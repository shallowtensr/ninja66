/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	StreamFn,
} from "./types.js";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Create a steering message.
 */
function steer(text: string): AgentMessage {
	return {
		role: "user",
		content: [{ type: "text", text }],
		timestamp: Date.now(),
	};
}

/**
 * Parse file paths from system prompt or task text.
 */
function parseExpectedFiles(text: string): string[] {
	const files: string[] = [];
	const seen = new Set<string>();
	const sectionPatterns = [
		/FILES EXPLICITLY NAMED IN THE TASK[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
		/LIKELY RELEVANT FILES[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
		/Pre-identified target files[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
	];
	for (const re of sectionPatterns) {
		const match = text.match(re);
		if (!match) continue;
		const lineRe = /^[-*]\s+(\S[^(]*?)(?:\s+\(|\s*$)/gm;
		let m: RegExpExecArray | null;
		while ((m = lineRe.exec(match[1])) !== null) {
			const file = m[1].trim();
			if (file && !seen.has(file)) { seen.add(file); files.push(file); }
		}
	}
	return files;
}

/**
 * Normalize a file path for comparison.
 */
function normPath(p: string): string {
	return p.replace(/^\.\//, "");
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 * Enhanced with competitive optimizations for SN66 duels.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<void> {
	let firstTurn = true;
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// --- Competitive state tracking ---
	let upstreamRetries = 0;
	const UPSTREAM_RETRY_LIMIT = 100;

	// Edit failure tracking
	const editFailMap = new Map<string, number>();
	const failNotified = new Set<string>();
	const EDIT_FAIL_CEILING = 2;
	const priorFailedAnchor = new Map<string, string>();

	// Exploration vs editing tracking
	let explorationCount = 0;
	let hasProducedEdit = false;
	let emptyTurnRetries = 0;
	const EMPTY_TURN_MAX = 2;

	// Timing
	const loopStart = Date.now();
	let earlyNudgeSent = false;
	let urgentNudgeSent = false;
	let finalNudgeSent = false;
	const EARLY_NUDGE_MS = 12_000;
	const URGENT_NUDGE_MS = 25_000;
	const LATE_NUDGE_MS = 60_000;
	const GRACEFUL_EXIT_MS = 180_000;

	// File tracking
	const pathsAlreadyRead = new Set<string>();
	const pathReadCounts = new Map<string, number>();
	let lastRereadNudgeAt = 0;
	const editedPaths = new Set<string>();
	let consecutiveEditsOnSameFile = 0;
	let lastEditedFile = "";

	// Work phases
	let workPhase: "search" | "absorb" | "apply" = "search";
	let foundFiles: string[] = [];
	const absorbedFiles = new Set<string>();

	// Extract expected files from system prompt or initial messages
	const systemPromptText = (currentContext as any).systemPrompt || "";
	let expectedFiles: string[] = parseExpectedFiles(systemPromptText);
	if (expectedFiles.length === 0) {
		for (const msg of currentContext.messages) {
			if (!("content" in msg) || !Array.isArray(msg.content)) continue;
			for (const block of msg.content as any[]) {
				if (block?.type === "text" && typeof block.text === "string") {
					const parsed = parseExpectedFiles(block.text);
					if (parsed.length > 0) { expectedFiles = parsed; break; }
				}
			}
			if (expectedFiles.length > 0) break;
		}
	}
	if (expectedFiles.length > 0) {
		foundFiles = [...expectedFiles];
		workPhase = "absorb";
	}

	let coverageRetries = 0;
	const MAX_COVERAGE_RETRIES = 2;
	let multiFileHintSent = false;
	let reviewPassDone = false;

	const missingExpectedFiles = (): string[] => {
		if (expectedFiles.length === 0) return [];
		const missing: string[] = [];
		for (const f of expectedFiles) {
			const norm = normPath(f);
			let touched = false;
			for (const e of editedPaths) {
				const en = normPath(e);
				if (en === norm || en.endsWith("/" + norm) || norm.endsWith("/" + en)) { touched = true; break; }
			}
			if (!touched) missing.push(f);
		}
		return missing;
	};

	const getUneditedTargets = (): string[] =>
		foundFiles.filter((f) => {
			const nf = normPath(f);
			return !editedPaths.has(f) && !editedPaths.has(nf) && !editedPaths.has("./" + nf);
		});

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				await emit({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			// Process pending messages
			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					await emit({ type: "message_start", message });
					await emit({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFn);
			newMessages.push(message);

			if (message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// Handle upstream errors with retry
			if (message.stopReason === "error") {
				if (upstreamRetries < UPSTREAM_RETRY_LIMIT) {
					upstreamRetries++;
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push(steer(
						"Transient upstream failure. Resume by calling a tool directly — avoid prose. Only file diffs count toward your score."
					));
					hasMoreToolCalls = false;
					continue;
				}
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			const toolCalls = message.content.filter((c) => c.type === "toolCall");

			// Fix Gemini hallucinated tool names
			for (const tc of toolCalls) {
				const lcName = tc.name.toLowerCase();
				if (lcName === "editedits" || lcName === "edits") {
					(tc as { name: string }).name = "edit";
				}
			}

			hasMoreToolCalls = toolCalls.length > 0;

			// Handle empty turns (no tool calls when we need edits)
			if (!hasMoreToolCalls && emptyTurnRetries < EMPTY_TURN_MAX) {
				const tokenCapped = message.stopReason === "length";
				const idleStopped = message.stopReason === "stop" && !hasProducedEdit;
				if (tokenCapped || idleStopped) {
					emptyTurnRetries++;
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push(steer(
						tokenCapped
							? "Output budget consumed without tool invocation. Invoke `read` or `edit` now. Text output contributes nothing to your score."
							: "No file modifications detected. A blank diff scores zero. Use `read` on the primary file, then `edit` it."
					));
					continue;
				}
			}

			// Force coverage: model stopping but expected files still untouched
			if (!hasMoreToolCalls && hasProducedEdit && coverageRetries < MAX_COVERAGE_RETRIES) {
				const missing = missingExpectedFiles();
				if (missing.length > 0) {
					coverageRetries++;
					await emit({ type: "turn_end", message, toolResults: [] });
					const list = missing.slice(0, 5).map((f) => `\`${f}\``).join(", ");
					pendingMessages.push(steer(
						`These target files have NOT been edited: ${list}. Read each and apply changes. Missing a file forfeits all its matched lines.`
					));
					hasMoreToolCalls = false;
					continue;
				}
			}

			// Execute tool calls
			const toolResults: ToolResultMessage[] = [];
			if (hasMoreToolCalls) {
				toolResults.push(...(await executeToolCalls(currentContext, message, config, signal, emit)));

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}

				// --- Process edit results ---
				for (let i = 0; i < toolResults.length; i++) {
					const tr = toolResults[i];
					const tc = toolCalls[i];
					if (!tc || tc.type !== "toolCall") continue;

					const isEdit = tc.name === "edit" || tc.name === "write";
					if (!isEdit) continue;

					const targetPath = (tc.arguments as any)?.path;
					if (!targetPath || typeof targetPath !== "string") continue;

					if (tr.isError) {
						// Track edit failures
						const count = (editFailMap.get(targetPath) ?? 0) + 1;
						editFailMap.set(targetPath, count);
						const anchorText = (tc.arguments as any)?.old_string ?? (tc.arguments as any)?.oldText ?? "";
						const prevAnchor = priorFailedAnchor.get(targetPath);
						if (anchorText && prevAnchor === anchorText && pendingMessages.length === 0) {
							pendingMessages.push(steer(
								`Identical anchor failed twice on \`${targetPath}\`. Use \`read\` to refresh before retrying.`
							));
						}
						priorFailedAnchor.set(targetPath, anchorText);
						if (count >= EDIT_FAIL_CEILING && !failNotified.has(targetPath)) {
							failNotified.add(targetPath);
							pendingMessages.push(steer(
								`Edit on \`${targetPath}\` failed ${count}x. Your cached view is stale. Either:\n1. Switch to another unedited file.\n2. \`read\` this file first, then use a short anchor (under 5 lines).\n3. Never paste from memory.`
							));
						}
					} else {
						// Successful edit
						editFailMap.set(targetPath, 0);
						priorFailedAnchor.delete(targetPath);
						const firstEdit = !hasProducedEdit;
						hasProducedEdit = true;
						explorationCount = 0;
						const normTarget = normPath(targetPath);
						editedPaths.add(targetPath);
						editedPaths.add(normTarget);
						editedPaths.add("./" + normTarget);

						// Breadth-first enforcement
						if (normTarget === lastEditedFile) {
							consecutiveEditsOnSameFile++;
						} else {
							consecutiveEditsOnSameFile = 1;
							lastEditedFile = normTarget;
						}

						const uneditedTargets = getUneditedTargets();
						let breadthHint = "";
						if (consecutiveEditsOnSameFile >= 3 && uneditedTargets.length > 0) {
							breadthHint = ` STOP editing \`${normTarget}\` — ${consecutiveEditsOnSameFile} consecutive edits. ${uneditedTargets.length} file(s) still need edits: ${uneditedTargets.slice(0, 5).map((f) => `\`${f}\``).join(", ")}. Move to next file NOW.`;
						} else if (uneditedTargets.length > 0) {
							breadthHint = ` ${uneditedTargets.length} target(s) still need edits: ${uneditedTargets.slice(0, 5).map((f) => `\`${f}\``).join(", ")}. Breadth across files scores higher than depth in one.`;
						}

						pendingMessages.push(steer(`\`${targetPath}\` updated.${breadthHint}`));

						if (firstEdit && !multiFileHintSent && (foundFiles.length >= 4 || pathsAlreadyRead.size >= 4)) {
							multiFileHintSent = true;
							pendingMessages.push(steer(
								"Multiple candidate files detected. If any acceptance criterion maps to an unedited file, continue there before stopping."
							));
						}
					}
				}

				// Detect connection failures in bash output
				for (const tr of toolResults) {
					if (tr.toolName === "bash" && !tr.isError) {
						const output = tr.content?.map((c: any) => c.text ?? "").join("") ?? "";
						if (output.includes("ConnectionRefusedError") || output.includes("ECONNREFUSED")) {
							pendingMessages.push(steer(
								"No services available. All network requests will fail. Use `read` and `edit` only."
							));
							break;
						}
					}
				}

				// --- Phase transitions ---
				if (workPhase === "search") {
					for (const tr of toolResults) {
						if ((tr.toolName === "bash" || tr.toolName === "grep" || tr.toolName === "find") && !tr.isError) {
							const output = tr.content?.map((c: any) => c.text ?? "").join("") ?? "";
							const paths = output.split("\n")
								.filter((l: string) => l.trim().match(/\.\w+$/))
								.map((l: string) => l.trim());
							if (paths.length > 0) {
								foundFiles = paths.slice(0, 20);
								workPhase = "absorb";
								pendingMessages.push(steer(
									`Located ${foundFiles.length} candidate files. Read each target before editing:\n${foundFiles.slice(0, 10).map((p) => `- ${p}`).join("\n")}`
								));
							}
						}
					}
				} else if (workPhase === "absorb") {
					for (const tr of toolResults) {
						if (tr.toolName === "read" && !tr.isError) {
							const tc2 = toolCalls.find((c: any) => c.type === "toolCall" && c.name === "read");
							if (tc2) {
								const path = (tc2.arguments as any)?.path ?? "";
								if (path) absorbedFiles.add(path);
							}
						}
						if ((tr.toolName === "edit" || tr.toolName === "write") && !tr.isError) {
							workPhase = "apply";
						}
					}
					const absorbLimit = Math.min(Math.max(3, foundFiles.length > 10 ? 6 : 3), 8);
					if (absorbedFiles.size >= absorbLimit && workPhase === "absorb" && pendingMessages.length === 0) {
						workPhase = "apply";
						pendingMessages.push(steer(
							`${absorbedFiles.size} files absorbed. Begin editing now — invoke \`edit\` directly. Cover every acceptance criterion.`
						));
					}
				}

				// Track exploration count
				for (let i = 0; i < toolResults.length; i++) {
					const tr = toolResults[i];
					const tc = toolCalls[i];
					if ((tr.toolName === "read" || tr.toolName === "bash") && !tr.isError) {
						if (!hasProducedEdit) explorationCount++;
					}
					if (tr.toolName === "read" && !tr.isError && tc && tc.type === "toolCall") {
						const readPath = (tc.arguments as any)?.path;
						if (readPath && typeof readPath === "string") {
							pathsAlreadyRead.add(readPath);
							pathReadCounts.set(readPath, (pathReadCounts.get(readPath) ?? 0) + 1);
						}
					}
				}

				// Re-read nudge: stop reading the same file over and over
				const now = Date.now();
				if (now - lastRereadNudgeAt >= 5_000 && pendingMessages.length === 0) {
					for (const [rp, cnt] of pathReadCounts) {
						if (cnt >= 3) {
							lastRereadNudgeAt = now;
							const others = getUneditedTargets().filter((f) => normPath(f) !== normPath(rp));
							pendingMessages.push(steer(
								`\`${rp}\` read ${cnt}x — stop re-reading. ${others.length > 0 ? `Move to: ${others.slice(0, 5).map((f) => `\`${f}\``).join(", ")}.` : "Apply `edit` or stop."}`
							));
							break;
						}
					}
				}

				// Exploration ceiling: too many reads without editing
				const dynamicCeiling = Math.max(3, Math.min(foundFiles.length + 1, 6));
				if (!hasProducedEdit && explorationCount >= dynamicCeiling && pendingMessages.length === 0) {
					pendingMessages.push(steer(
						`${explorationCount} reads without any edit. Apply your first edit NOW. A partial patch always outscores an empty diff.`
					));
					explorationCount = 0;
				}

				// Time-based nudges (no edits yet)
				if (!hasProducedEdit && pendingMessages.length === 0) {
					const elapsed = Date.now() - loopStart;
					const readInfo = pathsAlreadyRead.size > 0
						? `Read so far: ${[...pathsAlreadyRead].slice(0, 5).join(", ")}. `
						: "";
					if (!earlyNudgeSent && elapsed >= EARLY_NUDGE_MS) {
						earlyNudgeSent = true;
						pendingMessages.push(steer(
							`${Math.round(elapsed/1000)}s elapsed, zero edits. Empty diff = zero score. ${readInfo}Apply \`edit\` now.`
						));
					} else if (earlyNudgeSent && !urgentNudgeSent && elapsed >= URGENT_NUDGE_MS) {
						urgentNudgeSent = true;
						pendingMessages.push(steer(
							`${Math.round(elapsed/1000)}s, still no edits. Time running out. ${readInfo}Edit immediately or accept zero score.`
						));
					} else if (!finalNudgeSent && elapsed >= LATE_NUDGE_MS) {
						finalNudgeSent = true;
						pendingMessages.push(steer(
							"Over 60s without edits. Pick the clearest file and apply `edit` now — further discovery has diminishing returns."
						));
					}
				}

				// Mid-run breadth nudge (has edits but stuck on few files)
				if (hasProducedEdit && pendingMessages.length === 0) {
					const elapsed = Date.now() - loopStart;
					const uniqueEdited = new Set([...editedPaths].map(normPath));
					const uneditedFound = getUneditedTargets();
					if (uneditedFound.length > 0 && elapsed > 30_000 && uniqueEdited.size <= 2) {
						pendingMessages.push(steer(
							`30s+ and only ${uniqueEdited.size} file(s) edited. ${uneditedFound.length} target(s) remain: ${uneditedFound.slice(0, 8).map((f) => `\`${f}\``).join(", ")}. Edit each before revisiting.`
						));
					}
				}

				// Graceful exit on time limit
				if ((Date.now() - loopStart) >= GRACEFUL_EXIT_MS) {
					await emit({ type: "turn_end", message, toolResults });
					await emit({ type: "agent_end", messages: newMessages });
					return;
				}
			}

			await emit({ type: "turn_end", message, toolResults });
			pendingMessages = (await config.getSteeringMessages?.()) || [];
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			pendingMessages = followUpMessages;
			continue;
		}

		// Review pass: finished quickly with edits → check for missed files
		const reviewElapsed = Date.now() - loopStart;
		if (!reviewPassDone && hasProducedEdit && reviewElapsed < 60_000) {
			reviewPassDone = true;
			const uneditedTargets = getUneditedTargets();
			const hint = uneditedTargets.length > 0
				? `Unedited files: ${uneditedTargets.slice(0, 5).map((f) => `\`${f}\``).join(", ")}. Read and edit them.`
				: `Re-read acceptance criteria. Any missed files or criteria? If all covered, reply "done".`;
			pendingMessages = [steer(
				`REVIEW: Edited ${editedPaths.size} file(s): ${[...editedPaths].slice(0, 8).join(", ")}. ${hint}`
			)];
			continue;
		}

		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	const streamFunction = streamFn || streamSimple;

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	if (config.toolExecution === "sequential") {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			results.push(
				await finalizeExecutedToolCall(
					currentContext,
					assistantMessage,
					preparation,
					executed,
					config,
					signal,
					emit,
				),
			);
		}
	}

	return results;
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];
	const runnableCalls: PreparedToolCall[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
		} else {
			runnableCalls.push(preparation);
		}
	}

	const runningCalls = runnableCalls.map((prepared) => ({
		prepared,
		execution: executePreparedToolCall(prepared, signal, emit),
	}));

	for (const running of runningCalls) {
		const executed = await running.execution;
		results.push(
			await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				running.prepared,
				executed,
				config,
				signal,
				emit,
			),
		);
	}

	return results;
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
				};
			}
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		const afterResult = await config.afterToolCall(
			{
				assistantMessage,
				toolCall: prepared.toolCall,
				args: prepared.args,
				result,
				isError,
				context: currentContext,
			},
			signal,
		);
		if (afterResult) {
			result = {
				content: afterResult.content ?? result.content,
				details: afterResult.details ?? result.details,
			};
			isError = afterResult.isError ?? isError;
		}
	}

	return await emitToolCallOutcome(prepared.toolCall, result, isError, emit);
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolCallOutcome(
	toolCall: AgentToolCall,
	result: AgentToolResult<any>,
	isError: boolean,
	emit: AgentEventSink,
): Promise<ToolResultMessage> {
	await emit({
		type: "tool_execution_end",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		result,
		isError,
	});

	const toolResultMessage: ToolResultMessage = {
		role: "toolResult",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		content: result.content,
		details: result.details,
		isError,
		timestamp: Date.now(),
	};

	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
	return toolResultMessage;
}
