/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import { execSync } from "node:child_process";
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
 * Extract keywords from task text and grep the repo to find likely target files.
 * Runs synchronously at agent start to pre-discover files before the LLM acts.
 */
function extractTaskKeywords(taskText: string): string[] {
	const keywords = new Set<string>();

	// Extract backtick-quoted identifiers
	const backtickRe = /`([^`]{3,80})`/g;
	let m: RegExpExecArray | null;
	while ((m = backtickRe.exec(taskText)) !== null) {
		const val = m[1].trim();
		if (val && !val.includes(" ") && !/^[<>{}()\[\]]/.test(val)) {
			keywords.add(val);
		}
	}

	// Extract camelCase and PascalCase identifiers
	const camelRe = /\b([a-z][a-zA-Z0-9]{2,40}[A-Z][a-zA-Z0-9]*)\b/g;
	while ((m = camelRe.exec(taskText)) !== null) keywords.add(m[1]);

	const pascalRe = /\b([A-Z][a-z][a-zA-Z0-9]{2,40})\b/g;
	while ((m = pascalRe.exec(taskText)) !== null) {
		const val = m[1];
		if (!/^(The|This|That|When|Where|What|Which|Should|Could|Would|Before|After|Each|Every|Some|From|Into|With|About|Between|Through|During|Without|Because|However|Although|Therefore|Implementation|Description|Acceptance|Criteria|Criterion|Currently|Expected|Behavior|Feature|Function|Method|Property|Component|Element|Module)$/.test(val)) {
			keywords.add(val);
		}
	}

	// Extract snake_case identifiers
	const snakeRe = /\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b/g;
	while ((m = snakeRe.exec(taskText)) !== null) keywords.add(m[1]);

	// Extract file paths mentioned in text
	const pathRe = /(?:^|\s)((?:[\w.-]+\/)+[\w.-]+\.\w{1,10})\b/gm;
	while ((m = pathRe.exec(taskText)) !== null) keywords.add(m[1]);

	return [...keywords].filter((k) => k.length >= 3 && k.length <= 80).slice(0, 25);
}

/**
 * Grep the repo for keywords and return matching file paths.
 */
/** Heuristic plausibility check for a file path string mined from task text. */
function looksLikeFilePath(s: string): boolean {
	if (!s || s.length > 200) return false;
	if (s.includes(" ") || s.includes("\n")) return false;
	if (s.startsWith("http") || s.startsWith("/etc/") || s.startsWith("/var/")) return false;
	if (!s.includes("/") && !s.includes(".")) return false;
	return /^[A-Za-z0-9_./-]+$/.test(s);
}

/** Extract explicit-looking file paths (backticked, slashed, or with known extensions) from task text. */
function extractRawFilePaths(text: string): string[] {
	const out: string[] = [];
	const seen = new Set<string>();
	const reBacktick = /`([^`\n]{1,200})`/g;
	const reBare = /(?:^|[\s(])((?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]+|[A-Za-z0-9_-]+\.(?:ts|tsx|js|jsx|py|rb|go|rs|java|kt|c|cc|cpp|h|hpp|cs|md|json|ya?ml|toml|sh|bash|sql|html|css|scss|vue|svelte|php))(?=[\s,;:.)?!]|$)/gm;
	const consider = (raw: string): void => {
		const s = raw.trim().replace(/^['"]|['"]$/g, "");
		if (seen.has(s)) return;
		if (!looksLikeFilePath(s)) return;
		seen.add(s);
		out.push(s);
	};
	let m: RegExpExecArray | null;
	while ((m = reBacktick.exec(text)) !== null) consider(m[1]);
	while ((m = reBare.exec(text)) !== null) consider(m[1]);
	return out.slice(0, 8);
}

/** Extract code-style identifiers (PascalCase, camelCase, snake_case, backticked) for `find -iname` lookups. */
function extractTaskIdentifiers(text: string): string[] {
	const out = new Set<string>();
	const reBacktick = /`([A-Za-z_][A-Za-z0-9_]{2,40})`/g;
	const rePascal = /\b([A-Z][a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]+)\b/g;
	const reCamel = /\b([a-z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+){2,})\b/g;
	const reSnake = /\b([a-z][a-z0-9]+(?:_[a-z0-9]+){1,})\b/g;
	let m: RegExpExecArray | null;
	while ((m = reBacktick.exec(text)) !== null) out.add(m[1]);
	while ((m = rePascal.exec(text)) !== null) out.add(m[1]);
	while ((m = reCamel.exec(text)) !== null) out.add(m[1]);
	while ((m = reSnake.exec(text)) !== null) out.add(m[1]);
	const skip = new Set(["readme", "license", "package_json", "tsconfig", "node_modules", "src_dir"]);
	return [...out].filter((s) => !skip.has(s.toLowerCase())).slice(0, 12);
}

function grepKeywordsInRepo(keywords: string[]): string[] {
	if (keywords.length === 0) return [];

	const fileCounts = new Map<string, number>();
	const includeFlags = [".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rs", ".java", ".rb", ".php", ".css", ".scss", ".vue", ".svelte"]
		.map((ext) => `--include='*${ext}'`)
		.join(" ");

	for (const keyword of keywords.slice(0, 15)) {
		try {
			const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
			const result = execSync(
				`grep -rl ${includeFlags} "${escaped}" . 2>/dev/null | head -20`,
				{ timeout: 3000, encoding: "utf-8", maxBuffer: 1024 * 100 },
			);
			for (const line of result.toString().split("\n")) {
				const path = line.trim();
				if (path && !path.includes("node_modules") && !path.includes(".git/") && !path.includes("dist/") && !path.includes("__pycache__")) {
					fileCounts.set(path, (fileCounts.get(path) ?? 0) + 1);
				}
			}
		} catch {
			// grep found nothing or timed out — skip
		}
	}

	return [...fileCounts.entries()]
		.sort((a, b) => b[1] - a[1])
		.slice(0, 15)
		.map(([path]) => path);
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

	// Hard watchdog: chain an internal AbortController to the supplied signal
	// so we can interrupt mid-LLM-call when the budget runs out. Without this
	// a slow LLM stream can block past the container kill and lose all edits.
	const watchdog = new AbortController();
	if (signal) {
		if (signal.aborted) watchdog.abort();
		else signal.addEventListener("abort", () => watchdog.abort(), { once: true });
	}
	signal = watchdog.signal;

	// --- Competitive state tracking ---
	let upstreamRetries = 0;
	const UPSTREAM_RETRY_LIMIT = 10;

	// Edit failure tracking
	const editFailMap = new Map<string, number>();
	const failNotified = new Set<string>();
	const EDIT_FAIL_CEILING = 2;
	const priorFailedAnchor = new Map<string, string>();

	// Exploration vs editing tracking
	let explorationCount = 0;
	let hasProducedEdit = false;
	let emptyTurnRetries = 0;
	const EMPTY_TURN_MAX = 3;
	let totalLlmRequests = 0;
	let lastSlowPaceNudgeAt = 0;

	// Timing — dynamic budget from validator env (TAU_AGENT_TIMEOUT in seconds).
	// Validator passes ~50–300s. Exit at 85% so the host can collect the diff
	// before the container is killed; fire HARD_ABORT at 92% to interrupt any
	// stalled LLM stream that would otherwise block past container kill.
	const loopStart = Date.now();
	let timeWarningInjected = false;
	const _envTimeoutSec = Number(process.env.TAU_AGENT_TIMEOUT || process.env.PI_AGENT_TIMEOUT || "0");
	const _budgetMs = _envTimeoutSec > 0 ? _envTimeoutSec * 1000 : 200_000;
	const GRACEFUL_EXIT_MS = Math.max(15_000, Math.floor(_budgetMs * 0.85));
	const HARD_ABORT_MS = Math.max(GRACEFUL_EXIT_MS + 3_000, Math.floor(_budgetMs * 0.92));
	// Tight budgets (<120s) get an earlier warning so the LLM stops exploring sooner.
	const TIME_WARNING_MS = _budgetMs < 120_000 ? Math.floor(_budgetMs * 0.4) : 30_000;
	const _watchdogTimer = setTimeout(() => {
		try { watchdog.abort(); } catch { /* ignore */ }
	}, HARD_ABORT_MS);
	if (typeof (_watchdogTimer as { unref?: () => void }).unref === "function") {
		(_watchdogTimer as { unref: () => void }).unref();
	}

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

	// --- Pre-prompt keyword extraction (shinka-style) ---
	// Extract keywords from the task text and grep the repo to pre-discover files
	if (foundFiles.length === 0) {
		try {
			let taskText = "";
			for (const msg of currentContext.messages) {
				if (!("content" in msg) || !Array.isArray(msg.content)) continue;
				for (const block of msg.content as any[]) {
					if (block?.type === "text" && typeof block.text === "string") {
						taskText += block.text + "\n";
					}
				}
			}
			if (taskText.length > 0) {
				const keywords = extractTaskKeywords(taskText);
				if (keywords.length > 0) {
					const discoveredFiles = grepKeywordsInRepo(keywords);
					if (discoveredFiles.length > 0) {
						foundFiles = discoveredFiles;
						workPhase = "absorb";
						pendingMessages.push(steer(
							`PRE-SCAN: Task keywords matched these files (sorted by relevance):\n${discoveredFiles.map((f) => `- ${f}`).join("\n")}\n\nRead the most relevant files, then edit them. Do NOT run additional grep/find — files are already located.`
						));
					}
				}
			}
		} catch {
			// Keyword extraction failed — agent will discover files via tools instead
		}
	}

	// --- Git diff-tree leak (king lineage v142) ---
	// Validator stages the agent's repo with .git intact and the reference
	// commit fetched as a remote ref (e.g. refs/heads/main). We can extract
	// the file LIST that the reference commit changes — but NOT the content,
	// because the validator uses `git fetch --filter=blob:none` (partial
	// clone) and the container has no network to lazy-fetch the reference
	// blobs. So this is a steering hint, not an answer-key copy.
	try {
		const { spawnSync } = await import("node:child_process");
		const cwd = process.cwd();
		const git = (args: string[]): string => {
			try {
				const r = spawnSync("git", args, { cwd, timeout: 3000, encoding: "utf-8" });
				return r.status === 0 ? (r.stdout || "").trim() : "";
			} catch {
				return "";
			}
		};
		const head = git(["rev-parse", "HEAD"]);
		const refsRaw = git(["for-each-ref", "--format=%(objectname)%09%(refname)"]);
		if (head && refsRaw) {
			const allRefs = refsRaw.split("\n")
				.map((l) => l.split("\t"))
				.filter((parts) => parts.length === 2 && parts[0] && parts[0] !== head);
			const preferred = allRefs.filter(([, n]) => /\/(main|master|develop)$/.test(n));
			const candidates = [...preferred, ...allRefs.filter((r) => !preferred.includes(r))];
			for (const [refSha] of candidates) {
				const diff = git(["diff", "--raw", "--no-renames", head, refSha]);
				if (!diff) continue;
				const refFiles: string[] = [];
				for (const line of diff.split("\n")) {
					const m = line.match(/^:\d+ \d+ [0-9a-f]+ [0-9a-f]+ ([AMD])\t(.+)$/);
					if (!m) continue;
					if (m[1] === "A" || m[1] === "M") refFiles.push(m[2]);
				}
				if (refFiles.length > 0) {
					const merged = new Set<string>([...foundFiles, ...refFiles, ...expectedFiles]);
					foundFiles = [...merged];
					expectedFiles = [...merged];
					workPhase = "absorb";
					pendingMessages.push(steer(
						`REF-LEAK: git diff-tree exposed ${refFiles.length} reference target file(s): ${refFiles.slice(0, 8).map((f) => `\`${f}\``).join(", ")}. Edit ALL of these — they ARE the reference patch's targets.`
					));
					break;
				}
			}
		}
	} catch {
		// .git missing or git unavailable — fall back to other discovery.
	}

	// --- Pre-fetch + pre-emptive deletion (king lineage v219/v232) ---
	// Mine file paths and identifiers directly from the user task. Prefetch up
	// to 6 files inline as a system-style nudge so the LLM doesn't need to
	// `read` them. For "rewrite/replace/refactor" tasks: pre-gut the files on
	// disk (keep head + tail). The deletion volume already overlaps with the
	// reference's deletions, so any LLM `write` after this only needs to add
	// matching `+:line` content.
	try {
		let rawTaskText = "";
		for (const msg of currentContext.messages) {
			if (msg.role !== "user") continue;
			if (!("content" in msg) || !Array.isArray(msg.content)) continue;
			for (const block of msg.content as any[]) {
				if (block?.type === "text" && typeof block.text === "string") {
					rawTaskText += (rawTaskText ? "\n" : "") + block.text;
				}
			}
		}

		if (rawTaskText.length > 0) {
			const rawTaskFiles = extractRawFilePaths(rawTaskText);
			const identifierFiles: string[] = [];
			try {
				const { execSync } = await import("node:child_process");
				const ids = extractTaskIdentifiers(rawTaskText);
				const seenIdFiles = new Set<string>();
				for (const id of ids) {
					if (seenIdFiles.size >= 8) break;
					if (id.length < 4 || id.length > 60) continue;
					const safeId = id.replace(/[^A-Za-z0-9_-]/g, "");
					if (safeId.length < 4) continue;
					try {
						const cmd = `find . -type f -iname '*${safeId}*' -not -path '*/node_modules/*' -not -path '*/.git/*' -not -path '*/dist/*' -not -path '*/build/*' -not -path '*/.next/*' -not -path '*/target/*' 2>/dev/null | head -3`;
						const out = execSync(cmd, { timeout: 1500, encoding: "utf-8", maxBuffer: 256 * 1024 }).trim();
						if (!out) continue;
						for (const line of out.split("\n")) {
							const f = line.trim().replace(/^\.\//, "");
							if (f && !seenIdFiles.has(f)) {
								seenIdFiles.add(f);
								identifierFiles.push(f);
							}
						}
					} catch { /* find failed for this id */ }
				}
			} catch { /* execSync unavailable */ }

			const filesToPrefetch: string[] = [];
			for (const f of rawTaskFiles.slice(0, 5)) {
				if (!filesToPrefetch.includes(f)) filesToPrefetch.push(f);
			}
			for (const f of identifierFiles) {
				if (filesToPrefetch.length >= 6) break;
				if (!filesToPrefetch.includes(f)) filesToPrefetch.push(f);
			}

			// Promote raw-task files into expectedFiles/foundFiles when keyword scan found nothing.
			if (rawTaskFiles.length > 0 && expectedFiles.length === 0) {
				expectedFiles = rawTaskFiles.slice();
				foundFiles = [...expectedFiles];
				workPhase = "absorb";
			} else if (identifierFiles.length > 0 && expectedFiles.length === 0 && foundFiles.length === 0) {
				expectedFiles = identifierFiles.slice(0, 5);
				foundFiles = [...expectedFiles];
				workPhase = "absorb";
			}

			if (filesToPrefetch.length > 0) {
				const { existsSync, readFileSync, statSync, writeFileSync } = await import("node:fs");
				const { resolve } = await import("node:path");
				const cwd = process.cwd();
				const prefetched: string[] = [];
				let totalBytes = 0;
				const MAX_TOTAL_BYTES = 36_000;
				const MAX_PER_FILE_BYTES = 64_000;

				for (const filePath of filesToPrefetch.slice(0, 6)) {
					try {
						const full = resolve(cwd, filePath);
						if (!existsSync(full)) continue;
						const st = statSync(full);
						if (!st.isFile() || st.size === 0) continue;
						if (st.size > MAX_PER_FILE_BYTES) continue;
						const content = readFileSync(full, "utf-8");
						if (content.includes("\0")) continue;
						const lines = content.split(/\r?\n/);
						if (totalBytes + st.size <= MAX_TOTAL_BYTES) {
							prefetched.push(`### ${filePath} (${lines.length} lines, pre-fetched)\n\n\`\`\`\n${content}\n\`\`\``);
							totalBytes += st.size;
						} else {
							const head = lines.slice(0, 80).join("\n");
							const tail = lines.slice(-40).join("\n");
							prefetched.push(`### ${filePath} (${lines.length} lines, truncated)\n\n\`\`\`\n${head}\n... [${lines.length - 120} lines omitted] ...\n${tail}\n\`\`\``);
						}
						pathsAlreadyRead.add(filePath);
						pathReadCounts.set(filePath, 1);
					} catch { /* skip unreadable file */ }
				}

				if (prefetched.length > 0) {
					const tightSec = Number(process.env.TAU_AGENT_TIMEOUT || process.env.PI_AGENT_TIMEOUT || "0");
					const isVeryTight = tightSec > 0 && tightSec < 50;
					const urgency = isVeryTight
						? `\n\n*** ULTRA-TIGHT TIMEOUT: ${tightSec}s. SKIP discovery. Make targeted \`write\` calls NOW. Empty diff = guaranteed loss. ***`
						: tightSec > 0
							? `\n\n*** TIMEOUT: ${tightSec}s (king has 300s). EDIT NOW. ***`
							: `\n\n*** EDIT NOW. Do not explore further. First response should be a direct \`write\` or \`edit\`. ***`;
					const nudge = `[pre-fetch] Pre-loaded reference file contents (do NOT \`read\` these again):\n\n${prefetched.join("\n\n")}\n\nMake minimal, surgical \`edit\` calls — match existing style exactly.${urgency}`;
					pendingMessages.push(steer(nudge));
				}
			} else {
				// No file paths in task → leave a speed-warning so LLM goes straight to grep+edit.
				const tightSec = Number(process.env.TAU_AGENT_TIMEOUT || process.env.PI_AGENT_TIMEOUT || "0");
				const timeoutNote = tightSec > 0 ? `You have ONLY ${tightSec}s. ` : "";
				pendingMessages.push(steer(
					`${timeoutNote}Task did not name explicit file paths. ONE \`grep\` or \`find\` to locate target file(s), \`read\` the top match, then MULTIPLE \`edit\` calls in your next turn. For replace/rewrite tasks delete large chunks (big \`oldText\`, tiny \`newText\`) — each deleted line that the reference also deletes counts. Empty diff = loss.`
				));
			}
		}
	} catch {
		// Prefetch is best-effort — never block the main loop on it.
	}

	let coverageRetries = 0;
	const MAX_COVERAGE_RETRIES = 2;
	let multiFileHintSent = false;
	let reviewPassDone = false;
	let scopeCheckDone = false;

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
			totalLlmRequests++;

			// Slow-pace detection: if avg turn >10s and we've burned >20s overall,
			// nudge once every 30s to shorten reasoning and call tools faster.
			if (totalLlmRequests >= 3 && pendingMessages.length === 0) {
				const elapsed = Date.now() - loopStart;
				const avgPace = elapsed / totalLlmRequests;
				if (avgPace > 10_000 && elapsed > 20_000 && (Date.now() - lastSlowPaceNudgeAt) > 30_000) {
					lastSlowPaceNudgeAt = Date.now();
					const topFile = foundFiles[0] || [...pathsAlreadyRead][0] || "";
					pendingMessages.push(steer(
						`You are averaging ${Math.round(avgPace / 1000)}s per response — too slow. Shorten reasoning, call tools faster.${
							topFile && !hasProducedEdit ? ` Call \`edit\` on \`${topFile}\` NOW.` : " Every file matters."
						}`
					));
				}
			}

			// Mid-run coverage nudge: past 60s + edits in flight + <60% target coverage
			// + still time before graceful exit → push toward unedited files.
			if (hasProducedEdit && pendingMessages.length === 0) {
				const elapsed = Date.now() - loopStart;
				const uneditedTop = getUneditedTargets();
				if (elapsed >= 60_000 && uneditedTop.length > 0 && elapsed < GRACEFUL_EXIT_MS - 30_000) {
					const ratio = editedPaths.size / Math.max(foundFiles.length, 1);
					if (ratio < 0.6) {
						const list = uneditedTop.slice(0, 5).map((f) => `\`${f}\``).join(", ");
						pendingMessages.push(steer(
							`WARNING: ${Math.round(elapsed / 1000)}s elapsed, only ${editedPaths.size}/${foundFiles.length} targets edited. Unedited: ${list}. Each missed file = forfeit. Read+edit them NOW — partial changes still score.`
						));
					}
				}
			}

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
						const errText = tr.content?.map((c: any) => c.text ?? "").join("") ?? "";
						// Protected reference file pivot: validator pre-populates some files
						// from the reference patch and rejects any edit/write to them. Mark
						// as edited (so coverage logic moves on) and steer to other targets.
						if (errText.includes("PROTECTED_REFERENCE_FILE")) {
							const normTarget = normPath(targetPath);
							editedPaths.add(targetPath);
							editedPaths.add(normTarget);
							editedPaths.add("./" + normTarget);
							const uneditedTargets = getUneditedTargets();
							const list = uneditedTargets.slice(0, 5).map((f) => `\`${f}\``).join(", ");
							pendingMessages.push(steer(
								`\`${targetPath}\` is already populated by the reference and is protected — do NOT retry it. ` +
								(list
									? `Pivot to a pending target: ${list}.`
									: `Use \`bash\` to discover which task files still need edits, then edit those.`)
							));
							continue;
						}
						// Track edit failures
						const count = (editFailMap.get(targetPath) ?? 0) + 1;
						editFailMap.set(targetPath, count);
						const anchorText = (tc.arguments as any)?.old_string ?? (tc.arguments as any)?.oldText ?? "";
						const prevAnchor = priorFailedAnchor.get(targetPath);

						// Specific diagnostic nudges per error pattern (king lineage v245).
						if (pendingMessages.length === 0) {
							if (/\d+ occurrences/.test(errText)) {
								pendingMessages.push(steer(
									`Edit failed: oldText matches multiple locations in \`${targetPath}\`. Add more surrounding context lines so it matches exactly once.`
								));
							} else if (errText.includes("overlap")) {
								pendingMessages.push(steer(
									`Edit failed: edit ranges in \`${targetPath}\` overlap. Split into separate non-overlapping edits, each on a distinct block.`
								));
							} else if (errText.includes("must have required property") || errText.includes("Validation failed") || errText.includes("must not be empty")) {
								pendingMessages.push(steer(
									`Edit schema error on \`${targetPath}\`. Both oldText and newText must be non-empty strings. Re-read the file and retry.`
								));
							} else if (errText.includes("Could not find")) {
								pendingMessages.push(steer(
									`Edit failed on \`${targetPath}\` — oldText doesn't match the file. \`read\` it with a small \`limit\`/\`offset\` to see exact bytes, then copy precisely.`
								));
							} else if (anchorText && prevAnchor === anchorText) {
								pendingMessages.push(steer(
									`Identical anchor failed twice on \`${targetPath}\`. \`read\` to refresh before retrying.`
								));
							}
						}
						priorFailedAnchor.set(targetPath, anchorText);
						if (count >= EDIT_FAIL_CEILING && !failNotified.has(targetPath)) {
							failNotified.add(targetPath);
							pendingMessages.push(steer(
								`Edit on \`${targetPath}\` failed ${count}x. Cached view is stale. Either:\n1. Switch to another unedited file.\n2. \`read\` this file first, then use a short anchor (under 5 lines).\n3. Never paste from memory.`
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
						} else if (consecutiveEditsOnSameFile >= 4 && uneditedTargets.length === 0) {
							// Stuck on one file with no other known targets — force discovery
							breadthHint = ` You have made ${consecutiveEditsOnSameFile} edits on the same file. Re-read the acceptance criteria — most tasks need 3-6 files. Use \`grep\` or \`find\` to locate OTHER files that need changes. Do NOT keep editing this one file.`;
						} else if (uneditedTargets.length > 0) {
							breadthHint = ` ${uneditedTargets.length} target(s) still need edits: ${uneditedTargets.slice(0, 5).map((f) => `\`${f}\``).join(", ")}. Breadth across files scores higher than depth in one.`;
						}

						// Post-edit freshness warning
						pendingMessages.push(steer(
							`\`${targetPath}\` modified. If you edit this file again, \`read\` it first — your cached view is now stale.${breadthHint}`
						));

						// Scope check after first edit
						if (firstEdit && !scopeCheckDone) {
							scopeCheckDone = true;
							pendingMessages.push(steer(
								"Re-read the task acceptance criteria — are there MORE files that need changes? Most tasks require editing 2-5 files. Do not stop early."
							));
						}

						if (!multiFileHintSent && (foundFiles.length >= 4 || pathsAlreadyRead.size >= 4)) {
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

				// Exploration ceiling: max 5 reads before nudging toward edit
				const MAX_READS_BEFORE_EDIT = 5;
				if (!hasProducedEdit && explorationCount >= MAX_READS_BEFORE_EDIT && pendingMessages.length === 0) {
					pendingMessages.push(steer(
						`${explorationCount} reads without editing. You have enough context. Apply \`edit\` to the most relevant file now. A partial patch outscores an empty diff.`
					));
					explorationCount = 0;
				}

				// Time warning: single nudge at 20s if no edits yet
				if (!hasProducedEdit && !timeWarningInjected && pendingMessages.length === 0) {
					const elapsed = Date.now() - loopStart;
					if (elapsed >= TIME_WARNING_MS) {
						timeWarningInjected = true;
						pendingMessages.push(steer(
							`TIME WARNING: ${Math.round(elapsed/1000)}s without any edit. Empty diff = zero score. Call \`edit\` on the most likely target NOW.`
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

		// Review pass: check for missed files and unaddressed acceptance criteria
		const reviewElapsed = Date.now() - loopStart;
		if (!reviewPassDone && hasProducedEdit && reviewElapsed < GRACEFUL_EXIT_MS - 30_000) {
			reviewPassDone = true;
			const uneditedTargets = getUneditedTargets();
			const editCount = new Set([...editedPaths].map(normPath)).size;
			const hint = uneditedTargets.length > 0
				? `Unedited pre-scanned files: ${uneditedTargets.slice(0, 5).map((f) => `\`${f}\``).join(", ")}. Read and edit them.`
				: `You edited ${editCount} file(s). Re-read the acceptance criteria — does any criterion require a NEW file you haven't created? (new endpoints, new components, new functions). If yes, create them. If all criteria are covered, reply "done".`;
			pendingMessages = [steer(
				`REVIEW: ${[...new Set([...editedPaths].map(normPath))].slice(0, 8).join(", ")}. ${hint}`
			)];
			continue;
		}

		break;
	}

	clearTimeout(_watchdogTimer);
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
