export function resolveApiBase(
  publicApiUrl: string | undefined,
  hostname?: string,
  publicApiPort?: string,
): string {
  const configuredUrl = publicApiUrl?.trim().replace(/\/+$/, "");
  if (configuredUrl) return configuredUrl;

  const port = publicApiPort?.trim() || "8002";
  return `http://${hostname || "localhost"}:${port}/api`;
}

export const API_BASE = resolveApiBase(
  process.env.NEXT_PUBLIC_API_URL,
  typeof window !== "undefined" ? window.location.hostname : undefined,
  process.env.NEXT_PUBLIC_API_PORT,
);

export interface SSEEvent {
  type: string;
  content?: string;
  task?: string;
  tool?: string;
  name?: string;
  input?: any;
  input_preview?: string;
  args?: any;
  output?: string;
  output_preview?: string;
  error?: string;
  session_id?: string;
  title?: string;
  query?: string;
  results?: any[];
  response?: string;
  event?: string;
  run_id?: string;
  elapsed_s?: number;
  chars?: number;
  result_delivery_state?: string;
  usage?: TokenUsage;
  result?: any;
  approval_id?: string;
  context_utilization?: number;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  duration_ms: number;
}

export interface AgentStatus {
  agent_id: string;
  total_turns: number;
  total_input_tokens: number;
  total_output_tokens: number;
  compaction_count: number;
  thinking: boolean;
  verbose: boolean;
  reasoning: boolean;
  last_active: number;
  heartbeat_active: boolean;
}

export interface InitStatus {
  file_initialized: boolean;
  config_ready: boolean;
  providers_count: number;
  valid_providers_count: number;
  default_model: string | null;
  missing: string[];
}

async function readErrorMessage(resp: Response): Promise<string> {
  const fallback = `Request failed: ${resp.status}`;
  try {
    const data = await resp.json();
    if (typeof data?.detail === "string" && data.detail) return data.detail;
    if (typeof data?.error === "string" && data.error) return data.error;
    if (typeof data?.message === "string" && data.message) return data.message;
    return fallback;
  } catch {
    return fallback;
  }
}

/** Get stream inactivity timeout. timeoutSeconds=0 disables it. */
export async function fetchChatTimeout(): Promise<{ timeoutSeconds: number }> {
  const resp = await fetch(`${API_BASE}/config/chat`);
  return resp.json();
}

const TURN_POLL_MS = 500;
const TURN_STREAM_RECONNECT_MS = 500;

export interface ChatSubmitResponse {
  turn_id: string;
  position: number;
  status: string;
  session_id: string;
}

export async function submitChat(
  message: string,
  sessionId: string,
  agentId: string,
): Promise<ChatSubmitResponse> {
  const resp = await fetch(`${API_BASE}/chat/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, agent_id: agentId }),
  });
  if (resp.status === 409) {
    throw new Error((await readErrorMessage(resp)) || "Another user turn is active for this session");
  }
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

export async function getTurnStatus(turnId: string): Promise<{
  turn_id: string;
  status: string;
  position: number;
  session_id: string;
  agent_id: string;
  error?: string | null;
}> {
  const resp = await fetch(`${API_BASE}/chat/turn/${encodeURIComponent(turnId)}/status`);
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

export async function fetchPendingTurn(
  sessionId: string,
  agentId: string,
): Promise<{
  turn_id: string | null;
  status: string | null;
  position?: number;
  session_id: string;
  agent_id: string;
}> {
  const u = new URL(`${API_BASE}/chat/pending-turn`);
  u.searchParams.set("session_id", sessionId);
  u.searchParams.set("agent_id", agentId);
  const resp = await fetch(u.toString());
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

/** true = open SSE; false = turn already finished (reload messages from server). */
export async function waitUntilTurnRunning(turnId: string, signal?: AbortSignal): Promise<boolean> {
  while (true) {
    if (signal?.aborted) {
      const e = new Error("Aborted");
      e.name = "AbortError";
      throw e;
    }
    const s = await getTurnStatus(turnId);
    if (s.status === "running") return true;
    if (s.status === "done" || s.status === "error" || s.status === "cancelled") {
      return false;
    }
    await new Promise(r => setTimeout(r, TURN_POLL_MS));
  }
}

async function consumeSseStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: (event: SSEEvent) => void,
  onActivity?: () => void,
): Promise<boolean> {
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (buffer.trim()) {
        const remaining = buffer.trim();
        if (remaining.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(remaining.slice(6)) as SSEEvent;
            onEvent(parsed);
            return parsed.type === "done" || parsed.type === "error" || parsed.type === "aborted";
          } catch {
            // ignore
          }
        }
      }
      return false;
    }
    onActivity?.();
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    let reachedTerminalEvent = false;
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const parsed = JSON.parse(line.slice(6)) as SSEEvent;
          onEvent(parsed);
          if (parsed.type === "done" || parsed.type === "error" || parsed.type === "aborted") {
            reachedTerminalEvent = true;
            break;
          }
        } catch {
          // ignore
        }
      }
    }
    if (reachedTerminalEvent) {
      try {
        await reader.cancel();
      } catch {
        // ignore
      }
      return true;
    }
  }
}

export async function streamTurn(
  turnId: string,
  onEvent: (event: SSEEvent) => void,
  opts?: { signal?: AbortSignal; timeoutMs?: number },
): Promise<void> {
  const { signal: userSignal, timeoutMs } = opts ?? {};
  while (true) {
    if (userSignal?.aborted) {
      const error = new Error("Aborted");
      error.name = "AbortError";
      throw error;
    }

    const controller = new AbortController();
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let timeoutFired = false;
    let streamError: unknown = null;
    let terminalEventReceived = false;

    const resetInactivityTimeout = () => {
      if (timeoutId) clearTimeout(timeoutId);
      if (timeoutMs != null && timeoutMs > 0) {
        timeoutId = setTimeout(() => {
          timeoutFired = true;
          controller.abort();
        }, timeoutMs);
      }
    };
    const abortForUser = () => controller.abort();
    userSignal?.addEventListener("abort", abortForUser, { once: true });
    resetInactivityTimeout();

    try {
      const resp = await fetch(`${API_BASE}/chat/turn/${encodeURIComponent(turnId)}/stream`, {
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error(`Stream failed: ${resp.status}`);
      const reader = resp.body?.getReader();
      if (!reader) throw new Error("No response body");
      terminalEventReceived = await consumeSseStream(reader, onEvent, resetInactivityTimeout);
    } catch (error) {
      streamError = error;
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      userSignal?.removeEventListener("abort", abortForUser);
    }

    if (userSignal?.aborted) {
      if (streamError instanceof Error && streamError.name === "AbortError") throw streamError;
      const error = new Error("Aborted");
      error.name = "AbortError";
      throw error;
    }
    if (terminalEventReceived) return;

    try {
      const status = await getTurnStatus(turnId);
      if (status.status === "done" || status.status === "cancelled") return;
      if (status.status === "error") {
        throw new Error(status.error || "Turn failed");
      }
      if (status.status === "running" || status.status === "queued") {
        await new Promise(resolve => setTimeout(resolve, TURN_STREAM_RECONNECT_MS));
        continue;
      }
    } catch (statusError) {
      if (streamError) {
        if (timeoutFired && streamError instanceof Error && streamError.name === "AbortError") {
          throw new Error(`Request inactivity timeout (${Math.round((timeoutMs ?? 0) / 1000)}s)`);
        }
        throw streamError;
      }
      throw statusError;
    }

    if (streamError) throw streamError;
    throw new Error("Turn stream ended before the backend reached a terminal state");
  }
}

export async function abortChat(
  agentId: string,
  sessionId: string,
  opts?: { userInitiated?: boolean; turnId?: string },
): Promise<{ aborted: boolean }> {
  const body: Record<string, unknown> = {
    agent_id: agentId,
    session_id: sessionId,
    user_initiated: opts?.userInitiated !== false,
  };
  if (opts?.turnId) body.turn_id = opts.turnId;
  const resp = await fetch(`${API_BASE}/chat/abort`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    throw new Error(await readErrorMessage(resp));
  }
  return resp.json();
}

// ---------- REST API ----------

export async function fetchAgents(): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/agents`);
  return resp.json();
}

export async function createAgent(data: { id: string; name: string; description?: string; model?: string }) {
  const resp = await fetch(`${API_BASE}/agents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error((err as any)?.detail || `Create failed: ${resp.status}`);
  }
  return resp.json();
}

export async function deleteAgent(agentId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}`, { method: "DELETE" });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error((err as any)?.detail || `Delete failed: ${resp.status}`);
  }
  return resp.json();
}

export async function fetchAgentStatus(agentId: string): Promise<AgentStatus> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/status`);
  return resp.json();
}

export async function fetchAgentUsage(agentId: string, sessionId?: string): Promise<any> {
  const params = sessionId ? `?session_id=${sessionId}` : "";
  const resp = await fetch(`${API_BASE}/agents/${agentId}/usage${params}`);
  return resp.json();
}

export async function fetchAuditLog(agentId: string, limit: number = 50): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/audit-log?limit=${limit}`);
  return resp.json();
}

export async function fetchHeartbeatConfig(agentId: string): Promise<{ enabled: boolean; every: string; interval_seconds?: number }> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/heartbeat/config`);
  if (!resp.ok) throw new Error("Failed to fetch heartbeat config");
  return resp.json();
}

export async function fetchHeartbeatHistory(agentId: string, limit: number = 30): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/heartbeat/history?limit=${limit}`);
  return resp.json();
}

export async function updateHeartbeatEnabled(enabled: boolean): Promise<void> {
  await updateConfig({
    agents: {
      defaults: {
        heartbeat: { enabled },
      },
    },
  });
}

export async function resolveApproval(approvalId: string, decision: "approved" | "denied"): Promise<void> {
  const resp = await fetch(`${API_BASE}/approvals/${approvalId}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error((err as any)?.detail || `Confirm failed: ${resp.status}`);
  }
}

export async function fetchCronJobs(): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/cron/jobs`);
  return resp.json();
}

export async function createCronJob(body: {
  name: string;
  description?: string;
  agent_id?: string;
  enabled?: boolean;
  deleteAfterRun?: boolean;
  schedule: { kind?: string; at?: string; everyMs?: number; expr?: string; tz?: string };
  payload: { kind: string; text: string };
}) {
  const resp = await fetch(`${API_BASE}/cron/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`Create failed: ${resp.status}`);
  return resp.json();
}

export async function updateCronJob(jobId: string, body: Partial<{ name: string; description: string; agent_id: string; enabled: boolean; schedule: any; payload: any }>) {
  const resp = await fetch(`${API_BASE}/cron/jobs/${jobId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`Update failed: ${resp.status}`);
  return resp.json();
}

export async function deleteCronJob(jobId: string) {
  const resp = await fetch(`${API_BASE}/cron/jobs/${jobId}`, { method: "DELETE" });
  if (!resp.ok) throw new Error(`Delete failed: ${resp.status}`);
  return resp.json();
}

export async function runCronJob(jobId: string) {
  const resp = await fetch(`${API_BASE}/cron/jobs/${jobId}/run`, { method: "POST" });
  if (!resp.ok) throw new Error(`Trigger failed: ${resp.status}`);
  return resp.json();
}

// --- Main Session API ---

export async function fetchMainSession(agentId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/session`);
  return resp.json();
}

export async function fetchMainSessionMessages(agentId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/session/messages`);
  return resp.json();
}

export async function resetMainSession(agentId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/session/reset`, {
    method: "POST",
  });
  return resp.json();
}

// --- Model API ---

export async function fetchModels() {
  const resp = await fetch(`${API_BASE}/models`);
  return resp.json();
}

export async function fetchCurrentModel(agentId: string) {
  const resp = await fetch(`${API_BASE}/models/current/${agentId}`);
  if (resp.status === 404) return null;
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

export async function switchModel(agentId: string, model: string, scope: "agent" | "default" = "agent") {
  const resp = await fetch(`${API_BASE}/models/switch/${agentId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, scope }),
  });
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  const data = await resp.json();
  if (data?.status === "error") {
    throw new Error(data?.error || "Model switch failed");
  }
  return data;
}

export async function updateSecrets(path: string, value: string) {
  const resp = await fetch(`${API_BASE}/config/secrets`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, value }),
  });
  if (!resp.ok) throw new Error(`Update failed: ${resp.status}`);
  return resp.json();
}

export async function fetchRawConfig(): Promise<any> {
  const resp = await fetch(`${API_BASE}/config/raw`);
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

// --- Legacy API compat ---

export async function fetchSessions(agentId: string): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/sessions`);
  return resp.json();
}

export async function fetchHistory(agentId: string, sessionId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/sessions/${sessionId}/history`);
  return resp.json();
}

export async function fetchMessages(agentId: string, sessionId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/sessions/${sessionId}/messages`);
  return resp.json();
}

export async function readFile(agentId: string, path: string): Promise<{ path: string; content: string }> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/files?path=${encodeURIComponent(path)}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error((err as any)?.detail || `Read failed: ${resp.status}`);
  }
  return resp.json();
}

export async function saveFile(agentId: string, path: string, content: string) {
  return fetch(`${API_BASE}/agents/${agentId}/files`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content }),
  });
}

export async function fetchSkills(agentId: string): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/skills`);
  return resp.json();
}

export interface ToolItem {
  name: string;
  description: string;
  category: string;
  allowed: boolean;
}

export async function fetchTools(agentId: string): Promise<ToolItem[]> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/tools`);
  return resp.json();
}

export async function updateToolPolicy(agentId: string, toolName: string, allowed: boolean): Promise<{ status: string; config: any }> {
  const resp = await fetch(`${API_BASE}/config/tools-policy`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_id: agentId, tool_name: toolName, allowed }),
  });
  if (!resp.ok) throw new Error(`Update failed: ${resp.status}`);
  return resp.json();
}

export async function fetchToolsCatalog(): Promise<{ tools: string[] }> {
  const resp = await fetch(`${API_BASE}/tools/catalog`);
  return resp.json();
}

export interface SubagentTreeItem {
  run_id: string;
  label: string;
  task: string;
  target_agent_id: string;
  status: string;
  state?: "running" | "succeeded" | "failed" | "cancelled" | "timed_out" | "interrupted" | "orphaned";
  terminal_reason?: string | null;
  elapsed: number | null;
  duration_ms?: number | null;
  started_at?: number | null;
  ended_at?: number | null;
  result_summary: string;
  messages: { role: string; content: string; tool_calls?: any[] }[];
  created_at: number;
  spawn_depth?: number;
  requester_session_key?: string;
  child_session_key?: string;
  result_delivery_state?: "pending" | "queued" | "delivering" | "retrying" | "delivered" | "dropped";
  delivery_work_id?: string | null;
  announce_retry_count?: number;
  archive_at_ms?: number | null;
  descendants_active_count?: number;
  children?: SubagentTreeItem[];
}

export interface SubagentsResponse {
  tree: SubagentTreeItem[];
  flat: SubagentTreeItem[];
  include_recent_minutes?: number;
}

export async function fetchSubagents(
  agentId: string,
  sessionId?: string,
  includeRecentMinutes?: number
): Promise<SubagentsResponse> {
  const search = new URLSearchParams();
  if (sessionId) search.set("session_id", sessionId);
  if (includeRecentMinutes != null && includeRecentMinutes > 0) {
    search.set("include_recent_minutes", String(includeRecentMinutes));
  }
  const params = search.toString() ? `?${search}` : "";
  const resp = await fetch(`${API_BASE}/agents/${agentId}/subagents${params}`);
  const data = await resp.json();
  if (Array.isArray(data)) {
    return { tree: data, flat: data };
  }
  return data;
}

export async function killSubagent(
  agentId: string,
  target: string,
  sessionId?: string,
): Promise<{ ok: boolean;[k: string]: any }> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/subagents/kill`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target, session_id: sessionId }),
  });
  return resp.json();
}

export async function steerSubagent(
  agentId: string,
  runId: string,
  message: string,
): Promise<{ ok: boolean;[k: string]: any }> {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/subagents/steer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, message }),
  });
  return resp.json();
}

export function subscribeAgentEvents(
  agentId: string,
  onEvent: (event: SSEEvent) => void,
  onError?: (error: unknown) => void,
): () => void {
  const url = `${API_BASE}/agents/${agentId}/events`;
  let retryCount = 0;
  const maxRetries = 5;
  const baseDelay = 1000; // 1 second
  let es: EventSource | null = null;
  let closed = false;

  function connect() {
    if (closed) return;
    es = new EventSource(url);

    es.onmessage = (evt) => {
      retryCount = 0; // Reset retry count on success
      try {
        const parsed = JSON.parse(evt.data) as SSEEvent;
        onEvent(parsed);
      } catch {
        // ignore malformed events
      }
    };

    es.onerror = () => {
      if (closed) return;

      if (retryCount >= maxRetries) {
        es?.close();
        onError?.(new Error("Max retries reached"));
        return;
      }

      // Exponential backoff
      const delay = baseDelay * Math.pow(2, retryCount);
      retryCount++;

      es?.close();
      setTimeout(connect, delay);
    };
  }

  connect();

  return () => {
    closed = true;
    es?.close();
  };
}

export async function compressSession(agentId: string, sessionId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/sessions/${sessionId}/compress`, {
    method: "POST",
  });
  return resp.json();
}

export async function resetSession(agentId: string, sessionId: string) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/sessions/${sessionId}/reset`, {
    method: "POST",
  });
  return resp.json();
}

export async function fetchConfig(): Promise<any> {
  const resp = await fetch(`${API_BASE}/config`);
  return resp.json();
}

export async function fetchConfigPath(): Promise<{ path: string }> {
  const resp = await fetch(`${API_BASE}/config/path`);
  return resp.json();
}

export async function fetchInitStatus(): Promise<InitStatus> {
  const resp = await fetch(`${API_BASE}/init/status`);
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

export async function replaceConfig(config: Record<string, any>): Promise<any> {
  const resp = await fetch(`${API_BASE}/config/replace`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

export async function updateConfig(updates: Record<string, any>): Promise<any> {
  const resp = await fetch(`${API_BASE}/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updates }),
  });
  if (!resp.ok) throw new Error(await readErrorMessage(resp));
  return resp.json();
}

export async function updateRagMode(enabled: boolean) {
  return updateConfig({ rag_mode: enabled });
}

export async function updateSkillEnabled(agentId: string, skillName: string, enabled: boolean) {
  const resp = await fetch(`${API_BASE}/agents/${agentId}/skills`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ skill_name: skillName, enabled }),
  });
  if (!resp.ok) throw new Error(`Update failed: ${resp.status}`);
  return resp.json();
}

// ---- Memory API ----

export async function memStats(agentId: string): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId });
  const r = await fetch(`${API_BASE}/mem/stats?${q}`);
  return r.json();
}

export async function memTasks(agentId: string, params?: { status?: string; limit?: number; offset?: number }): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId });
  if (params?.status) q.set("status", params.status);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const r = await fetch(`${API_BASE}/mem/tasks?${q}`);
  return r.json();
}

export async function memTaskDetail(agentId: string, taskId: string): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId });
  const r = await fetch(`${API_BASE}/mem/task/${taskId}?${q}`);
  return r.json();
}

export async function memBoundaryReviews(agentId: string, limit = 100): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId, limit: String(limit) });
  const r = await fetch(`${API_BASE}/mem/boundary-reviews?${q}`);
  if (!r.ok) throw new Error(await readErrorMessage(r));
  return r.json();
}

export async function resolveMemBoundaryReview(
  agentId: string,
  reviewId: string,
  input: {
    action: "assign_current" | "create_new" | "assign_other" | "orphan";
    target_task_id?: string;
    note?: string;
  },
): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId });
  const r = await fetch(`${API_BASE}/mem/boundary-reviews/${encodeURIComponent(reviewId)}/resolve?${q}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await readErrorMessage(r));
  return r.json();
}

export async function memSkills(agentId: string, params?: { status?: string }): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId });
  if (params?.status) q.set("status", params.status);
  const r = await fetch(`${API_BASE}/mem/skills?${q}`);
  return r.json();
}

export async function memSkillDetail(agentId: string, skillId: string): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId });
  const r = await fetch(`${API_BASE}/mem/skill/${skillId}?${q}`);
  return r.json();
}

export async function memMemories(agentId: string, params?: { limit?: number; page?: number; session?: string; role?: string }): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId });
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.page) q.set("page", String(params.page));
  if (params?.session) q.set("session", params.session);
  if (params?.role) q.set("role", params.role);
  const r = await fetch(`${API_BASE}/mem/memories?${q}`);
  return r.json();
}

export async function memSearch(agentId: string, query: string, limit?: number): Promise<any> {
  const q = new URLSearchParams({ agent_id: agentId, q: query });
  if (limit) q.set("limit", String(limit));
  const r = await fetch(`${API_BASE}/mem/search?${q}`);
  return r.json();
}
