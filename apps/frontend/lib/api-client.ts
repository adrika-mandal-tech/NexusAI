/**
 * Typed fetch & SSE streaming client for the NexusAI Hybrid RAG Backend.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  latency_ms?: number;
  detail?: string;
  dependencies?: Record<string, any>;
}

export interface SystemHealth {
  api: HealthResponse | null;
  dependencies: Record<string, any> | null;
  fetchedAt: string;
}

export async function getSystemHealth(): Promise<SystemHealth> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/health/dependencies`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    const data = await res.json();
    return {
      api: { status: data.status === "ok" ? "ok" : "degraded" },
      dependencies: data.dependencies ?? null,
      fetchedAt: new Date().toISOString(),
    };
  } catch {
    return {
      api: { status: "error", detail: "Connection failed" },
      dependencies: null,
      fetchedAt: new Date().toISOString(),
    };
  }
}

export async function apiFetch<T = any>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(errorData?.detail || response.statusText || "Request failed");
  }

  if (response.status === 204) return null as T;
  return response.json();
}

export interface UploadResponse {
  source_id: string;
  document_id?: string;
  filename?: string;
  duplicate?: boolean;
  recommended_parser?: string;
  status: "processing" | "ready" | "failed";
  chunks: number;
  error?: string;
}

export async function uploadFile(
  file: File,
  sourceType = "pdf",
  forceReindex = false
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("source_type", sourceType);
  form.append("user_id", "00000000-0000-0000-0000-000000000000");
  form.append("force_reindex", String(forceReindex));

  const res = await fetch(`${API_BASE_URL}/api/v1/knowledge/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail || res.statusText);
  }
  return res.json();
}

export interface SourceStatus {
  id: string;
  title: string;
  filename?: string;
  source_type: string;
  status: "pending" | "processing" | "ready" | "failed";
  chunk_count: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
}

export async function getSources(): Promise<SourceStatus[]> {
  return apiFetch<SourceStatus[]>(
    `/api/v1/knowledge/sources?user_id=00000000-0000-0000-0000-000000000000`
  );
}

export async function getSource(sourceId: string): Promise<SourceStatus> {
  return apiFetch<SourceStatus>(
    `/api/v1/knowledge/sources/${sourceId}?user_id=00000000-0000-0000-0000-000000000000`
  );
}

export async function reindexSource(sourceId: string): Promise<UploadResponse> {
  return apiFetch<UploadResponse>(
    `/api/v1/documents/${sourceId}/reindex?user_id=00000000-0000-0000-0000-000000000000`,
    { method: "POST" }
  );
}

export async function deleteSource(sourceId: string): Promise<void> {
  await apiFetch<void>(
    `/api/v1/documents/${sourceId}?user_id=00000000-0000-0000-0000-000000000000`,
    { method: "DELETE" }
  );
}

export async function pollSourceUntilReady(
  sourceId: string,
  onUpdate: (source: SourceStatus) => void,
  intervalMs = 1500,
  timeoutMs = 300000
): Promise<SourceStatus> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    const source = await getSource(sourceId);
    onUpdate(source);
    if (source.status === "ready" || source.status === "failed") {
      return source;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error("Polling timeout");
}

export interface Citation {
  citation_id?: string;
  source_kind?: "knowledge_base" | "web";
  chunk_id?: string;
  document_id?: string | null;
  chunk_index?: number;
  text_snippet: string;
  page_number?: number | null;
  section_title?: string | null;
  filename: string;
  source_type?: string;
  url?: string;
  score: number;
  dense_rank?: number | null;
  sparse_rank?: number | null;
  graph_rank?: number | null;
  final_rank?: number | null;
  dense_score?: number | null;
  sparse_score?: number | null;
  bm25_score?: number | null;
  fusion_score?: number | null;
  rerank_score?: number | null;
  retrieval_channel?: string;
}

export interface LiveToolTrace {
  tool: string;
  status: string;
  candidates: number;
  latency_ms: number;
}

export interface LiveEvalMetrics {
  hallucination_rate_pct: number;
  correctness_pct: number;
  accuracy_pct: number;
  faithfulness_pct: number;
  context_precision_pct: number;
  context_recall_pct: number;
  tool_calling_accuracy_pct: number;
  task_success_rate_pct: number;
  abstained?: boolean;
  claims_evaluated?: number;
  claims_supported?: number;
  provider_used?: string;
  frameworks?: string[];
  tools_invoked?: LiveToolTrace[];
}

export interface RetrievalDiagnostics {
  request_id?: string;
  mode?: string;
  fusion_strategy?: string;
  reranked?: boolean;
  dense_candidates?: number;
  sparse_candidates?: number;
  graph_candidates?: number;
  graph_entities_count?: number;
  fused_unique_candidates?: number;
  final_selected_chunks?: number;
  web_results_count?: number;
  graph_status?: string;
  web_status?: string;
  provider_used?: string;
  model_used?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  expanded_acronyms?: Record<string, string>;
  latency_ms?: Record<string, number>;
  eval_metrics?: LiveEvalMetrics;
  citation_validation?: {
    valid_citation_ids: string[];
    cited_in_answer: string[];
    stripped_fabricated_citations: string[];
    grounding_verified: boolean;
  };
}

export interface ConversationItem {
  id: string;
  title: string | null;
  message_count?: number;
  created_at: string;
  updated_at: string;
  messages?: {
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    citations?: Citation[];
    metadata?: Record<string, any>;
    created_at: string;
  }[];
}

export async function listConversations(): Promise<ConversationItem[]> {
  return apiFetch<ConversationItem[]>(
    `/api/v1/conversations?user_id=00000000-0000-0000-0000-000000000000`
  );
}

export async function createConversation(title?: string): Promise<ConversationItem> {
  return apiFetch<ConversationItem>(
    `/api/v1/conversations?user_id=00000000-0000-0000-0000-000000000000`,
    {
      method: "POST",
      body: JSON.stringify({ title: title || "New Analysis" }),
    }
  );
}

export async function getConversation(conversationId: string): Promise<ConversationItem> {
  return apiFetch<ConversationItem>(
    `/api/v1/conversations/${conversationId}?user_id=00000000-0000-0000-0000-000000000000`
  );
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await apiFetch<void>(
    `/api/v1/conversations/${conversationId}?user_id=00000000-0000-0000-0000-000000000000`,
    { method: "DELETE" }
  );
}

export interface ChartSpec {
  type: "bar" | "line" | "donut" | "scatter";
  title: string;
  subtitle?: string;
  x_label?: string;
  y_label?: string;
  labels: string[];
  series: {
    name: string;
    data: number[];
    color?: string;
  }[];
  kpis?: {
    label: string;
    value: string;
  }[];
}

export interface ChatRequestOptions {
  conversation_id?: string | null;
  document_id?: string | null;
  document_ids?: string[];
  provider?: string;
  model?: string;
  mode?: "hybrid" | "dense" | "sparse" | "graph" | "web";
  fusion_strategy?: "rrf" | "normalized_linear" | "qdrant_native_rrf";
  enable_reranking?: boolean;
  include_web?: boolean;
  top_k?: number;
}

export async function streamChatMessage(
  message: string,
  onDelta: (text: string) => void,
  onSources: (sources: Citation[], diagnostics?: RetrievalDiagnostics) => void,
  onDone: (
    answer: string,
    citations: Citation[],
    diagnostics?: RetrievalDiagnostics,
    chart?: ChartSpec
  ) => void,
  onError: (error: Error) => void,
  options: ChatRequestOptions = {},
  abortSignal?: AbortSignal
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/knowledge/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, ...options }),
      signal: abortSignal ?? AbortSignal.timeout(120000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.detail || res.statusText);
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error("No response stream body");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = "message";
    let fullAnswer = "";
    let citations: Citation[] = [];
    let diagnostics: RetrievalDiagnostics | undefined;
    let doneCalled = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const rawLine of lines) {
        const trimmed = rawLine.trim();
        if (!trimmed) {
          currentEvent = "message";
          continue;
        }

        if (trimmed.startsWith("event:")) {
          currentEvent = trimmed.slice(6).trim();
          continue;
        }

        if (trimmed.startsWith("data:")) {
          const rawJson = trimmed.slice(5).trim();
          if (!rawJson) continue;
          try {
            const data = JSON.parse(rawJson);
            if (currentEvent === "error" || data.error) {
              throw new Error(data.error || "Stream generation failed");
            }
            if (currentEvent === "sources" || Array.isArray(data.sources)) {
              const incoming: Citation[] = Array.isArray(data.citations)
                ? data.citations
                : Array.isArray(data.sources)
                ? data.sources
                : [];
              citations = incoming;
              if (data.diagnostics) diagnostics = data.diagnostics;
              onSources(citations, diagnostics);
            } else if (typeof data.delta === "string") {
              fullAnswer += data.delta;
              onDelta(data.delta);
            }
            if (currentEvent === "done" || typeof data.answer === "string") {
              if (typeof data.answer === "string") fullAnswer = data.answer;
              if (Array.isArray(data.citations)) citations = data.citations;
              if (data.diagnostics) diagnostics = data.diagnostics;
              if (currentEvent === "done") {
                doneCalled = true;
                onDone(fullAnswer, citations, diagnostics, data.chart);
                return;
              }
            }
          } catch (parseErr) {
            if (parseErr instanceof Error && currentEvent === "error") {
              throw parseErr;
            }
          }
        }
      }
    }

    if (!doneCalled) {
      onDone(fullAnswer, citations, diagnostics);
    }
  } catch (e) {
    onError(e instanceof Error ? e : new Error(String(e)));
  }
}

export async function runHybridSearch(body: {
  query: string;
  mode?: "hybrid" | "vector" | "sparse" | "graph" | "web";
  fusion_strategy?: string;
  enable_reranking?: boolean;
  include_web?: boolean;
  top_k?: number;
}): Promise<any> {
  const endpoint =
    body.mode === "vector"
      ? "/api/v1/search/vector"
      : body.mode === "sparse"
      ? "/api/v1/search/sparse"
      : body.mode === "graph"
      ? "/api/v1/search/graph"
      : body.mode === "web"
      ? "/api/v1/search/web"
      : "/api/v1/search/hybrid";
  return apiFetch(endpoint, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getEvaluationDashboard(): Promise<any> {
  return apiFetch("/api/v1/evaluation");
}

export async function triggerEvaluationBenchmark(topK = 5): Promise<any> {
  return apiFetch("/api/v1/evaluation/run", {
    method: "POST",
    body: JSON.stringify({ top_k: topK }),
  });
}

export interface CypherConsoleResult {
  status: string;
  cypher: string;
  query_type: string;
  columns: string[];
  rows: Record<string, any>[];
  row_count: number;
  truncated: boolean;
  elapsed_ms: number;
  nodes: any[];
  edges: any[];
  stats?: Record<string, any>;
}

export async function executeCypherConsole(
  cypher: string,
  params: Record<string, any> = {},
  maxRows = 200
): Promise<CypherConsoleResult> {
  return apiFetch<CypherConsoleResult>("/api/v1/knowledge-graph/cypher", {
    method: "POST",
    body: JSON.stringify({ cypher, params, max_rows: maxRows }),
  });
}

export const executeCypher = executeCypherConsole;

export async function getRuntimeSettings(): Promise<any> {
  return apiFetch("/api/v1/settings");
}
