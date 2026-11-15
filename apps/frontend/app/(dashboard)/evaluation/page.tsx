"use client";

import { useState, useEffect, useCallback } from "react";
import { Activity, Play, Search, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SparkleBackground } from "@/components/ui/sparkle-background";
import {
  getEvaluationDashboard,
  triggerEvaluationBenchmark,
  runHybridSearch,
  getRuntimeSettings,
} from "@/lib/api-client";

export default function EvaluationPage() {
  const [telemetry, setTelemetry] = useState<any>(null);
  const [settings, setSettings] = useState<any>(null);
  const [benchmark, setBenchmark] = useState<any>(null);
  const [runningEval, setRunningEval] = useState(false);
  const [diagQuery, setDiagQuery] = useState("");
  const [diagMode, setDiagMode] = useState<"hybrid" | "vector" | "sparse" | "graph" | "web">("hybrid");
  const [diagResult, setDiagResult] = useState<any>(null);
  const [searching, setSearching] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [ev, st] = await Promise.all([getEvaluationDashboard(), getRuntimeSettings()]);
      setTelemetry(ev);
      setSettings(st);
      if (ev.benchmark_runs?.length > 0) {
        setBenchmark(ev.benchmark_runs[0]);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleRunBenchmark = async () => {
    setRunningEval(true);
    try {
      const res = await triggerEvaluationBenchmark(5);
      setBenchmark(res);
      await loadData();
    } finally {
      setRunningEval(false);
    }
  };

  const handleDiagnosticSearch = async () => {
    if (!diagQuery.trim()) return;
    setSearching(true);
    try {
      const res = await runHybridSearch({
        query: diagQuery,
        mode: diagMode,
        enable_reranking: true,
        top_k: 6,
      });
      setDiagResult(res);
      await loadData();
    } finally {
      setSearching(false);
    }
  };

  const summary = telemetry?.telemetry_summary;
  const comp = benchmark?.comparison;

  return (
    <div className="relative min-h-screen space-y-6 pb-12">
      <SparkleBackground density={12} />

      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 relative z-10">
        <div>
          <h1 className="text-3xl font-bold text-foreground">
            RAG Observability, Diagnostics & Evaluation
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Inspect Dense vs Sparse/BM25 vs RRF vs Reranked Hybrid retrieval, LangSmith telemetry, and empirical RAG metrics.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={loadData}>
            <RefreshCw className="h-3.5 w-3.5 mr-1.5" /> Refresh
          </Button>
          <Button
            className="btn-primary-gradient"
            size="sm"
            onClick={handleRunBenchmark}
            disabled={runningEval}
          >
            <Play className="h-3.5 w-3.5 mr-1.5" />
            {runningEval ? "Running Benchmark..." : "Compare Dense vs Hybrid vs Reranked"}
          </Button>
        </div>
      </div>

      {/* Live Telemetry Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 relative z-10">
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
          <CardContent className="p-4">
            <span className="text-xs text-muted-foreground">Traced Requests</span>
            <p className="text-2xl font-bold text-foreground mt-1">
              {summary?.total_requests ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
          <CardContent className="p-4">
            <span className="text-xs text-muted-foreground">Avg Latency</span>
            <p className="text-2xl font-bold text-foreground mt-1">
              {summary?.avg_total_ms ?? 0} ms
            </p>
          </CardContent>
        </Card>
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
          <CardContent className="p-4">
            <span className="text-xs text-muted-foreground">P95 Latency</span>
            <p className="text-2xl font-bold text-foreground mt-1">
              {summary?.p95_total_ms ?? 0} ms
            </p>
          </CardContent>
        </Card>
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
          <CardContent className="p-4">
            <span className="text-xs text-muted-foreground">Active Embedding Space</span>
            <p className="text-sm font-mono font-semibold text-primary mt-2 truncate">
              {settings?.embedding?.model ?? "BAAI/bge-m3"} ({settings?.embedding?.dimension ?? 1024}d)
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Comparative Benchmark Table */}
      {comp && (
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl relative z-10">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" /> Comparative Retrieval Evaluation (Dense vs Hybrid RRF vs Reranked Hybrid)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border/60 text-muted-foreground">
                    <th className="text-left py-2 px-3">Strategy</th>
                    <th className="text-left py-2 px-3">Context Precision</th>
                    <th className="text-left py-2 px-3">Context Recall</th>
                    <th className="text-left py-2 px-3">Retrieval Relevance</th>
                    <th className="text-left py-2 px-3">Faithfulness</th>
                    <th className="text-left py-2 px-3">Citation Correctness</th>
                    <th className="text-left py-2 px-3">Abstention Acc.</th>
                    <th className="text-left py-2 px-3">Avg Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {["dense_only", "hybrid_rrf", "hybrid_reranked"].map((key) => {
                    const row = comp[key];
                    if (!row) return null;
                    return (
                      <tr key={key} className="border-b border-border/30">
                        <td className="py-2.5 px-3 font-mono font-semibold text-primary">{key}</td>
                        <td className="py-2.5 px-3">{row.avg_context_precision}</td>
                        <td className="py-2.5 px-3">{row.avg_context_recall}</td>
                        <td className="py-2.5 px-3 font-semibold">{row.avg_retrieval_relevance}</td>
                        <td className="py-2.5 px-3">{row.avg_faithfulness}</td>
                        <td className="py-2.5 px-3">{row.avg_citation_correctness}</td>
                        <td className="py-2.5 px-3">{row.abstention_accuracy}</td>
                        <td className="py-2.5 px-3">{row.avg_latency_ms} ms</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Direct Retrieval Channel Inspector */}
      <Card className="border-border/60 bg-card/60 backdrop-blur-xl relative z-10">
        <CardHeader>
          <CardTitle className="text-base">Direct Retrieval & Fusion Inspector</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <input
              type="text"
              value={diagQuery}
              onChange={(e) => setDiagQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleDiagnosticSearch()}
              placeholder="Test a semantic, acronym (e.g. ECC, SIMT), exact keyword, or Tavily web query..."
              className="flex-1 min-w-[260px] rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
            <select
              value={diagMode}
              onChange={(e) => setDiagMode(e.target.value as any)}
              className="rounded-lg border border-border bg-background px-3 py-2 text-xs"
            >
              <option value="hybrid">Hybrid (Dense + Sparse + Graph + Rerank)</option>
              <option value="vector">Dense Vector (BGE-M3)</option>
              <option value="sparse">Sparse / BM25 Lexical</option>
              <option value="graph">Neo4j Graph Context</option>
              <option value="web">Tavily External Web</option>
            </select>
            <Button onClick={handleDiagnosticSearch} disabled={searching}>
              <Search className="h-4 w-4 mr-1.5" /> Inspect
            </Button>
          </div>

          {diagResult && (
            <div className="space-y-2 pt-2">
              {diagResult.status === "unconfigured" && (
                <p className="text-xs text-amber-400">{diagResult.detail}</p>
              )}
              {(diagResult.results || []).map((item: any, idx: number) => (
                <div
                  key={idx}
                  className="rounded-lg border border-border/60 bg-background/70 p-3 text-xs space-y-1"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono font-semibold text-primary">
                      [{item.citation_id || `#${idx + 1}`}] {item.filename || item.title}
                      {item.page_number ? ` (p. ${item.page_number})` : ""}
                    </span>
                    <span className="font-mono text-[11px] text-muted-foreground">
                      dense=#{item.dense_rank ?? "—"} | sparse=#{item.sparse_rank ?? "—"} | fusion=
                      {item.fusion_score ?? "—"} | rerank={item.rerank_score ?? "—"}
                    </span>
                  </div>
                  <p className="text-muted-foreground">{item.text_snippet || item.text}</p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

