"use client";

import { useState } from "react";
import { Globe, Search, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SparkleBackground } from "@/components/ui/sparkle-background";
import { runHybridSearch } from "@/lib/api-client";

export default function ResearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"hybrid" | "vector" | "sparse" | "graph" | "web">("hybrid");
  const [includeWeb, setIncludeWeb] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await runHybridSearch({
        query,
        mode,
        include_web: includeWeb,
        enable_reranking: true,
        top_k: 8,
      });
      setResult(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen space-y-6 pb-12">
      <SparkleBackground density={12} />

      <div className="relative z-10">
        <h1 className="text-3xl font-bold text-foreground">Retrieval Sources & Web Research</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Compare Private Knowledge Base evidence ([KB-n]) alongside separate Tavily external web research ([WEB-n]).
        </p>
      </div>

      <Card className="border-border/60 bg-card/60 backdrop-blur-xl relative z-10">
        <CardContent className="p-4 flex flex-wrap items-center gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Enter a research query across Qdrant, Neo4j, and Tavily..."
            className="flex-1 min-w-[260px] rounded-xl border border-border bg-background px-4 py-2.5 text-sm"
          />
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as any)}
            className="rounded-xl border border-border bg-background px-3 py-2.5 text-xs"
          >
            <option value="hybrid">Hybrid (Dense + Sparse + Graph)</option>
            <option value="vector">Dense Vector Only</option>
            <option value="sparse">Sparse / BM25 Only</option>
            <option value="graph">Neo4j Graph Only</option>
            <option value="web">Tavily External Web Only</option>
          </select>
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={includeWeb}
              onChange={(e) => setIncludeWeb(e.target.checked)}
            />
            Include Tavily Web
          </label>
          <Button onClick={handleSearch} disabled={loading} className="btn-primary-gradient">
            <Search className="h-4 w-4 mr-1.5" /> {loading ? "Searching..." : "Search"}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive relative z-10">
          {error}
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative z-10">
          <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Layers className="h-4 w-4 text-primary" /> Private Knowledge Base Evidence ([KB-n])
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {(result.results || []).filter((r: any) => r.source_kind !== "web").length === 0 ? (
                <p className="text-xs text-muted-foreground">No private KB chunks matched this query.</p>
              ) : (
                (result.results || [])
                  .filter((r: any) => r.source_kind !== "web")
                  .map((r: any, idx: number) => (
                    <div key={idx} className="rounded-lg border border-border/60 bg-background/70 p-3 text-xs space-y-1">
                      <div className="flex justify-between font-mono">
                        <span className="text-primary font-semibold">
                          [{r.citation_id || `KB-${idx + 1}`}] {r.filename}
                          {r.page_number ? ` (p.${r.page_number})` : ""}
                        </span>
                        <span className="text-muted-foreground">
                          dense=#{r.dense_rank ?? "—"} sparse=#{r.sparse_rank ?? "—"}
                        </span>
                      </div>
                      <p className="text-muted-foreground">{r.text_snippet || r.text}</p>
                    </div>
                  ))
              )}
            </CardContent>
          </Card>

          <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Globe className="h-4 w-4 text-emerald-400" /> External Web Research ([WEB-n])
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {result.status === "unconfigured" || result.diagnostics?.web_status === "unconfigured" ? (
                <p className="text-xs text-amber-400">
                  Tavily API key (`TAVILY_API_KEY`) is not configured in `apps/backend/.env`. Private KB retrieval remains active.
                </p>
              ) : (result.web_results || (mode === "web" ? result.results : []) || []).length === 0 ? (
                <p className="text-xs text-muted-foreground">No external web results returned.</p>
              ) : (
                (result.web_results || result.results || []).map((w: any, idx: number) => (
                  <div key={idx} className="rounded-lg border border-border/60 bg-background/70 p-3 text-xs space-y-1">
                    <div className="flex justify-between">
                      <span className="font-semibold text-emerald-400">
                        [{w.citation_id || `WEB-${idx + 1}`}] {w.title || w.filename}
                      </span>
                    </div>
                    {w.url && (
                      <a
                        href={w.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-[11px] text-primary underline block truncate"
                      >
                        {w.url}
                      </a>
                    )}
                    <p className="text-muted-foreground">{w.text_snippet || w.text}</p>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

