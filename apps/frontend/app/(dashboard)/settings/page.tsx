"use client";

import { useState, useEffect, useCallback } from "react";
import { Settings as SettingsIcon, Save, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SparkleBackground } from "@/components/ui/sparkle-background";
import { apiFetch, getRuntimeSettings } from "@/lib/api-client";

export default function SettingsPage() {
  const [config, setConfig] = useState<any>(null);
  const [saving, setSaving] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);
  const [form, setForm] = useState({
    EMBEDDING_PROVIDER: "bge-m3",
    EMBEDDING_MODEL: "BAAI/bge-m3",
    CHUNK_STRATEGY: "structure_aware",
    CHUNK_SIZE: 800,
    CHUNK_OVERLAP: 80,
    RETRIEVAL_TOP_K: 6,
    RETRIEVAL_RRF_K: 60,
    ENABLE_RERANKING: true,
    ENABLE_SPARSE_RETRIEVAL: true,
    LLM_PROVIDER: "gemini",
  });

  const load = useCallback(async () => {
    try {
      const data = await getRuntimeSettings();
      setConfig(data);
      setForm({
        EMBEDDING_PROVIDER: data.embedding?.provider ?? "bge-m3",
        EMBEDDING_MODEL: data.embedding?.model ?? "BAAI/bge-m3",
        CHUNK_STRATEGY: data.chunking?.strategy ?? "structure_aware",
        CHUNK_SIZE: data.chunking?.chunk_size ?? 800,
        CHUNK_OVERLAP: data.chunking?.chunk_overlap ?? 80,
        RETRIEVAL_TOP_K: data.retrieval?.top_k ?? 6,
        RETRIEVAL_RRF_K: data.retrieval?.rrf_k ?? 60,
        ENABLE_RERANKING: data.retrieval?.enable_reranking ?? true,
        ENABLE_SPARSE_RETRIEVAL: data.retrieval?.enable_sparse_retrieval ?? true,
        LLM_PROVIDER: data.llm?.default_provider ?? "gemini",
      });
    } catch (e) {
      setBanner((e as Error).message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleSave = async () => {
    setSaving(true);
    setBanner(null);
    try {
      const res = await apiFetch("/api/v1/settings", {
        method: "PATCH",
        body: JSON.stringify(form),
      });
      setConfig(res.settings);
      setBanner("Runtime RAG configuration updated.");
    } catch (e) {
      setBanner((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="relative min-h-screen space-y-6 pb-12">
      <SparkleBackground density={12} />

      <div className="flex items-center justify-between relative z-10">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Hybrid RAG Settings & Providers</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Inspect active BGE-M3 / MiniLM Qdrant collections, chunking parameters, LLM adapters, and LangSmith status.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5 mr-1.5" /> Refresh
          </Button>
          <Button className="btn-primary-gradient" size="sm" onClick={handleSave} disabled={saving}>
            <Save className="h-3.5 w-3.5 mr-1.5" /> {saving ? "Saving..." : "Save Settings"}
          </Button>
        </div>
      </div>

      {banner && (
        <div className="rounded-xl border border-primary/30 bg-primary/10 px-4 py-2.5 text-xs text-primary relative z-10">
          {banner}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <SettingsIcon className="h-4 w-4 text-primary" /> Chunking & Hybrid Retrieval Tuning
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1">
                <span className="text-muted-foreground">Embedding Provider (HF / OpenLM)</span>
                <select
                  value={form.EMBEDDING_PROVIDER}
                  onChange={(e) => {
                    const nextProv = e.target.value;
                    const nextModel =
                      nextProv === "bge-m3"
                        ? "BAAI/bge-m3"
                        : nextProv === "sentence-transformers"
                        ? "sentence-transformers/all-MiniLM-L6-v2"
                        : form.EMBEDDING_MODEL;
                    setForm({ ...form, EMBEDDING_PROVIDER: nextProv, EMBEDDING_MODEL: nextModel });
                  }}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                >
                  <option value="bge-m3">bge-m3 (Local / HF)</option>
                  <option value="sentence-transformers">sentence-transformers (MiniLM)</option>
                  <option value="huggingface">huggingface (HF Hub / Router)</option>
                  <option value="openlm">openlm (OpenLM / HF Fallback)</option>
                </select>
              </label>

              <label className="space-y-1">
                <span className="text-muted-foreground">Embedding Model</span>
                <select
                  value={form.EMBEDDING_MODEL}
                  onChange={(e) => {
                    const m = e.target.value;
                    const p =
                      m !== "BAAI/bge-m3" && form.EMBEDDING_PROVIDER === "bge-m3"
                        ? "sentence-transformers"
                        : form.EMBEDDING_PROVIDER;
                    setForm({ ...form, EMBEDDING_MODEL: m, EMBEDDING_PROVIDER: p });
                  }}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                >
                  <option value="BAAI/bge-m3">BAAI/bge-m3 (1024d)</option>
                  <option value="sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2 (384d)</option>
                </select>
              </label>

              <label className="space-y-1">
                <span className="text-muted-foreground">Chunk Strategy</span>
                <select
                  value={form.CHUNK_STRATEGY}
                  onChange={(e) => setForm({ ...form, CHUNK_STRATEGY: e.target.value })}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                >
                  <option value="structure_aware">structure_aware</option>
                  <option value="table_aware">table_aware</option>
                  <option value="parent_child">parent_child</option>
                  <option value="token_aware">token_aware</option>
                  <option value="recursive">recursive</option>
                </select>
              </label>

              <label className="space-y-1">
                <span className="text-muted-foreground">Default LLM Provider</span>
                <select
                  value={form.LLM_PROVIDER}
                  onChange={(e) => setForm({ ...form, LLM_PROVIDER: e.target.value })}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                >
                  <option value="auto">auto (Gemini &rarr; Groq &rarr; OpenLM)</option>
                  <option value="gemini">gemini</option>
                  <option value="openlm">openlm</option>
                  <option value="groq">groq</option>
                </select>
              </label>

              <label className="space-y-1">
                <span className="text-muted-foreground">Chunk Size (chars)</span>
                <input
                  type="number"
                  value={form.CHUNK_SIZE}
                  onChange={(e) => {
                    const sz = Number(e.target.value);
                    setForm({ ...form, CHUNK_SIZE: sz, CHUNK_OVERLAP: Math.max(16, Math.round(sz * 0.1)) });
                  }}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                />
              </label>

              <label className="space-y-1">
                <span className="text-muted-foreground">
                  Chunk Overlap ({form.CHUNK_SIZE > 0 ? Math.round((form.CHUNK_OVERLAP / form.CHUNK_SIZE) * 100) : 10}%)
                </span>
                <input
                  type="number"
                  value={form.CHUNK_OVERLAP}
                  onChange={(e) => setForm({ ...form, CHUNK_OVERLAP: Number(e.target.value) })}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                />
              </label>

              <label className="space-y-1">
                <span className="text-muted-foreground">Retrieval Top-K</span>
                <input
                  type="number"
                  value={form.RETRIEVAL_TOP_K}
                  onChange={(e) => setForm({ ...form, RETRIEVAL_TOP_K: Number(e.target.value) })}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                />
              </label>

              <label className="space-y-1">
                <span className="text-muted-foreground">RRF Constant (k)</span>
                <input
                  type="number"
                  value={form.RETRIEVAL_RRF_K}
                  onChange={(e) => setForm({ ...form, RETRIEVAL_RRF_K: Number(e.target.value) })}
                  className="w-full rounded-lg border border-border bg-background px-2.5 py-2"
                />
              </label>
            </div>

            <div className="flex gap-6 pt-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.ENABLE_SPARSE_RETRIEVAL}
                  onChange={(e) => setForm({ ...form, ENABLE_SPARSE_RETRIEVAL: e.target.checked })}
                />
                Enable Sparse / BM25 Branch
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.ENABLE_RERANKING}
                  onChange={(e) => setForm({ ...form, ENABLE_RERANKING: e.target.checked })}
                />
                Enable Cross-Feature Reranker
              </label>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 bg-card/60 backdrop-blur-xl">
          <CardHeader>
            <CardTitle className="text-base">Verified Service & Provider Readiness</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2.5 text-xs">
            <div className="rounded-lg bg-background/60 p-3 border border-border/50 space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Active Embedding Model</span>
                <span className="font-mono text-foreground">{config?.embedding?.model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Scoped Qdrant Collection</span>
                <span className="font-mono text-primary truncate max-w-[220px]">
                  {config?.embedding?.active_qdrant_collection}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Vector Dimension</span>
                <span className="font-mono">{config?.embedding?.dimension}d</span>
              </div>
            </div>

            <div className="rounded-lg bg-background/60 p-3 border border-border/50 space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Gemini API</span>
                <span>{config?.llm?.providers?.gemini?.configured ? `Configured (${config?.llm?.providers?.gemini?.default_model})` : "Unconfigured"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Groq API</span>
                <span>{config?.llm?.providers?.groq?.configured ? `Configured (${config?.llm?.providers?.groq?.default_model})` : "Unconfigured"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">OpenLM Endpoint</span>
                <span>{config?.llm?.providers?.openlm?.configured ? config?.llm?.providers?.openlm?.base_url : "Unconfigured"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Tavily Web Search</span>
                <span>{config?.tavily?.status ?? "unconfigured"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">LangSmith Tracing</span>
                <span>
                  {config?.observability?.tracing_active
                    ? `Active (${config?.observability?.project})`
                    : "Inactive (No API Key)"}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

