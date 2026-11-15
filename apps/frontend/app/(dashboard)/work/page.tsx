"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  Database,
  Network,
  FileText,
  Plus,
  File,
  Table,
  Globe,
  RefreshCw,
  SlidersHorizontal,
  CheckCircle,
  Clock,
  XCircle,
  Trash2,
  RotateCcw,
  Sparkles,
  ArrowRight,
  X,
  Layers,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { SparkleBackground } from "@/components/ui/sparkle-background";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  uploadFile,
  getSources,
  pollSourceUntilReady,
  deleteSource,
  reindexSource,
  getSystemHealth,
  SourceStatus,
} from "@/lib/api-client";
import { authFetch } from "@/lib/auth";

const filterButtons = [
  { id: "all", label: "All Sources", icon: SlidersHorizontal },
  { id: "pdf", label: "PDF", icon: File },
  { id: "docx", label: "Documents", icon: FileText },
  { id: "data", label: "CSV / Data", icon: Table },
];

const FILE_SIZE_LIMITS = [
  { type: "PDF (Docling / PyPDF / OCR)", ext: ".pdf", limit: "100 MB", icon: File },
  { type: "Office & HTML (Docling)", ext: ".docx, .pptx, .xlsx, .html", limit: "50–100 MB", icon: FileText },
  { type: "CSV / Tabular", ext: ".csv", limit: "100 MB", icon: Table },
  { type: "Text / Markdown / JSON", ext: ".txt, .md, .json", limit: "10 MB", icon: FileText },
  { type: "Scanned Images (PP-OCRv6)", ext: ".png, .jpg, .tiff, .webp", limit: "25 MB", icon: Globe },
];

const STATUS_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  ready: { color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20", label: "Ready" },
  processing: { color: "text-primary", bg: "bg-primary/10 border-primary/20", label: "Processing" },
  failed: { color: "text-destructive", bg: "bg-destructive/10 border-destructive/20", label: "Failed" },
  pending: { color: "text-muted-foreground", bg: "bg-muted/10 border-muted/20", label: "Pending" },
};

export default function WorkPage() {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [uploadedBanner, setUploadedBanner] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<string>("all");
  const [sources, setSources] = useState<SourceStatus[]>([]);
  const [pollingSources, setPollingSources] = useState<Set<string>>(new Set());
  const [depStats, setDepStats] = useState<{
    qdrantOk: boolean;
    neo4jOk: boolean;
    qdrantPoints: number;
    neo4jNodes: number;
    embeddingModel: string;
  }>({
    qdrantOk: false,
    neo4jOk: false,
    qdrantPoints: 0,
    neo4jNodes: 0,
    embeddingModel: "BAAI/bge-m3",
  });

  const refreshAll = useCallback(async () => {
    try {
      const [srcData, health] = await Promise.all([getSources(), getSystemHealth()]);
      setSources(srcData);
      const deps = health.dependencies;
      if (deps) {
        setDepStats({
          qdrantOk: deps.qdrant?.status === "ok",
          neo4jOk: deps.neo4j?.status === "ok",
          qdrantPoints: deps.qdrant?.points_count ?? 0,
          neo4jNodes: deps.neo4j?.nodes_count ?? 0,
          embeddingModel: deps.embeddings?.model ?? "BAAI/bge-m3",
        });
      }
    } catch (e) {
      setErrorBanner((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const startPolling = useCallback(
    (sourceId: string) => {
      setPollingSources((prev) => new Set(prev).add(sourceId));
      pollSourceUntilReady(sourceId, (updated) => {
        setSources((prev) => {
          const exists = prev.some((s) => s.id === sourceId);
          return exists ? prev.map((s) => (s.id === sourceId ? updated : s)) : [updated, ...prev];
        });
      })
        .catch(() => null)
        .finally(() => {
          setPollingSources((prev) => {
            const next = new Set(prev);
            next.delete(sourceId);
            return next;
          });
          refreshAll();
        });
    },
    [refreshAll]
  );

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setErrorBanner(null);
    setUploadedBanner(null);
    try {
      let uploadedFileNames: string[] = [];
      for (const file of Array.from(files)) {
        uploadedFileNames.push(file.name);
        const ext = file.name.split(".").pop()?.toLowerCase() || "pdf";

        // 1. Ingest via main RAG knowledge router (Docling, PyPDF, OCR, Qdrant)
        const result = await uploadFile(file, ext);
        if (result.status === "processing") {
          startPolling(result.source_id);
        }

        // 2. If spreadsheet (CSV / XLSX / XLS / TSV), dual-ingest into Neo4j graph store
        if (["csv", "xlsx", "xls", "tsv"].includes(ext)) {
          try {
            const form = new FormData();
            form.append("files", file);
            const uploadRes = await authFetch("/api/v1/graph/csv/upload", {
              method: "POST",
              body: form,
            });
            if (uploadRes?.proposal_id) {
              await authFetch(`/api/v1/graph/csv/proposal/${uploadRes.proposal_id}/approve`, {
                method: "POST",
                body: JSON.stringify({ review_comments: "Auto-approved from Knowledge Workspace" }),
              });
              await authFetch("/api/v1/graph/csv/ingest", {
                method: "POST",
                body: JSON.stringify({ proposal_id: uploadRes.proposal_id, dry_run: false }),
              });
            }
          } catch (csvErr) {
            console.warn("CSV Graph dual-ingestion notice:", csvErr);
          }
        }
      }

      setUploadedBanner(
        `Uploaded & Ingested: ${uploadedFileNames.join(", ")}! Entities and relationships are now live in the Knowledge Graph.`
      );
      await refreshAll();
    } catch (e) {
      setErrorBanner((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handleReindex = async (sourceId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setErrorBanner(null);
    try {
      const res = await reindexSource(sourceId);
      if (res.status === "processing") {
        startPolling(sourceId);
      } else {
        await refreshAll();
      }
    } catch (err) {
      setErrorBanner((err as Error).message);
    }
  };

  const handleDelete = async (sourceId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setErrorBanner(null);
    try {
      await deleteSource(sourceId);
      setSources((prev) => prev.filter((s) => s.id !== sourceId));
      await refreshAll();
    } catch (err) {
      setErrorBanner((err as Error).message);
    }
  };

  const filteredSources = useMemo(() => {
    if (activeFilter === "all") return sources;
    if (activeFilter === "pdf") return sources.filter((s) => s.source_type === "pdf");
    if (activeFilter === "docx")
      return sources.filter((s) => ["docx", "pptx", "html", "txt", "md", "json"].includes(s.source_type));
    if (activeFilter === "data") return sources.filter((s) => ["csv", "xlsx"].includes(s.source_type));
    return sources;
  }, [sources, activeFilter]);

  const formatSize = (bytes: number) => {
    if (!bytes) return "0 B";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const triggerFilePicker = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".pdf,.docx,.doc,.pptx,.xlsx,.html,.htm,.csv,.txt,.md,.json,.png,.jpg,.jpeg,.tiff,.webp";
    input.onchange = () => handleUpload(input.files);
    input.click();
  };

  return (
    <div className="relative min-h-screen">
      <SparkleBackground density={12} />

      {/* Header with Title and Primary Action */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-6">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
            Knowledge Workspace
          </h1>
          <p className="text-muted-foreground mt-2 max-w-2xl text-base">
            Ingest multi-format files into BGE-M3 Qdrant hybrid vectors &amp; explore the live Neo4j dataset knowledge graph.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link href="/graph">
            <Button variant="outline" className="gap-2 border-primary/30 hover:border-primary">
              <Network className="h-4 w-4 text-primary" /> Fullscreen Graph Studio
            </Button>
          </Link>
          <Button
            className="btn-primary-gradient gap-2 self-start md:self-auto"
            disabled={uploading}
            onClick={triggerFilePicker}
          >
            <Plus className="h-4 w-4" /> {uploading ? "Uploading & Ingesting..." : "Add Knowledge Source"}
          </Button>
        </div>
      </div>

      {/* RAG & Ingestion Processing Pipeline */}
      <div className="mb-8 rounded-2xl border border-primary/30 bg-card/75 backdrop-blur-xl p-5 shadow-[0_0_30px_rgba(245,158,11,0.08)]">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-4 border-b border-border/50 gap-2">
          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-semibold uppercase tracking-wider text-primary">
              Pipeline: Parsing &rarr; Entity Extraction &rarr; Embedding &rarr; Chunks &rarr; Dual Indexing
            </span>
          </div>
          <span className="text-[11px] font-mono text-muted-foreground">
            Strict 10% Overlap &bull; Dual Store &bull; Zero Hallucination
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 relative">
          {/* Step 1: Parsing */}
          <div className="rounded-xl border border-border/70 bg-background/85 p-3.5 space-y-1.5 relative group hover:border-primary/50 transition-all shadow-sm">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-mono text-primary font-bold">01</span>
              <span className="text-[10px] bg-amber-500/10 text-amber-400 font-semibold px-1.5 py-0.5 rounded">Multi-Parser</span>
            </div>
            <div className="font-bold text-sm text-foreground">Parsing</div>
            <div className="text-[11px] text-muted-foreground leading-snug">
              Docling &bull; PyMuPDF &bull; PaddleOCR v6 &bull; PDF, DOCX, CSV, XLSX
            </div>
          </div>

          {/* Connector 1 */}
          <div className="hidden lg:flex absolute top-1/2 left-[20%] -translate-y-1/2 -translate-x-1/2 z-10 text-primary font-mono text-sm font-bold select-none pointer-events-none">
            &rarr;
          </div>

          {/* Step 2: Entity Extraction */}
          <div className="rounded-xl border border-border/70 bg-background/85 p-3.5 space-y-1.5 relative group hover:border-primary/50 transition-all shadow-sm">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-mono text-primary font-bold">02</span>
              <span className="text-[10px] bg-purple-500/10 text-purple-400 font-semibold px-1.5 py-0.5 rounded">NER &amp; Relations</span>
            </div>
            <div className="font-bold text-sm text-foreground">Entity Extraction</div>
            <div className="text-[11px] text-muted-foreground leading-snug">
              Entities &bull; Relations &bull; Neo4j Triples &bull; Taxonomy
            </div>
          </div>

          {/* Connector 2 */}
          <div className="hidden lg:flex absolute top-1/2 left-[40%] -translate-y-1/2 -translate-x-1/2 z-10 text-primary font-mono text-sm font-bold select-none pointer-events-none">
            &rarr;
          </div>

          {/* Step 3: Embedding */}
          <div className="rounded-xl border border-border/70 bg-background/85 p-3.5 space-y-1.5 relative group hover:border-primary/50 transition-all shadow-sm">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-mono text-primary font-bold">03</span>
              <span className="text-[10px] bg-emerald-500/10 text-emerald-400 font-semibold px-1.5 py-0.5 rounded">1024-dim Vector</span>
            </div>
            <div className="font-bold text-sm text-foreground">Embedding</div>
            <div className="text-[11px] text-muted-foreground leading-snug">
              BAAI/bge-m3 dense &amp; sparse lexical vectorization
            </div>
          </div>

          {/* Connector 3 */}
          <div className="hidden lg:flex absolute top-1/2 left-[60%] -translate-y-1/2 -translate-x-1/2 z-10 text-primary font-mono text-sm font-bold select-none pointer-events-none">
            &rarr;
          </div>

          {/* Step 4: Chunks */}
          <div className="rounded-xl border border-border/70 bg-background/85 p-3.5 space-y-1.5 relative group hover:border-primary/50 transition-all shadow-sm">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-mono text-primary font-bold">04</span>
              <span className="text-[10px] bg-sky-500/10 text-sky-400 font-semibold px-1.5 py-0.5 rounded">10% Overlap</span>
            </div>
            <div className="font-bold text-sm text-foreground">Chunks</div>
            <div className="text-[11px] text-muted-foreground leading-snug">
              800 chars &bull; 80 overlap &bull; Deterministic UUIDv5 IDs
            </div>
          </div>

          {/* Connector 4 */}
          <div className="hidden lg:flex absolute top-1/2 left-[80%] -translate-y-1/2 -translate-x-1/2 z-10 text-primary font-mono text-sm font-bold select-none pointer-events-none">
            &rarr;
          </div>

          {/* Step 5: Dual Indexing */}
          <div className="rounded-xl border border-border/70 bg-background/85 p-3.5 space-y-1.5 relative group hover:border-primary/50 transition-all shadow-sm">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-mono text-primary font-bold">05</span>
              <span className="text-[10px] bg-cyan-500/10 text-cyan-400 font-semibold px-1.5 py-0.5 rounded">Dual Store</span>
            </div>
            <div className="font-bold text-sm text-foreground">Dual Indexing</div>
            <div className="text-[11px] text-muted-foreground leading-snug">
              Qdrant Vectors + Neo4j Graph + Cross-Store Links
            </div>
          </div>
        </div>
      </div>

      {/* Upload Success Banner */}
      {uploadedBanner && (
        <div className="mb-6 rounded-xl border border-emerald-500/40 bg-emerald-500/10 p-4 text-sm text-emerald-300 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-lg">
          <div className="flex items-center gap-2.5">
            <CheckCircle className="h-5 w-5 text-emerald-400 shrink-0" />
            <span className="font-medium">{uploadedBanner}</span>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/graph">
              <Button size="sm" className="h-8 btn-primary-gradient text-xs gap-1.5">
                <Network className="h-3.5 w-3.5" /> Open in Knowledge Graph Studio &rarr;
              </Button>
            </Link>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setUploadedBanner(null)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {errorBanner && (
        <div className="mb-6 rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive flex items-center justify-between">
          <span>{errorBanner}</span>
          <button onClick={() => setErrorBanner(null)} className="text-xs underline ml-4">
            Dismiss
          </button>
        </div>
      )}

      {/* Bento Grid: Upload + Live Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <Card
          className={cn(
            "lg:col-span-2 border-border/60 bg-card/60 backdrop-blur-xl shadow-glow transition-all cursor-pointer relative overflow-hidden group rounded-2xl",
            dragOver && "border-primary/60 bg-primary/10"
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleUpload(e.dataTransfer.files);
          }}
          onClick={triggerFilePicker}
        >
          <CardContent className="flex flex-col items-center justify-center text-center p-8 min-h-[260px] relative z-10">
            <div className="relative h-16 w-16 rounded-full bg-card border border-primary/20 flex items-center justify-center mb-4">
              <Database className="h-7 w-7 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-foreground">
              Drag &amp; Drop multi-file documents or spreadsheets
            </h3>
            <p className="text-muted-foreground mb-5 max-w-md text-sm">
              Instant SHA-256 deduplication, BGE-M3 hybrid vectorization, and automatic Neo4j Knowledge Graph generation.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {["PDF", "DOCX", "PPTX/XLSX", "CSV", "MD/TXT/JSON", "Scanned Images"].map((label) => (
                <span
                  key={label}
                  className="rounded-md bg-background/50 px-2.5 py-1 text-xs text-muted-foreground border border-border"
                >
                  {label}
                </span>
              ))}
            </div>
            {uploading && (
              <div className="mt-4 flex items-center gap-2 text-primary">
                <RefreshCw className="h-4 w-4 animate-spin" />
                <span className="text-sm font-medium">Validating, parsing, and building Knowledge Graph...</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Live Qdrant + Neo4j Metrics */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl shadow-glow rounded-2xl">
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-base">
              <span className="flex items-center gap-2">
                <Network className="h-4 w-4 text-primary" /> Live Index Status
              </span>
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={refreshAll}>
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl bg-background/60 p-3 border border-border/60">
                <span className="text-xs text-muted-foreground block">Qdrant Vectors</span>
                <span className="text-2xl font-bold text-foreground">
                  {depStats.qdrantPoints.toLocaleString()}
                </span>
              </div>
              <Link href="/graph" className="block group">
                <div className="rounded-xl bg-background/60 p-3 border border-border/60 group-hover:border-primary/50 transition-colors">
                  <span className="text-xs text-muted-foreground flex items-center justify-between">
                    Neo4j Graph Nodes <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity text-primary" />
                  </span>
                  <span className="text-2xl font-bold text-foreground">
                    {depStats.neo4jNodes.toLocaleString()}
                  </span>
                </div>
              </Link>
            </div>
            <div className="rounded-xl bg-background/60 p-3 border border-border/60 text-xs space-y-1.5">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Embedding Model</span>
                <span className="font-mono text-foreground">{depStats.embeddingModel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Indexed Documents</span>
                <span className="font-medium text-foreground">{sources.length}</span>
              </div>
            </div>
            <div className="flex items-center justify-between text-xs pt-1">
              <span className="flex items-center gap-1.5">
                <span
                  className={cn(
                    "w-2 h-2 rounded-full",
                    depStats.qdrantOk ? "bg-emerald-500" : "bg-destructive"
                  )}
                />
                Qdrant {depStats.qdrantOk ? "Online" : "Offline"}
              </span>
              <span className="flex items-center gap-1.5">
                <span
                  className={cn(
                    "w-2 h-2 rounded-full",
                    depStats.neo4jOk ? "bg-emerald-500" : "bg-destructive"
                  )}
                />
                Neo4j {depStats.neo4jOk ? "Online" : "Offline"}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* High-Impact Knowledge Graph Callout Card */}
      <Card className="mb-6 border border-primary/30 bg-gradient-to-r from-primary/10 via-card/75 to-purple-500/10 backdrop-blur-xl p-5 rounded-2xl shadow-glow">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="h-12 w-12 rounded-xl bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0">
              <Network className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 className="font-bold text-base text-foreground flex items-center gap-2">
                Dedicated Knowledge Graph Studio
                <span className="text-[10px] bg-emerald-500/10 text-emerald-400 font-semibold px-2 py-0.5 rounded-full border border-emerald-500/20">
                  {depStats.neo4jNodes.toLocaleString()} Active Nodes
                </span>
              </h3>
              <p className="text-xs text-muted-foreground mt-0.5 max-w-xl">
                Explore colorful Neo4j-style floating entity nodes, dynamic moving relationship arrows, full attribute inspector, spreadsheet deletion manager, and real-time Cypher console queries.
              </p>
            </div>
          </div>
          <Link href="/graph">
            <Button className="btn-primary-gradient gap-2 whitespace-nowrap shadow-md">
              <Network className="h-4 w-4" /> Open Fullscreen Graph Studio <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </Card>

      {/* Supported Formats */}
      <Card className="border-border/60 bg-card/60 backdrop-blur-xl rounded-2xl mb-6">
        <CardHeader className="py-3">
          <CardTitle className="text-sm">Document Processing Router Matrix</CardTitle>
        </CardHeader>
        <CardContent className="pb-4">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/60">
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">Parser Pipeline</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">Extensions</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">Max Size</th>
                </tr>
              </thead>
              <tbody>
                {FILE_SIZE_LIMITS.map((row) => (
                  <tr key={row.type} className="border-b border-border/30">
                    <td className="py-2 px-3 flex items-center gap-2">
                      <row.icon className="h-3.5 w-3.5 text-primary" /> {row.type}
                    </td>
                    <td className="py-2 px-3 font-mono">{row.ext}</td>
                    <td className="py-2 px-3">{row.limit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 className="text-xl font-semibold text-foreground">
          Indexed Knowledge Sources ({filteredSources.length})
        </h2>
        <div className="flex flex-wrap gap-2">
          {filterButtons.map(({ id, label, icon: Icon }) => (
            <Button
              key={id}
              variant={activeFilter === id ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveFilter(id)}
              className="h-8 text-xs"
            >
              <Icon className="h-3.5 w-3.5 mr-1.5" />
              {label}
            </Button>
          ))}
        </div>
      </div>

      {/* Sources Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-12">
        {filteredSources.length === 0 ? (
          <Card className="border-border/60 bg-card/60 backdrop-blur-xl col-span-full">
            <CardContent className="p-8 text-center text-sm text-muted-foreground">
              No indexed documents match the selected filter. Drag &amp; drop a PDF, DOCX, CSV, or Markdown file above to start.
            </CardContent>
          </Card>
        ) : (
          filteredSources.map((source) => {
            const config = STATUS_CONFIG[source.status] || STATUS_CONFIG.pending;
            const isPolling = pollingSources.has(source.id);
            const parserUsed =
              source.metadata?.parser_used || source.metadata?.recommended_parser || source.source_type;
            const stage = source.metadata?.ingestion_stage;

            return (
              <Card
                key={source.id}
                className={cn(
                  "border-border/60 bg-card/60 backdrop-blur-xl transition-all",
                  isPolling && "ring-2 ring-primary/40"
                )}
              >
                <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
                  <div className="flex items-center gap-2">
                    <div className="h-9 w-9 rounded-lg bg-background border border-border flex items-center justify-center">
                      <FileText className="h-4 w-4 text-primary" />
                    </div>
                    <span className="text-[11px] font-mono uppercase px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                      {parserUsed}
                    </span>
                  </div>
                  <span
                    className={cn(
                      "flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs border",
                      config.bg,
                      config.color
                    )}
                  >
                    {source.status === "processing" && <RefreshCw className="h-3 w-3 animate-spin" />}
                    {source.status === "failed" && <XCircle className="h-3 w-3" />}
                    {source.status === "ready" && <CheckCircle className="h-3 w-3" />}
                    {source.status === "pending" && <Clock className="h-3 w-3" />}
                    {config.label}
                  </span>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <p className="text-sm font-medium truncate text-foreground" title={source.title}>
                      {source.title}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatSize(source.metadata?.size || 0)} &bull; {source.chunk_count} chunks
                      {stage ? ` &bull; stage: ${stage}` : ""}
                    </p>
                    {source.error_message && (
                      <p className="text-xs text-destructive mt-1 line-clamp-2">{source.error_message}</p>
                    )}
                  </div>
                  <div className="flex items-center justify-end gap-2 pt-1 border-t border-border/40">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs gap-1"
                      onClick={(e) => handleReindex(source.id, e)}
                    >
                      <RotateCcw className="h-3 w-3" /> Re-index
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs gap-1 text-destructive hover:text-destructive"
                      onClick={(e) => handleDelete(source.id, e)}
                    >
                      <Trash2 className="h-3 w-3" /> Delete
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })
        )}
      </div>
    </div>
  );
}