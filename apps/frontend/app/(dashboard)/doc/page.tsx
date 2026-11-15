"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Database,
  Network,
  Bot,
  FileText,
  Activity,
  Code,
  Terminal,
  Layers,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  Copy,
  ChevronRight,
  ExternalLink,
  Cpu,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SparkleBackground } from "@/components/ui/sparkle-background";
import { cn } from "@/lib/utils";

const SECTIONS = [
  { id: "overview", title: "1. Architecture & Core Capabilities" },
  { id: "ingestion", title: "2. Ingestion & Multi-Parser Engine" },
  { id: "retrieval", title: "3. Hybrid RRF Retrieval & Reranking" },
  { id: "chat", title: "4. Zero-Hallucination AI Chat" },
  { id: "graph", title: "5. Neo4j Knowledge Graph & Cypher" },
  { id: "evals", title: "6. Real % Evaluation Metrics" },
  { id: "api", title: "7. Developer API Reference & Code" },
];

export default function DocPage() {
  const [activeSection, setActiveSection] = useState("overview");
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="relative min-h-screen">
      <SparkleBackground density={10} />
      <div className="flex flex-1 flex-col overflow-hidden relative z-10">
        <main className="flex-1 overflow-y-auto bg-background/50">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 md:py-12 flex flex-col lg:flex-row gap-8">
            {/* Table of Contents Sticky Sidebar */}
            <aside className="w-full lg:w-72 flex-shrink-0">
              <div className="sticky top-24 rounded-2xl border border-border/70 bg-card/75 backdrop-blur-xl p-5 shadow-lg space-y-4">
                <div className="flex items-center gap-2 pb-3 border-b border-border/60">
                  <FileText className="h-4 w-4 text-primary" />
                  <span className="text-xs font-semibold uppercase tracking-wider text-foreground">
                    Developer Guide
                  </span>
                </div>
                <nav className="space-y-1">
                  {SECTIONS.map((sec) => (
                    <a
                      key={sec.id}
                      href={`#${sec.id}`}
                      onClick={() => setActiveSection(sec.id)}
                      className={cn(
                        "flex items-center justify-between text-xs rounded-lg px-3 py-2 transition-all",
                        activeSection === sec.id
                          ? "bg-primary/15 text-primary font-semibold border-l-2 border-primary"
                          : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                      )}
                    >
                      <span>{sec.title}</span>
                      <ChevronRight className="h-3 w-3 opacity-60" />
                    </a>
                  ))}
                </nav>

                <div className="pt-3 border-t border-border/60 space-y-2">
                  <div className="text-[11px] text-muted-foreground font-mono">
                    NexusAI v2.0 &bull; Production Ready
                  </div>
                  <div className="flex gap-2">
                    <Link href="/chat" className="flex-1">
                      <Button variant="outline" size="sm" className="w-full text-xs h-7">
                        Open Chat
                      </Button>
                    </Link>
                    <Link href="/graph" className="flex-1">
                      <Button variant="outline" size="sm" className="w-full text-xs h-7">
                        Graph View
                      </Button>
                    </Link>
                  </div>
                </div>
              </div>
            </aside>

            {/* Main Documentation Body */}
            <div className="flex-1 max-w-4xl space-y-12">
              {/* Header Hero */}
              <div className="space-y-4">
                <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 border border-primary/25 px-3 py-1 text-xs text-primary font-mono">
                  <Sparkles className="h-3.5 w-3.5" /> NexusAI Enterprise Hybrid RAG Platform
                </div>
                <h1 className="text-3xl md:text-5xl font-bold tracking-tight text-foreground">
                  Developer Documentation &amp; Architecture
                </h1>
                <p className="text-muted-foreground text-sm md:text-base leading-relaxed">
                  Comprehensive developer manual for building, querying, and deploying production knowledge
                  pipelines using Qdrant vector similarity, Neo4j knowledge graphs, multi-engine document parsing,
                  and live LangSmith / Ragas / DeepEval evaluation metrics.
                </p>
              </div>

              {/* Section 1: Overview */}
              <section id="overview" className="scroll-mt-24 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
                    01
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-foreground">
                    Architecture &amp; Core Capabilities
                  </h2>
                </div>
                <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                  <CardContent className="p-6 space-y-4 text-sm leading-relaxed text-foreground/90">
                    <p>
                      NexusAI is an enterprise-grade <strong>Hybrid Retrieval-Augmented Generation (RAG)</strong> platform
                      orchestrated by a strict <strong>Single-Agent LangChain Assistant</strong>. It eliminates multi-agent
                      hallucination and coordination overhead by providing 7 modular, deterministic tools for search,
                      graph traversal, and citation validation.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
                      <div className="p-3 rounded-xl bg-background/80 border border-border/60">
                        <span className="font-semibold text-xs text-primary block mb-1">
                          Dense Vector Search
                        </span>
                        <p className="text-xs text-muted-foreground">
                          Qdrant vector engine using BAAI/bge-m3 (1024d) and all-MiniLM-L6-v2 embeddings with resilient local fallback.
                        </p>
                      </div>
                      <div className="p-3 rounded-xl bg-background/80 border border-border/60">
                        <span className="font-semibold text-xs text-amber-400 block mb-1">
                          Sparse &amp; BM25 Lexical
                        </span>
                        <p className="text-xs text-muted-foreground">
                          Okapi BM25 scoring across domain terms, acronyms, and identifiers to prevent keyword miss in dense representations.
                        </p>
                      </div>
                      <div className="p-3 rounded-xl bg-background/80 border border-border/60">
                        <span className="font-semibold text-xs text-purple-400 block mb-1">
                          Neo4j Knowledge Graph
                        </span>
                        <p className="text-xs text-muted-foreground">
                          PascalCase entity nodes and typed semantic relationships (USES, DEPENDS_ON) with live Cypher query console.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Section 2: Ingestion & Parsers */}
              <section id="ingestion" className="scroll-mt-24 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
                    02
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-foreground">
                    Document Ingestion &amp; Multi-Parser Suite
                  </h2>
                </div>
                <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                  <CardContent className="p-6 space-y-4 text-sm leading-relaxed text-foreground/90">
                    <p>
                      Every uploaded file undergoes automated magic-byte validation, format inspection, and routing to the optimal parser:
                    </p>
                    <ul className="space-y-2 text-xs">
                      <li className="flex items-start gap-2">
                        <span className="h-2 w-2 rounded-full bg-emerald-500 mt-1 flex-shrink-0" />
                        <span><strong>Docling Parser:</strong> Primary parser for complex PDFs and Word documents (.docx), converting multi-column structures and markdown tables accurately.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="h-2 w-2 rounded-full bg-sky-500 mt-1 flex-shrink-0" />
                        <span><strong>PyMuPDF (fitz):</strong> High-speed reading-order text block extractor with font-span heading detection (#, ##) and low-density scanned page identification.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="h-2 w-2 rounded-full bg-amber-500 mt-1 flex-shrink-0" />
                        <span><strong>PaddleOCR / RapidOCR:</strong> ONNX Runtime and PyTorch OCR engine for scanned PDFs and image files, reconstructing tabular columns with bounding box coordinates.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="h-2 w-2 rounded-full bg-indigo-500 mt-1 flex-shrink-0" />
                        <span><strong>StructuredText Parser:</strong> Native handler for .csv, .xlsx, .md, .txt, and .json data with deterministic UUIDv5 chunk assignment.</span>
                      </li>
                    </ul>
                    <div className="rounded-xl bg-background/80 p-3 border border-border/60 text-xs font-mono">
                      <span className="text-primary font-semibold">Strict 10% Overlap Policy:</span> Chunk size: 800 characters, Chunk overlap: 80 characters (10% ratio) dynamically enforced by ConfigurableChunker to ensure seamless semantic continuity.
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Section 3: Retrieval & Reranking */}
              <section id="retrieval" className="scroll-mt-24 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
                    03
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-foreground">
                    Hybrid RRF Retrieval &amp; Cross-Feature Reranking
                  </h2>
                </div>
                <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                  <CardContent className="p-6 space-y-4 text-sm leading-relaxed text-foreground/90">
                    <p>
                      NexusAI combines three retrieval channels using <strong>Reciprocal Rank Fusion (RRF, k=60)</strong> followed by a calibrated cross-feature reranker:
                    </p>
                    <div className="p-4 rounded-xl bg-background/80 border border-border/60 font-mono text-xs text-foreground/90 space-y-1">
                      <div className="text-primary font-bold">Calibrated Rerank Formula:</div>
                      <div>RerankScore = 0.48 * NormalizedRRF</div>
                      <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.24 * NormalizedBM25</div>
                      <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.16 * QueryTermCoverage</div>
                      <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.12 * ExactPhraseMatch</div>
                      <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ HeadingBonus + MultiChannelConsensus</div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Document scoping guarantees that selecting a specific document completely prevents noise or entity pollution from unrelated files (e.g. tabular sales data contaminating technical code queries).
                    </p>
                  </CardContent>
                </Card>
              </section>

              {/* Section 4: AI Chat & Anti-Hallucination */}
              <section id="chat" className="scroll-mt-24 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
                    04
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-foreground">
                    Zero-Hallucination AI Chat &amp; Natural Prose
                  </h2>
                </div>
                <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                  <CardContent className="p-6 space-y-4 text-sm leading-relaxed text-foreground/90">
                    <p>
                      The AI Assistant operates under a strict source-grounding prompt powered by <strong>Google Gemini (gemini-3.5-flash)</strong> with automatic failover to <strong>Groq (openai/gpt-oss-120b)</strong>.
                    </p>
                    <ul className="space-y-2 text-xs">
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Natural Prose Generation:</strong> Content generation strips distracting markdown symbols (###, **, *, ---, bullets) so output reads in clear natural sentences while retaining bracketed citations [KB-1].</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Automatic Citation Validation:</strong> Any fabricated citation tags not present in retrieved context are stripped prior to response delivery.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Honest Abstention:</strong> Queries without supporting evidence explicitly state missing evidence rather than synthesizing ungrounded claims.</span>
                      </li>
                    </ul>
                  </CardContent>
                </Card>
              </section>

              {/* Section 5: Knowledge Graph & Cypher */}
              <section id="graph" className="scroll-mt-24 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
                    05
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-foreground">
                    Neo4j Knowledge Graph &amp; Cypher Console
                  </h2>
                </div>
                <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                  <CardContent className="p-6 space-y-4 text-sm leading-relaxed text-foreground/90">
                    <p>
                      The Knowledge Graph visualizer replicates the Neo4j Browser experience with color-coded labels, directed arrows, movable nodes, and an interactive Cypher query console:
                    </p>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                      <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-center">
                        Technology
                      </div>
                      <div className="p-2 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-400 text-center">
                        Architecture
                      </div>
                      <div className="p-2 rounded-lg bg-sky-500/10 border border-sky-500/20 text-sky-400 text-center">
                        Document / CSV
                      </div>
                      <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-center">
                        Entity
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Execute read-only queries directly inside the <strong>neo4j$</strong> console with 1-click presets:
                    </p>
                    <pre className="rounded-xl bg-background/80 p-3 text-xs font-mono text-primary border border-border/60">
                      {"MATCH (d:Document)-[r]->(e) RETURN d, r, e LIMIT 25"}
                    </pre>
                  </CardContent>
                </Card>
              </section>

              {/* Section 6: Real % Evaluation Metrics */}
              <section id="evals" className="scroll-mt-24 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
                    06
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-foreground">
                    Live % Evaluation Metrics (LangSmith, DeepEval, Ragas)
                  </h2>
                </div>
                <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                  <CardContent className="p-6 space-y-4 text-sm leading-relaxed text-foreground/90">
                    <p>
                      Every chat turn dynamically evaluates and displays 7 real-time percentage metrics without artificial saturation:
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                      <div className="p-3 rounded-lg bg-background/80 border border-border/60">
                        <span className="font-semibold text-foreground block">Correctness (%)</span>
                        <span className="text-muted-foreground">Intent satisfaction and factual grounding alignment.</span>
                      </div>
                      <div className="p-3 rounded-lg bg-background/80 border border-border/60">
                        <span className="font-semibold text-foreground block">Hallucination Rate (%)</span>
                        <span className="text-muted-foreground">Percentage of unsupported claims (lower is better).</span>
                      </div>
                      <div className="p-3 rounded-lg bg-background/80 border border-border/60">
                        <span className="font-semibold text-foreground block">Accuracy / Faithfulness (%)</span>
                        <span className="text-muted-foreground">Verification of numeric tokens, facts, and lineage.</span>
                      </div>
                      <div className="p-3 rounded-lg bg-background/80 border border-border/60">
                        <span className="font-semibold text-foreground block">Context Precision (%)</span>
                        <span className="text-muted-foreground">Average Precision over retrieved candidates with rank discounts.</span>
                      </div>
                      <div className="p-3 rounded-lg bg-background/80 border border-border/60">
                        <span className="font-semibold text-foreground block">Context Recall (%)</span>
                        <span className="text-muted-foreground">Coverage of query facets in the retrieved context chunks.</span>
                      </div>
                      <div className="p-3 rounded-lg bg-background/80 border border-border/60">
                        <span className="font-semibold text-foreground block">Tool Calling Accuracy (%)</span>
                        <span className="text-muted-foreground">Tool invocation precision, candidate selectivity, and latency adherence.</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Section 7: Developer API Reference */}
              <section id="api" className="scroll-mt-24 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
                    07
                  </div>
                  <h2 className="text-xl md:text-2xl font-bold text-foreground">
                    Developer API Reference &amp; Code Examples
                  </h2>
                </div>

                <div className="space-y-4">
                  {/* cURL Example */}
                  <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                    <CardHeader className="p-4 border-b border-border/60 flex flex-row items-center justify-between">
                      <CardTitle className="text-xs font-mono text-primary flex items-center gap-2">
                        <Terminal className="h-3.5 w-3.5" /> 1. Chat Completion Endpoint (cURL)
                      </CardTitle>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() =>
                          copyToClipboard(
                            `curl -X POST http://localhost:8000/api/v1/chat \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "message": "Summarize the CUDA warp scheduling architecture",\n    "mode": "hybrid",\n    "fusion_strategy": "rrf",\n    "enable_reranking": true\n  }'`,
                            "curl-chat"
                          )
                        }
                      >
                        <Copy className="h-3 w-3 mr-1" />
                        {copiedCode === "curl-chat" ? "Copied" : "Copy"}
                      </Button>
                    </CardHeader>
                    <CardContent className="p-4">
                      <pre className="p-3 rounded-xl bg-background/90 text-xs font-mono text-muted-foreground overflow-x-auto leading-relaxed">
{`curl -X POST http://localhost:8000/api/v1/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Summarize the CUDA warp scheduling architecture",
    "mode": "hybrid",
    "fusion_strategy": "rrf",
    "enable_reranking": true
  }'`}
                      </pre>
                    </CardContent>
                  </Card>

                  {/* Cypher Console API */}
                  <Card className="border-border/70 bg-card/65 backdrop-blur-xl">
                    <CardHeader className="p-4 border-b border-border/60 flex flex-row items-center justify-between">
                      <CardTitle className="text-xs font-mono text-purple-400 flex items-center gap-2">
                        <Terminal className="h-3.5 w-3.5" /> 2. Execute Read-Only Cypher Query (cURL)
                      </CardTitle>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() =>
                          copyToClipboard(
                            `curl -X POST http://localhost:8000/api/v1/knowledge-graph/cypher \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "cypher": "MATCH (n:Technology) RETURN n.name, n.degree LIMIT 10"\n  }'`,
                            "curl-cypher"
                          )
                        }
                      >
                        <Copy className="h-3 w-3 mr-1" />
                        {copiedCode === "curl-cypher" ? "Copied" : "Copy"}
                      </Button>
                    </CardHeader>
                    <CardContent className="p-4">
                      <pre className="p-3 rounded-xl bg-background/90 text-xs font-mono text-muted-foreground overflow-x-auto leading-relaxed">
{`curl -X POST http://localhost:8000/api/v1/knowledge-graph/cypher \\
  -H "Content-Type: application/json" \\
  -d '{
    "cypher": "MATCH (n:Technology) RETURN n.name, n.degree LIMIT 10"
  }'`}
                      </pre>
                    </CardContent>
                  </Card>
                </div>
              </section>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
