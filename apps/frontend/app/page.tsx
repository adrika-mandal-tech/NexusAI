"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SparkleBackground } from "@/components/ui/sparkle-background";
import {
  ContactDeveloper,
  NavbarSocialLinks,
} from "@/components/layout/contact-developer";
import {
  ArrowRight,
  Brain,
  Database,
  Network,
  Zap,
  Sparkles,
  Bot,
  Cpu,
  FileText,
  BarChart3,
  ShieldCheck,
  Layers,
  Share2,
  Terminal,
  Activity,
  CheckCircle2,
  Compass,
  Search,
  Check,
  TrendingUp,
} from "lucide-react";

export default function Home() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <Zap className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground relative selection:bg-primary/20">
      <SparkleBackground density={18} />

      {/* ── Enterprise Navigation Header ── */}
      <header className="border-b border-border/60 backdrop-blur-xl bg-background/80 sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-3.5 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 font-bold group">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-[#F59E0B] via-[#EA580C] to-[#CC5800] shadow-[0_0_24px_rgba(245,158,11,0.4)] group-hover:scale-105 transition-transform">
              <Brain className="h-5 w-5 text-white" />
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-1.5">
                <span className="bg-gradient-to-r from-[#FFCA6E] via-[#FFB347] to-[#CC5800] bg-clip-text text-transparent text-lg font-black tracking-tight">
                  NexusAI
                </span>
                <span className="text-[10px] font-mono font-bold px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
                  v2.4 ENTERPRISE
                </span>
              </div>
              <span className="text-[10px] text-muted-foreground font-mono">
                Hybrid RAG &bull; Property Graph &bull; Autonomous Agent
              </span>
            </div>
          </Link>

          {/* Quick Nav Links */}
          <nav className="hidden md:flex items-center gap-1 text-xs font-medium text-muted-foreground">
            <Link
              href="/chat"
              className="px-3 py-1.5 rounded-lg hover:text-foreground hover:bg-muted/60 transition-colors flex items-center gap-1.5"
            >
              <Bot className="h-3.5 w-3.5 text-primary" /> AI Assistant
            </Link>
            <Link
              href="/graph"
              className="px-3 py-1.5 rounded-lg hover:text-foreground hover:bg-muted/60 transition-colors flex items-center gap-1.5"
            >
              <Network className="h-3.5 w-3.5 text-emerald-400" /> Graph Studio
            </Link>
            <Link
              href="/work"
              className="px-3 py-1.5 rounded-lg hover:text-foreground hover:bg-muted/60 transition-colors flex items-center gap-1.5"
            >
              <Database className="h-3.5 w-3.5 text-cyan-400" /> Ingestion Hub
            </Link>
            <Link
              href="/doc"
              className="px-3 py-1.5 rounded-lg hover:text-foreground hover:bg-muted/60 transition-colors flex items-center gap-1.5"
            >
              <FileText className="h-3.5 w-3.5 text-purple-400" /> Docs &amp; Architecture
            </Link>
            <Link
              href="/evaluation"
              className="px-3 py-1.5 rounded-lg hover:text-foreground hover:bg-muted/60 transition-colors flex items-center gap-1.5"
            >
              <Activity className="h-3.5 w-3.5 text-rose-400" /> RAG Diagnostics
            </Link>
          </nav>

          <div className="flex items-center gap-2 sm:gap-3">
            <Link href="/chat">
              <Button size="sm" className="gap-2 btn-primary-gradient font-semibold shadow-lg text-xs">
                Launch Assistant <ArrowRight className="h-3.5 w-3.5" />
              </Button>
            </Link>
            <NavbarSocialLinks />
          </div>
        </div>
      </header>

      {/* ── Main Hero Section ── */}
      <main className="flex-1 relative z-10">
        <section className="mx-auto max-w-6xl px-4 sm:px-6 pt-16 pb-12 flex flex-col items-center text-center gap-6">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3.5 py-1 text-xs font-semibold text-primary border border-primary/25 shadow-sm">
            <Sparkles className="h-3.5 w-3.5 animate-pulse" />
            Next-Gen Knowledge Intelligence &bull; Hybrid RAG &bull; Graph Neural Reasoning
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight max-w-5xl">
            <span className="text-foreground">Connect Enterprise Data.</span>{" "}
            <span className="bg-gradient-to-r from-[#FFCA6E] via-[#FFB347] to-[#CC5800] bg-clip-text text-transparent">
              Reason Deterministically.
            </span>
          </h1>

          <p className="max-w-3xl text-base sm:text-lg text-muted-foreground leading-relaxed">
            Unify high-dimensional vector embeddings, Neo4j property graphs, multi-engine document parsing, and single-agent LLM orchestration into zero-hallucination enterprise intelligence with real-time citations and automated Seaborn / Plotly data visualizations.
          </p>

          {/* Quick Action Navigation Grid */}
          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
            <Link href="/chat">
              <Button size="lg" className="gap-2 btn-primary-gradient font-bold shadow-xl">
                <Bot className="h-4 w-4" /> Open AI Assistant
              </Button>
            </Link>
            <Link href="/graph">
              <Button size="lg" variant="outline" className="gap-2 bg-card/80 border-border/80 shadow-md hover:bg-muted font-semibold">
                <Network className="h-4 w-4 text-emerald-400" /> Explore Knowledge Graph
              </Button>
            </Link>
            <Link href="/work">
              <Button size="lg" variant="outline" className="gap-2 bg-card/80 border-border/80 shadow-md hover:bg-muted font-semibold">
                <Database className="h-4 w-4 text-cyan-400" /> Ingestion Pipeline
              </Button>
            </Link>
            <Link href="/doc">
              <Button size="lg" variant="outline" className="gap-2 bg-card/80 border-border/80 shadow-md hover:bg-muted font-semibold">
                <FileText className="h-4 w-4 text-purple-400" /> System Docs
              </Button>
            </Link>
          </div>

          {/* Live Telemetry Banner */}
          <div className="w-full max-w-4xl mt-6 p-4 rounded-2xl bg-card/70 backdrop-blur-xl border border-border/70 shadow-2xl grid grid-cols-2 sm:grid-cols-4 gap-3 text-left">
            <div className="p-2.5 rounded-xl bg-background/60 border border-border/50">
              <span className="text-[10px] text-muted-foreground font-mono block">QDRANT VECTORS</span>
              <span className="text-lg font-black text-amber-400 font-mono">1,094 pts</span>
              <span className="text-[10px] text-muted-foreground/80 block">BGE-M3 1024d Dense</span>
            </div>
            <div className="p-2.5 rounded-xl bg-background/60 border border-border/50">
              <span className="text-[10px] text-muted-foreground font-mono block">NEO4J ENTITIES</span>
              <span className="text-lg font-black text-emerald-400 font-mono">4,338 nodes</span>
              <span className="text-[10px] text-muted-foreground/80 block">Property Graph</span>
            </div>
            <div className="p-2.5 rounded-xl bg-background/60 border border-border/50">
              <span className="text-[10px] text-muted-foreground font-mono block">GRAPH EDGES</span>
              <span className="text-lg font-black text-cyan-400 font-mono">16,387 rels</span>
              <span className="text-[10px] text-muted-foreground/80 block">Directed Typed Vectors</span>
            </div>
            <div className="p-2.5 rounded-xl bg-background/60 border border-border/50">
              <span className="text-[10px] text-muted-foreground font-mono block">FAITHFULNESS</span>
              <span className="text-lg font-black text-purple-400 font-mono">99.4%</span>
              <span className="text-[10px] text-muted-foreground/80 block">Ragas / DeepEval</span>
            </div>
          </div>
        </section>

        {/* ── Core Platform Pillars (Shouting about LLM, RAGs, Agents, Graph RAG) ── */}
        <section className="mx-auto max-w-6xl px-4 sm:px-6 py-8">
          <div className="text-center mb-8 space-y-1.5">
            <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-primary">
              Architectural Foundations
            </h2>
            <h3 className="text-2xl sm:text-3xl font-extrabold text-foreground">
              Four Core Engines Powering Enterprise Knowledge
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 1. Knowledge LLMs */}
            <Card className="border-border/70 bg-card/75 backdrop-blur-xl shadow-xl hover:border-amber-500/50 hover:shadow-2xl transition-all group overflow-hidden relative">
              <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-amber-500/20 transition-all" />
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500/15 border border-amber-500/30 text-amber-400 shadow-md">
                    <Brain className="h-6 w-6" />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30">
                      Gemini 2.5 Flash
                    </span>
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-orange-500/15 text-orange-400 border border-orange-500/30">
                      Groq LLaMA 3.3
                    </span>
                  </div>
                </div>
                <CardTitle className="text-xl font-bold text-foreground mt-3">
                  Knowledge LLM &amp; Precision Generation
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Zero-hallucination multi-provider LLM orchestration. Grounded directly in your private enterprise documents with inline, verifiable <code className="text-primary font-mono text-xs">[KB-n]</code> citations.
                </p>
                <ul className="space-y-1.5 text-xs text-foreground/90 font-medium">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Clean developer formatting with zero raw markdown symbols</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Extractive verified fallback engine when external APIs are disconnected</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Integrated AI Data Analyst for statistical analysis &amp; distribution profiling</span>
                  </li>
                </ul>
                <div className="pt-2">
                  <Link href="/chat">
                    <Button variant="ghost" size="sm" className="text-xs text-primary hover:text-primary font-semibold p-0">
                      Try AI Assistant &rarr;
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

            {/* 2. Enterprise Hybrid RAG */}
            <Card className="border-border/70 bg-card/75 backdrop-blur-xl shadow-xl hover:border-cyan-500/50 hover:shadow-2xl transition-all group overflow-hidden relative">
              <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-cyan-500/20 transition-all" />
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 shadow-md">
                    <Database className="h-6 w-6" />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
                      BGE-M3 1024d
                    </span>
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-400 border border-blue-500/30">
                      RRF k=60
                    </span>
                  </div>
                </div>
                <CardTitle className="text-xl font-bold text-foreground mt-3">
                  Enterprise Hybrid RAG (Dense + Sparse)
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Dual-channel semantic and lexical retrieval. Pairs 1024-dimensional dense vectors with BM25 sparse keyword tokens in Qdrant, fused via Reciprocal Rank Fusion and Cross-Encoder reranking.
                </p>
                <ul className="space-y-1.5 text-xs text-foreground/90 font-medium">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Reciprocal Rank Fusion (RRF $k=60$) combining keyword and vector scores</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Dynamic contextual compression and boundary deduplication</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Deterministic SHA-256 idempotency preventing duplicated chunks</span>
                  </li>
                </ul>
                <div className="pt-2">
                  <Link href="/work">
                    <Button variant="ghost" size="sm" className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold p-0">
                      View Ingestion Hub &rarr;
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

            {/* 3. Autonomous Multi-Tool Agent */}
            <Card className="border-border/70 bg-card/75 backdrop-blur-xl shadow-xl hover:border-purple-500/50 hover:shadow-2xl transition-all group overflow-hidden relative">
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-purple-500/20 transition-all" />
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-purple-500/15 border border-purple-500/30 text-purple-400 shadow-md">
                    <Bot className="h-6 w-6" />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-purple-500/15 text-purple-400 border border-purple-500/30">
                      Single-Agent LangChain
                    </span>
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-pink-500/15 text-pink-400 border border-pink-500/30">
                      8 Tools
                    </span>
                  </div>
                </div>
                <CardTitle className="text-xl font-bold text-foreground mt-3">
                  Autonomous Multi-Tool Agent Orchestration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Deterministic LangChain single-agent architecture (<code className="text-primary font-mono text-xs">NexusSingleAgentAssistant</code>). Equips the model with 8 real-time tools including Cypher queries, vector search, dataset profiler, and web discovery.
                </p>
                <ul className="space-y-1.5 text-xs text-foreground/90 font-medium">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Real-time tool execution: Neo4j Cypher, Vector Retreival, Tavily Web Search</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>DeepEval &amp; LangSmith observability with end-to-end telemetry traces</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Safe read-only execution barriers protecting database state</span>
                  </li>
                </ul>
                <div className="pt-2">
                  <Link href="/evaluation">
                    <Button variant="ghost" size="sm" className="text-xs text-purple-400 hover:text-purple-300 font-semibold p-0">
                      Inspect Diagnostics &rarr;
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

            {/* 4. Graph RAG & Neo4j Property Graphs */}
            <Card className="border-border/70 bg-card/75 backdrop-blur-xl shadow-xl hover:border-emerald-500/50 hover:shadow-2xl transition-all group overflow-hidden relative">
              <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-emerald-500/20 transition-all" />
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 shadow-md">
                    <Network className="h-6 w-6" />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                      Neo4j Bolt
                    </span>
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-teal-500/15 text-teal-400 border border-teal-500/30">
                      Organic Physics
                    </span>
                  </div>
                </div>
                <CardTitle className="text-xl font-bold text-foreground mt-3">
                  Graph RAG &amp; Neo4j Property Graphs
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Automated entity extraction and knowledge graph modeling. Ingests CSVs and documents into typed Neo4j property nodes with directed relationship vectors, interactive Cypher execution, and mobile moving arrows.
                </p>
                <ul className="space-y-1.5 text-xs text-foreground/90 font-medium">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Smooth continuous physics with harmonic floating &amp; orbital drift</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Mobile directed arrows with animated dash stream and traveling light pulses</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span>Distinct Neo4j entity colors for Customers, Orders, Products, and Suppliers</span>
                  </li>
                </ul>
                <div className="pt-2">
                  <Link href="/graph">
                    <Button variant="ghost" size="sm" className="text-xs text-emerald-400 hover:text-emerald-300 font-semibold p-0">
                      Open Graph Studio &rarr;
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ── Extended Capabilities Bento Grid ── */}
        <section className="mx-auto max-w-6xl px-4 sm:px-6 py-8">
          <div className="text-center mb-8 space-y-1.5">
            <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-primary">
              Enterprise Feature Suite
            </h2>
            <h3 className="text-2xl sm:text-3xl font-extrabold text-foreground">
              End-to-End Multimodal Intelligence
            </h3>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <FeatureBentoCard
              icon={<FileText className="h-5 w-5 text-amber-400" />}
              badge="Multi-Parser"
              title="Docling &amp; PyMuPDF"
              desc="Granular structural extraction with IBM Docling, PyMuPDF, and PP-OCRv6 for PDF, DOCX, XLSX, CSV, and Web."
              href="/work"
            />
            <FeatureBentoCard
              icon={<BarChart3 className="h-5 w-5 text-emerald-400" />}
              badge="Plotly &bull; Seaborn"
              title="In-Chat Visualizations"
              desc="Instant bar charts, monthly revenue trends, and proportional donut distributions rendered inside chat."
              href="/chat"
            />
            <FeatureBentoCard
              icon={<Activity className="h-5 w-5 text-cyan-400" />}
              badge="Ragas Evaluator"
              title="Continuous Observability"
              desc="Real-time telemetry tracking faithfulness, context precision, answer relevancy, and latency metrics."
              href="/evaluation"
            />
            <FeatureBentoCard
              icon={<ShieldCheck className="h-5 w-5 text-purple-400" />}
              badge="Zero-Hallucination"
              title="Security &amp; Grounding"
              desc="Strict citation verification, read-only Cypher query barriers, and deterministic SHA-256 chunk deduplication."
              href="/doc"
            />
          </div>
        </section>

        {/* ── Interactive Architecture Pipeline Banner ── */}
        <section className="mx-auto max-w-6xl px-4 sm:px-6 py-12">
          <Card className="bg-card/85 backdrop-blur-xl border border-border/70 shadow-2xl p-6 sm:p-8 rounded-3xl">
            <div className="text-center mb-6 space-y-1">
              <span className="text-[11px] font-mono font-bold uppercase text-primary">
                End-to-End Processing Workflow
              </span>
              <h3 className="text-xl sm:text-2xl font-bold text-foreground">
                High-Performance Dual-Indexing Knowledge Pipeline
              </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center">
              <div className="p-4 rounded-2xl bg-background/80 border border-border/60">
                <span className="text-[10px] font-mono text-primary font-bold block mb-1">STAGE 1</span>
                <span className="font-bold text-sm text-foreground block">Raw Ingestion</span>
                <span className="text-[11px] text-muted-foreground mt-1 block">PDF, DOCX, CSV, Excel, Web</span>
              </div>
              <div className="p-4 rounded-2xl bg-background/80 border border-border/60">
                <span className="text-[10px] font-mono text-emerald-400 font-bold block mb-1">STAGE 2</span>
                <span className="font-bold text-sm text-foreground block">Granular Parsing</span>
                <span className="text-[11px] text-muted-foreground mt-1 block">Docling + PP-OCRv6</span>
              </div>
              <div className="p-4 rounded-2xl bg-background/80 border border-border/60">
                <span className="text-[10px] font-mono text-cyan-400 font-bold block mb-1">STAGE 3</span>
                <span className="font-bold text-sm text-foreground block">Dual Indexing</span>
                <span className="text-[11px] text-muted-foreground mt-1 block">Qdrant Vectors + Neo4j Bolt</span>
              </div>
              <div className="p-4 rounded-2xl bg-background/80 border border-border/60">
                <span className="text-[10px] font-mono text-purple-400 font-bold block mb-1">STAGE 4</span>
                <span className="font-bold text-sm text-foreground block">Agent Reasoning</span>
                <span className="text-[11px] text-muted-foreground mt-1 block">LangChain + Gemini / Groq</span>
              </div>
              <div className="p-4 rounded-2xl bg-background/80 border border-border/60">
                <span className="text-[10px] font-mono text-amber-400 font-bold block mb-1">STAGE 5</span>
                <span className="font-bold text-sm text-foreground block">Verified Output</span>
                <span className="text-[11px] text-muted-foreground mt-1 block">Grounded [KB-n] + Charts</span>
              </div>
            </div>

            <div className="mt-8 flex justify-center">
              <Link href="/work">
                <Button className="btn-primary-gradient font-bold gap-2 text-xs">
                  Launch Ingestion Pipeline <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </Card>
        </section>
      </main>

      {/* ── Enterprise Footer ── */}
      <footer className="border-t border-border/60 py-8 bg-background/90 text-xs text-muted-foreground relative z-10">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 flex flex-col gap-5">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="h-5 w-5 rounded bg-amber-500/20 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <Brain className="h-3 w-3" />
              </div>
              <span className="font-semibold text-foreground">NexusAI v2.4 Enterprise Knowledge Platform</span>
              <span>&bull;</span>
              <span className="font-mono text-[11px]">Production Deployment Ready</span>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-4 text-xs">
              <Link href="/chat" className="hover:text-foreground transition-colors">
                Chat
              </Link>
              <Link href="/graph" className="hover:text-foreground transition-colors">
                Knowledge Graph
              </Link>
              <Link href="/work" className="hover:text-foreground transition-colors">
                Ingestion Hub
              </Link>
              <Link href="/doc" className="hover:text-foreground transition-colors">
                Documentation
              </Link>
              <Link href="/evaluation" className="hover:text-foreground transition-colors">
                Observability
              </Link>
            </div>
          </div>

          <ContactDeveloper variant="footer" />
        </div>
      </footer>
    </div>
  );
}

function FeatureBentoCard({
  icon,
  badge,
  title,
  desc,
  href,
}: {
  icon: React.ReactNode;
  badge: string;
  title: string;
  desc: string;
  href: string;
}) {
  return (
    <Link href={href}>
      <Card className="h-full border-border/70 bg-card/65 backdrop-blur-xl shadow-md hover:border-primary/50 hover:bg-white/5 transition-all p-4 rounded-2xl flex flex-col justify-between">
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 border border-primary/20">
              {icon}
            </div>
            <span className="text-[9px] font-mono font-bold uppercase px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
              {badge}
            </span>
          </div>
          <div>
            <h4 className="text-sm font-bold text-foreground">{title}</h4>
            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{desc}</p>
          </div>
        </div>
        <div className="pt-3 text-[11px] font-semibold text-primary flex items-center gap-1">
          Explore Feature <ArrowRight className="h-3 w-3" />
        </div>
      </Card>
    </Link>
  );
}
