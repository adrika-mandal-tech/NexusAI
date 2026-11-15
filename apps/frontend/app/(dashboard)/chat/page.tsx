"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Database,
  Network,
  Bot,
  FileText,
  Settings,
  Paperclip,
  Mic,
  ArrowUp,
  ThumbsUp,
  ThumbsDown,
  Copy,
  Share,
  Sliders,
  MoreVertical,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SparkleBackground } from "@/components/ui/sparkle-background";
import { cn } from "@/lib/utils";
import {
  streamChatMessage,
  listConversations,
  createConversation,
  getConversation,
  deleteConversation,
  getSources,
  Citation,
  ConversationItem,
  RetrievalDiagnostics,
  LiveEvalMetrics,
  SourceStatus,
  ChartSpec,
} from "@/lib/api-client";

interface ChatMessage {
  role: "user" | "ai";
  content: string;
  streaming?: boolean;
  sources?: Citation[];
  diagnostics?: RetrievalDiagnostics;
  chart?: ChartSpec;
}

function cleanSpecialSymbols(raw: string): string {
  if (!raw) return "";
  let text = raw;
  // 1. Remove markdown headings #, ##, ###, ####, etc.
  text = text.replace(/^#{1,6}\s+/gm, "");
  // 2. Remove horizontal divider rules ---, ***, ___
  text = text.replace(/^[-\*_]{3,}\s*$/gm, "");
  // 3. Remove bold and italic asterisks/underscores
  text = text.replace(/\*\*([^*]+)\*\*/g, "$1");
  text = text.replace(/__([^_]+)__/g, "$1");
  text = text.replace(/(?<!\w)\*([^*]+)\*(?!\w)/g, "$1");
  text = text.replace(/(?<!\w)_([^_]+)_(?!\w)/g, "$1");
  // 4. Remove code fences and inline backticks
  text = text.replace(/```[a-zA-Z0-9_-]*\n([\s\S]*?)```/g, "$1");
  text = text.replace(/`([^`]+)`/g, "$1");
  // 5. Remove bullet markers (* , - , + , •) at start of lines
  text = text.replace(/^[\s]*[\*\-\+•⁃]\s+/gm, "");
  // 6. Remove blockquotes >
  text = text.replace(/^>\s+/gm, "");
  // 7. Collapse excessive blank lines
  text = text.replace(/\n{3,}/g, "\n\n").trim();
  return text;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "ai",
      content:
        "Hello! Welcome to NexusAI. Upload documents in the Workspace, then ask me anything with zero-hallucination hybrid retrieval.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceStatus[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>("all");
  const [lastEvalMetrics, setLastEvalMetrics] = useState<LiveEvalMetrics | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const loadInitialData = useCallback(async () => {
    try {
      const [convList, srcList] = await Promise.all([
        listConversations().catch(() => []),
        getSources().catch(() => []),
      ]);
      setConversations(convList);
      setSources(srcList);
    } catch {
      // offline graceful fallback
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    const userPrompt = input.trim();
    setInput("");

    const newMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: userPrompt },
      { role: "ai", content: "", streaming: true, sources: [] },
    ];
    setMessages(newMessages);
    setIsTyping(true);

    const aiIdx = newMessages.length - 1;

    try {
      let activeId = activeConvId;
      if (!activeId) {
        try {
          const conv = await createConversation(
            userPrompt.slice(0, 32) || "New Analysis"
          );
          activeId = conv.id;
          setActiveConvId(activeId);
          setConversations((prev) => [conv, ...prev.filter((c) => c.id !== conv.id)]);
        } catch {
          // continue without saving conversation id
        }
      }

      await streamChatMessage(
        userPrompt,
        (delta) => {
          setMessages((prev) => {
            const next = [...prev];
            if (next[aiIdx]) {
              next[aiIdx] = {
                ...next[aiIdx],
                content: (next[aiIdx].content || "") + delta,
              };
            }
            return next;
          });
        },
        (retrievedSources) => {
          setMessages((prev) => {
            const next = [...prev];
            if (next[aiIdx]) {
              next[aiIdx] = {
                ...next[aiIdx],
                sources: retrievedSources,
              };
            }
            return next;
          });
        },
        (doneAnswer, doneCitations, doneDiagnostics, doneChart) => {
          setIsTyping(false);
          setMessages((prev) => {
            const next = [...prev];
            if (next[aiIdx]) {
              next[aiIdx] = {
                ...next[aiIdx],
                streaming: false,
                content: doneAnswer || next[aiIdx].content,
                sources: doneCitations?.length ? doneCitations : next[aiIdx].sources,
                diagnostics: doneDiagnostics,
                chart: doneChart,
              };
            }
            return next;
          });
          if (doneDiagnostics?.eval_metrics) {
            setLastEvalMetrics(doneDiagnostics.eval_metrics);
          }
        },
        (err) => {
          setIsTyping(false);
          setMessages((prev) => {
            const next = [...prev];
            if (next[aiIdx]) {
              next[aiIdx] = {
                ...next[aiIdx],
                streaming: false,
                content: `Error: ${err.message || "Failed to reach AI service"}`,
              };
            }
            return next;
          });
        },
        {
          mode: "hybrid",
          fusion_strategy: "rrf",
          enable_reranking: true,
          include_web: false,
          provider: "gemini",
          document_ids: selectedDocId !== "all" ? [selectedDocId] : undefined,
          conversation_id: activeId || undefined,
        }
      );
    } catch (e: any) {
      setIsTyping(false);
      setMessages((prev) => {
        const next = [...prev];
        if (next[aiIdx]) {
          next[aiIdx] = {
            ...next[aiIdx],
            streaming: false,
            content: `Error: ${e.message || "Failed to stream response"}`,
          };
        }
        return next;
      });
    }
  };

  const activeCitations = messages
    .filter((m) => m.role === "ai" && m.sources && m.sources.length > 0)
    .slice(-1)[0]?.sources || [];

  return (
    <div className="relative min-h-screen">
      <SparkleBackground density={12} />
      <div className="flex h-screen overflow-hidden relative z-10">
        {/* Left Sidebar: Recent Analyses & Document Scope (Slim Compact Width) */}
        <aside className="w-48 border-r border-border/60 bg-background/70 backdrop-blur-xl hidden lg:flex flex-col shrink-0">
          <div className="p-3 border-b border-border/60 flex items-center justify-between">
            <h2 className="text-xs font-bold text-foreground uppercase tracking-wider">
              Recent Analyses
            </h2>
            <Button
              variant="ghost"
              size="sm"
              className="text-[11px] h-6 px-2 text-primary font-medium"
              onClick={() => {
                setActiveConvId(null);
                setMessages([
                  {
                    role: "ai",
                    content:
                      "Started a new session. Choose your document scope or ask across all knowledge.",
                  },
                ]);
              }}
            >
              + New
            </Button>
          </div>

          {/* Document Scope Filter */}
          <div className="p-2.5 border-b border-border/60 bg-muted/20">
            <label className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1">
              Document Scope
            </label>
            <select
              value={selectedDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
              className="w-full text-xs rounded-lg bg-card/80 border border-border/70 px-2 py-1 text-foreground focus:outline-none focus:ring-1 focus:ring-primary truncate"
            >
              <option value="all">All Documents</option>
              {sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.filename || s.title || "Document"}
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
            {conversations.length === 0 ? (
              <div className="p-3 text-center text-[11px] text-muted-foreground">
                No saved analyses yet. Ask a question to begin!
              </div>
            ) : (
              conversations.map((item) => (
                <button
                  key={item.id}
                  onClick={async () => {
                    setActiveConvId(item.id);
                    try {
                      const detail = await getConversation(item.id);
                      if (detail.messages?.length) {
                        setMessages(
                          detail.messages.map((m) => ({
                            role: m.role as "user" | "ai",
                            content: m.content,
                          }))
                        );
                      }
                    } catch {
                      // fallback
                    }
                  }}
                  className={cn(
                    "w-full text-left p-2 rounded-lg transition-colors border text-xs",
                    activeConvId === item.id
                      ? "bg-primary/10 border-primary/30 text-primary font-medium"
                      : "border-transparent hover:bg-muted text-muted-foreground hover:text-foreground"
                  )}
                >
                  <div
                    className={cn(
                      "font-medium text-xs mb-0.5 truncate",
                      activeConvId === item.id ? "text-primary" : "text-foreground"
                    )}
                  >
                    {item.title || "Untitled Analysis"}
                  </div>
                  <div className="text-[10px] text-muted-foreground flex items-center justify-between">
                    <span className="flex items-center gap-1">
                      <Database className="h-3 w-3" />
                      {item.message_count ?? 0} msgs
                    </span>
                    <span className="text-muted-foreground/60">
                      {new Date(item.updated_at).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                </button>
              ))
            )}
          </div>

          <div className="p-3 border-t border-border/60 flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              Hybrid Qdrant + Neo4j
            </span>
            <Link href="/graph" className="hover:text-primary transition-colors">
              KG View &rarr;
            </Link>
          </div>
        </aside>

        {/* Main Chat Section */}
        <section className="flex-1 flex flex-col relative">
          <header className="p-4 md:p-6 border-b border-border/60 flex items-center justify-between bg-background/80 backdrop-blur-xl">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center border border-primary/20">
                <Bot className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h1 className="font-headline-md text-headline-md font-medium text-foreground">
                  NexusAI Assistant
                </h1>
                <p className="text-xs text-primary/80 flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                  Online &bull; BGE-M3 + Qdrant &bull; LangSmith Monitored
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link href="/doc">
                <Button variant="outline" size="sm" className="h-8 text-xs">
                  <FileText className="h-3.5 w-3.5 mr-1.5" /> Upload Docs
                </Button>
              </Link>
              <Link href="/graph">
                <Button variant="outline" size="sm" className="h-8 text-xs">
                  <Network className="h-3.5 w-3.5 mr-1.5" /> Knowledge Graph
                </Button>
              </Link>
            </div>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={cn("flex gap-4", msg.role === "user" && "flex-row-reverse")}
              >
                <div
                  className={cn(
                    "h-8 w-8 rounded-full flex items-center justify-center border flex-shrink-0",
                    msg.role === "ai"
                      ? "bg-primary/20 border-primary/30"
                      : "bg-muted border-border"
                  )}
                >
                  {msg.role === "ai" ? (
                    <Bot className="h-4 w-4 text-primary" />
                  ) : (
                    <span className="text-xs font-semibold">U</span>
                  )}
                </div>

                <div
                  className={cn(
                    "max-w-[85%] rounded-2xl p-4 md:p-5 space-y-2",
                    msg.role === "user"
                      ? "bg-primary/10 border border-primary/20 rounded-tr-sm"
                      : "glass-panel rounded-tl-sm border border-border/70"
                  )}
                >
                  {msg.role === "ai" && msg.streaming && (
                    <div className="text-xs text-primary flex items-center gap-2 mb-1">
                      <span className="animate-spin">⟳</span> Synthesizing from sources with BGE-M3 &amp; Qdrant...
                    </div>
                  )}

                  <div className="text-sm leading-relaxed whitespace-pre-wrap text-foreground">
                    {msg.role === "ai" ? cleanSpecialSymbols(msg.content) : msg.content}
                  </div>

                  {msg.chart && (
                    <DataAnalyticsChartCard chart={msg.chart} />
                  )}

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="pt-2 border-t border-border/40 mt-3 flex flex-wrap gap-1.5 items-center">
                      <span className="text-[11px] font-semibold text-muted-foreground mr-1">
                        Sources:
                      </span>
                      {msg.sources.map((s, idx) => (
                        <span
                          key={idx}
                          className="inline-flex items-center gap-1 text-[11px] bg-primary/10 text-primary border border-primary/20 px-2 py-0.5 rounded-md font-mono"
                        >
                          <FileText className="h-3 w-3" />
                          {s.citation_id || `KB-${idx + 1}`}: {s.filename || "doc"}
                          {s.page_number ? ` (p.${s.page_number})` : ""}
                        </span>
                      ))}
                    </div>
                  )}

                  {msg.role === "ai" && !msg.streaming && msg.content && (
                    <div className="flex items-center gap-2 pt-2 border-t border-border/30">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-foreground"
                      >
                        <ThumbsUp className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-foreground"
                      >
                        <ThumbsDown className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-foreground"
                        onClick={() => navigator.clipboard.writeText(cleanSpecialSymbols(msg.content))}
                      >
                        <Copy className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs gap-1 ml-auto text-muted-foreground hover:text-foreground"
                      >
                        <Share className="h-3 w-3" /> Export
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground pl-12">
                <span className="animate-spin text-primary">⟳</span> NexusAI is retrieving evidence &amp; running evaluation checks...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input Box */}
          <div className="p-4 bg-background/80 backdrop-blur-xl border-t border-border/60">
            <div className="max-w-4xl mx-auto relative">
              <div className="flex items-end rounded-2xl border border-border bg-card focus-within:ring-2 focus-within:ring-primary/50 transition-all">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 rounded-none rounded-l-xl text-muted-foreground hover:text-foreground"
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask anything about your documents, knowledge graph, or data..."
                  className="flex-1 bg-transparent border-none resize-none py-3 px-2 text-sm focus:ring-0 max-h-32 min-h-[44px] text-foreground placeholder:text-muted-foreground"
                  rows={1}
                />
                <div className="flex items-center gap-1 pr-1 pb-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-foreground"
                  >
                    <Mic className="h-4 w-4" />
                  </Button>
                  <Button
                    size="icon"
                    className="h-8 w-8 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                    onClick={handleSend}
                    disabled={!input.trim() || isTyping}
                  >
                    <ArrowUp className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <p className="text-center text-[11px] text-muted-foreground/60 mt-2">
                NexusAI Hybrid RAG with Qdrant Vector Search &amp; Neo4j Knowledge Graph. Verified with LangSmith, Ragas &amp; DeepEval.
              </p>
            </div>
          </div>
        </section>

        {/* Right Sidebar: Live Evals (%) & Citations */}
        <aside className="w-80 border-l border-border/60 bg-background/60 backdrop-blur-xl hidden xl:flex flex-col">
          {/* Section 1: Live % Evaluation Metrics */}
          <div className="p-4 border-b border-border/60">
            <div className="flex items-center justify-between">
              <h2 className="font-headline-md text-sm font-semibold text-foreground flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" /> Live RAG Evals (%)
              </h2>
              <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                LangSmith &bull; Ragas
              </span>
            </div>

            {lastEvalMetrics ? (
              <div className="mt-3 space-y-2.5">
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 rounded-lg bg-card/80 border border-border/60 text-center">
                    <span className="text-[10px] text-muted-foreground block">Correctness</span>
                    <span className="text-sm font-mono font-bold text-emerald-400">
                      {lastEvalMetrics.correctness_pct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-2 rounded-lg bg-card/80 border border-border/60 text-center">
                    <span className="text-[10px] text-muted-foreground block">Hallucination</span>
                    <span className={cn(
                      "text-sm font-mono font-bold",
                      lastEvalMetrics.hallucination_rate_pct < 20 ? "text-emerald-400" : "text-amber-400"
                    )}>
                      {lastEvalMetrics.hallucination_rate_pct.toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="space-y-1.5 pt-1 text-xs">
                  <div>
                    <div className="flex justify-between text-[11px] mb-0.5">
                      <span className="text-muted-foreground">Accuracy / Faithfulness</span>
                      <span className="font-mono text-foreground font-semibold">
                        {lastEvalMetrics.accuracy_pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 rounded-full"
                        style={{ width: `${Math.min(100, lastEvalMetrics.accuracy_pct)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] mb-0.5">
                      <span className="text-muted-foreground">Context Precision</span>
                      <span className="font-mono text-foreground font-semibold">
                        {lastEvalMetrics.context_precision_pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary rounded-full"
                        style={{ width: `${Math.min(100, lastEvalMetrics.context_precision_pct)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] mb-0.5">
                      <span className="text-muted-foreground">Context Recall</span>
                      <span className="font-mono text-foreground font-semibold">
                        {lastEvalMetrics.context_recall_pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-500 rounded-full"
                        style={{ width: `${Math.min(100, lastEvalMetrics.context_recall_pct)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] mb-0.5">
                      <span className="text-muted-foreground">Tool Calling Accuracy</span>
                      <span className="font-mono text-foreground font-semibold">
                        {lastEvalMetrics.tool_calling_accuracy_pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full"
                        style={{ width: `${Math.min(100, lastEvalMetrics.tool_calling_accuracy_pct)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="mt-3 p-3 rounded-lg bg-card/50 border border-dashed border-border/80 text-center">
                <p className="text-[11px] text-muted-foreground">
                  Send a message to see live LangSmith &amp; Ragas % accuracy and hallucination metrics.
                </p>
              </div>
            )}
          </div>

          {/* Section 2: Citations Panel */}
          <div className="p-3 border-b border-border/60 bg-muted/10">
            <h2 className="font-headline-md text-xs font-semibold text-foreground flex items-center gap-2">
              <FileText className="h-3.5 w-3.5 text-primary" /> Citations &amp; Evidence
            </h2>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {activeCitations.length === 0 ? (
              <div className="p-4 text-center text-xs text-muted-foreground">
                No citations for current message yet.
              </div>
            ) : (
              activeCitations.map((c, i) => (
                <Card key={i} className="border-border/60 bg-card/60 shadow-sm">
                  <CardContent className="p-3 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="flex items-center gap-1 text-primary font-medium">
                        <span className="bg-primary/20 px-1.5 py-0.2 rounded font-mono text-[10px]">
                          [{c.citation_id || i + 1}]
                        </span>
                        {c.source_type?.toUpperCase() || "DOC"}
                      </span>
                      <span className="text-muted-foreground text-[10px]">
                        {c.page_number ? `Page ${c.page_number}` : ""}
                      </span>
                    </div>
                    <p className="text-xs font-semibold truncate text-foreground">
                      {c.filename || c.section_title || "Retrieved Chunk"}
                    </p>
                    <p className="text-[11px] text-muted-foreground line-clamp-3 leading-relaxed">
                      &quot;{(c.text_snippet || "").slice(0, 180)}&quot;
                    </p>
                    {c.dense_score != null && (
                      <div className="pt-1 flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                        <span>Dense: {Number(c.dense_score).toFixed(3)}</span>
                        {c.sparse_score != null && (
                          <span>BM25: {Number(c.sparse_score).toFixed(1)}</span>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function DataAnalyticsChartCard({ chart }: { chart: ChartSpec }) {
  const [theme, setTheme] = useState<"plotly" | "seaborn">("plotly");
  const [showTable, setShowTable] = useState(false);

  const colors =
    theme === "plotly"
      ? ["#F59E0B", "#10B981", "#3B82F6", "#8B5CF6", "#EC4899", "#06B6D4", "#F97316", "#6366F1"]
      : ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#E377C2", "#7F7F7F"];

  const series = chart.series?.[0];
  const data = series?.data || [];
  const labels = chart.labels || [];
  const maxVal = Math.max(...data, 1);

  return (
    <div className="mt-3 rounded-xl border border-border/80 bg-background/90 backdrop-blur-md p-4 space-y-3 shadow-lg max-w-full overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/60 pb-2.5">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
            <h4 className="text-xs font-bold text-foreground">{chart.title}</h4>
            <span className="text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
              {chart.type}
            </span>
          </div>
          {chart.subtitle && (
            <p className="text-[11px] text-muted-foreground mt-0.5" dangerouslySetInnerHTML={{ __html: chart.subtitle }} />
          )}
        </div>

        <div className="flex items-center gap-1.5 self-end sm:self-center">
          <button
            onClick={() => setTheme((t) => (t === "plotly" ? "seaborn" : "plotly"))}
            className="text-[10px] font-mono px-2 py-0.5 rounded border border-border hover:bg-muted text-muted-foreground transition-colors"
          >
            Palette: {theme === "plotly" ? "Plotly Vibrant" : "Seaborn Grid"}
          </button>
          <button
            onClick={() => setShowTable((v) => !v)}
            className="text-[10px] font-mono px-2 py-0.5 rounded border border-border hover:bg-muted text-muted-foreground transition-colors"
          >
            {showTable ? "View Chart" : "View Data"}
          </button>
        </div>
      </div>

      {chart.kpis && chart.kpis.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {chart.kpis.map((kpi, idx) => (
            <div key={idx} className="p-2 rounded-lg bg-card/60 border border-border/60 text-center">
              <span className="text-[9px] text-muted-foreground block truncate">{kpi.label}</span>
              <span className="text-xs font-bold text-foreground font-mono truncate">{kpi.value}</span>
            </div>
          ))}
        </div>
      )}

      {showTable ? (
        <div className="max-h-48 overflow-y-auto rounded-lg border border-border/60 text-xs font-mono">
          <table className="w-full text-left">
            <thead className="bg-muted/60 text-muted-foreground">
              <tr>
                <th className="p-1.5">{chart.x_label || "Label"}</th>
                <th className="p-1.5 text-right">{chart.y_label || "Value"}</th>
              </tr>
            </thead>
            <tbody>
              {labels.map((lbl, idx) => (
                <tr key={idx} className="border-t border-border/40 hover:bg-muted/30">
                  <td className="p-1.5 text-foreground truncate max-w-xs">{lbl}</td>
                  <td className="p-1.5 text-right text-primary font-bold">{data[idx]?.toLocaleString() ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : chart.type === "line" ? (
        <div className="h-44 w-full relative pt-2">
          <svg className="w-full h-full overflow-visible" viewBox="0 0 500 140" preserveAspectRatio="none">
            <defs>
              <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={colors[0]} stopOpacity="0.35" />
                <stop offset="100%" stopColor={colors[0]} stopOpacity="0.0" />
              </linearGradient>
            </defs>
            <line x1="0" y1="120" x2="500" y2="120" stroke="rgba(100,116,139,0.2)" strokeWidth="1" />
            <line x1="0" y1="65" x2="500" y2="65" stroke="rgba(100,116,139,0.15)" strokeDasharray="3 3" />
            <line x1="0" y1="10" x2="500" y2="10" stroke="rgba(100,116,139,0.15)" strokeDasharray="3 3" />

            {data.length > 1 && (
              <>
                <polygon
                  points={`0,120 ${data
                    .map((val, idx) => `${(idx / (data.length - 1)) * 500},${120 - (val / maxVal) * 105}`)
                    .join(" ")} 500,120`}
                  fill="url(#lineGrad)"
                />
                <polyline
                  fill="none"
                  stroke={colors[0]}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={data
                    .map((val, idx) => `${(idx / (data.length - 1)) * 500},${120 - (val / maxVal) * 105}`)
                    .join(" ")}
                />
              </>
            )}

            {data.map((val, idx) => {
              const cx = (idx / Math.max(1, data.length - 1)) * 500;
              const cy = 120 - (val / maxVal) * 105;
              return (
                <g key={idx} className="group cursor-pointer">
                  <circle cx={cx} cy={cy} r="3.5" fill={colors[0]} stroke="#FFFFFF" strokeWidth="1.5" />
                  <title>{`${labels[idx] || `Item ${idx}`}: ${val}`}</title>
                </g>
              );
            })}
          </svg>
          <div className="flex justify-between text-[9px] text-muted-foreground font-mono mt-1">
            <span>{labels[0] || ""}</span>
            {labels.length > 2 && <span>{labels[Math.floor(labels.length / 2)] || ""}</span>}
            <span>{labels[labels.length - 1] || ""}</span>
          </div>
        </div>
      ) : chart.type === "donut" ? (
        <div className="flex flex-col sm:flex-row items-center justify-around gap-4 py-2">
          <div className="relative w-36 h-36">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
              {(() => {
                const total = data.reduce((a, b) => a + b, 0) || 1;
                let accAngle = 0;
                return data.map((val, idx) => {
                  const pct = val / total;
                  const dashLen = pct * 251.2;
                  const strokeOffset = -accAngle * 251.2;
                  accAngle += pct;
                  return (
                    <circle
                      key={idx}
                      cx="50"
                      cy="50"
                      r="40"
                      fill="transparent"
                      stroke={colors[idx % colors.length]}
                      strokeWidth="16"
                      strokeDasharray={`${dashLen} ${251.2 - dashLen}`}
                      strokeDashoffset={strokeOffset}
                    >
                      <title>{`${labels[idx]}: ${val} (${(pct * 100).toFixed(1)}%)`}</title>
                    </circle>
                  );
                });
              })()}
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
              <span className="text-[10px] text-muted-foreground">Share</span>
              <span className="text-xs font-bold text-foreground font-mono">100%</span>
            </div>
          </div>
          <div className="space-y-1 text-xs">
            {labels.slice(0, 5).map((lbl, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: colors[idx % colors.length] }} />
                <span className="text-muted-foreground text-[11px] truncate max-w-[120px]">{lbl}</span>
                <span className="text-foreground font-bold font-mono text-[11px] ml-auto">
                  {data[idx]?.toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Vertical Bar Chart */
        <div className="space-y-2 pt-2">
          <div className="flex items-end gap-2 h-36 px-2 border-b border-border/50">
            {data.slice(0, 8).map((val, idx) => {
              const hPct = Math.max(8, Math.min(100, (val / maxVal) * 100));
              const barColor = colors[idx % colors.length];
              return (
                <div key={idx} className="flex-1 flex flex-col items-center justify-end h-full group relative">
                  <span className="text-[9px] font-mono font-bold text-foreground opacity-0 group-hover:opacity-100 transition-opacity mb-1">
                    {val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val}
                  </span>
                  <div
                    className="w-full rounded-t-md transition-all group-hover:brightness-110 shadow-sm"
                    style={{
                      height: `${hPct}%`,
                      backgroundColor: barColor,
                    }}
                  />
                  <span className="text-[9px] font-mono text-muted-foreground truncate w-full text-center mt-1">
                    {labels[idx] ? (labels[idx].length > 6 ? `${labels[idx].slice(0, 5)}…` : labels[idx]) : idx + 1}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
