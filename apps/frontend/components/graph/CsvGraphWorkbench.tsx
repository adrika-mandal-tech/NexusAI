"use client";

/**
 * The CSV-to-knowledge-graph workbench.
 *
 * The workflow is deliberately multi-step and nothing is written to the
 * database until the schema proposal has been reviewed and approved:
 *
 *   1. Upload    — choose one or more CSV files
 *   2. Review    — inspect the inferred schema, edit it, approve it
 *   3. Ingest    — run the ingestion, with a dry-run option
 *   4. Explore   — the report, the statistics, the graph and a question box
 */

import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileText,
  Loader2,
  Upload,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { authFetch, getUser, isAuthenticated, register, signIn } from "@/lib/auth";

type Step = "upload" | "review" | "ingest" | "explore";

interface ColumnProfile {
  name: string;
  inferred_type: string;
  property_type: string | null;
  null_ratio: number;
  distinct_count: number;
  is_candidate_key: boolean;
  key_confidence: string;
  may_contain_pii: boolean;
  quality_flags: string[];
}

interface FileProfile {
  file_id: string;
  filename: string;
  total_rows: number;
  column_count: number;
  proposed_label: string;
  label_confidence: string;
  duplicate_row_count: number;
  warnings: string[];
  columns: ColumnProfile[];
}

interface NodeDefinition {
  label: string;
  source_file: string;
  identity_columns: string[];
  confidence: string;
  expected_node_count: number;
  properties: Record<string, string>;
  warnings: string[];
  unresolved_ambiguities: string[];
}

interface RelationshipDefinition {
  relationship_type: string;
  source_label: string;
  target_label: string;
  source_columns: string[];
  target_columns: string[];
  direction: string;
  match_ratio: number;
  confidence: string;
  unmatched_value_count: number;
  cardinality: string;
  warnings: string[];
}

interface Proposal {
  proposal_id: string;
  status: string;
  overall_confidence: string;
  files: FileProfile[];
  nodes: NodeDefinition[];
  relationships: RelationshipDefinition[];
  ambiguities: string[];
  warnings: string[];
}

interface ValidationCheck {
  check_id: string;
  status: string;
  expected: unknown;
  actual: unknown;
  detail: string;
}

interface IngestionReport {
  job_id: string;
  status: string;
  duration_seconds: number;
  dry_run: boolean;
  files: {
    filename: string;
    rows_read: number;
    rows_accepted: number;
    rows_rejected: number;
    nodes_created: number;
    nodes_matched: number;
    relationships_created: number;
    unmatched_relationship_rows: number;
    conversion_failures: { row: number; column: string; message: string }[];
    rejected_rows: { row: number; column: string | null; message: string }[];
  }[];
  validation: {
    checks: ValidationCheck[];
    errors: string[];
    warnings: string[];
  };
}

const CONFIDENCE_STYLES: Record<string, string> = {
  high: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  medium: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  low: "bg-rose-500/15 text-rose-400 border-rose-500/30",
};

function ConfidenceBadge({ value }: { value: string }) {
  return (
    <Badge
      variant="outline"
      className={`text-[10px] uppercase tracking-wide ${
        CONFIDENCE_STYLES[value] ?? CONFIDENCE_STYLES.low
      }`}
    >
      {value}
    </Badge>
  );
}

export default function CsvGraphWorkbench({
  onGraphChanged,
}: {
  onGraphChanged?: () => void;
}) {
  const [step, setStep] = useState<Step>("upload");
  const [authed, setAuthed] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [report, setReport] = useState<IngestionReport | null>(null);
  const [plan, setPlan] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");
  const [notice, setNotice] = useState<string>("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<any>(null);

  useEffect(() => {
    setAuthed(isAuthenticated());
  }, []);

  const run = useCallback(async (fn: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  const handleAuth = (mode: "register" | "login") =>
    run(async () => {
      const user =
        mode === "register"
          ? await register(email, password, fullName || email)
          : await signIn(email, password);
      setAuthed(true);
      setNotice(`Signed in as ${user.email}`);
    });

  const handleUpload = () =>
    run(async () => {
      if (files.length === 0) throw new Error("Choose at least one CSV file.");
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      const body = await authFetch("/api/v1/graph/csv/upload", {
        method: "POST",
        body: form,
      });
      setProposal(body);
      setStep("review");
    });

  const handleApprove = () =>
    run(async () => {
      if (!proposal) return;
      const body = await authFetch(
        `/api/v1/graph/csv/proposal/${proposal.proposal_id}/approve`,
        { method: "POST" }
      );
      setProposal(body);
      setNotice("Schema approved. You can now run the ingestion.");
      const planBody = await authFetch(
        `/api/v1/graph/csv/plan/${proposal.proposal_id}`
      );
      setPlan(planBody.cypher ?? "");
    });

  const handleReject = () =>
    run(async () => {
      if (!proposal) return;
      const body = await authFetch(
        `/api/v1/graph/csv/proposal/${proposal.proposal_id}/reject`,
        { method: "POST", body: JSON.stringify({ reason: "rejected by user" }) }
      );
      setProposal(body);
      setNotice("Schema rejected. Nothing was written to the graph.");
    });

  const handleIngest = (dryRun: boolean) =>
    run(async () => {
      if (!proposal) return;
      const body = await authFetch("/api/v1/graph/csv/ingest", {
        method: "POST",
        body: JSON.stringify({ proposal_id: proposal.proposal_id, dry_run: dryRun }),
      });
      setReport(body);
      setStep("explore");
      if (!dryRun) onGraphChanged?.();
    });

  const handleAsk = () =>
    run(async () => {
      const body = await authFetch("/api/v1/graph/csv/ask", {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      setAnswer(body);
    });

  return (
    <Card className="bg-card/95 backdrop-blur-xl border-border/60 max-h-[calc(100vh-12rem)] overflow-y-auto">
      <CardContent className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <Database className="h-4 w-4" />
            CSV → Knowledge Graph
          </h3>
          <div className="flex gap-1 text-[10px] uppercase tracking-wide">
            {(["upload", "review", "ingest", "explore"] as Step[]).map((s, i) => (
              <span
                key={s}
                className={
                  step === s
                    ? "text-primary font-semibold"
                    : "text-muted-foreground/60"
                }
              >
                {i + 1}. {s}
              </span>
            ))}
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 p-2 text-xs text-rose-300">
            <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {notice && (
          <div className="flex items-start gap-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 p-2 text-xs text-emerald-300">
            <CheckCircle2 className="h-3 w-3 mt-0.5 shrink-0" />
            <span>{notice}</span>
          </div>
        )}

        {/* ── 1. Upload ── */}
        {step === "upload" && (
          <div className="space-y-3">
            <input
              type="file"
              multiple
              accept=".csv,.tsv"
              className="block w-full text-xs text-muted-foreground file:mr-3 file:rounded-md file:border-0 file:bg-primary/10 file:px-3 file:py-1.5 file:text-xs file:text-primary"
              onChange={(e) =>
                setFiles(Array.from(e.target.files ?? []))
              }
            />
            {files.length > 0 && (
              <ul className="space-y-1 text-xs text-muted-foreground">
                {files.map((f) => (
                  <li key={f.name} className="flex items-center gap-2">
                    <FileText className="h-3 w-3" />
                    {f.name}
                    <span className="text-[10px] opacity-70">
                      {(f.size / 1024).toFixed(1)} KB
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <Button
              size="sm"
              className="w-full btn-primary gap-2"
              disabled={busy || files.length === 0}
              onClick={handleUpload}
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              Analyse and propose a schema
            </Button>
          </div>
        )}

        {/* ── 2. Review ── */}
        {step === "review" && proposal && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Overall confidence</span>
              <ConfidenceBadge value={proposal.overall_confidence} />
              <Badge variant="outline" className="text-[10px]">
                {proposal.status}
              </Badge>
            </div>

            {proposal.ambiguities.length > 0 && (
              <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-2 space-y-1">
                <p className="text-[11px] font-semibold text-amber-300">
                  Needs your attention
                </p>
                {proposal.ambiguities.map((a, i) => (
                  <p key={i} className="text-[11px] text-amber-200/90">
                    • {a}
                  </p>
                ))}
              </div>
            )}

            {proposal.files.map((file) => (
              <div key={file.file_id} className="rounded-md border border-border/60 p-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">{file.filename}</span>
                  <ConfidenceBadge value={file.label_confidence} />
                </div>
                <p className="text-[10px] text-muted-foreground mt-0.5">
                  {file.total_rows} rows · {file.column_count} columns · label{" "}
                  <span className="text-primary">{file.proposed_label}</span>
                </p>
              </div>
            ))}

            <div>
              <p className="text-[11px] font-semibold mb-1.5">Node types</p>
              <div className="space-y-2">
                {proposal.nodes.map((node) => (
                  <div
                    key={node.label}
                    className="rounded-md border border-border/60 p-2 space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-primary">
                        (:{node.label})
                      </span>
                      <ConfidenceBadge value={node.confidence} />
                    </div>
                    <p className="text-[10px] text-muted-foreground">
                      from {node.source_file} · key{" "}
                      <span className="text-foreground">
                        {node.identity_columns.join(", ")}
                      </span>{" "}
                      · ~{node.expected_node_count} nodes
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {Object.keys(node.properties)
                        .filter((p) => !p.startsWith("source_") && p !== "kind" && p !== "ingested_at")
                        .map((p) => (
                          <span
                            key={p}
                            className="rounded border border-border/60 px-1.5 py-0.5 text-[10px] text-muted-foreground"
                          >
                            {p}: {node.properties[p]}
                          </span>
                        ))}
                    </div>
                    {node.warnings.map((w, i) => (
                      <p key={i} className="text-[10px] text-amber-300/90">
                        ⚠ {w}
                      </p>
                    ))}
                  </div>
                ))}
              </div>
            </div>

            {proposal.relationships.length > 0 && (
              <div>
                <p className="text-[11px] font-semibold mb-1.5">Relationships</p>
                <div className="space-y-2">
                  {proposal.relationships.map((rel) => (
                    <div
                      key={`${rel.relationship_type}-${rel.source_label}-${rel.target_label}`}
                      className="rounded-md border border-border/60 p-2 space-y-1"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium">
                          ({rel.source_label})-[:{rel.relationship_type}]-&gt;(
                          {rel.target_label})
                        </span>
                        <ConfidenceBadge value={rel.confidence} />
                      </div>
                      <p className="text-[10px] text-muted-foreground">
                        {rel.source_columns.join(", ")} → {rel.target_columns.join(", ")}{" "}
                        · {(rel.match_ratio * 100).toFixed(0)}% of values resolve ·{" "}
                        {rel.cardinality.replace(/_/g, " ")}
                      </p>
                      {rel.unmatched_value_count > 0 && (
                        <p className="text-[10px] text-amber-300/90">
                          ⚠ {rel.unmatched_value_count} value(s) do not resolve
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {plan && (
              <details className="rounded-md border border-border/60 p-2">
                <summary className="text-[11px] font-semibold cursor-pointer">
                  Generated Cypher
                </summary>
                <pre className="mt-2 overflow-x-auto text-[10px] leading-relaxed text-muted-foreground">
                  {plan}
                </pre>
              </details>
            )}

            {proposal.status === "pending_review" ? (
              <div className="flex gap-2">
                <Button
                  size="sm"
                  className="flex-1 btn-primary"
                  disabled={busy}
                  onClick={handleApprove}
                >
                  Approve schema
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={busy}
                  onClick={handleReject}
                >
                  Reject
                </Button>
              </div>
            ) : proposal.status === "approved" ? (
              <div className="flex gap-2">
                <Button
                  size="sm"
                  className="flex-1 btn-primary"
                  disabled={busy}
                  onClick={() => handleIngest(false)}
                >
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                  Ingest into the graph
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={busy}
                  onClick={() => handleIngest(true)}
                >
                  Dry run
                </Button>
              </div>
            ) : (
              <p className="text-[11px] text-muted-foreground">
                This proposal is {proposal.status}. Upload the files again to
                start over.
              </p>
            )}
          </div>
        )}

        {/* ── 3-4. Ingest and explore ── */}
        {step === "explore" && report && (
          <div className="space-y-4">
            <div className="rounded-md border border-border/60 p-2 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium">
                  {report.dry_run ? "Dry run" : "Ingestion"} report
                </span>
                <Badge
                  variant="outline"
                  className={`text-[10px] ${
                    report.status === "completed"
                      ? "text-emerald-400"
                      : "text-amber-400"
                  }`}
                >
                  {report.status.replace(/_/g, " ")}
                </Badge>
              </div>
              <p className="text-[10px] text-muted-foreground">
                {report.duration_seconds.toFixed(2)}s
              </p>
            </div>

            <div className="space-y-1">
              {report.files.map((file) => (
                <div
                  key={file.filename}
                  className="rounded-md border border-border/60 p-2 text-[11px]"
                >
                  <p className="font-medium">{file.filename}</p>
                  <p className="text-muted-foreground mt-0.5">
                    {file.rows_read} read · {file.rows_accepted} accepted ·{" "}
                    {file.rows_rejected} rejected
                  </p>
                  <p className="text-muted-foreground">
                    {file.nodes_created} nodes created ({file.nodes_matched}{" "}
                    matched) · {file.relationships_created} relationships
                    {file.unmatched_relationship_rows > 0 &&
                      ` · ${file.unmatched_relationship_rows} unmatched`}
                  </p>
                  {file.rejected_rows.slice(0, 5).map((r, i) => (
                    <p key={i} className="text-rose-300/90 mt-0.5">
                      row {r.row}: {r.message}
                    </p>
                  ))}
                  {file.conversion_failures.slice(0, 5).map((f, i) => (
                    <p key={i} className="text-amber-300/90 mt-0.5">
                      row {f.row}, {f.column}: {f.message}
                    </p>
                  ))}
                </div>
              ))}
            </div>

            {report.validation.checks.length > 0 && (
              <details className="rounded-md border border-border/60 p-2" open>
                <summary className="text-[11px] font-semibold cursor-pointer">
                  Validation ({report.validation.checks.filter((c) => c.status === "pass").length}
                  /{report.validation.checks.length} passed)
                </summary>
                <ul className="mt-1.5 space-y-0.5">
                  {report.validation.checks.map((check) => (
                    <li key={check.check_id} className="text-[10px] flex gap-1.5">
                      <span
                        className={
                          check.status === "pass"
                            ? "text-emerald-400"
                            : check.status === "warn"
                              ? "text-amber-400"
                              : "text-rose-400"
                        }
                      >
                        {check.status === "pass" ? "✓" : check.status === "warn" ? "!" : "✕"}
                      </span>
                      <span className="text-muted-foreground">
                        {check.detail || check.check_id}
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            )}

            <div className="space-y-2">
              <p className="text-[11px] font-semibold">Ask the graph</p>
              <input
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-xs"
                placeholder="Which customers placed the most orders?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && question) handleAsk();
                }}
              />
              <Button
                size="sm"
                variant="outline"
                className="w-full"
                disabled={busy || !question}
                onClick={handleAsk}
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Run the question
              </Button>
              {answer && (
                <div className="space-y-2 rounded-md border border-border/60 p-2">
                  <p className="text-[11px] leading-relaxed whitespace-pre-wrap">
                    {answer.answer}
                  </p>
                  {answer.cypher && (
                    <details>
                      <summary className="text-[10px] cursor-pointer text-muted-foreground">
                        Executed query
                      </summary>
                      <pre className="mt-1 overflow-x-auto text-[10px] text-muted-foreground">
                        {answer.cypher}
                        {Object.keys(answer.parameters ?? {}).length > 0 &&
                          `\n\nparameters: ${JSON.stringify(answer.parameters)}`}
                      </pre>
                    </details>
                  )}
                  {answer.source_files?.length > 0 && (
                    <p className="text-[10px] text-muted-foreground">
                      Source: {answer.source_files.join(", ")}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
