import { useMemo } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, Loader2, RefreshCw } from "lucide-react";

import { ResultsPanel } from "@/components/results/ResultsPanel";
import { ActivityLog } from "@/components/workflow/ActivityLog";
import { WorkflowGraph } from "@/components/workflow/WorkflowGraph";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useWorkflowStream } from "@/hooks/useWorkflowStream";
import { deriveProgress, totalTokens } from "@/lib/workflow-graph";
import type { WorkflowStatus } from "@/types/workflow";

const STREAM_LABELS = {
  connecting: "Connecting…",
  streaming: "Live",
  finished: "Finished",
  error: "Disconnected",
} as const;

/** How a terminal status should be presented, and what it actually means. */
const STATUS_PRESENTATION: Record<
  WorkflowStatus,
  { tone: "default" | "secondary" | "destructive"; note?: string }
> = {
  pending: { tone: "secondary" },
  running: { tone: "secondary" },
  completed: { tone: "default" },
  max_revisions_reached: {
    tone: "secondary",
    note: "The revision limit was reached without editorial approval. The best available draft was returned.",
  },
  completed_unverified: {
    tone: "secondary",
    note: "The posts are publishable prose, but their claims are unverified or unsupported. Nothing crashed - review before publishing.",
  },
  completed_with_errors: {
    tone: "destructive",
    note: "An agent failed during the run. Usable output was still produced.",
  },
  failed: { tone: "destructive", note: "No usable content was produced." },
};

export default function WorkflowExecution() {
  const { runId = "" } = useParams();
  const { events, run, streamState, error, reload } = useWorkflowStream(runId);

  const progress = useMemo(() => deriveProgress(events), [events]);
  const tokens = useMemo(() => totalTokens(events), [events]);

  const result = run?.result ?? null;
  const status = result?.workflow_status;
  const presentation = status ? STATUS_PRESENTATION[status] : null;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <Link
        to="/"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-4" aria-hidden="true" />
        New workflow
      </Link>

      <div className="mt-4 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            {run?.topic || "Workflow run"}
          </h1>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{runId}</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <StreamIndicator state={streamState} />
          {status && (
            <Badge variant={presentation?.tone ?? "secondary"}>{status.replace(/_/g, " ")}</Badge>
          )}
          {tokens > 0 && (
            <span className="text-xs tabular-nums text-muted-foreground">
              {tokens.toLocaleString()} tokens
            </span>
          )}
          <Button variant="ghost" size="sm" onClick={reload}>
            <RefreshCw aria-hidden="true" />
            Reconnect
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mt-6">
          <AlertTitle>Stream interrupted</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {run?.error_message && (
        <Alert variant="destructive" className="mt-6">
          <AlertTitle>The run could not complete</AlertTitle>
          <AlertDescription>{run.error_message}</AlertDescription>
        </Alert>
      )}

      {presentation?.note && (
        <Alert className="mt-6" variant={presentation.tone === "destructive" ? "destructive" : "default"}>
          <AlertTitle>{status?.replace(/_/g, " ")}</AlertTitle>
          <AlertDescription>{presentation.note}</AlertDescription>
        </Alert>
      )}

      {result?.warnings.map((warning) => (
        <Alert key={warning} className="mt-3">
          <AlertDescription>{warning}</AlertDescription>
        </Alert>
      ))}

      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <WorkflowGraph progress={progress} />

        <Card className="overflow-hidden py-0">
          <CardHeader className="border-b py-4">
            <CardTitle className="text-sm">Activity</CardTitle>
          </CardHeader>
          <CardContent className="px-0">
            <ActivityLog events={events} />
          </CardContent>
        </Card>
      </div>

      {result && (
        <section className="mt-10">
          <h2 className="font-heading text-lg font-semibold tracking-tight">Results</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {result.final_posts.length} post(s) after {result.revision_count} revision(s), with the
            research and verification behind them.
          </p>
          <div className="mt-4">
            <ResultsPanel result={result} />
          </div>
        </section>
      )}
    </div>
  );
}

function StreamIndicator({ state }: { state: keyof typeof STREAM_LABELS }) {
  const live = state === "streaming";
  const connecting = state === "connecting";

  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
      {connecting ? (
        <Loader2 className="size-3 animate-spin" aria-hidden="true" />
      ) : (
        <span
          className={
            live
              ? "size-2 animate-pulse rounded-full bg-emerald-500"
              : state === "error"
                ? "size-2 rounded-full bg-destructive"
                : "size-2 rounded-full bg-muted-foreground/40"
          }
          aria-hidden="true"
        />
      )}
      {STREAM_LABELS[state]}
    </span>
  );
}
