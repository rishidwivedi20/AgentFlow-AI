import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, Loader2, RefreshCw } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError, getRun } from "@/services/api";
import { isRunFinished } from "@/types/run";
import type { RunStatusResponse } from "@/types/run";

/**
 * Workflow execution page.
 *
 * Chunk 2 scaffold: reads run state from the status endpoint so the routing and
 * data path are real end to end. The React Flow graph, the live event stream
 * and the results panel are added in the following chunks - this deliberately
 * shows only what it can actually observe today rather than faking a UI.
 */
export default function WorkflowExecution() {
  const { runId = "" } = useParams();
  const [run, setRun] = useState<RunStatusResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async (): Promise<void> => {
    try {
      setRun(await getRun(runId));
      setError(null);
    } catch (cause) {
      setRun(null);
      setError(
        cause instanceof ApiError ? cause : new ApiError(0, "unknown", "Could not load the run."),
      );
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void load();
  }, [load]);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Link
        to="/"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-4" aria-hidden="true" />
        New workflow
      </Link>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            {run?.topic || "Workflow run"}
          </h1>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{runId}</p>
        </div>
        <div className="flex items-center gap-3">
          {run && <Badge variant="secondary">{run.state}</Badge>}
          <Button variant="ghost" size="sm" onClick={() => void load()}>
            <RefreshCw aria-hidden="true" />
            Refresh
          </Button>
        </div>
      </div>

      {loading && (
        <p className="mt-8 flex items-center gap-2 text-muted-foreground">
          <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          Loading run…
        </p>
      )}

      {error && (
        <Alert variant="destructive" className="mt-8">
          <AlertTitle>
            {error.status === 404 ? "Run not found" : "Could not load the run"}
          </AlertTitle>
          <AlertDescription>{error.message}</AlertDescription>
        </Alert>
      )}

      {run && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Execution</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <Row label="State" value={run.state} />
            <Row label="Events recorded" value={String(run.event_count)} />
            <Row label="Started" value={new Date(run.created_at).toLocaleTimeString()} />
            {run.finished_at && (
              <Row label="Finished" value={new Date(run.finished_at).toLocaleTimeString()} />
            )}
            {run.result && <Row label="Workflow status" value={run.result.workflow_status} />}
            {run.error_message && <Row label="Error" value={run.error_message} />}

            {!isRunFinished(run.state) && (
              <p className="pt-2 text-muted-foreground">
                This run is still executing. Live progress arrives in the next chunk; refresh to
                see the current state.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b py-2 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}
