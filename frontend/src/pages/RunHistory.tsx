import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router";
import { ChevronLeft, ChevronRight, Loader2, RefreshCw } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  formatDateTime,
  formatNumber,
  formatPercent,
  formatSeconds,
  formatUsd,
} from "@/lib/format";
import { ApiError, getMetricsSummary, listRuns } from "@/services/api";
import type { MetricsSummaryResponse, RunHistoryResponse, RunSummary } from "@/types/history";

const PAGE_SIZE = 20;

/** Statuses that need a second look are not shown in the same colour as success. */
function statusVariant(run: RunSummary): "default" | "secondary" | "destructive" {
  if (run.state === "failed" || run.workflow_status === "failed") return "destructive";
  if (run.workflow_status === "completed_with_errors") return "destructive";
  if (run.workflow_status === "completed") return "default";
  return "secondary";
}

function statusLabel(run: RunSummary): string {
  return (run.workflow_status ?? run.state).replace(/_/g, " ");
}

/**
 * Every finished run, and what they add up to.
 *
 * The aggregates answer questions a single run cannot: how often the Editor
 * approves, how many revisions a run usually takes, which agent fails most,
 * and what all of it would cost on a paid plan.
 */
export default function RunHistory() {
  const [page, setPage] = useState(0);
  const [refreshes, setRefreshes] = useState(0);
  const [history, setHistory] = useState<RunHistoryResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsSummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);

  // Loading is derived rather than stored: the request in flight is whichever
  // one has not yet landed.
  const requestKey = `${page}:${refreshes}`;
  const loading = loadedKey !== requestKey;

  const refresh = useCallback(() => setRefreshes((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([listRuns(PAGE_SIZE, page * PAGE_SIZE), getMetricsSummary()])
      .then(([runs, summary]) => {
        if (cancelled) return;
        setHistory(runs);
        setMetrics(summary);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(cause instanceof ApiError ? cause.message : "Could not load run history.");
      })
      .finally(() => {
        if (!cancelled) setLoadedKey(requestKey);
      });
    return () => {
      cancelled = true;
    };
  }, [page, requestKey]);

  const totalPages = history ? Math.max(1, Math.ceil(history.total / PAGE_SIZE)) : 1;

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-6 py-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-heading text-2xl font-semibold tracking-tight">Run history</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Finished runs and the metrics across them.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
          {loading ? (
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          ) : (
            <RefreshCw className="size-4" aria-hidden="true" />
          )}
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>History unavailable</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {history && !history.history_enabled && (
        <Alert>
          <AlertTitle>Run history is switched off</AlertTitle>
          <AlertDescription>
            Set RUN_STORE_ENABLED=true in backend/.env to record finished runs.
          </AlertDescription>
        </Alert>
      )}

      {metrics?.summary && metrics.summary.total_runs > 0 && (
        <SummarySection metrics={metrics} />
      )}

      {history?.history_enabled && (
        <Card>
          <CardHeader>
            <CardTitle>Runs</CardTitle>
            <CardDescription>
              {history.total === 0
                ? "No finished runs yet."
                : `${history.total.toLocaleString()} finished run(s), newest first.`}
            </CardDescription>
          </CardHeader>
          {history.total === 0 ? (
            <CardContent>
              <Button asChild size="sm">
                <Link to="/">Start a run</Link>
              </Button>
            </CardContent>
          ) : (
            <CardContent className="space-y-4">
              <RunTable runs={history.runs} />
              {totalPages > 1 && (
                <div className="flex items-center justify-end gap-2 text-sm">
                  <span className="text-muted-foreground">
                    Page {page + 1} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    aria-label="Previous page"
                    disabled={page === 0 || loading}
                    onClick={() => setPage((p) => p - 1)}
                  >
                    <ChevronLeft className="size-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    aria-label="Next page"
                    disabled={page + 1 >= totalPages || loading}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    <ChevronRight className="size-4" />
                  </Button>
                </div>
              )}
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );
}

function SummarySection({ metrics }: { metrics: MetricsSummaryResponse }) {
  const summary = metrics.summary;
  if (!summary) return null;

  const outcomes = Object.entries(summary.runs_by_workflow_status).sort((a, b) => b[1] - a[1]);
  const crashed = summary.runs_by_state.failed ?? 0;

  return (
    <>
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Runs" value={formatNumber(summary.total_runs)} />
        <Stat label="Approval rate" value={formatPercent(summary.approval_rate)} />
        <Stat label="Mean quality" value={formatNumber(summary.mean_quality, 1)} />
        <Stat label="Mean revisions" value={formatNumber(summary.mean_revisions, 2)} />
        <Stat label="Mean duration" value={formatSeconds(summary.mean_duration_seconds)} />
        <Stat label="Mean tokens" value={formatNumber(summary.mean_tokens)} />
        <Stat label="Mean cost" value={formatUsd(summary.mean_cost_usd)} />
        <Stat label="Total cost" value={formatUsd(summary.total_cost_usd)} />
      </dl>
      <p className="-mt-2 text-xs text-muted-foreground">
        Costs are list-price estimates
        {metrics.pricing ? ` (${metrics.pricing.model}, ${metrics.pricing.source})` : ""}; free-tier
        runs are not billed. Quality, revisions and approval count only runs that reached a
        verdict.
      </p>

      <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
        <Card>
          <CardHeader>
            <CardTitle>Outcomes</CardTitle>
            <CardDescription>How finished runs ended.</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              {outcomes.map(([status, count]) => (
                <li key={status} className="flex items-center justify-between gap-3">
                  <span className="capitalize">{status.replace(/_/g, " ")}</span>
                  <span className="tabular-nums text-muted-foreground">{count}</span>
                </li>
              ))}
              {crashed > 0 && (
                <li className="flex items-center justify-between gap-3 text-destructive">
                  <span>Crashed before finishing</span>
                  <span className="tabular-nums">{crashed}</span>
                </li>
              )}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Agents</CardTitle>
            <CardDescription>Every execution across every run.</CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-3 font-medium">Agent</th>
                  <th className="py-2 pr-3 text-right font-medium">Executions</th>
                  <th className="py-2 pr-3 text-right font-medium">Failures</th>
                  <th className="py-2 pr-3 text-right font-medium">Mean latency</th>
                  <th className="py-2 text-right font-medium">Mean tokens</th>
                </tr>
              </thead>
              <tbody>
                {summary.agents.map((agent) => (
                  <tr key={agent.agent} className="border-b last:border-0">
                    <td className="py-2 pr-3">{agent.agent.replace(/_/g, " ")}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{agent.executions}</td>
                    <td
                      className={`py-2 pr-3 text-right tabular-nums ${
                        agent.failures > 0 ? "text-destructive" : ""
                      }`}
                    >
                      {agent.failures} ({formatPercent(agent.failure_rate)})
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      {formatSeconds(agent.mean_latency_seconds)}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {formatNumber(agent.mean_tokens)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function RunTable({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="py-2 pr-3 font-medium">Topic</th>
            <th className="py-2 pr-3 font-medium">Status</th>
            <th className="py-2 pr-3 text-right font-medium">Quality</th>
            <th className="py-2 pr-3 text-right font-medium">Revisions</th>
            <th className="py-2 pr-3 text-right font-medium">Duration</th>
            <th className="py-2 pr-3 text-right font-medium">Cost</th>
            <th className="py-2 text-right font-medium">Finished</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id} className="border-b last:border-0 hover:bg-muted/40">
              <td className="max-w-xs py-2 pr-3">
                <Link
                  to={`/runs/${run.run_id}`}
                  className="line-clamp-2 font-medium underline-offset-4 hover:underline"
                >
                  {run.topic}
                </Link>
                <span className="text-xs text-muted-foreground">
                  {run.number_of_posts} post(s) · {run.tone} · {run.model}
                </span>
              </td>
              <td className="py-2 pr-3">
                <Badge variant={statusVariant(run)} className="whitespace-nowrap font-normal">
                  {statusLabel(run)}
                </Badge>
              </td>
              <td className="py-2 pr-3 text-right tabular-nums">
                {formatNumber(run.quality_score, 1)}
              </td>
              <td className="py-2 pr-3 text-right tabular-nums">
                {formatNumber(run.revision_count)}
              </td>
              <td className="py-2 pr-3 text-right tabular-nums">
                {formatSeconds(run.duration_seconds)}
              </td>
              <td className="py-2 pr-3 text-right tabular-nums">{formatUsd(run.cost_usd)}</td>
              <td className="whitespace-nowrap py-2 text-right text-xs text-muted-foreground">
                {formatDateTime(run.finished_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border p-3">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 text-lg font-medium tabular-nums">{value}</dd>
    </div>
  );
}
