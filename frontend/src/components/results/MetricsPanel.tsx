import { Badge } from "@/components/ui/badge";
import { formatUsd } from "@/lib/format";
import type {
  AgentMetric,
  CostEstimate,
  ModelPricing,
  TokenUsage,
  WorkflowErrorRecord,
} from "@/types/workflow";

interface MetricsPanelProps {
  metrics: AgentMetric[];
  errors: WorkflowErrorRecord[];
  revisionCount: number;
  maxRevisions: number;
  cost: CostEstimate | null;
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

/**
 * One call's cost, matching `usage_cost` in `backend/app/services/pricing.py`.
 *
 * Output is total minus prompt because the providers disagree on whether
 * their output count includes reasoning; the difference is right for both.
 */
function callCost(usage: TokenUsage, pricing: ModelPricing): number {
  const fromTotal = usage.total_tokens - usage.prompt_tokens;
  const output = fromTotal > 0 ? fromTotal : usage.output_tokens + usage.thought_tokens;
  return (usage.prompt_tokens * pricing.input_per_mtok + output * pricing.output_per_mtok) / 1e6;
}

/**
 * Per-execution timing and token usage.
 *
 * One row per *execution*, not per agent: the Writer running three times is
 * three rows, which is what makes the cost of the revision loop visible.
 *
 * Token cells are blank where the provider reported nothing. Zero would read
 * as "this was free", which is a different claim from "we do not know".
 */
export function MetricsPanel({
  metrics,
  errors,
  revisionCount,
  maxRevisions,
  cost,
}: MetricsPanelProps) {
  if (metrics.length === 0) {
    return (
      <p className="py-10 text-center text-sm text-muted-foreground">No metrics were recorded.</p>
    );
  }

  const totalLatency = sum(metrics.map((m) => m.latency_seconds));
  const withUsage = metrics.filter((m) => m.usage !== null);
  const totalTokens = sum(withUsage.map((m) => m.usage?.total_tokens ?? 0));
  const thoughtTokens = sum(withUsage.map((m) => m.usage?.thought_tokens ?? 0));

  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-5">
        <Stat label="Executions" value={String(metrics.length)} />
        <Stat label="Total latency" value={`${totalLatency.toFixed(1)}s`} />
        <Stat label="Total tokens" value={totalTokens.toLocaleString()} />
        <Stat label="Revisions" value={`${revisionCount} / ${maxRevisions}`} />
        <Stat label="Est. cost" value={cost ? formatUsd(cost.total_usd) : "—"} />
      </dl>

      <p className="text-xs text-muted-foreground">
        {cost ? (
          <>
            Estimated at list price for {cost.pricing.model}: $
            {cost.pricing.input_per_mtok}/M input, ${cost.pricing.output_per_mtok}/M output (
            {cost.pricing.source}). Free-tier runs are not billed.
            {cost.unpriced_calls > 0 &&
              ` ${cost.unpriced_calls} call(s) reported no usage, so this is a lower bound.`}
          </>
        ) : (
          "No price is known for this model, so cost is not estimated. Set LLM_INPUT_PRICE_PER_MTOK and LLM_OUTPUT_PRICE_PER_MTOK in backend/.env to enable it."
        )}
      </p>

      {thoughtTokens > 0 && (
        <p className="text-xs text-muted-foreground">
          {thoughtTokens.toLocaleString()} of those (
          {Math.round((thoughtTokens / totalTokens) * 100)}%) were internal reasoning tokens, which
          the provider bills separately.
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-2 pr-3 font-medium">#</th>
              <th className="py-2 pr-3 font-medium">Agent</th>
              <th className="py-2 pr-3 font-medium">Status</th>
              <th className="py-2 pr-3 text-right font-medium">Latency</th>
              <th className="py-2 pr-3 text-right font-medium">Prompt</th>
              <th className="py-2 pr-3 text-right font-medium">Output</th>
              <th className="py-2 pr-3 text-right font-medium">Thinking</th>
              <th className="py-2 pr-3 text-right font-medium">Total</th>
              <th className="py-2 text-right font-medium">Est. cost</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => (
              <tr key={metric.sequence} className="border-b last:border-0">
                <td className="py-2 pr-3 font-mono text-xs text-muted-foreground">
                  {metric.sequence}
                </td>
                <td className="py-2 pr-3">{metric.agent}</td>
                <td className="py-2 pr-3">
                  <Badge
                    variant={metric.status === "failed" ? "destructive" : "secondary"}
                    className="font-normal"
                  >
                    {metric.status}
                  </Badge>
                </td>
                <td className="py-2 pr-3 text-right tabular-nums">
                  {metric.latency_seconds.toFixed(2)}s
                </td>
                <Tokens value={metric.usage?.prompt_tokens} />
                <Tokens value={metric.usage?.output_tokens} />
                <Tokens value={metric.usage?.thought_tokens} />
                <Tokens value={metric.usage?.total_tokens} />
                <td className="py-2 text-right tabular-nums">
                  {cost && metric.usage ? (
                    formatUsd(callCost(metric.usage, cost.pricing))
                  ) : (
                    <span className="text-muted-foreground/50">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {errors.length > 0 && (
        <section>
          <h3 className="text-sm font-medium">Errors</h3>
          <ul className="mt-2 space-y-2">
            {errors.map((error, index) => (
              <li
                key={`${error.agent}-${index}`}
                className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium">{error.agent}</span>
                  <Badge variant="outline" className="font-normal">
                    {error.error_type}
                  </Badge>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{error.message}</p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function Tokens({ value, last }: { value?: number; last?: boolean }) {
  return (
    <td className={`py-2 text-right tabular-nums ${last ? "" : "pr-3"}`}>
      {value === undefined ? (
        <span className="text-muted-foreground/50">—</span>
      ) : (
        value.toLocaleString()
      )}
    </td>
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
