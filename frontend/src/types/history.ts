/**
 * Mirrors `backend/app/services/run_store.py`, `RunHistoryResponse` in
 * `backend/app/schemas/run.py` and `backend/app/routes/metrics.py`.
 */

import type { RunState } from "@/types/run";
import type { AgentName, ModelPricing, Platform, Tone, WorkflowStatus } from "@/types/workflow";

export interface RunSummary {
  run_id: string;
  topic: string;
  platform: Platform;
  tone: Tone;
  number_of_posts: number;
  state: RunState;
  /** Null when the run crashed before the workflow reached a status. */
  workflow_status: WorkflowStatus | null;
  created_at: string;
  finished_at: string;
  duration_seconds: number;
  revision_count: number | null;
  approved: boolean | null;
  quality_score: number | null;
  total_tokens: number | null;
  cost_usd: number | null;
  provider: string;
  model: string;
  error_type: string | null;
}

export interface RunHistoryResponse {
  /** False when history is switched off on the backend; `runs` is then empty. */
  history_enabled: boolean;
  total: number;
  limit: number;
  offset: number;
  runs: RunSummary[];
}

export interface AgentAggregate {
  agent: AgentName;
  executions: number;
  failures: number;
  failure_rate: number;
  mean_latency_seconds: number;
  /** Null for nodes that make no LLM call. */
  mean_tokens: number | null;
}

/**
 * Aggregates over stored runs. Quality, revision and approval averages only
 * count runs that produced a verdict - a crash is not a score of zero.
 */
export interface MetricsSummary {
  total_runs: number;
  runs_by_state: Record<string, number>;
  runs_by_workflow_status: Record<string, number>;
  approval_rate: number | null;
  mean_revisions: number | null;
  mean_quality: number | null;
  mean_duration_seconds: number | null;
  mean_tokens: number | null;
  total_cost_usd: number | null;
  mean_cost_usd: number | null;
  first_run_at: string | null;
  last_run_at: string | null;
  agents: AgentAggregate[];
}

export interface MetricsSummaryResponse {
  history_enabled: boolean;
  summary: MetricsSummary | null;
  /** Rates for the currently configured model. Null when it is unpriced. */
  pricing: ModelPricing | null;
}
