/**
 * Mirrors `backend/app/schemas/run.py`.
 *
 * A *run* is the execution; a *workflow result* is what it produced. The two
 * are separate because a run can finish while its workflow reports failure.
 */

import type { WorkflowRunResponse } from "@/types/workflow";

export type RunState = "queued" | "running" | "completed" | "failed";

export interface RunAcceptedResponse {
  run_id: string;
  state: RunState;
  topic: string;
  created_at: string;
  events_url: string;
  status_url: string;
}

export interface RunStatusResponse {
  run_id: string;
  state: RunState;
  topic: string;
  created_at: string;
  finished_at: string | null;
  /** Events buffered so far; also the cursor a fresh stream would replay from. */
  event_count: number;
  result: WorkflowRunResponse | null;
  /** Set when the run itself failed to execute, as opposed to producing poor output. */
  error_type: string | null;
  error_message: string | null;
}

export function isRunFinished(state: RunState): boolean {
  return state === "completed" || state === "failed";
}
