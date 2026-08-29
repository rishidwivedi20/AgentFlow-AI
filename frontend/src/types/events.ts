/**
 * Mirrors `backend/app/models/events.py`.
 *
 * Events carry execution status only - never prompts, model reasoning or
 * internals. The backend enforces that with a test; this type documents it.
 */

import type { AgentName, AgentStatus, TokenUsage, WorkflowStatus } from "@/types/workflow";

export type EventType =
  | "workflow_started"
  | "agent_started"
  | "agent_completed"
  | "agent_failed"
  | "workflow_completed"
  | "workflow_failed";

export interface WorkflowEvent {
  /** Monotonic within the run. Doubles as the replay cursor and the SSE `id`. */
  sequence: number;
  type: EventType;
  timestamp: string;

  agent: AgentName | null;
  status: AgentStatus | null;
  /**
   * Execution index within the run, matching `AgentMetric.sequence`. Unique per
   * execution, so a second `writer` event with a higher index marks a pass
   * through the feedback loop. Not the revision counter.
   */
  agent_execution: number | null;
  latency_seconds: number | null;
  usage: TokenUsage | null;
  error_type: string | null;
  /** Client-safe explanation. Never a stack trace. */
  message: string | null;
  workflow_status: WorkflowStatus | null;
}

/** Whether this event ends the stream. */
export function isTerminalEvent(event: WorkflowEvent): boolean {
  return event.type === "workflow_completed" || event.type === "workflow_failed";
}
