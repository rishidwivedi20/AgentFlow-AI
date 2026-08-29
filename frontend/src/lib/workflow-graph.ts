/**
 * The workflow's shape, and how live events map onto it.
 *
 * The topology mirrors `build_workflow` in `backend/app/graph/workflow.py`,
 * maintained by hand like the type mirrors. The backend can describe its own
 * graph (LangGraph's `get_graph()`), so this could become an endpoint later;
 * for six fixed nodes a hand-kept copy is simpler and has no runtime cost.
 *
 * Node ids are `AgentName` values, which is what lets an event's `agent` field
 * address a node with no translation table.
 */

import type { WorkflowEvent } from "@/types/events";
import type { AgentName, AgentStatus } from "@/types/workflow";

export interface AgentNodeSpec {
  id: AgentName;
  label: string;
  /** One line on what this agent is responsible for. */
  role: string;
  position: { x: number; y: number };
}

const COLUMN_X = 140;
const ROW_HEIGHT = 104;

/** Executed top to bottom; the loop edges run back up the left-hand side. */
export const AGENT_NODES: AgentNodeSpec[] = [
  {
    id: "researcher",
    label: "Researcher",
    role: "Searches the web and extracts cited facts",
    position: { x: COLUMN_X, y: 0 },
  },
  {
    id: "planner",
    label: "Planner",
    role: "Assigns each post a distinct angle",
    position: { x: COLUMN_X, y: ROW_HEIGHT },
  },
  {
    id: "writer",
    label: "Writer",
    role: "Drafts and revises the posts",
    position: { x: COLUMN_X, y: ROW_HEIGHT * 2 },
  },
  {
    id: "editor",
    label: "Editor",
    role: "Critiques quality; never rewrites",
    position: { x: COLUMN_X, y: ROW_HEIGHT * 3 },
  },
  {
    id: "fact_checker",
    label: "Fact Checker",
    role: "Verifies claims against the research",
    position: { x: COLUMN_X, y: ROW_HEIGHT * 4 },
  },
  {
    id: "finalize",
    label: "Final Output",
    role: "Selects the best draft and reports caveats",
    position: { x: COLUMN_X, y: ROW_HEIGHT * 5 },
  },
];

export interface WorkflowEdgeSpec {
  id: string;
  source: AgentName;
  target: AgentName;
  label?: string;
  /** Feedback edges run backwards and are drawn distinctly. */
  isLoop?: boolean;
}

export const WORKFLOW_EDGES: WorkflowEdgeSpec[] = [
  { id: "researcher-planner", source: "researcher", target: "planner" },
  { id: "planner-writer", source: "planner", target: "writer" },
  { id: "writer-editor", source: "writer", target: "editor" },
  { id: "editor-fact_checker", source: "editor", target: "fact_checker", label: "approved" },
  { id: "fact_checker-finalize", source: "fact_checker", target: "finalize", label: "verified" },
  // Both critics can send work back to the Writer, sharing one revision budget.
  { id: "editor-writer", source: "editor", target: "writer", label: "revise", isLoop: true },
  {
    id: "fact_checker-writer",
    source: "fact_checker",
    target: "writer",
    label: "unsupported",
    isLoop: true,
  },
];

export interface AgentProgress {
  status: AgentStatus;
  /** How many times this agent has run. Above one means the loop was taken. */
  executions: number;
  /** Latency of the most recent completed or failed execution. */
  latencySeconds: number | null;
  totalTokens: number;
  errorMessage: string | null;
}

function emptyProgress(): AgentProgress {
  return {
    status: "pending",
    executions: 0,
    latencySeconds: null,
    totalTokens: 0,
    errorMessage: null,
  };
}

/**
 * Fold the event stream into per-agent progress.
 *
 * Events are replayed from the start on every reconnect, so this must be a
 * pure fold over the whole list rather than an incremental mutation - applying
 * the same event twice has to be harmless.
 *
 * An agent that ran several times shows its *latest* state: a Writer that
 * failed once and later succeeded is completed, not failed.
 */
export function deriveProgress(events: WorkflowEvent[]): Record<AgentName, AgentProgress> {
  const progress = Object.fromEntries(
    AGENT_NODES.map((node) => [node.id, emptyProgress()]),
  ) as Record<AgentName, AgentProgress>;

  for (const event of events) {
    if (!event.agent) continue;
    const entry = progress[event.agent];
    if (!entry) continue;

    switch (event.type) {
      case "agent_started":
        entry.status = "running";
        entry.executions += 1;
        entry.errorMessage = null;
        break;
      case "agent_completed":
        entry.status = "completed";
        entry.latencySeconds = event.latency_seconds;
        entry.totalTokens += event.usage?.total_tokens ?? 0;
        break;
      case "agent_failed":
        entry.status = "failed";
        entry.latencySeconds = event.latency_seconds;
        entry.errorMessage = event.message;
        break;
      default:
        break;
    }
  }

  return progress;
}

/** The agent currently executing, if any. */
export function activeAgent(progress: Record<AgentName, AgentProgress>): AgentName | null {
  const running = AGENT_NODES.find((node) => progress[node.id].status === "running");
  return running?.id ?? null;
}

/** Total tokens across every agent that reported usage. */
export function totalTokens(events: WorkflowEvent[]): number {
  return events.reduce((sum, event) => sum + (event.usage?.total_tokens ?? 0), 0);
}
