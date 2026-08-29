import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { AlertCircle, Check, Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import type { AgentProgress } from "@/lib/workflow-graph";
import type { AgentStatus } from "@/types/workflow";

export interface AgentNodeData extends Record<string, unknown> {
  label: string;
  role: string;
  progress: AgentProgress;
}

const STATUS_STYLES: Record<AgentStatus, string> = {
  pending: "border-border bg-card text-muted-foreground",
  running: "border-amber-500/60 bg-amber-500/10 text-foreground ring-2 ring-amber-500/30",
  completed: "border-emerald-500/50 bg-emerald-500/10 text-foreground",
  failed: "border-destructive/50 bg-destructive/10 text-foreground",
};

function StatusIcon({ status }: { status: AgentStatus }) {
  switch (status) {
    case "running":
      return <Loader2 className="size-3.5 animate-spin text-amber-600" aria-hidden="true" />;
    case "completed":
      return <Check className="size-3.5 text-emerald-600" aria-hidden="true" />;
    case "failed":
      return <AlertCircle className="size-3.5 text-destructive" aria-hidden="true" />;
    default:
      return <span className="size-3.5 rounded-full border border-current opacity-40" />;
  }
}

/**
 * One agent in the workflow graph.
 *
 * The running node is highlighted with a ring, and an execution count appears
 * once an agent has run more than once - which is the visible evidence that
 * the feedback loop was taken.
 */
export function AgentNode({ data }: NodeProps) {
  const { label, role, progress } = data as AgentNodeData;
  const { status, executions, latencySeconds, totalTokens, errorMessage } = progress;

  return (
    <div
      className={cn(
        "w-60 rounded-lg border px-3 py-2.5 shadow-sm transition-colors",
        STATUS_STYLES[status],
      )}
      aria-label={`${label}: ${status}`}
    >
      {/* Top target and bottom source carry the forward path; the left pair
          carries the feedback edges so they route clear of the column. */}
      <Handle type="target" position={Position.Top} className="!size-1.5 !bg-muted-foreground" />
      <Handle
        type="target"
        id="loop"
        position={Position.Left}
        className="!size-1.5 !bg-muted-foreground"
      />

      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-sm font-medium">
          <StatusIcon status={status} />
          {label}
        </span>
        {executions > 1 && (
          <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[10px] font-medium tabular-nums">
            ×{executions}
          </span>
        )}
      </div>

      <p className="mt-1 text-[11px] leading-snug opacity-70">{role}</p>

      {(latencySeconds !== null || totalTokens > 0) && (
        <div className="mt-1.5 flex gap-3 text-[10px] tabular-nums opacity-70">
          {latencySeconds !== null && <span>{latencySeconds.toFixed(1)}s</span>}
          {totalTokens > 0 && <span>{totalTokens.toLocaleString()} tokens</span>}
        </div>
      )}

      {errorMessage && (
        <p className="mt-1.5 line-clamp-2 text-[10px] leading-snug text-destructive">
          {errorMessage}
        </p>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="!size-1.5 !bg-muted-foreground"
      />
      <Handle
        type="source"
        id="loop"
        position={Position.Left}
        className="!size-1.5 !bg-muted-foreground"
      />
    </div>
  );
}
