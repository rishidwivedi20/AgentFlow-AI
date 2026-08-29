import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";
import type { EventType, WorkflowEvent } from "@/types/events";

const TYPE_LABELS: Record<EventType, string> = {
  workflow_started: "Workflow started",
  agent_started: "started",
  agent_completed: "completed",
  agent_failed: "failed",
  workflow_completed: "Workflow finished",
  workflow_failed: "Workflow failed",
};

const TYPE_STYLES: Record<EventType, string> = {
  workflow_started: "text-muted-foreground",
  agent_started: "text-amber-700 dark:text-amber-400",
  agent_completed: "text-emerald-700 dark:text-emerald-400",
  agent_failed: "text-destructive",
  workflow_completed: "text-foreground",
  workflow_failed: "text-destructive",
};

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
  });
}

function describe(event: WorkflowEvent): string {
  const label = TYPE_LABELS[event.type];
  if (!event.agent) return label;

  const parts = [`${event.agent} ${label}`];
  if (event.latency_seconds !== null) parts.push(`${event.latency_seconds.toFixed(1)}s`);
  if (event.usage) parts.push(`${event.usage.total_tokens.toLocaleString()} tok`);
  return parts.join(" · ");
}

/**
 * Chronological log of what the run did.
 *
 * Complements the graph, which only shows current state: an agent that ran
 * three times collapses to one node there, but every pass is a separate line
 * here.
 */
export function ActivityLog({ events }: { events: WorkflowEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Follow the tail as events arrive, the way a console does.
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [events.length]);

  if (events.length === 0) {
    return (
      <p className="px-4 py-8 text-center text-sm text-muted-foreground">
        Waiting for the first event…
      </p>
    );
  }

  return (
    <ol className="max-h-[560px] overflow-y-auto px-1 py-2 font-mono text-xs">
      {events.map((event) => (
        <li
          key={event.sequence}
          className="flex items-baseline gap-3 rounded px-3 py-1.5 hover:bg-muted/50"
        >
          <span className="tabular-nums text-muted-foreground">{formatTime(event.timestamp)}</span>
          <span className={cn("flex-1", TYPE_STYLES[event.type])}>
            {describe(event)}
            {event.workflow_status && (
              <span className="ml-2 text-muted-foreground">→ {event.workflow_status}</span>
            )}
            {event.message && (
              <span className="mt-0.5 block text-[11px] leading-snug text-muted-foreground">
                {event.message}
              </span>
            )}
          </span>
        </li>
      ))}
      <div ref={endRef} />
    </ol>
  );
}
