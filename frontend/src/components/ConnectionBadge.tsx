import { cn } from "@/lib/utils";
import type { ConnectionState } from "@/hooks/useHealth";

interface ConnectionBadgeProps {
  state: ConnectionState;
  label: string;
}

const STATE_STYLES: Record<ConnectionState, string> = {
  checking: "bg-amber-500/10 text-amber-700 ring-amber-500/25 dark:text-amber-400",
  connected: "bg-emerald-500/10 text-emerald-700 ring-emerald-500/25 dark:text-emerald-400",
  unreachable: "bg-destructive/10 text-destructive ring-destructive/25",
};

const DOT_STYLES: Record<ConnectionState, string> = {
  checking: "bg-amber-500 animate-pulse",
  connected: "bg-emerald-500",
  unreachable: "bg-destructive",
};

/** Status pill reporting backend reachability. */
export function ConnectionBadge({ state, label }: ConnectionBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset",
        STATE_STYLES[state],
      )}
    >
      <span className={cn("size-2 rounded-full", DOT_STYLES[state])} aria-hidden="true" />
      {label}
    </span>
  );
}
