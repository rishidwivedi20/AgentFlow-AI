import { Link, NavLink } from "react-router";

import { ConnectionBadge } from "@/components/ConnectionBadge";
import { useHealth } from "@/hooks/useHealth";

const CONNECTION_LABELS = {
  checking: "Checking backend…",
  connected: "Backend connected",
  unreachable: "Backend unreachable",
} as const;

const NAV_LINKS = [
  { to: "/", label: "New run" },
  { to: "/history", label: "History" },
] as const;

/**
 * Application header.
 *
 * Carries the backend connection badge on every page: a run that cannot start
 * is almost always a backend that is not running, and finding that out from a
 * failed submission is worse than seeing it beforehand.
 */
export function AppHeader() {
  const { state } = useHealth();

  return (
    <header className="border-b">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6 py-4">
        <Link to="/" className="group">
          <span className="font-heading text-lg font-semibold tracking-tight">AgentFlow-AI</span>
          <span className="ml-2 text-sm text-muted-foreground">
            multi-agent content generation
          </span>
        </Link>
        <div className="flex items-center gap-5">
          <nav className="flex items-center gap-4 text-sm">
            {NAV_LINKS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  isActive
                    ? "font-medium text-foreground"
                    : "text-muted-foreground transition-colors hover:text-foreground"
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <ConnectionBadge state={state} label={CONNECTION_LABELS[state]} />
        </div>
      </div>
    </header>
  );
}
