import { RefreshCw } from "lucide-react";

import { ConnectionBadge } from "@/components/ConnectionBadge";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/hooks/useHealth";
import { API_BASE_URL } from "@/services/api";

const CONNECTION_LABELS = {
  checking: "Checking backend…",
  connected: "Backend connected",
  unreachable: "Backend unreachable",
} as const;

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="font-mono text-sm">{value}</dd>
    </div>
  );
}

export default function App() {
  const { state, health, error, refresh } = useHealth();

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center px-6 py-16">
      <header className="mb-8">
        <h1 className="font-heading text-3xl font-semibold tracking-tight">OrchestrAI</h1>
        <p className="mt-2 text-muted-foreground">
          Multi-agent content generation and evaluation. Phase&nbsp;1 — foundation.
        </p>
      </header>

      <section className="rounded-xl border bg-card p-6 text-card-foreground shadow-sm">
        <div className="flex items-center justify-between gap-4">
          <ConnectionBadge state={state} label={CONNECTION_LABELS[state]} />
          <Button variant="ghost" size="sm" onClick={refresh}>
            <RefreshCw aria-hidden="true" />
            Retry
          </Button>
        </div>

        {error && (
          <p className="mt-4 rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}

        {health && (
          <dl className="mt-4 divide-y">
            <DetailRow label="API" value={API_BASE_URL} />
            <DetailRow label="Version" value={health.version} />
            <DetailRow label="Environment" value={health.environment} />
            <DetailRow label="LLM provider" value={`${health.llm_provider} · ${health.llm_model}`} />
            <DetailRow
              label="Provider key"
              value={health.llm_configured ? "configured" : "not set (expected in Phase 1)"}
            />
          </dl>
        )}
      </section>

      <p className="mt-6 text-sm text-muted-foreground">
        Next: Phase&nbsp;2 wires the LangGraph Researcher → Writer → Editor loop behind{" "}
        <code className="font-mono text-foreground">POST /api/workflows/run</code>.
      </p>
    </main>
  );
}
