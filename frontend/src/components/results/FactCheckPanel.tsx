import { AlertTriangle, CircleHelp, ExternalLink, ShieldCheck } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ClaimVerdict, FactCheckReport, Source, VerificationStatus } from "@/types/workflow";

const VERDICT_ORDER: ClaimVerdict[] = ["unsupported", "uncertain", "verified"];

const VERDICT_META: Record<
  ClaimVerdict,
  { label: string; description: string; className: string; icon: typeof ShieldCheck }
> = {
  unsupported: {
    label: "Unsupported",
    description: "Contradicted by the research, or specific with nothing behind it.",
    className: "border-destructive/40 bg-destructive/5",
    icon: AlertTriangle,
  },
  uncertain: {
    label: "Uncertain",
    description: "Plausible and related, but not directly established.",
    className: "border-amber-500/40 bg-amber-500/5",
    icon: CircleHelp,
  },
  verified: {
    label: "Verified",
    description: "Directly supported by a research fact.",
    className: "border-emerald-500/40 bg-emerald-500/5",
    icon: ShieldCheck,
  },
};

const STATUS_MESSAGE: Record<VerificationStatus, string> = {
  verified: "Every extracted claim is supported by the research, or the posts make no claims.",
  partially_verified:
    "No claim is contradicted, but some could not be confirmed against the research.",
  unsupported_claims:
    "At least one claim is not supported by the research. Review these before publishing.",
  not_checked: "The fact checker did not run, so these claims are unverified.",
};

interface FactCheckPanelProps {
  report: FactCheckReport | null;
  sources: Source[];
}

/**
 * How each factual claim stands against this run's research.
 *
 * Grouped worst-first: an unsupported claim is the thing a reader must act on,
 * and burying it under a list of verified ones would defeat the purpose.
 */
export function FactCheckPanel({ report, sources }: FactCheckPanelProps) {
  if (!report) {
    return (
      <Alert>
        <CircleHelp aria-hidden="true" />
        <AlertTitle>Not fact checked</AlertTitle>
        <AlertDescription>
          {STATUS_MESSAGE.not_checked} Absence of a check is not evidence that the claims are
          correct.
        </AlertDescription>
      </Alert>
    );
  }

  const grouped = VERDICT_ORDER.map((verdict) => ({
    verdict,
    claims: report.claims.filter((claim) => claim.verdict === verdict),
  })).filter((group) => group.claims.length > 0);

  return (
    <div className="space-y-6">
      <Alert variant={report.critical_issue_count > 0 ? "destructive" : "default"}>
        <AlertTitle className="capitalize">
          {report.overall_status.replace(/_/g, " ")}
        </AlertTitle>
        <AlertDescription>{STATUS_MESSAGE[report.overall_status]}</AlertDescription>
      </Alert>

      {report.claims.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">
          No factual assertions were found in these posts.
        </p>
      ) : (
        grouped.map(({ verdict, claims }) => {
          const meta = VERDICT_META[verdict];
          const Icon = meta.icon;
          return (
            <section key={verdict}>
              <h3 className="flex items-center gap-2 text-sm font-medium">
                <Icon className="size-4" aria-hidden="true" />
                {meta.label}
                <span className="text-muted-foreground">({claims.length})</span>
              </h3>
              <p className="mt-1 text-xs text-muted-foreground">{meta.description}</p>

              <ul className="mt-3 space-y-3">
                {claims.map((claim, index) => {
                  const source = sources[claim.supporting_source_index];
                  return (
                    <li
                      key={`${claim.claim}-${index}`}
                      className={cn("rounded-md border p-3 text-sm", meta.className)}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <p className="leading-relaxed">{claim.claim}</p>
                        <Badge variant="outline" className="shrink-0 font-normal">
                          Post {claim.post_index}
                        </Badge>
                      </div>
                      <p className="mt-2 text-xs leading-snug text-muted-foreground">
                        {claim.reasoning}
                      </p>
                      {/* -1 means nothing supports the claim; always the case
                          for an unsupported verdict. */}
                      {claim.supporting_source_index >= 0 && source && (
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="mt-2 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground hover:underline"
                        >
                          Source [{claim.supporting_source_index}] {source.title}
                          <ExternalLink className="size-3" aria-hidden="true" />
                        </a>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          );
        })
      )}
    </div>
  );
}
