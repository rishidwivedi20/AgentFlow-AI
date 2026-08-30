import { ExternalLink, Info } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import type { ResearchFact, Source } from "@/types/workflow";

function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

interface ResearchPanelProps {
  summary: string | null;
  facts: ResearchFact[];
  limitations: string[];
  sources: Source[];
}

/**
 * What the run actually found, and what it could not establish.
 *
 * Facts cite sources by index, so each one resolves to a real URL here. A fact
 * whose index does not resolve is shown without a link rather than silently
 * dropped - the backend already discards uncitable facts, so seeing one here
 * would mean something upstream changed.
 */
export function ResearchPanel({ summary, facts, limitations, sources }: ResearchPanelProps) {
  if (!summary && facts.length === 0 && sources.length === 0) {
    return (
      <p className="py-10 text-center text-sm text-muted-foreground">
        No research was gathered for this run.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {summary && (
        <section>
          <h3 className="text-sm font-medium">Summary</h3>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{summary}</p>
        </section>
      )}

      {limitations.length > 0 && (
        <Alert>
          <Info aria-hidden="true" />
          <AlertTitle>Not established by these sources</AlertTitle>
          <AlertDescription>
            <ul className="mt-1 list-disc space-y-1 pl-4">
              {limitations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {facts.length > 0 && (
        <section>
          <h3 className="text-sm font-medium">
            Findings <span className="text-muted-foreground">({facts.length})</span>
          </h3>
          <ul className="mt-3 space-y-3">
            {facts.map((fact, index) => {
              const source = sources[fact.source_index];
              return (
                <li key={`${fact.statement}-${index}`} className="flex gap-3 text-sm">
                  <Badge
                    variant={fact.kind === "fact" ? "default" : "secondary"}
                    className="mt-0.5 h-fit shrink-0 font-normal"
                  >
                    {fact.kind}
                  </Badge>
                  <div className="min-w-0">
                    <p className="leading-relaxed">{fact.statement}</p>
                    {source && (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-1 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground hover:underline"
                      >
                        {domainOf(source.url)}
                        <ExternalLink className="size-3" aria-hidden="true" />
                      </a>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {sources.length > 0 && (
        <section>
          <h3 className="text-sm font-medium">
            Sources <span className="text-muted-foreground">({sources.length})</span>
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Retrieved by web search. The model cites these by position and never produces a URL of
            its own.
          </p>
          <ol className="mt-3 space-y-3">
            {sources.map((source, index) => (
              <li key={source.url} className="flex gap-3 text-sm">
                <span className="mt-0.5 shrink-0 font-mono text-xs text-muted-foreground">
                  [{index}]
                </span>
                <div className="min-w-0">
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 font-medium hover:underline"
                  >
                    {source.title}
                    <ExternalLink className="size-3 shrink-0" aria-hidden="true" />
                  </a>
                  <p className="text-xs text-muted-foreground">{domainOf(source.url)}</p>
                  {source.snippet && (
                    <p className="mt-1 line-clamp-2 text-xs leading-snug text-muted-foreground">
                      {source.snippet}
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}
