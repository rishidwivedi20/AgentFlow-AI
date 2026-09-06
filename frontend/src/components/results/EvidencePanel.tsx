import { ExternalLink, FileSearch, Info, TriangleAlert } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import type { EvidenceSet, RetrievalStatus, Source } from "@/types/workflow";

function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

const STATUS_META: Record<
  RetrievalStatus,
  { title: string; description: string; icon: typeof Info }
> = {
  retrieved: {
    title: "Evidence retrieved",
    description:
      "Passages below are the source text ranked most relevant to the topic. They are extracts, not whole pages, so material relevant to the topic may still be missing.",
    icon: FileSearch,
  },
  weak_evidence: {
    title: "Weak evidence",
    description:
      "No passage cleared the relevance threshold. This means these sources do not clearly discuss the topic - it is not evidence that any claim is false.",
    icon: TriangleAlert,
  },
  no_evidence: {
    title: "No page text retrieved",
    description:
      "The sources returned no extractable page text - typically PDFs, paywalls, or pages that build themselves in the browser. The run fell back to search summaries.",
    icon: Info,
  },
  not_attempted: {
    title: "Retrieval did not run",
    description:
      "Passage retrieval was disabled, unconfigured, or failed. The run used search summaries alone. This says nothing about the sources themselves.",
    icon: Info,
  },
};

interface EvidencePanelProps {
  evidence: EvidenceSet | null;
  sources: Source[];
}

/**
 * The passages the run's claims were actually grounded in.
 *
 * Shows the retrieval's own limits as prominently as its results: how much of
 * the corpus was searched, and what the scores do and do not mean. A panel that
 * displayed only the winning passages would imply the sources were read whole,
 * which is exactly the impression a capped, chunked pipeline must not give.
 *
 * Chunk text is verbatim content from third-party web pages. It is rendered as
 * text - React escapes it - and never as markup.
 */
export function EvidencePanel({ evidence, sources }: EvidencePanelProps) {
  if (!evidence) {
    return (
      <p className="py-10 text-center text-sm text-muted-foreground">
        This run predates evidence retrieval.
      </p>
    );
  }

  const meta = STATUS_META[evidence.status];
  const Icon = meta.icon;
  const unsearched = evidence.chunks_indexed > 0;

  return (
    <div className="space-y-6">
      <Alert variant={evidence.status === "weak_evidence" ? "destructive" : "default"}>
        <Icon aria-hidden="true" />
        <AlertTitle>{meta.title}</AlertTitle>
        <AlertDescription>{meta.description}</AlertDescription>
      </Alert>

      {unsearched && (
        <dl className="grid grid-cols-3 gap-4">
          <Stat label="Passages used" value={String(evidence.chunks.length)} />
          <Stat label="Passages searched" value={evidence.chunks_indexed.toLocaleString()} />
          <Stat label="Sources indexed" value={String(evidence.documents_indexed)} />
        </dl>
      )}

      {evidence.warnings.length > 0 && (
        <ul className="space-y-2">
          {evidence.warnings.map((warning) => (
            <li
              key={warning}
              className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs leading-snug"
            >
              {warning}
            </li>
          ))}
        </ul>
      )}

      {evidence.chunks.length > 0 && (
        <section>
          <h3 className="text-sm font-medium">
            Retrieved passages{" "}
            <span className="text-muted-foreground">({evidence.chunks.length})</span>
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Ranked by cosine similarity to the topic. A score measures textual relevance to
            the query - not accuracy, and not how strongly the passage supports any claim.
          </p>

          <ol className="mt-3 space-y-3">
            {evidence.chunks.map((chunk) => {
              const source = sources[chunk.source_index];
              return (
                <li
                  key={`${chunk.source_index}-${chunk.position}`}
                  className="rounded-md border p-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    {source ? (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-xs font-medium hover:underline"
                      >
                        [{chunk.source_index}] {domainOf(source.url)}
                        <ExternalLink className="size-3 shrink-0" aria-hidden="true" />
                      </a>
                    ) : (
                      /* The backend re-points chunks when duplicate sources are
                         collapsed, so an unresolvable index means something
                         upstream changed rather than a model inventing one. */
                      <span className="text-xs text-muted-foreground">
                        source [{chunk.source_index}] unavailable
                      </span>
                    )}
                    <Badge variant="secondary" className="shrink-0 font-normal tabular-nums">
                      {chunk.similarity.toFixed(3)}
                    </Badge>
                  </div>
                  <blockquote className="mt-2 whitespace-pre-wrap border-l-2 pl-3 text-xs leading-relaxed text-muted-foreground">
                    {chunk.text}
                  </blockquote>
                </li>
              );
            })}
          </ol>
        </section>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border p-3">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 text-lg font-medium tabular-nums">{value}</dd>
    </div>
  );
}
