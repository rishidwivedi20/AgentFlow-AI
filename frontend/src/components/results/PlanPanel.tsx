import { Badge } from "@/components/ui/badge";
import type { ContentPlan, EditorOutput } from "@/types/workflow";

interface PlanPanelProps {
  plan: ContentPlan | null;
  editorFeedback: EditorOutput | null;
}

const SEVERITY_VARIANT = {
  high: "destructive",
  medium: "secondary",
  low: "outline",
} as const;

/**
 * The brief each post was written from, and the Editor's verdict on the result.
 *
 * Shown together because they answer the same question from opposite ends:
 * what the post was supposed to do, and whether it did it.
 */
export function PlanPanel({ plan, editorFeedback }: PlanPanelProps) {
  return (
    <div className="space-y-8">
      <section>
        <h3 className="text-sm font-medium">Content plan</h3>
        {plan && plan.angles.length > 0 ? (
          <ol className="mt-3 space-y-4">
            {plan.angles.map((angle) => (
              <li key={angle.post_number} className="rounded-md border p-4 text-sm">
                <div className="flex items-start justify-between gap-3">
                  <p className="font-medium">
                    Post {angle.post_number}: {angle.angle}
                  </p>
                  <Badge variant="secondary" className="shrink-0 font-normal">
                    {angle.content_type}
                  </Badge>
                </div>

                <dl className="mt-3 space-y-1.5 text-xs">
                  <Field label="Audience" value={angle.target_audience} />
                  <Field label="Hook strategy" value={angle.hook_strategy} />
                  <Field label="Takeaway" value={angle.intended_takeaway} />
                </dl>

                {angle.key_points.length > 0 && (
                  <ul className="mt-3 list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                    {angle.key_points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ol>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">No content plan was produced.</p>
        )}
      </section>

      <section>
        <h3 className="text-sm font-medium">Editorial review</h3>
        {editorFeedback ? (
          <div className="mt-3 space-y-4">
            <div className="flex items-center gap-3 text-sm">
              <Badge variant={editorFeedback.approved ? "default" : "secondary"}>
                {editorFeedback.approved ? "Approved" : "Not approved"}
              </Badge>
              <span className="tabular-nums text-muted-foreground">
                {editorFeedback.quality_score.toFixed(1)} / 10
              </span>
            </div>

            {editorFeedback.issues.length > 0 && (
              <ul className="space-y-2">
                {editorFeedback.issues.map((issue, index) => (
                  <li
                    key={`${issue.description}-${index}`}
                    className="flex items-start gap-2 text-sm"
                  >
                    <Badge
                      variant={SEVERITY_VARIANT[issue.severity]}
                      className="mt-0.5 shrink-0 font-normal"
                    >
                      {issue.severity}
                    </Badge>
                    <span>
                      <span className="text-muted-foreground">
                        {issue.dimension.replace(/_/g, " ")}:{" "}
                      </span>
                      {issue.description}
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {editorFeedback.feedback.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground">
                  Instructions given to the writer
                </p>
                <ul className="mt-1.5 list-disc space-y-1 pl-4 text-sm">
                  {editorFeedback.feedback.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">The editor produced no verdict.</p>
        )}
      </section>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="shrink-0 text-muted-foreground">{label}:</dt>
      <dd>{value}</dd>
    </div>
  );
}
