import { EvidencePanel } from "@/components/results/EvidencePanel";
import { FactCheckPanel } from "@/components/results/FactCheckPanel";
import { MetricsPanel } from "@/components/results/MetricsPanel";
import { PlanPanel } from "@/components/results/PlanPanel";
import { PostList } from "@/components/results/PostList";
import { ResearchPanel } from "@/components/results/ResearchPanel";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { WorkflowRunResponse } from "@/types/workflow";

/**
 * Everything the run produced, behind tabs.
 *
 * Posts come first because they are the deliverable; research, evidence, plan,
 * fact check and metrics are the support behind them. Evidence sits next to
 * research because it is the layer underneath it - the source passages the
 * facts were extracted from. Counts appear on the tabs so the
 * reader can see there are unsupported claims without opening the tab - the one
 * number that changes whether the posts are safe to use.
 */
export function ResultsPanel({ result }: { result: WorkflowRunResponse }) {
  const criticalIssues = result.fact_check?.critical_issue_count ?? 0;
  const evidenceCount = result.evidence?.chunks.length ?? 0;

  return (
    <Tabs defaultValue="posts" className="w-full">
      <TabsList>
        <TabsTrigger value="posts">
          Posts
          {result.final_posts.length > 0 && (
            <span className="ml-1.5 text-muted-foreground">{result.final_posts.length}</span>
          )}
        </TabsTrigger>
        <TabsTrigger value="research">
          Research
          {result.sources.length > 0 && (
            <span className="ml-1.5 text-muted-foreground">{result.sources.length}</span>
          )}
        </TabsTrigger>
        <TabsTrigger value="evidence">
          Evidence
          {evidenceCount > 0 && (
            <span className="ml-1.5 text-muted-foreground">{evidenceCount}</span>
          )}
        </TabsTrigger>
        <TabsTrigger value="plan">Plan &amp; review</TabsTrigger>
        <TabsTrigger value="factcheck">
          Fact check
          {criticalIssues > 0 && (
            <Badge variant="destructive" className="ml-1.5 h-4 px-1 text-[10px]">
              {criticalIssues}
            </Badge>
          )}
        </TabsTrigger>
        <TabsTrigger value="metrics">Metrics</TabsTrigger>
      </TabsList>

      <TabsContent value="posts" className="mt-6">
        <PostList posts={result.final_posts} />
      </TabsContent>

      <TabsContent value="research" className="mt-6">
        <ResearchPanel
          summary={result.research_summary}
          facts={result.research_facts}
          limitations={result.research_limitations}
          sources={result.sources}
        />
      </TabsContent>

      <TabsContent value="evidence" className="mt-6">
        <EvidencePanel evidence={result.evidence} sources={result.sources} />
      </TabsContent>

      <TabsContent value="plan" className="mt-6">
        <PlanPanel plan={result.content_plan} editorFeedback={result.editor_feedback} />
      </TabsContent>

      <TabsContent value="factcheck" className="mt-6">
        <FactCheckPanel report={result.fact_check} sources={result.sources} />
      </TabsContent>

      <TabsContent value="metrics" className="mt-6">
        <MetricsPanel
          metrics={result.agent_metrics}
          errors={result.errors}
          revisionCount={result.revision_count}
          maxRevisions={result.max_revisions}
        />
      </TabsContent>
    </Tabs>
  );
}
