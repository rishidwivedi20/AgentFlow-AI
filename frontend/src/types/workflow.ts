/**
 * Mirrors `backend/app/schemas/workflow.py` and the models it exposes.
 *
 * Maintained by hand. When a backend schema changes, update this file in the
 * same commit - there is no codegen step by design.
 */

// --- enums, mirroring app/models/enums.py -----------------------------------

export const PLATFORMS = ["linkedin"] as const;
export type Platform = (typeof PLATFORMS)[number];

export const TONES = ["professional", "conversational", "inspirational", "analytical"] as const;
export type Tone = (typeof TONES)[number];

export const AGENT_NAMES = [
  "retriever",
  "researcher",
  "planner",
  "writer",
  "editor",
  "fact_checker",
  "finalize",
] as const;
/** Also the node ids in the compiled graph, so events map to nodes directly. */
export type AgentName = (typeof AGENT_NAMES)[number];

export type AgentStatus = "pending" | "running" | "completed" | "failed";

export type WorkflowStatus =
  | "pending"
  | "running"
  | "completed"
  | "max_revisions_reached"
  | "completed_unverified"
  | "completed_with_errors"
  | "failed";

/** Verification standing. `not_checked` is a distinct state, not a synonym for verified. */
export type VerificationStatus =
  | "verified"
  | "partially_verified"
  | "unsupported_claims"
  | "not_checked";

// --- research ---------------------------------------------------------------

export interface Source {
  title: string;
  url: string;
  snippet: string;
}

/**
 * How much the retrieval step established.
 *
 * `weak_evidence` means the sources do not clearly discuss the topic - never
 * that a claim is false. `not_attempted` means retrieval did not run at all,
 * which is a statement about configuration, not about the sources.
 */
export type RetrievalStatus =
  | "retrieved"
  | "weak_evidence"
  | "no_evidence"
  | "not_attempted";

export interface EvidenceChunk {
  /** Index into the run's `sources`. */
  source_index: number;
  /** Zero-based position within that source's page text. */
  position: number;
  /** Verbatim page text. Render as text, never as markup. */
  text: string;
  /** Cosine similarity to the retrieval query. */
  similarity: number;
}

export interface EvidenceSet {
  query: string;
  status: RetrievalStatus;
  /** Selected chunks, most relevant first. */
  chunks: EvidenceChunk[];
  documents_indexed: number;
  chunks_indexed: number;
  warnings: string[];
}

export interface ResearchFact {
  statement: string;
  kind: "fact" | "opinion";
  /** Index into the run's `sources`. */
  source_index: number;
}

// --- planning ---------------------------------------------------------------

export type ContentType = "educational" | "storytelling" | "career" | "opinion" | "practical";

export interface ContentAngle {
  post_number: number;
  angle: string;
  content_type: ContentType;
  hook_strategy: string;
  key_points: string[];
  target_audience: string;
  intended_takeaway: string;
}

export interface ContentPlan {
  angles: ContentAngle[];
}

// --- content ----------------------------------------------------------------

export interface Post {
  angle: string;
  content: string;
  hashtags: string[];
}

// --- editorial review -------------------------------------------------------

export type ReviewDimension =
  | "clarity"
  | "structure"
  | "engagement"
  | "repetition"
  | "relevance"
  | "tone_consistency";

export type IssueSeverity = "low" | "medium" | "high";

export interface EditorIssue {
  dimension: ReviewDimension;
  severity: IssueSeverity;
  description: string;
}

export interface EditorOutput {
  approved: boolean;
  /** 0 to 10. */
  quality_score: number;
  issues: EditorIssue[];
  feedback: string[];
}

// --- fact checking ----------------------------------------------------------

export type ClaimVerdict = "verified" | "uncertain" | "unsupported";

export interface FactCheckClaim {
  claim: string;
  verdict: ClaimVerdict;
  reasoning: string;
  post_index: number;
  /** Index into `sources`, or -1 when nothing supports the claim. */
  supporting_source_index: number;
}

export interface FactCheckReport {
  claims: FactCheckClaim[];
  /** Computed server-side; only `unsupported` claims count as critical. */
  critical_issue_count: number;
  overall_status: VerificationStatus;
}

// --- observability ----------------------------------------------------------

export interface TokenUsage {
  prompt_tokens: number;
  output_tokens: number;
  /** Gemini 2.5+ bills internal reasoning separately. */
  thought_tokens: number;
  total_tokens: number;
}

export interface AgentMetric {
  /** Execution index within the run; matches an event's `agent_execution`. */
  sequence: number;
  agent: AgentName;
  status: AgentStatus;
  started_at: string;
  ended_at: string;
  latency_seconds: number;
  /** Null when the agent made no LLM call, or the provider reported none. */
  usage: TokenUsage | null;
  error: string | null;
}

export interface WorkflowErrorRecord {
  agent: AgentName;
  error_type: string;
  message: string;
  occurred_at: string;
}

// --- request and response ---------------------------------------------------

export const MIN_POSTS = 1;
export const MAX_POSTS = 5;

export interface WorkflowRunRequest {
  topic: string;
  platform: Platform;
  tone: Tone;
  number_of_posts: number;
}

// --- cost, mirroring app/services/pricing.py ---------------------------------

export interface ModelPricing {
  model: string;
  /** USD per million prompt tokens. */
  input_per_mtok: number;
  /** USD per million output tokens, reasoning included. */
  output_per_mtok: number;
  /** "settings" when overridden in backend/.env, else the price table's date. */
  source: string;
}

/**
 * Estimated cost at list price. On a free tier the actual charge is zero -
 * this is what the same run would cost on a paid plan.
 */
export interface CostEstimate {
  pricing: ModelPricing;
  total_usd: number;
  priced_calls: number;
  /** LLM calls with no reported usage. Non-zero makes the total a lower bound. */
  unpriced_calls: number;
}

export interface WorkflowRunResponse {
  workflow_status: WorkflowStatus;
  topic: string;
  platform: Platform;
  tone: Tone;
  final_posts: Post[];
  revision_count: number;
  max_revisions: number;
  approved: boolean;
  quality_score: number | null;
  editor_feedback: EditorOutput | null;
  research_summary: string | null;
  research_facts: ResearchFact[];
  research_limitations: string[];
  sources: Source[];
  /** Null on runs that predate retrieval. */
  evidence: EvidenceSet | null;
  content_plan: ContentPlan | null;
  fact_check: FactCheckReport | null;
  agent_metrics: AgentMetric[];
  /** List-price estimate of the run's LLM calls. Null when the model is unpriced. */
  cost: CostEstimate | null;
  errors: WorkflowErrorRecord[];
  /**
   * Caveats to display alongside the output. Distinct from `errors`: nothing
   * here necessarily broke, but the run cannot be presented as a clean success.
   */
  warnings: string[];
}
