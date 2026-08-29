import { useState } from "react";
import { useNavigate } from "react-router";
import { Loader2, Play } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ApiError, createRun } from "@/services/api";
import { MAX_POSTS, MIN_POSTS, PLATFORMS, TONES } from "@/types/workflow";
import type { Platform, Tone } from "@/types/workflow";

/** Matches the backend's `min_length=3` after stripping whitespace. */
const MIN_TOPIC_LENGTH = 3;
const MAX_TOPIC_LENGTH = 200;

const TONE_LABELS: Record<Tone, string> = {
  professional: "Professional",
  conversational: "Conversational",
  inspirational: "Inspirational",
  analytical: "Analytical",
};

const PLATFORM_LABELS: Record<Platform, string> = {
  linkedin: "LinkedIn",
};

/**
 * Explain a failed submission in terms the reader can act on.
 *
 * Configuration and quota problems are the two most likely failures on a free
 * tier, and a generic "request failed" leaves the user with nowhere to go.
 */
function describeError(error: ApiError): { title: string; detail: string } {
  switch (error.type) {
    case "configuration_error":
      return {
        title: "Backend is not configured",
        detail: `${error.message} The key lives in backend/.env and is never sent to the browser.`,
      };
    case "rate_limited":
    case "research_rate_limited":
      return {
        title: "Free-tier limit reached",
        detail: `${error.message} Wait a little, or reduce the number of posts.`,
      };
    case "network_error":
    case "timeout":
      return { title: "Cannot reach the backend", detail: error.message };
    default:
      return { title: "Could not start the run", detail: error.message };
  }
}

export default function CreateWorkflow() {
  const navigate = useNavigate();

  const [topic, setTopic] = useState("");
  const [platform, setPlatform] = useState<Platform>("linkedin");
  const [tone, setTone] = useState<Tone>("professional");
  const [numberOfPosts, setNumberOfPosts] = useState(2);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const trimmedTopic = topic.trim();
  const topicValid = trimmedTopic.length >= MIN_TOPIC_LENGTH;
  const canSubmit = topicValid && !submitting;

  async function handleSubmit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (!canSubmit) return;

    setSubmitting(true);
    setError(null);
    try {
      const run = await createRun({
        topic: trimmedTopic,
        platform,
        tone,
        number_of_posts: numberOfPosts,
      });
      // The run is queued, not finished - the execution page follows it live.
      void navigate(`/runs/${run.run_id}`);
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause
          : new ApiError(0, "unknown", "An unexpected error occurred."),
      );
      setSubmitting(false);
    }
  }

  const described = error ? describeError(error) : null;

  return (
    <div className="mx-auto max-w-2xl px-6 py-12">
      <h1 className="font-heading text-2xl font-semibold tracking-tight">Generate content</h1>
      <p className="mt-2 text-muted-foreground">
        Five agents research the topic, plan distinct angles, write, review and fact check the
        result. You can watch each step as it runs.
      </p>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Workflow input</CardTitle>
          <CardDescription>
            Every claim in the output is checked against sources gathered for this topic.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="topic">Topic</Label>
              <Input
                id="topic"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="How AI is changing software engineering"
                maxLength={MAX_TOPIC_LENGTH}
                autoFocus
                aria-describedby="topic-hint"
              />
              <p id="topic-hint" className="text-xs text-muted-foreground">
                {topic.length > 0 && !topicValid
                  ? `At least ${MIN_TOPIC_LENGTH} characters.`
                  : "Specific topics produce better research than broad ones."}
              </p>
            </div>

            <div className="grid gap-6 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="platform">Platform</Label>
                <Select value={platform} onValueChange={(v) => setPlatform(v as Platform)}>
                  <SelectTrigger id="platform" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PLATFORMS.map((value) => (
                      <SelectItem key={value} value={value}>
                        {PLATFORM_LABELS[value]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="tone">Tone</Label>
                <Select value={tone} onValueChange={(v) => setTone(v as Tone)}>
                  <SelectTrigger id="tone" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TONES.map((value) => (
                      <SelectItem key={value} value={value}>
                        {TONE_LABELS[value]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="posts">Number of posts</Label>
              <Select
                value={String(numberOfPosts)}
                onValueChange={(v) => setNumberOfPosts(Number(v))}
              >
                <SelectTrigger id="posts" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: MAX_POSTS - MIN_POSTS + 1 }, (_, i) => i + MIN_POSTS).map(
                    (count) => (
                      <SelectItem key={count} value={String(count)}>
                        {count} {count === 1 ? "post" : "posts"}
                      </SelectItem>
                    ),
                  )}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Each post costs additional model calls. Free-tier quotas are small, so start low.
              </p>
            </div>

            {described && (
              <Alert variant="destructive">
                <AlertTitle>{described.title}</AlertTitle>
                <AlertDescription>{described.detail}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" disabled={!canSubmit} size="lg">
              {submitting ? (
                <>
                  <Loader2 className="animate-spin" aria-hidden="true" />
                  Starting…
                </>
              ) : (
                <>
                  <Play aria-hidden="true" />
                  Generate content
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
