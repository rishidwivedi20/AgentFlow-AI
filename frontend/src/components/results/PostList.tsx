import { useState } from "react";
import { Check, Copy } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { Post } from "@/types/workflow";

/**
 * Copy a post's text, hashtags included, in the form it would be published.
 *
 * `navigator.clipboard` needs a secure context, so it is absent over plain HTTP
 * on anything but localhost. The button hides itself rather than failing when
 * clicked.
 */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  if (typeof navigator === "undefined" || !navigator.clipboard) return null;

  const handleCopy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Denied permission is not worth an error state for a convenience.
    }
  };

  return (
    <Button variant="ghost" size="sm" onClick={() => void handleCopy()}>
      {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

function postAsText(post: Post): string {
  const tags = post.hashtags.map((tag) => `#${tag}`).join(" ");
  return tags ? `${post.content}\n\n${tags}` : post.content;
}

export function PostList({ posts }: { posts: Post[] }) {
  if (posts.length === 0) {
    return (
      <p className="py-10 text-center text-sm text-muted-foreground">
        This run produced no posts.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {posts.map((post, index) => (
        <Card key={`${post.angle}-${index}`}>
          <CardContent className="pt-0">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-xs font-medium text-muted-foreground">Post {index + 1}</p>
                <p className="mt-0.5 text-sm font-medium">{post.angle}</p>
              </div>
              <CopyButton text={postAsText(post)} />
            </div>

            {/* whitespace-pre-wrap: the paragraph breaks are the post's own
                formatting, and collapsing them would misrepresent the output. */}
            <p className="mt-3 text-sm leading-relaxed whitespace-pre-wrap">{post.content}</p>

            {post.hashtags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {post.hashtags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="font-normal">
                    #{tag}
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
