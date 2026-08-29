import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getRun, runEventsUrl } from "@/services/api";
import { isTerminalEvent } from "@/types/events";
import type { EventType, WorkflowEvent } from "@/types/events";
import { isRunFinished } from "@/types/run";
import type { RunStatusResponse } from "@/types/run";

export type StreamState = "connecting" | "streaming" | "finished" | "error";

/**
 * Every event name the server emits.
 *
 * Frames carry an `event:` line, so the browser dispatches them as *named*
 * events. `EventSource.onmessage` only fires for unnamed ones, so each type
 * needs its own listener - a silent no-op otherwise.
 */
const EVENT_TYPES: EventType[] = [
  "workflow_started",
  "agent_started",
  "agent_completed",
  "agent_failed",
  "workflow_completed",
  "workflow_failed",
];

interface UseWorkflowStreamResult {
  events: WorkflowEvent[];
  run: RunStatusResponse | null;
  streamState: StreamState;
  error: string | null;
  reload: () => void;
}

/**
 * Follow a run: its buffered history, then its live events, then its result.
 *
 * Three properties this has to get right:
 *
 * **Replay is idempotent.** The server resends everything after the client's
 * cursor, and `EventSource` reconnects on its own, so the same event can
 * arrive twice. Events are keyed by `sequence` and deduplicated here rather
 * than assumed unique.
 *
 * **The stream must be closed deliberately.** When the server finishes a run
 * its generator returns and the connection closes - which `EventSource` treats
 * as a dropped connection and retries forever. Seeing a terminal event is what
 * tells us to close first.
 *
 * **The result comes from the status endpoint, not the stream.** Events carry
 * execution status only; the posts, sources and verdicts are fetched once the
 * run has finished.
 */
export function useWorkflowStream(runId: string): UseWorkflowStreamResult {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [run, setRun] = useState<RunStatusResponse | null>(null);
  const [streamState, setStreamState] = useState<StreamState>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const sourceRef = useRef<EventSource | null>(null);

  const reload = useCallback(() => setReloadToken((n) => n + 1), []);

  useEffect(() => {
    if (!runId) return;

    let cancelled = false;

    const closeStream = (): void => {
      sourceRef.current?.close();
      sourceRef.current = null;
    };

    /** Fetch the run record, which carries the result once it exists. */
    const loadRun = async (): Promise<RunStatusResponse | null> => {
      try {
        const record = await getRun(runId);
        if (!cancelled) {
          setRun(record);
          setError(null);
        }
        return record;
      } catch (cause) {
        if (!cancelled) {
          setError(
            cause instanceof ApiError ? cause.message : "Could not load this run.",
          );
          setStreamState("error");
        }
        return null;
      }
    };

    const openStream = (): void => {
      const source = new EventSource(runEventsUrl(runId));
      sourceRef.current = source;

      source.onopen = () => {
        if (!cancelled) setStreamState("streaming");
      };

      const handleFrame = (message: MessageEvent<string>): void => {
        if (cancelled) return;
        let event: WorkflowEvent;
        try {
          event = JSON.parse(message.data) as WorkflowEvent;
        } catch {
          // A malformed frame is not worth tearing the stream down for.
          return;
        }

        setEvents((current) => {
          if (current.some((existing) => existing.sequence === event.sequence)) {
            return current; // replayed after a reconnect
          }
          return [...current, event].sort((a, b) => a.sequence - b.sequence);
        });

        if (isTerminalEvent(event)) {
          // Close before the server does, so EventSource does not treat the
          // orderly end of the run as a dropped connection and reconnect.
          closeStream();
          setStreamState("finished");
          void loadRun();
        }
      };

      for (const type of EVENT_TYPES) {
        source.addEventListener(type, handleFrame as EventListener);
      }
      // Covers a server that ever emits an unnamed frame.
      source.onmessage = handleFrame;

      source.onerror = () => {
        if (cancelled) return;
        // EventSource retries on its own, echoing Last-Event-ID so the server
        // resumes rather than replaying. Only a closed socket is terminal.
        if (source.readyState === EventSource.CLOSED) {
          setStreamState("error");
          setError("Lost connection to the run. Reload to reconnect.");
        }
      };
    };

    void (async () => {
      setStreamState("connecting");
      setEvents([]);
      const record = await loadRun();
      if (cancelled || record === null) return;

      if (isRunFinished(record.state)) {
        // Finished runs still stream: the server replays the whole buffer,
        // which is what makes a reloaded page show the full history.
        setStreamState("finished");
      }
      openStream();
    })();

    return () => {
      cancelled = true;
      closeStream();
    };
  }, [runId, reloadToken]);

  return { events, run, streamState, error, reload };
}
