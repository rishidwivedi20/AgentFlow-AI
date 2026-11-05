import { ApiError, type ApiErrorBody } from "@/types/api";
import type { HealthResponse } from "@/types/health";
import type { MetricsSummaryResponse, RunHistoryResponse } from "@/types/history";
import type { RunAcceptedResponse, RunStatusResponse } from "@/types/run";
import type { WorkflowRunRequest } from "@/types/workflow";

/**
 * Base URL of the FastAPI backend.
 *
 * Only `VITE_`-prefixed variables reach the browser, and everything that
 * reaches the browser is public. No secret may ever live here - provider API
 * keys stay server-side, which is why all LLM calls route through the backend.
 */
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/** Requests that hang longer than this are aborted rather than left pending. */
const REQUEST_TIMEOUT_MS = 10_000;

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null || !("error" in value)) {
    return false;
  }
  const { error } = value as { error: unknown };
  return typeof error === "object" && error !== null && "type" in error && "message" in error;
}

/**
 * Perform a JSON request against the backend, normalising every failure mode
 * - network, timeout, HTTP error, malformed body - into an `ApiError`.
 */
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch (cause) {
    const timedOut = cause instanceof DOMException && cause.name === "AbortError";
    throw new ApiError(
      0,
      timedOut ? "timeout" : "network_error",
      timedOut
        ? `Request to ${path} timed out after ${REQUEST_TIMEOUT_MS}ms.`
        : `Could not reach the backend at ${API_BASE_URL}. Is it running?`,
    );
  } finally {
    clearTimeout(timeout);
  }

  const payload: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    if (isApiErrorBody(payload)) {
      throw new ApiError(
        response.status,
        payload.error.type,
        payload.error.message,
        payload.error.details,
      );
    }
    throw new ApiError(response.status, "http_error", `Request failed (${response.status}).`);
  }

  return payload as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

/**
 * Queue a workflow run. Returns as soon as the run is registered - the work
 * happens in the background and is followed via the event stream.
 */
export function createRun(body: WorkflowRunRequest): Promise<RunAcceptedResponse> {
  return request<RunAcceptedResponse>("/api/workflows", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * Read a run's current state and, once it has finished, its result.
 *
 * This is how a reloaded page recovers: the event stream is a convenience, not
 * the only way to learn what happened.
 */
export function getRun(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/api/workflows/${encodeURIComponent(runId)}`);
}

/** Finished runs from the backend's history, newest first. */
export function listRuns(limit = 25, offset = 0): Promise<RunHistoryResponse> {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return request<RunHistoryResponse>(`/api/workflows?${query.toString()}`);
}

/** Outcome, cost and per-agent aggregates across every stored run. */
export function getMetricsSummary(): Promise<MetricsSummaryResponse> {
  return request<MetricsSummaryResponse>("/api/metrics/summary");
}

/**
 * URL of a run's SSE stream, for `EventSource`.
 *
 * `EventSource` cannot send headers, so on reconnect the browser echoes the
 * last `id:` it saw as `Last-Event-ID` - which the backend prefers over any
 * cursor in the query string.
 */
export function runEventsUrl(runId: string, cursor?: number): string {
  const base = `${API_BASE_URL}/api/workflows/${encodeURIComponent(runId)}/events`;
  return cursor === undefined ? base : `${base}?cursor=${cursor}`;
}

export { API_BASE_URL, ApiError };
