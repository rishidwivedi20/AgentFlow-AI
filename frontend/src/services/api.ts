import { ApiError, type ApiErrorBody } from "@/types/api";
import type { HealthResponse } from "@/types/health";

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

export { API_BASE_URL, ApiError };
