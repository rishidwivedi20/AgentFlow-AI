/** Error envelope returned by every backend failure. Mirrors `core/exceptions.py`. */
export interface ApiErrorBody {
  error: {
    type: string;
    message: string;
    details: Record<string, unknown>;
  };
}

/** Thrown by the API client so callers have one error type to handle. */
export class ApiError extends Error {
  readonly status: number;
  readonly type: string;
  readonly details: Record<string, unknown>;

  constructor(status: number, type: string, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
    this.details = details;
  }
}
