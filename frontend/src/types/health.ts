/**
 * Mirrors `backend/app/schemas/health.py`.
 *
 * These types are maintained by hand. When a backend schema changes, update
 * this file in the same commit - there is no codegen step by design.
 */

export type HealthStatus = "ok" | "degraded";

export interface HealthChecks {
  config_loaded: boolean;
  api_key_present: boolean;
}

export interface HealthResponse {
  status: HealthStatus;
  version: string;
  environment: string;
  llm_provider: string;
  llm_model: string;
  llm_configured: boolean;
  checks: HealthChecks;
}
