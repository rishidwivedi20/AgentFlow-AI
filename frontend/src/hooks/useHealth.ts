import { useCallback, useEffect, useState } from "react";

import { getHealth } from "@/services/api";
import { ApiError } from "@/types/api";
import type { HealthResponse } from "@/types/health";

export type ConnectionState = "checking" | "connected" | "unreachable";

interface UseHealthResult {
  state: ConnectionState;
  health: HealthResponse | null;
  error: string | null;
  refresh: () => void;
}

/**
 * Fetch backend health on mount, with a manual refresh.
 *
 * This is the first end-to-end integration check of the stack: it proves the
 * dev server, CORS configuration and API client all agree before any agent
 * exists.
 */
export function useHealth(): UseHealthResult {
  const [state, setState] = useState<ConnectionState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Resolves state only after the request settles, so the effect below never
  // sets state synchronously during render.
  const load = useCallback(async (): Promise<void> => {
    try {
      setHealth(await getHealth());
      setState("connected");
      setError(null);
    } catch (cause) {
      setHealth(null);
      setState("unreachable");
      setError(cause instanceof ApiError ? cause.message : "Unknown error contacting backend.");
    }
  }, []);

  useEffect(() => {
    // Fetching on mount is the rule's own documented exception - synchronising
    // with an external system. The lint cannot see that `load` only sets state
    // after an await, so suppress it here rather than project-wide.
    // oxlint-disable-next-line react/set-state-in-effect
    void load();
  }, [load]);

  const refresh = useCallback((): void => {
    setState("checking");
    setError(null);
    void load();
  }, [load]);

  return { state, health, error, refresh };
}
