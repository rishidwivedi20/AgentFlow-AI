/** Shared number formatting. Every helper renders "—" for a missing value. */

const DASH = "—";

/**
 * Dollars with enough precision to be meaningful. A single run usually costs
 * fractions of a cent, which two decimal places would round to $0.00 - a
 * figure that reads as "free" rather than "small".
 */
export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined) return DASH;
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return DASH;
  return `${Math.round(value * 100)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined) return DASH;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return DASH;
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${Math.round(value - minutes * 60)}s`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return DASH;
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
