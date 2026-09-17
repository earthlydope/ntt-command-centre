/**
 * Number formatting. One implementation, used by the shell, the charts and the
 * mock semantic layer, so a figure reads identically in a KPI tile, an axis and
 * a tooltip. Nothing formats inline.
 */

export function money(v: number, dp = 2): string {
  const a = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (a >= 1e9) return `${sign}$${(a / 1e9).toFixed(dp)}bn`;
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(dp)}M`;
  if (a >= 1e3) return `${sign}$${(a / 1e3).toFixed(0)}k`;
  return `${sign}$${a.toFixed(0)}`;
}

/** Axis ticks want one decimal and no noise. */
export const moneyShort = (v: number) => money(v, v >= 1e6 || v <= -1e6 ? 1 : 0);

export const pct = (v: number, dp = 1) => `${v.toFixed(dp)}%`;

export const num = (v: number) => v.toLocaleString("en-US");

export const multiple = (v: number) => `${v.toFixed(1)}x`;

export type Format = "currency" | "percent" | "number" | "days";

/** A count of days reads as "146d" — a bare integer beside a dollar axis is
 *  ambiguous, and "146 days" is too wide for a tick. */
export const days = (v: number) => `${Math.round(v)}d`;

export function formatValue(v: number, kind?: Format): string {
  switch (kind) {
    case "percent":
      return pct(v);
    case "number":
      return num(Math.round(v));
    case "days":
      return days(v);
    default:
      return money(v);
  }
}

export function formatTick(v: number, kind?: Format): string {
  switch (kind) {
    case "percent":
      return `${Math.round(v)}%`;
    case "number":
      return num(Math.round(v));
    case "days":
      return days(v);
    default:
      return moneyShort(v);
  }
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function monthLabel(iso: string): string {
  const d = new Date(iso);
  return `${MONTHS[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}`;
}

export function longDate(iso: string): string {
  const d = new Date(iso);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}
