/**
 * The only place the app talks to the semantic layer.
 *
 * Every call carries the persona and identity, because the server resolves the
 * row-level predicate from them and there is no client-side filtering anywhere
 * in this app. If a number is wrong, it is wrong in one place.
 *
 * The AI calls additionally carry `chartsSay` — the claim keys of every chart
 * currently on screen. The server unions that with what it recomputes for the
 * same page, so a stale client cannot unlock a redundant answer, but sending it
 * means the model knows about charts the user has scrolled to as well.
 */
import type {
  ActionCard,
  AskResponse,
  DealDetail,
  Digest,
  MetaPayload,
  Narrative,
  ViewPayload,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

// ---- The access token -------------------------------------------------------
// The client half of api/auth.py. One shared demo password opens the whole
// platform; the server exchanges it for a bearer token and every request here
// carries that token. It is a door, not an identity: the token says nothing
// about who is calling, so row-level security still comes from the persona
// in the query string, exactly as before. Bearer-only, never a cookie — the
// browser sends nothing on its own, this module attaches it deliberately.

const ACCESS_KEY = "ntt.access";
export const LOCKED_EVENT = "ntt:locked";

declare global {
  interface WindowEventMap {
    [LOCKED_EVENT]: CustomEvent<void>;
  }
}

export interface Access {
  token: string;
  /** ISO-8601, from the server. The token is opaque to this side; the expiry
   *  travels beside it so the gate can show before a doomed first request. */
  expiresAt: string;
}

export function readAccess(): Access | null {
  try {
    const raw = localStorage.getItem(ACCESS_KEY);
    if (!raw) return null;
    const a = JSON.parse(raw) as Partial<Access>;
    if (typeof a.token !== "string" || typeof a.expiresAt !== "string") return null;
    if (!(Date.parse(a.expiresAt) > Date.now())) return null;
    return { token: a.token, expiresAt: a.expiresAt };
  } catch {
    // private window, blocked storage, or a hand-edited value: treat as absent
    return null;
  }
}

function writeAccess(a: Access): void {
  try {
    localStorage.setItem(ACCESS_KEY, JSON.stringify(a));
  } catch {
    // a token that cannot be persisted still opens this session
  }
}

export function clearAccess(): void {
  try {
    localStorage.removeItem(ACCESS_KEY);
  } catch {
    // nothing was stored to begin with
  }
}

function authHeaders(): Record<string, string> {
  const a = readAccess();
  return a ? { authorization: `Bearer ${a.token}` } : {};
}

/** A 401 from anywhere means the token is gone, expired, or the password was
 *  rotated underneath it. Forget it and tell the gate, which unmounts the app
 *  so no in-flight page keeps retrying against a closed door. */
function locked(): void {
  clearAccess();
  window.dispatchEvent(new CustomEvent(LOCKED_EVENT));
}

export interface Ctx {
  persona: string;
  identity: string;
  filters: Record<string, string | null>;
  measure: string;
}

function qs(ctx: Ctx, extra: Record<string, string | undefined> = {}): string {
  const p = new URLSearchParams();
  p.set("persona", ctx.persona);
  if (ctx.identity) p.set("identity", ctx.identity);
  p.set("measure", ctx.measure);
  for (const [k, v] of Object.entries(ctx.filters)) if (v) p.set(k, v);
  for (const [k, v] of Object.entries(extra)) if (v !== undefined) p.set(k, v);
  return p.toString();
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { signal, headers: authHeaders() });
  if (!r.ok) {
    if (r.status === 401) locked();
    let detail = r.statusText;
    try {
      detail = (await r.json())?.detail ?? detail;
    } catch {
      /* the body was not JSON; the status text is the best we have */
    }
    throw new ApiError(detail, r.status);
  }
  return (await r.json()) as T;
}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    if (r.status === 401) locked();
    throw new ApiError(r.statusText, r.status);
  }
  return (await r.json()) as T;
}

export const auth = {
  /** Exchange the password for a token and keep it. This does not go through
   *  `post`, because a wrong password is also a 401 and must not be read as
   *  "the door just closed" — the gate is already up when this runs. */
  login: async (password: string, signal?: AbortSignal): Promise<Access> => {
    const r = await fetch(`${BASE}/api/auth/login`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ password }),
      signal,
    });
    if (!r.ok) throw new ApiError(r.status === 401 ? "wrong password" : r.statusText, r.status);
    const a = (await r.json()) as Access;
    writeAccess(a);
    return a;
  },
};

export const api = {
  meta: (ctx: Ctx, signal?: AbortSignal) =>
    get<MetaPayload>(`/api/meta?${qs(ctx)}`, signal),

  view: (ctx: Ctx, page: string, signal?: AbortSignal) =>
    get<ViewPayload>(`/api/view?${qs(ctx, { page })}`, signal),

  actions: (ctx: Ctx, limit = 12, signal?: AbortSignal) =>
    get<{ actions: ActionCard[] }>(
      `/api/actions?${qs(ctx, { limit: String(limit) })}`, signal),

  deal: (ctx: Ctx, code: string, signal?: AbortSignal) =>
    get<DealDetail>(`/api/deal/${encodeURIComponent(code)}?${qs(ctx)}`, signal),

  anomalies: (ctx: Ctx, limit = 80, signal?: AbortSignal) =>
    get<Record<string, unknown>>(
      `/api/anomalies?${qs(ctx, { limit: String(limit) })}`, signal),

  accounts: (ctx: Ctx, signal?: AbortSignal) =>
    get<Record<string, unknown>>(`/api/accounts?${qs(ctx)}`, signal),

  budget: (ctx: Ctx, signal?: AbortSignal) =>
    get<Record<string, unknown>>(`/api/budget?${qs(ctx)}`, signal),

  // ---- AI surfaces. Each carries what is already on screen. ---------------
  brief: (ctx: Ctx, page: string, chartsSay: string[], signal?: AbortSignal) =>
    get<Narrative>(
      `/api/ai/brief?${qs(ctx, { page, chartsSay: JSON.stringify(chartsSay) })}`,
      signal),

  /** `chart` is the id of the chart the question is about, when it is about
   *  one; the server then answers from that chart's rows rather than the
   *  whole slice, and the reply is words only. */
  ask: (ctx: Ctx, q: string, chartsSay: string[], signal?: AbortSignal,
        chart?: { id: string; page: string }) =>
    get<AskResponse>(
      `/api/ai/ask?${qs(ctx, { q, chartsSay: JSON.stringify(chartsSay),
                               chart: chart?.id, page: chart?.page })}`, signal),

  explain: (ctx: Ctx, card: ActionCard, chartsSay: string[], signal?: AbortSignal) =>
    post<Narrative & { evidence?: unknown[] }>(
      `/api/ai/explain?${qs(ctx, { chartsSay: JSON.stringify(chartsSay) })}`,
      card, signal),

  nextAction: (ctx: Ctx, code: string, signal?: AbortSignal) =>
    get<Narrative & { deal: DealDetail; evidence: unknown[] }>(
      `/api/ai/next-action/${encodeURIComponent(code)}?${qs(ctx)}`, signal),

  digest: (ctx: Ctx, days = 7, signal?: AbortSignal) =>
    get<Digest>(`/api/ai/digest?${qs(ctx, { days: String(days) })}`, signal),

  health: (signal?: AbortSignal) =>
    get<Record<string, unknown>>(`/api/health`, signal),
};
