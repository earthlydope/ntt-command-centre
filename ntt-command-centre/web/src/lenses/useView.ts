/**
 * Fetch hooks.
 *
 * Two behaviours worth naming:
 *
 * **The page renders before the AI does.** `useView` returns the computed
 * payload — every KPI, chart and action card — and `useBrief` fetches the
 * generated narrative separately. The page is therefore complete and correct in
 * one round trip, and the prose upgrades in place when it arrives. A product
 * that waits on a language model to paint its first pixel is a product that
 * looks broken whenever the model is slow, which on a free tier is often.
 *
 * **Every fetch is abortable and keyed.** Switching persona while a request is
 * in flight must not let the old persona's payload land in the new persona's
 * page; the abort plus the key check makes that impossible rather than unlikely.
 *
 * **A key change drops the previous payload, it does not keep it under a
 * spinner.** `useView` used to hold the old page's data while the new one
 * loaded, so clicking a sidebar item moved the highlight and left the previous
 * page's question, tiles and charts painted until the response came back —
 * on a cold Cloud Run instance that is seconds, and it is exactly what the
 * customer reported as "the right side does not change". Worse, `useBrief`
 * saw the stale `chartsSay` with the NEW page key and asked the model for a
 * brief with the wrong forbidden list; the correct request followed a moment
 * later, so every navigation cost two generations on a per-minute allowance
 * and the wrong one was what got cached. Nulling the data on a key change is
 * what makes the skeleton show, and it is what stops the brief firing before
 * the page it must not repeat has arrived.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type Ctx } from "../api/client";
import type { MetaPayload, Narrative, ViewPayload } from "../api/types";
import { useApp } from "../state/AppStateProvider";

export interface Async<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

function key(ctx: Ctx, extra = ""): string {
  return JSON.stringify([ctx.persona, ctx.identity, ctx.measure, ctx.filters, extra]);
}

/**
 * In-flight de-duplication.
 *
 * The shell and the page both need the view payload — the shell for the scope
 * label and for the claim keys it hands the Ask panel, the page for everything
 * it draws. Without this they would issue the same request twice on every
 * navigation. Keyed on the request itself, so two callers asking the same
 * question share one answer and a third asking a different one is unaffected.
 *
 * Entries are dropped as soon as the promise settles: this is a request
 * coalescer, not a data cache, and holding results here would mean a stale
 * payload after the user changes something the key does not capture.
 */
const INFLIGHT = new Map<string, Promise<unknown>>();

function share<T>(k: string, run: () => Promise<T>): Promise<T> {
  const hit = INFLIGHT.get(k) as Promise<T> | undefined;
  if (hit) return hit;
  const p = run().finally(() => INFLIGHT.delete(k));
  INFLIGHT.set(k, p);
  return p;
}

export function useMeta(): Async<MetaPayload> {
  const { ctx } = useApp();
  const [s, set] = useState<Async<MetaPayload>>({ data: null, error: null, loading: true });
  const k = useMemo(() => `${ctx.persona}|${ctx.identity}`, [ctx.persona, ctx.identity]);

  useEffect(() => {
    const ac = new AbortController();
    let live = true;
    set((p) => ({ ...p, loading: true, error: null }));
    api
      .meta(ctx, ac.signal)
      .then((data) => live && set({ data, error: null, loading: false }))
      .catch((e) => {
        if (ac.signal.aborted || !live) return;
        set({ data: null, error: describe(e), loading: false });
      });
    return () => {
      live = false;
      ac.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k]);

  return s;
}

export function useView(): Async<ViewPayload> {
  const { ctx, state } = useApp();
  const [s, set] = useState<Async<ViewPayload>>({ data: null, error: null, loading: true });
  const k = useMemo(() => key(ctx, state.page), [ctx, state.page]);
  const last = useRef("");

  useEffect(() => {
    last.current = k;
    // The previous payload is dropped, not carried: it answers a different
    // question (or a different scope) and a figure from it is wrong on this
    // one. The page shows its skeleton for the round trip instead.
    set({ data: null, error: null, loading: true });
    // Not passing the AbortSignal: the promise is shared with the other caller,
    // so cancelling it here would cancel theirs too. The key check below is what
    // prevents a late response from the previous scope painting.
    share(k, () => api.view(ctx, state.page))
      .then((data) => last.current === k && set({ data, error: null, loading: false }))
      .catch((e) => {
        if (last.current !== k) return;
        set({ data: null, error: describe(e), loading: false });
      });
    return () => {
      last.current = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k]);

  return s;
}

/**
 * The generated narrative, fetched after the page is already on screen.
 *
 * `chartsSay` is what the charts on THIS page claim; the server unions it with
 * what it recomputes and forbids the model from restating any of it.
 */
export function useBrief(chartsSay: string[], ready: boolean): Async<Narrative> {
  const { ctx, state } = useApp();
  const [s, set] = useState<Async<Narrative>>({ data: null, error: null, loading: false });
  const k = useMemo(
    () => key(ctx, `${state.page}|${chartsSay.join(",")}`),
    [ctx, state.page, chartsSay],
  );
  const last = useRef("");

  useEffect(() => {
    // Dropped rather than carried for the same reason as the view: the page
    // falls back to the computed narrative the payload already carries, which
    // is about THIS page, rather than showing the previous page's generated
    // prose under a spinner for as long as the model takes. The not-ready
    // branch clears too, so the old brief is gone before the new page's first
    // frame rather than one frame after it.
    if (!ready) {
      set({ data: null, error: null, loading: false });
      return;
    }
    last.current = k;
    set({ data: null, error: null, loading: true });
    share(`brief:${k}`, () => api.brief(ctx, state.page, chartsSay))
      .then((data) => last.current === k && set({ data, error: null, loading: false }))
      .catch((e) => {
        if (last.current !== k) return;
        set({ data: null, error: describe(e), loading: false });
      });
    return () => {
      last.current = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k, ready]);

  return s;
}

export function describe(e: unknown): string {
  if (e instanceof ApiError) return `${e.message} (${e.status})`;
  if (e instanceof Error) return e.message;
  return String(e);
}
