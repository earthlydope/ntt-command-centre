/**
 * All app state, and its mirror in the URL.
 *
 * Nothing in this app ever opens a window. Persona, identity, page, every
 * filter, the measure toggle, the theme and the Ask overlay all live in the
 * query string and are written with `pushState`, so any view is reproducible
 * from its URL — which is also how a link in an email lands someone on exactly
 * the screen it is about.
 *
 * Identity is state, not a filter, and the distinction matters: a filter
 * narrows what you look at, an identity changes what you are ALLOWED to look
 * at. The server resolves the row-level predicate from the identity; putting it
 * in the same bucket as `lob=Security` would make widening your own scope a
 * query-string edit.
 */
import type { AskResponse, DimKey, Lens, Measure, PersonaKey } from "../api/types";

/**
 * A conversation carried from a chart's own Ask into the main Ask panel.
 * Held in state, never in the URL: it is a handoff, not a view.
 */
export interface AskSeedTurn {
  question: string;
  response: AskResponse | null;
  /** The chart the question was asked about, so the transcript says so. */
  chartTitle?: string;
}
export type AskSeed = AskSeedTurn[];

export const DIM_KEYS: DimKey[] = [
  "stage", "forecast", "lob", "portfolio", "industry",
  "orderType", "quarter", "country", "rep", "account",
  "riskBand", "anomalyCategory",
];

export type FilterMap = Record<DimKey, string | null>;

export const EMPTY_FILTERS: FilterMap = DIM_KEYS.reduce((acc, k) => {
  acc[k] = null;
  return acc;
}, {} as FilterMap);

export const PERSONA_HOME: Record<PersonaKey, Lens> = {
  ae: "my-day",
  manager: "pod-pulse",
  executive: "tldr",
};

export interface AppState {
  persona: PersonaKey;
  identity: string;
  page: Lens;
  filters: FilterMap;
  measure: Measure;
  /** The Ask panel, and what it was opened about. */
  ask: boolean;
  askQuery: string;
  /** Turns handed over from a chart's Ask when the user expands into the main chat. */
  askSeed: AskSeed | null;
  /** An open drawer: "deal:<code>" | "account:<code>" | "card:<key>". */
  drawer: string | null;
}

export const INITIAL: AppState = {
  persona: "executive",
  identity: "",
  page: "tldr",
  filters: { ...EMPTY_FILTERS },
  measure: "gp",
  ask: false,
  askQuery: "",
  askSeed: null,
  drawer: null,
};

export type Action =
  | { type: "persona"; persona: PersonaKey; identity?: string }
  | { type: "identity"; identity: string }
  | { type: "page"; page: Lens }
  | { type: "toggleFilter"; dim: DimKey; value: string }
  | { type: "setFilter"; dim: DimKey; value: string | null }
  | { type: "clearFilters" }
  | { type: "measure"; measure: Measure }
  | { type: "ask"; open: boolean; query?: string; seed?: AskSeed | null }
  | { type: "drawer"; drawer: string | null }
  | { type: "fromUrl"; state: Partial<AppState> };

export function reducer(s: AppState, a: Action): AppState {
  switch (a.type) {
    case "persona": {
      if (a.persona === s.persona && !a.identity) return s;
      // A persona change moves to that persona's OWN home, never staying on the
      // current page: the current page usually does not exist for the new
      // persona, and landing on a redirect is a worse first impression than
      // landing where you meant to go. Filters are cleared for the same reason
      // the query cache is — a filter value legal in one scope may not exist in
      // the next, and a silently-empty page reads as "there is none of this"
      // rather than "not yours".
      return {
        ...s,
        persona: a.persona,
        identity: a.identity ?? "",
        page: PERSONA_HOME[a.persona],
        filters: { ...EMPTY_FILTERS },
        drawer: null,
      };
    }
    case "identity":
      return { ...s, identity: a.identity, filters: { ...EMPTY_FILTERS }, drawer: null };
    case "page":
      return { ...s, page: a.page, drawer: null };
    case "toggleFilter": {
      // Clicking an active value clears it, like a pivot-table slicer.
      const cur = s.filters[a.dim];
      return {
        ...s,
        filters: { ...s.filters, [a.dim]: cur === a.value ? null : a.value },
      };
    }
    case "setFilter":
      return { ...s, filters: { ...s.filters, [a.dim]: a.value } };
    case "clearFilters":
      return { ...s, filters: { ...EMPTY_FILTERS } };
    case "measure":
      return { ...s, measure: a.measure };
    case "ask":
      // Opening with a seed replaces the previous seed; opening without one
      // starts clean. Closing drops it, so a stale handoff cannot reappear.
      return {
        ...s,
        ask: a.open,
        askQuery: a.query ?? (a.open ? s.askQuery : ""),
        askSeed: a.open ? (a.seed ?? null) : null,
      };
    case "drawer":
      return { ...s, drawer: a.drawer };
    case "fromUrl":
      return { ...s, ...a.state, filters: { ...EMPTY_FILTERS, ...(a.state.filters ?? {}) } };
    default:
      return s;
  }
}

export function toQuery(s: AppState): string {
  const p = new URLSearchParams();
  p.set("as", s.persona);
  if (s.identity) p.set("id", s.identity);
  p.set("page", s.page);
  if (s.measure !== "gp") p.set("measure", s.measure);
  for (const k of DIM_KEYS) if (s.filters[k]) p.set(k, s.filters[k] as string);
  if (s.ask) p.set("ask", "1");
  if (s.drawer) p.set("drawer", s.drawer);
  return p.toString();
}

export function fromQuery(search: string): Partial<AppState> {
  const p = new URLSearchParams(search);
  const persona = p.get("as");
  const page = p.get("page");
  const filters: Partial<FilterMap> = {};
  for (const k of DIM_KEYS) {
    const v = p.get(k);
    if (v) filters[k] = v;
  }
  const out: Partial<AppState> = { filters: filters as FilterMap };
  if (persona === "ae" || persona === "manager" || persona === "executive") {
    out.persona = persona;
  }
  if (page) out.page = page as Lens;
  const id = p.get("id");
  if (id) out.identity = id;
  const m = p.get("measure");
  if (m === "revenue" || m === "gp") out.measure = m;
  if (p.get("ask") === "1") out.ask = true;
  const d = p.get("drawer");
  if (d) out.drawer = d;
  return out;
}

export function activeFilters(s: AppState): { dim: DimKey; value: string }[] {
  return DIM_KEYS.filter((k) => s.filters[k]).map((k) => ({
    dim: k,
    value: s.filters[k] as string,
  }));
}
