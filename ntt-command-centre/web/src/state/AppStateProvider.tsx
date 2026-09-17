/**
 * One reducer, mirrored into the URL, provided to the tree.
 *
 * The URL write is `pushState` on a real state change and `replaceState` on the
 * initial hydrate, so the back button walks the user's actual decisions rather
 * than every keystroke in the Ask box.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";
import type { Ctx } from "../api/client";
import type { DimKey, Lens, Measure, PersonaKey } from "../api/types";
import {
  INITIAL,
  PERSONA_HOME,
  type Action,
  type AppState,
  type AskSeed,
  fromQuery,
  reducer,
  toQuery,
} from "./filters";

interface Api {
  state: AppState;
  dispatch: (a: Action) => void;
  /** The context every API call carries. Memoised so effects do not re-fire. */
  ctx: Ctx;
  setPersona: (p: PersonaKey, identity?: string) => void;
  setIdentity: (id: string) => void;
  setPage: (p: Lens) => void;
  onFilter: (dim: DimKey, value: string) => void;
  clearFilters: () => void;
  setMeasure: (m: Measure) => void;
  /** Open the main Ask panel, optionally about a question and with turns carried over from a chart's Ask. */
  openAsk: (query?: string, seed?: AskSeed | null) => void;
  closeAsk: () => void;
  openDrawer: (d: string) => void;
  closeDrawer: () => void;
}

const C = createContext<Api | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL, (init) => {
    const fromUrl = fromQuery(window.location.search);
    const merged = { ...init, ...fromUrl };
    // A URL naming a page that belongs to another persona lands on this
    // persona's home rather than on an error.
    if (fromUrl.persona && !fromUrl.page) merged.page = PERSONA_HOME[fromUrl.persona];
    return { ...merged, filters: { ...init.filters, ...(fromUrl.filters ?? {}) } };
  });

  const first = useRef(true);
  useEffect(() => {
    const q = toQuery(state);
    const url = `${window.location.pathname}?${q}`;
    if (first.current) {
      first.current = false;
      window.history.replaceState(null, "", url);
      return;
    }
    if (`?${q}` !== window.location.search) window.history.pushState(null, "", url);
  }, [state]);

  useEffect(() => {
    const onPop = () => dispatch({ type: "fromUrl", state: fromQuery(window.location.search) });
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // The persona accent is a document-level attribute so CSS can resolve
  // --persona once rather than every component threading a colour.
  useEffect(() => {
    document.documentElement.setAttribute("data-persona", state.persona);
  }, [state.persona]);

  const ctx: Ctx = useMemo(
    () => ({
      persona: state.persona,
      identity: state.identity,
      filters: state.filters as Record<string, string | null>,
      measure: state.measure,
    }),
    [state.persona, state.identity, state.filters, state.measure],
  );

  const value: Api = useMemo(
    () => ({
      state,
      dispatch,
      ctx,
      setPersona: (p, identity) => dispatch({ type: "persona", persona: p, identity }),
      setIdentity: (identity) => dispatch({ type: "identity", identity }),
      setPage: (page) => dispatch({ type: "page", page }),
      onFilter: (dim, value) => dispatch({ type: "toggleFilter", dim, value }),
      clearFilters: () => dispatch({ type: "clearFilters" }),
      setMeasure: (measure) => dispatch({ type: "measure", measure }),
      openAsk: (query, seed) => dispatch({ type: "ask", open: true, query, seed }),
      closeAsk: () => dispatch({ type: "ask", open: false }),
      openDrawer: (d) => dispatch({ type: "drawer", drawer: d }),
      closeDrawer: () => dispatch({ type: "drawer", drawer: null }),
    }),
    [state, ctx],
  );

  return <C.Provider value={value}>{children}</C.Provider>;
}

export function useApp(): Api {
  const v = useContext(C);
  if (!v) throw new Error("useApp must be used inside AppStateProvider");
  return v;
}

/** A stable callback that scopes the page from a chart mark or an action card. */
export function useScopeTo() {
  const { onFilter, setPage } = useApp();
  return useCallback(
    (scope: { dim: DimKey; value: string } | null | undefined, page?: Lens) => {
      if (scope) onFilter(scope.dim, scope.value);
      if (page) setPage(page);
    },
    [onFilter, setPage],
  );
}
