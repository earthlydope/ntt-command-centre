/**
 * The shell: header, sidebar, filter deck, page, overlays. Nothing else.
 *
 * Navigation is built from `meta.pages`, which the server derives from the
 * persona — so the sidebar is genuinely different for each of the three
 * profiles rather than the same list with some items disabled. A persona cannot
 * even see the name of a page that belongs to someone else's job.
 *
 * Three things were removed from this file rather than restyled, because they
 * were repetition and no amount of styling fixes that: the scope strip (which
 * said what the persona control already says), the second Ask button (the
 * header has one), and the footer's provenance paragraph (the date was on
 * screen three times).
 */
import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FilterBar } from "./components/FilterBar";
import { Header } from "./components/Header";
import { SidebarNav } from "./components/SidebarNav";
import { SUGGESTIONS } from "./components/askSuggestions";
import { Glyph } from "./components/askGlyphs";
import { PageView } from "./lenses/PageView";
import { useMeta, useView } from "./lenses/useView";
import { useApp } from "./state/AppStateProvider";
import type { DimKey } from "./api/types";

const AskPanel = lazy(() =>
  import("./components/AskPanel").then((m) => ({ default: m.AskPanel })),
);
const DealDrawer = lazy(() =>
  import("./components/DealDrawer").then((m) => ({ default: m.DealDrawer })),
);

export default function App() {
  const { state, setPage, dispatch, clearFilters, setMeasure, openAsk, closeAsk, closeDrawer } =
    useApp();
  const meta = useMeta();
  const view = useView();
  const [railOpen, setRailOpen] = useState(true);

  // The floating button is unmounted while the panel it opens is up, so the
  // panel cannot capture it as the active element on arrival the way it does
  // the header's button. It is named here instead: by the time the panel's
  // teardown runs the button is mounted again and takes focus back. The flag
  // is cleared on close so a later open from the header returns focus there.
  const fabRef = useRef<HTMLButtonElement | null>(null);
  const openedByFab = useRef(false);
  const closeAskPanel = useCallback(() => {
    openedByFab.current = false;
    closeAsk();
  }, [closeAsk]);

  const pages = meta.data?.pages ?? [];
  const chartsSay = useMemo(() => view.data?.chartsSay ?? [], [view.data]);

  // Narrow screens start with the rail closed; it is a drawer there, not a column.
  useEffect(() => {
    const narrow = window.matchMedia("(max-width: 1100px)");
    const sync = () => setRailOpen(!narrow.matches);
    sync();
    narrow.addEventListener("change", sync);
    return () => narrow.removeEventListener("change", sync);
  }, []);


  if (meta.error) {
    return (
      <div className="app">
        <Header meta={null} />
        <main className="main">
          <div className="panel-error" role="alert">
            <h2>The semantic layer did not answer</h2>
            <p>{meta.error}</p>
            <p className="muted">
              Nothing on this page is mocked, so rather than show numbers that look real,
              it shows nothing. Start the API and reload.
            </p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className={`app${railOpen ? "" : " app--rail-closed"}`}>
      <Header meta={meta.data} onToggleRail={() => setRailOpen((v) => !v)} />

      <div className="shell">
        <aside className={`rail${railOpen ? "" : " rail--closed"}`}>
          {meta.data && (
            <SidebarNav
              persona={meta.data.persona.active.persona}
              pages={pages}
              current={state.page}
              onNavigate={(p) => setPage(p)}
            />
          )}
        </aside>

        <div className="body">
          <FilterBar
            meta={meta.data}
            persona={state.persona}
            active={state.filters}
            measure={state.measure}
            onSet={(dim: DimKey, value) => dispatch({ type: "setFilter", dim, value })}
            onClear={clearFilters}
            onMeasure={setMeasure}
          />
          <main className="main">
            <PageView />
          </main>
        </div>
      </div>

      {state.ask && (
        <Suspense fallback={null}>
          <AskPanel
            onClose={closeAskPanel}
            triggerRef={openedByFab.current ? fabRef : undefined}
            chartsSay={chartsSay}
            scopeLabel={view.data?.scope.label ?? ""}
            suggestions={SUGGESTIONS[state.persona] ?? SUGGESTIONS.executive}
            initialQuery={state.askQuery}
            seed={state.askSeed}
            scope={view.data?.scope ?? null}
            asOf={view.data?.asOf}
          />
        </Suspense>
      )}

      {state.drawer?.startsWith("deal:") && (
        <Suspense fallback={null}>
          <DealDrawer code={state.drawer.slice(5)} onClose={closeDrawer} />
        </Suspense>
      )}

      {/* The AI expert is reachable from every page, from the same corner,
          whatever is scrolled into view. It hides while the panel it opens is
          up, so there is never a button under the thing it summoned. The mark
          on it is the same spark the Ask surfaces use, so the button and the
          panel it opens read as one thing. */}
      {!state.ask && (
        <button
          type="button"
          className="fab"
          ref={fabRef}
          onClick={() => {
            openedByFab.current = true;
            openAsk();
          }}
          aria-label="Ask AI Expert"
          aria-haspopup="dialog"
        >
          <Glyph kind="spark" className="fab__glyph" size={18} />
          <span className="fab__label">Ask AI Expert</span>
        </button>
      )}

      <footer className="foot">
        <span>
          {meta.data?.data.lines.toLocaleString() ?? "—"} deal lines ·{" "}
          {meta.data?.data.movementRows.toLocaleString() ?? "—"} recorded changes ·{" "}
          {meta.data?.data.anomalies.toLocaleString() ?? "—"} findings
        </span>
      </footer>
    </div>
  );
}
