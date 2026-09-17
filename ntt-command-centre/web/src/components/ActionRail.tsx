/**
 * ActionRail — the band of AI-surfaced actions ("Do these next").
 *
 * This is the component the product is for. A chart tells you the shape of the
 * quarter; this tells you what to do in the next two hours, and — crucially —
 * it tells you WHY it is allowed to say it.
 *
 * The shape of the thing: a header strip that lets the reader narrow the feed
 * (All · Critical · High · Risks · Opportunities) and re-order it (the server's
 * rank, value at stake, urgency); a grid of compact, equal-height cards that
 * each carry one headline, one big number and one urgency; and ONE expanded
 * panel at a time that opens under the selected card's row with the why, the
 * rule, the next step and the explanation. Selection is component state, not
 * URL state: which card is open is a reading position, not a view of the data,
 * and it should not survive a reload or pollute a shared link.
 *
 * Three rules are load-bearing here and none of them are decoration:
 *
 * 1. NO RESTATEMENT. The rail never paraphrases a chart. Every sentence the
 *    server returns carries a `lens` — cause / norm / delta / action / answer /
 *    state — which is the licence for that sentence to exist at all. The lens
 *    is therefore rendered, in plain words, next to the sentence: the reader
 *    can see that a line is here because it explains a cause, not because it
 *    read a bar off a chart they are already looking at.
 *
 * 2. REFUSALS ARE SHOWN, NOT SWALLOWED. `dropped` (sentences the server threw
 *    away as redundant with what is on screen) and `rejected` (sentences that
 *    used tokens the data does not support) stay reachable in a provenance
 *    disclosure. Being able to read what the product REFUSED to say is the
 *    reason to believe what it does say.
 *
 * 3. OPPORTUNITY IS NOT A PROBLEM. `framing: "opportunity"` — the cross-sell
 *    and whitespace findings — gets a positive treatment: --good, never red,
 *    never amber, and the value reads "in play", not "at risk". An upside
 *    finding presented in the colour of a fire drill trains people to close
 *    the rail.
 *
 * Nothing here computes a business figure. Filtering and sorting re-arrange
 * the server's cards on fields the server already sent; the segment counts are
 * counts of cards, not of anything in the data. `valueAtStake` is the only raw
 * figure on a card and it goes through lib/format so it reads identically to
 * the same dollar in a KPI tile. Nothing calls Date.now(): the business date
 * is AS_OF and it arrives on the payload.
 *
 * Motion is FLIP: before any change that re-orders the grid the visual rects
 * are snapshotted, and after React commits each surviving card is given the
 * inverse transform and released, so the list slides into its new order rather
 * than jumping. The expanded row animates its height with a grid-template-rows
 * transition (0fr → 1fr). Both run on the --dur tokens, which reduced-motion
 * zeroes, and the snapshot is skipped entirely under that preference.
 *
 * Interaction: the card body toggles the panel (mouse), and its "Details"
 * button does the same thing (keyboard, with aria-expanded). Escape collapses
 * and hands focus back to that button. Every other affordance is a real
 * <button> or a native <details>, so there is no custom role handler to get
 * wrong.
 */
import {
  Fragment,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
  type RefObject,
  type TransitionEvent as ReactTransitionEvent,
} from "react";
import type { ActionCard, DimKey, Narrative, Sentence, Tone, Urgency } from "../api/types";
import { money, num } from "../lib/format";

type LensKey = Sentence["lens"];

/** The lens in plain words. The taxonomy names are the server's; the reader
 *  gets the question each one answers. */
const LENS_LABEL: Record<LensKey, string> = {
  cause: "Why",
  norm: "Compared with",
  delta: "What changed",
  action: "What to do",
  answer: "Answer",
  state: "Where it stands",
};

/** What each lens is FOR. Shown as the label's tooltip so the taxonomy is
 *  learnable on the spot rather than documented in a deck nobody opens. */
const LENS_WHY: Record<LensKey, string> = {
  cause: "Why this is happening — a mechanism no chart on screen states.",
  norm: "What normal looks like, so the figure has something to be judged against.",
  delta: "What changed since the last look.",
  action: "What to do about it.",
  answer: "A direct answer to the question this page asks.",
  state: "A statement of the current state — allowed only because no chart here says it.",
};

const LENS_ORDER: LensKey[] = ["answer", "cause", "norm", "delta", "state", "action"];

type FilterKey = "all" | "critical" | "high" | "risk" | "opportunity";
type SortKey = "ranked" | "value" | "urgency";

const FILTERS: { key: FilterKey; label: string; match: (c: ActionCard) => boolean }[] = [
  { key: "all", label: "All", match: () => true },
  { key: "critical", label: "Critical", match: (c) => c.urgencyLabel === "Critical" },
  { key: "high", label: "High", match: (c) => c.urgencyLabel === "High" },
  { key: "risk", label: "Risks", match: (c) => c.framing === "risk" },
  { key: "opportunity", label: "Opportunities", match: (c) => c.framing === "opportunity" },
];

/** "Ranked" is the server's own order — value at stake against urgency, which
 *  is what the rank numeral in each card's circle refers to. The other two are
 *  single-axis re-orderings of the same cards; the numeral stays the server's
 *  rank rather than being renumbered, so a card keeps its identity as it moves. */
const SORTS: { key: SortKey; label: string }[] = [
  { key: "ranked", label: "Ranked" },
  { key: "value", label: "By value at stake" },
  { key: "urgency", label: "By urgency" },
];

const URGENCY_RANK: Record<Urgency, number> = { Critical: 3, High: 2, Medium: 1, Low: 0 };

function orderBy(sort: SortKey, cards: ActionCard[]): ActionCard[] {
  const out = [...cards];
  switch (sort) {
    case "value":
      out.sort((a, b) => b.valueAtStake - a.valueAtStake || a.n - b.n);
      break;
    case "urgency":
      out.sort(
        (a, b) =>
          URGENCY_RANK[b.urgencyLabel] - URGENCY_RANK[a.urgencyLabel] ||
          b.urgency - a.urgency ||
          a.n - b.n,
      );
      break;
    default:
      out.sort((a, b) => a.n - b.n);
  }
  return out;
}

/**
 * Colour is goodness, not sign. An opportunity card is always --good however
 * loudly it scored, because urgency on an upside means "this expires", not
 * "this is broken".
 */
function toneFor(card: ActionCard): Tone {
  if (card.framing === "opportunity") return "good";
  switch (card.urgencyLabel) {
    case "Critical":
      return "danger";
    case "High":
      return "warn";
    case "Medium":
      return "accent";
    default:
      return "neutral";
  }
}

/** Emphasise the declared substrings, in order, once each. The payload ships
 *  plain text and a list of runs — never markup — so the same Sentence can be
 *  rendered by a client that has no DOM. */
function emphasise(text: string, bold?: string[]): ReactNode[] {
  if (!bold || bold.length === 0) return [text];
  const out: ReactNode[] = [];
  let rest = text;
  let k = 0;
  for (const b of bold) {
    if (!b) continue;
    const i = rest.indexOf(b);
    if (i < 0) continue; // a run the sentence does not contain is simply skipped
    if (i > 0) out.push(<Fragment key={k++}>{rest.slice(0, i)}</Fragment>);
    out.push(<b key={k++}>{b}</b>);
    rest = rest.slice(i + b.length);
  }
  if (rest) out.push(<Fragment key={k++}>{rest}</Fragment>);
  return out;
}

/** A click that landed on a nested control is that control's business. */
const INTERACTIVE =
  "button, a[href], summary, input, select, textarea, [role='button'], [contenteditable='true']";

const reducedMotion = () =>
  typeof window !== "undefined" &&
  (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);

/** The grid item that carries the expanded content. Keyed so the FLIP pass can
 *  slide it between rows when the selection moves. */
const DETAIL_KEY = "__detail";

type ExplainState =
  | { status: "loading" }
  | { status: "ready"; narrative: Narrative }
  | { status: "error"; message: string };

/** The panel's life: mounted closed, opened on the next style flush so the
 *  height transition has a start value, and kept mounted while it closes so
 *  the rows above it can shrink smoothly before it leaves the DOM. */
type PanelState = { key: string; phase: "opening" | "open" | "closing" };

/**
 * How many columns the grid is laying out right now, read from the resolved
 * grid-template-columns rather than from a breakpoint duplicated in TypeScript.
 * The panel is inserted after the last card of the selected card's row, which
 * is the one place the layout and the DOM order have to agree.
 */
function useColumnCount(ref: RefObject<HTMLElement | null>, enabled: boolean): number {
  const [cols, setCols] = useState(1);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!enabled || !el) return;
    const read = () => {
      const tracks = getComputedStyle(el).gridTemplateColumns;
      // A laid-out grid resolves to explicit track widths ("312px 312px 312px").
      // Anything else means it is not a grid yet; one column is the safe read.
      const n = tracks.includes("(") ? 1 : tracks.split(" ").filter(Boolean).length;
      setCols((c) => (c === Math.max(1, n) ? c : Math.max(1, n)));
    };
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, enabled]);
  return cols;
}

export interface ActionRailProps {
  actions: ActionCard[];
  /** Scope the page to the card's subject. Accepts `useScopeTo()` directly. */
  onScope: (scope: { dim: DimKey; value: string }) => void;
  /** Open the working behind the card — the deal drawer, usually. */
  onDrill: (drill: string) => void;
  /** Ask the server to explain this card. Resolves to a checked Narrative. */
  onExplain: (card: ActionCard) => Promise<Narrative>;
  title?: string;
  /** The payload's AS_OF. Never derived on the client. */
  asOf?: string;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  /** Skeleton cards to paint while the first payload is in flight. */
  skeletonCount?: number;
}

export function ActionRail({
  actions,
  onScope,
  onDrill,
  onExplain,
  title = "Do these next",
  asOf,
  loading = false,
  error = null,
  onRetry,
  skeletonCount = 3,
}: ActionRailProps) {
  const uid = useId();
  const headingId = `${uid}-title`;
  const panelId = `${uid}-panel`;

  const [filter, setFilter] = useState<FilterKey>("all");
  const [sort, setSort] = useState<SortKey>("ranked");
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [panel, setPanel] = useState<PanelState | null>(null);
  const [explains, setExplains] = useState<Record<string, ExplainState>>({});

  const listRef = useRef<HTMLUListElement>(null);
  const detailRef = useRef<HTMLLIElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const before = useRef<Map<string, DOMRect> | null>(null);
  const alive = useRef(true);
  const actionsRef = useRef(actions);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  // A new payload is a new set of cards, possibly under a new scope: the open
  // panel and the cached explanations belong to the old one. The functional
  // updates bail out when there is nothing to reset, so the first mount does
  // not pay for a second render.
  useEffect(() => {
    actionsRef.current = actions;
    setExplains((m) => (Object.keys(m).length ? {} : m));
    setExpandedKey(null);
    setPanel(null);
  }, [actions]);

  const counts = useMemo(() => {
    const out = {} as Record<FilterKey, number>;
    for (const f of FILTERS) out[f.key] = actions.filter(f.match).length;
    return out;
  }, [actions]);

  const visible = useMemo(() => {
    const match = FILTERS.find((f) => f.key === filter)?.match ?? (() => true);
    return orderBy(sort, actions.filter(match));
  }, [actions, filter, sort]);

  const showList = !error && !(loading && actions.length === 0) && actions.length > 0;
  const cols = useColumnCount(listRef, showList && visible.length > 0);

  /* ---------------------------------------------------------- FLIP motion */

  /** Record where every card is right now, including any in-flight transform,
   *  so the next commit can animate from the position the reader is looking
   *  at rather than from where the last animation was heading. */
  const snapshot = useCallback(() => {
    const list = listRef.current;
    if (!list || reducedMotion()) return;
    const m = new Map<string, DOMRect>();
    list.querySelectorAll<HTMLElement>("[data-flip]").forEach((el) => {
      m.set(el.dataset.flip as string, el.getBoundingClientRect());
    });
    before.current = m;
  }, []);

  useLayoutEffect(() => {
    const prev = before.current;
    before.current = null;
    const list = listRef.current;
    if (!prev || !list) return;
    const els = Array.from(list.querySelectorAll<HTMLElement>("[data-flip]"));
    // Three passes rather than one loop: measure everything at its natural
    // position first, then invert, then release — one forced layout, not one
    // per card.
    const moves: [HTMLElement, number, number][] = [];
    for (const el of els) {
      el.style.transition = "none";
      el.style.transform = "";
    }
    for (const el of els) {
      const was = prev.get(el.dataset.flip as string);
      if (!was) continue;
      const now = el.getBoundingClientRect();
      const dx = was.left - now.left;
      const dy = was.top - now.top;
      if (Math.abs(dx) >= 0.5 || Math.abs(dy) >= 0.5) moves.push([el, dx, dy]);
    }
    for (const [el, dx, dy] of moves) el.style.transform = `translate(${dx}px, ${dy}px)`;
    void list.getBoundingClientRect(); // commit the inverse before releasing it
    for (const el of els) el.style.transition = "";
    for (const [el] of moves) el.style.transform = "";
  });

  /* ------------------------------------------------------------ selection */

  const expand = useCallback(
    (key: string) => {
      snapshot();
      setExpandedKey(key);
      setPanel((p) => (p && p.phase !== "closing" ? { key, phase: "open" } : { key, phase: "opening" }));
    },
    [snapshot],
  );

  const collapse = useCallback(() => {
    snapshot();
    setExpandedKey(null);
    setPanel((p) => (p ? { ...p, phase: "closing" } : null));
  }, [snapshot]);

  const toggle = useCallback(
    (key: string) => {
      if (expandedKey === key) collapse();
      else expand(key);
    },
    [expandedKey, expand, collapse],
  );

  // A card that the filter has just removed cannot stay selected; its panel
  // leaves with it in the same commit, so there is no closing phase to run.
  useEffect(() => {
    if (expandedKey && !visible.some((c) => c.key === expandedKey)) {
      setExpandedKey(null);
      setPanel(null);
    }
  }, [visible, expandedKey]);

  // "opening" exists for exactly one style flush: the panel mounts at 0fr,
  // that state is forced into the computed style, and then it is told to be
  // open so the grid-template-rows transition has somewhere to start from.
  useLayoutEffect(() => {
    if (panel?.phase !== "opening") return;
    void detailRef.current?.getBoundingClientRect();
    setPanel((p) => (p && p.phase === "opening" ? { ...p, phase: "open" } : p));
  }, [panel]);

  // Focus follows the selection into the panel so Tab reaches its buttons
  // without crossing the rest of the row, and Escape works immediately. A
  // panel that is already at full height (the selection moved from one card
  // to another) is scrolled into view here; one that is still growing waits
  // for its transition to end, because "nearest" on a zero-height box would
  // reveal only its top edge.
  useEffect(() => {
    if (panel?.phase !== "open") return;
    const el = panelRef.current;
    if (!el) return;
    el.focus({ preventScroll: true });
    if ((detailRef.current?.getBoundingClientRect().height ?? 0) > 40) {
      el.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
    }
  }, [panel]);

  // Belt and braces for the closing phase: if the transition never reports
  // its end (a tab in the background, say), the panel still leaves.
  useEffect(() => {
    if (panel?.phase !== "closing") return;
    const t = window.setTimeout(() => {
      snapshot();
      setPanel((p) => (p && p.phase === "closing" ? null : p));
    }, 400);
    return () => window.clearTimeout(t);
  }, [panel, snapshot]);

  const onDetailTransitionEnd = useCallback(
    (e: ReactTransitionEvent<HTMLLIElement>) => {
      if (e.target !== e.currentTarget || e.propertyName !== "grid-template-rows") return;
      if (panel?.phase === "closing") {
        snapshot();
        setPanel(null);
      } else if (panel?.phase === "open") {
        panelRef.current?.scrollIntoView({
          block: "nearest",
          behavior: reducedMotion() ? "auto" : "smooth",
        });
      }
    },
    [panel, snapshot],
  );

  const onKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLElement>) => {
      if (e.key !== "Escape" || !expandedKey) return;
      e.preventDefault();
      e.stopPropagation();
      const key = expandedKey;
      collapse();
      const btn = listRef.current?.querySelector<HTMLElement>(
        `[data-flip="${CSS.escape(key)}"] .arail-card__more`,
      );
      btn?.focus();
    },
    [expandedKey, collapse],
  );

  /* ---------------------------------------------------------- explanation */

  const runExplain = useCallback(
    (card: ActionCard) => {
      const payload = actionsRef.current;
      setExplains((m) => ({ ...m, [card.key]: { status: "loading" } }));
      onExplain(card)
        .then((narrative) => {
          if (!alive.current || actionsRef.current !== payload) return;
          setExplains((m) => ({ ...m, [card.key]: { status: "ready", narrative } }));
        })
        .catch((err: unknown) => {
          if (!alive.current || actionsRef.current !== payload) return;
          setExplains((m) => ({
            ...m,
            [card.key]: { status: "error", message: err instanceof Error ? err.message : String(err) },
          }));
        });
    },
    [onExplain],
  );

  /* ------------------------------------------------------------ the panel */

  const panelCard = panel ? visible.find((c) => c.key === panel.key) ?? null : null;
  const panelIndex = panelCard ? visible.indexOf(panelCard) : -1;
  // After the last card of the selected card's row — full width, in place.
  const panelAfter =
    panelIndex < 0 ? -1 : Math.min(visible.length, Math.ceil((panelIndex + 1) / cols) * cols) - 1;

  const activeFilter = FILTERS.find((f) => f.key === filter) ?? FILTERS[0];

  const items: ReactNode[] = [];
  visible.forEach((card, i) => {
    items.push(
      <ActionRailCard
        key={card.key}
        card={card}
        index={i}
        open={expandedKey === card.key}
        panelId={panelId}
        onToggle={toggle}
      />,
    );
    if (i === panelAfter && panelCard && panel) {
      items.push(
        <ActionRailDetail
          key={DETAIL_KEY}
          card={panelCard}
          phase={panel.phase}
          panelId={panelId}
          detailRef={detailRef}
          panelRef={panelRef}
          explain={explains[panelCard.key]}
          onExplain={runExplain}
          onScope={onScope}
          onDrill={onDrill}
          onCollapse={collapse}
          onTransitionEnd={onDetailTransitionEnd}
        />,
      );
    }
  });

  return (
    <section className="arail" aria-labelledby={headingId} onKeyDown={onKeyDown}>
      <header className="arail-head">
        <div className="arail-head__lead">
          <h2 className="arail-head__title" id={headingId}>
            {title}
          </h2>
          {!loading && !error && actions.length > 0 ? (
            <span className="arail-head__count">
              {num(actions.length)}
              <span className="arail__sr"> ranked actions</span>
            </span>
          ) : null}
          {asOf ? <span className="arail-head__asof">as of {asOf}</span> : null}
        </div>
        <p className="arail-head__note">
          Ranked by value at stake against urgency. Open a card for why it fired and what to
          do about it.
        </p>

        {showList ? (
          <div className="arail-head__controls">
            <div className="arail-seg" role="group" aria-label="Show">
              {FILTERS.map((f) => {
                const n = counts[f.key];
                const on = filter === f.key;
                return (
                  <button
                    key={f.key}
                    type="button"
                    className="arail-seg__btn"
                    aria-pressed={on}
                    disabled={f.key !== "all" && n === 0}
                    onClick={() => {
                      if (on) return;
                      snapshot();
                      setFilter(f.key);
                    }}
                  >
                    {f.label}
                    <span className="arail-seg__n">
                      <span className="arail__sr">, </span>
                      {num(n)}
                    </span>
                  </button>
                );
              })}
            </div>
            <div className="arail-seg" role="group" aria-label="Sort">
              {SORTS.map((s) => {
                const on = sort === s.key;
                return (
                  <button
                    key={s.key}
                    type="button"
                    className="arail-seg__btn"
                    aria-pressed={on}
                    onClick={() => {
                      if (on) return;
                      snapshot();
                      setSort(s.key);
                    }}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}
      </header>

      {error ? (
        <div className="arail-error" role="alert">
          <p className="arail-error__title">The action feed could not be loaded.</p>
          <p className="arail-error__text">{error}</p>
          {onRetry ? (
            <button type="button" className="arail-btn arail-btn--primary" onClick={onRetry}>
              Try again
            </button>
          ) : null}
        </div>
      ) : loading && actions.length === 0 ? (
        <ul className="arail-list" aria-busy="true" aria-live="polite">
          {Array.from({ length: Math.max(1, skeletonCount) }, (_, i) => (
            <li className="arail-skeleton" key={i} aria-hidden="true">
              <span className="arail-skeleton__top">
                <span className="arail-skeleton__rank" />
                <span className="arail-skeleton__line arail-skeleton__line--short" />
              </span>
              <span className="arail-skeleton__line arail-skeleton__line--title" />
              <span className="arail-skeleton__line" />
              <span className="arail-skeleton__foot" />
            </li>
          ))}
          <li className="arail__sr">Loading the ranked actions for this scope.</li>
        </ul>
      ) : actions.length === 0 ? (
        <div className="arail-empty">
          <p className="arail-empty__title">Nothing fired in this scope.</p>
          <p className="arail-empty__text">
            No rule crossed its threshold for the current filters. Widen the scope, or read this
            as the quiet it looks like.
          </p>
        </div>
      ) : visible.length === 0 ? (
        <div className="arail-empty">
          <p className="arail-empty__title">
            No {activeFilter.label.toLowerCase()} cards in this scope.
          </p>
          <p className="arail-empty__text">
            The other {num(actions.length)} still stand — this filter just does not match any of
            them here.
          </p>
          <button
            type="button"
            className="arail-btn"
            onClick={() => {
              snapshot();
              setFilter("all");
            }}
          >
            Show all
          </button>
        </div>
      ) : (
        // While one card is open the list says so, and the stylesheet softens
        // every other card so the eye stays on the chosen one and its detail row.
        <ul className="arail-list" ref={listRef} data-focused={expandedKey ? "true" : undefined}>
          {items}
        </ul>
      )}
    </section>
  );
}

/* ========================================================================== *
 * The compact card. One height for every card in a row: a one-line top strip,
 * a headline clamped to two lines and reserved at two, and a one-line foot.
 * ========================================================================== */

function ActionRailCard({
  card,
  index,
  open,
  panelId,
  onToggle,
}: {
  card: ActionCard;
  index: number;
  open: boolean;
  panelId: string;
  onToggle: (key: string) => void;
}) {
  const tone = toneFor(card);
  const opportunity = card.framing === "opportunity";

  // Mouse only: the keyboard path is the Details button below, which is a
  // real button, so it needs no key handler of its own.
  const onCardClick = useCallback(
    (e: ReactMouseEvent<HTMLLIElement>) => {
      const t = e.target;
      if (t instanceof Element && t.closest(INTERACTIVE)) return;
      onToggle(card.key);
    },
    [card.key, onToggle],
  );

  return (
    <li
      className={[
        "arail-card",
        `arail-card--${tone}`,
        opportunity ? "arail-card--opportunity" : "arail-card--risk",
        open ? "is-open" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      data-flip={card.key}
      style={{ "--i": index } as CSSProperties}
      onClick={onCardClick}
    >
      <div className="arail-card__top">
        <span className="arail-card__rank" aria-hidden="true">
          {card.n}
        </span>
        <span className="arail-card__entity" title={`${card.entity.type} · ${card.entity.label}`}>
          <span className="arail-card__entity-type">{card.entity.type}</span>
          <span className="arail-card__entity-label">{card.entity.label}</span>
        </span>
        <span className="arail-pill">
          <span className="arail__sr">Urgency: </span>
          {card.urgencyLabel}
        </span>
      </div>

      <h3 className="arail-card__headline" title={card.headline}>
        <span className="arail__sr">Priority {card.n}. </span>
        {card.headline}
      </h3>

      <div className="arail-card__foot">
        <span className="arail-card__value">
          <span className="arail-card__value-num">{money(card.valueAtStake)}</span>
          <span className="arail-card__value-label">{opportunity ? "in play" : "at risk"}</span>
        </span>
        <span className="arail-card__owner" title={card.owner}>
          <span className="arail__sr">Owner: </span>
          {card.owner}
        </span>
        <button
          type="button"
          className="arail-card__more"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => onToggle(card.key)}
        >
          {open ? "Close" : "Details"}
          <svg className="arail-card__chev" viewBox="0 0 12 12" aria-hidden="true" focusable="false">
            <path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </li>
  );
}

/* ========================================================================== *
 * The expanded panel: a full-width grid row under the selected card's row.
 * Left, the evidence — why it fired and the literal rule. Right, the verb —
 * the next step, the actions, and the explanation that streams in beneath.
 * ========================================================================== */

function ActionRailDetail({
  card,
  phase,
  panelId,
  detailRef,
  panelRef,
  explain,
  onExplain,
  onScope,
  onDrill,
  onCollapse,
  onTransitionEnd,
}: {
  card: ActionCard;
  phase: PanelState["phase"];
  panelId: string;
  detailRef: RefObject<HTMLLIElement>;
  panelRef: RefObject<HTMLDivElement>;
  explain: ExplainState | undefined;
  onExplain: (card: ActionCard) => void;
  onScope: (scope: { dim: DimKey; value: string }) => void;
  onDrill: (drill: string) => void;
  onCollapse: () => void;
  onTransitionEnd: (e: ReactTransitionEvent<HTMLLIElement>) => void;
}) {
  const tone = toneFor(card);
  const opportunity = card.framing === "opportunity";
  const scopeTo = card.scopeTo ?? null;
  const explainId = `${panelId}-explain`;
  const status = explain?.status ?? "idle";

  return (
    <li
      className={`arail-detail arail-detail--${tone} arail-detail--${phase}`}
      role="presentation"
      data-flip={DETAIL_KEY}
      ref={detailRef}
      onTransitionEnd={onTransitionEnd}
    >
      <div className="arail-detail__clip">
        <div
          className="arail-detail__panel"
          role="region"
          id={panelId}
          aria-label={`Details: ${card.headline}`}
          tabIndex={-1}
          ref={panelRef}
        >
          <header className="arail-detail__head">
            <span className="arail-card__rank" aria-hidden="true">
              {card.n}
            </span>
            <span className="arail-detail__crumb">
              {card.entity.type} · {card.entity.label}
            </span>
            <span className="arail-pill">
              <span className="arail__sr">Urgency: </span>
              {card.urgencyLabel}
            </span>
            <span className="arail-detail__value">
              {money(card.valueAtStake)}
              <span className="arail-detail__value-label">{opportunity ? "in play" : "at risk"}</span>
            </span>
            <button
              type="button"
              className="arail-detail__close"
              onClick={onCollapse}
              aria-label="Collapse details"
              title="Collapse (Esc)"
            >
              <svg viewBox="0 0 12 12" aria-hidden="true" focusable="false">
                <path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </button>
          </header>

          <div className="arail-detail__cols">
            <div className="arail-detail__why">
              <h4 className="arail-detail__label">Why this fired</h4>
              {card.why.length > 0 ? (
                <ul className="arail-why">
                  {card.why.map((w, i) => (
                    <li className="arail-why__item" key={i}>
                      {w}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="arail-why__none">The rule alone selected this card.</p>
              )}
              <p className="arail-rule">
                <span className="arail-detail__label">Rule</span>
                <code className="arail-rule__code">{card.predicate}</code>
              </p>
            </div>

            <div className="arail-detail__do">
              <h4 className="arail-detail__label">Next step</h4>
              <blockquote className="arail-next">{card.nextStep}</blockquote>
              <p className="arail-detail__owner">
                <span className="arail-detail__label">Owner</span>
                {card.owner}
              </p>

              <div className="arail-detail__btns">
                {card.drill ? (
                  <button
                    type="button"
                    className="arail-btn arail-btn--primary"
                    onClick={() => onDrill(card.drill as string)}
                    title={`Open ${card.entity.label}`}
                  >
                    Open
                  </button>
                ) : null}
                {scopeTo ? (
                  <button
                    type="button"
                    className="arail-btn"
                    onClick={() => onScope(scopeTo)}
                    title={`Scope the page to ${scopeTo.value}`}
                  >
                    Filter the page to this
                  </button>
                ) : null}
                {status === "idle" || status === "loading" ? (
                  <button
                    type="button"
                    className="arail-btn"
                    onClick={() => onExplain(card)}
                    aria-controls={explainId}
                    disabled={status === "loading"}
                  >
                    {status === "loading" ? "Explaining…" : "Explain"}
                  </button>
                ) : null}
              </div>

              {/* Always mounted: an aria-live region has to exist before the
                  answer arrives for a screen reader to announce it. Empty, it
                  paints nothing. */}
              <div
                className="arail-explain"
                id={explainId}
                aria-live="polite"
                aria-busy={status === "loading"}
              >
                {status === "loading" ? (
                  <div className="arail-skel" aria-hidden="true">
                    <span className="arail-skel__line" />
                    <span className="arail-skel__line" />
                    <span className="arail-skel__line" />
                  </div>
                ) : null}
                {status === "loading" ? (
                  <p className="arail__sr">Checking this card against what the charts already say.</p>
                ) : null}

                {explain?.status === "error" ? (
                  <div className="arail-explain__error" role="alert">
                    <p className="arail-explain__error-title">
                      The explanation could not be generated.
                    </p>
                    <p className="arail-explain__error-text">{explain.message}</p>
                    <button type="button" className="arail-btn" onClick={() => onExplain(card)}>
                      Try again
                    </button>
                  </div>
                ) : null}

                {explain?.status === "ready" ? <Explanation narrative={explain.narrative} /> : null}
              </div>
            </div>
          </div>
        </div>
      </div>
    </li>
  );
}

function Explanation({ narrative }: { narrative: Narrative }) {
  const dropped = narrative.dropped ?? [];
  const rejected = narrative.rejected ?? [];
  const refusals = dropped.length + rejected.length;
  const sentences = [...narrative.sentences].sort(
    (a, b) => LENS_ORDER.indexOf(a.lens) - LENS_ORDER.indexOf(b.lens),
  );

  return (
    <>
      {narrative.headline ? <p className="arail-explain__headline">{narrative.headline}</p> : null}

      <ol className="arail-explain__list">
        {sentences.map((s, i) => (
          <li className="arail-explain__s" key={i} style={{ "--i": i } as CSSProperties}>
            <span className="arail-explain__lens" title={LENS_WHY[s.lens]}>
              {LENS_LABEL[s.lens]}
            </span>
            <span className="arail-explain__text">{emphasise(s.text, s.bold)}</span>
          </li>
        ))}
      </ol>

      {narrative.degraded ? (
        <p className="arail-explain__degraded" title={narrative.reason ?? undefined}>
          Written from the computed figures.
        </p>
      ) : null}

      {refusals > 0 ? (
        <details className="arail-prov">
          <summary className="arail-prov__summary">What this refused to say ({num(refusals)})</summary>

          {dropped.length > 0 ? (
            <div className="arail-prov__group">
              <p className="arail-prov__label">Dropped — a chart on this page already says it</p>
              <ul>
                {dropped.map((d, i) => (
                  <li className="arail-prov__item" key={i}>
                    <q>{d.text}</q>
                    <span className="arail-prov__reason">{d.reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {rejected.length > 0 ? (
            <div className="arail-prov__group">
              <p className="arail-prov__label">Rejected — the data does not support these terms</p>
              <ul>
                {rejected.map((r, i) => (
                  <li className="arail-prov__item" key={i}>
                    <q>{r.text}</q>
                    <span className="arail-prov__reason">
                      {r.rejectedTokens.map((t, j) => (
                        <code className="arail__token" key={j}>
                          {t}
                        </code>
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <p className="arail-prov__meta">
            {narrative.provider ? `${narrative.provider}` : "computed"}
            {narrative.model ? ` · ${narrative.model}` : ""}
            {typeof narrative.latencyMs === "number"
              ? ` · ${num(Math.round(narrative.latencyMs))} ms`
              : ""}
            {narrative.cached ? " · cached" : ""}
          </p>
        </details>
      ) : null}
    </>
  );
}
