/**
 * The KPI tile row.
 *
 * THE RULE THIS COMPONENT EXISTS TO ENFORCE: colour encodes goodness, not sign.
 * `tone` is the colour; `direction` says which way is good. Past-due pipeline
 * rising is DANGER even though the number went up, and pipeline coverage
 * rising is GOOD — so nothing here ever derives a colour from an arrow, a sign
 * or a delta. The tile paints `tone`; the direction glyph is muted chrome that
 * states, in words a screen reader reads out, whether higher is better or
 * worse. Two channels, two meanings, neither inferred from the other.
 *
 * The tile also holds no number of its own. `Kpi.formatted` is rendered
 * verbatim — the server formats, the client displays. There is no arithmetic
 * in this file and no call to a formatter, because a figure that is computed
 * twice is a figure that can disagree with itself.
 *
 * EVERY TILE IS A PLACE TO ASK. The customer's word was "referenceable": a
 * number on a tile should be something you can point at and interrogate, not
 * a label you read past. So every tile is a real button that opens a small
 * popover under itself — the label, the figure, the server's own sub line,
 * what its direction means in words, and one action that takes the question
 * to the Ask panel. The popover is the only place the tile explains itself;
 * the tile face stays as quiet as before, because six tiles that each carry
 * their own paragraph are a wall. One popover is open at a time, Escape and
 * an outside click close it, and it flips to stay on screen at the right edge.
 *
 * The raised icon square is the tile's handle: a tone-tinted solid, drawn in
 * the same isometric style as the persona marks, that lifts and tilts when the
 * tile is hovered so the whole card reads as something you can pick up.
 */
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type FocusEvent as ReactFocusEvent,
} from "react";
import type { Kpi, Tone } from "../api/types";
import { useApp } from "../state/AppStateProvider";
import { SIZE, Solid, solidFor } from "./icons";

export interface KpiRowProps {
  kpis: Kpi[];
  /** Renders the skeleton in place of the tiles. The layout does not shift. */
  loading?: boolean;
  /** What failed, in the server's own words. Shown with a retry. */
  error?: string | null;
  onRetry?: () => void;
  /**
   * When present, the popover offers "Scope the page to this" as well as the
   * question, and the tile whose key matches `selectedKey` is marked current.
   */
  onSelect?: (kpi: Kpi) => void;
  /** `Kpi.key` of the tile currently scoping the page, if any. */
  selectedKey?: string | null;
  /** How many placeholder tiles the skeleton draws. Matches the usual row. */
  skeletonCount?: number;
  /** Labels the row for assistive tech when a page shows more than one. */
  label?: string;
}

/** Spelled out rather than implied: this is what the glyph means. */
const DIRECTION_TEXT: Record<Kpi["direction"], string> = {
  "up-good": "Higher is better on this measure",
  "up-bad": "Higher is worse on this measure",
  neutral: "",
};

/** The arrow points at the good end of the scale, never at a movement. */
const DIRECTION_GLYPH: Record<Kpi["direction"], string> = {
  "up-good": "↑",
  "up-bad": "↓",
  neutral: "",
};

/**
 * The visible word must agree with the glyph.
 *
 * This read "better" for both directions, so a tile whose meaning is "higher is
 * worse" — top-account concentration, pipeline at risk — rendered as
 * "↑ better" to anyone looking at it, while the screen-reader label said the
 * opposite. That is precisely the confusion this component exists to prevent,
 * and it is worse than no flag at all: a reader who trusts it reads the most
 * dangerous tiles on the page backwards.
 */
const DIRECTION_WORD: Record<Kpi["direction"], string> = {
  "up-good": "higher is better",
  "up-bad": "lower is better",
  neutral: "",
};

/**
 * The popover's explanation of direction, in whole sentences. It says only
 * what `direction` and `tone` mean — what the measure counts is the server's
 * sub line, printed beside this, and nothing here guesses at it.
 */
const DIRECTION_EXPLAINED: Record<Kpi["direction"], { lead: string; rest: string }> = {
  "up-good": {
    lead: "Higher is better.",
    rest: "A rising figure here is good news; the colour of the tile says how it stands right now.",
  },
  "up-bad": {
    lead: "Lower is better.",
    rest: "A rising figure here is a warning, not a win; the colour of the tile says how it stands right now.",
  },
  neutral: {
    lead: "Neither direction is better.",
    rest: "This figure describes the scope rather than scoring it.",
  },
};

function DirectionFlag({ direction }: { direction: Kpi["direction"] }) {
  if (direction === "neutral") return null;
  const text = DIRECTION_TEXT[direction];
  return (
    <span className={`kpi-row__dir kpi-row__dir--${direction}`} role="img" aria-label={text}>
      <span className="kpi-row__dir-glyph" aria-hidden="true">
        {DIRECTION_GLYPH[direction]}
      </span>
      <span className="kpi-row__dir-word" aria-hidden="true">
        {DIRECTION_WORD[direction]}
      </span>
    </span>
  );
}

/**
 * The sparkline.
 *
 * Drawn only when the server sent a real series; a snapshot measure with no
 * honest history gets no line at all, because a flat stroke under a number
 * implies a trend was checked when it was not. The line is the tone colour at
 * low opacity with the final point marked, so the eye lands on "where it is
 * now" rather than on the shape.
 */
function Spark({ points, tone }: { points: number[]; tone: Tone }) {
  if (points.length < 3) return null;
  const W = 96;
  const H = 22;
  const lo = Math.min(...points);
  const hi = Math.max(...points);
  const span = hi - lo || 1;
  const x = (i: number) => (i / (points.length - 1)) * (W - 2) + 1;
  const y = (v: number) => H - 2 - ((v - lo) / span) * (H - 4);
  const d = points.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
  const last = points.length - 1;
  return (
    <svg
      className={`kpi-row__spark kpi-row__spark--${tone}`}
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
    >
      <path d={`${d} L${W - 1} ${H} L1 ${H} Z`} className="kpi-row__spark-fill" />
      <path d={d} className="kpi-row__spark-line" />
      <circle cx={x(last)} cy={y(points[last])} r="2.1" className="kpi-row__spark-dot" />
    </svg>
  );
}

function TileBody({ kpi }: { kpi: Kpi }) {
  return (
    <>
      <span className="kpi-row__top">
        <span className="kpi-row__label">{kpi.label}</span>
        <span className="kpi-row__icon" aria-hidden="true">
          <Solid name={solidFor(kpi.icon)} size={SIZE.tile} />
        </span>
      </span>
      {/* A tile whose value is a name rather than a figure ("Networking /
          Product") cannot wear the number size — it wraps into a headline
          and pushes the sub line off the card. It takes the heading size. */}
      <span
        className={`kpi-row__value${/[A-Za-z]{3,}/.test(kpi.formatted) ? " kpi-row__value--text" : ""}`}
      >
        {kpi.formatted}
      </span>
      <Spark points={kpi.spark ?? []} tone={kpi.tone} />
      <span className="kpi-row__sub">{kpi.sub}</span>
      <DirectionFlag direction={kpi.direction} />
    </>
  );
}

/**
 * The popover under an open tile.
 *
 * It is laid out below the tile, aligned to its left edge, then measured once
 * before paint: if that puts it past the right edge of the viewport it hangs
 * from the tile's right edge instead, and if it would run off the bottom and
 * there is room above, it opens upward. Layout, not a business figure.
 *
 * Focus lands on the dialog itself on arrival so its name and the figure are
 * announced before the actions; Tab then reaches "Ask" and "Close".
 */
function TilePopover({
  kpi,
  id,
  anchor,
  onAsk,
  onScope,
  onClose,
}: {
  kpi: Kpi;
  id: string;
  anchor: HTMLElement | null;
  onAsk: () => void;
  onScope?: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const headId = `${id}-head`;
  const descId = `${id}-desc`;
  const [place, setPlace] = useState({ flip: false, above: false });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const margin = 12;
    const flip = r.right > window.innerWidth - margin;
    const roomAbove = anchor ? anchor.getBoundingClientRect().top - margin : 0;
    const above = r.bottom > window.innerHeight - margin && roomAbove >= r.height;
    setPlace({ flip, above });
  }, [anchor]);

  useEffect(() => {
    ref.current?.focus({ preventScroll: true });
  }, []);

  const why = DIRECTION_EXPLAINED[kpi.direction];

  return (
    <div
      className={[
        "kpi-pop",
        `kpi-pop--${kpi.tone}`,
        place.flip ? "kpi-pop--flip" : "",
        place.above ? "kpi-pop--above" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      id={id}
      ref={ref}
      role="dialog"
      aria-labelledby={headId}
      aria-describedby={descId}
      tabIndex={-1}
    >
      <div className="kpi-pop__head">
        <span className="kpi-row__icon kpi-pop__mark" aria-hidden="true">
          <Solid name={solidFor(kpi.icon)} size={SIZE.hero} />
        </span>
        <div className="kpi-pop__title">
          <h3 className="kpi-pop__label" id={headId}>
            {kpi.label}
          </h3>
          <span className="kpi-pop__value">{kpi.formatted}</span>
        </div>
      </div>

      <div id={descId}>
        <p className="kpi-pop__sub">{kpi.sub}</p>
        <p className="kpi-pop__dir">
          <strong>{why.lead}</strong> {why.rest}
        </p>
      </div>

      <div className="kpi-pop__actions">
        <button type="button" className="kpi-pop__btn kpi-pop__btn--ask" onClick={onAsk}>
          Ask about this number
        </button>
        {onScope ? (
          <button type="button" className="kpi-pop__btn" onClick={onScope}>
            Scope the page to this
          </button>
        ) : null}
        <button type="button" className="kpi-pop__btn" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}

function Tile({
  kpi,
  open,
  onToggle,
  onClose,
  onSelect,
  selected,
}: {
  kpi: Kpi;
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  onSelect?: (kpi: Kpi) => void;
  selected: boolean;
}) {
  const { openAsk } = useApp();
  const slotRef = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const popId = useId();
  const tone = `kpi-row__tile--${kpi.tone}`;

  // The row hands down a fresh closure every render; the document listeners
  // below read it through a ref so they are bound once per opening, not once
  // per render of the row.
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  // Escape closes and hands focus back to the tile, so a keyboard user is
  // never dropped at the top of the document; a press anywhere outside the
  // tile and its popover closes without moving focus.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      closeRef.current();
      buttonRef.current?.focus();
    };
    const onDown = (e: MouseEvent) => {
      if (!slotRef.current?.contains(e.target as Node)) closeRef.current();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [open]);

  // Tabbing out of the popover closes it. A null target is the window losing
  // focus, which is not the user going anywhere.
  const onBlur = (e: ReactFocusEvent<HTMLDivElement>) => {
    if (!open) return;
    const next = e.relatedTarget;
    if (next && !slotRef.current?.contains(next)) onClose();
  };

  const ask = () => {
    onClose();
    openAsk(`What is behind ${kpi.label.toLowerCase()}?`);
  };

  const close = () => {
    onClose();
    buttonRef.current?.focus();
  };

  return (
    <div className={`kpi-row__slot${open ? " kpi-row__slot--open" : ""}`} ref={slotRef} onBlur={onBlur}>
      {/* The whole tile is the hit target, so it is a real <button> —
          focusable, Enter/Space operable and announced as expanded when its
          popover is open, none of which a div with an onClick gets for free. */}
      <button
        type="button"
        ref={buttonRef}
        className={`kpi-row__tile ${tone} kpi-row__tile--button${selected ? " kpi-row__tile--selected" : ""}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? popId : undefined}
        aria-current={selected ? "true" : undefined}
        onClick={onToggle}
      >
        <TileBody kpi={kpi} />
      </button>

      {open ? (
        <TilePopover
          kpi={kpi}
          id={popId}
          anchor={slotRef.current}
          onAsk={ask}
          onScope={
            onSelect
              ? () => {
                  onClose();
                  onSelect(kpi);
                }
              : undefined
          }
          onClose={close}
        />
      ) : null}
    </div>
  );
}

/**
 * The skeleton mirrors the tile anatomy — rule, label, value, sub — so the
 * row does not resize when the payload lands.
 */
export function KpiRowSkeleton({ count = 6 }: { count?: number }) {
  const tiles = Array.from({ length: Math.max(1, count) }, (_, i) => i);
  return (
    <div className="kpi-row kpi-row--skeleton" data-count={tiles.length} aria-hidden="true">
      {tiles.map((i) => (
        <div className="kpi-row__tile kpi-row__tile--skeleton" key={i}>
          <span className="kpi-row__bone kpi-row__bone--label" />
          <span className="kpi-row__bone kpi-row__bone--value" />
          <span className="kpi-row__bone kpi-row__bone--sub" />
        </div>
      ))}
    </div>
  );
}

export function KpiRow({
  kpis,
  loading = false,
  error = null,
  onRetry,
  onSelect,
  selectedKey = null,
  skeletonCount = 6,
  label = "Key figures for the current scope",
}: KpiRowProps) {
  // One popover at a time, so the row owns which tile is open. A new payload
  // is a new row; anything that was open belongs to figures that are gone.
  const [openKey, setOpenKey] = useState<string | null>(null);
  useEffect(() => {
    setOpenKey(null);
  }, [kpis]);

  // The announcement, not the tiles, is what goes into the live region: a
  // polite region wrapped around six tiles re-reads the whole row on every
  // filter change, which is how a screen-reader user learns to ignore it.
  const status = loading
    ? "Loading key figures"
    : error
      ? `Key figures unavailable. ${error}`
      : kpis.length
        ? `${kpis.length} key figures updated`
        : "No key figures for this scope";

  return (
    <section className="kpi-row-region" aria-label={label} aria-busy={loading || undefined}>
      <p className="visually-hidden" role="status" aria-live="polite">
        {status}
      </p>

      {loading ? (
        <KpiRowSkeleton count={skeletonCount} />
      ) : error ? (
        // Says what failed and offers the way back, rather than leaving a gap
        // the reader has to interpret as either "zero" or "broken".
        <div className="kpi-row__error" role="alert">
          <span className="kpi-row__error-title">Key figures did not load</span>
          <span className="kpi-row__error-detail">{error}</span>
          {onRetry ? (
            <button type="button" className="kpi-row__retry" onClick={onRetry}>
              Retry
            </button>
          ) : null}
        </div>
      ) : kpis.length === 0 ? (
        <p className="kpi-row__empty">No key figures are defined for this scope.</p>
      ) : (
        <div className="kpi-row" data-count={kpis.length}>
          {kpis.map((kpi) => (
            <Tile
              key={kpi.key}
              kpi={kpi}
              open={openKey === kpi.key}
              onToggle={() => setOpenKey((k) => (k === kpi.key ? null : kpi.key))}
              onClose={() => setOpenKey((k) => (k === kpi.key ? null : k))}
              onSelect={onSelect}
              selected={selectedKey === kpi.key}
            />
          ))}
        </div>
      )}
    </section>
  );
}
