/**
 * Ask about THIS chart.
 *
 * Every chart carries its own Ask. The panel opens beside the chart it belongs
 * to — pinned to the right of that card, never centred over the page — because
 * the answer is about the marks you are looking at and you should be able to
 * read both at once. The rest of the page goes quiet behind a blur; the chart
 * itself is lifted above it and stays sharp. That is the whole idea, and it is
 * why this is not simply the main Ask panel opened with a different question.
 *
 * Three things are worth knowing about how it behaves:
 *
 *   THE BOX IS BLANK. No suggested questions, no example in the placeholder:
 *   the customer asked for a plain space to type into, beside the chart, with
 *   the answer in words. The trigger is the spark alone for the same reason.
 *
 *   NOTHING HERE COMPUTES A NUMBER, AND NOTHING HERE DRAWS ONE. The question
 *   goes to the server with the chart's id and the claims already on screen,
 *   the server answers from that chart's own rows, and this panel shows the
 *   sentences only. A chart you are already looking at does not need to be
 *   drawn again beside the words about it; "Open in full chat" is where a
 *   follow-up gets its own chart.
 *
 *   BEING TURNED DOWN IS NOT AN ERROR. When the data cannot answer a question
 *   the server says so, and this panel states it plainly instead of dressing it
 *   up as a failure. A product that guesses is worse than one that declines.
 *
 * The conversation here is deliberately short. "Open in full chat" hands the
 * last question to the main Ask panel, which keeps the longer history.
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
  type ReactNode,
  type Ref,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { api } from "../api/client";
import type { AskResponse, ChartSpec, Sentence } from "../api/types";
import { describe } from "../lenses/useView";
import { useApp } from "../state/AppStateProvider";
import type { AskSeed } from "../state/filters";
import { Glyph } from "./askGlyphs";

/* ------------------------------------------------------------------- lenses */

/**
 * Why a sentence is allowed to exist next to a chart, said in plain words.
 * The model may not restate what the chart already shows, so every sentence
 * has to earn its place one of these six ways.
 */
const LENS_META: Record<Sentence["lens"], { label: string; why: string }> = {
  answer: { label: "Answer", why: "The direct answer to what you asked." },
  cause: { label: "Why", why: "What produced this. The chart shows the level, not the reason." },
  norm: {
    label: "Compared with",
    why: "Sets the figure against a peer or a baseline that is not on this chart.",
  },
  delta: { label: "Change", why: "What moved, and since when." },
  action: { label: "Do next", why: "The move this suggests. A chart describes; it does not advise." },
  state: {
    label: "Where it stands",
    why: "A plain fact, allowed only because no chart on screen already says it.",
  },
};

/* ------------------------------------------------------------------ helpers */

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "summary",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

/**
 * Emphasise each declared run once. `bold` carries exact substrings of `text`,
 * never markup — the server does not ship HTML into this client — and a run the
 * text does not contain is skipped rather than guessed at.
 */
function emphasise(text: string, bold: string[] | undefined): ReactNode[] {
  if (!bold || bold.length === 0) return [text];
  const out: ReactNode[] = [];
  let rest = text;
  let k = 0;
  for (const b of bold) {
    if (!b) continue;
    const i = rest.indexOf(b);
    if (i < 0) continue;
    if (i > 0) out.push(<Fragment key={k++}>{rest.slice(0, i)}</Fragment>);
    out.push(<b key={k++}>{b}</b>);
    rest = rest.slice(i + b.length);
  }
  if (rest) out.push(<Fragment key={k++}>{rest}</Fragment>);
  return out;
}

/* ---------------------------------------------------------------- geometry */

/**
 * Layout arithmetic, not styling. The panel is placed from a measured
 * rectangle, so these are plain numbers; everything the stylesheet needs is
 * handed over as a custom property and every colour, radius and inset stays in
 * the theme where it belongs.
 *
 * The rule is that the chart and the answer are ALWAYS both on screen. A card
 * in the left column has room to its right and the panel simply sits there.
 * A card in the right column, or one spanning the page, has no room — so the
 * card makes some: it slides left over the blurred page, as far as the left
 * gutter, and only if that is still not enough does it give up width from its
 * right edge. Sliding is free (a transform; the chart is not redrawn) and
 * narrowing costs one redraw at the new width, which is why sliding comes
 * first. The card never gives up more than half the viewport, and the panel
 * never takes less than it needs to be readable. Under 720px there is no
 * "beside" to speak of and the panel becomes a sheet along the bottom with
 * the card scrolled to the top of the screen above it.
 */
const SIDE_MIN_W = 340;
const SIDE_MAX_W = 520;
/** The least the panel accepts once the card has retreated as far as it can. */
const OVER_MIN_W = 300;
/** The card keeps at least this, or half the viewport, whichever is larger. */
const CARD_MIN_W = 320;
const CARD_SHARE = 0.5;
/** The panel is placed so it can be at least this tall before it is pinned to the top of the card. */
const PANEL_PREF_H = 560;
const PANEL_MIN_H = 260;
/** Below this the panel is a bottom sheet. Matches [data-place="sheet"] in the stylesheet. */
const SHEET_BELOW = 720;

interface Geometry {
  top: number;
  left: number;
  width: number;
  maxHeight: number;
  /**
   * "side" when the panel fits beside the chart as it stands; "over" when the
   * card had to slide or narrow to make the room; "sheet" on a phone.
   */
  place: "side" | "over" | "sheet";
  /** How far the host card is slid to the left, in px. */
  shift: number;
  /** How much narrower the host card is drawn than its natural width, in px. */
  shrink: number;
}

/** The lift currently applied to the host card, so a re-measure can undo it. */
interface Lift {
  shift: number;
  shrink: number;
}

const NO_LIFT: Lift = { shift: 0, shrink: 0 };

/** The gutter comes from the spacing scale, so the panel breathes like the rest of the app. */
function tokenPx(name: string, fallback: number): number {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const n = Number.parseFloat(raw);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Where everything goes. `lift` is what the card is already wearing from the
 * previous measurement: the slide is a pure translateX and the narrowing is
 * `calc(100% - shrink)`, so the card's natural rectangle is recovered by
 * arithmetic rather than by un-styling it, measuring, and re-styling it — which
 * would restart its transition on every resize event.
 */
function measure(host: Element | null, lift: Lift): Geometry | null {
  if (!host) return null;
  const gap = tokenPx("--s-4", 16);
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  const b = host.getBoundingClientRect();
  const natural = { top: b.top, left: b.left + lift.shift, width: b.width + lift.shrink };
  const right = natural.left + natural.width;

  if (vw < SHEET_BELOW) {
    const maxHeight = Math.round(vh * 0.7);
    return { top: vh - maxHeight, left: 0, width: vw, maxHeight, place: "sheet", shift: 0, shrink: 0 };
  }

  let place: Geometry["place"];
  let width: number;
  let left: number;
  let shift = 0;
  let shrink = 0;

  const room = vw - right - gap * 2;
  if (room >= SIDE_MIN_W) {
    place = "side";
    width = Math.min(room, SIDE_MAX_W);
    left = right + gap;
  } else {
    place = "over";
    // How far the card's right edge is able to retreat: all the way to the
    // left gutter by sliding, then down to its floor by narrowing.
    const floor = Math.min(natural.width, Math.max(CARD_MIN_W, Math.round(vw * CARD_SHARE)));
    const slideMax = Math.max(0, natural.left - gap);
    const shrinkMax = Math.max(0, natural.width - floor);
    const retreatMax = slideMax + shrinkMax;
    // The panel takes what a full retreat would free, within its own range.
    const panelMax = vw - gap * 2 - right + retreatMax;
    width = Math.max(Math.min(panelMax, SIDE_MAX_W), OVER_MIN_W);
    // And the card retreats only as far as that panel actually needs.
    const retreat = Math.min(retreatMax, Math.max(0, width + gap * 2 - (vw - right)));
    shift = Math.round(Math.min(retreat, slideMax));
    shrink = Math.round(retreat - shift);
    left = vw - gap - width;
  }

  // Level with the top of the chart, but never so low that the panel is left
  // stubby: it moves up until it can be a comfortable height, and the height
  // it is allowed is whatever is left below that point.
  const prefer = Math.min(vh - gap * 2, PANEL_PREF_H);
  const top = Math.min(Math.max(natural.top, gap), Math.max(gap, vh - gap - Math.max(prefer, PANEL_MIN_H)));

  return {
    top: Math.round(top),
    left: Math.round(left),
    width: Math.round(width),
    maxHeight: Math.round(vh - gap - top),
    place,
    shift,
    shrink,
  };
}

/* -------------------------------------------------------------- the trigger */

export interface ChartAskButtonProps {
  /** Open the panel for this chart. */
  onClick: () => void;
  /** True while this chart's panel is on screen. */
  open?: boolean;
  /** The chart's title, so the button says which chart it asks about. */
  chartTitle?: string;
  /** Focus comes back here when the panel closes. */
  buttonRef?: Ref<HTMLButtonElement>;
}

/**
 * The trigger that sits in a chart card's header: a spark and the word Ask,
 * as a small pill on a wash of the accent. Always visible, never hover-only —
 * a feature nobody can see is a feature nobody uses, and the customer asked
 * for this one by name. It stays small so that eight of them down a page of
 * charts read as eight affordances rather than eight distractions, and it
 * reads as pressed while its panel is open so the lifted card itself says
 * which control opened the panel beside it.
 */
export function ChartAskButton({ onClick, open = false, chartTitle, buttonRef }: ChartAskButtonProps) {
  return (
    <button
      type="button"
      ref={buttonRef}
      className="chart-ask-button"
      onClick={onClick}
      aria-haspopup="dialog"
      aria-expanded={open}
      aria-label={chartTitle ? `Ask about ${chartTitle}` : "Ask about this chart"}
    >
      <Glyph kind="spark" className="chart-ask-button__glyph" />
    </button>
  );
}

/* ---------------------------------------------------------------- the panel */

export interface ChartAskProps {
  /** The chart being asked about. Carries its own suggested questions. */
  spec: ChartSpec;
  /** What the charts on this page already say. The server forbids repeating it. */
  chartsSay: string[];
  /** Hand the conversation so far to the main Ask panel, which keeps the history. */
  onExpand: (question: string, seed: AskSeed) => void;
  onClose: () => void;
  open: boolean;
}

/**
 * Mount the panel only while it is open, so focus, the scroll lock and the
 * lift on the host card are all tied to one mount and one unmount rather than
 * to a pile of conditions. The anchor left behind is how the panel finds the
 * card it belongs to.
 */
export function ChartAsk(props: ChartAskProps) {
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  return (
    <span className="chart-ask" ref={anchorRef}>
      {props.open ? <ChartAskPanel {...props} anchorRef={anchorRef} /> : null}
    </span>
  );
}

type TurnStatus = "pending" | "done" | "failed";

interface Turn {
  id: number;
  question: string;
  status: TurnStatus;
  response: AskResponse | null;
  error: string | null;
}

function ChartAskPanel({
  spec,
  chartsSay,
  onExpand,
  onClose,
  anchorRef,
}: ChartAskProps & { anchorRef: RefObject<HTMLSpanElement | null> }) {
  const { ctx, state } = useApp();

  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [geom, setGeom] = useState<Geometry | null>(null);

  const panelRef = useRef<HTMLElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const nextId = useRef(0);
  const inflight = useRef<Map<number, AbortController>>(new Map());

  const titleId = useId();
  const introId = useId();
  const inputId = useId();
  const hintId = useId();

  // This chart's own claims travel with the page's, so the server knows not to
  // read the marks back to you.
  const says = useMemo(
    () => Array.from(new Set([...chartsSay, ...(spec.says ?? [])])),
    [chartsSay, spec.says],
  );

  const close = useCallback(() => onClose(), [onClose]);

  /* ---- asking ---------------------------------------------------------- */

  const run = useCallback(
    (id: number, question: string) => {
      inflight.current.get(id)?.abort();
      const ac = new AbortController();
      inflight.current.set(id, ac);
      api
        .ask(ctx, question, says, ac.signal, { id: spec.id, page: state.page })
        .then((response) => {
          if (ac.signal.aborted) return;
          inflight.current.delete(id);
          if (!response.answer) {
            setTurns((t) =>
              t.map((x) =>
                x.id === id
                  ? { ...x, status: "failed", error: "No answer came back for that question." }
                  : x,
              ),
            );
            return;
          }
          setTurns((t) =>
            t.map((x) => (x.id === id ? { ...x, status: "done", response, error: null } : x)),
          );
        })
        .catch((e: unknown) => {
          if (ac.signal.aborted) return;
          inflight.current.delete(id);
          setTurns((t) =>
            t.map((x) =>
              x.id === id ? { ...x, status: "failed", response: null, error: describe(e) } : x,
            ),
          );
        });
    },
    [ctx, says, spec.id, state.page],
  );

  const submit = useCallback(
    (raw: string) => {
      const question = raw.trim();
      if (!question) return;
      const id = nextId.current++;
      setTurns((t) => [...t, { id, question, status: "pending", response: null, error: null }]);
      setLastQuestion(question);
      setDraft("");
      run(id, question);
    },
    [run],
  );

  const retry = useCallback(
    (turn: Turn) => {
      setTurns((t) => t.map((x) => (x.id === turn.id ? { ...x, status: "pending", error: null } : x)));
      run(turn.id, turn.question);
    },
    [run],
  );

  // Nothing in flight outlives the panel.
  useEffect(() => {
    const map = inflight.current;
    return () => {
      for (const ac of map.values()) ac.abort();
      map.clear();
    };
  }, []);

  /* ---- anchoring ------------------------------------------------------- */

  // The card this Ask belongs to. Found once, from the anchor left in the card.
  const hostRef = useRef<HTMLElement | null>(null);
  // What the card is currently wearing, so the next measurement can undo it.
  const liftRef = useRef<Lift>(NO_LIFT);

  // Measure, then hand the card its slide and its narrowing as two custom
  // properties. The stylesheet turns them into a transform and a width; this
  // file never writes a style value that is not a length it measured.
  const place = useCallback(() => {
    const host = hostRef.current;
    const g = measure(host, liftRef.current);
    if (!host || !g) return;
    liftRef.current = { shift: g.shift, shrink: g.shrink };
    host.style.setProperty("--chart-ask-shift", `${g.shift}px`);
    host.style.setProperty("--chart-ask-shrink", `${g.shrink}px`);
    setGeom(g);
  }, []);

  // A layout effect, not a plain one: the panel must be measured and the card
  // lifted before the browser paints, or the panel shows for one frame at the
  // top-left corner and the card jumps into place a frame after the scrim.
  useLayoutEffect(() => {
    const anchor = anchorRef.current;
    const host =
      (anchor?.closest(".card") as HTMLElement | null) ?? (anchor?.parentElement ?? null);
    hostRef.current = host;
    if (!host) return;

    // The page is frozen for as long as the panel is open. Lock it BEFORE
    // measuring — losing a scrollbar changes the viewport width, and a panel
    // placed from the wrong width sits a scrollbar's width off its card.
    const body = document.body;
    const prevOverflow = body.style.overflow;
    body.style.overflow = "hidden";

    // If the chart is half off screen, bring it fully into view — the lock
    // stops the reader scrolling, not this code, and being frozen on a view of
    // half a chart is the one way this interaction can strand somebody. On a
    // phone the panel is a sheet along the bottom, so the card goes to the top
    // of the screen where it stays visible above the sheet.
    host.scrollIntoView({
      block: window.innerWidth < SHEET_BELOW ? "start" : "nearest",
      inline: "nearest",
    });
    host.classList.add("chart-ask-host");
    place();

    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("resize", place);
      body.style.overflow = prevOverflow;
      host.style.removeProperty("--chart-ask-shift");
      host.style.removeProperty("--chart-ask-shrink");
      host.classList.remove("chart-ask-host");
      liftRef.current = NO_LIFT;
    };
  }, [anchorRef, place]);

  /* ---- modal behaviour ------------------------------------------------- */

  // Focus moves in on open and goes back to the Ask button on close, so a
  // keyboard user is never dropped at the top of the document.
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const t = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => {
      window.clearTimeout(t);
      const trigger = hostRef.current?.querySelector<HTMLElement>(".chart-ask-button") ?? null;
      const back = opener && document.contains(opener) ? opener : trigger;
      if (back && document.contains(back)) back.focus();
    };
  }, []);

  // Escape closes; Tab is trapped inside the panel.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close();
        return;
      }
      if (e.key !== "Tab") return;
      const root = panelRef.current;
      if (!root) return;
      const nodes = Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
        (n) => n.getClientRects().length > 0,
      );
      if (nodes.length === 0) {
        e.preventDefault();
        root.focus();
        return;
      }
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      const active = document.activeElement;
      if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      } else if (e.shiftKey && (active === first || !root.contains(active))) {
        e.preventDefault();
        last.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [close]);

  // The newest answer is the one you want to read.
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const onInputKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit(draft);
    }
  };

  const busy = turns.some((t) => t.status === "pending");

  const style = {
    "--chart-ask-top": `${geom?.top ?? 0}px`,
    "--chart-ask-left": `${geom?.left ?? 0}px`,
    "--chart-ask-width": `${geom?.width ?? SIDE_MIN_W}px`,
    "--chart-ask-max-height": `${geom?.maxHeight ?? PANEL_MIN_H}px`,
  } as CSSProperties;

  return createPortal(
    <>
      <div className="chart-ask__scrim" onClick={close} aria-hidden="true" />
      <aside
        className="chart-ask__panel"
        style={style}
        data-place={geom?.place ?? "side"}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={introId}
        tabIndex={-1}
        ref={panelRef}
        // The panel is drawn on the body but still sits inside the card in the
        // React tree, so a click in here would otherwise reach the card's own
        // handlers.
        onClick={(e) => e.stopPropagation()}
      >
        <header className="chart-ask__head">
          <div className="chart-ask__heading">
            <Glyph kind="spark" className="chart-ask__glyph" />
            <p className="chart-ask__eyebrow">Ask about this chart</p>
            <h2 className="chart-ask__title" id={titleId}>
              {spec.title}
            </h2>
          </div>
          <button
            type="button"
            className="chart-ask__close"
            onClick={close}
            aria-label="Close and go back to the chart"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </header>

        <div className="chart-ask__body" ref={bodyRef}>
          <div
            className="chart-ask__thread"
            role="log"
            aria-live="polite"
            aria-relevant="additions"
            aria-busy={busy}
            aria-label="Answers"
          >
            {turns.length === 0 ? (
              <div className="chart-ask__intro" id={introId}>
                <p className="chart-ask__intro-body">
                  Ask anything about this chart. The answer is worked out from the rows behind
                  it, in the same slice the page is in.
                </p>
              </div>
            ) : null}

            {turns.map((turn) => (
              <article
                className="chart-ask__turn"
                key={turn.id}
                aria-label={`You asked: ${turn.question}`}
              >
                <p className="chart-ask__asked">
                  <span className="chart-ask__asked-label">You asked</span>
                  <span className="chart-ask__asked-text">{turn.question}</span>
                </p>

                {turn.status === "pending" ? <AnswerSkeleton /> : null}

                {turn.status === "failed" ? (
                  <div className="chart-ask-error" role="alert">
                    <p className="chart-ask-error__title">That question did not get through.</p>
                    <p className="chart-ask-error__detail">{turn.error}</p>
                    <button
                      type="button"
                      className="chart-ask-error__retry"
                      onClick={() => retry(turn)}
                    >
                      Ask it again
                    </button>
                  </div>
                ) : null}

                {turn.status === "done" && turn.response ? <Answer response={turn.response} /> : null}
              </article>
            ))}
          </div>
        </div>

        <form
          className="chart-ask__compose"
          onSubmit={(e) => {
            e.preventDefault();
            submit(draft);
          }}
        >
          <label className="chart-ask__compose-label pv-sr" htmlFor={inputId}>
            Ask a question about {spec.title}
          </label>
          <div className="chart-ask__field">
            <textarea
              id={inputId}
              ref={inputRef}
              className="chart-ask__input"
              rows={2}
              value={draft}
              placeholder="Type a question about this chart"
              aria-describedby={hintId}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onInputKeyDown}
            />
            <button type="submit" className="chart-ask__send" disabled={draft.trim().length === 0}>
              Ask
            </button>
          </div>
          <p className="chart-ask__hint" id={hintId}>
            Enter to send · Shift + Enter for a new line
          </p>
        </form>

        <footer className="chart-ask__foot">
          <button
            type="button"
            className="chart-ask__expand"
            onClick={() => {
              onExpand(
                lastQuestion || draft.trim(),
                turns
                  .filter((t) => t.status === "done" || t.status === "failed")
                  .map((t) => ({ question: t.question, response: t.response, chartTitle: spec.title })),
              );
              close();
            }}
          >
            <Glyph kind="spark" className="chart-ask__expand-glyph" />
            Open in full chat
          </button>
        </footer>
      </aside>
    </>,
    document.body,
  );
}

/* ------------------------------------------------------------------ answer */

function Answer({ response }: { response: AskResponse }) {
  const { answer, refused, degraded } = response;

  return (
    <div className={`chart-ask-answer${refused ? " chart-ask-answer--declined" : ""}`}>
      {refused ? (
        <p className="chart-ask-answer__declined">
          This one cannot be answered from the data behind this chart.
        </p>
      ) : null}

      <h4 className="chart-ask-answer__headline">{answer.headline}</h4>

      {answer.sentences.length > 0 ? (
        <ul className="chart-ask-answer__sentences">
          {answer.sentences.map((s, i) => {
            const meta = LENS_META[s.lens];
            return (
              <li className="chart-ask-answer__sentence" key={`${s.lens}-${i}`}>
                <span className="chart-ask-answer__lens" data-lens={s.lens} title={meta.why}>
                  {meta.label}
                </span>
                <span className="chart-ask-answer__text">{emphasise(s.text, s.bold)}</span>
              </li>
            );
          })}
        </ul>
      ) : null}

      {degraded ? (
        <p className="chart-ask-answer__quiet">
          Written from the computed figures this time. The numbers are the same either way.
        </p>
      ) : null}
    </div>
  );
}

/* ---------------------------------------------------------------- skeleton */

/** The shape of the answer that is coming, so nothing jumps when it lands. */
function AnswerSkeleton() {
  return (
    <div className="chart-ask-skeleton">
      <div className="chart-ask-skeleton__bars" aria-hidden="true">
        <div className="chart-ask-skeleton__line" />
        <div className="chart-ask-skeleton__line" />
        <div className="chart-ask-skeleton__line chart-ask-skeleton__line--short" />
      </div>
      <p className="chart-ask-skeleton__caption">Working it out against this chart…</p>
    </div>
  );
}
