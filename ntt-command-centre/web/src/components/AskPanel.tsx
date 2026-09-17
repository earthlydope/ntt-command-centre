/**
 * Ask AI Expert — the right-hand overlay.
 *
 * It is drawn ABOVE the page, not instead of it: a scrim, a panel, and the
 * lens still visible behind. Nothing navigates, nothing opens a window, and
 * the filter state behind the panel is untouched — closing it returns you to
 * exactly the screen you asked from.
 *
 * The panel is written for someone who does not want to know how it works.
 * An answer is a headline, a few short sentences each introduced by a plain
 * word ("Why", "Compared with", "What to do"), and — when the answer is a
 * breakdown or a trend — the chart that shows it. Everything that proves the
 * answer is honest sits behind one quiet "Details" disclosure at the end:
 *
 *   SCOPE.  The header states, in words, the slice the question is being
 *           answered within. The row-level predicate the server resolved from
 *           the persona is there too, behind a small disclosure, for anyone
 *           who wants to see the rule rather than its name.
 *
 *   PLAN.   Every answer carries the query plan it ran, shown as formatted
 *           JSON. For a typed question the model wrote the plan; for one of
 *           the suggested questions the server chose it itself. Either way
 *           the SERVER executed it and produced every figure, and the plan is
 *           the receipt for that division of labour.
 *
 *   LENS.   Each sentence carries the lens that licenses it to exist —
 *           cause, norm, delta, action, answer, state. The governing rule of
 *           this product is that the AI may not restate what a chart already
 *           shows, so the lens is not decoration: it is the sentence's reason
 *           for being prose rather than a mark. On screen it is a plain word.
 *           Sentences the server threw away (`dropped`, `rejected`) are listed
 *           in Details rather than hidden, because being able to read what the
 *           product refused to say is why the rest of it can be believed.
 *
 * `refused: true` is a first-class, calm state. A question the data cannot
 * support is declined in plain language, with something to ask instead when
 * the server offers it. That is correct behaviour, not an error, and it is
 * styled as a statement rather than as a failure.
 *
 * A thread can arrive already started: a chart's own Ask hands its turns over
 * as `seed`, so expanding into the main chat continues the conversation the
 * reader was having beside the chart rather than restarting it.
 */
import {
  Fragment,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  type RefObject,
} from "react";
import { api } from "../api/client";
import type { AskResponse, Scope, Sentence } from "../api/types";
import { describe } from "../lenses/useView";
import { num } from "../lib/format";
import { useApp } from "../state/AppStateProvider";
import type { AskSeed } from "../state/filters";
import { Glyph, glyphFor } from "./askGlyphs";
import { ChartCard } from "./ChartCard";

/**
 * Why a sentence exists at all, given that a chart is cheaper than prose —
 * said in the words a reader would use, not the lens's name. The `why` is the
 * tooltip; the `label` is what is printed above the sentence.
 */
const LENS_META: Record<Sentence["lens"], { label: string; why: string }> = {
  answer: { label: "Answer", why: "The direct answer to what you asked." },
  cause: { label: "Why", why: "What produced this. A chart shows the level; it cannot say why." },
  norm: {
    label: "Compared with",
    why: "Sets the figure against a peer, a plan or a baseline that is not on screen.",
  },
  delta: { label: "What changed", why: "What moved, and since when." },
  action: { label: "What to do", why: "The next move this suggests. Charts describe; they do not advise." },
  state: {
    label: "Where it stands",
    why: "A plain fact, allowed only because no chart on screen already says it.",
  },
};

/** How many of the server's alternatives a declined answer offers. */
const REFUSAL_ALTERNATIVES = 2;

/** A chart inside an answer is a compact one; the page's charts are taller. */
const ANSWER_CHART_HEIGHT = 240;

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
 * Emphasise each declared substring once, in order.
 *
 * `Sentence.bold` carries exact substrings of `Sentence.text` rather than
 * markup, so the server never ships HTML into the client. A run the text does
 * not contain is skipped rather than guessed at.
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

/**
 * The alternatives the server offers with a declined answer, read
 * defensively. `suggestions` rides along on the response; reading it this way
 * rather than widening the shared wire type means an answer served before the
 * field existed simply offers none instead of throwing.
 */
function alternatives(response: AskResponse): string[] {
  const raw: unknown = (response as AskResponse & { suggestions?: unknown }).suggestions;
  if (!Array.isArray(raw)) return [];
  const out: string[] = [];
  for (const q of raw as unknown[]) {
    if (typeof q === "string" && q.trim()) out.push(q.trim());
  }
  return out;
}

/** A provider name a reader can understand. Anything unknown is shown as sent. */
function providerLabel(provider: string): string {
  if (provider === "computed" || provider === "template") return "computed on the server";
  return provider;
}

type TurnStatus = "pending" | "done" | "failed";

interface Turn {
  id: number;
  question: string;
  status: TurnStatus;
  response: AskResponse | null;
  error: string | null;
  /** Set when the turn was asked about a specific chart and carried over. */
  chartTitle?: string;
}

export interface AskPanelProps {
  /** The scope label off the payload — what this panel is answering within. */
  scopeLabel: string;
  /** Suggested questions, chosen by the caller for the active persona. */
  suggestions: string[];
  /** Claim keys already on screen. The server forbids the model to restate them. */
  chartsSay: string[];
  onClose: () => void;
  /** A question the panel was opened about; asked once, on arrival. */
  initialQuery?: string;
  /** The full scope object, when the caller has the payload. Adds the
   *  row-level predicate to the header — the rule, not just its name. */
  scope?: Scope | null;
  /** The pinned business date on the payload. Never Date.now(). */
  asOf?: string;
  /** The control that opened the panel; focus returns to it on close. */
  triggerRef?: RefObject<HTMLElement | null>;
  /** Turns carried over from a chart's own Ask, shown first so the thread continues. */
  seed?: AskSeed | null;
}

export function AskPanel({
  scopeLabel,
  suggestions,
  chartsSay,
  onClose,
  initialQuery,
  scope,
  asOf,
  triggerRef,
  seed,
}: AskPanelProps) {
  const { ctx } = useApp();
  const label = scope?.label || scopeLabel || "your current scope";

  // A thread handed over from a chart's Ask starts the transcript, already
  // answered, so expanding into the main chat continues a conversation rather
  // than restarting one. Failed turns come across as failed and can be retried.
  const [turns, setTurns] = useState<Turn[]>(() =>
    (seed ?? []).map((t, i) => ({
      id: i,
      question: t.question,
      status: t.response ? "done" : "failed",
      response: t.response,
      error: t.response ? null : "That question did not get an answer on the chart.",
      chartTitle: t.chartTitle,
    })),
  );
  const [draft, setDraft] = useState("");
  // Once a question has been asked the chips fold away under "More questions"
  // so the transcript has the room; this is whether they are unfolded.
  const [moreOpen, setMoreOpen] = useState(false);

  const panelRef = useRef<HTMLElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const lastTurnRef = useRef<HTMLElement | null>(null);
  const nextId = useRef((seed ?? []).length);
  const inflight = useRef<Map<number, AbortController>>(new Map());
  // A query that arrives WITH a seed is usually the seed's last question,
  // already answered beside the chart — asking it again would duplicate the
  // last turn. It is matched against the seed's questions rather than assumed,
  // so a question the chart's Ask was still waiting on (not in the seed) is
  // asked here as the reader expects.
  const seeded = useRef<string | null>(
    (() => {
      const q = (initialQuery ?? "").trim();
      return q && (seed ?? []).some((t) => t.question.trim() === q) ? q : null;
    })(),
  );

  const titleId = useId();
  const scopeId = useId();
  const inputId = useId();
  const hintId = useId();

  const close = useCallback(() => onClose(), [onClose]);

  // ---- asking -----------------------------------------------------------
  const run = useCallback(
    (id: number, question: string) => {
      inflight.current.get(id)?.abort();
      const ac = new AbortController();
      inflight.current.set(id, ac);
      api
        .ask(ctx, question, chartsSay, ac.signal)
        .then((response) => {
          if (ac.signal.aborted) return;
          inflight.current.delete(id);
          if (!response.answer) {
            setTurns((t) =>
              t.map((x) =>
                x.id === id
                  ? {
                      ...x,
                      status: "failed",
                      error: "The semantic layer returned no answer for that question.",
                    }
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
    [ctx, chartsSay],
  );

  const submit = useCallback(
    (raw: string) => {
      const question = raw.trim();
      if (!question) return;
      const id = nextId.current++;
      setTurns((t) => [...t, { id, question, status: "pending", response: null, error: null }]);
      setDraft("");
      // Picking a chip is choosing; the rest of the row can fold away again.
      setMoreOpen(false);
      run(id, question);
    },
    [run],
  );

  const retry = useCallback(
    (turn: Turn) => {
      setTurns((t) =>
        t.map((x) => (x.id === turn.id ? { ...x, status: "pending", error: null } : x)),
      );
      run(turn.id, turn.question);
    },
    [run],
  );

  // A question carried in on `openAsk(query)` is asked once, on arrival.
  const submitRef = useRef(submit);
  useEffect(() => {
    submitRef.current = submit;
  });
  useEffect(() => {
    const q = (initialQuery ?? "").trim();
    if (!q || seeded.current === q) return;
    seeded.current = q;
    submitRef.current(q);
  }, [initialQuery]);

  // Nothing in flight outlives the panel.
  useEffect(() => {
    const map = inflight.current;
    return () => {
      for (const ac of map.values()) ac.abort();
      map.clear();
    };
  }, []);

  // ---- modal behaviour --------------------------------------------------
  // Focus moves in on open and returns to whatever opened the panel on close,
  // so a keyboard user is never dropped at the top of the document.
  useEffect(() => {
    const opener =
      triggerRef?.current ??
      (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    const t = window.setTimeout(() => inputRef.current?.focus(), 0);
    const body = document.body;
    const prevOverflow = body.style.overflow;
    body.style.overflow = "hidden";
    return () => {
      window.clearTimeout(t);
      body.style.overflow = prevOverflow;
      const back = triggerRef?.current ?? opener;
      if (back && document.contains(back)) back.focus();
    };
  }, [triggerRef]);

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

  // The newest turn is the one you want to read, so it is brought to the top
  // of the transcript — its question first, the answer flowing in beneath —
  // rather than the transcript being pinned to its very end, which would show
  // the tail of a long answer and hide its headline. On arrival with a seed
  // this lands on the turn carried over from the chart. Two frames of delay
  // let the charts above it settle their heights first: a chart lays out once
  // its wrapper has been measured, which is a frame after mount.
  useEffect(() => {
    const scroller = scrollerRef.current;
    const target = lastTurnRef.current;
    if (!scroller || !target) return;
    let second = 0;
    const first = window.requestAnimationFrame(() => {
      second = window.requestAnimationFrame(() => {
        const inset = Number.parseFloat(getComputedStyle(scroller).paddingTop) || 0;
        const offset = target.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
        scroller.scrollTop += offset - inset;
      });
    });
    return () => {
      window.cancelAnimationFrame(first);
      window.cancelAnimationFrame(second);
    };
  }, [turns.length]);

  const onInputKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit(draft);
    }
  };

  const busy = turns.some((t) => t.status === "pending");
  const started = turns.length > 0;

  return (
    <>
      <div className="ask-scrim" onClick={close} aria-hidden="true" />
      <aside
        className="ask-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={scopeId}
        tabIndex={-1}
        ref={panelRef}
      >
        <header className="ask-panel__header">
          <div className="ask-panel__heading">
            <Glyph kind="spark" className="ask-panel__glyph" />
            <h2 className="ask-panel__title" id={titleId}>
              Ask AI Expert
            </h2>
          </div>
          <button
            type="button"
            className="ask-panel__close"
            onClick={close}
            aria-label="Close Ask AI Expert"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </header>

        <div className="ask-panel__scope" id={scopeId}>
          <p className="ask-panel__scope-line">
            <span className="ask-panel__scope-label">Answering within</span>
            <strong className="ask-panel__scope-value">{label}</strong>
            {asOf ? (
              <span className="ask-panel__scope-meta">
                as of <time dateTime={asOf}>{asOf}</time>
              </span>
            ) : null}
          </p>
          {scope?.predicate ? (
            <details className="ask-panel__scope-details">
              <summary className="ask-panel__scope-summary">
                <span className="ask-panel__scope-caret" aria-hidden="true" />
                How this scope is applied
              </summary>
              <div className="ask-panel__scope-body">
                <p className="ask-panel__scope-note">
                  Only the rows inside this scope are counted, and they are chosen before anything
                  is added up. The page behind this panel keeps its filters; closing changes
                  nothing. The rule, as the server applies it:
                </p>
                <code className="ask-panel__predicate">{scope.predicate}</code>
              </div>
            </details>
          ) : null}
        </div>

        <div className="ask-panel__body" ref={scrollerRef}>
          {!started ? (
            <div className="ask-empty">
              <h3 className="ask-empty__title">Ask in your own words.</h3>
              <p className="ask-empty__body">
                Every number comes from the same data as the page — nothing is made up. If the
                data can&rsquo;t answer, you&rsquo;ll be told.
              </p>
            </div>
          ) : null}

          <div
            className="ask-transcript"
            role="log"
            aria-live="polite"
            aria-relevant="additions"
            aria-busy={busy}
            aria-label="Answers"
          >
            {turns.map((turn, i) => (
              <article
                className="ask-turn"
                key={turn.id}
                ref={i === turns.length - 1 ? lastTurnRef : undefined}
                aria-label={`Question: ${turn.question}`}
              >
                {turn.chartTitle ? (
                  <p className="ask-turn__about">
                    <Glyph kind="spark" className="ask-turn__about-glyph" />
                    <span>
                      Asked about the chart:{" "}
                      <span className="ask-turn__about-title">{turn.chartTitle}</span>
                    </span>
                  </p>
                ) : null}

                <p className="ask-turn__question">
                  <span className="ask-turn__q-label">You asked</span>
                  <span className="ask-turn__q-text">{turn.question}</span>
                </p>

                {turn.status === "pending" ? <AnswerSkeleton /> : null}

                {turn.status === "failed" ? (
                  <div className="ask-error" role="alert">
                    <p className="ask-error__title">That question did not get through.</p>
                    <p className="ask-error__detail">{turn.error}</p>
                    <button type="button" className="ask-error__retry" onClick={() => retry(turn)}>
                      Ask it again
                    </button>
                  </div>
                ) : null}

                {turn.status === "done" && turn.response ? (
                  <Answer
                    response={turn.response}
                    onAsk={submit}
                    onRetry={() => retry(turn)}
                  />
                ) : null}
              </article>
            ))}
          </div>

          {suggestions.length > 0 ? (
            <Suggestions
              questions={suggestions}
              folded={started}
              open={moreOpen}
              onToggle={() => setMoreOpen((v) => !v)}
              onPick={submit}
            />
          ) : null}
        </div>

        <form
          className="ask-compose"
          onSubmit={(e) => {
            e.preventDefault();
            submit(draft);
          }}
        >
          <label className="ask-compose__label" htmlFor={inputId}>
            Ask a question about {label}
          </label>
          <div className="ask-compose__field">
            <textarea
              id={inputId}
              ref={inputRef}
              className="ask-compose__input"
              rows={2}
              value={draft}
              placeholder="Which accounts are carrying the most stalled GP?"
              aria-describedby={hintId}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onInputKeyDown}
            />
            <button type="submit" className="ask-compose__send" disabled={draft.trim().length === 0}>
              Ask
            </button>
          </div>
          <p className="ask-compose__hint" id={hintId}>
            Enter to send · Shift + Enter for a new line
          </p>
        </form>
      </aside>
    </>
  );
}

/* ------------------------------------------------------------- suggestions */

interface SuggestionsProps {
  questions: string[];
  /** True once a question has been asked: the chips fold under a toggle. */
  folded: boolean;
  /** Whether a folded row is currently unfolded. Ignored while not folded. */
  open: boolean;
  onToggle: () => void;
  onPick: (question: string) => void;
}

/**
 * The pre-computed questions, each with a glyph chosen from the shape of the
 * question — a ranking, an amount, a where, a when, a why — so the row can be
 * read at a glance rather than word by word. Before anything has been asked
 * the row is open under "Try one of these"; afterwards it folds under "More
 * questions" so the transcript has the room, and it sits at the end of the
 * transcript, where the eye is after reading an answer.
 */
function Suggestions({ questions, folded, open, onToggle, onPick }: SuggestionsProps) {
  const listId = useId();
  const labelId = useId();
  const showList = !folded || open;

  return (
    <section className="ask-suggest" aria-labelledby={labelId}>
      {folded ? (
        <h3 className="ask-suggest__label" id={labelId}>
          <button
            type="button"
            className="ask-suggest__toggle"
            aria-expanded={open}
            aria-controls={listId}
            onClick={onToggle}
          >
            <span className="ask-suggest__toggle-caret" aria-hidden="true" />
            More questions
          </button>
        </h3>
      ) : (
        <h3 className="ask-suggest__label" id={labelId}>
          Try one of these
        </h3>
      )}
      {showList ? (
        <ul className="ask-suggest__list" id={listId}>
          {questions.map((q) => (
            <li key={q}>
              <SuggestionChip question={q} onPick={onPick} />
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function SuggestionChip({ question, onPick }: { question: string; onPick: (q: string) => void }) {
  return (
    <button type="button" className="ask-suggest__chip" onClick={() => onPick(question)}>
      <Glyph kind={glyphFor(question)} className="ask-suggest__glyph" />
      <span className="ask-suggest__text">{question}</span>
    </button>
  );
}

/* ------------------------------------------------------------------ answer */

interface AnswerProps {
  response: AskResponse;
  /** Ask one of the alternatives a declined answer offers. */
  onAsk: (question: string) => void;
  /** Ask the same question again, when no service was there to answer it. */
  onRetry: () => void;
}

function Answer({ response, onAsk, onRetry }: AnswerProps) {
  const { answer, chart, plan, chartWhy, refused } = response;
  // The server stamps these on the response; older payloads carried them on
  // the narrative. Either is read, so a cached answer is not misdescribed.
  const provider = response.provider ?? answer.provider;
  const degraded = !!response.degraded || !!answer.degraded;

  if (refused) return <Refusal response={response} onAsk={onAsk} />;

  // Degraded WITHOUT a plan means nothing was computed: no service was there
  // to plan the question, so there are no figures to write from. That is a
  // transient state worth a second try, not an answer.
  const unavailable = degraded && !plan && !chart;
  if (unavailable) {
    return (
      <div className="ask-answer ask-answer--quiet">
        <h4 className="ask-answer__headline">{answer.headline}</h4>
        {answer.sentences.map((s, i) => (
          <p className="ask-answer__plain" key={i}>
            {s.text}
          </p>
        ))}
        <button type="button" className="ask-error__retry" onClick={onRetry}>
          Ask it again
        </button>
        <Details response={response} />
      </div>
    );
  }

  // The computed template is INFORMATION, not a warning: the numbers are the
  // same either way, so it is said quietly and never in a warning colour.
  const computed = degraded || provider === "computed" || provider === "template";

  return (
    <div className="ask-answer">
      <h4 className="ask-answer__headline">{answer.headline}</h4>

      {answer.sentences.length > 0 ? (
        <ul className="ask-sentences">
          {answer.sentences.map((s, i) => {
            const meta = LENS_META[s.lens];
            return (
              <li className="ask-sentence" key={`${s.lens}-${i}`}>
                <span
                  className={`ask-sentence__lens ask-sentence__lens--${s.lens}`}
                  title={s.claim ? `${meta.why} Claim: ${s.claim}` : meta.why}
                >
                  {meta.label}
                </span>
                <span className="ask-sentence__text">{emphasise(s.text, s.bold)}</span>
              </li>
            );
          })}
        </ul>
      ) : null}

      {chart ? (
        <div className="ask-chart">
          <ChartCard spec={chart} height={ANSWER_CHART_HEIGHT} />
          {chartWhy ? (
            <p className="ask-chart__why">
              <span className="ask-chart__why-label">Why this chart</span>
              {chartWhy}
            </p>
          ) : null}
        </div>
      ) : null}

      {computed ? (
        <p className="ask-answer__quiet">
          Written from the computed figures. The numbers are the same either way.
        </p>
      ) : null}

      <Details response={response} />
    </div>
  );
}

/**
 * The calm decline. The server's sentence is the reason, printed as prose
 * rather than under a lens label, and its alternatives — when it offers any —
 * are the same chips as the suggested questions, so a dead end has a way out.
 */
function Refusal({ response, onAsk }: { response: AskResponse; onAsk: (q: string) => void }) {
  const { answer } = response;
  const next = alternatives(response).slice(0, REFUSAL_ALTERNATIVES);
  // The reason is the server's sentence. When it sent none, its note stands
  // in — printed here, so Details does not print it a second time.
  const reasons = answer.sentences.length > 0 ? answer.sentences.map((s) => s.text) : [];
  const noteAsReason = reasons.length === 0 && !!answer.reason;

  return (
    <div className="ask-answer ask-answer--refused">
      <h4 className="ask-answer__headline">The data can&rsquo;t answer that</h4>
      {reasons.map((text, i) => (
        <p className="ask-refusal__reason" key={i}>
          {text}
        </p>
      ))}
      {noteAsReason ? <p className="ask-refusal__reason">{answer.reason}</p> : null}
      {next.length > 0 ? (
        <div className="ask-refusal__try">
          <p className="ask-suggest__label">Try instead</p>
          <ul className="ask-suggest__list">
            {next.map((q) => (
              <li key={q}>
                <SuggestionChip question={q} onPick={onAsk} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <Details response={response} omitReason={noteAsReason} />
    </div>
  );
}

/**
 * One quiet disclosure for everything that proves the answer rather than
 * states it: which service wrote the prose and how long it took, the query
 * plan the server executed, and the sentences it refused to show. Rendered
 * only when there is something to disclose.
 */
function Details({ response, omitReason = false }: { response: AskResponse; omitReason?: boolean }) {
  const { answer, plan } = response;
  const provider = response.provider ?? answer.provider;
  const cached = !!response.cached || !!answer.cached;
  const dropped = answer.dropped ?? [];
  const rejected = answer.rejected ?? [];
  const suppressed = dropped.length + rejected.length;
  const reason = omitReason ? undefined : answer.reason;
  const hasLatency = typeof answer.latencyMs === "number";
  const hasTags = !!provider || !!answer.model || hasLatency || cached;
  const computed = provider === "computed" || provider === "template";
  const hasNote = suppressed > 0 || !!reason;

  if (!hasTags && !plan && !hasNote) return null;

  return (
    <details className="ask-disclosure ask-details">
      <summary className="ask-disclosure__summary">
        Details
        <span className="ask-disclosure__aside">how this answer was produced</span>
      </summary>
      <div className="ask-disclosure__body">
        {hasTags ? (
          <ul className="ask-details__tags" aria-label="Provenance">
            {provider ? <li className="ask-tag">{providerLabel(provider)}</li> : null}
            {answer.model ? <li className="ask-tag">{answer.model}</li> : null}
            {hasLatency ? <li className="ask-tag">{num(answer.latencyMs as number)} ms</li> : null}
            {cached ? <li className="ask-tag">cached</li> : null}
          </ul>
        ) : null}

        {reason ? <p className="ask-disclosure__note">{reason}</p> : null}

        {plan ? (
          <section className="ask-details__section">
            <h5 className="ask-details__title">The query the server ran</h5>
            <p className="ask-disclosure__note">
              {computed
                ? "The server chose this query for the question and ran it over the same data as the page, so every figure above is computed, not written."
                : "The model chose this plan. The server checked it against the semantic layer and ran it, so every figure above is the server\u2019s, not the model\u2019s."}
            </p>
            <pre className="ask-plan">
              <code>{JSON.stringify(plan, null, 2)}</code>
            </pre>
          </section>
        ) : null}

        {suppressed > 0 ? (
          <section className="ask-details__section">
            <h5 className="ask-details__title">
              What was not said ({num(suppressed)} sentence{suppressed === 1 ? "" : "s"} removed)
            </h5>

            {dropped.length > 0 ? (
              <section className="ask-provenance__group">
                <h6 className="ask-provenance__title">
                  Dropped as repeating a chart or out of scope ({num(dropped.length)})
                </h6>
                <ul className="ask-provenance__list">
                  {dropped.map((d, i) => (
                    <li className="ask-provenance__item" key={`d-${i}`}>
                      <span className="ask-provenance__text">{d.text}</span>
                      <span className="ask-provenance__reason">{d.reason}</span>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}

            {rejected.length > 0 ? (
              <section className="ask-provenance__group">
                <h6 className="ask-provenance__title">
                  Rejected as not backed by the figures ({num(rejected.length)})
                </h6>
                <ul className="ask-provenance__list">
                  {rejected.map((r, i) => (
                    <li className="ask-provenance__item" key={`r-${i}`}>
                      <span className="ask-provenance__text">{r.text}</span>
                      <span className="ask-provenance__reason">
                        not in the figures given to the model:{" "}
                        {r.rejectedTokens.map((t) => (
                          <code className="ask-provenance__token" key={t}>
                            {t}
                          </code>
                        ))}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
          </section>
        ) : null}
      </div>
    </details>
  );
}

/* ---------------------------------------------------------------- skeleton */

/** The shape of the answer that is coming, so nothing jumps when it lands. */
function AnswerSkeleton() {
  return (
    <div className="ask-skeleton">
      <div className="ask-skeleton__bars" aria-hidden="true">
        <div className="ask-skeleton__headline" />
        <div className="ask-skeleton__line" />
        <div className="ask-skeleton__line ask-skeleton__line--short" />
        <div className="ask-skeleton__chart" />
      </div>
      <p className="ask-skeleton__caption">Working it out from the data…</p>
    </div>
  );
}
