/**
 * The situation brief — the one place on a lens where prose is allowed.
 *
 * THE RULE THIS COMPONENT EXISTS TO ENFORCE, on screen rather than in a deck:
 * the AI does not restate a chart. Every chart ships `says: string[]`, the
 * server unions those claim keys and forbids the model from repeating any of
 * them, and what survives arrives here as `Sentence[]` — each sentence carrying
 * a `lens` that names WHY it is allowed to exist next to the charts. So the
 * lens is not metadata: it is the licence, and it is rendered as such. A chip
 * beside every sentence, and the chip explains itself when you open it.
 *
 * Three consequences follow, and all three are deliberate:
 *
 * 1. `bold` carries exact substrings of `text`, never markup. The API never
 *    ships HTML into this client, there is no `dangerouslySetInnerHTML` here,
 *    and the identical payload can be read by a surface with no DOM at all.
 *    A bold run the text does not contain is skipped in silence.
 *
 * 2. The provenance footer names the provider, the latency and whether the
 *    answer came out of a cache. When the model was not reached, `degraded`
 *    is true and the footer says the computed template wrote this — stated
 *    plainly, not as an incident. The numbers are the page's own either way.
 *
 * 3. Sentences the server threw away are NOT hidden. `dropped` (redundant with
 *    a chart) and `rejected` (contained a figure the server never supplied) sit
 *    behind one disclosure. Being able to read what the product refused to say
 *    is the whole reason to believe what it does say.
 *
 * This component computes nothing. Every figure inside it was formatted by the
 * server; the only thing formatted here is the latency and the as-of date, both
 * through lib/format.
 */
import { Fragment, useId, useState, type ReactNode } from "react";
import type { Narrative, Sentence } from "../api/types";
import { num } from "../lib/format";

type LensKey = Sentence["lens"];

/**
 * What each lens means, and — the part that matters — why a sentence wearing it
 * is not a duplicate of the chart above it. Written for the reader, not for us:
 * the label is the plain words a reader would use ("Why", "What changed"),
 * never the lens's wire name, which is jargon from the grounding contract.
 * The explanation sits in the label's title and opens on click.
 */
const LENS_META: Record<LensKey, { label: string; tip: string }> = {
  cause: {
    label: "Why",
    tip: "Why it moved. The charts plot the movement; none of them can attribute it. This sentence names the driver behind a shape you can already see.",
  },
  norm: {
    label: "Compared with",
    tip: "What normal looks like here. A bar is only high or low against a baseline, and the baseline lives in history the chart does not plot.",
  },
  delta: {
    label: "What changed",
    tip: "The change against the prior period or the benchmark. Two states compared — the page shows one of them, so the comparison is not on screen.",
  },
  action: {
    label: "What to do",
    tip: "What to do next, and to what. A chart never carries an instruction; this sentence is the handover from reading to doing.",
  },
  answer: {
    label: "Answer",
    tip: "A direct reply to the question this page asks, in a sentence. The evidence is below it; this is the verdict the evidence supports.",
  },
  state: {
    label: "Where it stands",
    tip: "A plain statement of where things stand. Permitted only when no chart on this screen already makes the same claim — otherwise the server drops it.",
  },
};

/**
 * Split `text` into runs, emboldening every declared substring.
 *
 * Substring matching, never markup parsing: the ranges are collected, sorted
 * and merged so that overlapping or repeated `bold` entries cannot nest a <b>
 * inside a <b> or drop a character of the original text. What is rendered is
 * always exactly `text`, only partly heavier.
 */
function emphasise(text: string, bold?: string[]): ReactNode[] {
  if (!bold || bold.length === 0) return [text];

  const hits: { start: number; end: number }[] = [];
  for (const b of bold) {
    if (!b) continue;
    let from = 0;
    for (;;) {
      const i = text.indexOf(b, from);
      if (i < 0) break;
      hits.push({ start: i, end: i + b.length });
      from = i + b.length;
    }
  }
  if (hits.length === 0) return [text];

  hits.sort((a, b) => a.start - b.start || b.end - a.end);
  const merged: { start: number; end: number }[] = [];
  for (const h of hits) {
    const last = merged[merged.length - 1];
    if (last && h.start <= last.end) last.end = Math.max(last.end, h.end);
    else merged.push({ ...h });
  }

  const out: ReactNode[] = [];
  let cursor = 0;
  let k = 0;
  for (const r of merged) {
    if (r.start > cursor) out.push(<Fragment key={k++}>{text.slice(cursor, r.start)}</Fragment>);
    out.push(<b key={k++}>{text.slice(r.start, r.end)}</b>);
    cursor = r.end;
  }
  if (cursor < text.length) out.push(<Fragment key={k++}>{text.slice(cursor)}</Fragment>);
  return out;
}

export interface AiPanelProps {
  /** The brief. Null while it is in flight, or after it failed. */
  narrative: Narrative | null;
  /** True while the brief is being fetched. */
  loading?: boolean;
  /** A human-readable failure, already described by `useView.describe`. */
  error?: string | null;
  /** Refetch. Omit it and the error state simply states what failed. */
  onRetry?: () => void;
  /** The eyebrow above the headline. */
  label?: string;
  /** The slice this brief was written about, e.g. `payload.scope.label`. */
  scope?: string;
  /** `payload.asOf`. The business date always arrives on the payload. */
  asOf?: string;
}

export function AiPanel({
  narrative,
  loading = false,
  error = null,
  onRetry,
  label = "Situation brief",
  scope,
  asOf: _asOf,
}: AiPanelProps) {
  const uid = useId();
  // Which sentence has its lens explanation open. One at a time: the chips are
  // an aside, and three open asides are longer than the brief they annotate.
  const [openLens, setOpenLens] = useState<number | null>(null);
  const [showFiltered, setShowFiltered] = useState(false);

  // The skeleton only replaces a panel that has nothing in it. While the
  // generated brief is being written, the page hands in the computed brief it
  // already carries, so this branch is not taken; that brief stays painted,
  // marked busy, and the generated one replaces it in place when it lands.
  // Flashing a skeleton over prose the reader is mid-sentence in is worse than
  // a computed line for a second or two.
  if (loading && !narrative && !error) {
    return <AiPanelSkeleton label={label} scope={scope} />;
  }

  if (error && !narrative) {
    return (
      <section className="ai-panel ai-panel--error" role="region" aria-label={label}>
        <div className="ai-panel__head">
          <span className="ai-panel__glyph" aria-hidden="true">
            ✦
          </span>
          <span className="ai-panel__label">{label}</span>
        </div>
        <div className="ai-panel__error" role="alert">
          <p className="ai-panel__error-what">
            The situation brief did not load. The page below it is complete and correct — the
            charts, the KPIs and the actions come from the semantic layer, not from the model.
          </p>
          <p className="ai-panel__error-why">{error}</p>
          {onRetry ? (
            <button type="button" className="ai-panel__retry" onClick={onRetry}>
              Try the brief again
            </button>
          ) : null}
        </div>
      </section>
    );
  }

  if (!narrative) return null;

  const sentences = narrative.sentences ?? [];
  const dropped = narrative.dropped ?? [];
  const rejected = narrative.rejected ?? [];
  const filteredCount = dropped.length + rejected.length;
  const filteredId = `${uid}-filtered`;

  // Who wrote this, and nothing else. The model id, the latency and the as-of
  // date were all here too — the date for a third time on the page — which made
  // a two-word provenance note into a row of debug output.
  const prov: string[] = [];
  if (!narrative.degraded && narrative.provider && narrative.provider !== "computed") {
    prov.push(`written by ${narrative.provider}`);
  }
  if (narrative.cached) prov.push("cached");

  return (
    <section className="ai-panel" role="region" aria-label={label}>
      <div className="ai-panel__head">
        <span className="ai-panel__glyph" aria-hidden="true">
          ✦
        </span>
        <span className="ai-panel__label">{label}</span>
        {scope ? <span className="ai-panel__scope">{scope}</span> : null}
        {/* Always in the row, faded in and out, so the head never reflows
            when a refresh starts or ends. The body's aria-busy carries the
            state to a screen reader; this is the sighted reader's copy. */}
        <span
          className={`ai-panel__refreshing${loading ? " ai-panel__refreshing--on" : ""}`}
          aria-hidden={!loading}
        >
          Refreshing…
        </span>
      </div>

      {/* The brief replaces itself when the scope changes, so it is announced
          rather than silently swapped under a screen reader. */}
      <div className="ai-panel__body" aria-live="polite" aria-busy={loading}>
        <h2 className="ai-panel__headline">{narrative.headline}</h2>

        {sentences.length === 0 ? (
          <p className="ai-panel__empty">
            Nothing to add here. Every claim worth making about this view is already on a chart
            below, so the brief stayed quiet rather than repeating one.
          </p>
        ) : (
          <ul className="ai-panel__list">
            {sentences.map((s, i) => {
              const meta = LENS_META[s.lens];
              const open = openLens === i;
              const tipId = `${uid}-tip-${i}`;
              return (
                <li className="ai-sentence" key={`${s.lens}-${i}`}>
                  <div className="ai-sentence__row">
                    <button
                      type="button"
                      className="ai-sentence__lens"
                      data-lens={s.lens}
                      aria-expanded={open}
                      aria-controls={tipId}
                      aria-label={`${meta.label} — what this label means`}
                      title={meta.tip}
                      onClick={() => setOpenLens(open ? null : i)}
                    >
                      {meta.label}
                    </button>
                    <p className="ai-sentence__text">{emphasise(s.text, s.bold)}</p>
                  </div>
                  <p
                    className="ai-sentence__tip"
                    id={tipId}
                    role="note"
                    hidden={!open}
                  >
                    {meta.tip}
                    {s.claim ? (
                      <span className="ai-sentence__claim">
                        grounded on <code>{s.claim}</code>
                      </span>
                    ) : null}
                  </p>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <footer className="ai-panel__foot">
        {/* One short sentence. The paragraph that used to sit here explained the
            fallback at length and then printed the raw provider attempt list —
            an HTTP stack trace on a leadership screen. The server now hands back
            a plain reason; the attempt list stays in the payload for debugging
            and never reaches the page. */}
        {narrative.degraded ? (
          <p className="ai-panel__degraded">
            Written from the computed figures{narrative.reason ? ` — ${narrative.reason}` : ""}.
            The numbers are the same either way.
          </p>
        ) : null}

        {prov.length > 0 ? (
          <p className="ai-panel__prov">
            {prov.map((p, i) => (
              <Fragment key={p}>
                {i > 0 ? (
                  <span className="ai-panel__dot" aria-hidden="true">
                    ·
                  </span>
                ) : null}
                <span className="ai-panel__prov-item">{p}</span>
              </Fragment>
            ))}
          </p>
        ) : null}

        {filteredCount > 0 ? (
          <div className="ai-panel__filtered">
            <button
              type="button"
              className="ai-panel__filter-toggle"
              aria-expanded={showFiltered}
              aria-controls={filteredId}
              onClick={() => setShowFiltered((v) => !v)}
            >
              <span className="ai-panel__chevron" aria-hidden="true">
                {showFiltered ? "▾" : "▸"}
              </span>
              What was filtered out ({num(filteredCount)})
            </button>

            <div className="ai-panel__filter-body" id={filteredId} hidden={!showFiltered}>
              <p className="ai-panel__filter-intro">
                These sentences were generated and then removed before this panel was rendered. Each
                one names the rule that removed it.
              </p>

              {dropped.length > 0 ? (
                <div className="ai-panel__filter-group">
                  <h3 className="ai-panel__filter-grouphead">
                    Redundant with a chart
                    <span className="ai-panel__filter-count">{num(dropped.length)}</span>
                  </h3>
                  <ul className="ai-panel__filter-list">
                    {dropped.map((d, i) => (
                      <li className="ai-panel__filter-item" key={`d-${i}`}>
                        <q className="ai-panel__filter-quote">{d.text}</q>
                        <span className="ai-panel__filter-reason">{d.reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {rejected.length > 0 ? (
                <div className="ai-panel__filter-group">
                  <h3 className="ai-panel__filter-grouphead">
                    Carried a figure the server never supplied
                    <span className="ai-panel__filter-count">{num(rejected.length)}</span>
                  </h3>
                  <ul className="ai-panel__filter-list">
                    {rejected.map((r, i) => (
                      <li className="ai-panel__filter-item" key={`r-${i}`}>
                        <q className="ai-panel__filter-quote">{r.text}</q>
                        <span className="ai-panel__filter-reason">
                          ungrounded:
                          {r.rejectedTokens.map((t) => (
                            <code className="ai-panel__token" key={t}>
                              {t}
                            </code>
                          ))}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </footer>
    </section>
  );
}

/**
 * The loading state, shaped like the thing it is standing in for: eyebrow,
 * headline, three chip-and-sentence rows, footer. A skeleton whose geometry
 * does not match the arrival causes a reflow that reads as a bug, so this one
 * is built out of the same elements rather than out of grey boxes.
 */
function AiPanelSkeleton({ label, scope }: { label: string; scope?: string }) {
  return (
    <section
      className="ai-panel ai-panel--loading"
      role="region"
      aria-label={label}
      aria-busy="true"
    >
      <div className="ai-panel__head">
        <span className="ai-panel__glyph" aria-hidden="true">
          ✦
        </span>
        <span className="ai-panel__label">{label}</span>
        {scope ? <span className="ai-panel__scope">{scope}</span> : null}
      </div>

      <p className="ai-panel__sr" aria-live="polite">
        Writing the situation brief. The charts and figures on this page are already final.
      </p>

      <div className="ai-panel__sk" aria-hidden="true">
        <div className="ai-panel__sk-head" />
        {[0, 1, 2].map((i) => (
          <div className="ai-panel__sk-row" key={i}>
            <div className="ai-panel__sk-chip" />
            <div className="ai-panel__sk-lines">
              <div className="ai-panel__sk-line" />
              <div className="ai-panel__sk-line ai-panel__sk-line--short" />
            </div>
          </div>
        ))}
        <div className="ai-panel__sk-foot" />
      </div>
    </section>
  );
}
