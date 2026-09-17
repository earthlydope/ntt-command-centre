/**
 * The card a chart lives in: the title, the stated grain, the SVG the D3 module
 * owns, and a footer that names which repository module was selected and why.
 *
 * **The footer is not decoration.** The client asked for a repository of charts
 * chosen by the shape of the data; printing `shape → repositoryKey` on every
 * card is how that claim is checked on screen rather than asserted in a deck.
 * The server resolves the key and stamps it on the spec, the client re-runs the
 * same rule, and a disagreement shows as "(fallback)" instead of being hidden.
 *
 * **The grain is printed too.** Stage counts opportunities, LOB counts lines,
 * and a reader comparing two cards has to be able to see which is which — the
 * same query at two grains gives two different right answers.
 *
 * **Ask is opt-in, by passing `ask`.** A card on the page gets one; the card
 * that renders an ANSWER's chart inside the Ask panel does not, because a chart
 * you can ask about inside the answer to a question about a chart is a hall of
 * mirrors, not a feature.
 *
 * **The footnote is a caveat about the marks above it**, not small print: "the
 * red tail is time already overrun", "these months have no closed history yet".
 * Dropping it is how a green pill ends up on a quarter nobody should trust.
 */
import { useRef, useState } from "react";
import type { ChartSpec } from "../api/types";
import type { AskSeed } from "../state/filters";
import { useChart } from "../charts/useChart";
import { useApp } from "../state/AppStateProvider";
import { ChartAsk, ChartAskButton } from "./ChartAsk";

/** Does this payload have anything to draw? Both wire shapes are checked. */
function isEmpty(data: ChartSpec["data"]): boolean {
  if (Array.isArray(data)) return data.length === 0;
  if (!data || typeof data !== "object") return true;
  const arrays = Object.values(data).filter(Array.isArray) as unknown[][];
  return arrays.length > 0 && arrays.every((a) => a.length === 0);
}

export interface ChartCardProps {
  spec: ChartSpec;
  height?: number;
  /**
   * Give the card its own Ask. `chartsSay` is what the whole page already
   * claims — the server refuses to answer by repeating it — and `onExpand`
   * hands the question up to the main Ask panel.
   */
  ask?: { chartsSay: string[]; onExpand: (question: string, seed: AskSeed) => void };
}

export function ChartCard({ spec, height, ask }: ChartCardProps) {
  const { onFilter, state } = useApp();
  const [askOpen, setAskOpen] = useState(false);
  // Focus goes back to the trigger on close. Without this, dismissing the
  // panel drops the caret at the top of the document and a keyboard user has
  // to tab through the whole page to get back to where they were.
  const askButtonRef = useRef<HTMLButtonElement | null>(null);
  // A chart the server knows can be legitimately empty says so in its own
  // words, and the module is never mounted — a generic "No rows in this slice"
  // over a true finding ("every account in your book already buys all four
  // lines of business") is the product failing to say the useful thing.
  const empty = !!spec.emptyMessage && isEmpty(spec.data);
  const { wrapRef, svgRef, resolution } = useChart({
    spec: empty ? { ...spec, data: [] } : spec,
    onFilter,
    selected: state.filters,
    height: height ?? spec.height ?? undefined,
  });

  const basis =
    spec.countBasis === "opportunities"
      ? "counted on distinct opportunities"
      : spec.countBasis === "lines"
        ? "counted on line items"
        : null;

  return (
    <section className="card">
      <header className="card-head">
        <div className="card-head__text">
          <h3>{spec.title}</h3>
          {spec.subtitle && <p className="csub">{spec.subtitle}</p>}
        </div>
        {ask && !empty ? (
          <ChartAskButton
            onClick={() => setAskOpen(true)}
            open={askOpen}
            chartTitle={spec.title}
            buttonRef={askButtonRef}
          />
        ) : null}
      </header>

      {ask && !empty ? (
        <ChartAsk
          spec={spec}
          chartsSay={ask.chartsSay}
          open={askOpen}
          onExpand={(question, seed) => {
            setAskOpen(false);
            ask.onExpand(question, seed);
          }}
          onClose={() => {
            setAskOpen(false);
            askButtonRef.current?.focus();
          }}
        />
      ) : null}

      {empty ? (
        <p className="card-empty">{spec.emptyMessage}</p>
      ) : (
        <div ref={wrapRef} className="chartwrap">
          <svg ref={svgRef} role="img" aria-label={`${spec.title}. ${spec.subtitle}`} />
        </div>
      )}

      {spec.footnote && !empty && <p className="cnote">{spec.footnote}</p>}

      {/* The provenance line — which repository module was selected, on what
          grain, and what a click does — is a claim worth being able to CHECK,
          not a caption worth reading on every card. It moves behind a disclosure
          so the page stays legible and the proof stays available. */}
      <details className="cmeta">
        <summary className="cmeta__toggle">
          {basis ?? "About this chart"}
          {spec.clickDim ? " · click to filter" : ""}
        </summary>
        <div className="cmeta__body">
          <p>
            Shape <code>{spec.shape}</code> resolved to <code>{resolution.key}</code>
            {resolution.degraded ? " (fallback)" : ""}.
          </p>
          {spec.basisNote && <p>{spec.basisNote}</p>}
          {spec.clickDim && <p>Clicking a mark filters {spec.clickDim}.</p>}
        </div>
      </details>
    </section>
  );
}
