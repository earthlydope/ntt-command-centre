/**
 * The glyphs the Ask surfaces share.
 *
 * One set for the per-chart Ask, the main Ask AI Expert panel and the floating
 * button, so a "why" question looks the same wherever it is offered. The
 * drawings live in `icons.tsx` with the rest of the shell's marks — same
 * 24-unit grid, same 1.75 stroke — so a question chip and a sidebar item are
 * recognisably from one hand. This file keeps the question-shape vocabulary
 * (rank, value, place, time, cause, compare, act, spark) and the rule that
 * picks one from a sentence.
 */
import { IconPaths, type IconName } from "./icons";

export type GlyphKey =
  | "rank"
  | "value"
  | "place"
  | "time"
  | "cause"
  | "compare"
  | "act"
  | "spark";

/** Question shape to drawing. Every key maps onto the shell's own set. */
const ICON_FOR: Record<GlyphKey, IconName> = {
  rank: "rank",
  value: "value",
  place: "place",
  time: "time",
  cause: "cause",
  compare: "compare",
  act: "act",
  spark: "spark",
};

/**
 * Question shape to glyph. Order matters: "why did this month fall" is a why,
 * not a when; "who should I coach first" is a what-to-do, not a who; "are we
 * on track against plan this quarter" is a comparison, not a date; and "how
 * much closes this quarter" is an amount, not a date.
 */
const GLYPH_RULES: { kind: GlyphKey; test: RegExp }[] = [
  { kind: "cause", test: /\b(why|cause[sd]?|causing|driving|drove|behind|reason)\b/ },
  {
    kind: "act",
    test: /\b(what should i|should i|what to do|do first|first today|next step|coach|fix|fixing)\b/,
  },
  {
    kind: "compare",
    test: /\b(compares?|compared|comparison|versus|vs\.?|against|vary|varies|differs?|on track)\b/,
  },
  { kind: "value", test: /\b(how much|how many|how big|how large|worth|value|size)\b/ },
  { kind: "place", test: /\b(where|which region|which country|country|region)\b/ },
  {
    kind: "time",
    test: /\b(when|month|monthly|quarter|quarterly|year|week|trend|over time|since|ageing|aging)\b/,
  },
  {
    kind: "rank",
    test: /\b(which|who|whose|top|most|least|biggest|largest|smallest|best|worst|highest|lowest|rank)\b/,
  },
];

export function glyphFor(question: string): GlyphKey {
  const q = question.toLowerCase();
  for (const rule of GLYPH_RULES) if (rule.test.test(q)) return rule.kind;
  return "spark";
}

/**
 * The Ask surfaces size their glyphs from their own stylesheets, which is why
 * this draws its own element rather than rendering `Icon`: `size` writes width
 * and height on the element itself, for a glyph whose stylesheet is not this
 * component's to control (the floating button). It is a presentation
 * attribute, so any CSS rule on the class still wins; without it an inline
 * SVG with only a viewBox fills its container.
 */
export function Glyph({
  kind,
  className,
  size,
}: {
  kind: GlyphKey;
  className: string;
  size?: number;
}) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
    >
      <IconPaths name={ICON_FOR[kind]} />
    </svg>
  );
}
