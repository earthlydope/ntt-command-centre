/**
 * gantt.timeline — serves `entity×start×end`.
 * One open deal per row, from its own start to its own close date, against a
 * calendar. The question it answers is the one an AE actually asks: which of my
 * deals has already run out of road?
 *
 * THE AS-OF RULE IS THE CHART. Every other Gantt decoration is optional; the
 * vertical line at `asOf` is not, because "past due" is a statement about a bar
 * and a date, and a reader who cannot see the date cannot check the claim. It
 * is drawn on top of the rows, in `p.text` — the same token `bullet.target`
 * uses for its target marker and `combo.columnline` for its plan line, because
 * in this app a reference marker is text-coloured and never a RAG hue.
 * `AS_OF` arrives on the payload (2026-09-15 on this dataset) and is never
 * computed here: a chart that reads the clock disagrees with the page around it
 * the moment the page is left open overnight.
 *
 * PAST DUE IS DRAWN AS OVERSHOOT, NOT AS A COLOURED ROW. The part of a
 * `pastDue` bar that lies beyond the as-of rule is painted in `p.danger` and
 * the part before it keeps its normal fill, so the red is literally the time
 * the deal has overrun — length, not decoration. The server may send either of
 * the two honest encodings of that fact and both are handled: a bar whose `end`
 * runs past `asOf` (the tail beyond the rule is red), and a bar whose `end` is
 * already behind `asOf` (the run from its own close date up to the rule is
 * red — that IS the overshoot, and 256 of the 350 open lines on this dataset
 * are in exactly that state). A one-pixel separator in the surface colour marks
 * where the deal's own close date sits, so the two segments can never be read
 * as one span.
 *
 * STALLED IS A TEXTURE, NOT A HUE. 113 open opportunities have had no logged
 * change in 60+ days, and staleness is orthogonal to lateness — a deal can be
 * either, both or neither. Hue is already carrying `tone` and the past-due
 * overshoot, so "stalled" is a diagonal hatch laid over whatever the bar
 * already is. The hatch is an SVG `<pattern>`, defined once per render with an
 * id unique to that render: a lens can show two of these cards at once and a
 * fixed id means both documents define `#gantt-hatch`, every `url(#gantt-hatch)`
 * on the page resolves to whichever node the browser finds first, and the
 * surviving chart's bars silently lose (or inherit) a texture when the other
 * one re-renders or is torn down. React StrictMode's double-invoke makes that a
 * certainty rather than a risk. The same applies to the scroll clip path.
 *
 * ORDER COMES FROM THE SERVER. The rows are drawn in the order they arrive and
 * are never re-sorted here — the ranking is a business rule (`api/semantic/`),
 * and a client that re-sorts turns "the worst deal is at the top" into a lie
 * the moment the two rules disagree.
 *
 * Rows are 22px, so a 30-deal list does not fit the card. The rows scroll
 * inside a clipped viewport while the axis, the gridlines and the as-of rule
 * stay pinned — an axis that scrolls away from its bars is worse than no axis.
 */
import { scaleTime } from "d3-scale";
import { select } from "d3-selection";
import type { Selection } from "d3-selection";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec, Tone } from "../api/types";
import type { Palette } from "../theme/palette";
import { toneColor } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  charsIn,
  createTooltip,
  formatValue,
  selectionState,
  textWidth,
  tipHtml,
  truncate,
  truncateLabel,
} from "./util";
import { longDate, monthLabel } from "../lib/format";

/** A row. Tight enough that a quarter of pipeline fits a card, tall enough that
 *  a 14px bar and a two-line label at the data and note sizes still read. */
const ROW = 28;
const BAR_H = 14;
/** The pinned header: tick labels plus the as-of chip. */
const HEAD = 36;
/** The legend strip under the plot. */
const FOOT = 24;
const PAD_R = 6;
/** The scrollbar gutter, taken out of the plot only when the list overflows. */
const SCROLL_W = 9;
/** Beyond this many marks the group takes one tab stop. the palette-as-parameter rule. */
const DENSE = 20;
/** A viewport shorter than this is not a chart, it is a peephole. */
const MIN_VIEW_ROWS = 3;
const DAY = 86_400_000;

/** Per-render id counter — see the `<pattern>` note in the header comment. */
let SEQ = 0;

type Format = "currency" | "percent" | "number" | "days";

// ---------------------------------------------------------------- the payload
// docs/CHART_CONTRACT.md §"gantt.timeline", field for field. Declared here rather than
// imported because `api/types.ts` is shared and this module does not own it.

interface GanttBar {
  id: string;
  label: string;
  sublabel?: string;
  start: string;
  end: string;
  value: number;
  tone?: Tone;
  pastDue?: boolean;
  stalled?: boolean;
  quietDays?: number;
}

interface GanttPayload {
  bars: GanttBar[];
  asOf: string;
  rangeStart: string;
  rangeEnd: string;
}

/**
 * The spec as this module needs to see it. `ChartSpec.data` in `api/types.ts`
 * is still typed as a row array — every shape that shipped first was one — so
 * the object payload is read through a structural view instead of by widening a
 * shared wire type from inside a chart module. Three wire spellings are
 * accepted so this module is correct whichever way the seam is closed:
 * `data` as the contract object, `payload` as the contract object, or a flat
 * `data` array of bars with the three dates carried on the spec itself.
 */
interface GanttSpecView {
  data?: unknown;
  payload?: unknown;
  asOf?: unknown;
  rangeStart?: unknown;
  rangeEnd?: unknown;
  measureLabel?: string;
  format?: Format;
}

const TONES: Tone[] = ["accent", "good", "warn", "danger", "neutral"];

function toneOf(raw: unknown): Tone | undefined {
  return typeof raw === "string" && (TONES as string[]).includes(raw) ? (raw as Tone) : undefined;
}

/**
 * ISO in, Date out, `null` for anything that is not a date. A date-only string
 * is pinned to UTC midnight: `lib/format` reads dates with `getUTC*`, so a
 * local-midnight parse here would print the day before for anyone west of
 * Greenwich, and the as-of rule would land on the wrong side of a bar.
 */
function parseISO(raw: unknown): Date | null {
  if (raw instanceof Date) return Number.isFinite(+raw) ? raw : null;
  const s = String(raw ?? "").trim();
  if (!s) return null;
  const t = Date.parse(/^\d{4}-\d{2}-\d{2}$/.test(s) ? `${s}T00:00:00Z` : s);
  return Number.isFinite(t) ? new Date(t) : null;
}

function readBars(list: unknown): GanttBar[] {
  if (!Array.isArray(list)) return [];
  const out: GanttBar[] = [];
  list.forEach((row, i) => {
    if (!row || typeof row !== "object") return;
    const d = row as Record<string, unknown>;
    const id = String(d.id ?? "").trim() || `row-${i}`;
    out.push({
      id,
      label: String(d.label ?? id),
      sublabel:
        d.sublabel === null || d.sublabel === undefined || String(d.sublabel).trim() === ""
          ? undefined
          : String(d.sublabel),
      start: String(d.start ?? ""),
      end: String(d.end ?? ""),
      value: Number(d.value),
      tone: toneOf(d.tone),
      pastDue: d.pastDue === true,
      stalled: d.stalled === true,
      quietDays:
        d.quietDays === null || d.quietDays === undefined ? undefined : Number(d.quietDays),
    });
  });
  return out;
}

interface Payload {
  bars: GanttBar[];
  asOf: Date | null;
  rangeStart: Date | null;
  rangeEnd: Date | null;
}

function readPayload(spec: ChartSpec): Payload {
  const v = spec as unknown as GanttSpecView;
  const objs = [v.data, v.payload].filter(
    (o): o is Record<string, unknown> => !!o && typeof o === "object" && !Array.isArray(o),
  );
  const obj = objs.find((o) => Array.isArray((o as Partial<GanttPayload>).bars)) ?? objs[0] ?? {};
  const bars = readBars(
    Array.isArray((obj as Partial<GanttPayload>).bars)
      ? (obj as Partial<GanttPayload>).bars
      : Array.isArray(v.data)
        ? v.data
        : [],
  );
  const pick = (key: "asOf" | "rangeStart" | "rangeEnd"): Date | null =>
    parseISO((obj as Record<string, unknown>)[key]) ?? parseISO(v[key]);
  return { bars, asOf: pick("asOf"), rangeStart: pick("rangeStart"), rangeEnd: pick("rangeEnd") };
}

// ---------------------------------------------------------------- formatting
// docs/CHART_CONTRACT.md: `format: "days"` is a bare integer with a `d` suffix. The
// shared formatters in `lib/format` predate that case, so the suffix is added
// here and every other format still goes through them. Nothing is reimplemented.

function fmtValue(v: number, f?: Format): string {
  if (!Number.isFinite(v)) return "—";
  return f === "days" ? `${Math.round(v)}d` : formatValue(v, f);
}

/** A day-grain tick label. `longDate` is the one date formatter in the app; the
 *  year is dropped because every tick on one axis shares it. */
function dayLabel(d: Date): string {
  return longDate(d.toISOString()).replace(/\s\d{4}$/, "");
}

/** UTC month starts inside [a, b], used when the window is wide enough that a
 *  day-grain axis would be noise. */
function monthStarts(a: Date, b: Date): Date[] {
  const out: Date[] = [];
  const cur = new Date(Date.UTC(a.getUTCFullYear(), a.getUTCMonth(), 1));
  if (+cur < +a) cur.setUTCMonth(cur.getUTCMonth() + 1);
  while (+cur <= +b && out.length < 64) {
    out.push(new Date(+cur));
    cur.setUTCMonth(cur.getUTCMonth() + 1);
  }
  return out;
}

interface Row {
  bar: GanttBar;
  t0: Date;
  t1: Date;
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  const view = spec as unknown as GanttSpecView;
  const fmt: Format | undefined = view.format ?? spec.format;
  const measureLabel =
    typeof view.measureLabel === "string" && view.measureLabel ? view.measureLabel : "Value";
  const width = Math.max(260, opts.width);

  /** Degrade, never throw. the chart repository contract rule 7. */
  const centred = (lines: string[]): Teardown => {
    svg
      .attr("viewBox", `0 0 ${width} 96`)
      .attr("width", "100%")
      .attr("height", 96)
      .attr("role", "group")
      .attr("aria-label", `${spec.title}. ${lines.join(". ")}`);
    lines.forEach((t, i) =>
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 44 + i * 16)
        .attr("text-anchor", "middle")
        .attr("fill", p.muted)
        .attr("font-size", i === 0 ? FONT.title : FONT.tick)
        .text(t),
    );
    return () => svg.selectAll("*").remove();
  };

  const pay = readPayload(spec);
  const t0 = pay.rangeStart;
  const t1 = pay.rangeEnd;
  // A Gantt with no window is not a Gantt with a default window. Say so.
  if (!t0 || !t1 || +t1 <= +t0) {
    return centred([
      "This timeline has no usable date range",
      "rangeStart and rangeEnd must be ISO dates, and the end must follow the start",
    ]);
  }
  if (!pay.bars.length) return centred(["No open rows in this slice"]);

  // A bar without two parsable dates cannot be placed. It is dropped and
  // counted, and the count is printed — a silently missing row reads as a
  // shorter list, which is the one thing this chart must not say.
  const rows: Row[] = [];
  let dropped = 0;
  for (const bar of pay.bars) {
    const a = parseISO(bar.start);
    const b = parseISO(bar.end);
    if (!a || !b || +b < +a) {
      dropped += 1;
      continue;
    }
    rows.push({ bar, t0: a, t1: b });
  }
  if (!rows.length) {
    return centred([
      "No rows in this slice with usable start and end dates",
      `${dropped} row${dropped === 1 ? "" : "s"} arrived without a parsable start and end`,
    ]);
  }
  // NOT sorted here. The server's order is the ranking. See the header comment.

  // ------------------------------------------------------------- the layout
  // Height is derived from `opts.height`, never fixed: the chart repository contract rule 6.
  const budget = Math.max(ROW * MIN_VIEW_ROWS, (opts.height || 240) - HEAD - FOOT - 6);
  const viewRows = Math.max(MIN_VIEW_ROWS, Math.floor(budget / ROW));
  const scrolls = rows.length > viewRows;
  const plotH = (scrolls ? viewRows : rows.length) * ROW;
  const contentH = rows.length * ROW;
  const height = HEAD + plotH + FOOT;

  const gutter = scrolls ? SCROLL_W : 0;
  const valueW = width >= 420 ? textWidth(9, FONT.label) + 8 : 0;
  const right = width - PAD_R - gutter - valueW;
  // The label column yields to the plot, never the other way round: a Gantt
  // whose bars are 40px wide is a list with decoration.
  const labelW = Math.max(56, Math.min(264, Math.max(112, width * 0.28), right - 96));
  const plotW = Math.max(60, right - labelW);
  const px0 = right - plotW;
  const px1 = right;

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr(
      "aria-label",
      `${spec.title}. ${rows.length} rows from ${longDate(t0.toISOString())} to ${longDate(
        t1.toISOString(),
      )}${pay.asOf ? `, as of ${longDate(pay.asOf.toISOString())}` : ""}. ${
        rows.filter((r) => r.bar.pastDue).length
      } past due.`,
    );

  const x = scaleTime().domain([t0, t1]).range([px0, px1]);
  /** Bars are clamped into the window: a row that starts before it is drawn
   *  from the edge, and the tooltip still reports its real dates. */
  const cx = (d: Date) => Math.max(px0, Math.min(px1, x(d)));

  // ------------------------------------------------------- per-render ids
  // Two of these cards on one lens would otherwise share `#gantt-hatch` and
  // `#gantt-clip`, and every `url(#…)` in the document resolves to the first
  // matching node — so one chart would paint with the other's pattern and keep
  // painting with it after that chart was torn down. StrictMode guarantees the
  // collision rather than merely risking it. Unique per render, not per module.
  const uid = ++SEQ;
  const hatchId = `gantt-hatch-${uid}`;
  const clipId = `gantt-clip-${uid}`;

  const defs = svg.append("defs");
  const hatch = defs
    .append("pattern")
    .attr("id", hatchId)
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 6)
    .attr("height", 6)
    .attr("patternTransform", "rotate(45)");
  // Surface-coloured strokes, so the hatch reads as a texture cut out of the
  // bar in either theme and the bar's own hue survives underneath it.
  hatch
    .append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 0)
    .attr("y2", 6)
    .attr("stroke", p.surface)
    .attr("stroke-width", 2.6)
    .attr("stroke-opacity", 0.75);
  defs
    .append("clipPath")
    .attr("id", clipId)
    .append("rect")
    .attr("x", 0)
    .attr("y", HEAD)
    .attr("width", width)
    .attr("height", plotH);

  const tip = createTooltip(p);

  // ------------------------------------------------ the axis (pinned, on top)
  // Month grain past a quarter or so; day grain below that. Either way the
  // ticks are vertical, so they do not move when the rows scroll and they are
  // drawn outside the clipped viewport.
  const spanDays = (+t1 - +t0) / DAY;
  const slots = Math.max(2, Math.floor(plotW / 74));
  const ticks: Date[] =
    spanDays > 75
      ? (() => {
          const all = monthStarts(t0, t1);
          const step = Math.max(1, Math.ceil(all.length / slots));
          return all.filter((_, i) => i % step === 0);
        })()
      : x.ticks(slots);
  const tickLabel = (d: Date) => (spanDays > 75 ? monthLabel(d.toISOString()) : dayLabel(d));

  const axis = svg.append("g").attr("aria-hidden", "true");
  ticks.forEach((d) => {
    const gx = x(d);
    axis
      .append("line")
      .attr("x1", gx)
      .attr("x2", gx)
      .attr("y1", HEAD - 8)
      .attr("y2", HEAD + plotH)
      .attr("stroke", p.grid)
      .attr("stroke-width", 1);
    axis
      .append("text")
      .attr("x", gx)
      .attr("y", HEAD - 12)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(tickLabel(d));
  });
  // The plot's own frame, so an empty right-hand side still reads as calendar.
  axis
    .append("line")
    .attr("x1", px0)
    .attr("x2", px1)
    .attr("y1", HEAD + plotH)
    .attr("y2", HEAD + plotH)
    .attr("stroke", p.line)
    .attr("stroke-width", 1);

  // --------------------------------------------------------- the rows
  const viewport = svg.append("g").attr("clip-path", `url(#${clipId})`);
  // A transparent floor, so the wheel is caught over the whitespace between
  // bars and not only over a mark.
  viewport
    .append("rect")
    .attr("x", 0)
    .attr("y", HEAD)
    .attr("width", width)
    .attr("height", plotH)
    .attr("fill", "transparent")
    .attr("aria-hidden", "true");
  const scroller = viewport.append("g");

  const basisWord = spec.countBasis === "opportunities" ? "opportunities" : "lines";
  const asOfX = pay.asOf && +pay.asOf >= +t0 && +pay.asOf <= +t1 ? x(pay.asOf) : null;
  const hits: SVGGraphicsElement[] = [];
  let overdueShown = 0;
  let stalledShown = 0;

  rows.forEach((r, i) => {
    const b = r.bar;
    const top = HEAD + i * ROW;
    const by = top + (ROW - BAR_H) / 2;
    const g = scroller.append("g");

    // Zebra banding: at this pitch a 30-row list needs the eye anchored to a row.
    if (i % 2 === 1) {
      g.append("rect")
        .attr("x", 0)
        .attr("y", top)
        .attr("width", width)
        .attr("height", ROW)
        .attr("fill", p.surface2)
        .attr("aria-hidden", "true");
    }

    const bx0 = cx(r.t0);
    const bx1 = cx(r.t1);
    // The overshoot. Both wire encodings of "past due" land here: a tail that
    // runs beyond the rule, or a close date already behind it — in which case
    // the red run is from that close date up to the rule.
    let mainX1 = bx1;
    let dangerFrom = 0;
    let dangerTo = 0;
    if (b.pastDue && asOfX !== null) {
      if (bx1 > asOfX) {
        mainX1 = asOfX;
        dangerFrom = asOfX;
        dangerTo = bx1;
      } else {
        dangerFrom = bx1;
        dangerTo = asOfX;
      }
    }
    const hasDanger = dangerTo - dangerFrom > 0.5;
    if (hasDanger) overdueShown += 1;
    if (b.stalled) stalledShown += 1;
    const runEnd = Math.max(bx1, hasDanger ? dangerTo : bx1);
    const base = b.tone ? toneColor(p, b.tone) : p.accent;

    // Selection: outline plus dimming of the rest, never a hue change.
    // the one-helper interaction rule. `GanttBar` carries no dimension value of its own, so
    // a row is a filter only when the server both names a `clickDim` and sends
    // a `sublabel` to carry that dimension's value — clicking a deal name would
    // set a filter matching nothing, which is why `treemap.nested` leaves its
    // "Other" tile inert for the same reason.
    const filterValue = spec.clickDim && b.sublabel ? b.sublabel : undefined;
    const { anySelected, isSelected } = selectionState(opts, spec.clickDim, filterValue ?? b.id);
    const dimmed = filterValue ? anySelected && !isSelected : anySelected;
    g.attr("opacity", dimmed ? DIM_OPACITY : 1);

    const bar = g
      .append("rect")
      .attr("x", bx0)
      .attr("y", by)
      .attr("width", Math.max(2, mainX1 - bx0))
      .attr("height", BAR_H)
      .attr("rx", 3)
      .attr("fill", base);

    if (hasDanger) {
      g.append("rect")
        .attr("x", dangerFrom)
        .attr("y", by)
        .attr("width", Math.max(2, dangerTo - dangerFrom))
        .attr("height", BAR_H)
        .attr("rx", 3)
        .attr("fill", p.danger);
      // Where the deal's own close date sits. Without it the two segments read
      // as one span and the overshoot loses its anchor.
      g.append("line")
        .attr("x1", bx1)
        .attr("x2", bx1)
        .attr("y1", by - 1)
        .attr("y2", by + BAR_H + 1)
        .attr("stroke", p.surface)
        .attr("stroke-width", 1.25);
    }

    if (b.stalled) {
      g.append("rect")
        .attr("x", bx0)
        .attr("y", by)
        .attr("width", Math.max(2, runEnd - bx0))
        .attr("height", BAR_H)
        .attr("rx", 3)
        .attr("fill", `url(#${hatchId})`)
        .attr("pointer-events", "none");
    }

    if (isSelected && filterValue) {
      g.append("rect")
        .attr("x", bx0 - 2.5)
        .attr("y", by - 2.5)
        .attr("width", Math.max(2, runEnd - bx0) + 5)
        .attr("height", BAR_H + 5)
        .attr("rx", 5)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5);
    }

    // ------------------------------------------------------------ the label
    if (b.sublabel) {
      g.append("text")
        .attr("x", 2)
        .attr("y", top + 12)
        .attr("fill", p.text)
        .attr("font-size", FONT.label)
        .attr("font-weight", isSelected ? 700 : 500)
        .text(truncateLabel(b.label, labelW - 8));
      g.append("text")
        .attr("x", 2)
        .attr("y", top + 24)
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .text(truncateLabel(b.sublabel, labelW - 8, FONT.note));
    } else {
      g.append("text")
        .attr("x", 2)
        .attr("y", top + ROW / 2 + 4.5)
        .attr("fill", p.text)
        .attr("font-size", FONT.label)
        .attr("font-weight", isSelected ? 700 : 500)
        .text(truncateLabel(b.label, labelW - 8));
    }

    if (valueW) {
      g.append("text")
        .attr("x", width - PAD_R - gutter)
        .attr("y", top + ROW / 2 + 4.5)
        .attr("text-anchor", "end")
        .attr("fill", p.text2)
        .attr("font-size", FONT.label)
        .attr("font-weight", 700)
        .attr("font-variant-numeric", "tabular-nums")
        .text(fmtValue(b.value, fmt));
    }

    // ---------------------------------------------------------- the mark
    const spanD = Math.max(0, Math.round((+r.t1 - +r.t0) / DAY));
    const overdueD = pay.asOf ? Math.round((+pay.asOf - +r.t1) / DAY) : 0;
    const tipRows: [string, string][] = [
      [measureLabel, fmtValue(b.value, fmt)],
      ["Window", `${longDate(r.t0.toISOString())} → ${longDate(r.t1.toISOString())}`],
      ["Span", `${spanD}d`],
    ];
    if (b.pastDue && overdueD > 0) tipRows.push(["Past its close date", `${overdueD}d ago`]);
    if (b.pastDue && overdueD <= 0)
      tipRows.push(["Beyond the as-of date", `${Math.abs(overdueD)}d`]);
    if (b.stalled)
      tipRows.push([
        "No logged change",
        b.quietDays !== undefined && Number.isFinite(b.quietDays)
          ? `${Math.round(b.quietDays)}d`
          : "60d+",
      ]);
    const foot = [
      b.sublabel,
      filterValue ? `Click to filter ${spec.clickDim} = ${filterValue}` : undefined,
    ]
      .filter(Boolean)
      .join(" · ");

    const hit = g
      .append("rect")
      .attr("x", 0)
      .attr("y", top)
      .attr("width", width)
      .attr("height", ROW)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, b.label, tipRows, foot || undefined),
      aria: `${b.label}${b.sublabel ? `, ${b.sublabel}` : ""}, ${fmtValue(b.value, fmt)}, ${longDate(
        r.t0.toISOString(),
      )} to ${longDate(r.t1.toISOString())}${
        b.pastDue ? `, past due by ${Math.max(0, overdueD)} days` : ""
      }${b.stalled ? ", no logged change" : ""}`,
      dim: filterValue ? spec.clickDim : undefined,
      value: filterValue,
      opts,
      onEnter: () => {
        g.attr("opacity", 1);
        bar.attr("stroke", p.text).attr("stroke-width", 1);
      },
      onLeave: () => {
        g.attr("opacity", dimmed ? DIM_OPACITY : 1);
        bar.attr("stroke", "none");
      },
    });
    hits.push(hit);
  });

  // ------------------------------------------------------- the as-of rule
  // Drawn after the rows and outside the scrolled group: it is a property of
  // the calendar, not of any row, so it stays put and stays on top. Inert to
  // the pointer, so it never steals a hover from the bar underneath it.
  if (asOfX !== null && pay.asOf) {
    const rule = svg.append("g").attr("pointer-events", "none").attr("aria-hidden", "true");
    rule
      .append("line")
      .attr("x1", asOfX)
      .attr("x2", asOfX)
      .attr("y1", HEAD - 8)
      .attr("y2", HEAD + plotH)
      .attr("stroke", p.text)
      .attr("stroke-width", 1.5);
    const text = `as of ${longDate(pay.asOf.toISOString())}`;
    const tw = textWidth(text.length, FONT.note) + 12;
    const flip = asOfX + tw + 4 > width - PAD_R;
    const chipX = flip ? asOfX - tw - 4 : asOfX + 4;
    rule
      .append("rect")
      .attr("x", chipX)
      .attr("y", 2)
      .attr("width", tw)
      .attr("height", 17)
      .attr("rx", 4)
      .attr("fill", p.surface)
      .attr("stroke", p.line)
      .attr("stroke-width", 1);
    rule
      .append("text")
      .attr("x", chipX + tw / 2)
      .attr("y", 14.5)
      .attr("text-anchor", "middle")
      .attr("fill", p.text)
      .attr("font-size", FONT.note)
      .attr("font-weight", 700)
      .text(text);
  }

  // ----------------------------------------------------------- the scroll
  // The viewport scrolls, the axis does not. `util.ts` has no scroller helper
  // and every shared file is owned elsewhere, so the handling lives here and
  // only here. Every listener sits on a node inside this subtree — including
  // the thumb's pointer capture — so dropping the subtree drops them all and
  // the teardown stays a one-liner that is safe to call twice.
  const maxScroll = Math.max(0, contentH - plotH);
  let scrollY = 0;
  const thumbH = maxScroll ? Math.max(20, (plotH * plotH) / contentH) : 0;
  // Typed as the concrete <rect> selection rather than `ReturnType<typeof
  // svg.append>`: that alias resolves to a BaseType selection, which is not
  // assignable from the SVGRectElement selection `track.append("rect")` returns.
  let thumb: Selection<SVGRectElement, unknown, null, undefined> | null = null;

  const paint = () => {
    scroller.attr("transform", `translate(0,${-scrollY})`);
    if (thumb && maxScroll) {
      thumb.attr("y", HEAD + (scrollY / maxScroll) * (plotH - thumbH));
    }
  };
  const setScroll = (v: number): boolean => {
    const next = Math.max(0, Math.min(maxScroll, v));
    if (Math.abs(next - scrollY) < 0.01) return false;
    scrollY = next;
    paint();
    return true;
  };
  /** viewBox units per CSS pixel. The card scales the SVG to its own width, so
   *  a wheel delta in pixels is not a delta in plot units. */
  const unitPerPx = (): number => {
    const r = root.getBoundingClientRect();
    return r.width > 0 ? width / r.width : 1;
  };

  if (maxScroll > 0) {
    const track = svg.append("g").attr("aria-hidden", "true");
    track
      .append("rect")
      .attr("x", width - SCROLL_W + 1)
      .attr("y", HEAD)
      .attr("width", 5)
      .attr("height", plotH)
      .attr("rx", 2.5)
      .attr("fill", p.track);
    thumb = track
      .append("rect")
      .attr("x", width - SCROLL_W + 1)
      .attr("y", HEAD)
      .attr("width", 5)
      .attr("height", thumbH)
      .attr("rx", 2.5)
      .attr("fill", p.muted)
      .attr("opacity", 0.75)
      .attr("cursor", "grab");
    paint();

    const viewNode = viewport.node();
    if (viewNode) {
      // Non-passive, because the page must not scroll while the list still
      // has somewhere to go — and must scroll normally once it does not.
      viewNode.addEventListener(
        "wheel",
        (ev: WheelEvent) => {
          const px =
            ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaMode === 2 ? ev.deltaY * plotH : ev.deltaY;
          if (setScroll(scrollY + px * unitPerPx())) ev.preventDefault();
        },
        { passive: false },
      );
    }

    const thumbNode = thumb?.node() ?? null;
    if (thumbNode) {
      let from = 0;
      let at = 0;
      thumbNode.addEventListener("pointerdown", (ev: PointerEvent) => {
        ev.preventDefault();
        from = ev.clientY;
        at = scrollY;
        thumbNode.setPointerCapture(ev.pointerId);
        thumbNode.setAttribute("cursor", "grabbing");
      });
      thumbNode.addEventListener("pointermove", (ev: PointerEvent) => {
        if (!thumbNode.hasPointerCapture(ev.pointerId)) return;
        const travel = Math.max(1, plotH - thumbH);
        setScroll(at + ((ev.clientY - from) * unitPerPx() * maxScroll) / travel);
      });
      const release = (ev: PointerEvent) => {
        if (thumbNode.hasPointerCapture(ev.pointerId)) thumbNode.releasePointerCapture(ev.pointerId);
        thumbNode.setAttribute("cursor", "grab");
      };
      thumbNode.addEventListener("pointerup", release);
      thumbNode.addEventListener("pointercancel", release);
    }
  }

  /** Keyboard navigation must move the viewport, not just the focus ring: a
   *  focused row clipped out of sight is a focus ring nobody can see. */
  const reveal = (i: number) => {
    const top = i * ROW;
    if (top < scrollY) setScroll(top);
    else if (top + ROW > scrollY + plotH) setScroll(top + ROW - plotH);
  };

  // ------------------------------------------------------- keyboard, rule 2
  // 30 open deals is 30 tab stops, and a keyboard user cannot get past the
  // card. `util.ts` has no `markGroup()` yet, so the roving tabindex lives
  // here: the group holds exactly one focusable row and the arrows move which.
  const dense = hits.length > DENSE;
  let cur = Math.max(
    0,
    rows.findIndex((r) => {
      const v = spec.clickDim && r.bar.sublabel ? r.bar.sublabel : null;
      return v !== null && selectionState(opts, spec.clickDim, v).isSelected;
    }),
  );
  const setRoving = (i: number) => {
    if (!dense) return;
    hits.forEach((node, j) => node.setAttribute("tabindex", j === i ? "0" : "-1"));
  };
  setRoving(cur);
  reveal(cur);
  hits.forEach((node, i) => {
    node.addEventListener("focus", () => {
      cur = i;
      setRoving(i);
    });
    node.addEventListener("keydown", (ev: KeyboardEvent) => {
      const step =
        ev.key === "ArrowDown" || ev.key === "ArrowRight"
          ? 1
          : ev.key === "ArrowUp" || ev.key === "ArrowLeft"
            ? -1
            : 0;
      let next = -1;
      if (step !== 0) next = Math.max(0, Math.min(hits.length - 1, i + step));
      else if (ev.key === "Home") next = 0;
      else if (ev.key === "End") next = hits.length - 1;
      else if (ev.key === "PageDown") next = Math.min(hits.length - 1, i + Math.max(1, viewRows - 1));
      else if (ev.key === "PageUp") next = Math.max(0, i - Math.max(1, viewRows - 1));
      if (next < 0) return;
      ev.preventDefault();
      // Scroll first, focus second: `attachMark` positions its tooltip from the
      // node's client rect, and a node moved after focus is a tooltip adrift.
      reveal(next);
      setRoving(next);
      hits[next].focus();
    });
  });

  // ------------------------------------------------------------ the legend
  // The two textures are the whole vocabulary of this chart, so they are named
  // on the card rather than in a tooltip nobody opens.
  const legend = svg.append("g").attr("transform", `translate(0,${HEAD + plotH + 16})`);
  let lx = 2;
  const swatch = (fill: string, hatched: boolean, text: string) => {
    legend
      .append("rect")
      .attr("x", lx)
      .attr("y", -7)
      .attr("width", 14)
      .attr("height", 8)
      .attr("rx", 2)
      .attr("fill", fill);
    if (hatched) {
      legend
        .append("rect")
        .attr("x", lx)
        .attr("y", -7)
        .attr("width", 14)
        .attr("height", 8)
        .attr("rx", 2)
        .attr("fill", `url(#${hatchId})`);
    }
    legend
      .append("text")
      .attr("x", lx + 18)
      .attr("y", 0)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(text);
    lx += 18 + textWidth(text.length, FONT.note) + 14;
  };
  if (width >= 380) {
    if (overdueShown) swatch(p.danger, false, "past its close date");
    if (stalledShown) swatch(p.accent, true, "no logged change");
  }
  const tail = [
    scrolls ? `${viewRows} of ${rows.length} rows — scroll or use ↑↓` : `${rows.length} rows`,
    `${basisWord} as ranked by the server`,
    dropped ? `${dropped} row${dropped === 1 ? "" : "s"} without usable dates not drawn` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  legend
    .append("text")
    .attr("x", width - PAD_R)
    .attr("y", 0)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .text(truncate(tail, Math.max(10, charsIn(width - lx - 10, FONT.note))));

  return () => {
    tip.destroy();
    // Every listener this module added — attachMark's, the wheel, the thumb's
    // pointer capture and the arrow keys — sits on a node inside this subtree,
    // so dropping the subtree drops them. Safe to call twice: the second call
    // removes nothing.
    svg.selectAll("*").remove();
  };
}

export const ganttTimeline: ChartModule = {
  key: "gantt.timeline",
  // `entity×start×end` is not a member of the `Shape` union in `api/types.ts`
  // yet — that file is generated from real payloads and the registry wiring
  // lands with the orchestrator, not from inside a chart module. The cast is
  // that seam, and it becomes a plain literal the moment the union is widened.
  serves: "entity×start×end" as unknown as ChartModule["serves"],
  render,
  // The same viewport arithmetic as render(): the plot shows as many rows as
  // the offer holds and scrolls the rest, so the drawn height tracks the offer
  // up to the list's own length.
  drawnHeight: (spec, offered) => {
    const n = readPayload(spec).bars.length;
    if (!n) return 96;
    const budget = Math.max(ROW * MIN_VIEW_ROWS, offered - HEAD - FOOT - 6);
    const viewRows = Math.max(MIN_VIEW_ROWS, Math.floor(budget / ROW));
    return HEAD + Math.min(n, viewRows) * ROW + FOOT;
  },
};
