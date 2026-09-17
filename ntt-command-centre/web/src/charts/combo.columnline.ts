/**
 * combo.columnline — serves `temporal×measure×measure`.
 * Won GP per month as columns, against the monthly GP plan as a line.
 *
 * TWO MEASURES, TWO AXES, SCALED INDEPENDENTLY. The left axis belongs to the
 * columns and carries `spec.format`; the right axis belongs to the line and
 * carries `lineFormat`. Each is captioned with its own measure name, because
 * hanging two units off one axis is the exact lie this chart type is usually
 * used to tell — a $46k column "crossing" a 94% line means nothing, and a
 * reader who cannot see which axis a series is read against cannot know that.
 * The gridlines are therefore drawn from the LEFT scale alone and the right
 * axis gets tick text only: a shared grid would imply a shared unit.
 *
 * The line is drawn in `p.text`, the same token `bullet.target` uses for its
 * target marker, and with straight segments rather than a smoothed curve — a
 * plan is a reference series, not a hue in the RAG vocabulary, and it does not
 * interpolate between months. It is painted over the columns on a surface-
 * coloured halo so it stays legible where it crosses a saturated bar.
 *
 * ONE MARK PER MONTH, not two. A month is a single fact with two measures, so
 * the column, the line's point marker and the whitespace of that slot share one
 * hit area whose tooltip reports both figures — otherwise a keyboard user pays
 * two tab stops per month for one datum. Past ~20 months the marks collapse to
 * a single tab stop with arrow keys, the palette-as-parameter rule; `util.ts` has no
 * `markGroup()` yet, so that handling is local to this file.
 *
 * `ComboPoint.tone` shades its own column and nothing else infers a tone: a
 * column whose value the server did not grade stays accent rather than being
 * secretly graded here against the line. `ComboPoint.annotation` renders as a
 * caret and a line of text below the axis — on this dataset, Oct–Dec carrying
 * "no closed history yet, calendar artefact", which is why those columns are
 * short and why that is not a collapse.
 */
import { max, min } from "d3-array";
import { scaleBand, scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import type { Selection } from "d3-selection";
import { line as d3line } from "d3-shape";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette, Tone } from "../theme/palette";
import { toneColor } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  charsIn,
  createTooltip,
  formatTick,
  formatValue,
  measureTickWidth,
  selectionState,
  textWidth,
  tipHtml,
  truncate,
  truncateLabel,
} from "./util";
import { monthLabel } from "../lib/format";

/** Beyond this many marks the group takes one tab stop. the palette-as-parameter rule. */
const DENSE = 20;
/** A column wider than this reads as a slab, not a column. */
const MAX_BAR = 56;

type Format = "currency" | "percent" | "number" | "days";

// ---------------------------------------------------------------- the payload
// CONTRACT "Payload types — exact", field for field. Declared here rather than
// imported because `api/types.ts` is shared and this module does not own it.

interface ComboPoint {
  key: string;
  bar: number;
  line: number;
  label?: string;
  tone?: Tone;
  annotation?: string;
}

interface ComboPayload {
  points: ComboPoint[];
  barLabel: string;
  lineLabel: string;
  lineFormat: Format;
}

/**
 * The spec as this module needs to see it. `ChartSpec` in `api/types.ts` is
 * still typed for the first six repositories — `data` + `encoding` — so the
 * combo payload is read through a structural view instead of by editing a
 * shared file. Both wire shapes are accepted: `payload` when the server sends
 * the contract object, flat rows on `data` when it does not.
 */
interface ComboSpecView {
  payload?: Partial<ComboPayload> | null;
  data?: unknown;
  encoding?: { x?: string; y?: string; series?: string; target?: string; label?: string };
  measureLabel?: string;
}

/** One column, already resolved. `null` is an absence, never a zero: a month
 *  with no closed history yet is not a month that closed nothing. */
interface Col {
  key: string;
  label: string;
  bar: number | null;
  line: number | null;
  tone?: Tone;
  annotation?: string;
}

const ISO = /^\d{4}-\d{2}(-\d{2})?$/;
const TONES: Tone[] = ["good", "warn", "danger", "neutral", "accent"];

function numberOrNull(raw: unknown): number | null {
  if (raw === null || raw === undefined || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function toneOf(raw: unknown): Tone | undefined {
  return typeof raw === "string" && (TONES as string[]).includes(raw) ? (raw as Tone) : undefined;
}

function readPayload(spec: ChartSpec): { cols: Col[]; barLabel: string; lineLabel: string; lineFormat: Format } {
  const v = spec as unknown as ComboSpecView;
  const enc = v.encoding ?? {};
  const kKey = enc.x ?? "key";
  const bKey = enc.y ?? "bar";
  const lKey = enc.series ?? enc.target ?? "line";

  // docs/CHART_CONTRACT.md §"combo.columnline" puts `{points, barLabel, lineLabel,
  // lineFormat}` on `data`; `payload` is accepted as an alias, and a flat row
  // array on `data` still works.
  const obj = (v.data && !Array.isArray(v.data) ? v.data : v.payload) as
    | { points?: unknown; barLabel?: string; lineLabel?: string; lineFormat?: Format }
    | undefined;
  const rows: Record<string, unknown>[] = Array.isArray(obj?.points)
    ? (obj?.points as Record<string, unknown>[])
    : Array.isArray(v.data)
      ? (v.data as Record<string, unknown>[])
      : [];

  const cols: Col[] = rows
    .map((r) => {
      const key = String(r.key ?? r[kKey] ?? "");
      const label = typeof r.label === "string" && r.label ? r.label : "";
      return {
        key,
        label:
          label ||
          (ISO.test(key) && !Number.isNaN(Date.parse(key)) ? monthLabel(key) : key),
        bar: numberOrNull(r.bar ?? r[bKey]),
        line: numberOrNull(r.line ?? r[lKey]),
        tone: toneOf(r.tone),
        annotation: typeof r.annotation === "string" && r.annotation ? r.annotation : undefined,
      };
    })
    // A row with no key cannot be placed and a row with neither measure has
    // nothing to say; both are dropped rather than drawn at the origin.
    .filter((c) => c.key !== "" && (c.bar !== null || c.line !== null));

  return {
    cols,
    barLabel: obj?.barLabel ?? v.payload?.barLabel ?? v.measureLabel ?? "Actual",
    lineLabel: obj?.lineLabel ?? v.payload?.lineLabel ?? "Plan",
    lineFormat:
      obj?.lineFormat ?? v.payload?.lineFormat ?? (spec.format as Format | undefined) ?? "currency",
  };
}

// ------------------------------------------------------------- number shaping
// docs/CHART_CONTRACT.md: `format: "days"` is a bare integer with a `d` suffix. The
// shared formatters in `lib/format` predate that case, so the suffix is added
// here and every other format still goes through them. Nothing is reimplemented.

function fmtValue(v: number, f?: Format): string {
  return f === "days" ? `${Math.round(v)}d` : formatValue(v, f);
}

function fmtTick(v: number, f?: Format): string {
  return f === "days" ? `${Math.round(v)}d` : formatTick(v, f);
}

function tickKind(f?: Format): "currency" | "percent" | "number" | undefined {
  return f === "days" ? "number" : f;
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  const { cols, barLabel, lineLabel, lineFormat } = readPayload(spec);
  const barFormat: Format = spec.format ?? "currency";

  // Annotations sit below the axis, so they buy their own band rather than
  // stealing it from the plot. Adjacent points carrying the SAME text are one
  // run with one caption — Oct, Nov and Dec share a calendar artefact, they do
  // not have three of them.
  const runs: { text: string; from: number; to: number }[] = [];
  cols.forEach((c, i) => {
    if (!c.annotation) return;
    const last = runs[runs.length - 1];
    if (last && last.text === c.annotation && last.to === i - 1) last.to = i;
    else runs.push({ text: c.annotation, from: i, to: i });
  });

  const width = Math.max(240, opts.width);
  const annoH = runs.length ? 30 : 0;
  const height = Math.max(200, opts.height || 240) + annoH;

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr(
      "aria-label",
      `${spec.title}. ${cols.length} periods. Columns are ${barLabel} on the left axis; the line is ${lineLabel} on the right axis.` +
        (cols.length > DENSE ? " Use the arrow keys to move between periods." : ""),
    );

  if (!cols.length) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text("No periods in this slice");
    return () => svg.selectAll("*").remove();
  }

  // Two domains, computed apart. Each keeps zero in range so a negative month
  // reads as a drop below the baseline rather than as a short column.
  const barLo = Math.min(0, min(cols, (c) => c.bar ?? Infinity) ?? 0);
  const barHi = Math.max(0, max(cols, (c) => c.bar ?? -Infinity) ?? 0);
  const lineLo = Math.min(0, min(cols, (c) => c.line ?? Infinity) ?? 0);
  const lineHi = Math.max(0, max(cols, (c) => c.line ?? -Infinity) ?? 0);

  const yBarProbe = scaleLinear().domain([barLo, barHi || 1]).nice();
  const yLineProbe = scaleLinear().domain([lineLo, lineHi || 1]).nice();
  const m = {
    top: 26,
    bottom: 26 + annoH,
    left: measureTickWidth(yBarProbe.ticks(4), tickKind(barFormat)),
    right: measureTickWidth(yLineProbe.ticks(4), tickKind(lineFormat)),
  };
  const plotBottom = height - m.bottom;

  const yBar = yBarProbe.range([plotBottom, m.top]);
  const yLine = yLineProbe.range([plotBottom, m.top]);
  const x = scaleBand<string>()
    .domain(cols.map((c) => c.key))
    .range([m.left, width - m.right])
    .paddingInner(0.34)
    .paddingOuter(0.18);

  const tip = createTooltip(p);
  const cleanups: (() => void)[] = [];

  // --------------------------------------------------- left axis: the columns
  // Gridlines come off this scale only. A grid drawn for both would say the two
  // units share a ruler.
  const barTicks = yBar.ticks(4);
  const grid = svg.append("g");
  grid
    .selectAll("line")
    .data(barTicks)
    .join("line")
    .attr("x1", m.left)
    .attr("x2", width - m.right)
    .attr("y1", (d) => yBar(d))
    .attr("y2", (d) => yBar(d))
    .attr("stroke", p.grid);
  grid
    .selectAll("text")
    .data(barTicks)
    .join("text")
    .attr("x", m.left - 8)
    .attr("y", (d) => yBar(d) + 4)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((d) => fmtTick(d, barFormat));

  // ----------------------------------------------------- right axis: the line
  const lineTicks = yLine.ticks(4);
  const rightAxis = svg.append("g");
  rightAxis
    .selectAll("line")
    .data(lineTicks)
    .join("line")
    .attr("x1", width - m.right + 1)
    .attr("x2", width - m.right + 5)
    .attr("y1", (d) => yLine(d))
    .attr("y2", (d) => yLine(d))
    .attr("stroke", p.line);
  rightAxis
    .selectAll("text")
    .data(lineTicks)
    .join("text")
    .attr("x", width - m.right + 9)
    .attr("y", (d) => yLine(d) + 4)
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((d) => fmtTick(d, lineFormat));

  // The zero rule of the column scale, drawn only when the columns cross it.
  if (barLo < 0) {
    svg
      .append("line")
      .attr("x1", m.left)
      .attr("x2", width - m.right)
      .attr("y1", yBar(0))
      .attr("y2", yBar(0))
      .attr("stroke", p.line)
      .attr("stroke-width", 1.4);
  }

  // ------------------------------------------ axis captions = the legend, too
  // Each caption sits over its own axis with its own swatch, so "which axis is
  // this series read against" is answered without a lookup.
  // Each caption may take just under half the width, swatch included.
  const capChars = Math.max(6, charsIn((width - 60) / 2, FONT.note));
  const cap = svg.append("g").attr("aria-hidden", "true");
  cap
    .append("rect")
    .attr("x", 2)
    .attr("y", 3.5)
    .attr("width", 8)
    .attr("height", 8)
    .attr("rx", 2)
    .attr("fill", p.accent);
  cap
    .append("text")
    .attr("x", 14)
    .attr("y", 11.5)
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .attr("font-weight", 650)
    .attr("letter-spacing", "0.04em")
    .text(`${truncate(barLabel, capChars)} (left)`);

  const rightCap = `${truncate(lineLabel, capChars)} (right)`;
  const rightCapW = textWidth(rightCap.length, FONT.note);
  cap
    .append("line")
    .attr("x1", width - 4 - rightCapW - 16)
    .attr("x2", width - 4 - rightCapW - 4)
    .attr("y1", 7.5)
    .attr("y2", 7.5)
    .attr("stroke", p.text)
    .attr("stroke-width", 2);
  cap
    .append("circle")
    .attr("cx", width - 4 - rightCapW - 10)
    .attr("cy", 7.5)
    .attr("r", 2.6)
    .attr("fill", p.surface)
    .attr("stroke", p.text)
    .attr("stroke-width", 1.6);
  cap
    .append("text")
    .attr("x", width - 2)
    .attr("y", 11.5)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .attr("font-weight", 650)
    .attr("letter-spacing", "0.04em")
    .text(rightCap);

  // ------------------------------------------------------------- the columns
  const bw = Math.min(x.bandwidth(), MAX_BAR);
  const cx = (c: Col) => (x(c.key) ?? 0) + x.bandwidth() / 2;
  const barsG = svg.append("g");
  const bars: (Selection<SVGRectElement, unknown, null, undefined> | null)[] = [];
  const state = cols.map((c) => selectionState(opts, spec.clickDim, c.key));

  cols.forEach((c, i) => {
    const dimmed = state[i].anySelected && !state[i].isSelected;
    const g = barsG.append("g").attr("opacity", dimmed ? DIM_OPACITY : 1);
    if (c.bar === null || c.bar === 0) {
      // No column at all. A hairline zero would read as "measured, and it was
      // nothing"; an absent month has not been measured.
      bars.push(null);
      return;
    }
    const y0 = yBar(0);
    const yv = yBar(c.bar);
    const rect = g
      .append("rect")
      .attr("x", (x(c.key) ?? 0) + (x.bandwidth() - bw) / 2)
      .attr("y", Math.min(y0, yv))
      .attr("width", bw)
      .attr("height", Math.max(1, Math.abs(yv - y0)))
      .attr("rx", 3)
      .attr("fill", c.tone ? toneColor(p, c.tone) : p.accent);
    bars.push(rect);
  });

  // ----------------------------------------------------------- the line above
  const withLine = cols.filter((c) => c.line !== null);
  if (withLine.length > 1) {
    const path = d3line<Col>()
      .x((c) => cx(c))
      .y((c) => yLine(c.line as number));
    const d = path(withLine) ?? "";
    // halo first, so the plan stays readable where it crosses a column
    svg
      .append("path")
      .attr("d", d)
      .attr("fill", "none")
      .attr("stroke", p.surface)
      .attr("stroke-width", 4.5)
      .attr("stroke-linejoin", "round")
      .attr("opacity", 0.85)
      .attr("aria-hidden", "true");
    svg
      .append("path")
      .attr("d", d)
      .attr("fill", "none")
      .attr("stroke", p.text)
      .attr("stroke-width", 2)
      .attr("stroke-linejoin", "round")
      .attr("aria-hidden", "true");
  }

  const dotsG = svg.append("g").attr("aria-hidden", "true");
  const dots = cols.map((c, i) => {
    if (c.line === null) return null;
    const dimmed = state[i].anySelected && !state[i].isSelected;
    return dotsG
      .append("circle")
      .attr("cx", cx(c))
      .attr("cy", yLine(c.line))
      .attr("r", 3.2)
      .attr("fill", p.surface)
      .attr("stroke", p.text)
      .attr("stroke-width", 2)
      .attr("opacity", dimmed ? DIM_OPACITY : 1);
  });

  // ------------------------------------------------------------- the x labels
  const labStep = Math.max(
    1,
    Math.ceil(
      cols.length / Math.max(1, Math.floor((width - m.left - m.right) / textWidth(7, FONT.tick))),
    ),
  );
  svg
    .append("g")
    .selectAll("text")
    .data(cols.filter((_, i) => i % labStep === 0 || i === cols.length - 1))
    .join("text")
    .attr("x", (c) => cx(c))
    .attr("y", plotBottom + 17)
    .attr("text-anchor", "middle")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((c) => truncateLabel(c.label, x.step() * labStep, FONT.tick));

  // ----------------------------------------------------------- the annotations
  // A caret per annotated column, one caption per run, full text in the tooltip.
  if (runs.length) {
    const anno = svg.append("g").attr("aria-hidden", "true");
    runs.forEach((r) => {
      for (let i = r.from; i <= r.to; i++) {
        const k = cx(cols[i]);
        anno
          .append("path")
          .attr("d", `M${k - 4},${height - annoH + 8} L${k},${height - annoH + 1} L${k + 4},${height - annoH + 8} Z`)
          .attr("fill", p.muted);
      }
      const from = x(cols[r.from].key) ?? 0;
      const to = (x(cols[r.to].key) ?? 0) + x.bandwidth();
      const est = Math.max(8, charsIn(to - from + 40, FONT.note));
      const text = truncate(r.text, est);
      const half = textWidth(text.length, FONT.note) / 2;
      anno
        .append("text")
        .attr("x", Math.min(Math.max((from + to) / 2, half + 2), width - half - 2))
        .attr("y", height - annoH + 21)
        .attr("text-anchor", "middle")
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .text(text);
    });
  }

  // ------------------------------------------------------------- the marks
  // One per period: the column, its plan marker and the slot's whitespace are a
  // single fact and therefore a single hit area.
  const guide = svg
    .append("line")
    .attr("stroke", p.muted)
    .attr("stroke-dasharray", "3 3")
    .attr("y1", m.top)
    .attr("y2", plotBottom)
    .attr("opacity", 0)
    .attr("aria-hidden", "true");

  const hitsG = svg.append("g");
  const marks: SVGGraphicsElement[] = [];
  const sameUnit = barFormat === lineFormat;

  cols.forEach((c, i) => {
    const sel = state[i];
    const dimmed = sel.anySelected && !sel.isSelected;
    const g = hitsG.append("g");

    if (sel.isSelected) {
      // Selection is outline plus dimming of the rest. Never a hue change —
      // hue is already spoken for by `tone`.
      g.append("rect")
        .attr("x", x(c.key) ?? 0)
        .attr("y", m.top - 3)
        .attr("width", x.bandwidth())
        .attr("height", plotBottom - m.top + 6)
        .attr("rx", 4)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5);
    }

    const rows: [string, string][] = [
      [barLabel, c.bar === null ? "no value in this slice" : fmtValue(c.bar, barFormat)],
      [lineLabel, c.line === null ? "not planned" : fmtValue(c.line, lineFormat)],
    ];
    // A gap is only a gap when the two measures share a unit. Where they do
    // not, the two numbers stand alone rather than being subtracted.
    if (sameUnit && c.bar !== null && c.line !== null) {
      const delta = c.bar - c.line;
      rows.push([
        "Gap",
        barFormat === "percent"
          ? `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pts`
          : `${delta >= 0 ? "+" : ""}${fmtValue(delta, barFormat)}`,
      ]);
      if (c.line !== 0) rows.push(["Attainment", `${((c.bar / c.line) * 100).toFixed(0)}%`]);
    }

    const filterHint = spec.clickDim ? `Click to filter ${spec.clickDim} = ${c.key}` : undefined;
    const foot = c.annotation
      ? filterHint
        ? `${c.annotation} · ${filterHint}`
        : c.annotation
      : filterHint;

    const hit = g
      .append("rect")
      .attr("x", x(c.key) ?? 0)
      .attr("y", m.top)
      .attr("width", Math.max(2, x.step()))
      .attr("height", plotBottom - m.top)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    const bar = bars[i];
    const dot = dots[i];
    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, c.label, rows, foot),
      aria:
        `${c.label}. ${barLabel} ${c.bar === null ? "no value" : fmtValue(c.bar, barFormat)}. ` +
        `${lineLabel} ${c.line === null ? "not planned" : fmtValue(c.line, lineFormat)}.` +
        (c.annotation ? ` ${c.annotation}.` : ""),
      dim: spec.clickDim,
      value: c.key,
      opts,
      onEnter: () => {
        guide.attr("x1", cx(c)).attr("x2", cx(c)).attr("opacity", 0.7);
        bar?.attr("opacity", 0.82);
        dot?.attr("r", 5).attr("opacity", 1);
      },
      onLeave: () => {
        guide.attr("opacity", 0);
        bar?.attr("opacity", 1);
        dot?.attr("r", 3.2).attr("opacity", dimmed ? DIM_OPACITY : 1);
      },
    });
    marks.push(hit);
  });

  // ------------------------------------------------- one tab stop when dense
  // the palette-as-parameter rule. `util.ts` exposes no `markGroup()` yet, so the roving
  // tabindex lives here and nowhere else: the group is entered once, the arrow
  // keys walk it, Home and End jump to the ends. `attachMark` already owns
  // Enter, Space and Escape on each mark, so nothing here duplicates it.
  if (marks.length > DENSE) {
    hitsG.attr("role", "group").attr("aria-label", `${cols.length} periods. Use the arrow keys to move between them.`);
    marks.forEach((node, i) => {
      node.setAttribute("tabindex", i === 0 ? "0" : "-1");
      const onKey = (ev: KeyboardEvent) => {
        let next = -1;
        if (ev.key === "ArrowRight" || ev.key === "ArrowDown") next = Math.min(marks.length - 1, i + 1);
        else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") next = Math.max(0, i - 1);
        else if (ev.key === "Home") next = 0;
        else if (ev.key === "End") next = marks.length - 1;
        else return;
        ev.preventDefault();
        node.setAttribute("tabindex", "-1");
        marks[next].setAttribute("tabindex", "0");
        (marks[next] as unknown as HTMLElement).focus();
      };
      node.addEventListener("keydown", onKey as EventListener);
      cleanups.push(() => node.removeEventListener("keydown", onKey as EventListener));
    });
  }

  let torn = false;
  return () => {
    // StrictMode invokes the cleanup twice; the second call must be a no-op.
    if (torn) return;
    torn = true;
    cleanups.forEach((f) => f());
    cleanups.length = 0;
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

/**
 * `Shape` in `api/types.ts` still enumerates only the five shapes the first six
 * modules serve, and that file is shared. Widening through `string` lets this
 * module declare CONTRACT's own shape string for the combo without editing it;
 * the registry resolves on `key` and `selectChart()` maps the shape.
 */
const SERVES = "temporal×measure×measure" as string as ChartModule["serves"];

export const comboColumnLine: ChartModule = {
  key: "combo.columnline",
  serves: SERVES,
  render,
};
