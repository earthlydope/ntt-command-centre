/**
 * bar.categorical — serves `categorical×measure`.
 * Open pipeline by stage / LOB / portfolio.
 *
 * Horizontal, because the categories are words ("Technology Solution
 * Consulting", "Proposal / Price Quote") and a vertical axis turns them into
 * rotated confetti.
 */
import { max } from "d3-array";
import { scaleBand, scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette } from "../theme/palette";
import { alpha } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  createTooltip,
  formatTick,
  formatValue,
  pitch,
  selectionState,
  textWidth,
  tipHtml,
  wrapLabel,
  enc,
  rows as specRows,
} from "./util";

/**
 * The row pitch grows from its floor towards the height the card offers, so a
 * four-bar chart beside a taller neighbour fills its card rather than leaving
 * a strip under itself — but never past the cap, because a bar as thick as a
 * KPI tile reads as a block of colour, not a length.
 */
const ROW_MIN = 34;
const ROW_MAX = 56;
const GAP_TOP = 12;
const AXIS = 28;
/** Leading of a wrapped gutter label. Two lines of it fit inside ROW_MIN. */
const LABEL_LEAD = 14;
/** Room the empty-state line needs. */
const EMPTY_HEIGHT = 96;

function positiveRows(spec: ChartSpec) {
  const yKey = enc(spec).y;
  return specRows(spec).filter((d) => Number(d[yKey]) > 0);
}

/** One sizing rule, read by render() and by the layout pass alike. */
function layout(spec: ChartSpec, offered: number): { row: number; height: number; n: number } {
  const n = positiveRows(spec).length;
  if (!n) return { row: ROW_MIN, height: EMPTY_HEIGHT, n };
  const row = pitch(offered, GAP_TOP + AXIS, n, ROW_MIN, ROW_MAX);
  return { row, height: GAP_TOP + n * row + AXIS, n };
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  const xKey = enc(spec).x;
  const yKey = enc(spec).y;
  const rows = positiveRows(spec);
  const width = Math.max(240, opts.width);
  const { row: ROW, height } = layout(spec, opts.height);

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr("aria-label", `${spec.title}. ${rows.length} categories.`);

  if (!rows.length) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text("No rows in this slice");
    return () => svg.selectAll("*").remove();
  }

  // The label gutter is a share of the width, capped at what the longest
  // label would take; the value column is sized to the widest figure it will
  // print so a $250.03M never clips and a chart of small numbers is not given
  // a gutter it does not need.
  const longest = max(rows, (d) => String(d[xKey]).length) ?? 0;
  const labelW = Math.min(
    Math.max(96, width * 0.26),
    Math.max(96, textWidth(longest, FONT.label) + 14),
  );
  const maxV = max(rows, (d) => Number(d[yKey])) ?? 0;
  const widestValue = max(rows, (d) => formatValue(Number(d[yKey]), spec.format).length) ?? 4;
  const valueW = textWidth(widestValue, FONT.label) + 14;
  const x = scaleLinear()
    .domain([0, maxV || 1])
    .range([0, Math.max(40, width - labelW - valueW - 8)]);
  const y = scaleBand<string>()
    .domain(rows.map((d) => String(d[xKey])))
    .range([GAP_TOP, GAP_TOP + rows.length * ROW])
    .padding(0.3);

  const tip = createTooltip(p);
  const plot = svg.append("g").attr("transform", `translate(${labelW},0)`);

  // gridlines first, so every mark paints over them
  const ticks = x.ticks(Math.max(2, Math.floor(x.range()[1] / textWidth(14, FONT.tick))));
  plot
    .append("g")
    .selectAll("line")
    .data(ticks)
    .join("line")
    .attr("x1", (d) => x(d))
    .attr("x2", (d) => x(d))
    .attr("y1", GAP_TOP - 6)
    .attr("y2", GAP_TOP + rows.length * ROW)
    .attr("stroke", p.grid)
    .attr("stroke-width", 1);

  plot
    .append("g")
    .selectAll("text")
    .data(ticks)
    .join("text")
    .attr("x", (d) => x(d))
    .attr("y", GAP_TOP + rows.length * ROW + 18)
    .attr("text-anchor", "middle")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((d) => formatTick(d, spec.format));

  const basisWord = spec.countBasis === "opportunities" ? "opportunities" : "lines";

  const g = plot
    .selectAll<SVGGElement, Record<string, unknown>>("g.mark")
    .data(rows)
    .join("g")
    .attr("class", "mark");

  g.each(function (d) {
    const node = select(this);
    const name = String(d[xKey]);
    const v = Number(d[yKey]);
    // The count field is named by `encoding.label`; the API sends one count
    // already on the right grain (`countBasis` says which), so the chart never
    // has to choose between an opportunity count and a line count itself.
    const countKey = enc(spec).label ?? "count";
    const count = Number(d[countKey] ?? d.count ?? d.lines ?? d.opps ?? 0);
    const share = maxV ? (v / rows.reduce((s, r) => s + Number(r[yKey]), 0)) * 100 : 0;
    const { anySelected, isSelected } = selectionState(opts, spec.clickDim, name);
    const dimmed = anySelected && !isSelected;

    const rect = node
      .append("rect")
      .attr("x", 0)
      .attr("y", y(name)!)
      .attr("height", y.bandwidth())
      .attr("rx", 3)
      .attr("width", Math.max(2, x(v)))
      .attr("fill", p.accent)
      .attr("opacity", dimmed ? DIM_OPACITY : 1);

    // selection = outline + dimming of the rest. Never a hue change.
    if (isSelected) {
      node
        .append("rect")
        .attr("x", -2.5)
        .attr("y", y(name)! - 2.5)
        .attr("height", y.bandwidth() + 5)
        .attr("width", Math.max(2, x(v)) + 5)
        .attr("rx", 5)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5);
    }

    // The gutter label wraps to a second line rather than being cut where the
    // row is tall enough to hold two lines of type, which every row is: a
    // stage name or a play name read whole is the point of the chart.
    const lines = wrapLabel(name, labelW - 14, FONT.label, 2);
    const mid = y(name)! + y.bandwidth() / 2;
    const label = node
      .append("text")
      .attr("x", -10)
      .attr("text-anchor", "end")
      .attr("fill", dimmed ? p.muted : p.text)
      .attr("font-size", FONT.label)
      .attr("font-weight", isSelected ? 700 : 500);
    lines.forEach((line, i) => {
      label
        .append("tspan")
        .attr("x", -10)
        .attr("y", mid + 4.5 + (i - (lines.length - 1) / 2) * LABEL_LEAD)
        .text(line);
    });

    node
      .append("text")
      .attr("x", Math.max(2, x(v)) + 8)
      .attr("y", y(name)! + y.bandwidth() / 2 + 4.5)
      .attr("fill", dimmed ? p.muted : p.text)
      .attr("font-size", FONT.label)
      .attr("font-weight", 700)
      .attr("font-variant-numeric", "tabular-nums")
      .text(formatValue(v, spec.format));

    // a full-row hit area so the label and the whitespace are hoverable too
    const hit = node
      .append("rect")
      .attr("x", -labelW)
      .attr("y", y(name)! - y.step() * 0.15)
      .attr("width", width)
      .attr("height", y.step())
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(
        p,
        name,
        [
          [spec.format === "percent" ? "Value" : "Amount", formatValue(v, spec.format)],
          ["Share of shown", `${share.toFixed(1)}%`],
          [basisWord.replace(/^./, (c) => c.toUpperCase()), count.toLocaleString("en-US")],
        ],
        spec.clickDim ? `Click to filter ${spec.clickDim} = ${name}` : undefined,
      ),
      aria: `${name}, ${formatValue(v, spec.format)}, ${count.toLocaleString("en-US")} ${basisWord}`,
      dim: spec.clickDim,
      value: name,
      opts,
      onEnter: () => rect.attr("fill", alpha(p.accentRgb, 0.75)).attr("opacity", 1),
      onLeave: () => rect.attr("fill", p.accent).attr("opacity", dimmed ? DIM_OPACITY : 1),
    });
  });

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const barCategorical: ChartModule = {
  key: "bar.categorical",
  serves: "categorical×measure",
  render,
  drawnHeight: (spec, offered) => layout(spec, offered).height,
};
