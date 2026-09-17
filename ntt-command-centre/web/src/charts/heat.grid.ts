/**
 * heat.grid — serves `categorical×categorical×measure`.
 * LOB × Portfolio. This is the roll-up grain Praveen asked coverage to be read
 * at: "analysis is not done at a row level compared to the budget. It's rolled
 * up." CONTRACT P7.
 *
 * An empty intersection is NOT drawn — that is why the light theme's heatLow is
 * #CFE0F9 and not a near-white tint: too pale a floor makes "low value" and
 * "no rows at all" look the same. the theme contract.
 *
 * Both axes are marks. A column header filters LOB, a row label filters
 * Portfolio, a cell filters the spec's clickDim.
 */
import { scaleBand, scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { interpolateRgb } from "d3-interpolate";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec, DimKey } from "../api/types";
import type { Palette } from "../theme/palette";
import { divergingAt } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  charsIn,
  createTooltip,
  formatValue,
  pitch,
  selectionState,
  textWidth,
  tipHtml,
  truncate,
  truncateLabel,
  enc,
  payload,
  rows as specRows,
} from "./util";

/** The column-header band above the grid and the legend strip under it. */
const HEAD_H = 36;
const LEGEND_H = 30;
/**
 * When the columns are too narrow for their names to lie flat, the names lean
 * at 45 degrees instead — the standard answer for a dense grid — and the band
 * grows to hold them, up to this cap. A fourteen-column grid at full width
 * gives each column about sixty pixels, which at note size is eight
 * characters: every "Digital Workplace / …" column would read the same.
 */
const HEAD_MAX = 150;
const HEAD_LEAN = Math.SQRT1_2; // sin 45 = cos 45
/** A leaning label is cut here whatever the band allows; past it the band
 *  is taller than the grid it labels. */
const HEAD_CHARS = 28;
/** A cell is a row of label text tall at least; it grows towards the offered
 *  height until the grid would read as a table of empty boxes. */
const CELL_MIN = 32;
const CELL_MAX = 44;
const EMPTY_HEIGHT = 96;

/** The axes the payload declares, or the ones its rows imply. Shared by
 *  render() and the layout pass so both count the same rows. */
function axes(spec: ChartSpec): { xs: string[]; ys: string[]; n: number } {
  const obj = payload<{ rows?: unknown; cols?: unknown; cells?: unknown }>(spec, {});
  const contractCells = Array.isArray(obj.cells) ? (obj.cells as Record<string, unknown>[]) : null;
  const xKey = contractCells ? "col" : enc(spec).x;
  const yKey = contractCells ? "row" : enc(spec).y;
  const rows = (contractCells ?? (specRows(spec) as Record<string, unknown>[])).filter(
    (d) => d.hasBudget === undefined || d.hasBudget === true,
  );
  const xs = Array.isArray(obj.cols) && obj.cols.length
    ? (obj.cols as string[]).map(String)
    : [...new Set(rows.map((d) => String(d[xKey])))].sort();
  const ys = Array.isArray(obj.rows) && obj.rows.length
    ? (obj.rows as string[]).map(String)
    : [...new Set(rows.map((d) => String(d[yKey])))].sort();
  return { xs, ys, n: rows.length };
}

/** `headH` is the header band render() has settled on from the width; the
 *  layout pass, which has no width, takes the flat band. */
function layout(
  spec: ChartSpec,
  offered: number,
  headH: number = HEAD_H,
): { cellH: number; height: number } {
  const { ys, n } = axes(spec);
  if (!n) return { cellH: CELL_MIN, height: EMPTY_HEIGHT };
  const cellH = pitch(offered, headH + LEGEND_H, ys.length, CELL_MIN, CELL_MAX);
  return { cellH, height: headH + ys.length * cellH + LEGEND_H };
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  // The encoding names the fields AND the two dimensions: the axis keys are
  // dimension keys ("lob", "portfolio"), so which axis filters which slicer is
  // read off the spec rather than assumed. Swapping the axes server-side needs
  // no change here.
  // docs/CHART_CONTRACT.md §"heat.grid" sends `{rows, cols, cells}` with each cell carrying
  // `{row, col, value, secondary, zero}`. This module predates that contract and
  // read a flat array through `encoding`; both are accepted so a spec built
  // either way renders, but the contract shape is the one the server sends.
  const obj = payload<{
    rows?: unknown;
    cols?: unknown;
    cells?: unknown;
  }>(spec, {});
  const contractCells = Array.isArray(obj.cells)
    ? (obj.cells as Record<string, unknown>[])
    : null;

  const xKey = contractCells ? "col" : enc(spec).x;
  const yKey = contractCells ? "row" : enc(spec).y;
  // The axis keys double as the dimension a header click filters. Under the
  // contract shape the payload's own keys are "row"/"col", which are not
  // dimensions, so the filter dimension comes from `clickDim` instead.
  const xDim = (contractCells ? spec.clickDim : (xKey as DimKey)) as DimKey;
  const yDim = (contractCells ? spec.clickDim : (yKey as DimKey)) as DimKey;
  // A grid with no `clickDim` is a picture, not a filter control — the
  // cross-sell matrix, for one — so the axis words fall back to the payload's
  // own row/column nouns instead of dereferencing a null dimension.
  const label = (k: string | null | undefined) =>
    !k ? "this" : k === "lob" ? "LOB" : k === "col" ? "Portfolio" : k === "row" ? "LOB"
      : k.charAt(0).toUpperCase() + k.slice(1);
  const vKey = contractCells ? "value" : (enc(spec).series ?? enc(spec).label ?? "v");
  const countKey = contractCells ? "secondary" : (enc(spec).label ?? "lines");

  const rows = (contractCells ?? (specRows(spec) as Record<string, unknown>[])).filter(
    // A pair with no plan at all is an absence, not a zero, and is drawn as one.
    (d) => d.hasBudget === undefined || d.hasBudget === true,
  );

  const { xs, ys } = axes(spec);
  const width = Math.max(260, opts.width);

  // The row-label gutter is a share of the width, capped at what the longest
  // label needs so a grid of short names keeps its columns wide.
  const longest = ys.reduce((a, k) => Math.max(a, k.length), 0);
  const labelW = Math.min(
    Math.max(92, width * 0.24),
    Math.max(92, textWidth(longest, FONT.label) + 16),
  );
  // Flat column names if every one of them fits its column; a leaning band
  // otherwise. A leaning label runs up and to the right of its column, so the
  // last column's name would run past the right edge and be clipped by the
  // viewport — the grid gives up that overhang on the right instead.
  const gridRight = width - 6;
  const flat = scaleBand<string>().domain(xs).range([labelW, gridRight]).paddingInner(0.06);
  const longestCol = xs.reduce((a, k) => Math.max(a, k.length), 0);
  const lean = xs.length > 0 && textWidth(longestCol, FONT.note) + 4 > flat.bandwidth();
  let headH = HEAD_H;
  let leanChars = 0;
  let overhang = 0;
  if (lean) {
    headH = Math.min(HEAD_MAX, Math.ceil(textWidth(Math.min(HEAD_CHARS, longestCol), FONT.note) * HEAD_LEAN) + 18);
    // Cut each name to what the band can hold along the diagonal.
    leanChars = Math.min(HEAD_CHARS, longestCol, charsIn((headH - 18) / HEAD_LEAN, FONT.note));
    const lastLen = textWidth(Math.min(leanChars, xs[xs.length - 1].length), FONT.note);
    overhang = Math.max(0, Math.ceil(lastLen * HEAD_LEAN) - flat.bandwidth() / 2);
  }
  const { cellH, height } = layout(spec, opts.height, headH);

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr("aria-label", `${spec.title}. ${xs.length} by ${ys.length} grid.`);

  if (!rows.length) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text("No open rows in this slice");
    return () => svg.selectAll("*").remove();
  }

  const x = scaleBand<string>()
    .domain(xs)
    .range([labelW, gridRight - overhang])
    .paddingInner(0.06);
  const y = scaleBand<string>()
    .domain(ys)
    .range([headH, headH + ys.length * cellH])
    .paddingInner(0.08);

  const vals = rows.map((d) => Number(d[vKey])).filter((v) => Number.isFinite(v));
  const maxV = vals.length ? Math.max(...vals) : 1;
  const minV = vals.length ? Math.min(...vals) : 0;

  // A domain that crosses zero is DIVERGING, not sequential. The rep-behaviour
  // grid is standard deviations from a peer norm, where the sign is the whole
  // message — below the norm and above it are opposite findings, not "less" and
  // "more" of one. Passing a negative through the sequential sqrt ramp yields
  // NaN and paints the cell black, which is how that grid first rendered.
  const diverging = minV < -0.01;
  const extent = Math.max(Math.abs(minV), Math.abs(maxV)) || 1;

  // sqrt so the long tail of small cells stays distinguishable from empty
  const t = scaleLinear().domain([0, Math.sqrt(Math.max(maxV, 0) || 1)]).range([0, 1]);
  const ramp = interpolateRgb(p.heatLow, p.accent);
  const fill = (v: number) =>
    diverging ? divergingAt(p, v, 0, extent) : ramp(t(Math.sqrt(Math.max(v, 0))));
  /** How saturated this cell is, for deciding whether its label knocks out. */
  const intensity = (v: number) =>
    diverging ? Math.abs(v) / extent : t(Math.sqrt(Math.max(v, 0)));

  const tip = createTooltip(p);
  const cellDim: DimKey = spec.clickDim ?? yDim;
  const total = rows.reduce((s, d) => s + Number(d[vKey]), 0);

  // A coverage multiple reads as "2.4x", never as "2". Anything else keeps the
  // spec's own format, so the same module serves a value grid unchanged.
  const isCoverage = /coverage/i.test(spec.id) || /coverage/i.test(spec.title);
  const cellText = (v: number) =>
    isCoverage ? `${v.toFixed(v < 10 ? 1 : 0)}x` : formatValue(v, spec.format);
  const money = (v: unknown) => formatValue(Number(v ?? 0), "currency");
  const tipRowsFor = (d: Record<string, unknown>): [string, string][] => {
    const rowsOut: [string, string][] = [];
    if (isCoverage) {
      rowsOut.push(["Coverage", cellText(Number(d[vKey]))]);
      if (d.pipeline !== undefined) rowsOut.push(["Pipeline not yet lost", money(d.pipeline)]);
      if (d.budget !== undefined) rowsOut.push(["Revenue plan", money(d.budget)]);
      if (d.gp !== undefined) rowsOut.push(["Pipeline GP", money(d.gp)]);
      if (d.gpBudget !== undefined) rowsOut.push(["GP plan", money(d.gpBudget)]);
    } else {
      rowsOut.push(["Value", formatValue(Number(d[vKey]), spec.format)]);
      rowsOut.push(["Share of grid", `${((Number(d[vKey]) / (total || 1)) * 100).toFixed(1)}%`]);
    }
    rowsOut.push([
      spec.countBasis === "opportunities" ? "Opportunities" : "Lines",
      Number(d[countKey] ?? d.lines ?? 0).toLocaleString("en-US"),
    ]);
    if (d.material === false) rowsOut.push(["Materiality", "below the 0.5% floor"]);
    return rowsOut;
  };
  const byCell = new Map(rows.map((d) => [`${d[xKey]}||${d[yKey]}`, d]));

  // ------------------------------------------------- column headers (x axis)
  const head = svg.append("g");
  xs.forEach((cx) => {
    const g = head.append("g");
    const sel = selectionState(opts, xDim, cx);
    const text = g
      .append("text")
      .attr("fill", sel.isSelected ? p.text : p.muted)
      .attr("font-size", FONT.note)
      .attr("font-weight", sel.isSelected ? 750 : 650)
      .attr("letter-spacing", "0.05em");
    if (lean) {
      // Anchored at the column's centre on the band's baseline and leaning
      // up and to the right, so the name is read towards its column.
      const ax = (x(cx) ?? 0) + x.bandwidth() / 2;
      const ay = headH - 8;
      text
        .attr("x", ax)
        .attr("y", ay)
        .attr("text-anchor", "start")
        .attr("transform", `rotate(-45 ${ax} ${ay})`)
        .text(truncate(cx, Math.max(3, leanChars)));
    } else {
      text
        .attr("x", (x(cx) ?? 0) + x.bandwidth() / 2)
        .attr("y", headH - 14)
        .attr("text-anchor", "middle")
        .text(truncateLabel(cx, x.bandwidth() - 4, FONT.note));
    }
    // A leaning name crosses the columns to its right, so the name itself is
    // the mark; a flat band is hit anywhere in its column.
    const hit = lean
      ? (text.node() as SVGGraphicsElement)
      : (g
          .append("rect")
          .attr("x", x(cx) ?? 0)
          .attr("y", 2)
          .attr("width", x.bandwidth())
          .attr("height", headH - 6)
          .attr("fill", "transparent")
          .node() as SVGGraphicsElement);
    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, cx, [["Column total", formatValue(
        rows.filter((d) => String(d[xKey]) === cx).reduce((s, d) => s + Number(d[vKey]), 0),
        spec.format,
      )]], xDim ? `Click to filter ${label(xKey)}` : undefined),
      aria: `${label(xKey)} ${cx}`,
      dim: xDim ?? undefined,
      value: cx,
      opts,
    });
  });

  // ------------------------------------------------- row labels (y axis)
  const side = svg.append("g");
  ys.forEach((cy) => {
    const sel = selectionState(opts, yDim, cy);
    const g = side.append("g");
    g.append("text")
      .attr("x", labelW - 10)
      .attr("y", (y(cy) ?? 0) + y.bandwidth() / 2 + 4.5)
      .attr("text-anchor", "end")
      .attr("fill", sel.isSelected ? p.text : p.text2)
      .attr("font-size", FONT.label)
      .attr("font-weight", sel.isSelected ? 700 : 500)
      .text(truncateLabel(cy, labelW - 16));
    const hit = g
      .append("rect")
      .attr("x", 0)
      .attr("y", y(cy) ?? 0)
      .attr("width", labelW - 4)
      .attr("height", y.bandwidth())
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;
    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, cy, [["Row total", formatValue(
        rows.filter((d) => String(d[yKey]) === cy).reduce((s, d) => s + Number(d[vKey]), 0),
        spec.format,
      )]], yDim ? `Click to filter ${label(yKey)}` : undefined),
      aria: `${label(yKey)} ${cy}`,
      dim: yDim ?? undefined,
      value: cy,
      opts,
    });
  });

  // ------------------------------------------------------------- the cells
  const cells = svg.append("g");
  xs.forEach((cx) =>
    ys.forEach((cy) => {
      const d = byCell.get(`${cx}||${cy}`);
      const gx = x(cx) ?? 0;
      const gy = y(cy) ?? 0;
      if (!d) {
        // An unpopulated intersection is not a zero — it is drawn as an absence.
        cells
          .append("rect")
          .attr("x", gx)
          .attr("y", gy)
          .attr("width", x.bandwidth())
          .attr("height", y.bandwidth())
          .attr("rx", 3)
          .attr("fill", "none")
          .attr("stroke", p.line)
          .attr("stroke-dasharray", "2 3")
          .attr("aria-hidden", "true");
        return;
      }
      const v = Number(d[vKey]);
      const lobSel = selectionState(opts, xDim, cx);
      const pfSel = selectionState(opts, yDim, cy);
      const dimmed = (lobSel.anySelected && !lobSel.isSelected) || (pfSel.anySelected && !pfSel.isSelected);
      const g = cells.append("g");
      const rect = g
        .append("rect")
        .attr("x", gx)
        .attr("y", gy)
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .attr("rx", 3)
        .attr("fill", fill(v))
        .attr("stroke", p.cellStroke)
        .attr("stroke-width", 1)
        .attr("opacity", dimmed ? DIM_OPACITY : 1);

      if (lobSel.isSelected && pfSel.isSelected) {
        g.append("rect")
          .attr("x", gx - 1.5)
          .attr("y", gy - 1.5)
          .attr("width", x.bandwidth() + 3)
          .attr("height", y.bandwidth() + 3)
          .attr("rx", 4)
          .attr("fill", "none")
          .attr("stroke", p.text)
          .attr("stroke-width", 1.5);
      }

      // Label only where the cell is wide enough; the value is always in the tooltip.
      if (x.bandwidth() > textWidth(cellText(v).length, FONT.note) + 12) {
        g.append("text")
          .attr("x", gx + x.bandwidth() / 2)
          .attr("y", gy + y.bandwidth() / 2 + 4)
          .attr("text-anchor", "middle")
          .attr("font-size", FONT.note)
          .attr("font-variant-numeric", "tabular-nums")
          .attr("pointer-events", "none")
          .attr("fill", intensity(v) > 0.55 ? p.onAccent : p.text)
          .attr("opacity", dimmed ? 0.5 : 1)
          .text(cellText(v));
      }

      attachMark(rect.node() as SVGGraphicsElement, {
        tip,
        palette: p,
        // A cell is an intersection of two dimensions and a filter click sets
        // one, so the server says which: `clickDim`. Clicking the column header
        // or the row label instead filters that axis explicitly — which is how
        // you reach the other half of the intersection.
        html: tipHtml(
          p,
          `${cx} · ${cy}`,
          tipRowsFor(d),
          cellDim ? `Click to filter ${label(cellDim)} = ${cellDim === xDim ? cx : cy}` : undefined,
        ),
        aria: `${cx} by ${cy}, ${cellText(v)}`,
        dim: cellDim ?? undefined,
        value: cellDim === xDim ? cx : cy,
        opts,
        onEnter: () => rect.attr("stroke", p.text).attr("stroke-width", 1.6).attr("opacity", 1),
        onLeave: () =>
          rect.attr("stroke", p.cellStroke).attr("stroke-width", 1).attr("opacity", dimmed ? DIM_OPACITY : 1),
      });
    }),
  );

  // ------------------------------------------------------------- the legend
  const legend = svg.append("g").attr("transform", `translate(${labelW},${headH + ys.length * cellH + 14})`);
  const lw = Math.min(160, width - labelW - 90);
  const lgId = `heat-lg-${spec.id}-${p.name}`;
  const lg = svg
    .append("defs")
    .append("linearGradient")
    .attr("id", lgId)
    .attr("x1", 0).attr("x2", 1).attr("y1", 0).attr("y2", 0);
  if (diverging) {
    lg.append("stop").attr("offset", "0%").attr("stop-color", divergingAt(p, -extent, 0, extent));
    lg.append("stop").attr("offset", "50%").attr("stop-color", divergingAt(p, 0, 0, extent));
    lg.append("stop").attr("offset", "100%").attr("stop-color", divergingAt(p, extent, 0, extent));
  } else {
    lg.append("stop").attr("offset", "0%").attr("stop-color", p.heatLow);
    lg.append("stop").attr("offset", "100%").attr("stop-color", p.accent);
  }
  legend.append("rect").attr("width", lw).attr("height", 7).attr("rx", 3.5).attr("fill", `url(#${lgId})`);
  legend.append("text").attr("x", 0).attr("y", 21).attr("fill", p.muted).attr("font-size", FONT.note).text(diverging ? `${(-extent).toFixed(1)} SD` : isCoverage ? "0x" : "low");
  legend
    .append("text")
    .attr("x", lw)
    .attr("y", 21)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .text(diverging ? `+${extent.toFixed(1)} SD` : cellText(maxV));
  legend
    .append("text")
    .attr("x", lw + 14)
    .attr("y", 7.5)
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .text("dashed = not sold / no plan");

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const heatGrid: ChartModule = {
  key: "heat.grid",
  serves: "categorical×categorical×measure",
  render,
  drawnHeight: (spec, offered) => layout(spec, offered).height,
};
