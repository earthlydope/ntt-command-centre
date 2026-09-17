/**
 * bar.stacked — serves `categorical×series×measure`.
 * Past-due ageing split by Forecast Category.
 *
 * The series ramp runs danger -> warn rather than through a categorical rainbow.
 * Everything in this chart is overdue pipeline, so the palette has to say
 * "adverse" before it says "which category" — the direction-aware colour rule, direction-aware
 * colour. The categories are told apart by the legend and the tooltip, which is
 * where the exact value lives anyway.
 */
import { max } from "d3-array";
import { scaleBand, scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import { stack } from "d3-shape";
import { interpolateRgb } from "d3-interpolate";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  createTooltip,
  formatTick,
  formatValue,
  measureTickWidth,
  selectionState,
  textWidth,
  tipHtml,
  truncate,
  wrapLabel,
  enc,
  payload,
  rows as specRows,
} from "./util";

/** The x labels and the legend row under the plot: one line of labels, or
 *  two when a bucket name ("31-90 days over") is wider than its band and
 *  wraps rather than being cut. */
const BOTTOM = 58;
const BOTTOM_WRAPPED = 72;
const LABEL_LEAD = 14;

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
  const sKey = enc(spec).series ?? "series";

  // docs/CHART_CONTRACT.md §"bar.stacked" sends `{categories, series:[{name, values}]}` — a
  // wide payload, because that is how the server already holds it. This module
  // draws long rows, so the wide form is unpivoted here rather than the server
  // sending the same numbers twice in two shapes.
  const widePayload = payload<{ categories?: unknown; series?: unknown }>(spec, {});
  const rows: Record<string, unknown>[] =
    Array.isArray(widePayload.categories) && Array.isArray(widePayload.series)
      ? (widePayload.series as { name?: unknown; values?: unknown }[]).flatMap((se) =>
          (Array.isArray(se.values) ? (se.values as unknown[]) : []).map((v, i) => ({
            [xKey]: String((widePayload.categories as unknown[])[i] ?? ""),
            [sKey]: String(se.name ?? ""),
            [yKey]: Number(v) || 0,
          })),
        )
      : (specRows(spec) as Record<string, unknown>[]);

  const width = Math.max(240, opts.width);
  const height = Math.max(200, opts.height || 250);
  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr("aria-label", spec.title);

  if (!rows.length) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text("Nothing in this slice is past its close date");
    return () => svg.selectAll("*").remove();
  }

  // keep the order the payload supplied for x (age buckets are ordinal)
  const xs: string[] = [];
  for (const r of rows) {
    const v = String(r[xKey]);
    if (!xs.includes(v)) xs.push(v);
  }
  const seriesTotals = new Map<string, number>();
  for (const r of rows) {
    const s = String(r[sKey]);
    seriesTotals.set(s, (seriesTotals.get(s) ?? 0) + Number(r[yKey]));
  }
  const series = [...seriesTotals.entries()].sort((a, b) => b[1] - a[1]).map(([s]) => s);

  const wide = xs.map((cat) => {
    const o: Record<string, number | string> = { cat };
    for (const s of series) o[s] = 0;
    for (const r of rows) if (String(r[xKey]) === cat) o[String(r[sKey])] = Number(r[yKey]);
    return o;
  });

  const totals = wide.map((d) => series.reduce((s, k) => s + Number(d[k]), 0));
  const maxV = max(totals) ?? 1;
  const yTicks = scaleLinear().domain([0, maxV]).nice().ticks(4);
  const left = measureTickWidth(yTicks, spec.format);
  const right = 8;

  // The x labels wrap to a second line where the band is narrower than the
  // words, and the plot gives up the extra line; the tooltip carries the full
  // bucket name either way.
  const step = (width - left - right) / Math.max(1, xs.length);
  const xLabels = new Map(xs.map((c) => [c, wrapLabel(c, step - 6, FONT.label, 2)]));
  const wrapped = [...xLabels.values()].some((lines) => lines.length > 1);
  const m = { top: 12, right, bottom: wrapped ? BOTTOM_WRAPPED : BOTTOM, left };

  const x = scaleBand<string>().domain(xs).range([m.left, width - m.right]).padding(0.32);
  const y = scaleLinear().domain([0, maxV]).nice().range([height - m.bottom, m.top]);

  const ramp = interpolateRgb(p.danger, p.warn);
  const colorOf = (s: string) => ramp(series.length < 2 ? 0 : series.indexOf(s) / (series.length - 1));

  const tip = createTooltip(p);

  const grid = svg.append("g");
  grid
    .selectAll("line")
    .data(y.ticks(4))
    .join("line")
    .attr("x1", m.left)
    .attr("x2", width - m.right)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d))
    .attr("stroke", p.grid);
  grid
    .selectAll("text")
    .data(y.ticks(4))
    .join("text")
    .attr("x", m.left - 8)
    .attr("y", (d) => y(d) + 4)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((d) => formatTick(d, spec.format));

  const xAxis = svg.append("g");
  xs.forEach((d) => {
    const t = xAxis
      .append("text")
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.label);
    (xLabels.get(d) ?? [d]).forEach((line, i) => {
      t.append("tspan")
        .attr("x", (x(d) ?? 0) + x.bandwidth() / 2)
        .attr("y", height - m.bottom + 18 + i * LABEL_LEAD)
        .text(line);
    });
  });

  const stacked = stack<Record<string, number | string>>().keys(series)(wide);
  const g = svg.append("g");

  stacked.forEach((layer) => {
    const s = layer.key;
    const sel = selectionState(opts, spec.clickDim, s);
    const dimmed = sel.anySelected && !sel.isSelected;
    layer.forEach((seg, i) => {
      const v = Number(wide[i][s]);
      if (!v) return;
      const cat = xs[i];
      const h = Math.max(1, y(seg[0]) - y(seg[1]));
      const rect = g
        .append("rect")
        .attr("x", x(cat) ?? 0)
        .attr("y", y(seg[1]))
        .attr("width", x.bandwidth())
        .attr("height", h)
        .attr("fill", colorOf(s))
        .attr("stroke", p.surface)
        .attr("stroke-width", 0.75)
        .attr("opacity", dimmed ? DIM_OPACITY : 1);

      if (sel.isSelected) {
        g.append("rect")
          .attr("x", (x(cat) ?? 0) - 1.5)
          .attr("y", y(seg[1]) - 1.5)
          .attr("width", x.bandwidth() + 3)
          .attr("height", h + 3)
          .attr("fill", "none")
          .attr("stroke", p.text)
          .attr("stroke-width", 1.5)
          .attr("pointer-events", "none");
      }

      attachMark(rect.node() as SVGGraphicsElement, {
        tip,
        palette: p,
        html: tipHtml(
          p,
          `${cat} overdue`,
          [
            [s, formatValue(v, spec.format)],
            ["Bucket total", formatValue(totals[i], spec.format)],
            ["Share of bucket", `${((v / (totals[i] || 1)) * 100).toFixed(0)}%`],
          ],
          spec.clickDim ? `Click to filter ${spec.clickDim} = ${s}` : undefined,
        ),
        aria: `${cat}, ${s}, ${formatValue(v, spec.format)}`,
        dim: spec.clickDim,
        value: s,
        opts,
        onEnter: () => rect.attr("opacity", 1).attr("stroke", p.text).attr("stroke-width", 1.4),
        onLeave: () =>
          rect.attr("opacity", dimmed ? DIM_OPACITY : 1).attr("stroke", p.surface).attr("stroke-width", 0.75),
      });
    });
  });

  // ------------------------------------------------ legend: also marks
  const lg = svg.append("g").attr("transform", `translate(${m.left},${height - 20})`);
  let lx = 0;
  series.forEach((s) => {
    const label = truncate(s, 16);
    const w = textWidth(label.length, FONT.note) + 24;
    if (lx + w > width - m.left - m.right) return; // silently drop what will not fit
    const item = lg.append("g").attr("transform", `translate(${lx},0)`);
    const sel = selectionState(opts, spec.clickDim, s);
    item
      .append("rect")
      .attr("width", 9)
      .attr("height", 9)
      .attr("rx", 2)
      .attr("y", -8)
      .attr("fill", colorOf(s))
      .attr("opacity", sel.anySelected && !sel.isSelected ? DIM_OPACITY : 1);
    item
      .append("text")
      .attr("x", 14)
      .attr("y", 0)
      .attr("fill", sel.isSelected ? p.text : p.muted)
      .attr("font-size", FONT.note)
      .attr("font-weight", sel.isSelected ? 700 : 500)
      .text(label);
    const hit = item
      .append("rect")
      .attr("x", -3)
      .attr("y", -12)
      .attr("width", w)
      .attr("height", 18)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;
    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, s, [["Past-due total", formatValue(seriesTotals.get(s) ?? 0, spec.format)]],
        spec.clickDim ? `Click to filter ${spec.clickDim} = ${s}` : undefined),
      aria: `Series ${s}`,
      dim: spec.clickDim,
      value: s,
      opts,
    });
    lx += w;
  });

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const barStacked: ChartModule = {
  key: "bar.stacked",
  serves: "categorical×series×measure",
  render,
};
