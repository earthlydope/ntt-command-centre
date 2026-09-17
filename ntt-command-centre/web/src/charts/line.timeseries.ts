/**
 * line.timeseries — serves `temporal×measure`.
 * Closed-won by month: area + line + points, with every point interactive.
 *
 * Time is not one of the four validated dimensions, so the points carry no
 * clickDim — they hover and focus but they do not filter. That is deliberate:
 * a month is not a slicer in this model. the regression harness.
 */
import { extent, max } from "d3-array";
import { scaleLinear, scaleTime } from "d3-scale";
import { select } from "d3-selection";
import { area, line as d3line, curveMonotoneX } from "d3-shape";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette } from "../theme/palette";
import { alpha } from "../theme/palette";
import {
  FONT,
  attachMark,
  createTooltip,
  formatTick,
  formatValue,
  measureTickWidth,
  textWidth,
  tipHtml,
  enc,
  rows as specRows,
} from "./util";
import { monthLabel } from "../lib/format";

interface Pt {
  d: Date;
  v: number;
  n: number;
  raw: Record<string, unknown>;
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
  const pts: Pt[] = specRows(spec)
    .map((d) => ({
      d: new Date(String(d[xKey])),
      v: Number(d[yKey]),
      // `count` is on the grain `spec.countBasis` names — opportunities for a
      // closed-won series, since Stage is constant within an opportunity.
      n: Number(d.count ?? d.lines ?? 0),
      raw: d,
    }))
    .sort((a, b) => +a.d - +b.d);

  const width = Math.max(240, opts.width);
  const height = Math.max(180, opts.height || 230);
  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr("aria-label", `${spec.title}. ${pts.length} months.`);

  if (pts.length < 2) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text(pts.length ? "Only one month in this slice" : "No closed-won rows in this slice");
    return () => svg.selectAll("*").remove();
  }

  const maxV = max(pts, (d) => d.v) ?? 0;
  const yTicksProbe = scaleLinear().domain([0, maxV || 1]).nice().ticks(4);
  const m = { top: 14, right: 12, bottom: 28, left: measureTickWidth(yTicksProbe, spec.format) };

  const xs = scaleTime()
    .domain(extent(pts, (d) => d.d) as [Date, Date])
    .range([m.left, width - m.right]);
  const ys = scaleLinear()
    .domain([0, maxV || 1])
    .nice()
    .range([height - m.bottom, m.top]);

  const tip = createTooltip(p);

  // gridlines + y ticks
  const yTicks = ys.ticks(4);
  const grid = svg.append("g");
  grid
    .selectAll("line")
    .data(yTicks)
    .join("line")
    .attr("x1", m.left)
    .attr("x2", width - m.right)
    .attr("y1", (d) => ys(d))
    .attr("y2", (d) => ys(d))
    .attr("stroke", p.grid);
  grid
    .selectAll("text")
    .data(yTicks)
    .join("text")
    .attr("x", m.left - 8)
    .attr("y", (d) => ys(d) + 4)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((d) => formatTick(d, spec.format));

  // x ticks — first, last and a couple between; a month label is about eight
  // characters and wants a clear gap either side.
  const step = Math.max(
    1,
    Math.round(pts.length / Math.max(2, Math.floor(width / textWidth(15, FONT.tick)))),
  );
  const xTicks = pts.filter((_, i) => i % step === 0 || i === pts.length - 1);
  svg
    .append("g")
    .selectAll("text")
    .data(xTicks)
    .join("text")
    .attr("x", (d) => xs(d.d))
    .attr("y", height - 8)
    .attr("text-anchor", "middle")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((d) => monthLabel(d.d.toISOString()));

  const gradId = `ts-grad-${spec.id}-${p.name}`;
  const grad = svg
    .append("defs")
    .append("linearGradient")
    .attr("id", gradId)
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 0)
    .attr("y2", 1);
  grad.append("stop").attr("offset", "0%").attr("stop-color", alpha(p.accentRgb, 0.34));
  grad.append("stop").attr("offset", "100%").attr("stop-color", alpha(p.accentRgb, 0));

  svg
    .append("path")
    .datum(pts)
    .attr("fill", `url(#${gradId})`)
    .attr(
      "d",
      area<Pt>()
        .x((d) => xs(d.d))
        .y0(ys(0))
        .y1((d) => ys(d.v))
        .curve(curveMonotoneX),
    );

  svg
    .append("path")
    .datum(pts)
    .attr("fill", "none")
    .attr("stroke", p.accent)
    .attr("stroke-width", 2)
    .attr("stroke-linejoin", "round")
    .attr(
      "d",
      d3line<Pt>()
        .x((d) => xs(d.d))
        .y((d) => ys(d.v))
        .curve(curveMonotoneX),
    );

  const guide = svg
    .append("line")
    .attr("stroke", p.muted)
    .attr("stroke-dasharray", "3 3")
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("opacity", 0);

  // Every point is a mark: hover, keyboard focus, real value in the tooltip.
  const mean = pts.reduce((s, d) => s + d.v, 0) / pts.length;
  const g = svg.append("g");
  pts.forEach((d) => {
    const node = g.append("g");
    const dot = node
      .append("circle")
      .attr("cx", xs(d.d))
      .attr("cy", ys(d.v))
      .attr("r", 3)
      .attr("fill", p.surface)
      .attr("stroke", p.accent)
      .attr("stroke-width", 2);
    const hit = node
      .append("circle")
      .attr("cx", xs(d.d))
      .attr("cy", ys(d.v))
      .attr("r", 13)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, monthLabel(d.d.toISOString()), [
        ["Closed-won", formatValue(d.v, spec.format)],
        [spec.countBasis === "lines" ? "Lines" : "Opportunities", d.n.toLocaleString("en-US")],
        ["vs 18-month mean", `${d.v >= mean ? "+" : ""}${(((d.v - mean) / (mean || 1)) * 100).toFixed(0)}%`],
      ]),
      aria: `${monthLabel(d.d.toISOString())}, ${formatValue(d.v, spec.format)}, ${d.n} ${
        spec.countBasis === "lines" ? "lines" : "opportunities"
      }`,
      opts,
      onEnter: () => {
        dot.attr("r", 5.5);
        guide.attr("x1", xs(d.d)).attr("x2", xs(d.d)).attr("opacity", 0.7);
      },
      onLeave: () => {
        dot.attr("r", 3);
        guide.attr("opacity", 0);
      },
    });
  });

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const lineTimeseries: ChartModule = {
  key: "line.timeseries",
  serves: "temporal×measure",
  render,
};
