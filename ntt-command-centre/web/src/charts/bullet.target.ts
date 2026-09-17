/**
 * bullet.target — serves `target×actual`.
 * Services GP against the 30% target; pipeline against the revenue plan; the
 * quarter against its own quarterly plan.
 *
 * Praveen: "services GP should be 30%... that's a target. We're not saying that
 * we have that." The gap IS the product — nothing here rescales to make a bar
 * reach its marker. CONTRACT P9.
 *
 * **Every row carries its own target.** A margin chart happens to share one
 * (30% for all portfolios); a coverage chart does not — each LOB has its own
 * plan — so the target is read per row and drawn per row. Reading rows[0] would
 * silently measure every LOB against Networking's budget.
 *
 * Colour is direction-aware (rule 4). Where the semantic layer has already
 * classified the row it is obeyed, because the thresholds are a business rule
 * and business rules live in `api/semantic/`, not in a D3 module.
 */
import { max } from "d3-array";
import { scaleBand, scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  createTooltip,
  formatValue,
  pitch,
  selectionState,
  textWidth,
  tipHtml,
  truncateLabel,
  enc,
  rows as specRows,
} from "./util";

/** A row holds a label line, the track and the status word under the value;
 *  it grows towards the offered height only as far as the cap, past which the
 *  rows drift apart and stop reading as one list. */
const ROW_MIN = 48;
const ROW_MAX = 64;
const PAD = 8;
const EMPTY_HEIGHT = 96;

function layout(spec: ChartSpec, offered: number): { row: number; height: number } {
  const n = specRows(spec).length;
  if (!n) return { row: ROW_MIN, height: EMPTY_HEIGHT };
  const row = pitch(offered, PAD * 2, n, ROW_MIN, ROW_MAX);
  return { row, height: PAD + n * row + PAD };
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
  const tKey = enc(spec).target ?? "target";
  const rows = specRows(spec) as Record<string, unknown>[];

  const width = Math.max(240, opts.width);
  const { row: ROW, height } = layout(spec, opts.height);
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
      .text("No rows in this slice");
    return () => svg.selectAll("*").remove();
  }

  // One scale across the chart, sized to the largest of every value and every
  // target, so two rows with different targets stay visually comparable.
  // A row may legitimately have NO target: the 30% is a services target, so
  // Product and Net VBR appear on the margin chart for mix but are not graded.
  // `null` must stay null — coercing it to 0 pins the marker to the axis and
  // makes an ungraded row read as "+86.36 pts clear of a zero target".
  const targetOf = (d: Record<string, unknown>): number | null => {
    const raw = d[tKey];
    if (raw === null || raw === undefined || raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  };
  const maxTarget = max(rows, (d) => targetOf(d) ?? 0) ?? 0;
  const domainMax = Math.max(maxTarget * 1.15, (max(rows, (d) => Number(d[yKey])) ?? 0) * 1.08);
  const left = 4;
  // The value column has to hold the widest figure and the status word under it.
  const right = Math.max(textWidth(9, FONT.title) + 12, Math.min(150, width * 0.24));
  const x = scaleLinear().domain([0, domainMax]).range([left, width - right]);
  const y = scaleBand<string>()
    .domain(rows.map((d) => String(d[xKey])))
    .range([PAD, PAD + rows.length * ROW]);

  const tip = createTooltip(p);
  const isPct = spec.format === "percent";
  const TONES: Record<string, string> = {
    good: p.good,
    warn: p.warn,
    danger: p.danger,
    accent: p.accent,
    neutral: p.muted,
  };
  /** The semantic layer's own verdict wins; the fallback is a ratio rule, and
   *  for a percentage target "within five points" is the margin conversation
   *  Praveen actually has. */
  const statusOf = (d: Record<string, unknown>, v: number): string => {
    if (typeof d.status === "string") return d.status;
    const t = targetOf(d);
    if (t === null || t === 0) return "";
    if (v >= t) return "On Target";
    return (isPct ? v >= t - 5 : v >= t * 0.9) ? "Near" : "Below";
  };
  const toneOf = (d: Record<string, unknown>, v: number): string => {
    if (typeof d.tone === "string" && TONES[d.tone]) return TONES[d.tone];
    const st = statusOf(d, v);
    return st === "On Target" ? p.good : st === "Near" ? p.warn : p.danger;
  };

  rows.forEach((d) => {
    const name = String(d[xKey]);
    const v = Number(d[yKey]);
    const target = targetOf(d);
    const status = statusOf(d, v);
    const tone = toneOf(d, v);
    // `naive` is AVG(GM %) — the wrong way to compute a margin, carried by the
    // API so the tooltip can show the trap rather than describe it.
    const averaged =
      d.averaged !== undefined
        ? Number(d.averaged)
        : d.naive !== undefined
          ? Number(d.naive)
          : null;
    const sel = selectionState(opts, spec.clickDim, name);
    const dimmed = sel.anySelected && !sel.isSelected;
    const top = y(name)!;
    const g = svg.append("g").attr("opacity", dimmed ? DIM_OPACITY : 1);

    g.append("text")
      .attr("x", left)
      .attr("y", top + 14)
      .attr("fill", p.text)
      .attr("font-size", FONT.label)
      .attr("font-weight", sel.isSelected ? 750 : 600)
      .text(truncateLabel(name, width - right - 8));

    g.append("text")
      .attr("x", width - right + 10)
      .attr("y", top + 14)
      .attr("fill", tone)
      .attr("font-size", FONT.title)
      .attr("font-weight", 800)
      .attr("font-variant-numeric", "tabular-nums")
      .text(formatValue(v, spec.format));

    g.append("text")
      .attr("x", width - right + 10)
      .attr("y", top + 31)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .attr("letter-spacing", "0.08em")
      .text(status.toUpperCase());

    const barY = top + 22;
    g.append("rect")
      .attr("x", left)
      .attr("y", barY)
      .attr("width", x(domainMax) - left)
      .attr("height", 8)
      .attr("rx", 4)
      .attr("fill", p.track);

    const bar = g
      .append("rect")
      .attr("x", left)
      .attr("y", barY)
      .attr("width", Math.max(2, x(v) - left))
      .attr("height", 8)
      .attr("rx", 4)
      .attr("fill", tone);

    // the target marker — a hard line, never the end of the track. Drawn only
    // where a target exists; an ungraded row gets no marker at all rather than
    // a marker sitting on zero.
    if (target !== null) {
      g.append("line")
        .attr("x1", x(target))
        .attr("x2", x(target))
        .attr("y1", barY - 5)
        .attr("y2", barY + 13)
        .attr("stroke", p.text)
        .attr("stroke-width", 2);
    }

    if (sel.isSelected) {
      g.append("rect")
        .attr("x", left - 3)
        .attr("y", barY - 3)
        .attr("width", x(domainMax) - left + 6)
        .attr("height", 14)
        .attr("rx", 7)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.4);
    }

    const tipRows: [string, string][] = [
      [isPct ? "Weighted GM%" : "Actual", formatValue(v, spec.format)],
    ];
    if (target === null) {
      tipRows.push([isPct ? "Target" : "Plan", "not graded against this target"]);
    } else {
      const gap = isPct
        ? `${v >= target ? "+" : ""}${(v - target).toFixed(2)} pts`
        : `${v >= target ? "+" : ""}${formatValue(v - target, spec.format)}`;
      tipRows.push([isPct ? "Target" : "Plan", formatValue(target, spec.format)]);
      tipRows.push(["Gap", gap]);
    }
    if (d.coverage !== undefined)
      tipRows.push(["Coverage", `${Number(d.coverage).toFixed(2)}x`]);
    if (d.won !== undefined) tipRows.push(["Closed won", formatValue(Number(d.won), "currency")]);
    if (d.open !== undefined) tipRows.push(["Still open", formatValue(Number(d.open), "currency")]);
    if (d.gp !== undefined && !isPct)
      tipRows.push(["Pipeline GP", formatValue(Number(d.gp), "currency")]);
    const rev = d.rev ?? d.revenue;
    if (rev !== undefined) tipRows.push(["ACV revenue", formatValue(Number(rev), "currency")]);
    const count = d.count ?? d.lines;
    if (count !== undefined)
      tipRows.push([
        spec.countBasis === "opportunities" ? "Opportunities" : "Lines",
        Number(count).toLocaleString("en-US"),
      ]);
    if (averaged !== null) tipRows.push(["AVG(GM %) — wrong", `${averaged.toFixed(2)}%`]);
    if (d.material === false) tipRows.push(["Materiality", "below the 0.5% floor"]);
    const foot =
      target !== null && averaged !== null && Math.round(averaged) >= target && v < target
        ? "The averaged figure rounds to target and the weighted one misses it. The weighted figure is the true one."
        : spec.clickDim
          ? `Click to filter ${spec.clickDim} = ${name}`
          : undefined;

    const hit = g
      .append("rect")
      .attr("x", 0)
      .attr("y", top)
      .attr("width", width)
      .attr("height", ROW - 4)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, name, tipRows, foot),
      aria:
        target === null
          ? `${name}, ${formatValue(v, spec.format)}, not graded against this target`
          : `${name}, ${formatValue(v, spec.format)} against a ${formatValue(target, spec.format)} target, ${status}`,
      dim: spec.clickDim,
      value: name,
      opts,
      onEnter: () => bar.attr("opacity", 0.8),
      onLeave: () => bar.attr("opacity", 1),
    });
  });

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const bulletTarget: ChartModule = {
  key: "bullet.target",
  serves: "target×actual",
  render,
  drawnHeight: (spec, offered) => layout(spec, offered).height,
};
