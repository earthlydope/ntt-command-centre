/**
 * table.compact — the fallback. Serves nothing and therefore serves everything.
 *
 * `select()` returns this key for any shape the repository does not register,
 * so an unknown shape degrades to a readable table instead of throwing and
 * blanking the lens. the chart repository contract.
 *
 * It is drawn in SVG rather than HTML on purpose: every module has to run
 * unchanged under jsdom for the Outlook card PNG (the build contract), and an HTML
 * table would be the one card that could not.
 */
import { max } from "d3-array";
import { scaleLinear } from "d3-scale";
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
  formatValue,
  pitch,
  selectionState,
  textWidth,
  tipHtml,
  truncateLabel,
  enc,
  payload,
  rows as specRows,
} from "./util";

const HEAD = 28;
/** A row is a line of label text with room to breathe; it grows a little
 *  towards an offered height, but a table with airy rows stops being a table. */
const ROW_MIN = 32;
const ROW_MAX = 40;
const MAX_ROWS = 12;
/** The "+n more rows" line, or the bare bottom margin when nothing is cut. */
const MORE_H = 20;
const BOTTOM_PAD = 6;
const EMPTY_HEIGHT = 96;

function tableRows(spec: ChartSpec): Record<string, unknown>[] {
  const tbl = payload<{ rows?: unknown }>(spec, {});
  return Array.isArray(tbl.rows)
    ? (tbl.rows as Record<string, unknown>[])
    : (specRows(spec) as Record<string, unknown>[]);
}

function layout(spec: ChartSpec, offered: number): { row: number; height: number } {
  const all = tableRows(spec);
  const n = Math.min(all.length, MAX_ROWS);
  if (!n) return { row: ROW_MIN, height: EMPTY_HEIGHT };
  const tail = all.length > MAX_ROWS ? MORE_H : BOTTOM_PAD;
  const row = pitch(offered, HEAD + tail, n, ROW_MIN, ROW_MAX);
  return { row, height: HEAD + n * row + tail };
}

/** Columns beyond x/y that are worth printing, in the order they read best. */
const EXTRA = ["opportunities", "lines", "share", "n"];

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
  // docs/CHART_CONTRACT.md §"table.compact" sends `{columns, rows}`; a bare row array is
  // still accepted, which is what the shape fallback produces.
  const tbl = payload<{ columns?: unknown; rows?: unknown }>(spec, {});
  const all = tableRows(spec);
  const declaredCols = Array.isArray(tbl.columns)
    ? (tbl.columns as { key: string; label: string; format?: string; align?: string }[])
    : null;
  const rows = all.slice(0, MAX_ROWS);

  const width = Math.max(240, opts.width);
  const { row: ROW, height } = layout(spec, opts.height);
  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "table")
    .attr("aria-label", `${spec.title}. ${all.length} rows.`);

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

  // When the server declares its own columns, they win: the first is the label,
  // the first numeric one is the ranked value, and up to two more ride along as
  // extras. Without this a contract table renders its bars against a `value`
  // field it never sent and every row draws empty.
  const numericish = (k: string) =>
    rows.some((r) => r[k] !== undefined && r[k] !== null && !Number.isNaN(Number(r[k])));
  const nameKey = declaredCols?.[0]?.key ?? xKey;
  const valueKey =
    declaredCols?.slice(1).find((c) => numericish(c.key))?.key ?? yKey;
  const extras = declaredCols
    ? declaredCols
        .map((c) => c.key)
        .filter((k) => k !== nameKey && k !== valueKey && rows[0][k] !== undefined)
        .slice(0, 2)
    : EXTRA.filter((k) => rows[0][k] !== undefined).slice(0, 2);
  const headLabel = (k: string) =>
    declaredCols?.find((c) => c.key === k)?.label ?? k;
  // Column widths follow the type: a value column holds a twelve-character
  // figure, an extra column a shorter count or share.
  const valX = width - 8;
  const valueW = textWidth(12, FONT.label) + 8;
  const extraW = textWidth(10, FONT.label) + 8;
  const extraX = extras.map((_, i) => valX - valueW - i * extraW);
  const nameW = (extras.length ? extraX[extras.length - 1] : valX - valueW) - 12;

  const tip = createTooltip(p);
  const maxV = max(rows, (d) => Number(d[valueKey])) ?? 1;
  const w = scaleLinear().domain([0, maxV || 1]).range([0, Math.max(30, nameW)]);

  const head = svg.append("g");
  const th = (x: number, t: string, anchor: string) =>
    head
      .append("text")
      .attr("x", x)
      .attr("y", 15)
      .attr("text-anchor", anchor)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .attr("font-weight", 700)
      .attr("letter-spacing", "0.1em")
      .text(t.toUpperCase());
  th(0, declaredCols ? headLabel(nameKey) : (spec.clickDim ?? "category"), "start");
  extras.forEach((k, i) => th(extraX[i], headLabel(k), "end"));
  th(valX, declaredCols ? headLabel(valueKey) : "value", "end");
  head
    .append("line")
    .attr("x1", 0)
    .attr("x2", width)
    .attr("y1", HEAD - 6)
    .attr("y2", HEAD - 6)
    .attr("stroke", p.line);

  rows.forEach((d, i) => {
    const name = String(d[nameKey]);
    const v = Number(d[valueKey]);
    const top = HEAD + i * ROW;
    const sel = selectionState(opts, spec.clickDim, name);
    const dimmed = sel.anySelected && !sel.isSelected;
    const g = svg.append("g").attr("opacity", dimmed ? DIM_OPACITY : 1);

    // a faint value bar behind the label — the ranking is the point of a table
    g.append("rect")
      .attr("x", 0)
      .attr("y", top + 4)
      .attr("width", Math.max(1, w(v)))
      .attr("height", ROW - 10)
      .attr("rx", 3)
      .attr("fill", alpha(p.accentRgb, 0.1));

    g.append("text")
      .attr("x", 7)
      .attr("y", top + ROW / 2 + 4.5)
      .attr("fill", p.text)
      .attr("font-size", FONT.label)
      .attr("font-weight", sel.isSelected ? 750 : 500)
      .text(truncateLabel(name, nameW - 10));

    extras.forEach((k, j) => {
      const raw = Number(d[k]);
      const txt = k === "share" ? `${raw.toFixed(0)}%` : raw.toLocaleString("en-US");
      g.append("text")
        .attr("x", extraX[j])
        .attr("y", top + ROW / 2 + 4.5)
        .attr("text-anchor", "end")
        .attr("fill", p.muted)
        .attr("font-size", FONT.label)
        .attr("font-variant-numeric", "tabular-nums")
        .text(txt);
    });

    g.append("text")
      .attr("x", valX)
      .attr("y", top + ROW / 2 + 4.5)
      .attr("text-anchor", "end")
      .attr("fill", p.text)
      .attr("font-size", FONT.label)
      .attr("font-weight", 700)
      .attr("font-variant-numeric", "tabular-nums")
      .text(formatValue(v, spec.format));

    g.append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", top + ROW)
      .attr("y2", top + ROW)
      .attr("stroke", p.line)
      .attr("stroke-opacity", 0.6);

    const hit = g
      .append("rect")
      .attr("x", 0)
      .attr("y", top)
      .attr("width", width)
      .attr("height", ROW)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    const tipRows: [string, string][] = [["Value", formatValue(v, spec.format)]];
    for (const k of extras) tipRows.push([k, Number(d[k]).toLocaleString("en-US")]);
    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, name, tipRows, spec.clickDim ? `Click to filter ${spec.clickDim} = ${name}` : undefined),
      aria: `${name}, ${formatValue(v, spec.format)}`,
      dim: spec.clickDim,
      value: name,
      opts,
      onEnter: () => g.attr("opacity", 1),
      onLeave: () => g.attr("opacity", dimmed ? DIM_OPACITY : 1),
    });
  });

  if (all.length > MAX_ROWS) {
    svg
      .append("text")
      .attr("x", 0)
      .attr("y", height - 5)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(`+${all.length - MAX_ROWS} more rows`);
  }

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const tableCompact: ChartModule = {
  key: "table.compact",
  serves: "fallback",
  render,
  drawnHeight: (spec, offered) => layout(spec, offered).height,
};
