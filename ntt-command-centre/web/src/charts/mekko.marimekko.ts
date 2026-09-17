/**
 * mekko.marimekko — serves `categorical×measure×width`.
 * LOB (column width = ACV revenue) × Portfolio (segment height = mix inside the
 * LOB), shaded by margin against the blended rate.
 *
 * The point of a Marimekko over a stacked bar is that BOTH axes carry a number:
 * column width ∝ the column's own total, segment height ∝ that segment's share
 * within its column, so a rectangle's AREA is its value. Networking at 62% of
 * revenue gets 62% of the width; a fat Product band inside it is Product's share
 * of Networking, not of the file. Reading area across columns is only honest
 * because of that, which is why nothing here rescales a column to fill the card.
 *
 * Fill is a third number on a DIVERGING ramp centred on `fillMid` — GM% against
 * the 16.23% blended rate — red below, palette-neutral at the rate, green above.
 * The ramp is symmetric about the midpoint by construction (one half-width taken
 * from the wider side), because a diverging ramp whose arms differ in length
 * turns "slightly below" into "as red as the worst cell".
 *
 * Density: 4 LOBs × 5 portfolios is already ~20 rects plus headers, and the same
 * module has to survive a wider cut. `util.ts` has no `markGroup()` helper, so
 * the roving-tabindex handling lives locally at the bottom of this file: the
 * header row is one tab stop and the segment grid is one tab stop, arrow keys
 * move inside each. the palette-as-parameter rule.
 *
 * A filter click sets `clickDim`, and in a Marimekko the columns ARE that
 * dimension — the segment breakdown is the second, unnamed axis of the payload.
 * So a segment filters its own column's key, exactly as its header does, and the
 * tooltip says so rather than leaving the user to guess which half they hit.
 */
import { max, min } from "d3-array";
import { rgb } from "d3-color";
import { interpolateRgb } from "d3-interpolate";
import { select } from "d3-selection";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  charsIn,
  createTooltip,
  formatValue,
  selectionState,
  tipHtml,
  truncate,
  truncateLabel,
  textWidth,
} from "./util";

/** docs/CHART_CONTRACT.md §"Payload types — exact", `mekko.marimekko`. */
interface MekkoSegment {
  key: string;
  value: number;
  /** The diverging-ramp value, e.g. GM% against `fillMid`. */
  fill: number;
}
interface MekkoColumn {
  key: string;
  total: number;
  segments: MekkoSegment[];
}
interface MekkoPayload {
  columns: MekkoColumn[];
  fillLabel: string;
  fillMid: number;
}

const HEAD = 36; // column key + its share of the grand total
const LEGEND = 40; // the diverging ramp and its three readings
/** The cumulative-mix gutter, 0% at the top: room for "100%" at note size. */
const AXIS_W = textWidth(4, FONT.note) + 10;
const GAP = 3; // without a gutter a Marimekko reads as one blob
const MIN_LABEL_W = 50;
const MIN_LABEL_H = 17;

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  const width = Math.max(260, opts.width);
  // The mix axis is 0–100% by construction, so the height is free: take the
  // measured card height (rule 6) and clamp it to something a stack reads in.
  const boxH = Math.max(220, Math.min(opts.height || 260, 560));
  // `format` is the measure's, not the fill's. Narrowed locally so a future
  // "days" on the spec cannot leak into a formatter that has no case for it.
  const fmt = (spec.format as "currency" | "percent" | "number" | undefined) ?? "currency";
  // `measureLabel` is a CONTRACT base key the wire projection has not caught up
  // with yet; read it if it is there, fall back to the reference wording if not.
  const measureLabel =
    (spec as unknown as { measureLabel?: string }).measureLabel?.trim() ||
    (fmt === "percent" ? "Value" : "Amount");

  /** Degrade, never throw: a centred message that names the problem. Rule 7/8. */
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

  const payload = (spec.data as unknown as Partial<MekkoPayload> | null) ?? {};
  const raw = Array.isArray(payload.columns) ? payload.columns : [];
  const fillLabel = String(payload.fillLabel ?? "Fill");

  // Area cannot be signed. A negative total or segment is named, not drawn.
  let negatives = 0;
  for (const c of raw) {
    if (Number(c?.total) < 0) negatives += 1;
    for (const s of Array.isArray(c?.segments) ? c.segments : []) {
      if (Number(s?.value) < 0) negatives += 1;
    }
  }
  if (negatives > 0) {
    return centred([
      `${negatives} negative ${negatives === 1 ? "value" : "values"} in this slice`,
      "A Marimekko encodes value as area, so it cannot draw them",
    ]);
  }

  const cols = raw
    .filter((c) => Number.isFinite(Number(c?.total)) && Number(c.total) > 0)
    .map((c) => ({
      key: String(c.key),
      total: Number(c.total),
      segments: (Array.isArray(c.segments) ? c.segments : [])
        .filter((s) => Number.isFinite(Number(s?.value)) && Number(s.value) > 0)
        .map((s) => ({ key: String(s.key), value: Number(s.value), fill: Number(s.fill) })),
    }));

  const grand = cols.reduce((s, c) => s + c.total, 0);
  if (!cols.length || !(grand > 0)) return centred(["No rows in this slice"]);

  // One stacking order for every column. The server's order is honoured where
  // the columns agree (first appearance wins); where they disagree the stack is
  // normalised, because a Marimekko whose bands swap places column to column
  // cannot be compared across columns at all.
  const order: string[] = [];
  for (const c of cols) for (const s of c.segments) if (!order.includes(s.key)) order.push(s.key);
  const rank = new Map(order.map((k, i) => [k, i] as const));
  for (const c of cols) c.segments.sort((a, b) => (rank.get(a.key) ?? 0) - (rank.get(b.key) ?? 0));

  // Widths come from `total`; heights from the segments that actually arrived.
  // The two agree on a whole payload and the shares stay truthful when they do
  // not, because each is read against its own denominator.
  const colSum = new Map(cols.map((c) => [c.key, c.segments.reduce((a, s) => a + s.value, 0)] as const));
  const grandVal = cols.reduce((s, c) => s + (colSum.get(c.key) ?? 0), 0);

  // -------------------------------------------------------- the diverging ramp
  const fills = cols.flatMap((c) => c.segments.map((s) => s.fill)).filter((v) => Number.isFinite(v));
  const weighted = (list: MekkoSegment[]): number => {
    let w = 0;
    let acc = 0;
    for (const s of list) {
      if (!Number.isFinite(s.fill)) continue;
      w += s.value;
      acc += s.fill * s.value;
    }
    return w > 0 ? acc / w : 0;
  };
  const midGiven = Number.isFinite(Number(payload.fillMid));
  const mid = midGiven ? Number(payload.fillMid) : weighted(cols.flatMap((c) => c.segments));
  const lo = min(fills) ?? mid;
  const hi = max(fills) ?? mid;
  // Symmetric arms: the wider side sets the half-width, so "at the rate" is dead
  // centre and a small miss never paints like the worst cell on the card.
  const spread = Math.max(mid - lo, hi - mid, 1e-9);
  const below = interpolateRgb(p.danger, p.muted);
  const above = interpolateRgb(p.muted, p.good);
  const fillFor = (v: number): string => {
    if (!Number.isFinite(v)) return p.track;
    const t = Math.max(-1, Math.min(1, (v - mid) / spread));
    return t < 0 ? below(1 + t) : above(t);
  };
  // The fill is a rate on this dataset (GM% against the blended 16.23%) and the
  // payload carries no format for it, so the label decides and the fallback is a
  // plain number rather than a percent sign the server never asked for.
  const fillIsPct = /%|pct|percent|margin|\bgm\b|rate/i.test(fillLabel);
  const fillText = (v: number): string =>
    Number.isFinite(v) ? formatValue(v, fillIsPct ? "percent" : "number") : "n/a";
  const delta = (v: number): string => {
    if (!Number.isFinite(v)) return "n/a";
    const d = v - mid;
    const sign = d >= 0 ? "+" : "-";
    return fillIsPct
      ? `${sign}${Math.abs(d).toFixed(1)} pts`
      : `${sign}${formatValue(Math.abs(d), "number")}`;
  };

  // Ink on a fill is chosen by that fill's luminance, from palette tokens only —
  // `onAccent` is calibrated for the accent hue and this ramp is not that hue.
  const lightInk = p.name === "dark" ? p.text : p.surface;
  const darkInk = p.name === "dark" ? p.bg : p.text;
  const inkOn = (c: string): string => {
    const q = rgb(c);
    const lum = (0.2126 * q.r + 0.7152 * q.g + 0.0722 * q.b) / 255;
    return Number.isFinite(lum) && lum > 0.5 ? darkInk : lightInk;
  };

  // ------------------------------------------------------------------ layout
  const axisW = width > 340 ? AXIS_W : 0;
  const plotW = Math.max(60, width - axisW - 4);
  const plotH = Math.max(140, boxH - HEAD - LEGEND);
  const height = HEAD + plotH + LEGEND;
  const innerW = Math.max(20, plotW - GAP * (cols.length - 1));
  const segCount = cols.reduce((s, c) => s + c.segments.length, 0);

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr(
      "aria-label",
      `${spec.title}. ${cols.length} columns, ${segCount} segments. Column width is its share of the total; segment height is its share within the column; colour is ${fillLabel} against ${fillText(mid)}.`,
    );

  const tip = createTooltip(p);

  // ------------------------------------------------- the cumulative mix gutter
  if (axisW) {
    const gutter = svg.append("g").attr("aria-hidden", "true");
    [0, 50, 100].forEach((t) => {
      gutter
        .append("text")
        .attr("x", axisW - 7)
        .attr("y", HEAD + (plotH * t) / 100 + (t === 0 ? 8 : t === 100 ? -1 : 3.5))
        .attr("text-anchor", "end")
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .attr("font-variant-numeric", "tabular-nums")
        .text(`${t}%`);
      gutter
        .append("line")
        .attr("x1", axisW - 4)
        .attr("x2", axisW - 1)
        .attr("y1", HEAD + (plotH * t) / 100)
        .attr("y2", HEAD + (plotH * t) / 100)
        .attr("stroke", p.grid)
        .attr("stroke-width", 1);
    });
  }

  const headG = svg
    .append("g")
    .attr("role", "group")
    .attr("aria-label", `Column headers. ${cols.length} columns; left and right arrows move between them.`);
  const gridG = svg
    .append("g")
    .attr("role", "group")
    .attr("aria-label", `${segCount} segments; arrow keys move within the grid.`);
  const overG = svg.append("g").attr("aria-hidden", "true");

  const headNodes: SVGGraphicsElement[] = [];
  const cellNodes: SVGGraphicsElement[] = [];
  const cellAt: number[][] = cols.map(() => []);
  const cellPos: { c: number; s: number }[] = [];

  let cursor = axisW;
  cols.forEach((c, ci) => {
    const w = (c.total / grand) * innerW;
    const x0 = cursor;
    cursor += w + GAP;
    const drawW = Math.max(1.5, w);
    const share = (c.total / grand) * 100;
    const segSum = colSum.get(c.key) ?? 0;
    const sel = selectionState(opts, spec.clickDim, c.key);
    const dimmed = sel.anySelected && !sel.isSelected;
    const clickFoot = spec.clickDim
      ? `Click to filter ${spec.clickDim} = ${c.key}`
      : "Width is share of total; height is mix within the column";

    // --------------------------------------------------------- column header
    const hg = headG.append("g");
    const headChars = charsIn(drawW - 4, FONT.label);
    if (drawW >= 34 && headChars >= 3) {
      hg.append("text")
        .attr("x", x0 + drawW / 2)
        .attr("y", 14)
        .attr("text-anchor", "middle")
        .attr("fill", dimmed ? p.muted : sel.isSelected ? p.text : p.text2)
        .attr("font-size", FONT.label)
        .attr("font-weight", sel.isSelected ? 750 : 650)
        .attr("letter-spacing", "0.04em")
        .attr("pointer-events", "none")
        .text(truncate(c.key, headChars));
    }
    if (drawW >= 24) {
      hg.append("text")
        .attr("x", x0 + drawW / 2)
        .attr("y", 28)
        .attr("text-anchor", "middle")
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .attr("font-variant-numeric", "tabular-nums")
        .attr("pointer-events", "none")
        .text(formatValue(share, "percent"));
    }
    const hhit = hg
      .append("rect")
      .attr("x", x0)
      .attr("y", 0)
      .attr("width", drawW)
      .attr("height", HEAD - 4)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;
    attachMark(hhit, {
      tip,
      palette: p,
      html: tipHtml(
        p,
        c.key,
        [
          [`${measureLabel} total`, formatValue(c.total, fmt)],
          ["Share of total", formatValue(share, "percent")],
          ["Segments", String(c.segments.length)],
          [`${fillLabel}, weighted`, fillText(weighted(c.segments))],
        ],
        clickFoot,
      ),
      aria: `Column ${c.key}, ${formatValue(c.total, fmt)}, ${formatValue(share, "percent")} of the total`,
      dim: spec.clickDim,
      value: c.key,
      opts,
    });
    headNodes.push(hhit);

    // ------------------------------------------------------------- the stack
    if (!(segSum > 0)) {
      // Width but no breakdown is an absence, not a zero, and is drawn as one.
      overG
        .append("rect")
        .attr("x", x0)
        .attr("y", HEAD)
        .attr("width", drawW)
        .attr("height", plotH)
        .attr("rx", 3)
        .attr("fill", "none")
        .attr("stroke", p.line)
        .attr("stroke-dasharray", "2 3");
      return;
    }

    let y = HEAD;
    c.segments.forEach((s) => {
      const h = (s.value / segSum) * plotH;
      const y0 = y;
      y += h;
      const paint = fillFor(s.fill);
      const g = gridG.append("g");
      const rect = g
        .append("rect")
        .attr("x", x0)
        .attr("y", y0)
        .attr("width", drawW)
        .attr("height", Math.max(0.75, h))
        .attr("fill", paint)
        .attr("stroke", p.cellStroke)
        .attr("stroke-width", h >= 3 ? 1 : 0)
        .attr("opacity", dimmed ? DIM_OPACITY : 1);

      // Label only where the rect can hold the text; everything else is in the
      // tooltip. A clipped word is worse than no word.
      const chars = charsIn(drawW - 12, FONT.label);
      if (drawW >= MIN_LABEL_W && h >= MIN_LABEL_H && chars >= 4) {
        const ink = inkOn(paint);
        const twoLine = h >= 34 && drawW >= 72;
        g.append("text")
          .attr("x", x0 + drawW / 2)
          .attr("y", y0 + h / 2 + (twoLine ? -2.5 : 4.5))
          .attr("text-anchor", "middle")
          .attr("fill", ink)
          .attr("font-size", FONT.label)
          .attr("font-weight", 600)
          .attr("opacity", dimmed ? 0.55 : 1)
          .attr("pointer-events", "none")
          .text(truncate(s.key, chars));
        if (twoLine) {
          g.append("text")
            .attr("x", x0 + drawW / 2)
            .attr("y", y0 + h / 2 + 13)
            .attr("text-anchor", "middle")
            .attr("fill", ink)
            .attr("font-size", FONT.note)
            .attr("font-variant-numeric", "tabular-nums")
            .attr("opacity", dimmed ? 0.5 : 0.85)
            .attr("pointer-events", "none")
            .text(formatValue(s.value, fmt));
        }
      }

      const node = rect.node() as SVGGraphicsElement;
      attachMark(node, {
        tip,
        palette: p,
        html: tipHtml(
          p,
          `${c.key} · ${s.key}`,
          [
            [measureLabel, formatValue(s.value, fmt)],
            [`Share of ${truncate(c.key, 18)}`, formatValue((s.value / segSum) * 100, "percent")],
            ["Share of total", formatValue((s.value / (grandVal || 1)) * 100, "percent")],
            [fillLabel, fillText(s.fill)],
            [`vs ${fillText(mid)}${midGiven ? "" : " mix mean"}`, delta(s.fill)],
          ],
          clickFoot,
        ),
        aria: `${c.key}, ${s.key}, ${formatValue(s.value, fmt)}, ${formatValue((s.value / segSum) * 100, "percent")} of ${c.key}, ${fillLabel} ${fillText(s.fill)}`,
        dim: spec.clickDim,
        value: c.key,
        opts,
        onEnter: () => rect.attr("stroke", p.text).attr("stroke-width", 1.6).attr("opacity", 1),
        onLeave: () =>
          rect
            .attr("stroke", p.cellStroke)
            .attr("stroke-width", h >= 3 ? 1 : 0)
            .attr("opacity", dimmed ? DIM_OPACITY : 1),
      });
      cellAt[ci].push(cellNodes.length);
      cellPos.push({ c: ci, s: cellAt[ci].length - 1 });
      cellNodes.push(node);
    });

    // Selection is an outline round the whole column plus the dimming above —
    // never a hue change, because hue already means margin here. Rule 3.
    if (sel.isSelected) {
      overG
        .append("rect")
        .attr("x", x0 - 1.5)
        .attr("y", HEAD - 1.5)
        .attr("width", drawW + 3)
        .attr("height", plotH + 3)
        .attr("rx", 3)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5);
    }
  });

  // ------------------------------------------------------------- the legend
  const legend = svg
    .append("g")
    .attr("transform", `translate(${axisW},${HEAD + plotH + 14})`)
    .attr("aria-hidden", "true");
  const lw = Math.min(190, Math.max(80, plotW - 120));
  const gid = `mekko-lg-${String(spec.id).replace(/[^A-Za-z0-9_-]/g, "")}-${p.name}`;
  const lg = svg
    .append("defs")
    .append("linearGradient")
    .attr("id", gid)
    .attr("x1", 0)
    .attr("x2", 1)
    .attr("y1", 0)
    .attr("y2", 0);
  lg.append("stop").attr("offset", "0%").attr("stop-color", p.danger);
  lg.append("stop").attr("offset", "50%").attr("stop-color", p.muted);
  lg.append("stop").attr("offset", "100%").attr("stop-color", p.good);
  legend.append("rect").attr("width", lw).attr("height", 7).attr("rx", 3.5).attr("fill", `url(#${gid})`);
  legend
    .append("line")
    .attr("x1", lw / 2)
    .attr("x2", lw / 2)
    .attr("y1", -3)
    .attr("y2", 10)
    .attr("stroke", p.text)
    .attr("stroke-width", 1.2);
  legend
    .append("text")
    .attr("y", 22)
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .text(fillText(mid - spread));
  legend
    .append("text")
    .attr("x", lw / 2)
    .attr("y", 22)
    .attr("text-anchor", "middle")
    .attr("fill", p.text2)
    .attr("font-size", FONT.note)
    .attr("font-weight", 650)
    .text(midGiven ? fillText(mid) : `${fillText(mid)} mean`);
  legend
    .append("text")
    .attr("x", lw)
    .attr("y", 22)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .text(fillText(mid + spread));
  if (width - axisW - lw > 104) {
    legend
      .append("text")
      .attr("x", lw + 14)
      .attr("y", 7.5)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(truncateLabel(`${fillLabel} — below / at / above`, width - axisW - lw - 18, FONT.note));
  }

  // -------------------------------------------------------- keyboard grouping
  // `util.ts` ships no markGroup(), so it is done here and only here: one tab
  // stop per group, arrows inside it. Rule 2.
  rove(headNodes, (i, key) => {
    if (key === "ArrowRight") return Math.min(i + 1, headNodes.length - 1);
    if (key === "ArrowLeft") return Math.max(i - 1, 0);
    if (key === "Home") return 0;
    if (key === "End") return headNodes.length - 1;
    return null;
  });
  rove(cellNodes, (i, key) => {
    const at = cellPos[i];
    if (!at) return null;
    const col = (k: number): number[] => cellAt[k] ?? [];
    const here = col(at.c);
    if (key === "ArrowDown") return here[Math.min(at.s + 1, here.length - 1)] ?? null;
    if (key === "ArrowUp") return here[Math.max(at.s - 1, 0)] ?? null;
    if (key === "Home") return here[0] ?? null;
    if (key === "End") return here[here.length - 1] ?? null;
    if (key === "ArrowRight" || key === "ArrowLeft") {
      const step = key === "ArrowRight" ? 1 : -1;
      for (let k = at.c + step; k >= 0 && k < cellAt.length; k += step) {
        const target = col(k);
        if (target.length) return target[Math.min(at.s, target.length - 1)];
      }
    }
    return null;
  });

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

/**
 * Roving tabindex over one group of marks: the group is a single tab stop and
 * `next` decides where a key takes the focus. Only the keyboard is wired here —
 * hover, click-to-filter and the focus tooltip all still come from attachMark(),
 * which has already set tabindex 0 on every node; this demotes the rest.
 */
function rove(nodes: SVGGraphicsElement[], next: (i: number, key: string) => number | null): void {
  if (!nodes.length) return;
  nodes.forEach((n, i) => n.setAttribute("tabindex", i === 0 ? "0" : "-1"));
  const focusTo = (i: number) => {
    nodes.forEach((n, j) => n.setAttribute("tabindex", j === i ? "0" : "-1"));
    (nodes[i] as unknown as HTMLElement).focus();
  };
  nodes.forEach((n, i) =>
    n.addEventListener("keydown", ((ev: KeyboardEvent) => {
      const to = next(i, ev.key);
      if (to === null || to === i) return;
      ev.preventDefault();
      focusTo(to);
    }) as EventListener),
  );
}

export const mekkoMarimekko: ChartModule = {
  key: "mekko.marimekko",
  // The shape is docs/CHART_CONTRACT.md §"The fourteen repository keys"; the wire projection's
  // `Shape` union has not been widened for the eight new keys yet, and widening
  // it is the orchestrator's job, not this module's.
  serves: "categorical×measure×width" as unknown as ChartModule["serves"],
  render,
};
