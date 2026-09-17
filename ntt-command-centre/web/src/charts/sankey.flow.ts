/**
 * sankey.flow — serves `source×target×measure`.
 * The stage path including its skips, and Industry → LOB → Outcome weighted by
 * GP.
 *
 * `d3-sankey` is not a dependency, so the layout is in this file: columns come
 * from `depth`, a node is sized by the LARGER of what enters and what leaves
 * it, the nodes in a column are stacked with a gap, and the order within each
 * column is then relaxed by a few weighted-median sweeps down and back up — the
 * same idea d3-sankey uses, and the only part of a Sankey that is not
 * arithmetic. Crossing reduction is the entire readability budget of this chart
 * type: two orderings of the same graph are the same numbers and a different
 * picture, so the sweep is not cosmetic.
 *
 * A NODE IS SIZED BY max(in, out), NOT BY in + out. The difference is the
 * leakage, and it is the fact the reader came for: what enters "Proposal /
 * Price Quote" on the stage path and never leaves it is the work that stopped
 * there. It is reported in the tooltip as "Stops here" rather than being folded
 * into the node's own thickness, where it would read as throughput.
 *
 * THE SKIPS ARE REAL EDGES. A link whose target is more than one column along
 * is drawn flying past the columns in between, because a line that jumps from
 * qualification straight to closed is the anomaly this chart exists to show. A
 * link that runs BACKWARD (target depth ≤ source depth) cannot be drawn as a
 * ribbon without implying a direction it does not have; it is dropped and
 * counted in a line under the chart rather than bent into a lie. Same for a
 * zero or negative value — a ribbon has no way to be less than nothing.
 *
 * Hue: a link takes its own `tone` when the server sent one, otherwise it
 * inherits its SOURCE node's tone, otherwise accent. Emphasis is opacity, never
 * hue — a hovered ribbon deepens and every other falls to DIM_OPACITY, a
 * hovered node lights every ribbon that touches it, and selection stays outline
 * plus dimming per the one-helper interaction rule.
 *
 * Only NODES carry a filter click, and only when the spec sends `clickDim`: a
 * ribbon is an intersection of two values of (sometimes) two different
 * dimensions, and there is no honest single filter for "Industry Retail into
 * LOB Networking". The server sends `clickDim` for the stage path, where every
 * node is a value of one dimension, and omits it for the three-dimension flow.
 */
import { max } from "d3-array";
import { select } from "d3-selection";
import type { Selection } from "d3-selection";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette, Tone } from "../theme/palette";
import { toneColor } from "../theme/palette";
import {
  CHAR_WIDTH,
  DIM_OPACITY,
  FONT,
  attachMark,
  createTooltip,
  formatValue,
  selectionState,
  tipHtml,
  truncate,
} from "./util";

/** Beyond this many marks the whole graph takes one tab stop. the palette-as-parameter rule. */
const DENSE = 20;
/** The node bar. Wide enough to hit, narrow enough not to become the chart. */
const NODE_W = 12;
/** Base vertical gap between nodes in a column; compressed when a column is crowded. */
const GAP = 14;
const MIN_NODE_H = 2;
const MIN_RIBBON = 1;
/** Sweeps of the ordering relaxation. Past ~6 the ordering stops changing. */
const SWEEPS = 6;
/** Advance of one character of the label font. Only ever used to decide
 *  whether a label fits, never to position one. */
const CHAR_W = FONT.label * CHAR_WIDTH;
/** A ribbon is translucent so overlaps stay legible; hover deepens it. */
const RIBBON_FILL = 0.34;
const RIBBON_HOT = 0.78;
/** Bezier control point placement. 0.5 is the d3-sankey house curve. */
const CURVE = 0.5;

type Format = "currency" | "percent" | "number" | "days";

const TONES: Tone[] = ["good", "warn", "danger", "neutral", "accent"];

// ---------------------------------------------------------------- the payload
// CONTRACT "Payload types — exact", field for field. Declared here rather than
// imported because `api/types.ts` is shared and this module does not own it.

interface SankeyNode {
  id: string;
  label: string;
  depth: number;
  tone?: Tone;
}

interface SankeyLink {
  source: string;
  target: string;
  value: number;
  tone?: Tone;
  note?: string;
}

interface SankeyPayload {
  nodes: SankeyNode[];
  links: SankeyLink[];
}

/**
 * The spec as this module needs to see it. `ChartSpec` in `api/types.ts` is
 * still typed for the first six repositories — `data` + `encoding` — so the
 * Sankey payload is read through a structural view instead of by editing a
 * shared file. Both wire shapes are accepted: `payload` when the server sends
 * the contract object, and flat link rows on `data` when it does not, in which
 * case the nodes are inferred from the links and `depth` is derived.
 */
interface SankeySpecView {
  payload?: Partial<SankeyPayload> | null;
  data?: unknown;
  encoding?: { x?: string; y?: string; series?: string; target?: string; label?: string };
  measureLabel?: string;
}

// ------------------------------------------------------------ the laid-out graph

interface LayoutNode {
  id: string;
  label: string;
  tone?: Tone;
  depth: number;
  /** Column index after the depths present are packed to consecutive columns. */
  col: number;
  inValue: number;
  outValue: number;
  /** max(in, out) — see the header. */
  value: number;
  x: number;
  y: number;
  h: number;
  in: LayoutLink[];
  out: LayoutLink[];
}

interface LayoutLink {
  source: LayoutNode;
  target: LayoutNode;
  value: number;
  tone?: Tone;
  note?: string;
  /** Ribbon thickness, and the centre y where it meets each end. */
  w: number;
  sy: number;
  ty: number;
}

function toneOf(raw: unknown): Tone | undefined {
  return typeof raw === "string" && (TONES as string[]).includes(raw) ? (raw as Tone) : undefined;
}

function textOf(raw: unknown): string | undefined {
  return typeof raw === "string" && raw !== "" ? raw : undefined;
}

// ------------------------------------------------------------------ the parse

function readPayload(spec: ChartSpec): {
  nodes: SankeyNode[];
  links: SankeyLink[];
  measureLabel: string;
} {
  const v = spec as unknown as SankeySpecView;
  const enc = v.encoding ?? {};
  const sKey = enc.x ?? "source";
  const tKey = enc.target ?? enc.series ?? "target";
  const vKey = enc.y ?? "value";

  // docs/CHART_CONTRACT.md §"sankey.flow" puts the object on `data`; `payload` is accepted as
  // an alias so the module is correct whichever way the seam is closed, and a
  // flat link array on `data` still works for a server that sends only links.
  const obj = (v.data && !Array.isArray(v.data) ? v.data : v.payload) as
    | { nodes?: unknown; links?: unknown }
    | undefined;
  const rawNodes = Array.isArray(obj?.nodes)
    ? (obj?.nodes as Record<string, unknown>[])
    : [];
  const rawLinks = Array.isArray(obj?.links)
    ? (obj?.links as Record<string, unknown>[])
    : Array.isArray(v.data)
      ? (v.data as Record<string, unknown>[])
      : [];

  const nodes: SankeyNode[] = [];
  const claimed = new Set<string>();
  for (const r of rawNodes) {
    const id = String(r.id ?? "");
    // A repeated id is one node stated twice, not two nodes: merging them is
    // the only reading under which the flows still balance.
    if (!id || claimed.has(id)) continue;
    claimed.add(id);
    const d = Number(r.depth);
    nodes.push({
      id,
      label: textOf(r.label) ?? id,
      // NaN means "unstated" and is resolved in buildGraph, never drawn.
      depth: Number.isFinite(d) ? d : NaN,
      tone: toneOf(r.tone),
    });
  }

  const links: SankeyLink[] = [];
  for (const r of rawLinks) {
    const source = String(r.source ?? r[sKey] ?? "");
    const target = String(r.target ?? r[tKey] ?? "");
    if (!source || !target) continue;
    links.push({
      source,
      target,
      value: Number(r.value ?? r[vKey]),
      tone: toneOf(r.tone),
      note: textOf(r.note),
    });
  }

  return { nodes, links, measureLabel: v.measureLabel ?? "Value" };
}

// ------------------------------------------------------------------ the graph

interface Graph {
  cols: LayoutNode[][];
  flows: LayoutLink[];
  /** Links the payload carried that a ribbon cannot represent. */
  dropped: number;
  negatives: number;
}

function buildGraph(nodes: SankeyNode[], links: SankeyLink[]): Graph {
  const byId = new Map<string, LayoutNode>();
  const make = (id: string, label: string, depth: number, tone?: Tone): LayoutNode => ({
    id,
    label,
    tone,
    depth,
    col: 0,
    inValue: 0,
    outValue: 0,
    value: 0,
    x: 0,
    y: 0,
    h: 0,
    in: [],
    out: [],
  });
  for (const n of nodes) byId.set(n.id, make(n.id, n.label, n.depth, n.tone));

  let dropped = 0;
  let negatives = 0;
  const usable: SankeyLink[] = [];
  for (const l of links) {
    if (!Number.isFinite(l.value) || l.value <= 0) {
      if (l.value < 0) negatives++;
      dropped++;
      continue;
    }
    // A self-link has no width to run along and no direction to run in.
    if (l.source === l.target) {
      dropped++;
      continue;
    }
    if (!byId.has(l.source)) byId.set(l.source, make(l.source, l.source, NaN));
    if (!byId.has(l.target)) byId.set(l.target, make(l.target, l.target, NaN));
    usable.push(l);
  }

  // Depth is the server's to state. It is derived only when at least one node
  // arrived without one — a graph half-placed by hand and half by longest path
  // would put the skip ribbons in the wrong columns, which is worse than
  // re-deriving the lot.
  const everyDepthStated = [...byId.values()].every((n) => Number.isFinite(n.depth));
  if (!everyDepthStated) {
    for (const n of byId.values()) n.depth = 0;
    let moved = true;
    // Longest path from the sources. The guard is what stops a cycle — which a
    // stage ladder should not contain — from widening the graph for ever.
    for (let guard = 0; moved && guard <= byId.size; guard++) {
      moved = false;
      for (const l of usable) {
        const s = byId.get(l.source);
        const t = byId.get(l.target);
        if (!s || !t) continue;
        if (t.depth < s.depth + 1) {
          t.depth = s.depth + 1;
          moved = true;
        }
      }
    }
  }

  const flows: LayoutLink[] = [];
  for (const l of usable) {
    const source = byId.get(l.source);
    const target = byId.get(l.target);
    if (!source || !target) continue;
    // Backward and lateral edges are not drawable as a left-to-right ribbon.
    if (!(target.depth > source.depth)) {
      dropped++;
      continue;
    }
    const f: LayoutLink = {
      source,
      target,
      value: l.value,
      tone: l.tone,
      note: l.note,
      w: 0,
      sy: 0,
      ty: 0,
    };
    source.out.push(f);
    target.in.push(f);
    flows.push(f);
  }

  for (const n of byId.values()) {
    n.inValue = n.in.reduce((a, f) => a + f.value, 0);
    n.outValue = n.out.reduce((a, f) => a + f.value, 0);
    n.value = Math.max(n.inValue, n.outValue);
  }

  // A node the surviving links never touch has no thickness; drawing it as a
  // hairline would invent a category the slice does not contain.
  const live = [...byId.values()].filter((n) => n.value > 0);
  const depths = [...new Set(live.map((n) => n.depth))].sort((a, b) => a - b);
  // The depths PRESENT become the columns, so a ladder that arrives as 0,2,5
  // draws three columns rather than three columns and three empty gutters. A
  // skip still spans more than one column, because it skips a populated step.
  const cols: LayoutNode[][] = depths.map(() => []);
  // Deterministic seed order — biggest first, ties by label — so the same
  // payload relaxes to the same picture on every render. the chart repository contract rule 5.
  live.sort((a, b) => b.value - a.value || a.label.localeCompare(b.label));
  for (const n of live) {
    n.col = depths.indexOf(n.depth);
    cols[n.col].push(n);
  }

  return { cols, flows, dropped, negatives };
}

// ----------------------------------------------------------------- the layout

interface Box {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * The value-weighted median of a set of positions. d3-sankey relaxes an
 * ordering against the linked nodes on the other side; the median is used here
 * rather than the mean because one very large flow should pull a node to ITS
 * row, and a mean lets a handful of slivers drag it back off again.
 */
function weightedMedian(pos: number[], w: number[]): number | null {
  if (!pos.length) return null;
  const order = pos.map((_, i) => i).sort((a, b) => pos[a] - pos[b]);
  const half = w.reduce((a, b) => a + b, 0) / 2;
  let acc = 0;
  for (const i of order) {
    acc += w[i];
    if (acc >= half) return pos[i];
  }
  return pos[order[order.length - 1]];
}

function layout(g: Graph, box: Box): void {
  const { cols, flows } = g;
  const inner = Math.max(1, box.bottom - box.top);
  const widest = max(cols, (c) => c.length) ?? 1;
  // The gap yields before the data does: a crowded column keeps its ribbons
  // sized truthfully and loses its whitespace instead.
  const gap = Math.max(3, Math.min(GAP, (inner * 0.4) / Math.max(1, widest - 1)));

  // One vertical scale for the whole chart — a per-column scale would make a
  // ribbon change thickness between its two ends for no reason the reader can see.
  let ky = Infinity;
  for (const c of cols) {
    const total = c.reduce((a, n) => a + n.value, 0);
    if (total <= 0) continue;
    ky = Math.min(ky, (inner - (c.length - 1) * gap) / total);
  }
  if (!Number.isFinite(ky) || ky <= 0) {
    const biggest = cols.reduce((m, c) => Math.max(m, c.reduce((a, n) => a + n.value, 0)), 1);
    ky = (inner * 0.8) / biggest;
  }

  const step = cols.length > 1 ? (box.right - box.left - NODE_W) / (cols.length - 1) : 0;
  cols.forEach((c, i) => {
    for (const n of c) {
      n.x = box.left + i * step;
      n.h = Math.max(MIN_NODE_H, n.value * ky);
    }
  });

  const centre = (n: LayoutNode) => n.y + n.h / 2;

  /** Push apart in the current order, then keep the column inside the plot. */
  const settle = (c: LayoutNode[]) => {
    let y = box.top;
    for (const n of c) {
      if (n.y < y) n.y = y;
      y = n.y + n.h + gap;
    }
    let floor = box.bottom;
    for (let i = c.length - 1; i >= 0; i--) {
      const n = c[i];
      if (n.y + n.h > floor) n.y = floor - n.h;
      floor = n.y - gap;
    }
  };

  // Initial stack: centred, in the seed order.
  for (const c of cols) {
    const total = c.reduce((a, n) => a + n.h, 0) + gap * Math.max(0, c.length - 1);
    let y = box.top + Math.max(0, (inner - total) / 2);
    for (const n of c) {
      n.y = y;
      y += n.h + gap;
    }
  }

  const sweep = (c: LayoutNode[], want: (n: LayoutNode) => number | null) => {
    const desired = new Map<LayoutNode, number>();
    for (const n of c) desired.set(n, want(n) ?? centre(n));
    c.sort((a, b) => (desired.get(a) ?? 0) - (desired.get(b) ?? 0) || a.label.localeCompare(b.label));
    for (const n of c) n.y = (desired.get(n) ?? 0) - n.h / 2;
    settle(c);
  };

  for (let s = 0; s < SWEEPS; s++) {
    for (let i = 1; i < cols.length; i++) {
      sweep(cols[i], (n) =>
        weightedMedian(n.in.map((f) => centre(f.source)), n.in.map((f) => f.value)),
      );
    }
    for (let i = cols.length - 2; i >= 0; i--) {
      sweep(cols[i], (n) =>
        weightedMedian(n.out.map((f) => centre(f.target)), n.out.map((f) => f.value)),
      );
    }
  }

  for (const f of flows) f.w = Math.max(MIN_RIBBON, f.value * ky);

  // Ribbons meet a node in the order of the node at the OTHER end, which is
  // what stops two flows from the same node crossing each other immediately.
  for (const c of cols) {
    for (const n of c) {
      n.out.sort((a, b) => centre(a.target) - centre(b.target) || a.target.label.localeCompare(b.target.label));
      n.in.sort((a, b) => centre(a.source) - centre(b.source) || a.source.label.localeCompare(b.source.label));
      let sy = n.y;
      for (const f of n.out) {
        f.sy = sy + f.w / 2;
        sy += f.w;
      }
      let ty = n.y;
      for (const f of n.in) {
        f.ty = ty + f.w / 2;
        ty += f.w;
      }
    }
  }
}

/** The ribbon: two mirrored cubics closed into one fillable band. */
function ribbon(x0: number, y0: number, x1: number, y1: number, w: number): string {
  const h = w / 2;
  const c0 = x0 + (x1 - x0) * CURVE;
  const c1 = x1 - (x1 - x0) * CURVE;
  return (
    `M${x0},${y0 - h}` +
    `C${c0},${y0 - h} ${c1},${y1 - h} ${x1},${y1 - h}` +
    `L${x1},${y1 + h}` +
    `C${c1},${y1 + h} ${c0},${y0 + h} ${x0},${y0 + h}Z`
  );
}

// ---------------------------------------------------------------- the numbers
// docs/CHART_CONTRACT.md: `format: "days"` is a bare integer with a `d` suffix. The
// shared formatter predates that case, so the suffix is added here and every
// other format still goes through it. Nothing is reimplemented.

function fmt(v: number, f?: Format): string {
  return f === "days" ? `${Math.round(v)}d` : formatValue(v, f);
}

function share(part: number, whole: number): string {
  return whole > 0 ? `${((part / whole) * 100).toFixed(1)}%` : "—";
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  const { nodes, links, measureLabel } = readPayload(spec);
  const g = buildGraph(nodes, links);
  const { cols, flows } = g;
  const format: Format = spec.format ?? "currency";

  const width = Math.max(260, opts.width);
  const height = Math.max(220, opts.height || 300);
  const noteH = g.dropped > 0 ? 16 : 0;

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr(
      "aria-label",
      `${spec.title}. Flow diagram: ${cols.reduce((a, c) => a + c.length, 0)} nodes in ` +
        `${cols.length} columns, ${flows.length} flows.` +
        (flows.length + cols.reduce((a, c) => a + c.length, 0) > DENSE
          ? " Use the arrow keys to move between marks."
          : ""),
    );

  // Degrade, never throw. Each of the three ways this payload can be undrawable
  // says which one it was, because "no rows" and "every row is negative" are
  // different problems and only one of them is the reader's.
  if (!flows.length || cols.length < 2) {
    const message = !links.length
      ? "No flows in this slice"
      : g.negatives > 0 && g.negatives >= links.length
        ? "No flow to draw — every value in this slice is negative"
        : g.dropped >= links.length
          ? "No forward flow to draw — every link is zero, negative or runs backwards"
          : "No flows in this slice";
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text(message);
    return () => svg.selectAll("*").remove();
  }

  // Margins are the label gutters: the end columns hang their labels outside
  // the plot, so the plot has to make room for them rather than the other way
  // round. Everything between is labelled inside, against its own node.
  const longest = (c: LayoutNode[]) => c.reduce((a, n) => Math.max(a, n.label.length), 0);
  const gutter = Math.max(48, width * 0.26);
  const box: Box = {
    left: Math.min(gutter, Math.max(36, longest(cols[0]) * CHAR_W + 12)),
    right: width - Math.min(gutter, Math.max(36, longest(cols[cols.length - 1]) * CHAR_W + 12)),
    top: 14,
    bottom: height - 14 - noteH,
  };
  layout(g, box);

  const tip = createTooltip(p);
  const cleanups: (() => void)[] = [];
  const flat = cols.flat();
  const step = cols.length > 1 ? (box.right - box.left - NODE_W) / (cols.length - 1) : 0;

  // Selection is read once: a link belongs to the selection when either end
  // does, so selecting a stage keeps the flows into and out of it lit.
  const anySelected = selectionState(opts, spec.clickDim, "").anySelected;
  const isSelected = (id: string) => selectionState(opts, spec.clickDim, id).isSelected;
  const nodeDim = (n: LayoutNode) => anySelected && !isSelected(n.id);
  const linkDim = (f: LayoutLink) =>
    anySelected && !isSelected(f.source.id) && !isSelected(f.target.id);

  const linkFill = (f: LayoutLink) =>
    f.tone ? toneColor(p, f.tone) : f.source.tone ? toneColor(p, f.source.tone) : p.accent;
  const nodeFill = (n: LayoutNode) => (n.tone ? toneColor(p, n.tone) : p.accent);

  // ------------------------------------------------------------- the ribbons
  const ribbons = svg.append("g");
  const paths: Selection<SVGPathElement, unknown, null, undefined>[] = flows.map((f) =>
    ribbons
      .append("path")
      .attr("d", ribbon(f.source.x + NODE_W, f.sy, f.target.x, f.ty, f.w))
      .attr("fill", linkFill(f))
      .attr("fill-opacity", RIBBON_FILL)
      .attr("stroke", "none")
      .attr("opacity", linkDim(f) ? DIM_OPACITY : 1),
  );

  // --------------------------------------------------------------- the nodes
  const bars = svg.append("g");
  const rects: Selection<SVGRectElement, unknown, null, undefined>[] = flat.map((n) => {
    const holder = bars.append("g");
    const rect = holder
      .append("rect")
      .attr("x", n.x)
      .attr("y", n.y)
      .attr("width", NODE_W)
      .attr("height", Math.max(MIN_NODE_H, n.h))
      .attr("rx", 2)
      .attr("fill", nodeFill(n))
      .attr("opacity", nodeDim(n) ? DIM_OPACITY : 1);
    // Selection = outline + dimming of the rest. Never a hue change.
    if (isSelected(n.id)) {
      holder
        .append("rect")
        .attr("x", n.x - 2.5)
        .attr("y", n.y - 2.5)
        .attr("width", NODE_W + 5)
        .attr("height", Math.max(MIN_NODE_H, n.h) + 5)
        .attr("rx", 4)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5);
    }
    return rect;
  });

  // -------------------------------------------------------------- the labels
  // Outside the plot for the two end columns, inside it for everything between,
  // where a halo keeps the text legible over whatever ribbon runs behind it.
  const labels = svg.append("g").attr("pointer-events", "none");
  const insideChars = Math.max(4, Math.floor((step - NODE_W - 14) / CHAR_W));
  flat.forEach((n) => {
    const last = n.col === cols.length - 1;
    const end = n.col === 0 || last;
    const room = end
      ? Math.floor(((n.col === 0 ? box.left : width - box.right) - 12) / CHAR_W)
      : insideChars;
    // Between the columns, a label only earns its place if it fits; the tooltip
    // carries the name either way.
    if (!end && step - NODE_W < 56) return;
    const dimmed = nodeDim(n);
    const t = labels
      .append("text")
      .attr("x", n.col === 0 ? n.x - 8 : n.x + NODE_W + 7)
      .attr("y", n.y + n.h / 2 + (end && n.h >= 26 ? -1.5 : 4.5))
      .attr("text-anchor", n.col === 0 ? "end" : "start")
      .attr("fill", dimmed ? p.muted : p.text)
      .attr("font-size", FONT.label)
      .attr("font-weight", isSelected(n.id) ? 700 : 500)
      .text(truncate(n.label, Math.max(3, room)));
    if (!end) {
      t.attr("stroke", p.surface).attr("stroke-width", 3).attr("paint-order", "stroke");
    }
    // The end columns are the punchline of a Sankey, so they carry their figure
    // where there is height for a second line.
    if (end && n.h >= 26) {
      labels
        .append("text")
        .attr("x", n.col === 0 ? n.x - 8 : n.x + NODE_W + 7)
        .attr("y", n.y + n.h / 2 + 12)
        .attr("text-anchor", n.col === 0 ? "end" : "start")
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .attr("font-variant-numeric", "tabular-nums")
        .text(fmt(n.value, format));
    }
  });

  // ------------------------------------------------------------- emphasis
  // Hover is opacity, not hue: the tones on this chart are the server's
  // classification and a hover must not overwrite one.
  const restore = () => {
    flows.forEach((f, i) =>
      paths[i]
        .attr("opacity", linkDim(f) ? DIM_OPACITY : 1)
        .attr("fill-opacity", RIBBON_FILL)
        .attr("stroke", "none")
        .attr("stroke-width", 0),
    );
    flat.forEach((n, i) => rects[i].attr("opacity", nodeDim(n) ? DIM_OPACITY : 1).attr("stroke", "none"));
  };
  const spotlight = (lit: (f: LayoutLink) => boolean) => {
    flows.forEach((f, i) => {
      const on = lit(f);
      paths[i]
        .attr("opacity", on ? 1 : DIM_OPACITY)
        .attr("fill-opacity", on ? RIBBON_HOT : RIBBON_FILL)
        .attr("stroke", on ? p.text : "none")
        .attr("stroke-width", on ? 0.75 : 0);
    });
    const touched = new Set<string>();
    flows.filter(lit).forEach((f) => {
      touched.add(f.source.id);
      touched.add(f.target.id);
    });
    flat.forEach((n, i) =>
      rects[i]
        .attr("opacity", touched.has(n.id) ? 1 : DIM_OPACITY)
        .attr("stroke", touched.has(n.id) ? p.text : "none")
        .attr("stroke-width", touched.has(n.id) ? 1 : 0),
    );
  };

  // ---------------------------------------------------------------- the marks
  // Walked in reading order — a node, then the flows leaving it — so the arrow
  // keys trace the graph rather than the DOM.
  const hits = svg.append("g");
  const marks: SVGGraphicsElement[] = [];

  const nodeMark = (n: LayoutNode) => {
    const hitH = Math.max(n.h, 12);
    const hit = hits
      .append("rect")
      .attr("x", n.x - 5)
      .attr("y", n.y + n.h / 2 - hitH / 2)
      .attr("width", NODE_W + 10)
      .attr("height", hitH)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    const colTotal = cols[n.col].reduce((a, m) => a + m.value, 0);
    const rows: [string, string][] = [[measureLabel, fmt(n.value, format)]];
    if (n.in.length) rows.push(["In", fmt(n.inValue, format)]);
    if (n.out.length) rows.push(["Out", fmt(n.outValue, format)]);
    // The leak. Only where there IS an onward path — a terminal node keeping
    // everything that reaches it is the end of the flow, not a loss.
    if (n.in.length && n.out.length && n.inValue > n.outValue) {
      rows.push(["Stops here", fmt(n.inValue - n.outValue, format)]);
    }
    rows.push(["Share of this column", share(n.value, colTotal)]);

    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(
        p,
        n.label,
        rows,
        spec.clickDim ? `Click to filter ${spec.clickDim} = ${n.id}` : undefined,
      ),
      aria:
        `${n.label}, ${fmt(n.value, format)}, ` +
        `${n.in.length} flow${n.in.length === 1 ? "" : "s"} in, ` +
        `${n.out.length} flow${n.out.length === 1 ? "" : "s"} out`,
      dim: spec.clickDim,
      value: n.id,
      opts,
      // A node lights everything that touches it: that is the node's own story.
      onEnter: () => spotlight((f) => f.source === n || f.target === n),
      onLeave: restore,
    });
    marks.push(hit);
  };

  const linkMark = (f: LayoutLink) => {
    const path = paths[flows.indexOf(f)];
    const hop = f.target.col - f.source.col;
    const rows: [string, string][] = [
      [measureLabel, fmt(f.value, format)],
      [`Share of ${truncate(f.source.label, 16)} out`, share(f.value, f.source.outValue)],
      [`Share of ${truncate(f.target.label, 16)} in`, share(f.value, f.target.inValue)],
    ];
    // A skip is the fact this chart exists to show, so it is named, not implied
    // by the length of the ribbon.
    if (hop > 1) rows.push(["Path", `skips ${hop - 1} step${hop - 1 === 1 ? "" : "s"}`]);

    attachMark(path.node() as SVGGraphicsElement, {
      tip,
      palette: p,
      html: tipHtml(p, `${f.source.label} → ${f.target.label}`, rows, f.note),
      aria:
        `${f.source.label} to ${f.target.label}, ${fmt(f.value, format)}` +
        (hop > 1 ? `, skipping ${hop - 1} step${hop - 1 === 1 ? "" : "s"}` : "") +
        (f.note ? `. ${f.note}` : ""),
      // No dim/value: a ribbon is an intersection of two node values, and on
      // Industry → LOB → Outcome those are two different dimensions. There is
      // no single filter it could honestly set, so it hovers and reads but does
      // not click. The nodes at either end carry the filter.
      opts,
      onEnter: () => spotlight((other) => other === f),
      onLeave: restore,
    });
    marks.push(path.node() as SVGGraphicsElement);
  };

  cols.forEach((c) => {
    [...c]
      .sort((a, b) => a.y - b.y)
      .forEach((n) => {
        nodeMark(n);
        [...n.out].forEach(linkMark);
      });
  });

  // ------------------------------------------------- one tab stop when dense
  // the palette-as-parameter rule. `util.ts` exposes no `markGroup()` yet and every shared
  // file belongs to another module this cycle, so the roving tabindex lives
  // here and touches only this module's nodes: the group is entered once, the
  // arrow keys walk it, Home and End jump to the ends. `attachMark` still owns
  // hover, focus, Enter, Space and Escape — this rewrites the tab ORDER only.
  if (marks.length > DENSE) {
    hits
      .attr("role", "group")
      .attr("aria-label", `${marks.length} marks. Use the arrow keys to move between them.`);
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

  // --------------------------------------------------------- what was dropped
  // Said on the card, not in the console: a Sankey that quietly omits a link is
  // a Sankey whose columns do not add up and whose reader cannot tell.
  if (g.dropped > 0) {
    svg
      .append("text")
      .attr("x", box.left)
      .attr("y", height - 4)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(
        `${g.dropped} ${g.dropped === 1 ? "link" : "links"} not drawn — ` +
          `${g.negatives > 0 ? "negative, " : ""}zero-value or running backwards`,
      );
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
 * modules serve, and that file belongs to the registry wiring rather than to
 * this module, so CONTRACT's own shape string is asserted here. The assertion
 * becomes a no-op the moment `source×target×measure` joins the union.
 */
const SERVES = "source×target×measure" as string as ChartModule["serves"];

export const sankeyFlow: ChartModule = {
  key: "sankey.flow",
  serves: SERVES,
  render,
};
