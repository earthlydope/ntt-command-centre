/**
 * treemap.nested — serves `hierarchy×measure`.
 * Account concentration. One account carries 14.5% of all ACV GP and 362 carry
 * the rest; a ranked bar chart of that cannot be read on one screen, and a pie
 * of it is a pie. Area is the argument here, so area is what carries the number.
 *
 * Squarified (`treemapSquarify`) because the alternative tilings produce
 * slivers, and a sliver cannot be labelled, hovered or clicked — which is the
 * whole job.
 *
 * **The server has already collapsed the tail.** CONTRACT: "Server applies
 * top-N + Other." So this module never re-ranks and never re-collapses; it
 * draws what it is given. The "Other" tile is drawn in a muted neutral and is
 * deliberately NOT a filter — clicking it would set a filter to the literal
 * string "Other", which matches no account.
 *
 * Shade is reinforcement, not the encoding: `tone` drives the fill where the
 * semantic layer has classified a node (business rules live in `api/semantic/`,
 * not in a D3 module), and otherwise a sequential ramp by value repeats what
 * the area already says. That is why there is no colour legend — area is the
 * scale, and a second scale bar would imply a second reading.
 *
 * Parent headers are labels, not marks. A leaf and its parent belong to two
 * different dimensions and a spec carries exactly one `clickDim`, so making the
 * header a filter would have to guess which dimension it sets. The leaves are
 * the marks; the header states what they sit inside.
 */
import { max } from "d3-array";
import { stratify, treemap, treemapSquarify } from "d3-hierarchy";
import type { HierarchyNode, HierarchyRectangularNode } from "d3-hierarchy";
import { interpolateRgb } from "d3-interpolate";
import { scaleLinear } from "d3-scale";
import { select } from "d3-selection";
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

/** The parent's label strip. A child is laid out below it, never behind it. */
const HEADER = 20;
const PAD = 2;
/** Room under the plot for the one-line honesty note. */
const FOOT = 22;
/** Below a pixel a tile is invisible, so it is neither drawn nor focusable — a
 *  tab stop on something a sighted user cannot see is not an affordance. */
const MIN_TILE = 1;
/** More than this many marks and one tab stop each becomes a keyboard trap.
 *  the palette-as-parameter rule; `util.ts` has no `markGroup()` yet, so the roving
 *  tabindex is implemented below and stays inside this module. */
const DENSE = 20;

const TONES = new Set(["accent", "good", "warn", "danger", "neutral"]);

/** docs/CHART_CONTRACT.md §"treemap.nested": flat list, `parent` builds the hierarchy. */
interface TreeNode {
  id: string;
  label: string;
  parent: string | null;
  value: number;
  tone?: Tone;
  secondary?: number;
}

/**
 * The payload is `{ nodes: TreeNode[] }`. `ChartSpec.data` is still typed as a
 * row array because every shape that shipped first was one, so the object form
 * is read through `unknown` here rather than by widening the wire type from
 * inside a chart module — `api/types.ts` is generated from real payloads and is
 * not a chart's to edit. Both spellings are accepted, so this module renders
 * correctly whichever way the seam is closed.
 */
function readNodes(spec: ChartSpec): TreeNode[] {
  const raw: unknown = spec.data;
  const list: unknown[] = Array.isArray(raw)
    ? raw
    : Array.isArray((raw as { nodes?: unknown } | null)?.nodes)
      ? (raw as { nodes: unknown[] }).nodes
      : [];
  const out: TreeNode[] = [];
  for (const row of list) {
    if (!row || typeof row !== "object") continue;
    const d = row as Record<string, unknown>;
    const id = String(d.id ?? "").trim();
    if (!id) continue;
    const rawParent = d.parent;
    const parent =
      rawParent === null || rawParent === undefined || String(rawParent).trim() === ""
        ? null
        : String(rawParent).trim();
    const tone = String(d.tone ?? "");
    out.push({
      id,
      label: String(d.label ?? id),
      parent,
      value: Number(d.value),
      tone: TONES.has(tone) ? (tone as Tone) : undefined,
      secondary:
        d.secondary === null || d.secondary === undefined ? undefined : Number(d.secondary),
    });
  }
  return out;
}

/**
 * The collapsed tail. The contract gives no `isOther` flag, so it is recognised
 * by the two spellings the server can produce — an `other` id (optionally
 * namespaced, e.g. `account:other`) or a label that is exactly "Other" or
 * "Other (312 accounts)". Deliberately tight: "Other Networking" is a plausible
 * real account name and must stay a real, clickable tile.
 */
function isOtherNode(n: TreeNode): boolean {
  const id = n.id.trim().toLowerCase();
  if (id === "other" || /[:|/_-]other$/.test(id)) return true;
  return /^other(\s*\(|$)/i.test(n.label.trim());
}

/** Relative luminance of a `#rrggbb` or `rgb(...)` fill. The ramp interpolates
 *  to `rgb()` strings and `tone` resolves to hex, so a label has to pick its
 *  ink from the fill it actually landed on rather than from a ramp position. */
function luminance(color: string): number {
  let r = 0;
  let g = 0;
  let b = 0;
  const hex = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (hex) {
    const n = parseInt(hex[1], 16);
    r = (n >> 16) & 255;
    g = (n >> 8) & 255;
    b = n & 255;
  } else {
    const m = /rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(color);
    if (!m) return 0.5;
    r = Number(m[1]);
    g = Number(m[2]);
    b = Number(m[3]);
  }
  const lin = (c: number): number => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

interface Mark {
  node: SVGGraphicsElement;
  cx: number;
  cy: number;
}

/**
 * Arrow-key navigation over a 2-D field of tiles. Array order is layout order,
 * which after a squarified pass is not reading order, so "right" has to mean
 * the nearest tile actually to the right rather than the next one in the array.
 */
function nearest(marks: Mark[], from: number, dx: number, dy: number): number {
  const a = marks[from];
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < marks.length; i += 1) {
    if (i === from) continue;
    const vx = marks[i].cx - a.cx;
    const vy = marks[i].cy - a.cy;
    const along = vx * dx + vy * dy;
    if (along <= 0) continue;
    const across = Math.abs(vx * dy - vy * dx);
    // too far off-axis to be "that way" at all
    if (across > along * 1.6) continue;
    const d = along + across * 1.8;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  // Rule 6: both come from the wrapper's ResizeObserver, never from a constant.
  const width = Math.max(240, opts.width);
  const height = Math.max(200, Math.round(opts.height || 260));

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr("aria-label", spec.title);

  /** Rules 7 and 8: degrade to a centred message that names the problem. Never
   *  a throw, and never a drawing that would be a lie about the data. */
  const fail = (...lines: string[]): Teardown => {
    svg.selectAll("*").remove();
    lines.forEach((line, i) =>
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", height / 2 - (lines.length - 1) * 10 + i * 19)
        .attr("text-anchor", "middle")
        .attr("fill", p.muted)
        .attr("font-size", i === 0 ? FONT.title : FONT.tick)
        .text(line),
    );
    return () => svg.selectAll("*").remove();
  };

  const nodes = readNodes(spec);
  if (!nodes.length) return fail("No rows in this slice");

  // stratify throws on a duplicate id, a dangling parent or a cycle. Each is a
  // different data defect and each gets its own sentence, because "could not
  // render" tells the person holding the deck nothing at all.
  const byId = new Map<string, TreeNode>();
  for (const n of nodes) {
    if (byId.has(n.id)) {
      return fail("This hierarchy has a duplicate id", `"${truncate(n.label, 42)}" appears twice`);
    }
    byId.set(n.id, n);
  }
  const orphan = nodes.find((n) => n.parent !== null && !byId.has(n.parent));
  if (orphan) {
    return fail(
      "This hierarchy is incomplete",
      `"${truncate(orphan.label, 42)}" names a parent that is not in the list`,
    );
  }

  // Rule 8. A treemap has no way to draw a negative area, and drawing |v|
  // instead would make a loss look like a contribution.
  const negatives = nodes.filter((n) => Number.isFinite(n.value) && n.value < 0);
  if (negatives.length) {
    return fail(
      "A treemap cannot draw negative values",
      `${negatives.length} of ${nodes.length} nodes are below zero`,
    );
  }
  const unusable = nodes.filter((n) => !Number.isFinite(n.value));
  if (unusable.length) {
    return fail(
      "Some values are not numbers",
      `${unusable.length} of ${nodes.length} nodes have no usable value`,
    );
  }

  // One level or two, both served: a flat list is a list of roots, and a list
  // of roots is a single level once they hang off a virtual root never drawn.
  const ROOT_ID = "__treemap_virtual_root__";
  const parentIds = new Set(nodes.filter((n) => n.parent).map((n) => n.parent as string));
  const seeded: TreeNode[] = [
    { id: ROOT_ID, label: spec.title, parent: null, value: 0 },
    ...nodes.map((n) => ({ ...n, parent: n.parent ?? ROOT_ID })),
  ];

  let h: HierarchyNode<TreeNode>;
  try {
    h = stratify<TreeNode>()
      .id((d) => d.id)
      .parentId((d) => d.parent)(seeded);
  } catch {
    return fail("This hierarchy could not be built", "a node is its own ancestor");
  }

  // A parent's own `value` is ignored on purpose: the children's sum is the
  // only total that reconciles against the tiles on screen. Where the two
  // disagree the server has a rounding bug, and inventing a third number here
  // would hide it.
  h.sum((d) => (d.id === ROOT_ID || parentIds.has(d.id) ? 0 : Math.max(0, d.value))).sort(
    (a, b) => (b.value ?? 0) - (a.value ?? 0),
  );

  const total = h.value ?? 0;
  if (total <= 0) return fail("Every value in this slice is zero");

  const plotH = Math.max(120, height - FOOT);
  const layout = treemap<TreeNode>()
    .tile(treemapSquarify)
    .size([width, plotH])
    .paddingInner(PAD)
    .paddingOuter((d) => (d.depth > 0 && d.children ? PAD : 0))
    // paddingOuter sets all four sides; paddingTop then reclaims the strip the
    // parent's own label sits in. The virtual root keeps a top of 0.
    .paddingTop((d) => (d.depth > 0 && d.children ? HEADER : 0))
    .round(true);
  const laid = layout(h);

  const groups = laid.descendants().filter((d) => d.depth > 0 && !!d.children);
  const leaves = laid.leaves();
  if (!leaves.length) return fail("No rows in this slice");

  const maxLeaf = max(leaves, (d) => d.value ?? 0) ?? 0;
  // sqrt, as in heat.grid: a linear shade collapses the long tail into one
  // colour, and the tail is exactly what a concentration chart is about.
  const t = scaleLinear()
    .domain([0, Math.sqrt(maxLeaf || 1)])
    .range([0, 1])
    .clamp(true);
  const ramp = interpolateRgb(p.heatLow, p.accent);
  const anyTone = nodes.some((n) => n.tone);
  const fillOf = (d: HierarchyRectangularNode<TreeNode>): string => {
    if (isOtherNode(d.data)) return p.track;
    if (d.data.tone) return toneColor(p, d.data.tone);
    return ramp(t(Math.sqrt(d.value ?? 0)));
  };

  // Ink is picked from the fill the label actually landed on, because `tone`
  // fills do not sit anywhere on the ramp. 0.19 is not a guess: it is the
  // luminance at which this theme's two inks score the same WCAG ratio against
  // the same fill, so either side of it the better of the two always wins. It
  // reproduces heat.grid's choice of `onAccent` on a saturated accent cell, and
  // it is what keeps a `warn` tile and a `danger` tile legible in both themes.
  const onDarkFill = p.name === "dark" ? p.text : p.onAccent;
  const onLightFill = p.name === "dark" ? p.onAccent : p.text;
  const inkOn = (fill: string): string => (luminance(fill) < 0.19 ? onDarkFill : onLightFill);

  // `secondary` arrives unlabelled in the contract. `countBasis` is the only
  // thing on the spec that says what a second number per node means, so it is
  // named from that and left as a bare "Secondary" where the spec is silent —
  // rather than asserting a grain the server never claimed.
  const secondLabel =
    spec.countBasis === "opportunities"
      ? "Opportunities"
      : spec.countBasis === "lines"
        ? "Lines"
        : "Secondary";
  const secondText = (n: number): string =>
    spec.countBasis ? Math.round(n).toLocaleString("en-US") : formatValue(n, spec.format);

  const tip = createTooltip(p);
  const gGroups = svg.append("g");
  const gLeaves = svg
    .append("g")
    .attr("role", "group")
    .attr(
      "aria-label",
      `${leaves.length} tiles, largest first. Use the arrow keys to move between them.`,
    );

  // ------------------------------------------------------ the parent frames
  // Drawn first, so every leaf paints over its own parent's fill and border.
  groups.forEach((d) => {
    const w = d.x1 - d.x0;
    const hgt = d.y1 - d.y0;
    if (w < 6 || hgt < HEADER) return;
    // A group dims only when the selection lives in some *other* group — the
    // selection is on a leaf, so the group reads it off its own leaves rather
    // than pretending the group label is a filter value.
    const holdsSelection = d
      .leaves()
      .some((l) => selectionState(opts, spec.clickDim, l.data.label).isSelected);
    const { anySelected } = selectionState(opts, spec.clickDim, d.data.label);
    const dimmed = anySelected && !holdsSelection;

    const g = gGroups.append("g").attr("opacity", dimmed ? DIM_OPACITY : 1);
    g.append("rect")
      .attr("x", d.x0)
      .attr("y", d.y0)
      .attr("width", w)
      .attr("height", hgt)
      .attr("rx", 4)
      .attr("fill", p.surface2)
      .attr("stroke", p.line)
      .attr("stroke-width", 1)
      .attr("aria-hidden", "true");

    if (w <= 48) return;
    const totalW = textWidth(9, FONT.note);
    const showTotal = w > 132;
    g.append("text")
      .attr("x", d.x0 + 6)
      .attr("y", d.y0 + HEADER - 6)
      .attr("fill", p.text2)
      .attr("font-size", FONT.label)
      .attr("font-weight", 650)
      .attr("letter-spacing", "0.02em")
      .attr("pointer-events", "none")
      .text(truncateLabel(d.data.label, w - 14 - (showTotal ? totalW : 0)));
    if (showTotal) {
      g.append("text")
        .attr("x", d.x1 - 6)
        .attr("y", d.y0 + HEADER - 6)
        .attr("text-anchor", "end")
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .attr("font-variant-numeric", "tabular-nums")
        .attr("pointer-events", "none")
        .text(formatValue(d.value ?? 0, spec.format));
    }
  });

  // -------------------------------------------------------------- the tiles
  const marks: Mark[] = [];
  let otherPresent = false;
  let biggest = leaves[0];

  leaves.forEach((d) => {
    if ((d.value ?? 0) > (biggest.value ?? 0)) biggest = d;
    const w = d.x1 - d.x0;
    const hgt = d.y1 - d.y0;
    if (w < MIN_TILE || hgt < MIN_TILE) return;

    const other = isOtherNode(d.data);
    if (other) otherPresent = true;
    const v = d.value ?? 0;
    const share = (v / total) * 100;
    // The filter value is the label, not the id: the chip row carries display
    // strings, and a filter the chips cannot echo back is one you cannot clear.
    const { anySelected, isSelected } = selectionState(opts, spec.clickDim, d.data.label);
    const dimmed = anySelected && !isSelected;
    const base = fillOf(d);
    const stroke = other ? p.line : p.cellStroke;

    const g = gLeaves.append("g");
    const rect = g
      .append("rect")
      .attr("x", d.x0)
      .attr("y", d.y0)
      .attr("width", w)
      .attr("height", hgt)
      .attr("rx", Math.min(3, w / 2, hgt / 2))
      .attr("fill", base)
      .attr("stroke", stroke)
      .attr("stroke-width", 1)
      .attr("opacity", dimmed ? DIM_OPACITY : 1);

    // Rule 3: selection is outline plus dimming of the rest. Never a hue change.
    if (isSelected) {
      g.append("rect")
        .attr("x", d.x0 - 1.5)
        .attr("y", d.y0 - 1.5)
        .attr("width", w + 3)
        .attr("height", hgt + 3)
        .attr("rx", 4.5)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5)
        .attr("pointer-events", "none");
    }

    // Labels only where the tile fits one. The value is always in the tooltip
    // and always in the aria-label, so a tile too small to write on is still
    // readable — by pointer and by screen reader alike.
    const ink = other ? p.muted : inkOn(base);
    const size = w > 150 && hgt > 46 ? FONT.title : FONT.label;
    const chars = charsIn(w - 10, size);
    if (hgt >= size + 9 && chars >= 4) {
      g.append("text")
        .attr("x", d.x0 + 5)
        .attr("y", d.y0 + 5 + size * 0.85)
        .attr("fill", ink)
        .attr("font-size", size)
        .attr("font-weight", isSelected ? 750 : 600)
        .attr("opacity", dimmed ? 0.55 : 1)
        .attr("pointer-events", "none")
        .text(truncate(d.data.label, chars));
      if (hgt >= size + 27) {
        g.append("text")
          .attr("x", d.x0 + 5)
          .attr("y", d.y0 + 20 + size * 0.85)
          .attr("fill", ink)
          .attr("font-size", FONT.note)
          .attr("font-variant-numeric", "tabular-nums")
          .attr("opacity", dimmed ? 0.45 : 0.82)
          .attr("pointer-events", "none")
          .text(
            truncateLabel(`${formatValue(v, spec.format)} · ${share.toFixed(1)}%`, w - 10, FONT.note),
          );
      }
    }

    const rows: [string, string][] = [
      [spec.format === "percent" ? "Value" : "Amount", formatValue(v, spec.format)],
      ["Share of shown", `${share.toFixed(1)}%`],
    ];
    if (d.parent && d.parent.depth > 0) rows.push(["Within", d.parent.data.label]);
    if (d.data.secondary !== undefined && Number.isFinite(d.data.secondary)) {
      rows.push([secondLabel, secondText(d.data.secondary)]);
    }

    // The collapsed tail is informative, not actionable: it carries no single
    // dimension value, so it gets a tooltip and a focus stop but no `dim` —
    // which is what makes attachMark render it as an image, not a button.
    const clickable = !other && !!spec.clickDim;
    attachMark(rect.node() as SVGGraphicsElement, {
      tip,
      palette: p,
      html: tipHtml(
        p,
        d.data.label,
        rows,
        other
          ? "The tail, already collapsed by the server — not a filter"
          : clickable
            ? `Click to filter ${spec.clickDim} = ${d.data.label}`
            : undefined,
      ),
      aria: `${d.data.label}, ${formatValue(v, spec.format)}, ${share.toFixed(1)}% of shown${
        other ? ", the collapsed tail" : ""
      }`,
      dim: clickable ? spec.clickDim : undefined,
      value: clickable ? d.data.label : undefined,
      opts,
      onEnter: () => rect.attr("stroke", p.text).attr("stroke-width", 1.6).attr("opacity", 1),
      onLeave: () =>
        rect
          .attr("stroke", stroke)
          .attr("stroke-width", 1)
          .attr("opacity", dimmed ? DIM_OPACITY : 1),
    });

    marks.push({
      node: rect.node() as SVGGraphicsElement,
      cx: (d.x0 + d.x1) / 2,
      cy: (d.y0 + d.y1) / 2,
    });
  });

  // ---------------------------------------------------- one tab stop, arrows
  // the palette-as-parameter rule. A 40-tile account treemap is 40 tab stops otherwise and
  // a keyboard user cannot get past the card. Roving tabindex: the group holds
  // exactly one focusable tile, and the arrows move which one that is.
  const holder = gLeaves.node();
  if (holder && marks.length) {
    const dense = marks.length > DENSE;
    if (dense) {
      marks.forEach((m, i) => m.node.setAttribute("tabindex", i === 0 ? "0" : "-1"));
    }
    let cur = 0;
    const focusAt = (i: number): void => {
      if (i < 0 || i >= marks.length) return;
      if (dense) {
        marks[cur].node.setAttribute("tabindex", "-1");
        marks[i].node.setAttribute("tabindex", "0");
      }
      cur = i;
      (marks[i].node as unknown as HTMLElement).focus();
    };
    holder.addEventListener("focusin", (ev: Event) => {
      const i = marks.findIndex((m) => m.node === ev.target);
      if (i >= 0) cur = i;
    });
    holder.addEventListener("keydown", (ev: Event) => {
      const e = ev as KeyboardEvent;
      if (e.key === "Home") {
        e.preventDefault();
        focusAt(0);
        return;
      }
      if (e.key === "End") {
        e.preventDefault();
        focusAt(marks.length - 1);
        return;
      }
      const step: Record<string, [number, number]> = {
        ArrowRight: [1, 0],
        ArrowLeft: [-1, 0],
        ArrowDown: [0, 1],
        ArrowUp: [0, -1],
      };
      const dir = step[e.key];
      if (!dir) return;
      e.preventDefault();
      const next = nearest(marks, cur, dir[0], dir[1]);
      if (next >= 0) focusAt(next);
    });
  }

  // ----------------------------------------------------------- the footnote
  // No colour legend, by design. What does need saying is the concentration
  // itself — it is the reason this shape was chosen over a bar chart.
  const topShare = ((biggest.value ?? 0) / total) * 100;
  const foot = svg.append("g").attr("transform", `translate(0,${plotH + 15})`);
  const note = otherPresent
    ? "Other = the collapsed tail, not a filter"
    : anyTone
      ? "Colour = status"
      : "Shade = size";
  const noteW = width >= 420 ? textWidth(note.length, FONT.note) + 10 : 0;
  foot
    .append("text")
    .attr("x", 1)
    .attr("y", 0)
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    .text(
      truncate(
        `Largest tile: ${biggest.data.label} — ${topShare.toFixed(1)}% of shown`,
        Math.max(8, charsIn(width - noteW - 6, FONT.note)),
      ),
    );
  if (noteW) {
    foot
      .append("text")
      .attr("x", width - 1)
      .attr("y", 0)
      .attr("text-anchor", "end")
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(note);
  }

  return () => {
    tip.destroy();
    // Every listener this module added — attachMark's and the group's arrow
    // keys alike — sits on a node inside this subtree, so dropping the subtree
    // drops them. Safe to call twice: the second call removes nothing.
    svg.selectAll("*").remove();
  };
}

export const treemapNested: ChartModule = {
  key: "treemap.nested",
  // `hierarchy×measure` is not in the wire `Shape` union yet — `api/types.ts`
  // is generated from real payloads, and the registry wiring lands with the
  // orchestrator, not from inside a chart module. The cast is that seam, and it
  // becomes a plain string literal the moment the union is widened.
  serves: "hierarchy×measure" as unknown as ChartModule["serves"],
  render,
};
