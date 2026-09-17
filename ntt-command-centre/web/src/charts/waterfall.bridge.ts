/**
 * waterfall.bridge — serves `bridge`.
 * The entity GP bridge: FY26 budget GP → Q1 won → Q2 won → Q3 open-weighted →
 * the residual gap. The question it answers is not "how much GP is there" but
 * "where did the plan go", and that question is only answerable if every step
 * is drawn against the one before it.
 *
 * Geometry comes from `type`, never from the sign (docs/CHART_CONTRACT.md §"Payload types"):
 * a `delta` floats from the running total, while `start`, `subtotal` and `end`
 * are anchored to the zero baseline because they assert a LEVEL rather than a
 * movement. An anchor therefore also gets the heavier fill — it is a total, and
 * a total that looks like a step invites the reader to add it in twice.
 *
 * COLOUR HERE ENCODES SIGN, NOT PERFORMANCE. That is the one deliberate
 * exception to the direction-aware colour rule in this repository, and it is confined to
 * this module: rises use `good`, falls use `danger`, anchors use `accent`. A
 * bridge whose rises and falls share a hue is unreadable — the eye has to
 * re-derive the direction of every bar from its neighbours' edges — and the
 * sign of a bridge step is not a value judgement the palette is overriding,
 * because "won GP went up" and "the gap grew" are already good and bad in the
 * plain sense. Where the semantic layer HAS classified a step it still wins:
 * `tone` is obeyed when the server sends one, since thresholds are a business
 * rule and business rules live in `api/semantic/`, not in a D3 module.
 */
import { max, min } from "d3-array";
import { scaleBand, scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec } from "../api/types";
import type { Palette } from "../theme/palette";
import { toneColor } from "../theme/palette";
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
  truncateLabel,
  enc,
  rows as specRows,
} from "./util";

/** Above this many marks the group takes ONE tab stop. the palette-as-parameter rule. */
const DENSE_MARKS = 20;
const LEGEND_H = 20;

const TYPES = ["start", "delta", "subtotal", "end"] as const;
type StepType = (typeof TYPES)[number];

const TONES = ["accent", "good", "warn", "danger", "neutral"] as const;
type StepTone = (typeof TONES)[number];

/** The payload's Step, after validation. Field for field from the contract. */
interface Step {
  key: string;
  label: string;
  value: number;
  type: StepType;
  tone?: StepTone;
  note?: string;
}

interface Bar {
  s: Step;
  /** Bottom and top of the drawn rectangle, in data units. */
  lo: number;
  hi: number;
  /** The running total on each side of this step. */
  before: number;
  after: number;
}

/** An unknown `type` degrades to `delta` — the one geometry that asserts no
 *  level it was not given. */
function asType(v: unknown): StepType {
  return TYPES.includes(v as StepType) ? (v as StepType) : "delta";
}

function asTone(v: unknown): StepTone | undefined {
  return TONES.includes(v as StepTone) ? (v as StepTone) : undefined;
}

/**
 * the palette-as-parameter rule — a dense chart exposes one tab stop and moves within it
 * on the arrow keys, or a keyboard user pays 200 tabs to cross one card.
 * `util.ts` has no `markGroup()` yet and every shared file is owned elsewhere
 * this cycle, so the behaviour lives here and touches only this module's own
 * nodes. `attachMark()` still owns hover, focus, activation and the tooltip;
 * this rewrites the tab ORDER and adds the arrows, nothing else.
 */
function rovingTabStop(nodes: SVGGraphicsElement[]): void {
  if (nodes.length <= DENSE_MARKS) return;
  let at = 0;
  const put = (i: number) => {
    nodes.forEach((n, j) => n.setAttribute("tabindex", j === i ? "0" : "-1"));
    at = i;
  };
  put(0);
  nodes.forEach((n, i) => {
    n.addEventListener("focus", () => put(i));
    n.addEventListener("keydown", (ev) => {
      const e = ev as KeyboardEvent;
      const move =
        e.key === "ArrowRight" || e.key === "ArrowDown"
          ? 1
          : e.key === "ArrowLeft" || e.key === "ArrowUp"
            ? -1
            : 0;
      let next = -1;
      if (move !== 0) next = Math.min(nodes.length - 1, Math.max(0, at + move));
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = nodes.length - 1;
      if (next < 0) return;
      e.preventDefault();
      put(next);
      (nodes[next] as unknown as HTMLElement).focus();
    });
  });
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  // The payload is `Step[]`. `encoding` may rename the two axis fields — the
  // server has done so for other shapes — but `type`, `tone` and `note` are
  // fixed by the contract and are read by their contract names.
  const keyKey = enc(spec).x || "key";
  const vKey = enc(spec).y || "value";
  const labelKey = enc(spec).label || "label";
  const rows = (specRows(spec) ?? []) as Record<string, unknown>[];
  const steps: Step[] = rows.map((d) => ({
    key: String(d[keyKey] ?? ""),
    label: String(d[labelKey] ?? d[keyKey] ?? ""),
    value: Number(d[vKey]),
    type: asType(d.type),
    tone: asTone(d.tone),
    note: d.note === undefined || d.note === null ? undefined : String(d.note),
  }));

  const width = Math.max(240, opts.width);
  const height = Math.max(210, opts.height || 260);

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr("aria-label", `${spec.title}. Bridge of ${steps.length} steps.`);

  const centred = (msg: string): Teardown => {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text(msg);
    return () => svg.selectAll("*").remove();
  };

  if (!steps.length) return centred("No bridge steps in this slice");
  // A bridge is a running sum: one unusable value poisons every bar after it,
  // so the walk is refused rather than drawn wrong. the chart repository contract rule 7.
  const broken = steps.filter((s) => !Number.isFinite(s.value)).length;
  if (broken) {
    return centred(
      `${broken} of ${steps.length} bridge steps carry no usable value — walk not drawn`,
    );
  }

  // ------------------------------------------------------------- the walk
  let run = 0;
  const bars: Bar[] = steps.map((s) => {
    const before = run;
    if (s.type === "delta") {
      run = before + s.value;
      return { s, lo: Math.min(before, run), hi: Math.max(before, run), before, after: run };
    }
    // An anchor declares the level it sits at; the server's number wins over
    // our own arithmetic, and any disagreement is reported in the tooltip
    // rather than quietly redrawn.
    run = s.value;
    return { s, lo: Math.min(0, s.value), hi: Math.max(0, s.value), before, after: run };
  });

  // The domain spans the whole walk — every floor and every ceiling the running
  // total passes through — not just the endpoints. A bridge that dips below its
  // opening and its closing level and is scaled to those two alone draws the
  // dip off the bottom of the card.
  const lo = Math.min(0, min(bars, (b) => b.lo) ?? 0);
  const hi = Math.max(0, max(bars, (b) => b.hi) ?? 0);
  const domain: [number, number] = lo === hi ? [0, 1] : [lo, hi];

  const probe = scaleLinear().domain(domain).nice().ticks(4);
  const m = {
    top: 26,
    right: 12,
    bottom: 24,
    left: measureTickWidth(probe, spec.format),
  };
  const plotBottom = Math.max(m.top + 40, height - m.bottom - LEGEND_H);

  const x = scaleBand<string>()
    .domain(bars.map((b, i) => `${i}|${b.s.key}`))
    .range([m.left, width - m.right])
    .paddingInner(0.34)
    .paddingOuter(0.18);
  const y = scaleLinear().domain(domain).nice().range([plotBottom, m.top]);
  const at = (i: number) => x(`${i}|${bars[i].s.key}`) ?? m.left;

  const tip = createTooltip(p);

  // ---------------------------------------------------- gridlines and ticks
  const yTicks = y.ticks(4);
  const grid = svg.append("g");
  grid
    .selectAll("line")
    .data(yTicks)
    .join("line")
    .attr("x1", m.left)
    .attr("x2", width - m.right)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d))
    .attr("stroke", p.grid);
  grid
    .selectAll("text")
    .data(yTicks)
    .join("text")
    .attr("x", m.left - 8)
    .attr("y", (d) => y(d) + 4)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .text((d) => formatTick(d, spec.format));

  // The zero baseline is drawn heavier than a gridline: the anchors are read
  // against it, and a bridge that crosses zero must show where zero is.
  svg
    .append("line")
    .attr("x1", m.left)
    .attr("x2", width - m.right)
    .attr("y1", y(0))
    .attr("y2", y(0))
    .attr("stroke", p.line)
    .attr("stroke-width", 1.3);

  // ------------------------------------------------------------ connectors
  // Drawn before the bars so every mark paints over them. A connector leaves
  // the top of step i at the level the WALK reached and meets the left edge of
  // step i+1; where the next step is an anchor whose declared level differs,
  // the connector visibly misses it, which is the honest picture.
  const links = svg.append("g").attr("aria-hidden", "true");
  for (let i = 0; i < bars.length - 1; i += 1) {
    const level = y(bars[i].after);
    links
      .append("line")
      .attr("x1", at(i) + x.bandwidth())
      .attr("x2", at(i + 1))
      .attr("y1", level)
      .attr("y2", level)
      .attr("stroke", p.muted)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3 3")
      .attr("opacity", 0.55);
  }

  // ------------------------------------------------------------- the marks
  const isAnchor = (b: Bar) => b.s.type !== "delta";
  const fillOf = (b: Bar): string => {
    if (b.s.tone) return toneColor(p, b.s.tone);
    if (isAnchor(b)) return p.accent;
    return b.s.value < 0 ? p.danger : p.good;
  };
  /** A rise labels above its top, a fall below its bottom. For an anchor the
   *  same test reads as "does this level sit above zero". */
  const rises = (b: Bar) => b.s.value >= 0;
  const signed = (v: number) => `${v > 0 ? "+" : ""}${formatValue(v, spec.format)}`;
  const levelWord = (t: StepType) =>
    t === "start" ? "Opening level" : t === "end" ? "Closing level" : "Running level";

  const cols = svg.append("g");
  const overlay = svg.append("g");
  const hits: SVGGraphicsElement[] = [];

  bars.forEach((b, i) => {
    const g = cols.append("g");
    const x0 = at(i);
    const yTop = y(b.hi);
    const yBot = y(b.lo);
    // A zero-valued step still shows as a hairline at its own level: it is a
    // step that happened and moved nothing, not a step that is missing.
    const h = Math.max(1.5, yBot - yTop);
    const { anySelected, isSelected } = selectionState(opts, spec.clickDim, b.s.key);
    const dimmed = anySelected && !isSelected;
    g.attr("opacity", dimmed ? DIM_OPACITY : 1);

    const rect = g
      .append("rect")
      .attr("x", x0)
      .attr("y", yTop)
      .attr("width", x.bandwidth())
      .attr("height", h)
      .attr("rx", 2.5)
      .attr("fill", fillOf(b))
      // The heavier fill is what separates a total from a movement.
      .attr("fill-opacity", isAnchor(b) ? 1 : 0.85)
      .attr("stroke", fillOf(b))
      .attr("stroke-width", isAnchor(b) ? 1.25 : 1);

    // Selection = outline + dim the rest. Never a hue change — hue is already
    // spoken for here, twice over.
    if (isSelected) {
      g.append("rect")
        .attr("x", x0 - 2.5)
        .attr("y", yTop - 2.5)
        .attr("width", x.bandwidth() + 5)
        .attr("height", h + 5)
        .attr("rx", 4.5)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5);
    }

    // Value labels sit above a rise and below a fall, and are dropped whenever
    // the column is narrower than the text — a clipped or overlapping number is
    // worse than no number, and the value is always in the tooltip.
    const text = b.s.type === "delta" ? signed(b.s.value) : formatValue(b.s.value, spec.format);
    if (x.bandwidth() >= textWidth(text.length, FONT.label)) {
      g.append("text")
        .attr("x", x0 + x.bandwidth() / 2)
        .attr("y", rises(b) ? yTop - 6 : yBot + 13)
        .attr("text-anchor", "middle")
        .attr("fill", p.text)
        .attr("font-size", FONT.label)
        .attr("font-weight", 700)
        .attr("font-variant-numeric", "tabular-nums")
        .attr("pointer-events", "none")
        .text(text);
    }

    g.append("text")
      .attr("x", x0 + x.bandwidth() / 2)
      .attr("y", plotBottom + 16)
      .attr("text-anchor", "middle")
      .attr("fill", isAnchor(b) ? p.text2 : p.muted)
      .attr("font-size", FONT.tick)
      .attr("font-weight", isAnchor(b) ? 650 : 500)
      .attr("pointer-events", "none")
      .text(truncateLabel(b.s.label, x.step(), FONT.tick));

    // A full-column hit area, so the axis label and the gap above the bar are
    // hoverable too — a $12k step is three pixels tall and unhittable otherwise.
    const hit = overlay
      .append("rect")
      .attr("x", x0 - (x.step() - x.bandwidth()) / 2)
      .attr("y", m.top - 14)
      .attr("width", x.step())
      .attr("height", plotBottom - m.top + 32)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;
    hits.push(hit);

    const tipRows: [string, string][] = [];
    if (b.s.type === "delta") {
      tipRows.push(["Change", signed(b.s.value)]);
      tipRows.push(["From", formatValue(b.before, spec.format)]);
      tipRows.push(["To", formatValue(b.after, spec.format)]);
    } else {
      tipRows.push([levelWord(b.s.type), formatValue(b.s.value, spec.format)]);
      // Only when the declared level and the walked total actually disagree.
      // Rounding in the semantic layer is not a discrepancy worth shouting at.
      const tol = Math.max(0.5, Math.abs(b.s.value) * 0.005);
      if (b.s.type !== "start" && Math.abs(b.s.value - b.before) > tol) {
        tipRows.push(["Steps above arrive at", formatValue(b.before, spec.format)]);
      }
    }

    const foot = [
      b.s.note,
      spec.clickDim ? `Click to filter ${spec.clickDim} = ${b.s.key}` : undefined,
    ]
      .filter(Boolean)
      .join(" · ");

    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(p, b.s.label, tipRows, foot || undefined),
      aria:
        b.s.type === "delta"
          ? `${b.s.label}, change ${signed(b.s.value)}, running total ${formatValue(b.after, spec.format)}`
          : `${b.s.label}, ${levelWord(b.s.type).toLowerCase()} ${formatValue(b.s.value, spec.format)}`,
      dim: spec.clickDim,
      value: b.s.key,
      opts,
      onEnter: () => {
        g.attr("opacity", 1);
        rect.attr("stroke", p.text).attr("stroke-width", 1.6).attr("fill-opacity", 1);
      },
      onLeave: () => {
        g.attr("opacity", dimmed ? DIM_OPACITY : 1);
        rect
          .attr("stroke", fillOf(b))
          .attr("stroke-width", isAnchor(b) ? 1.25 : 1)
          .attr("fill-opacity", isAnchor(b) ? 1 : 0.85);
      },
    });
  });

  rovingTabStop(hits);

  // -------------------------------------------------------------- legend
  // Hue means something different on this card than on every other one, so the
  // card says what it means rather than leaving the reader to infer it.
  const legend = svg.append("g").attr("transform", `translate(${m.left},${height - 9})`);
  const keys: [string, string][] = [
    [p.good, "rise"],
    [p.danger, "fall"],
    [p.accent, "total"],
  ];
  let lx = 0;
  keys.forEach(([c, text]) => {
    legend
      .append("rect")
      .attr("x", lx)
      .attr("y", -7)
      .attr("width", 8)
      .attr("height", 8)
      .attr("rx", 2)
      .attr("fill", c);
    legend
      .append("text")
      .attr("x", lx + 12)
      .attr("y", 0)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(text);
    lx += 12 + textWidth(text.length, FONT.note) + 14;
  });

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const waterfallBridge: ChartModule = {
  key: "waterfall.bridge",
  // docs/CHART_CONTRACT.md §"The fourteen repository keys" indexes this module under the
  // `bridge` shape. `Shape` in `api/types.ts` is still the five-shape union of
  // the first cycle and that file belongs to the registry wiring, not to this
  // module, so the shape is asserted here and the assertion becomes a no-op the
  // moment `bridge` joins the union.
  serves: "bridge" as unknown as ChartModule["serves"],
  render,
};
