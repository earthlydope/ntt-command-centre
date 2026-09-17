/**
 * bubble.scatter — serves `x×y×size`.
 *
 * Two measures as position, a third as area, and a quadrant read laid over the
 * top. The chart exists for two questions and is built around them:
 *   • AE triage — x = days silent, y = GP, size = ACV, colour = risk band:
 *     "which of the 235 open opportunities is big, stale and unloved?"
 *   • Manager calibration — x = median value drift, y = win rate, size = book
 *     GP: "which rep wins on deals they never move?"
 *
 * AREA, not radius, is proportional to `size` — which is the whole reason the
 * radius comes from `scaleSqrt` and not `scaleLinear`. A circle's area grows
 * with r², so mapping value straight to radius squares the encoding: a $400k
 * line beside a $100k line would paint sixteen times the ink for four times the
 * money. With sqrt, four times the money is four times the ink.
 *
 * The quadrant guides are the point of this chart rather than decoration — the
 * only question an AE asks of a scatter is "which box is this deal in". They
 * are MEDIANS, not means: one $6M line drags a mean across the plot and drops
 * every other deal into the same quadrant.
 *
 * Overplotting is handled rather than ignored. Above DENSE_LIMIT marks the
 * per-mark tooltip becomes a lottery — the cursor lands on whichever circle
 * happens to be painted last — so one transparent surface takes the pointer and
 * routes it to the nearest centre. It routes by re-dispatching the same mouse
 * event at that mark, so hover, emphasis and the filter click still come out of
 * attachMark() and this module never grows a second hover implementation.
 */
import { max, median, min } from "d3-array";
import { scaleLinear, scaleSqrt } from "d3-scale";
import { select } from "d3-selection";
import type { ChartModule, RenderOpts, Teardown } from "./types";
import type { ChartSpec, Tone } from "../api/types";
import type { Palette } from "../theme/palette";
import { alpha, toneRgb } from "../theme/palette";
import {
  DIM_OPACITY,
  FONT,
  attachMark,
  charsIn,
  createTooltip,
  formatTick,
  formatValue,
  measureTickWidth,
  selectionState,
  textWidth,
  tipHtml,
  truncate,
} from "./util";

/** Above this many marks the per-mark tooltip is replaced by nearest-point
 *  hover and the fills are softened, because the picture is now a cloud. */
const DENSE_LIMIT = 120;
/** the palette-as-parameter rule: past this many marks the group takes ONE tab stop. */
const ROVE_LIMIT = 20;
/** How far the nearest-point hover reaches, in viewBox units. Past this the
 *  cursor is in genuine whitespace and the tooltip stays down. */
const NEAREST_REACH = 44;
/** A visibility floor, not part of the encoding: a bubble whose true area
 *  rounds to nothing still has to be findable and hoverable. The tooltip always
 *  carries the real number. */
const R_FLOOR = 2.2;

/** docs/CHART_CONTRACT.md §"Payload types". `days` exists here and not in lib/format's union,
 *  so it is the one thing handled locally; everything else goes through the
 *  shared formatter. */
type Format = "currency" | "percent" | "number" | "days";

/** docs/CHART_CONTRACT.md §"Payload types" — bubble.scatter, field for field. */
interface Bubble {
  id: string;
  label: string;
  x: number;
  y: number;
  size: number;
  category?: string;
  tone?: Tone;
  href?: string;
}

interface Payload {
  points: Bubble[];
  xLabel: string;
  yLabel: string;
  sizeLabel: string;
  xFormat: Format;
  yFormat: Format;
}

/** The payload plus the one thing the payload cannot say: how many rows the
 *  server sent, so the footer can own up to the ones this module could not
 *  place. */
interface Parsed {
  pay: Payload;
  supplied: number;
}

/** One drawn bubble, kept so the nearest-point router and the arrow keys can
 *  find it again without re-reading the DOM. */
interface Mark {
  node: SVGGraphicsElement;
  cx: number;
  cy: number;
  r: number;
}

const asRecord = (v: unknown): Record<string, unknown> | null =>
  v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

const asFormat = (v: unknown, fallback: Format): Format =>
  v === "currency" || v === "percent" || v === "number" || v === "days" ? v : fallback;

const asTone = (v: unknown): Tone | undefined =>
  v === "good" || v === "warn" || v === "danger" || v === "neutral" || v === "accent" ? v : undefined;

const prettify = (k: string): string =>
  k.replace(/[_-]+/g, " ").replace(/^./, (c) => c.toUpperCase());

/** `days` prints as a bare integer with a `d` suffix. docs/CHART_CONTRACT.md. */
const fmtValue = (v: number, f: Format): string =>
  f === "days" ? `${Math.round(v)}d` : formatValue(v, f);
const fmtTick = (v: number, f: Format): string =>
  f === "days" ? `${Math.round(v)}d` : formatTick(v, f);
/** measureTickWidth() takes the three-way union; days is a number plus one
 *  glyph, so it is measured as a number and given that glyph back. */
const narrow = (f: Format): "currency" | "percent" | "number" => (f === "days" ? "number" : f);

/**
 * The payload, as the wire actually delivers it.
 *
 * `ChartSpec.data` is typed as a flat row array (api/types.ts) while CONTRACT
 * writes this payload as one object carrying `points` and its labels. Both
 * forms are read here: an envelope row `[{ points: [...], xLabel: ... }]` is
 * unwrapped, and a bare `Bubble[]` is taken as the points with the field names
 * resolved off `spec.encoding`. No field is invented and none is renamed.
 */
function readPayload(spec: ChartSpec): Parsed {
  // docs/CHART_CONTRACT.md §"bubble.scatter" sends the envelope AS `data`:
  // `{points, xLabel, yLabel, sizeLabel, xFormat, yFormat}`. Two other spellings
  // are tolerated — a bare row array, and a single-row array wrapping the
  // envelope — so the module is correct however the seam is closed.
  const rows = Array.isArray(spec.data) ? spec.data : [];
  const direct =
    spec.data && !Array.isArray(spec.data) ? asRecord(spec.data) : null;
  const first = rows.length === 1 ? asRecord(rows[0]) : null;
  const envelope =
    direct && Array.isArray(direct.points)
      ? direct
      : first && Array.isArray(first.points)
        ? first
        : null;
  const raw: unknown[] = envelope ? (envelope.points as unknown[]) : rows;

  const xKey = spec.encoding?.x || "x";
  const yKey = spec.encoding?.y || "y";
  const sizeKey = spec.encoding?.series || "size";
  const labelKey = spec.encoding?.label || "label";

  const points: Bubble[] = [];
  raw.forEach((r, i) => {
    const d = asRecord(r);
    if (!d) return;
    const x = Number(d[xKey]);
    const y = Number(d[yKey]);
    // A point without a usable position is not a point. It is dropped here and
    // counted in the footer below, never silently swallowed.
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const size = Number(d[sizeKey]);
    const id = d.id === undefined || d.id === null ? `pt-${i}` : String(d.id);
    const label = d[labelKey] === undefined || d[labelKey] === null ? id : String(d[labelKey]);
    points.push({
      id,
      label,
      x,
      y,
      size: Number.isFinite(size) ? size : 0,
      category: d.category === undefined || d.category === null ? undefined : String(d.category),
      tone: asTone(d.tone),
      href: typeof d.href === "string" && d.href ? d.href : undefined,
    });
  });

  const lbl = (v: unknown, fallback: string) =>
    v === undefined || v === null || String(v) === "" ? fallback : String(v);

  return {
    supplied: raw.length,
    pay: {
      points,
      xLabel: lbl(envelope?.xLabel, prettify(xKey)),
      yLabel: lbl(envelope?.yLabel, prettify(yKey)),
      sizeLabel: lbl(envelope?.sizeLabel, prettify(sizeKey)),
      xFormat: asFormat(envelope?.xFormat, "number"),
      // y is the measure the card is titled with, so the spec's own format is
      // the honest default when the payload does not override it.
      yFormat: asFormat(envelope?.yFormat, spec.format ?? "currency"),
    },
  };
}

/** The only generic words available: the payload carries tones, not legend
 *  text, so a RAG legend cannot borrow a phrase it was never given. */
const TONE_LABEL: Record<Tone, string> = {
  danger: "At risk",
  warn: "Watch",
  good: "Healthy",
  accent: "Highlighted",
  neutral: "Unflagged",
};
const TONE_ORDER: Tone[] = ["danger", "warn", "good", "accent", "neutral"];

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  const { pay, supplied } = readPayload(spec);
  const pts = pay.points;
  const dropped = Math.max(0, supplied - pts.length);
  const width = Math.max(260, opts.width);
  const height = Math.max(240, opts.height || 300);

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr(
      "aria-label",
      `${spec.title}. Scatter of ${pts.length} points: ${pay.xLabel} against ${pay.yLabel}, area is ${pay.sizeLabel}. Median guides split the plot into four quadrants.`,
    );

  if (!pts.length) {
    // Degrade, never throw. the chart repository contract rule 7.
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", FONT.title)
      .text(
        supplied
          ? `No rows in this slice with a usable ${pay.xLabel} and ${pay.yLabel}`
          : "No rows in this slice",
      );
    return () => svg.selectAll("*").remove();
  }

  // ------------------------------------------------------------------ colour
  // Tone and category never share the plot. If ANY point carries a tone the
  // chart is in RAG mode and every fill is a tone, because a category hue
  // sitting next to a red risk band would read as a risk band. the chart repository contract
  // rule 3 keeps hue out of selection for the same reason.
  const ragMode = pts.some((d) => !!d.tone);
  const categories = ragMode ? [] : [...new Set(pts.map((d) => d.category).filter((c): c is string => !!c))].sort();
  // Accent and muted lead, so a two- or three-category plot never reaches for a
  // RAG hue it does not mean.
  const CYCLE = [p.accentRgb, p.mutedRgb, p.goodRgb, p.warnRgb, p.dangerRgb];
  const catRgb = new Map<string, string>();
  categories.forEach((c, i) => catRgb.set(c, CYCLE[i % CYCLE.length]));
  // Past one lap of the cycle the hue repeats, so the legend prints the names
  // and the tooltip always names the category outright.
  const rgbOf = (d: Bubble): string =>
    d.tone ? toneRgb(p, d.tone) : (d.category && catRgb.get(d.category)) || p.accentRgb;

  const dense = pts.length > DENSE_LIMIT;
  const fillA = dense ? 0.4 : 0.62;
  const strokeA = dense ? 0.85 : 1;

  // --------------------------------------------------------- domain, margins
  // Extents through d3, not a spread: 3,034 lines spread into Math.min() is an
  // argument list long enough to matter and buys nothing.
  const xLo = min(pts, (d) => d.x) ?? 0;
  const xHi = max(pts, (d) => d.x) ?? 0;
  const yLo = min(pts, (d) => d.y) ?? 0;
  const yHi = max(pts, (d) => d.y) ?? 0;
  // A scatter encodes position, not length, so a non-zero baseline is not the
  // lie it would be on a bar. The domain is padded off the data, then nice()d.
  const padX = (xHi - xLo || Math.abs(xHi) || 1) * 0.06;
  const padY = (yHi - yLo || Math.abs(yHi) || 1) * 0.08;
  const xDomain: [number, number] = [xLo - padX, xHi + padX];
  const yDomain: [number, number] = [yLo - padY, yHi + padY];

  // Domain first, margins second: the left gutter is measured from the ticks
  // the y scale will really print. nice() and ticks() do not read the range, so
  // the probe and the scale below cannot disagree.
  const legendRow = 22;
  const yTickProbe = scaleLinear().domain(yDomain).nice().ticks(4);
  const m = {
    top: 16,
    right: 14,
    bottom: 28 + legendRow,
    left: measureTickWidth(yTickProbe, narrow(pay.yFormat)) + (pay.yFormat === "days" ? 8 : 0),
  };
  const plotW = Math.max(60, width - m.left - m.right);
  const plotH = Math.max(60, height - m.top - m.bottom);

  // Radius from the plot's own area and the point count — never a constant. A
  // 900px card and a 320px card cannot carry the same bubble. CONTRACT rule 6.
  const rMax = Math.max(4, Math.min(22, Math.sqrt((plotW * plotH) / Math.max(24, pts.length * 12))));

  // The ranges are inset by rMax so the widest bubble sitting on the edge of
  // the domain is not half-clipped by its own axis.
  const x = scaleLinear().domain(xDomain).nice().range([m.left + rMax, width - m.right - rMax]);
  const y = scaleLinear().domain(yDomain).nice().range([height - m.bottom - rMax, m.top + rMax]);

  // Area ∝ value: r = k·√v. Negative sizes cannot be an area at all, so they
  // are clamped to the floor and counted out loud in the footer.
  const maxSize = max(pts, (d) => Math.max(0, d.size)) ?? 0;
  const rScale = scaleSqrt().domain([0, maxSize || 1]).range([0, rMax]);
  const negSized = pts.filter((d) => d.size < 0).length;
  // `size` has no format of its own in the payload, so it borrows the card's.
  const sizeFormat: Format = spec.format ?? "currency";

  const tip = createTooltip(p);

  // ------------------------------------------------------------- axes + grid
  const grid = svg.append("g").attr("aria-hidden", "true");
  const yTicks = y.ticks(4);
  grid
    .selectAll("line.gy")
    .data(yTicks)
    .join("line")
    .attr("class", "gy")
    .attr("x1", m.left)
    .attr("x2", width - m.right)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d))
    .attr("stroke", p.grid);
  grid
    .selectAll("text.ty")
    .data(yTicks)
    .join("text")
    .attr("class", "ty")
    .attr("x", m.left - 8)
    .attr("y", (d) => y(d) + 4)
    .attr("text-anchor", "end")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .attr("font-variant-numeric", "tabular-nums")
    .text((d) => fmtTick(d, pay.yFormat));

  const xTicks = x.ticks(Math.max(2, Math.floor(plotW / textWidth(14, FONT.tick))));
  grid
    .selectAll("line.gx")
    .data(xTicks)
    .join("line")
    .attr("class", "gx")
    .attr("x1", (d) => x(d))
    .attr("x2", (d) => x(d))
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("stroke", p.grid);
  grid
    .selectAll("text.tx")
    .data(xTicks)
    .join("text")
    .attr("class", "tx")
    .attr("x", (d) => x(d))
    .attr("y", height - m.bottom + 17)
    .attr("text-anchor", "middle")
    .attr("fill", p.muted)
    .attr("font-size", FONT.tick)
    .attr("font-variant-numeric", "tabular-nums")
    .text((d) => fmtTick(d, pay.xFormat));

  // axis titles, only where there is room for them to be read
  if (plotW > 220) {
    grid
      .append("text")
      .attr("x", width - m.right)
      .attr("y", height - m.bottom + 17)
      .attr("text-anchor", "end")
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .attr("letter-spacing", "0.05em")
      .text(truncate(pay.xLabel, 24));
  }
  if (plotH > 150) {
    grid
      .append("text")
      .attr("transform", `translate(${m.left - 8},${m.top + 2}) rotate(-90)`)
      .attr("text-anchor", "end")
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .attr("letter-spacing", "0.05em")
      .text(truncate(pay.yLabel, charsIn(plotH, FONT.note)));
  }

  // -------------------------------------------------------- quadrant guides
  const mx = median(pts, (d) => d.x) ?? 0;
  const my = median(pts, (d) => d.y) ?? 0;
  const quad = svg.append("g").attr("aria-hidden", "true").attr("pointer-events", "none");
  quad
    .append("line")
    .attr("x1", x(mx))
    .attr("x2", x(mx))
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("stroke", p.muted)
    .attr("stroke-width", 1)
    .attr("stroke-dasharray", "4 4")
    .attr("opacity", 0.45);
  quad
    .append("line")
    .attr("x1", m.left)
    .attr("x2", width - m.right)
    .attr("y1", y(my))
    .attr("y2", y(my))
    .attr("stroke", p.muted)
    .attr("stroke-width", 1)
    .attr("stroke-dasharray", "4 4")
    .attr("opacity", 0.45);

  if (plotW > 300) {
    quad
      .append("text")
      .attr("x", x(mx) + 5)
      .attr("y", m.top + 26)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .attr("opacity", 0.8)
      .text(`med ${fmtValue(mx, pay.xFormat)}`);
    quad
      .append("text")
      .attr("x", width - m.right - 2)
      .attr("y", y(my) - 5)
      .attr("text-anchor", "end")
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .attr("opacity", 0.8)
      .text(`med ${fmtValue(my, pay.yFormat)}`);
  }

  // The corner words are the read: "big and silent" is the top-right box, and
  // saying so beats making the reader reconstruct it from two axis titles.
  if (plotW > 300 && plotH > 140) {
    // The corner labels name the QUADRANT, not the axes — the axes are already
    // titled, so repeating both measure names four times ("low Days since last
    // change · high ACV GP") spent 150px a corner to say what the axis titles
    // and the median guides already say, and the four labels collided on a
    // card-width chart. Two words each: where you are on x, where you are on y.
    const lo = (t: string) => truncate(t.split(" ")[0].toLowerCase(), 10);
    const sx = lo(pay.xLabel);
    const sy = lo(pay.yLabel);
    const corner = (cx: number, cy: number, anchor: string, text: string) =>
      quad
        .append("text")
        .attr("x", cx)
        .attr("y", cy)
        .attr("text-anchor", anchor)
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .attr("opacity", 0.7)
        .text(text);
    corner(m.left + 6, m.top + 12, "start", `↓${sx} ↑${sy}`);
    corner(width - m.right - 6, m.top + 12, "end", `↑${sx} ↑${sy}`);
    corner(m.left + 6, height - m.bottom - 6, "start", `↓${sx} ↓${sy}`);
    corner(width - m.right - 6, height - m.bottom - 6, "end", `↑${sx} ↓${sy}`);
  }

  // A crosshair back to both axes, so a hovered bubble can be read off the
  // scales without counting gridlines.
  const guides = svg.append("g").attr("aria-hidden", "true").attr("pointer-events", "none");
  const guideX = guides
    .append("line")
    .attr("stroke", p.text)
    .attr("stroke-dasharray", "3 3")
    .attr("opacity", 0);
  const guideY = guides
    .append("line")
    .attr("stroke", p.text)
    .attr("stroke-dasharray", "3 3")
    .attr("opacity", 0);

  // ----------------------------------------------------------------- bubbles
  const bubbles = svg.append("g").attr("class", "bubbles");
  const marks: Mark[] = [];
  const basisWord = spec.countBasis === "lines" ? "lines" : "opportunities";
  // Largest first, so the small bubbles end up on top and stay clickable.
  const drawn = [...pts].sort((a, b) => b.size - a.size);

  drawn.forEach((d) => {
    // A click sets one filter and the server says which dimension; the value a
    // bubble carries for that dimension is its category, or its own label when
    // the payload has no category to give.
    const filterValue = d.category ?? d.label;
    const { anySelected, isSelected } = selectionState(opts, spec.clickDim, filterValue);
    const dimmed = anySelected && !isSelected;
    const cx = x(d.x);
    const cy = y(d.y);
    const r = Math.max(R_FLOOR, rScale(Math.max(0, d.size)));
    const rgb = rgbOf(d);

    const mg = bubbles.append("g").attr("class", "mark");
    // `href` is honoured only when a click cannot already mean "filter": one
    // click, one meaning. With a clickDim present the link is the row detail's
    // job, not this mark's.
    const linked = !!d.href && !spec.clickDim;
    const holder = linked
      ? mg
          .append("a")
          .attr("href", d.href as string)
          .attr("target", "_blank")
          .attr("rel", "noopener noreferrer")
      : mg;

    const circle = holder
      .append("circle")
      .attr("cx", cx)
      .attr("cy", cy)
      .attr("r", r)
      .attr("fill", alpha(rgb, dimmed ? fillA * 0.5 : fillA))
      .attr("stroke", alpha(rgb, dimmed ? 0.4 : strokeA))
      .attr("stroke-width", 1)
      .attr("opacity", dimmed ? DIM_OPACITY : 1);

    // Selection = outline + dim the rest. Never a hue change. CONTRACT rule 3.
    if (isSelected) {
      mg.append("circle")
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("r", r + 3)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5)
        .attr("pointer-events", "none");
    }

    const quadWord = `${d.y >= my ? "high" : "low"} ${truncate(pay.yLabel, 16)} · ${
      d.x >= mx ? "high" : "low"
    } ${truncate(pay.xLabel, 16)}`;
    const rows: [string, string][] = [
      [pay.xLabel, fmtValue(d.x, pay.xFormat)],
      [pay.yLabel, fmtValue(d.y, pay.yFormat)],
      [pay.sizeLabel, d.size < 0 ? `${fmtValue(d.size, sizeFormat)} (negative)` : fmtValue(d.size, sizeFormat)],
      ["Quadrant", quadWord],
    ];
    if (d.category) rows.push([ragMode ? "Segment" : "Category", d.category]);
    if (d.tone) rows.push(["Band", TONE_LABEL[d.tone]]);
    const foot = spec.clickDim
      ? `Click to filter ${spec.clickDim} = ${filterValue}`
      : linked
        ? "Click to open the record in a new tab"
        : undefined;

    const node = (linked ? holder : circle).node() as SVGGraphicsElement;
    attachMark(node, {
      tip,
      palette: p,
      html: tipHtml(p, d.label, rows, foot),
      aria: `${d.label}. ${pay.xLabel} ${fmtValue(d.x, pay.xFormat)}, ${pay.yLabel} ${fmtValue(
        d.y,
        pay.yFormat,
      )}, ${pay.sizeLabel} ${fmtValue(d.size, sizeFormat)}${d.category ? `, ${d.category}` : ""}${
        d.tone ? `, ${TONE_LABEL[d.tone]}` : ""
      }. ${quadWord}.`,
      dim: spec.clickDim,
      value: filterValue,
      opts,
      onEnter: () => {
        mg.raise();
        circle
          .attr("opacity", 1)
          .attr("fill", alpha(rgb, Math.min(0.9, fillA + 0.22)))
          .attr("stroke", p.text)
          .attr("stroke-width", 1.6);
        guideX
          .attr("x1", m.left)
          .attr("x2", cx)
          .attr("y1", cy)
          .attr("y2", cy)
          .attr("opacity", 0.5);
        guideY
          .attr("x1", cx)
          .attr("x2", cx)
          .attr("y1", cy)
          .attr("y2", height - m.bottom)
          .attr("opacity", 0.5);
      },
      onLeave: () => {
        circle
          .attr("opacity", dimmed ? DIM_OPACITY : 1)
          .attr("fill", alpha(rgb, dimmed ? fillA * 0.5 : fillA))
          .attr("stroke", alpha(rgb, dimmed ? 0.4 : strokeA))
          .attr("stroke-width", 1);
        guideX.attr("opacity", 0);
        guideY.attr("opacity", 0);
      },
    });

    marks.push({ node, cx, cy, r });
  });

  // ------------------------------------------- nearest-point hover (dense)
  // The circles keep their marks — focus, keyboard and the filter click all
  // still live on them — but they stop taking the pointer, and one surface
  // relays each mouse event to the nearest centre instead. Nothing here decides
  // what a hover looks like; attachMark still does.
  let hot: Mark | null = null;
  let overlay: SVGRectElement | null = null;
  const toLocal = (ev: MouseEvent): { lx: number; ly: number } | null => {
    const ctm = root.getScreenCTM();
    if (!ctm || !ctm.a || !ctm.d) return null;
    return { lx: (ev.clientX - ctm.e) / ctm.a, ly: (ev.clientY - ctm.f) / ctm.d };
  };
  const nearest = (lx: number, ly: number): Mark | null => {
    let best: Mark | null = null;
    let bestD = Infinity;
    for (const mk of marks) {
      const dd = (mk.cx - lx) * (mk.cx - lx) + (mk.cy - ly) * (mk.cy - ly);
      if (dd < bestD) {
        bestD = dd;
        best = mk;
      }
    }
    if (!best) return null;
    const reach = Math.max(best.r + 10, NEAREST_REACH);
    return bestD <= reach * reach ? best : null;
  };
  const relay = (node: SVGGraphicsElement, type: string, ev: MouseEvent) =>
    node.dispatchEvent(
      new MouseEvent(type, { clientX: ev.clientX, clientY: ev.clientY, bubbles: false, cancelable: true }),
    );
  const onMove = (ev: MouseEvent) => {
    const loc = toLocal(ev);
    const next = loc ? nearest(loc.lx, loc.ly) : null;
    if (next !== hot) {
      if (hot) relay(hot.node, "mouseleave", ev);
      hot = next;
      if (hot) relay(hot.node, "mouseenter", ev);
    } else if (hot) {
      relay(hot.node, "mousemove", ev);
    }
  };
  const onOut = (ev: MouseEvent) => {
    if (hot) relay(hot.node, "mouseleave", ev);
    hot = null;
  };
  // A stray click in a cloud must not navigate, so only a filter is relayed.
  const onClick = (ev: MouseEvent) => {
    if (hot && spec.clickDim) relay(hot.node, "click", ev);
  };

  if (dense) {
    bubbles.attr("pointer-events", "none");
    overlay = svg
      .append("rect")
      .attr("x", m.left)
      .attr("y", m.top)
      .attr("width", Math.max(1, width - m.left - m.right))
      .attr("height", Math.max(1, height - m.top - m.bottom))
      .attr("fill", "transparent")
      .attr("aria-hidden", "true")
      .style("cursor", spec.clickDim && opts.onFilter ? "pointer" : "default")
      .node() as SVGRectElement;
    overlay.addEventListener("mousemove", onMove as EventListener);
    overlay.addEventListener("mouseleave", onOut as EventListener);
    overlay.addEventListener("click", onClick as EventListener);
  }

  // ------------------------------------------------- one tab stop, arrow keys
  // the palette-as-parameter rule. util.ts has no markGroup() yet, so the roving index
  // lives here and nowhere else. Left/Right walk the points in x order and
  // Up/Down in y order, which is the only traversal that matches what the
  // reader sees on a scatter.
  const groupNode = bubbles.node() as SVGGElement;
  let onKey: ((ev: KeyboardEvent) => void) | null = null;
  if (marks.length > ROVE_LIMIT) {
    groupNode.setAttribute("role", "group");
    groupNode.setAttribute(
      "aria-label",
      `${marks.length} points. Arrow keys move between them, Enter activates.`,
    );
    marks.forEach((mk, i) => mk.node.setAttribute("tabindex", i === 0 ? "0" : "-1"));

    const orderX = marks.map((_, i) => i).sort((a, b) => marks[a].cx - marks[b].cx || a - b);
    const orderY = marks.map((_, i) => i).sort((a, b) => marks[a].cy - marks[b].cy || a - b);
    const posX = new Array<number>(marks.length);
    const posY = new Array<number>(marks.length);
    orderX.forEach((idx, k) => (posX[idx] = k));
    orderY.forEach((idx, k) => (posY[idx] = k));
    let rover = 0;

    const focusAt = (i: number) => {
      const next = Math.min(marks.length - 1, Math.max(0, i));
      marks[rover].node.setAttribute("tabindex", "-1");
      rover = next;
      marks[rover].node.setAttribute("tabindex", "0");
      // focus() fires attachMark's own focus handler, so the tooltip and the
      // emphasis come from the same place they do on hover.
      (marks[rover].node as unknown as SVGElement).focus();
    };

    onKey = (ev: KeyboardEvent) => {
      const k = ev.key;
      if (k === "ArrowRight") focusAt(orderX[Math.min(orderX.length - 1, posX[rover] + 1)]);
      else if (k === "ArrowLeft") focusAt(orderX[Math.max(0, posX[rover] - 1)]);
      else if (k === "ArrowUp") focusAt(orderY[Math.max(0, posY[rover] - 1)]);
      else if (k === "ArrowDown") focusAt(orderY[Math.min(orderY.length - 1, posY[rover] + 1)]);
      else if (k === "Home") focusAt(orderX[0]);
      else if (k === "End") focusAt(orderX[orderX.length - 1]);
      else return;
      ev.preventDefault();
    };
    groupNode.addEventListener("keydown", onKey as EventListener);
  }

  // ------------------------------------------------------------------ legend
  const legend = svg
    .append("g")
    .attr("aria-hidden", "true")
    .attr("transform", `translate(${m.left},${height - 6})`);
  const avail = Math.max(40, width - m.left - m.right);
  // The footer carries the things that are nowhere else on the chart: what the
  // area means, and every point the encoding could not tell the truth about.
  const footText = `area ∝ ${truncate(pay.sizeLabel, 20)}${
    negSized ? ` · ${negSized} negative drawn at the floor` : ""
  }${dropped ? ` · ${dropped} without a usable position` : ""}`;
  const footW = textWidth(footText.length, FONT.note) + 6;
  const roomForBoth = avail > footW + 90;

  const entries: { rgb: string; label: string }[] = ragMode
    ? TONE_ORDER.filter((t) => pts.some((d) => d.tone === t)).map((t) => ({
        rgb: toneRgb(p, t),
        label: TONE_LABEL[t],
      }))
    : categories.map((c) => ({ rgb: catRgb.get(c) ?? p.accentRgb, label: c }));

  if (!roomForBoth) {
    // Too narrow to carry both. The area rule wins: a category name is in every
    // tooltip already, and "area ∝ ACV" is written nowhere else.
    legend
      .append("text")
      .attr("x", 0)
      .attr("y", 0)
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(truncate(footText, charsIn(avail, FONT.note)));
  } else {
    legend
      .append("text")
      .attr("x", avail)
      .attr("y", 0)
      .attr("text-anchor", "end")
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .text(footText);

    const entryW = (label: string) => 13 + textWidth(truncate(label, 18).length, FONT.note) + 12;
    const budget = avail - footW - 12;
    // Fit twice: once for the happy case, and again against a budget that keeps
    // room for the "+n more" tail, so the tail never lands under the footer.
    const fits = (room: number) => {
      let used = 0;
      let n = 0;
      for (const e of entries) {
        const w = entryW(e.label);
        if (used + w > room) break;
        used += w;
        n += 1;
      }
      return n;
    };
    let shown = fits(budget);
    if (shown < entries.length) shown = fits(budget - 46);

    let cursor = 0;
    entries.slice(0, shown).forEach((e) => {
      const text = truncate(e.label, 18);
      const g = legend.append("g").attr("transform", `translate(${cursor},0)`);
      g.append("circle")
        .attr("cx", 4)
        .attr("cy", -3.5)
        .attr("r", 4)
        .attr("fill", alpha(e.rgb, 0.62))
        .attr("stroke", alpha(e.rgb, 1))
        .attr("stroke-width", 1);
      g.append("text").attr("x", 13).attr("y", 0).attr("fill", p.muted).attr("font-size", FONT.note).text(text);
      cursor += entryW(e.label);
    });
    if (shown < entries.length) {
      legend
        .append("text")
        .attr("x", cursor)
        .attr("y", 0)
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .text(`+${entries.length - shown} more`);
    } else if (dense && cursor + 100 < budget) {
      // Say out loud that the hover is not per-mark any more, so a user who
      // finds the tooltip snapping to a neighbour knows it is deliberate.
      legend
        .append("text")
        .attr("x", cursor)
        .attr("y", 0)
        .attr("fill", p.muted)
        .attr("font-size", FONT.note)
        .text(`${pts.length} ${basisWord} · nearest-point hover`);
    }
  }

  // Teardown is safe to call twice: StrictMode calls it twice. the idempotent-teardown rule.
  return () => {
    if (overlay) {
      overlay.removeEventListener("mousemove", onMove as EventListener);
      overlay.removeEventListener("mouseleave", onOut as EventListener);
      overlay.removeEventListener("click", onClick as EventListener);
      overlay = null;
    }
    if (onKey) {
      groupNode.removeEventListener("keydown", onKey as EventListener);
      onKey = null;
    }
    hot = null;
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

export const bubbleScatter: ChartModule = {
  key: "bubble.scatter",
  // `Shape` in api/types.ts is the five shapes the semantic layer ships today;
  // `x×y×size` (docs/CHART_CONTRACT.md §"The fourteen repository keys") is not in that union
  // yet and api/types.ts is a shared file this module may not widen. The cast is
  // the seam and nothing else depends on it — when the union grows, delete it.
  serves: "x×y×size" as unknown as ChartModule["serves"],
  render,
};
