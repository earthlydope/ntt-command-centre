/**
 * funnel.stage — serves `cohort×stage`.
 * Where a pod leaks deals: the stage ladder, one band per rung.
 *
 * A raw stage count is not a funnel. On this file it reaches 131% at Proposal,
 * because a deal is not obliged to be born at the top of the ladder — it is
 * logged wherever the rep first touched it. The semantic layer therefore
 * anchors every rung to ONE entry cohort and sends `cohortPct` already monotone
 * non-increasing. This module does not take that on trust: it re-checks the
 * sequence on every render, and where the sequence widens it draws the
 * violation instead of the funnel. A funnel that widens is the most convincing
 * wrong chart in the deck, so it is the one thing this module refuses to draw.
 * Contract rules 7 and 8.
 *
 * Geometry. A band's width encodes its own `cohortPct` and nothing else, so the
 * band is a rectangle — constant across its own height. The taper lives in the
 * GAP between two bands, which is exactly where the loss happens, and the two
 * outer wedges of that taper are the leak: their area is proportional to the
 * cohort width lost, and they are filled in `danger`. A band that narrowed over
 * its own height would be drawing a loss at a rung the data does not put it at.
 *
 * Hue is flat `accent` on every band. Width already carries the quantity, and
 * rule 3 reserves hue for performance — a per-rung ramp would say "worse" and
 * mean "later". There is no axis here, so no `formatTick`/`measureTickWidth`,
 * and no `directionColor`: no stage on this chart is above or below a target.
 */
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
  charsIn,
  createTooltip,
  formatValue,
  selectionState,
  textWidth,
  tipHtml,
  truncate,
  truncateLabel,
  enc,
  rows as specRows,
} from "./util";

/** The payload, exactly as CHART_CONTRACT.md defines it for this key. Declared
 *  locally because `api/types.ts` is shared and not this module's to edit. */
interface FunnelStage {
  key: string;
  label: string;
  count: number;
  value: number;
  /** % of the entry cohort. Monotone non-increasing — verified, not assumed. */
  cohortPct: number;
  /** % of the immediately previous stage. */
  stepPct?: number;
  /** Cohort lost AT this stage, i.e. in the gap below its band. */
  lostHere?: number;
}

const BAND = 34;          // fixed: this chart encodes width, never height
const TOP_PAD = 8;
const FOOT_PAD = 30;      // the "100% =" line that names the cohort
const MIN_HALF = 2.5;     // a 0.4% rung stays a visible sliver, not a gap
const MONOTONE_EPS = 0.1; // pts — absorbs the server's own rounding, not a rise

const numOr = (v: unknown): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
};

const optNum = (v: unknown): number | undefined => {
  if (v === null || v === undefined || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
};

/** 100 reads as "100%", 62.37 as "62.4%". A funnel full of trailing .0s is noise. */
const pctText = (v: number): string => `${v.toFixed(v >= 99.95 || v === 0 ? 0 : 1)}%`;

/** Greedy wrap for the honesty messages, which are sentences, not labels. */
function wrapLines(s: string, maxChars: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const w of s.split(/\s+/)) {
    if (!line) line = w;
    else if (line.length + 1 + w.length <= maxChars) line += ` ${w}`;
    else {
      out.push(line);
      line = w;
    }
  }
  if (line) out.push(line);
  return out;
}

function render(
  root: SVGSVGElement,
  spec: ChartSpec,
  p: Palette,
  opts: RenderOpts,
): Teardown {
  const svg = select(root);
  svg.selectAll("*").remove();

  const width = Math.max(240, opts.width);

  // `countBasis` is "opportunity" in api/types.ts and "opportunities" in the
  // chart contract; matching the stem lands either spelling on one word.
  const isOpps = String(spec.countBasis ?? "").startsWith("opportunit");
  const basisWord = isOpps ? "opportunities" : "lines";
  const basisShort = isOpps ? "opps" : "lines";

  // The contract names this payload's fields exactly, so they are read by name.
  // `encoding` is honoured only as a fallback, for the day the server renames
  // the measure column. Nothing here is invented.
  const nameKey = enc(spec)?.x || "key";
  const vKey = enc(spec)?.y || "value";
  const rows: FunnelStage[] = (specRows(spec) ?? []).map((d, i) => {
    const key = String(d.key ?? d[nameKey] ?? i);
    return {
      key,
      label: String(d.label ?? d[nameKey] ?? key),
      count: numOr(d.count),
      value: numOr(d.value ?? d[vKey]),
      cohortPct: numOr(d.cohortPct),
      stepPct: optNum(d.stepPct),
      lostHere: optNum(d.lostHere),
    };
  });

  /** Rule 7: degrade, never throw — and name the reason on screen. */
  const drawMessage = (lines: string[]): Teardown => {
    const maxChars = Math.max(20, charsIn(width - 36, FONT.tick));
    const flat: { text: string; head: boolean }[] = [];
    lines.forEach((l, i) =>
      wrapLines(l, maxChars).forEach((t) => flat.push({ text: t, head: i === 0 && lines.length > 1 })),
    );
    const h = Math.max(96, 40 + flat.length * 20);
    svg
      .attr("viewBox", `0 0 ${width} ${h}`)
      .attr("width", "100%")
      .attr("height", h)
      .attr("role", "img")
      .attr("aria-label", lines.join(" "));
    const y0 = h / 2 - ((flat.length - 1) * 20) / 2;
    flat.forEach((l, i) => {
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", y0 + i * 20)
        .attr("text-anchor", "middle")
        .attr("fill", l.head ? p.text : p.muted)
        .attr("font-size", l.head ? FONT.title : FONT.tick)
        .attr("font-weight", l.head ? 700 : 500)
        .text(l.text);
    });
    return () => svg.selectAll("*").remove();
  };

  // ------------------------------------------------------------- the checks
  if (!rows.length) return drawMessage(["No open stages in this slice"]);

  const broken = rows.find(
    (r) => !Number.isFinite(r.cohortPct) || !Number.isFinite(r.count) || !Number.isFinite(r.value),
  );
  if (broken) {
    return drawMessage([
      "This funnel cannot be drawn",
      `${broken.label} arrived without a count, a value or a cohort percentage.`,
    ]);
  }

  // Rule 8. A funnel has no geometry for a negative: a negative band would have
  // to be drawn as a positive width, which is a lie with a legend on it.
  const neg = rows.find((r) => r.cohortPct < 0 || r.count < 0 || r.value < 0);
  if (neg) {
    const what =
      neg.cohortPct < 0 ? "cohort percentage" : neg.count < 0 ? `count of ${basisWord}` : "value";
    return drawMessage([
      "This funnel cannot be drawn",
      `${neg.label} carries a negative ${what}, and a funnel band has no way to represent one.`,
    ]);
  }

  // THE honesty check. The server cohort-anchors this series; if it ever stops,
  // the reader gets the sentence rather than a funnel that widens.
  const rise = rows.findIndex((r, i) => i > 0 && r.cohortPct > rows[i - 1].cohortPct + MONOTONE_EPS);
  if (rise > 0) {
    return drawMessage([
      "This is not a cohort-anchored funnel",
      `${rows[rise].label} is ${pctText(rows[rise].cohortPct)} of the entry cohort, wider than ${rows[rise - 1].label} at ${pctText(rows[rise - 1].cohortPct)}.`,
      `A stage that widens means deals were counted where they were logged, not where the cohort started. The funnel is withheld rather than drawn.`,
    ]);
  }

  const topPct = rows[0].cohortPct;
  if (!(topPct > 0)) {
    return drawMessage([
      "The entry cohort is empty",
      `${rows[0].label} holds no ${basisWord} in this slice, so there is nothing for the later stages to be a share of.`,
    ]);
  }

  // ------------------------------------------------------------- the layout
  const n = rows.length;
  const gap = n > 7 ? 20 : 26;
  const height = TOP_PAD + n * BAND + (n - 1) * gap + FOOT_PAD;

  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "group")
    .attr("aria-label", `${spec.title}. ${n} stages, every one a share of the ${rows[0].label} cohort.`);

  const labelW = Math.min(204, Math.max(96, width * 0.25));
  const rightW = Math.min(136, Math.max(84, width * 0.19));
  const plotL = labelW + 6;
  const plotR = width - rightW - 6;
  const funnelHalf = Math.max(24, (plotR - plotL) / 2);
  const cx = (plotL + plotR) / 2;
  const half = scaleLinear().domain([0, topPct]).range([0, funnelHalf]);
  const hOf = (pct: number) => Math.max(MIN_HALF, half(Math.max(0, pct)));
  const bandTop = (i: number) => TOP_PAD + i * (BAND + gap);

  // The contract carries `measureLabel` ("ACV GP" | "ACV Revenue") on every
  // spec; api/types.ts does not declare it yet, so it is read defensively and
  // falls back to the neutral word rather than being invented.
  const declared = (spec as unknown as { measureLabel?: unknown }).measureLabel;
  const measureLabel =
    typeof declared === "string" && declared ? declared : spec.format === "percent" ? "Value" : "Amount";

  /** The drop into the next rung. The server's own `lostHere` wins; the count
   *  delta is the fallback, never an override. */
  const lostAt = (i: number): number => {
    const explicit = rows[i].lostHere;
    if (explicit !== undefined && explicit >= 0) return explicit;
    const d = rows[i].count - rows[i + 1].count;
    return d > 0 ? d : 0;
  };

  /** Step retention. Falls back to the cohort ratio, not the count ratio, so
   *  the number agrees with the widths actually drawn. */
  const stepOf = (i: number): string => {
    if (i === 0) return "entry stage";
    const s = rows[i].stepPct;
    if (s !== undefined) return pctText(s);
    const prev = rows[i - 1].cohortPct;
    return prev > 0 ? pctText((rows[i].cohortPct / prev) * 100) : "—";
  };

  // Selection is resolved on the key AND the label: a mark click sends the
  // server's key, while a chip filter carries the display name. Either match
  // lights the same rung. Outline plus dimming only — never a hue change.
  const state = rows.map((r) => {
    const byKey = selectionState(opts, spec.clickDim, r.key);
    const byLabel = selectionState(opts, spec.clickDim, r.label);
    return { anySelected: byKey.anySelected, isSelected: byKey.isSelected || byLabel.isSelected };
  });
  const anySelected = state.some((s) => s.anySelected);

  const tip = createTooltip(p);

  // ------------------------------------------------- the necks, drawn first
  const necks = svg.append("g").attr("aria-hidden", "true");
  for (let i = 0; i < n - 1; i++) {
    const ha = hOf(rows[i].cohortPct);
    const hb = hOf(rows[i + 1].cohortPct);
    const y0 = bandTop(i) + BAND;
    const y1 = bandTop(i + 1);
    const lit = state[i].isSelected || state[i + 1].isSelected;
    const g = necks.append("g").attr("opacity", anySelected && !lit ? DIM_OPACITY : 1);

    // what survives into the next rung
    g.append("polygon")
      .attr("points", `${cx - ha},${y0} ${cx + ha},${y0} ${cx + hb},${y1} ${cx - hb},${y1}`)
      .attr("fill", alpha(p.accentRgb, 0.22));

    // the leak: two wedges whose area is the cohort width lost at this rung
    if (ha - hb > 0.4) {
      g.append("polygon")
        .attr("points", `${cx - ha},${y0} ${cx - ha},${y1} ${cx - hb},${y1}`)
        .attr("fill", alpha(p.dangerRgb, 0.42));
      g.append("polygon")
        .attr("points", `${cx + ha},${y0} ${cx + ha},${y1} ${cx + hb},${y1}`)
        .attr("fill", alpha(p.dangerRgb, 0.42));
    }

    const lost = lostAt(i);
    if (lost > 0) {
      g.append("text")
        .attr("x", width - 6)
        .attr("y", (y0 + y1) / 2 + 3.5)
        .attr("text-anchor", "end")
        .attr("fill", p.danger)
        .attr("font-size", FONT.note)
        .attr("font-weight", 700)
        .attr("font-variant-numeric", "tabular-nums")
        .text(truncateLabel(`−${Math.round(lost).toLocaleString("en-US")} ${basisShort}`, rightW, FONT.note));
    }
  }

  // -------------------------------------------------------------- the bands
  const bands = svg
    .append("g")
    .attr("role", "group")
    .attr("aria-label", `${n} stages. Use the arrow keys to move between stages.`);
  const hits: SVGGraphicsElement[] = [];

  rows.forEach((r, i) => {
    const h = hOf(r.cohortPct);
    const top = bandTop(i);
    const mid = top + BAND / 2;
    // Nothing reached this rung. Drawn as an absence — outlined, not filled —
    // for the same reason heat.grid leaves an unpopulated intersection dashed.
    const empty = r.count <= 0;
    const dimmed = anySelected && !state[i].isSelected;
    const g = bands.append("g").attr("class", "mark");

    const band = g
      .append("rect")
      .attr("x", cx - h)
      .attr("y", top)
      .attr("width", h * 2)
      .attr("height", BAND)
      .attr("rx", 3)
      .attr("fill", empty ? "none" : p.accent)
      .attr("stroke", empty ? p.line : "none")
      .attr("opacity", dimmed ? DIM_OPACITY : 1);
    if (empty) band.attr("stroke-dasharray", "2 3");

    if (state[i].isSelected) {
      g.append("rect")
        .attr("x", cx - h - 2.5)
        .attr("y", top - 2.5)
        .attr("width", h * 2 + 5)
        .attr("height", BAND + 5)
        .attr("rx", 5)
        .attr("fill", "none")
        .attr("stroke", p.text)
        .attr("stroke-width", 1.5);
    }

    g.append("text")
      .attr("x", labelW - 10)
      .attr("y", mid - 3)
      .attr("text-anchor", "end")
      .attr("fill", dimmed ? p.muted : p.text)
      .attr("font-size", FONT.label)
      .attr("font-weight", state[i].isSelected ? 700 : 500)
      .text(truncateLabel(r.label, labelW - 14));

    g.append("text")
      .attr("x", labelW - 10)
      .attr("y", mid + 11)
      .attr("text-anchor", "end")
      .attr("fill", p.muted)
      .attr("font-size", FONT.note)
      .attr("font-variant-numeric", "tabular-nums")
      .text(`${r.count.toLocaleString("en-US")} ${basisShort}`);

    g.append("text")
      .attr("x", width - 6)
      .attr("y", mid + 4.5)
      .attr("text-anchor", "end")
      .attr("fill", dimmed ? p.muted : p.text)
      .attr("font-size", FONT.title)
      .attr("font-weight", 800)
      .attr("font-variant-numeric", "tabular-nums")
      .text(pctText(r.cohortPct));

    // The money goes inside the band only where the band can hold it; a late
    // rung is 8% wide and the figure is always in the tooltip regardless.
    const vText = formatValue(r.value, spec.format);
    if (!empty && h * 2 > textWidth(vText.length, FONT.label) + 18) {
      g.append("text")
        .attr("x", cx)
        .attr("y", mid + 4.5)
        .attr("text-anchor", "middle")
        .attr("fill", p.onAccent)
        .attr("font-size", FONT.label)
        .attr("font-weight", 700)
        .attr("font-variant-numeric", "tabular-nums")
        .attr("pointer-events", "none")
        .attr("opacity", dimmed ? 0.55 : 1)
        .text(vText);
    }

    // A full-row hit area, so the label column and the empty air beside a
    // narrow rung are hoverable too.
    const hit = g
      .append("rect")
      .attr("x", 0)
      .attr("y", top - 3)
      .attr("width", width)
      .attr("height", BAND + 6)
      .attr("fill", "transparent")
      .node() as SVGGraphicsElement;

    const lost = i < n - 1 ? lostAt(i) : 0;
    const tipRows: [string, string][] = [
      [isOpps ? "Opportunities" : "Lines", r.count.toLocaleString("en-US")],
      [measureLabel, vText],
      ["Of entry cohort", pctText(r.cohortPct)],
      ["Of previous stage", stepOf(i)],
    ];
    if (lost > 0) {
      tipRows.push(["Lost at this stage", `${Math.round(lost).toLocaleString("en-US")} ${basisWord}`]);
    }

    const paint = (hot: boolean) => {
      band.attr("opacity", hot ? 1 : dimmed ? DIM_OPACITY : 1);
      if (empty) band.attr("stroke", hot ? p.text : p.line);
      else band.attr("fill", hot ? alpha(p.accentRgb, 0.75) : p.accent);
    };

    attachMark(hit, {
      tip,
      palette: p,
      html: tipHtml(
        p,
        r.label,
        tipRows,
        spec.clickDim
          ? `Click to filter ${spec.clickDim} = ${r.label}`
          : `100% = the ${rows[0].label} cohort`,
      ),
      aria:
        `${r.label}, ${r.count.toLocaleString("en-US")} ${basisWord}, ${pctText(r.cohortPct)} of the entry cohort, ${vText}` +
        (lost > 0 ? `, ${Math.round(lost).toLocaleString("en-US")} ${basisWord} lost here` : ""),
      dim: spec.clickDim,
      value: r.key,
      opts,
      onEnter: () => paint(true),
      onLeave: () => paint(false),
    });

    hits.push(hit);
  });

  // ------------------------------------------------------- keyboard, rule 2
  // A ladder is six or seven marks, so it would survive one tab stop each —
  // but the group is still given ONE, because a funnel is read top to bottom
  // and the arrow keys are how a keyboard user reads it. util.ts has no
  // markGroup() yet, so the roving tabindex lives here and nowhere else.
  const setRoving = (i: number) =>
    hits.forEach((node, j) => node.setAttribute("tabindex", j === i ? "0" : "-1"));
  setRoving(Math.max(0, state.findIndex((s) => s.isSelected)));
  hits.forEach((node, i) => {
    node.addEventListener("focus", () => setRoving(i));
    node.addEventListener("keydown", (ev) => {
      const step =
        ev.key === "ArrowDown" || ev.key === "ArrowRight"
          ? 1
          : ev.key === "ArrowUp" || ev.key === "ArrowLeft"
            ? -1
            : 0;
      let next = -1;
      if (step !== 0) next = Math.max(0, Math.min(hits.length - 1, i + step));
      else if (ev.key === "Home") next = 0;
      else if (ev.key === "End") next = hits.length - 1;
      if (next < 0) return;
      ev.preventDefault();
      setRoving(next);
      hits[next].focus();
    });
  });

  // --------------------------------------------------------- the cohort line
  // The one sentence that makes the picture true: what 100% is, and that every
  // later band is the same cohort rather than whoever happened to be at that
  // rung on the day.
  const entry = `100% = ${rows[0].label}: ${rows[0].count.toLocaleString("en-US")} ${basisWord} entering the ladder`;
  const clause = `${entry} — every band below is that same cohort.`;
  const cap = charsIn(width - 12, FONT.note);
  svg
    .append("text")
    .attr("x", 6)
    .attr("y", height - 9)
    .attr("fill", p.muted)
    .attr("font-size", FONT.note)
    // The clause is dropped whole rather than ellipsed: a sentence cut at
    // "that same…" is worse than the shorter sentence that still says what
    // 100% is.
    .text(clause.length <= cap ? clause : truncate(`${entry}.`, cap));

  return () => {
    tip.destroy();
    svg.selectAll("*").remove();
  };
}

/** The ladder's own height: a band per stage and the cohort line. This chart
 *  encodes width, never height, so an offer taller than that is declined. */
function naturalHeight(spec: ChartSpec): number {
  const n = (specRows(spec) ?? []).length;
  if (!n) return 96;
  const gap = n > 7 ? 20 : 26;
  return TOP_PAD + n * BAND + (n - 1) * gap + FOOT_PAD;
}

export const funnelStage: ChartModule = {
  key: "funnel.stage",
  drawnHeight: (spec) => naturalHeight(spec),
  // `cohort×stage` is not a member of the Shape union in api/types.ts yet, and
  // that file is shared — the orchestrator owns it. The cast is confined to
  // this line and becomes a no-op the moment the union gains the member.
  serves: "cohort×stage" as unknown as ChartModule["serves"],
  render,
};
