/**
 * Shared chart plumbing. Everything in here exists so that the six modules
 * obey the chart repository contract identically rather than each inventing its own hover.
 */
import type { Palette } from "../theme/palette";
import { alpha } from "../theme/palette";
import type { DimKey, RenderOpts } from "./types";
import { formatTick, formatValue, type Format } from "../lib/format";

export { formatTick, formatValue };
export type { Format };

/**
 * A row-shaped spec payload, narrowed once so no module has to.
 *
 * `ChartSpec.data` is a union — row array for the simple types, object for the
 * structured ones (heat grid, mekko, sankey, gantt) — because those payloads
 * genuinely differ. Every module that wants rows calls this rather than
 * casting at each use site, which is where the implicit-any creep started.
 */
export type Row = Record<string, unknown>;

export function rows(spec: { data: unknown }): Row[] {
  return Array.isArray(spec.data) ? (spec.data as Row[]) : [];
}

export function payload<T>(spec: { data: unknown }, fallback: T): T {
  return spec.data && !Array.isArray(spec.data) ? (spec.data as T) : fallback;
}

/**
 * The type sizes a chart draws with, in SVG user units, which are CSS pixels
 * at the width the chart is rendered.
 *
 * A chart cannot read a custom property (see the note at the top of
 * palette.ts), so the scale that density.css sets for the shell is restated
 * here for the marks: tick labels one step under the shell's small text, data
 * labels level with it, a chart's own heading one step above. A module that
 * sizes a gutter or truncates a label should do so from these numbers and
 * `CHAR_WIDTH`, never from a literal, so that the gutter grows with the type
 * instead of clipping it.
 */
export const FONT = {
  /** Axis ticks and gridline values. */
  tick: 12,
  /** Category labels, values beside a mark, anything read as data. */
  label: 13,
  /** A chart's own heading or the empty-state line. */
  title: 13.5,
  /** Legend keys and the smallest annotations; never below this. */
  note: 11.5,
} as const;

/** Average advance of one character of the UI face, as a fraction of the
 *  font size. Inter's digits sit at 0.6em, letters a touch under. */
export const CHAR_WIDTH = 0.6;

/** The horizontal room `n` characters take at `size`, in user units. Used
 *  to size a gutter or a legend entry before the text is drawn. */
export function textWidth(n: number, size: number): number {
  return Math.ceil(n * size * CHAR_WIDTH);
}

/** How many characters at `size` fit inside `px`. Never negative. */
export function charsIn(px: number, size: number): number {
  return Math.max(0, Math.floor(px / (size * CHAR_WIDTH)));
}

/**
 * Truncate a label to what fits in `px` at `size`, with an ellipsis. The
 * floor of three characters keeps a category recognisable ("Te…") rather than
 * reducing it to punctuation when the gutter is very narrow.
 */
export function truncateLabel(s: string, px: number, size: number = FONT.label): string {
  return truncate(s, Math.max(3, charsIn(px, size)));
}

/**
 * Break a label into at most `maxLines` lines that each fit `px` at `size`,
 * breaking at spaces, with the last line cut with an ellipsis. Where a row is
 * tall enough for two lines of type, "Concentration & Cross-Sell" reads whole
 * on two rather than as "Concentration & C…" on one. A label that fits on one
 * line comes back as one; a single word wider than the line is cut on its own.
 */
export function wrapLabel(
  s: string,
  px: number,
  size: number = FONT.label,
  maxLines: number = 2,
): string[] {
  const cap = Math.max(3, charsIn(px, size));
  if (s.length <= cap || maxLines <= 1) return [truncate(s, cap)];
  const words = s.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = "";
  for (let i = 0; i < words.length; i += 1) {
    const w = words[i];
    const joined = line ? `${line} ${w}` : w;
    if (joined.length <= cap) {
      line = joined;
      continue;
    }
    // The next word does not fit. On the last line allowed, whatever remains
    // goes on it and is cut; otherwise this line closes and the word opens
    // the next, cut on its own if it is wider than the line.
    if (lines.length === maxLines - 1) {
      const rest = words.slice(i).join(" ");
      lines.push(truncate(line ? `${line} ${rest}` : rest, cap));
      return lines;
    }
    if (line) lines.push(line);
    line = w.length > cap ? truncate(w, cap) : w;
  }
  if (line) lines.push(line);
  return lines;
}

/** Clamp a per-row pitch between its floor and its cap. Modules that size
 *  themselves from their rows use it to grow towards an offered height
 *  without drawing bars as thick as a card. */
export function pitch(offered: number, chrome: number, n: number, min: number, max: number): number {
  if (n <= 0) return min;
  return Math.max(min, Math.min(max, Math.floor((offered - chrome) / n)));
}

export function numOf(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
}

export function strOf(v: unknown): string {
  return v === null || v === undefined ? "" : String(v);
}

/**
 * The spec's encoding, with the wire defaults applied.
 *
 * The server always sends `{x:"key", y:"value"}` unless a builder overrides it,
 * but the field is optional on the type because the structured payloads do not
 * use it — so reading it directly is an undefined access on exactly the types
 * that would never notice.
 */
export function enc(spec: {
  encoding?: { x: string; y: string; series?: string; target?: string; label?: string };
}): { x: string; y: string; series: string; target: string; label: string } {
  const e = spec.encoding;
  return {
    x: e?.x ?? "key",
    y: e?.y ?? "value",
    series: e?.series ?? "series",
    target: e?.target ?? "target",
    label: e?.label ?? "label",
  };
}

export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** A floating tooltip on <body>, not inside the SVG: an SVG cannot overflow its
 *  own viewport, so an in-SVG tooltip is clipped on the last bar every time. */
export interface Tooltip {
  show(html: string, x: number, y: number): void;
  hide(): void;
  destroy(): void;
}

export function createTooltip(p: Palette): Tooltip {
  const el = document.createElement("div");
  el.className = "d3tip";
  el.setAttribute("role", "status");
  el.style.position = "fixed";
  // No z-index here: the stylesheet's .d3tip rule owns where the tooltip
  // stacks, because it has to sit above the chart-ask overlay and an inline
  // value would beat that rule and bury the tooltip under the scrim.
  el.style.pointerEvents = "none";
  el.style.opacity = "0";
  // The fade is the stylesheet's too (.d3tip), so it runs on the app's motion
  // tokens and stops with them under prefers-reduced-motion.
  el.style.maxWidth = "300px";
  el.style.padding = "10px 12px";
  el.style.borderRadius = "10px";
  // The tooltip is read at arm's length like the shell, so it takes the
  // shell's small size rather than the chart's tick size.
  el.style.font =
    `500 ${FONT.title}px/1.5 "Inter","Segoe UI","Helvetica Neue",Arial,sans-serif`;
  el.style.background = p.surface;
  el.style.color = p.text;
  el.style.border = `1px solid ${p.line}`;
  el.style.boxShadow = `0 10px 30px ${alpha(p.name === "dark" ? "0,0,0" : "14,23,38", p.name === "dark" ? 0.5 : 0.16)}`;
  document.body.appendChild(el);

  let raf = 0;
  return {
    show(html, x, y) {
      el.innerHTML = html;
      el.style.opacity = "1";
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        const left = Math.min(Math.max(8, x - r.width / 2), window.innerWidth - r.width - 8);
        const top = y - r.height - 12 < 8 ? y + 18 : y - r.height - 12;
        el.style.left = `${left}px`;
        el.style.top = `${top}px`;
      });
    },
    hide() {
      el.style.opacity = "0";
    },
    destroy() {
      cancelAnimationFrame(raf);
      el.remove();
    },
  };
}

export function tipHtml(p: Palette, title: string, rows: [string, string][], foot?: string): string {
  const head = `<div style="font-weight:700;margin-bottom:5px;color:${p.text}">${esc(title)}</div>`;
  const body = rows
    .map(
      ([k, v]) =>
        `<div style="display:flex;gap:14px;justify-content:space-between"><span style="color:${p.muted}">${esc(k)}</span><span style="font-weight:700;font-variant-numeric:tabular-nums">${esc(v)}</span></div>`,
    )
    .join("");
  const f = foot
    ? `<div style="margin-top:6px;padding-top:6px;border-top:1px solid ${p.line};color:${p.muted};font-size:${FONT.label}px">${esc(foot)}</div>`
    : "";
  return head + body + f;
}

export function esc(s: string): string {
  return String(s).replace(/[&<>"]/g, (c) =>
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&quot;",
  );
}

/**
 * Wires one mark. EVERY mark in every module goes through here, so hover,
 * keyboard focus and click-to-filter cannot be forgotten in one chart and not
 * another. the one-helper interaction rule.
 */
export interface MarkOpts {
  tip: Tooltip;
  palette: Palette;
  html: string;
  /** Screen-reader text. The chart is otherwise a wall of <rect>. */
  aria: string;
  /** `null` is how the wire says "these marks are not filters"; treated
   *  identically to absent so a spec can round-trip through JSON. */
  dim?: DimKey | null;
  value?: string;
  opts: RenderOpts;
  /** Called on hover/focus so the module can draw its own emphasis. */
  onEnter?: () => void;
  onLeave?: () => void;
}

export function attachMark(node: SVGGraphicsElement, m: MarkOpts): void {
  const clickable = !!(m.dim && m.value && m.opts.onFilter);
  node.setAttribute("tabindex", "0");
  node.setAttribute("role", clickable ? "button" : "img");
  node.setAttribute("aria-label", m.aria + (clickable ? " — activate to filter" : ""));
  if (clickable) node.style.cursor = "pointer";

  const at = (): { x: number; y: number } => {
    const r = node.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top };
  };
  const enter = (ev?: MouseEvent) => {
    m.onEnter?.();
    if (ev) m.tip.show(m.html, ev.clientX, ev.clientY);
    else {
      const { x, y } = at();
      m.tip.show(m.html, x, y);
    }
  };
  const move = (ev: MouseEvent) => m.tip.show(m.html, ev.clientX, ev.clientY);
  const leave = () => {
    m.onLeave?.();
    m.tip.hide();
  };
  const fire = () => {
    if (m.dim && m.value) m.opts.onFilter?.(m.dim, m.value);
  };
  const key = (ev: KeyboardEvent) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      fire();
    }
    if (ev.key === "Escape") (node as unknown as HTMLElement).blur();
  };

  node.addEventListener("mouseenter", enter as EventListener);
  node.addEventListener("mousemove", move as EventListener);
  node.addEventListener("mouseleave", leave);
  node.addEventListener("focus", () => enter());
  node.addEventListener("blur", leave);
  node.addEventListener("click", fire);
  node.addEventListener("keydown", key as EventListener);
}

/**
 * Selection is signalled by outline plus dimming of the unselected — never by
 * hue, because hue already means performance. the selection rule.
 */
export function selectionState(
  opts: RenderOpts,
  dim: DimKey | undefined | null,
  value: string,
): { anySelected: boolean; isSelected: boolean } {
  const cur = dim ? opts.selected?.[dim] ?? null : null;
  return { anySelected: !!cur, isSelected: cur === value };
}

export const DIM_OPACITY = 0.28;

/** Direction-aware colour: a rise in overdue pipeline is red, not green.
 *  the direction-aware colour rule. */
export function directionColor(
  p: Palette,
  value: number,
  ref: number,
  higherIsBetter: boolean,
): string {
  if (!Number.isFinite(ref) || value === ref) return p.accent;
  const better = value > ref === higherIsBetter;
  return better ? p.good : p.danger;
}

/** Left margin sized to the widest tick, so a $250.03M label is never clipped
 *  and a small chart is not given a 70px gutter it does not need. The advance
 *  is derived from the tick size so the gutter grows with the type. */
export function measureTickWidth(
  ticks: number[],
  format?: Format,
): number {
  let w = 0;
  for (const t of ticks) w = Math.max(w, formatTick(t, format).length);
  return Math.max(28, Math.ceil(w * FONT.tick * CHAR_WIDTH) + 10);
}

/** Truncate a category label to the space available, with an ellipsis. */
export function truncate(s: string, max: number): string {
  return s.length <= max ? s : `${s.slice(0, Math.max(1, max - 1))}…`;
}
