/**
 * The palette, as data.
 *
 * This file is the single source of truth for colour and `tokens.css` mirrors
 * it key for key — the two must agree or a D3 chart floats off its card.
 *
 * It exists as a typed object and not only as CSS custom properties because a
 * D3 module cannot read a CSS variable: `getComputedStyle` on an SVG element
 * returns the *resolved* value only for properties that element actually uses,
 * and scales, gradients and interpolators need the raw colour at build time,
 * not at paint time. So the palette is passed into every `render()` as a
 * parameter and a theme change re-renders rather than re-cascades.
 *
 * ── NTT DATA brand ────────────────────────────────────────────────────────
 * NTT Blue is #007BC1 and the wordmark black is #231F20, both read off the
 * official lockup. #007BC1 is kept as `brand` and used for the brand rule and
 * the logo surround, but it is NOT the UI accent on the dark shell: it measures
 * 3.89:1 on #111827, under the 4.5:1 that 12px interface text needs. The accent
 * is a tint of the same hue (202°) — #2E9BDB at 5.77:1 — so the brand hue is
 * everywhere and still legible. Light does the reverse and darkens to #006AA8
 * (5.79:1 on white).
 *
 * ── RAG under colour-vision deficiency ────────────────────────────────────
 * Roughly 8% of men cannot separate red from green, so "good" is pulled to
 * teal (hue 174°) and "danger" to red-orange (hue 10°) — 164° of separation,
 * and they differ in the blue channel as well as in lightness, which is what
 * actually survives deuteranopia. Amber sits between them at 36°. Every status
 * colour also carries a non-colour partner somewhere in the UI (a glyph, a
 * hatch, a position), because colour alone is never the only encoding.
 *
 * Measured contrast, status on its own surface:
 *   dark  (#111827)   good 7.01   warn 8.23   danger 6.35   accent 5.77
 *   light (#FFFFFF)   good 5.47   warn 5.02   danger 5.44   accent 5.79
 */

export type ThemeName = "dark" | "light";
export type PersonaKey = "ae" | "manager" | "executive";

export interface Palette {
  name: ThemeName;
  bg: string;
  surface: string;
  surface2: string;
  line: string;
  accent: string;
  accentRgb: string;
  accentDim: string;
  /** True NTT Blue. Brand rule and logo surround only — never interface text. */
  brand: string;
  warn: string;
  warnRgb: string;
  danger: string;
  dangerRgb: string;
  good: string;
  goodRgb: string;
  muted: string;
  mutedRgb: string;
  text: string;
  text2: string;
  onAccent: string;
  accentSoftText: string;
  track: string;
  /** Chart-only tokens. */
  grid: string;
  heatLow: string;
  cellStroke: string;
  /** Alpha of the wash a pill paints behind its own colour. Theme-dependent. */
  pillWash: number;
  /** ── The three chart ramps ──────────────────────────────────────────── */
  /** Qualitative. Okabe-Ito, adapted per theme. Eight is the hard ceiling:
   *  past that no set stays separable and the answer is to group the tail. */
  categorical: string[];
  /** Sequential, low → high. For heat grids and treemaps. Single hue, so the
   *  reader never has to decide whether a hue change means a value change. */
  sequential: string[];
  /** Diverging, low → mid → high. For variance against a plan or a norm,
   *  where the midpoint is meaningful and both directions matter. */
  diverging: string[];
  /** Per-persona identity accent. Applied to the persona pill, the scope strip
   *  and the action-card rule — never inside a chart, where hue is a measure. */
  persona: Record<PersonaKey, string>;
}

export const DARK: Palette = {
  name: "dark",
  bg: "#0A0E1A",
  surface: "#111827",
  surface2: "#0E1522",
  line: "#1E2A3F",

  // A 202° tint of NTT Blue. 5.77:1 on surface, 6.26:1 on the page ground.
  accent: "#2E9BDB",
  accentRgb: "46,155,219",
  accentDim: "#0F2A3F",
  brand: "#007BC1",

  // Amber, 36°. 8.23:1.
  warn: "#E8A33D",
  warnRgb: "232,163,61",
  // Red-ORANGE, 10° — not a pure red, so it separates from teal under
  // deuteranopia by more than lightness alone. 6.35:1.
  danger: "#F2765C",
  dangerRgb: "242,118,92",
  // TEAL, 174° — not a pure green, for the same reason. 7.01:1.
  good: "#1FB6A6",
  goodRgb: "31,182,166",

  muted: "#8B96A8",
  mutedRgb: "139,150,168",
  text: "#E7ECF3",
  text2: "#C7D0DC",

  // White on #2E9BDB is only 3.08:1; the page's own near-black is 6.26:1 and is
  // the standard knock-out for a bright accent in a dark UI.
  onAccent: "#0A0E1A",
  accentSoftText: "#9ED0F2",
  track: "#131C2E",

  grid: "#1E2A3F",
  heatLow: "#0F2A3F",
  cellStroke: "#111827",
  pillWash: 0.14,

  categorical: [
    "#3D9BE0", // blue        5.88
    "#E69F00", // orange      7.88
    "#28B892", // blue-green  7.06
    "#DE93BC", // purple      7.61
    "#6FC3F0", // sky         9.07
    "#F07A3C", // vermillion  6.38
    "#D9CE3F", // yellow     11.0
    "#9AA6B5", // grey        7.18
  ],
  // Single-hue NTT Blue ramp, dark floor to bright peak. The floor is a visible
  // navy rather than the card colour: an unpopulated cell is not drawn at all,
  // so too dark a floor makes "lowest value" and "no value" identical.
  sequential: ["#12293D", "#154A6B", "#176D9B", "#1E90C8", "#4FB0E4", "#8FD0F1"],
  // danger → neutral → good, matching the RAG hues so a variance chart and a
  // status pill never disagree about which end is which.
  diverging: ["#C8452C", "#F2765C", "#5C6675", "#1FB6A6", "#0E8478"],
  persona: { ae: "#00A3A1", manager: "#E8A33D", executive: "#8A93F9" },
};

export const LIGHT: Palette = {
  name: "light",
  bg: "#F4F6FA",
  surface: "#FFFFFF",
  surface2: "#F9FAFC",
  line: "#DDE3EC",

  // A 202° shade of NTT Blue. 5.79:1 on white; white knocks out on it at 5.79.
  accent: "#006AA8",
  accentRgb: "0,106,168",
  accentDim: "#E3F0F9",
  brand: "#007BC1",

  warn: "#B45309", // 26°, 5.02:1
  warnRgb: "180,83,9",
  danger: "#C0392B", // 6°, 5.44:1
  dangerRgb: "192,57,43",
  good: "#0F766E", // 175°, 5.47:1
  goodRgb: "15,118,110",

  muted: "#55606F",
  mutedRgb: "85,96,111",
  text: "#0E1726",
  text2: "#2C3849",

  onAccent: "#FFFFFF",
  accentSoftText: "#006AA8",
  track: "#E6EAF1",

  grid: "#DDE3EC",
  // NOT a near-white tint: an unpopulated heat cell is not drawn at all, so too
  // pale a floor makes "low value" and "empty" indistinguishable.
  heatLow: "#D6E7F4",
  cellStroke: "#FFFFFF",
  pillWash: 0.08,

  categorical: [
    "#0072B2", // 5.19
    "#B36A00", // 4.22
    "#00795C", // 5.40
    "#A65384", // 5.01
    "#2B7FB8", // 4.35
    "#C8500A", // 4.56
    "#8A7A00", // 4.32
    "#5A6472", // 6.00
  ],
  sequential: ["#D6E7F4", "#A9CDEA", "#71ADDA", "#3C8FC7", "#12719F", "#004E7C"],
  diverging: ["#A32B1D", "#C0392B", "#9AA3AF", "#0F766E", "#0A554F"],
  persona: { ae: "#00807E", manager: "#9A6A12", executive: "#4B54C4" },
};

export const PALETTES: Record<ThemeName, Palette> = { dark: DARK, light: LIGHT };

/** Semantic tone -> palette colour. The only place tone is resolved to hue. */
export type Tone = "accent" | "good" | "warn" | "danger" | "neutral";

export function toneColor(p: Palette, tone: Tone): string {
  switch (tone) {
    case "good":
      return p.good;
    case "warn":
      return p.warn;
    case "danger":
      return p.danger;
    case "neutral":
      return p.muted;
    default:
      return p.accent;
  }
}

export function toneRgb(p: Palette, tone: Tone): string {
  switch (tone) {
    case "good":
      return p.goodRgb;
    case "warn":
      return p.warnRgb;
    case "danger":
      return p.dangerRgb;
    case "neutral":
      return p.mutedRgb;
    default:
      return p.accentRgb;
  }
}

/** `rgba()` from an *-rgb token. Charts never hardcode an alpha colour. */
export function alpha(rgb: string, a: number): string {
  return `rgba(${rgb}, ${a})`;
}

/**
 * Pick from a ramp by position in 0..1, with linear interpolation between
 * stops. Used by heat grids, treemaps and the mekko so every one of them reads
 * the same ramp rather than inventing its own gradient.
 */
export function rampAt(ramp: string[], t: number): string {
  if (!ramp.length) return "#888888";
  const x = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1) * (ramp.length - 1);
  const i = Math.floor(x);
  const j = Math.min(i + 1, ramp.length - 1);
  return mix(ramp[i], ramp[j], x - i);
}

/** Diverging ramp keyed to a midpoint: `v` below `mid` runs to the low end. */
export function divergingAt(p: Palette, v: number, mid: number, span: number): string {
  const t = span > 0 ? (v - mid) / span : 0;
  return rampAt(p.diverging, (Math.min(Math.max(t, -1), 1) + 1) / 2);
}

/** The categorical ramp, cycled. Past eight the answer is to group the tail,
 *  and any caller relying on the cycle is drawing too many series. */
export function categoricalAt(p: Palette, i: number): string {
  return p.categorical[((i % p.categorical.length) + p.categorical.length) % p.categorical.length];
}

function mix(a: string, b: string, t: number): string {
  const pa = hex(a);
  const pb = hex(b);
  const c = (k: 0 | 1 | 2) => Math.round(pa[k] + (pb[k] - pa[k]) * t);
  return `#${[c(0), c(1), c(2)].map((n) => n.toString(16).padStart(2, "0")).join("")}`;
}

function hex(h: string): [number, number, number] {
  const s = h.replace("#", "");
  const f = s.length === 3 ? s.split("").map((c) => c + c).join("") : s;
  return [
    parseInt(f.slice(0, 2), 16),
    parseInt(f.slice(2, 4), 16),
    parseInt(f.slice(4, 6), 16),
  ];
}
