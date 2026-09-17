/**
 * Theme persistence. the theme contract — dark is the default, both themes are
 * first-class, the choice survives a reload.
 *
 * The same resolution runs as a blocking inline script in index.html. That
 * duplication is deliberate: if the attribute is only written once the bundle
 * has parsed, a light-theme user gets a dark flash on every load.
 */
import { PALETTES, type Palette, type ThemeName } from "./palette";

export const STORAGE_KEY = "acc.theme";

export function readTheme(): ThemeName {
  // `?theme=` wins over the stored choice, so a view is fully reproducible from
  // its URL — the same property that lets an Outlook card link back into an
  // exact slice, and what makes a screenshot of a lens repeatable.
  try {
    const q = new URLSearchParams(location.search).get("theme");
    if (q === "light" || q === "dark") return q;
  } catch {
    // no location (a jsdom render for the card leg) — fall through
  }
  try {
    const t = localStorage.getItem(STORAGE_KEY);
    if (t === "light" || t === "dark") return t;
  } catch {
    // private window, blocked storage — fall through to the default
  }
  return "dark";
}

export function writeTheme(t: ThemeName): void {
  try {
    localStorage.setItem(STORAGE_KEY, t);
  } catch {
    // a theme that cannot be persisted still has to apply for this session
  }
}

/** The single place `data-theme` is written. `color-scheme` comes with it via
 *  the token blocks, so form controls and scrollbars follow. */
export function applyTheme(t: ThemeName): void {
  document.documentElement.setAttribute("data-theme", t);
}

export const paletteFor = (t: ThemeName): Palette => PALETTES[t];
