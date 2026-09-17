/**
 * Theme as React state AND as a DOM attribute AND as a typed object.
 *
 * The shell reads it through CSS custom properties; the D3 modules cannot (a
 * chart is handed the `Palette` object as a render parameter, the chart repository contract rule
 * 2), so both representations have to change together. Putting the palette in
 * context is what guarantees that: every chart re-renders because its palette
 * prop changed identity, not because something remembered to tell it to.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { Palette, ThemeName } from "./palette";
import { applyTheme, paletteFor, readTheme, writeTheme } from "./store";

interface ThemeCtx {
  theme: ThemeName;
  palette: Palette;
  setTheme(t: ThemeName): void;
  toggle(): void;
}

const Ctx = createContext<ThemeCtx | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeName>(() => readTheme());

  useEffect(() => {
    applyTheme(theme);
    writeTheme(theme);
  }, [theme]);

  const setTheme = useCallback((t: ThemeName) => setThemeState(t), []);
  const toggle = useCallback(() => setThemeState((t) => (t === "dark" ? "light" : "dark")), []);

  const value = useMemo<ThemeCtx>(
    () => ({ theme, palette: paletteFor(theme), setTheme, toggle }),
    [theme, setTheme, toggle],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTheme(): ThemeCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useTheme outside ThemeProvider");
  return c;
}
