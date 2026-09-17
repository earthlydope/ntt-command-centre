/**
 * The React <-> D3 boundary, and the only place the two meet.
 *
 * React owns the <svg> element's existence; the chart module owns everything
 * inside it and nothing outside it. The effect therefore has exactly one job:
 * call render, keep the teardown, and call the teardown before anything else
 * happens to that element — including React StrictMode's deliberate
 * double-invocation in development, which is why the teardown is guarded so it
 * can be called twice safely. the render/teardown contract.
 *
 * Width comes from a ResizeObserver on the wrapper, never from a constant:
 * rule 6.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ChartSpec, DimKey } from "../api/types";
import { useTheme } from "../theme/ThemeProvider";
import { resolve } from "./registry";
import type { RenderOpts, Teardown } from "./types";

export interface UseChartArgs {
  spec: ChartSpec;
  onFilter?: (dim: DimKey, value: string) => void;
  selected?: Partial<Record<DimKey, string | null>>;
  /**
   * The height offered to the module. Modules that draw to a height use it
   * (each keeping its own floor); modules that derive their height from the
   * data ignore it. It comes from the spec, or from the layout pass when a
   * card is asked to match the card beside it — never from a measurement of
   * the wrapper, which would make the chart's height depend on itself.
   */
  height?: number;
}

/** What a module is offered when neither the spec nor the layout says. The
 *  layout pass reads the same number when it estimates a card's height. */
export const DEFAULT_HEIGHT = 240;

export function useChart({ spec, onFilter, selected, height }: UseChartArgs) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const { palette } = useTheme();
  const [width, setWidth] = useState(0);
  const resolution = resolve(spec);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // Seed synchronously: the first ResizeObserver callback lands a frame late
    // and an unseeded chart would render once at width 0 and flash.
    setWidth(el.clientWidth);
    const ro = new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      setWidth((prev) => (Math.abs(prev - w) > 1 ? w : prev));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || width < 40) return;
    const opts: RenderOpts = {
      width,
      height: height ?? DEFAULT_HEIGHT,
      onFilter,
      selected,
      reduceMotion: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    };
    let teardown: Teardown | null = resolution.module.render(svg, spec, palette, opts);
    return () => {
      const t = teardown;
      teardown = null; // idempotent: StrictMode calls the cleanup twice
      t?.();
    };
    // `palette` is a dependency on purpose — theme is a render parameter, so a
    // theme change re-renders the chart rather than re-cascading CSS into it.
  }, [spec, palette, width, height, onFilter, selected, resolution.module]);

  return { wrapRef, svgRef, width, resolution };
}
