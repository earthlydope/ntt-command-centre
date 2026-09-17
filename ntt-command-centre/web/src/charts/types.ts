/**
 * The chart repository's own contract. the chart repository contract.
 *
 * Praveen asked for "a repository of charts from which to use for a particular
 * data… and render based on the question asked." So the repository is indexed
 * by the SHAPE of the data and resolved by a rule, never by a component picked
 * by hand at the call site.
 */
import type { Palette } from "../theme/palette";
import type { ChartSpec, DimKey, Shape } from "../api/types";

export type Teardown = () => void;

export interface RenderOpts {
  /** Measured from the container by ResizeObserver. No module hardcodes a width. */
  width: number;
  height: number;
  /** A mark click is a filter click. This is the reason D3 was chosen. */
  onFilter?: (dim: DimKey, value: string) => void;
  /** Current filter state, so a mark can show itself selected. */
  selected?: Partial<Record<DimKey, string | null>>;
  /** Honour prefers-reduced-motion; set by the wrapper, not sniffed per module. */
  reduceMotion?: boolean;
}

export interface ChartModule {
  key: string;
  serves: Shape | "fallback";
  /**
   * Renders into `root`'s subtree and owns nothing outside it. Returns a
   * teardown that must be safe to call twice — React StrictMode double-invokes
   * every effect in development.
   */
  render(root: SVGSVGElement, spec: ChartSpec, palette: Palette, opts: RenderOpts): Teardown;
  /**
   * The height the module will draw when offered `offered`, before the card's
   * own chrome. The layout pass asks this to decide whether two half-width
   * cards can share a row at one height. Absent, the module is taken to draw
   * to the offer exactly. A module that sizes itself from the data — a row per
   * category — answers with what the data needs, stretched towards the offer
   * as far as its pitch allows, so the layout learns whether the pair will
   * meet or whether each card must take a row of its own.
   */
  drawnHeight?(spec: ChartSpec, offered: number): number;
}

export type { ChartSpec, Shape, DimKey };
