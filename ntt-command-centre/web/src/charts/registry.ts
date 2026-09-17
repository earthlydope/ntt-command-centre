/**
 * THE CHART REPOSITORY.
 *
 * The client asked for "a repository of charts from which to use for a particular
 * data… and render based on the question asked." The index is the SHAPE of the
 * data, and `selectChart()` is a rule — a lookup in one table — never a guess
 * and never a component chosen by hand at a call site.
 *
 * Adding a chart type means adding a module and one row in SHAPE_TO_KEY.
 * Nothing else in the app changes.
 *
 * Fourteen modules now. The nine advanced types were added because each answers
 * a question the basic set cannot: a bridge shows *what moved* a number rather
 * than its level, a mekko shows size and mix and margin at once, a sankey shows
 * where deals actually flow rather than where they sit, a gantt shows which
 * deals have run out of road. A chart type earns its place by being the best
 * answer to a real question in this dataset — not by being available.
 */
import type { ChartModule, Shape } from "./types";
import type { ChartSpec } from "../api/types";
import { barCategorical } from "./bar.categorical";
import { lineTimeseries } from "./line.timeseries";
import { heatGrid } from "./heat.grid";
import { barStacked } from "./bar.stacked";
import { bulletTarget } from "./bullet.target";
import { tableCompact } from "./table.compact";
import { waterfallBridge } from "./waterfall.bridge";
import { comboColumnLine } from "./combo.columnline";
import { funnelStage } from "./funnel.stage";
import { bubbleScatter } from "./bubble.scatter";
import { mekkoMarimekko } from "./mekko.marimekko";
import { treemapNested } from "./treemap.nested";
import { sankeyFlow } from "./sankey.flow";
import { ganttTimeline } from "./gantt.timeline";

export const REPOSITORY: Record<string, ChartModule> = {
  [barCategorical.key]: barCategorical,
  [lineTimeseries.key]: lineTimeseries,
  [heatGrid.key]: heatGrid,
  [barStacked.key]: barStacked,
  [bulletTarget.key]: bulletTarget,
  [tableCompact.key]: tableCompact,
  [waterfallBridge.key]: waterfallBridge,
  [comboColumnLine.key]: comboColumnLine,
  [funnelStage.key]: funnelStage,
  [bubbleScatter.key]: bubbleScatter,
  [mekkoMarimekko.key]: mekkoMarimekko,
  [treemapNested.key]: treemapNested,
  [sankeyFlow.key]: sankeyFlow,
  [ganttTimeline.key]: ganttTimeline,
};

/** The fallback. An unregistered shape degrades to a table, never to a throw. */
export const FALLBACK_KEY = "table.compact";

const SHAPE_TO_KEY: Record<Shape, string> = {
  "categorical×measure": "bar.categorical",
  "temporal×measure": "line.timeseries",
  "categorical×categorical×measure": "heat.grid",
  "categorical×series×measure": "bar.stacked",
  "target×actual": "bullet.target",
  bridge: "waterfall.bridge",
  "temporal×measure×measure": "combo.columnline",
  "cohort×stage": "funnel.stage",
  "x×y×size": "bubble.scatter",
  "categorical×measure×width": "mekko.marimekko",
  "hierarchy×measure": "treemap.nested",
  "source×target×measure": "sankey.flow",
  "entity×start×end": "gantt.timeline",
  // A server-chosen table is a real answer for a ranked list with several
  // numeric columns, not only a degradation.
  table: "table.compact",
};

/** The rule. The API calls the same rule server-side and stamps the result on
 *  the spec as `repositoryKey`; the client re-runs it as a check. */
export function selectChart(shape: Shape | string): string {
  return SHAPE_TO_KEY[shape as Shape] ?? FALLBACK_KEY;
}

export interface Resolution {
  module: ChartModule;
  key: string;
  /** True when the spec asked for something the repository does not have and
   *  we fell back. Surfaced in the card footer rather than swallowed. */
  degraded: boolean;
}

/**
 * Resolve a spec to a module.
 *
 * `repositoryKey` wins when it names a module we actually have — the server may
 * legitimately override the shape rule (a ranked list with two counts reads
 * better as a table than as bars). Otherwise the shape rule decides. Either way
 * the result is a module, so a lens never renders an exception.
 */
export function resolve(spec: ChartSpec): Resolution {
  const asked = spec.repositoryKey;
  if (asked && REPOSITORY[asked]) {
    return { module: REPOSITORY[asked], key: asked, degraded: false };
  }
  const key = selectChart(spec.shape);
  const module = REPOSITORY[key] ?? REPOSITORY[FALLBACK_KEY];
  return {
    module,
    key: module.key,
    degraded: !!asked && asked !== module.key,
  };
}

export const REPOSITORY_KEYS = Object.keys(REPOSITORY);
