/**
 * How the evidence grid is packed.
 *
 * Layout arithmetic only — which cell is wide, and how tall a chart may draw
 * itself — never a business figure. The grid is two columns; a shape that
 * reads as a lie at half width takes both. Everything else pairs up, with two
 * exceptions that both exist for the same reason: a half-width card is only a
 * good idea when the card beside it is about as tall, because the grid row is
 * as tall as its taller card and whatever the shorter one does not fill is a
 * hole on the page. So:
 *
 *   - a half-width chart left without a partner in its row takes the full row;
 *   - a pair is offered ONE height, the taller of the two, and each module
 *     answers with what it would actually draw at that offer. A module that
 *     draws to whatever it is offered meets the offer exactly; a module that
 *     sizes itself from its rows — a bar per category — grows its pitch
 *     towards the offer and stops at its cap. If the two answers land within
 *     a footer's worth of each other the pair stands, both cards carry the
 *     same offer, and the grid stretches the shorter card so the residue sits
 *     between its chart and its footer rather than under the card. If they do
 *     not, each takes a row of its own.
 *
 * The heights are the modules' OWN answers, asked through `drawnHeight` on
 * the module (see charts/types.ts), so a change to a module's row pitch or
 * chrome cannot leave this file believing a stale copy of it. A module without
 * `drawnHeight` draws to the offer. The answers are used only to decide
 * packing; they are never shown and never measured back — a wrong estimate
 * costs a slightly worse packing, nothing more.
 */
import type { ChartSpec } from "../api/types";
import { resolve } from "../charts/registry";
import type { ChartModule } from "../charts/types";
import { DEFAULT_HEIGHT } from "../charts/useChart";

/**
 * Shapes that read as a lie at half width: a funnel needs its ladder, a flow
 * needs its span, a grid needs its columns. Everything else pairs up.
 */
export const WIDE_SHAPES = new Set<ChartSpec["shape"]>([
  "cohort×stage",
  "source×target×measure",
  "entity×start×end",
  "categorical×categorical×measure",
  "hierarchy×measure",
  "categorical×measure×width",
  "bridge",
  "table",
]);

/** A footnote under a chart is a wrapped line or two in a padded box. */
const FOOTNOTE_HEIGHT = 52;

/**
 * Below this, two cards of unequal height still read as a pair: the grid
 * stretches the shorter card to the row and the difference sits between its
 * chart and its footer, where it looks like breathing room rather than a hole.
 */
const PAIR_TOLERANCE = 40;

/** A chart is never asked to draw taller than this, whatever it is paired with. */
const MAX_OFFERED_HEIGHT = 560;

export interface PlacedChart {
  spec: ChartSpec;
  /** Spans both columns. */
  wide: boolean;
}

/** The height the module will draw when offered `offered`, before the card's
 *  own chrome. A module that does not answer draws to the offer. */
function drawnHeight(spec: ChartSpec, module: ChartModule, offered: number): number {
  return module.drawnHeight ? module.drawnHeight(spec, offered) : offered;
}

/** What the spec asks for, or what a module is offered when it asks nothing. */
function naturalOffer(spec: ChartSpec): number {
  return spec.height ?? DEFAULT_HEIGHT;
}

/**
 * The same charts array yields the same placement, by identity. A view
 * payload is one object for as long as it is on screen, and a placement that
 * hands back a fresh spec object on every render would make useChart tear the
 * chart down and redraw it each time the page re-rendered for an unrelated
 * reason, such as the brief arriving.
 */
const memo = new WeakMap<ChartSpec[], PlacedChart[]>();

/** Pack the charts into rows so no row is left half empty. */
export function layoutCharts(charts: ChartSpec[]): PlacedChart[] {
  const hit = memo.get(charts);
  if (hit) return hit;

  const placed: PlacedChart[] = charts.map((spec) => ({
    spec,
    wide: WIDE_SHAPES.has(spec.shape),
  }));

  // Walk the rows. A half-width card waits for a partner; a wide card ends
  // the row and orphans whatever was waiting.
  let pending: PlacedChart | null = null;
  for (const p of placed) {
    if (p.wide) {
      if (pending) pending.wide = true;
      pending = null;
    } else if (pending) {
      settlePair(pending, p);
      pending = null;
    } else {
      pending = p;
    }
  }
  if (pending) pending.wide = true;

  memo.set(charts, placed);
  return placed;
}

/**
 * Two half-width cards that will share a row. Either they end up about the
 * same height, or they each take a row of their own.
 */
function settlePair(a: PlacedChart, b: PlacedChart): void {
  const modA = resolve(a.spec).module;
  const modB = resolve(b.spec).module;
  const footA = a.spec.footnote ? FOOTNOTE_HEIGHT : 0;
  const footB = b.spec.footnote ? FOOTNOTE_HEIGHT : 0;

  // Each card's height left to itself, chart plus footnote.
  const naturalA = drawnHeight(a.spec, modA, naturalOffer(a.spec)) + footA;
  const naturalB = drawnHeight(b.spec, modB, naturalOffer(b.spec)) + footB;

  // One card height for the row: the taller of the two, capped. Each chart is
  // offered that height less its own footnote, so the CARDS meet even when
  // only one of them carries a caveat.
  const target = Math.min(MAX_OFFERED_HEIGHT, Math.max(naturalA, naturalB));
  const offerA = Math.max(1, target - footA);
  const offerB = Math.max(1, target - footB);
  const metA = drawnHeight(a.spec, modA, offerA) + footA;
  const metB = drawnHeight(b.spec, modB, offerB) + footB;

  if (Math.abs(metA - metB) > PAIR_TOLERANCE) {
    a.wide = true;
    b.wide = true;
    return;
  }

  // The pair stands. Both carry the shared offer — new spec objects, because
  // the payload's are shared with everything else that reads them — and a
  // spec whose own height already equals the offer keeps its identity.
  if (a.spec.height !== offerA) a.spec = { ...a.spec, height: offerA };
  if (b.spec.height !== offerB) b.spec = { ...b.spec, height: offerB };
}
