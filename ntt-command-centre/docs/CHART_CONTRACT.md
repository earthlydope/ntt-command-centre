# Chart spec contract — the binding agreement between `api/semantic/charts.py` and `web/src/charts/`

The server builds a `ChartSpec`; the client renders it. Neither side may invent a field.
Every spec carries these common keys:

```ts
interface ChartSpecBase {
  id: string;                 // stable per lens+slice, used as a React key
  title: string;
  subtitle?: string;
  shape: Shape;               // drives repository resolution
  repositoryKey: string;      // the server's own resolution; the client re-runs the rule as a check
  measureLabel: string;       // "ACV GP" | "ACV Revenue"
  format: "currency" | "percent" | "number" | "days";
  countBasis?: "lines" | "opportunities";
  basisNote?: string;         // printed in the card footer
  clickDim?: DimKey;          // set => marks are filter buttons
  says: string[];             // claim keys this chart puts on screen (see AI layer)
  footnote?: string;          // honesty note, e.g. "Oct–Dec has no closed history yet"
  height?: number;            // server hint; client clamps
}
```

`format: "days"` renders a bare integer with a `d` suffix. `countBasis`/`basisNote` come from
`dimensions.REGISTRY` and must be printed verbatim.

---

## The fourteen repository keys

| key | shape | data payload |
|---|---|---|
| `bar.categorical` | `categorical×measure` | `Cat[]` |
| `line.timeseries` | `temporal×measure` | `Point[]` |
| `bar.stacked` | `categorical×series×measure` | `{ categories: string[]; series: Series[] }` |
| `heat.grid` | `categorical×categorical×measure` | `{ rows: string[]; cols: string[]; cells: Cell[] }` |
| `bullet.target` | `target×actual` | `Bullet[]` |
| `table.compact` | *fallback* | `{ columns: Col[]; rows: Record<string, unknown>[] }` |
| `waterfall.bridge` | `bridge` | `Step[]` |
| `combo.columnline` | `temporal×measure×measure` | `{ points: ComboPoint[]; barLabel: string; lineLabel: string; lineFormat: Format }` |
| `funnel.stage` | `cohort×stage` | `FunnelStage[]` |
| `bubble.scatter` | `x×y×size` | `{ points: Bubble[]; xLabel: string; yLabel: string; sizeLabel: string; xFormat: Format; yFormat: Format }` |
| `mekko.marimekko` | `categorical×measure×width` | `{ columns: MekkoCol[]; fillLabel: string; fillMid: number }` |
| `treemap.nested` | `hierarchy×measure` | `{ nodes: TreeNode[] }` |
| `sankey.flow` | `source×target×measure` | `{ nodes: SankeyNode[]; links: SankeyLink[] }` |
| `gantt.timeline` | `entity×start×end` | `{ bars: GanttBar[]; asOf: string; rangeStart: string; rangeEnd: string }` |

---

## Payload types — exact

```ts
type Format = "currency" | "percent" | "number" | "days";
type Tone = "good" | "warn" | "danger" | "neutral" | "accent";

interface Cat   { key: string; value: number; label?: string; share?: number;
                  lines?: number; opps?: number; gm?: number; tone?: Tone }

interface Point { key: string; value: number; label?: string }

interface Series { name: string; values: number[] }          // values align to categories[]

interface Cell  { row: string; col: string; value: number;   // value drives the colour ramp
                  label?: string; secondary?: number; secondaryLabel?: string;
                  zero?: boolean }                            // true => outlined, not shaded

interface Bullet { key: string; actual: number; target: number; label?: string;
                   format?: Format; higherIsBetter?: boolean }

interface Col   { key: string; label: string; format?: Format; align?: "left" | "right" }

// --- waterfall.bridge -----------------------------------------------------
// Steps run left to right. `type` decides the geometry, not the sign.
interface Step  { key: string; label: string; value: number;
                  type: "start" | "delta" | "subtotal" | "end";
                  tone?: Tone; note?: string }

// --- combo.columnline -----------------------------------------------------
interface ComboPoint { key: string; bar: number; line: number; label?: string;
                       tone?: Tone; annotation?: string }

// --- funnel.stage ---------------------------------------------------------
// MUST be cohort-anchored by the server. A raw stage count is not a funnel:
// on this dataset it exceeds 100% because deals are born mid-ladder.
interface FunnelStage { key: string; label: string; count: number; value: number;
                        cohortPct: number;      // % of the entry cohort, monotone non-increasing
                        stepPct?: number;       // % of the immediately previous stage
                        lostHere?: number }

// --- bubble.scatter -------------------------------------------------------
interface Bubble { id: string; label: string; x: number; y: number; size: number;
                   category?: string; tone?: Tone; href?: string }

// --- mekko.marimekko ------------------------------------------------------
// Column width ∝ column total; segment height ∝ share within the column.
interface MekkoCol { key: string; total: number;
                     segments: { key: string; value: number; fill: number }[] }
                     // `fill` is the diverging-ramp value (e.g. GM% vs fillMid)

// --- treemap.nested -------------------------------------------------------
// Flat list; `parent` builds the hierarchy. Server applies top-N + "Other".
interface TreeNode { id: string; label: string; parent: string | null;
                     value: number; tone?: Tone; secondary?: number }

// --- sankey.flow ----------------------------------------------------------
interface SankeyNode { id: string; label: string; depth: number; tone?: Tone }
interface SankeyLink { source: string; target: string; value: number;
                       tone?: Tone; note?: string }

// --- gantt.timeline -------------------------------------------------------
interface GanttBar { id: string; label: string; sublabel?: string;
                     start: string; end: string;      // ISO dates
                     value: number; tone?: Tone;
                     pastDue?: boolean;               // tail beyond asOf drawn in danger
                     stalled?: boolean;               // hatched
                     quietDays?: number }
```

---

## Non-negotiable rules for every module

1. **`attachMark()` for every interactive mark.** Hover, keyboard focus and click-to-filter come
   from that one helper; a module that wires its own listeners will be rejected.
2. **`markGroup()` for dense charts.** A chart with more than ~20 marks must expose ONE tab stop
   for the group and use arrow keys to move within it, or a keyboard user hits 200 tab stops.
3. **Selection = outline + dim the rest.** Never a hue change: hue already encodes performance.
   `selectionState()` and `DIM_OPACITY` are provided.
4. **The palette is a render parameter.** A D3 module cannot read a CSS variable; a theme change
   re-renders rather than re-cascading.
5. **Pure and idempotent.** React StrictMode double-invokes effects; the teardown must be safe to
   call twice, and a second render into the same root must produce the same picture.
6. **No hardcoded width or height.** `opts.width`/`opts.height` are measured by `ResizeObserver`.
7. **Degrade, never throw.** Empty data renders a centred "No rows in this slice" message.
8. **Negative values.** `treemap`, `mekko` and `funnel` cannot represent them; if any arrive,
   render the empty-state message naming the problem rather than drawing a lie.
