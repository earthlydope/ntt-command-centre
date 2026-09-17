/**
 * The wire contract, as the semantic layer actually serves it.
 *
 * Do not add a field here that `api/` does not send. Add it to the Python
 * payload first, verify it against a real response, then mirror it here.
 */

export type Tone = "accent" | "good" | "warn" | "danger" | "neutral";
export type CountBasis = "opportunities" | "lines";
export type PersonaKey = "ae" | "manager" | "executive";

/** Pages are persona-scoped; the union is every page any persona can reach. */
export type Lens =
  | "my-day" | "my-deals" | "my-accounts" | "my-record"
  | "pod-pulse" | "rep-benchmark" | "process" | "calibration" | "pod-whitespace"
  | "tldr" | "performance" | "structure" | "risks" | "growth" | "actions";

export type DimKey =
  | "stage" | "forecast" | "lob" | "portfolio" | "industry"
  | "orderType" | "quarter" | "country" | "rep" | "account"
  | "riskBand" | "anomalyCategory";

export type Measure = "gp" | "revenue";
export type Urgency = "Critical" | "High" | "Medium" | "Low";
export type Format = "currency" | "percent" | "number" | "days";
export type RiskBand = "Low" | "Watch" | "High" | "Critical";

export type Shape =
  | "categorical×measure"
  | "temporal×measure"
  | "categorical×categorical×measure"
  | "categorical×series×measure"
  | "target×actual"
  | "bridge"
  | "temporal×measure×measure"
  | "cohort×stage"
  | "x×y×size"
  | "categorical×measure×width"
  | "hierarchy×measure"
  | "source×target×measure"
  | "entity×start×end"
  | "table";

/** One generated or computed sentence. `lens` is what licenses it to exist:
 *  a `state` sentence is only allowed when no chart on screen says the same. */
export interface Sentence {
  text: string;
  lens: "cause" | "norm" | "delta" | "action" | "answer" | "state";
  claim?: string | null;
  bold?: string[];
}

export interface Narrative {
  headline: string;
  sentences: Sentence[];
  provider?: string;
  model?: string;
  latencyMs?: number;
  degraded?: boolean;
  cached?: boolean;
  reason?: string;
  /** Provider diagnostics. Carried for debugging; never rendered to a user. */
  attempts?: { provider: string; ms: number; outcome: string }[];
  /** Sentences the server removed. Shown in the provenance drawer, not hidden. */
  dropped?: { text: string; reason: string }[];
  rejected?: { text: string; rejectedTokens: string[] }[];
}

export interface Kpi {
  key: string;
  label: string;
  value: number;
  formatted: string;
  sub: string;
  tone: Tone;
  /** Colour encodes goodness, not sign: past-due rising is red. */
  direction: "up-good" | "up-bad" | "neutral";
  /** A glyph key, resolved to a mark by the tile. Never a brand or product name. */
  icon?: string;
  /** A genuine monthly series, or empty. The server omits it rather than
   *  faking one for a snapshot measure — see `measures.trend`. */
  spark?: number[];
}

/** A row-shaped payload, or an object payload for the structured chart types
 *  (heat grid, mekko, sankey, gantt, combo, treemap, bubble). Each module
 *  narrows it to the payload its own CONTRACT entry declares. */
export type ChartData = Record<string, unknown>[] | Record<string, unknown>;

export interface ChartSpec {
  id: string;
  title: string;
  subtitle: string;
  shape: Shape;
  /** Resolved server-side; the client re-runs the same rule as a check. */
  repositoryKey: string;
  data: ChartData;
  /** Which field of a row carries x, y, the series and the target. Only the
   *  row-shaped types use it; the structured payloads name their own fields. */
  encoding?: {
    x: string;
    y: string;
    series?: string;
    target?: string;
    label?: string;
  };
  measureLabel: string;
  format: Format;
  clickDim?: DimKey | null;
  countBasis?: CountBasis | null;
  basisNote?: string | null;
  /** Claim keys this chart puts on screen. The AI may not restate them. */
  says: string[];
  footnote?: string | null;
  height?: number | null;
  /** What "no rows" means for THIS chart. Rendered instead of mounting the
   *  module at all, so a legitimately empty slice reads as a finding rather
   *  than as a generic blank. */
  emptyMessage?: string | null;
}

export interface RiskFactor {
  key: string;
  label: string;
  points: number;
  detail: string;
}

export interface ActionCard {
  n: number;
  key: string;
  persona: PersonaKey;
  entity: { type: string; id: string; label: string };
  headline: string;
  why: string[];
  /** The literal rule that fired, shown behind "Why this fired". */
  predicate: string;
  owner: string;
  nextStep: string;
  severity: number;
  urgency: number;
  priority: number;
  urgencyLabel: Urgency;
  valueAtStake: number;
  framing: "risk" | "opportunity";
  /** What sort of decision it is. A page ranks its rail by this, so the cards
   *  that answer the page's question come first — see `actions.PAGE_FOCUS`. */
  kind?: "deal" | "rep_pattern" | "coverage" | "concentration" | "growth";
  scopeTo?: { dim: DimKey; value: string } | null;
  drill?: string | null;
}

/**
 * One of the three things the platform watches, as the executive brief states
 * it: a name, one line, two or three figures already formatted by the server,
 * and the page that proves it. Sent under `extras.useCases` on the brief only.
 */
export interface UseCase {
  key: "opportunities" | "anomalies" | "closure";
  title: string;
  oneLine: string;
  figures: { label: string; formatted: string }[];
  page: Lens;
  cta: string;
}

export interface Scope {
  label: string;
  predicate: string;
  persona: PersonaKey;
  identity: string;
}

export interface ViewPayload {
  page: Lens;
  label: string;
  question: string;
  persona: PersonaKey;
  asOf: string;
  fy: string;
  quarter: string;
  scope: Scope;
  filters: { dim: string; label: string; value: string }[];
  measure: Measure;
  kpis: Kpi[];
  narrative: Narrative;
  actions: ActionCard[];
  charts: ChartSpec[];
  /** Union of every chart's `says`. Passed to the AI so it cannot repeat them. */
  chartsSay: string[];
  extras: Record<string, unknown>;
  measures: Record<string, unknown>;
}

export interface PersonaDef {
  key: PersonaKey;
  label: string;
  role: string;
  jtbd: string;
  home: Lens;
  pages: Lens[];
  accent: string;
  scopeColumn: string | null;
}

export interface IdentityOption {
  id: string;
  label: string;
  openDeals?: number;
  opportunities?: number;
  gp?: number;
  reps?: number;
  pod?: string;
}

export interface MetaPayload {
  asOf: string;
  fy: string;
  quarter: string;
  persona: {
    active: {
      persona: PersonaKey;
      label: string;
      role: string;
      jtbd: string;
      identity: string;
      identityLabel: string;
      predicate: string;
      home: Lens;
      pages: Lens[];
      denyMeasures: string[];
      accent: string;
    };
    personas: PersonaDef[];
    identities: Record<PersonaKey, IdentityOption[]>;
    rosterRule: string;
  };
  pages: { key: Lens; label: string; question: string }[];
  allPages: { key: Lens; persona: PersonaKey; label: string; question: string }[];
  dimensions: {
    key: DimKey;
    label: string;
    column: string;
    countBasis: CountBasis;
    basisNote: string;
    validated: boolean;
    description: string;
    values: string[];
  }[];
  measures: { key: Measure; label: string; default: boolean }[];
  data: Record<string, number>;
  provenance: {
    distribution: {
      column: string;
      category: string;
      given_pct: number;
      synthetic_pct: number;
      difference_pp: number;
      status: string;
    }[];
    note: string;
  };
  modelCard: ModelCard;
  dsModel: Record<string, unknown>;
  anomalyTaxonomy: { categories: { name: string; question: string }[] };
  stallThreshold: number;
}

export interface ModelCard {
  primary: Record<string, unknown> & { available: boolean; honestRead?: string };
  independent: Record<string, unknown>;
  riskModel: {
    name: string;
    kind: string;
    why: string;
    factors: { key: string; label: string; maxPoints: number }[];
    bands: { band: RiskBand; from: number; to: number }[];
  };
}

export interface DealDetail {
  opportunityCode: string;
  name: string;
  account: string;
  owner: string;
  stage: string;
  lob: string;
  portfolio: string;
  acvRevenue: number;
  acvGp: number;
  closeDate: string | null;
  riskScore: number;
  riskBand: RiskBand;
  riskFactors: RiskFactor[];
  valueAtRisk: number;
  quietDays: number | null;
  stagePath: string;
  pWin?: number;
  pWinSegment?: number | null;
  repConfidence?: number;
  confidenceGap?: number;
  expectedGp?: number;
  /** Which model produced pWin. "independent" means the DS drop does not
   *  cover this deal and this layer's own classifier stood in, in which case
   *  every DS field below is null: there is no SHAP driver for a number the
   *  DS model did not produce. */
  pWinSource?: "ds-model" | "independent";
  /** The DS model's per-deal SHAP sentence, verbatim from the workbook —
   *  "Stage progression (Identification) decreased win probability the most". */
  drivingForce?: string | null;
  /** The workbook feature behind that sentence, raw and in plain words. */
  driverFeature?: string | null;
  driverLabel?: string | null;
  /** Whether that feature pushed pWin up or down for this deal. */
  driverDirection?: "up" | "down" | null;
  /** The two above as one table-cell line, "Stage reached · lowers pWin",
   *  worded by the server so every surface prints the same words. */
  dsDriver?: string | null;
  /** The workbook's literal risk bucket (1 = Dark Red / Very High Risk) and
   *  its label, plus the label of the quantile-relative bucket. */
  riskBucket?: number | null;
  riskBucketLabel?: string | null;
  riskBucketRelativeLabel?: string | null;
  timeline: { date: string; field: string; from: string; to: string; by: string }[];
  benchmarks: {
    feature: string;
    label: string;
    value: number;
    benchmark: number;
    benchmarkMethod: string;
    direction: string;
    correlationWithWin: number;
    verdict: "GREEN" | "RED";
    aboveBenchmark: boolean;
    weakSignal: boolean;
    note: string | null;
  }[];
  findings: Record<string, unknown>[];
}

export interface AskResponse {
  question: string;
  plan?: Record<string, unknown>;
  answer: Narrative;
  chart?: ChartSpec | null;
  rows?: Record<string, unknown>[];
  shape?: Shape;
  chartWhy?: string;
  claim?: string;
  provider?: string;
  degraded?: boolean;
  refused?: boolean;
  cached?: boolean;
  /** Questions the server can answer, sent with a refusal so the panel can
   *  offer a way forward rather than a dead end. */
  suggestions?: string[];
}

export interface Digest {
  windowDays: number;
  asOf: string;
  totalChanges: number;
  opportunitiesTouched: number;
  byField: Record<string, number>;
  stageMoves: { opportunity: string; from: string; to: string; by: string; date: string }[];
  busiest: { opportunity_name: string; changes: number }[];
  note: string;
}
