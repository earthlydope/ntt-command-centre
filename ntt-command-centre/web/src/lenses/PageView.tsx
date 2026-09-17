/**
 * ONE page template, fourteen pages.
 *
 * The composition is fixed and deliberate, because a person who learns one page
 * has learned all of them:
 *
 *   the question, stated  →  KPIs  →  the AI panel  →  the action rail  →
 *   the active filters  →  the evidence grid  →  the block that makes this page
 *   itself
 *
 * Two rules govern everything below.
 *
 * NOTHING HERE COMPUTES. Every figure on screen is a field of `ViewPayload`,
 * formatted by `lib/format` and nothing else. The only arithmetic in this file
 * is fraction→percent (`0.42` → `42%`), which is a display unit and not a
 * claim. There is no total, no average and no count of anything except the
 * length of a list the server already decided to send.
 *
 * THE AI MAY NOT RESTATE A CHART. `payload.chartsSay` is the union of every
 * chart's claim keys; it is handed to `useBrief` so the server can refuse any
 * sentence that duplicates a mark already on screen. The page renders complete
 * and correct in one round trip and the prose upgrades in place when it lands —
 * a product that waits on a language model to paint its first pixel looks
 * broken every time the model is slow.
 *
 * AS_OF arrives on the payload. This file never asks the browser what day it is.
 */
import { useMemo, useState, type ReactNode } from "react";
import type {
  DimKey,
  Lens,
  ModelCard,
  Tone,
  UseCase,
  ViewPayload,
} from "../api/types";
import { ActionRail } from "../components/ActionRail";
import { AiPanel } from "../components/AiPanel";
import { ChartCard } from "../components/ChartCard";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { Findings } from "../components/Findings";
import { KpiRow } from "../components/KpiRow";
import { days, longDate, money, monthLabel, multiple, num, pct } from "../lib/format";
import { api } from "../api/client";
import { useApp } from "../state/AppStateProvider";
import { DIM_KEYS } from "../state/filters";
import { layoutCharts } from "./chartLayout";
import { useBrief, useView } from "./useView";

/* ========================================================================== *
 * Entry point
 * ========================================================================== */

/**
 * The outer shell exists for one reason: a retry that actually retries.
 *
 * `useView` is keyed on the request, not on a nonce, so the only honest way to
 * re-issue a failed fetch is to remount the component that owns it. The counter
 * is that remount, and it means the error state offers a real button rather
 * than an apology.
 */
export function PageView() {
  const [attempt, setAttempt] = useState(0);
  return <PageBody key={attempt} onRetry={() => setAttempt((a) => a + 1)} />;
}

/** A module-level constant so the brief's cache key does not churn on identity. */
const NO_CLAIMS: string[] = [];

function PageBody({ onRetry }: { onRetry: () => void }) {
  const { state } = useApp();
  const view = useView();
  // Hooks run unconditionally and in the same order on every render; the brief
  // simply does not fire until the payload it must not repeat has arrived.
  const chartsSay = view.data?.chartsSay ?? NO_CLAIMS;
  const brief = useBrief(chartsSay, view.data !== null);

  if (view.error !== null) {
    return <PageError page={state.page} detail={view.error} onRetry={onRetry} />;
  }
  if (view.data === null) return <PageSkeleton page={state.page} />;
  return <Page payload={view.data} brief={brief} />;
}

/* ========================================================================== *
 * Loading and failure — designed, not deferred
 * ========================================================================== */

/**
 * The skeleton is the real layout with the content removed, so nothing moves
 * when the payload lands. Screen readers get the one sentence that matters
 * instead of a field of empty boxes.
 */
function PageSkeleton({ page }: { page: Lens }) {
  return (
    <div className="pv pv--loading">
      <p className="pv-sr" role="status">
        Loading the {page} page.
      </p>
      <div className="pv-head" aria-hidden="true">
        <div className="pv-skel pv-skel--eyebrow" />
        <div className="pv-skel pv-skel--title" />
        <div className="pv-skel pv-skel--meta" />
      </div>
      <div className="pv-skel-kpis" aria-hidden="true">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="pv-skel pv-skel--kpi" />
        ))}
      </div>
      <div className="pv-skel pv-skel--panel" aria-hidden="true" />
      <div className="pv-skel-rail" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <div key={i} className="pv-skel pv-skel--action" />
        ))}
      </div>
      <div className="pv-charts" aria-hidden="true">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="pv-charts__cell">
            <div className="pv-skel pv-skel--chart" />
          </div>
        ))}
      </div>
    </div>
  );
}

/** What failed, in the user's terms, and the button that tries it again. */
function PageError({
  page,
  detail,
  onRetry,
}: {
  page: Lens;
  detail: string;
  onRetry: () => void;
}) {
  return (
    <div className="pv pv--error">
      <section className="pv-error" role="alert">
        <p className="pv-error__label">This page did not load</p>
        <h2 className="pv-error__title">
          The semantic layer did not answer for <code>{page}</code>
        </h2>
        <p className="pv-error__detail">{detail}</p>
        <p className="pv-error__note">
          Nothing is shown rather than something stale: a figure from the previous
          scope would be wrong on this one.
        </p>
        <button type="button" className="pv-error__retry" onClick={onRetry}>
          Try again
        </button>
      </section>
    </div>
  );
}

/* ========================================================================== *
 * The page
 * ========================================================================== */

function Page({
  payload,
  brief,
}: {
  payload: ViewPayload;
  brief: ReturnType<typeof useBrief>;
}) {
  const { clearFilters, onFilter, openDrawer, openAsk, ctx } = useApp();

  return (
    <section className="pv" aria-labelledby="pv-question">
      {/* 1. The question, stated. Every page is an answer to a sentence a
          person would actually say out loud, so the sentence is the heading —
          and it is the ONLY heading. The page's name is already lit in the
          sidebar, the scope is already in the persona control, and the as-of
          date was on screen twice more; all three are gone from here. */}
      <header className="pv-head">
        <h1 className="pv-head__question" id="pv-question">
          {payload.question}
        </h1>
        <p className="pv-head__meta">
          {payload.scope.label} · {payload.quarter} · as of {longDate(payload.asOf)}
        </p>
      </header>

      {/* 2. KPIs. */}
      {payload.kpis.length > 0 ? <KpiRow kpis={payload.kpis} /> : null}

      {/* 3. The AI, which may say only what no chart here already says. */}
      <AiPanel
        narrative={brief.data ?? payload.narrative}
        loading={brief.loading}
        error={brief.error}
        scope={payload.scope.label}
        asOf={payload.asOf}
      />

      {/* 4. What to do about it. */}
      {payload.actions.length > 0 ? (
        <ActionRail
          actions={payload.actions}
          asOf={payload.asOf}
          onScope={(scope) => onFilter(scope.dim, scope.value)}
          onDrill={openDrawer}
          onExplain={(card) => api.explain(ctx, card, payload.chartsSay)}
        />
      ) : null}

      {/* 5. What is currently narrowing the page, and the way out. */}
      <FilterBar
        filters={payload.filters}
        onToggle={onFilter}
        onClearAll={clearFilters}
      />

      {/* 6. The evidence. */}
      {payload.charts.length > 0 ? (
        <div className="pv-charts">
          {layoutCharts(payload.charts).map(({ spec: c, wide }) => (
            <div
              key={c.id}
              className={`pv-charts__cell${wide ? " pv-charts__cell--wide" : ""}`}
            >
              {/* Each card is fenced on its own: a module that cannot draw its
                  slice fails as one calm card, not as a blank page. The spec
                  object is the reset key, so a new payload retries it. */}
              <ErrorBoundary title={c.title} resetKey={c}>
                <ChartCard
                  spec={c}
                  height={c.height ?? undefined}
                  ask={{
                    chartsSay: payload.chartsSay,
                    onExpand: (question, seed) => openAsk(question, seed),
                  }}
                />
              </ErrorBoundary>
            </div>
          ))}
        </div>
      ) : null}

      {/* 7. The block that makes this page itself rather than a generic lens. */}
      <PageExtras payload={payload} />
    </section>
  );
}

/* ========================================================================== *
 * Active filters
 * ========================================================================== */

const KNOWN_DIMS = new Set<string>(DIM_KEYS);

function asDim(d: string): DimKey | null {
  return KNOWN_DIMS.has(d) ? (d as DimKey) : null;
}

/**
 * The filter deck, read back off the payload rather than off local state: what
 * is shown is what the server actually applied, which is the only version that
 * can be trusted to match the numbers beside it.
 */
function FilterBar({
  filters,
  onToggle,
  onClearAll,
}: {
  filters: ViewPayload["filters"];
  onToggle: (dim: DimKey, value: string) => void;
  onClearAll: () => void;
}) {
  if (filters.length === 0) return null;
  return (
    <div className="pv-filters" role="group" aria-label="Active filters">
      <span className="pv-filters__label">Filtered</span>
      {filters.map((f) => {
        const dim = asDim(f.dim);
        if (dim === null) {
          return (
            <span className="pv-filters__chip pv-filters__chip--fixed" key={f.dim}>
              <span className="pv-filters__dim">{f.label}</span>
              <span className="pv-filters__value">{f.value}</span>
            </span>
          );
        }
        return (
          <button
            type="button"
            className="pv-filters__chip"
            key={f.dim}
            onClick={() => onToggle(dim, f.value)}
            aria-label={`Remove the ${f.label} filter on ${f.value}`}
          >
            <span className="pv-filters__dim">{f.label}</span>
            <span className="pv-filters__value">{f.value}</span>
            <span className="pv-filters__x" aria-hidden="true">
              ✕
            </span>
          </button>
        );
      })}
      <button type="button" className="pv-filters__clear" onClick={onClearAll}>
        Clear all
      </button>
    </div>
  );
}

/* ========================================================================== *
 * Reading `extras`
 *
 * `extras` is `Record<string, unknown>` on the wire because each page puts a
 * different block in it. These readers narrow it without a cast and without
 * `any`: a field the server did not send reads as null and renders as an em
 * dash, which is the honest answer to "what is this number" when there isn't
 * one. Nothing below ever substitutes a zero for a missing value.
 * ========================================================================== */

type Row = Record<string, unknown>;

const isRow = (v: unknown): v is Row => typeof v === "object" && v !== null && !Array.isArray(v);

function rowsAt(extras: Row, key: string): Row[] {
  const v = extras[key];
  return Array.isArray(v) ? v.filter(isRow) : [];
}

function rowAt(extras: Row, key: string): Row | null {
  const v = extras[key];
  return isRow(v) ? v : null;
}

function text(r: Row, k: string): string | null {
  const v = r[k];
  return typeof v === "string" && v.length > 0 ? v : null;
}

function number(r: Row, k: string): number | null {
  const v = r[k];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function flag(r: Row, k: string): boolean {
  return r[k] === true;
}

function strings(r: Row, k: string): string[] {
  const v = r[k];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

/** A percentage the server sends as a fraction. Unit conversion, not a claim. */
const rate = (v: number | null): string => (v === null ? "—" : pct(v * 100, 0));
/** A percentage the server already sends as 0–100. */
const percent = (v: number | null, dp = 0): string => (v === null ? "—" : pct(v, dp));
const cash = (v: number | null): string => (v === null ? "—" : money(v));
const count = (v: number | null): string => (v === null ? "—" : num(Math.round(v)));

/** A z-score reads as a distance from the norm, so it always carries its sign. */
const signed = (v: number | null): string =>
  v === null ? "—" : `${v > 0 ? "+" : ""}${num(Math.round(v * 100) / 100)}`;

function ragTone(rag: string | null): Tone {
  const r = (rag ?? "").trim().toLowerCase();
  if (r.startsWith("g")) return "good";
  if (r.startsWith("a") || r.startsWith("y")) return "warn";
  if (r.startsWith("r")) return "danger";
  return "neutral";
}

function bandTone(band: string | null): Tone {
  switch ((band ?? "").trim()) {
    case "Low":
      return "good";
    case "Watch":
      return "neutral";
    case "High":
      return "warn";
    case "Critical":
      return "danger";
    default:
      return "neutral";
  }
}

/** "2026-09" is not a month anybody reads out loud. */
function monthName(m: string | null): string {
  if (m === null) return "—";
  return /^\d{4}-\d{2}$/.test(m) ? monthLabel(`${m}-01`) : m;
}

/* -------------------------------------------------------------------------- *
 * Shared primitives
 * -------------------------------------------------------------------------- */

function Tag({
  tone = "neutral",
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span className={`pv-tag pv-tag--${tone}${className ? ` ${className}` : ""}`}>
      {children}
    </span>
  );
}

function Block({
  title,
  lead,
  children,
  tone,
}: {
  title: string;
  lead?: string;
  children: ReactNode;
  tone?: "upside";
}) {
  return (
    <section className={`pv-block${tone === "upside" ? " pv-block--upside" : ""}`}>
      <h2 className="pv-block__title">{title}</h2>
      {lead ? <p className="pv-block__lead">{lead}</p> : null}
      {children}
    </section>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="pv-empty">{children}</p>;
}

/* ========================================================================== *
 * The page-specific block
 * ========================================================================== */

function PageExtras({ payload }: { payload: ViewPayload }) {
  const x = payload.extras;
  switch (payload.page) {
    case "my-deals":
      return <DealsBlock extras={x} />;
    case "my-accounts":
    case "pod-whitespace":
      return <WhitespaceBlock extras={x} />;
    case "rep-benchmark":
    case "calibration":
    case "pod-pulse":
      return <RepsBlock extras={x} />;
    case "performance":
      return <PerformanceBlock extras={x} />;
    case "structure":
      return <StructureBlock extras={x} />;
    case "risks":
      return <RisksBlock extras={x} />;
    case "tldr":
      return <TldrBlock extras={x} />;
    default:
      // my-day, my-record, process and actions carry no block of their own:
      // their answer is entirely in the KPIs, the rail and the charts.
      return null;
  }
}

/* -------------------------------------------------------------------- deals */

/**
 * The open book, ranked by risk. Risk is deterministic — every point comes from
 * an observable fact about the deal — which is why each row can name the driver
 * that put it here, and why opening the row is worth doing.
 */
function DealsBlock({ extras }: { extras: Row }) {
  const { openDrawer } = useApp();
  const deals = rowsAt(extras, "deals");

  return (
    <Block
      title="Your open deals, riskiest first"
      lead="Risk is scored from what the deal has actually done — silence, slippage, shrinkage — not from a model's opinion of it. Open a deal for the factors that fired."
    >
      {deals.length === 0 ? (
        <Empty>No open opportunities in this slice.</Empty>
      ) : (
        <div className="pv-tablewrap">
          <table className="pv-table pv-table--deals">
            <caption className="pv-sr">
              Open opportunities ranked by deal risk score
            </caption>
            <thead>
              <tr>
                <th scope="col">Deal</th>
                <th scope="col">Stage</th>
                <th scope="col">LOB</th>
                <th scope="col" className="pv-num">
                  Risk
                </th>
                <th scope="col">Top driver</th>
                <th scope="col" className="pv-num">
                  Quiet
                </th>
                <th scope="col">Close</th>
                <th scope="col" className="pv-num">
                  ACV GP
                </th>
              </tr>
            </thead>
            <tbody>
              {deals.map((d, i) => {
                const code = text(d, "opportunityCode");
                const name = text(d, "name") ?? code ?? "—";
                const band = text(d, "riskBand");
                const close = text(d, "closeDate");
                return (
                  <tr key={code ?? i}>
                    <th scope="row" className="pv-table__deal">
                      {code ? (
                        <button
                          type="button"
                          className="pv-linkbtn"
                          onClick={() => openDrawer(`deal:${code}`)}
                        >
                          {name}
                        </button>
                      ) : (
                        <span>{name}</span>
                      )}
                      <span className="pv-table__sub">{text(d, "account") ?? "—"}</span>
                    </th>
                    <td>{text(d, "stage") ?? "—"}</td>
                    <td>{text(d, "lob") ?? "—"}</td>
                    <td className="pv-num">
                      <span className="pv-risk">
                        <span className="pv-risk__score">{count(number(d, "riskScore"))}</span>
                        <Tag tone={bandTone(band)}>{band ?? "—"}</Tag>
                      </span>
                    </td>
                    <td className="pv-table__driver">{text(d, "topDriver") ?? "—"}</td>
                    <td className="pv-num">
                      {number(d, "quietDays") === null
                        ? "—"
                        : days(number(d, "quietDays") as number)}
                    </td>
                    <td>{close === null ? "—" : longDate(close)}</td>
                    <td className="pv-num">{cash(number(d, "gp"))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Block>
  );
}

/* --------------------------------------------------------------- whitespace */

/**
 * Whitespace is upside, and the page says so in as many words. The same figures
 * framed as "gaps" read as an accusation of neglect; framed as room to grow
 * they read as a plan. The estimate's method is printed on every card because a
 * cross-sell number nobody can reproduce is a number nobody will act on.
 */
function WhitespaceBlock({ extras }: { extras: Row }) {
  const { openDrawer } = useApp();
  const accounts = rowsAt(extras, "whitespace");

  return (
    <Block
      title="Room to grow"
      tone="upside"
      lead="Accounts that already buy from us and do not yet buy everything. Each estimate is the MEDIAN gross profit that comparable accounts earn from the missing line — never the best comparable."
    >
      {accounts.length === 0 ? (
        <Empty>
          Every account in this slice already holds the full line-up, or none is large
          enough to compare.
        </Empty>
      ) : (
        <ul className="pv-ws">
          {accounts.map((a, i) => {
            const code = text(a, "accountCode");
            const name = text(a, "accountName") ?? code ?? "—";
            const holds = strings(a, "holds");
            const comparables = number(a, "comparables");
            return (
              <li className="pv-ws__card" key={code ?? i}>
                <div className="pv-ws__top">
                  <h3 className="pv-ws__name">
                    {code ? (
                      <button
                        type="button"
                        className="pv-linkbtn"
                        onClick={() => openDrawer(`account:${code}`)}
                      >
                        {name}
                      </button>
                    ) : (
                      name
                    )}
                  </h3>
                  <Tag tone="good" className="pv-ws__flag">
                    Upside
                  </Tag>
                </div>
                <p className="pv-ws__who">
                  {text(a, "industry") ?? "—"} · {text(a, "owner") ?? "unowned"} ·{" "}
                  {cash(number(a, "gp"))} ACV GP today
                </p>

                <p className="pv-ws__caption">Buys today</p>
                <ul className="pv-ws__holds">
                  {holds.length === 0 ? (
                    <li className="pv-ws__hold pv-ws__hold--none">not yet recorded</li>
                  ) : (
                    holds.map((h) => (
                      <li className="pv-ws__hold" key={h}>
                        {h}
                      </li>
                    ))
                  )}
                </ul>

                <div className="pv-ws__rec">
                  <p className="pv-ws__caption">Recommended next line</p>
                  <p className="pv-ws__lob">{text(a, "recommendedLob") ?? "—"}</p>
                  <dl className="pv-ws__figs">
                    <div className="pv-ws__fig">
                      <dt>Peer attach rate</dt>
                      <dd>{rate(number(a, "peerAttachRate"))}</dd>
                    </div>
                    <div className="pv-ws__fig">
                      <dt>Estimated ACV GP</dt>
                      <dd className="pv-ws__value">{cash(number(a, "estimatedGp"))}</dd>
                    </div>
                    <div className="pv-ws__fig">
                      <dt>Comparables</dt>
                      <dd>
                        {comparables === null || comparables === 0
                          ? "not size-matched"
                          : count(comparables)}
                      </dd>
                    </div>
                  </dl>
                </div>

                <p className="pv-ws__method">{text(a, "method") ?? ""}</p>
              </li>
            );
          })}
        </ul>
      )}
    </Block>
  );
}

/* --------------------------------------------------------------------- reps */

/**
 * The six rates the server scores a rep on. `phrase` is the behaviour in the
 * words a manager would use for it, and it heads every chip; `column` is the
 * shorter form a table head can hold. `suffix` disambiguates the two rates that
 * read as a size when they are really a frequency — "deal value shrinks 57%"
 * would be read as the deal losing 57% of its value.
 */
const REP_METRICS = [
  { key: "win_rate", phrase: "Wins", column: "Wins", suffix: "", goodHigh: true },
  {
    key: "shrink_rate",
    phrase: "Deal value shrinks",
    column: "Value shrinks",
    suffix: " of the time",
    goodHigh: false,
  },
  {
    key: "inflate_rate",
    phrase: "Deal value inflates",
    column: "Value inflates",
    suffix: " of the time",
    goodHigh: false,
  },
  {
    key: "regression_rate",
    phrase: "Forecast goes backwards",
    column: "Forecast back",
    suffix: "",
    goodHigh: false,
  },
  { key: "stall_rate", phrase: "Deals go quiet", column: "Go quiet", suffix: "", goodHigh: false },
  {
    key: "new_business_share",
    phrase: "New business",
    column: "New business",
    suffix: "",
    goodHigh: true,
  },
] as const;

type RepMetric = (typeof REP_METRICS)[number];

/**
 * Where a rate stops being noise. These are DISPLAY thresholds on the z-scores
 * the server already computed — they decide whether a sentence is worth
 * printing, never what the number is. One σ is the conventional "worth a
 * look"; two is "clearly not the same pattern".
 */
const NOTABLE_Z = 1;
const CLEAR_Z = 2;

/** One behaviour that is far enough from the team to be said out loud. */
type Deviation = {
  metric: RepMetric;
  value: number | null;
  peer: number | null;
  z: number;
  /** True when the distance runs in the direction the business wants. */
  good: boolean;
};

/**
 * The verdict tiers, ordered so a sort on `level` puts the rep who most needs
 * a conversation first. Thin reps sit under everyone because a 100% rate on
 * two deals is a sample-size artefact, not a pattern.
 */
type RepRead = {
  name: string;
  deals: number | null;
  thin: boolean;
  level: 0 | 1 | 2 | 3;
  verdict: string;
  tone: Tone;
  /** Every behaviour at or past NOTABLE_Z: the bad-direction ones first, then the good, each by distance. */
  deviations: Deviation[];
  /** The largest |z| in any direction — the tie-break inside a tier. */
  largest: number;
};

/**
 * Turns one server row into the sentence a card prints. This is a display
 * rule, not a business figure: every input is a z-score the server sent, and
 * the only thing decided here is which of four fixed phrases describes it.
 *
 * The verdict is driven by the BAD direction only. A rep who wins far more
 * than the team is different, but not in a way that needs coaching, so the
 * headline stays "on the team pattern" and the good chips underneath say the
 * rest. The tiers are: any bad-direction distance at or past CLEAR_Z reads
 * "well off"; at or past NOTABLE_Z reads "a little off"; otherwise on pattern.
 */
function readRep(r: Row, norms: Row | null): RepRead {
  const name = text(r, "rep") ?? "—";
  const deals = number(r, "deals");
  const thin = flag(r, "thin");

  const deviations: Deviation[] = [];
  let largest = 0;
  let worstBad = 0;
  for (const metric of REP_METRICS) {
    const z = number(r, `${metric.key}_z`);
    if (z === null) continue;
    const size = Math.abs(z);
    largest = Math.max(largest, size);
    const good = (z > 0) === metric.goodHigh;
    if (!good) worstBad = Math.max(worstBad, size);
    if (size >= NOTABLE_Z) {
      deviations.push({
        metric,
        value: number(r, metric.key),
        peer: number(r, `${metric.key}_peer`) ?? (norms ? number(norms, metric.key) : null),
        z,
        good,
      });
    }
  }
  // The chips that explain the verdict come first, then the good news, each
  // group with the largest distance leading.
  deviations.sort(
    (a, b) => Number(a.good) - Number(b.good) || Math.abs(b.z) - Math.abs(a.z),
  );

  if (thin) {
    return {
      name,
      deals,
      thin,
      level: 0,
      verdict: "Too few deals to judge",
      tone: "neutral",
      deviations: [],
      largest,
    };
  }
  if (worstBad >= CLEAR_Z) {
    return {
      name,
      deals,
      thin,
      level: 3,
      verdict: "Well off the team pattern",
      tone: "danger",
      deviations,
      largest,
    };
  }
  if (worstBad >= NOTABLE_Z) {
    return {
      name,
      deals,
      thin,
      level: 2,
      verdict: "A little off the pattern",
      tone: "warn",
      deviations,
      largest,
    };
  }
  return {
    name,
    deals,
    thin,
    level: 1,
    verdict: "On the team pattern",
    tone: "good",
    deviations,
    largest,
  };
}

type SortDir = "asc" | "desc";

/**
 * How each rep compares with the team.
 *
 * The customer's words were that the behaviour against the pod had to be
 * simple to understand, and a table of value / peer / σ triplets was the
 * opposite of that. So the default view is a card per rep carrying one verdict
 * and, for each behaviour that is clearly away from the team, one sentence in
 * plain words with the rep's number beside the team's. Nothing else is shown
 * unless it is notable — a rep on the team pattern gets a card that says so
 * and stops. The full comparison table survives behind a disclosure for the
 * manager who wants every rate, and the σ that drives the flags is explained
 * there rather than printed on every cell.
 */
function RepsBlock({ extras }: { extras: Row }) {
  const reps = rowsAt(extras, "reps");
  const norms = rowAt(extras, "peerNorms");
  const [sort, setSort] = useState<{ key: string; dir: SortDir }>({
    key: "deals",
    dir: "desc",
  });

  // Most off-pattern first: the verdict tier, then the largest distance in
  // any direction inside the tier, then the name so the order is stable.
  // Thin reps are tier 0 and so always sit last.
  const cards = useMemo(
    () =>
      reps
        .map((r) => readRep(r, norms))
        .sort(
          (a, b) => b.level - a.level || b.largest - a.largest || a.name.localeCompare(b.name),
        ),
    [reps, norms],
  );

  const sorted = useMemo(() => {
    const isRateSort = REP_METRICS.some((m) => m.key === sort.key);
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...reps].sort((a, b) => {
      if (isRateSort) {
        // Thin reps are not comparable on a rate; they sit under the sort.
        const ta = flag(a, "thin") ? 1 : 0;
        const tb = flag(b, "thin") ? 1 : 0;
        if (ta !== tb) return ta - tb;
      }
      if (sort.key === "rep") {
        return (text(a, "rep") ?? "").localeCompare(text(b, "rep") ?? "") * dir;
      }
      const va = number(a, sort.key);
      const vb = number(b, sort.key);
      if (va === null && vb === null) return 0;
      if (va === null) return 1; // missing sorts last in both directions
      if (vb === null) return -1;
      return (va - vb) * dir;
    });
  }, [reps, sort]);

  const toggle = (key: string) =>
    setSort((s) =>
      s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" },
    );

  const ariaSort = (key: string): "ascending" | "descending" | "none" =>
    sort.key !== key ? "none" : sort.dir === "asc" ? "ascending" : "descending";

  // A column head is an element in a mapped list, so it carries the key itself.
  const head = (key: string, label: string, numeric = true) => (
    <th
      key={key}
      scope="col"
      className={`pv-sortable${numeric ? " pv-num" : ""}`}
      aria-sort={ariaSort(key)}
    >
      <button type="button" className="pv-sortbtn" onClick={() => toggle(key)}>
        {label}
        <span className="pv-sortbtn__dir" aria-hidden="true">
          {sort.key === key ? (sort.dir === "asc" ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </th>
  );

  return (
    <Block
      title="How each rep compares with the team"
      lead="Only clear differences from the team's usual rates are shown. Everything is judged against this team, not against a target."
    >
      {reps.length === 0 ? (
        <Empty>No reps in this scope.</Empty>
      ) : (
        <>
          <p className="pv-reps__how">
            Each rep is compared with the other reps in the team. We only flag a
            behaviour when it is clearly away from the team's usual rate.
          </p>

          <ul className="pv-reps" aria-label="Reps, most off the team pattern first">
            {cards.map((c, i) => (
              <RepCard key={c.name === "—" ? i : c.name} read={c} />
            ))}
          </ul>

          <details className="pv-reps-table">
            <summary className="pv-reps-table__summary">
              <span className="pv-reps-table__show">Show the full table</span>
              <span className="pv-reps-table__hide">Hide the full table</span>
            </summary>
            <div className="pv-tablewrap">
              <table className="pv-table pv-table--reps">
                <caption className="pv-sr">
                  Every rate for every rep, with the team's usual rate under each.
                  Sortable by column.
                </caption>
                <thead>
                  <tr>
                    {head("rep", "Rep", false)}
                    {head("deals", "Deals")}
                    {REP_METRICS.map((m) => head(m.key, m.column))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((r, i) => {
                    const thin = flag(r, "thin");
                    const name = text(r, "rep") ?? "—";
                    return (
                      <tr key={name === "—" ? i : name} className={thin ? "pv-row--thin" : undefined}>
                        <th scope="row" className="pv-table__rep">
                          <span className="pv-table__repname">{name}</span>
                          {thin ? (
                            <Tag tone="neutral" className="pv-thin">
                              too few deals to judge
                            </Tag>
                          ) : null}
                        </th>
                        <td className="pv-num">{count(number(r, "deals"))}</td>
                        {REP_METRICS.map((m) => {
                          const v = number(r, m.key);
                          const peer =
                            number(r, `${m.key}_peer`) ?? (norms ? number(norms, m.key) : null);
                          // A thin rep is never scored, so its value never takes a tone.
                          const z = thin ? null : number(r, `${m.key}_z`);
                          const tone = zTone(z, m.goodHigh);
                          return (
                            <td className="pv-num pv-reps-td" key={m.key}>
                              <span className={`pv-reps-td__v pv-reps-td__v--${tone}`}>
                                {rate(v)}
                                {tone === "neutral" || z === null ? null : (
                                  // The colour says "far enough to matter"; a screen
                                  // reader gets the distance the colour stands for.
                                  <span className="pv-sr">, {signed(z)}σ from the team</span>
                                )}
                              </span>
                              <span className="pv-reps-td__peer">team {rate(peer)}</span>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="pv-note">
              A coloured rate is one that sits at least 1σ from the team's usual rate;
              the flags above use the same rule and call a rate clearly away at 2σ. σ is
              the distance from the team's usual rate, which the server measures on the
              credible reps only — a handful of two-deal reps cannot drag the norm the
              rest are measured against. Rows marked <em>too few deals to judge</em> are
              shown but never scored, and never sort to the top of a rate.
            </p>
          </details>
        </>
      )}
    </Block>
  );
}

/**
 * One rep. The verdict is the whole card for most reps; the chips appear only
 * when there is something specific to say, each one a sentence with the rep's
 * rate beside the team's so the reader never has to hold a norm in their head.
 */
function RepCard({ read }: { read: RepRead }) {
  const kind = read.thin ? "thin" : read.level >= 2 ? "off" : "on";
  return (
    <li className={`pv-rep pv-rep--${kind}`}>
      <div className="pv-rep__head">
        <h3 className="pv-rep__name">{read.name}</h3>
        <span className="pv-rep__deals">
          {count(read.deals)} {read.deals === 1 ? "deal" : "deals"}
        </span>
      </div>
      <p className={`pv-rep__verdict pv-rep__verdict--${read.tone}`}>{read.verdict}</p>
      {read.deviations.length > 0 ? (
        <ul className="pv-rep__chips" aria-label="Behaviours clearly away from the team">
          {read.deviations.map((d) => (
            <li
              key={d.metric.key}
              className={`pv-rep__chip pv-tag pv-tag--${d.good ? "good" : "danger"}`}
            >
              <span className="pv-rep__chip-what">{d.metric.phrase}</span>{" "}
              <span className="pv-rep__chip-v">{rate(d.value)}</span>
              {d.metric.suffix}
              <span className="pv-rep__chip-sep" aria-hidden="true">
                {" · "}
              </span>
              <span className="pv-sr">, </span>
              <span className="pv-rep__chip-peer">team {rate(d.peer)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="pv-rep__none">
          {read.thin
            ? "Rates are in the full table but are not compared with the team."
            : "No behaviour is clearly away from the team's usual rate."}
        </p>
      )}
    </li>
  );
}

/** Colour only where the distance is large enough to be worth a conversation. */
function zTone(z: number | null, goodHigh: boolean): Tone {
  if (z === null || Math.abs(z) < NOTABLE_Z) return "neutral";
  const above = z > 0;
  return above === goodHigh ? "good" : "danger";
}

/* -------------------------------------------------------------- performance */

function PerformanceBlock({ extras }: { extras: Row }) {
  const quarters = rowsAt(extras, "quarters");
  const months = rowsAt(extras, "months");

  return (
    <Block
      title="Plan against actual"
      lead="The RAG status is the export's own, carried through rather than recomputed here. A future period carries its full plan and almost no closed business — that is a calendar position, not a performance finding."
    >
      <QuartersTable quarters={quarters} />
      <h3 className="pv-subhead">By month</h3>
      {months.length === 0 ? (
        <Empty>No monthly plan in this slice.</Empty>
      ) : (
        <div className="pv-tablewrap">
          <table className="pv-table pv-table--months">
            <caption className="pv-sr">Monthly plan against won gross profit</caption>
            <thead>
              <tr>
                <th scope="col">Month</th>
                <th scope="col" className="pv-num">
                  Plan GP
                </th>
                <th scope="col" className="pv-num">
                  Won GP
                </th>
                <th scope="col" className="pv-num">
                  Closed + commit
                </th>
                <th scope="col" className="pv-num">
                  Attainment
                </th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {months.map((m, i) => {
                const label = monthName(text(m, "month"));
                const annotation = text(m, "annotation");
                return (
                  <tr key={label === "—" ? i : label} className={flag(m, "isFuture") ? "pv-row--future" : undefined}>
                    <th scope="row">
                      {label}
                      {annotation ? (
                        <span className="pv-table__sub pv-annot">{annotation}</span>
                      ) : null}
                    </th>
                    <td className="pv-num">{cash(number(m, "budgetGp"))}</td>
                    <td className="pv-num">{cash(number(m, "wonGp"))}</td>
                    <td className="pv-num">{cash(number(m, "closedCommitGp"))}</td>
                    <td className="pv-num">{percent(number(m, "attainmentPct"))}</td>
                    <td>
                      <Tag tone={ragTone(text(m, "rag"))}>{text(m, "rag") ?? "—"}</Tag>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Block>
  );
}

/** Shared by Performance and the TL;DR, which asks the same question shorter. */
function QuartersTable({ quarters }: { quarters: Row[] }) {
  if (quarters.length === 0) return <Empty>No quarterly plan in this slice.</Empty>;
  return (
    <div className="pv-tablewrap">
      <table className="pv-table pv-table--quarters">
        <caption className="pv-sr">
          Quarterly plan, won, open pipeline, coverage and attainment
        </caption>
        <thead>
          <tr>
            <th scope="col">Quarter</th>
            <th scope="col" className="pv-num">
              Plan GP
            </th>
            <th scope="col" className="pv-num">
              Won GP
            </th>
            <th scope="col" className="pv-num">
              Open GP
            </th>
            <th scope="col" className="pv-num">
              Remaining
            </th>
            <th scope="col" className="pv-num">
              Coverage
            </th>
            <th scope="col" className="pv-num">
              Attainment
            </th>
            <th scope="col">Total pipeline</th>
            <th scope="col">Qualified</th>
          </tr>
        </thead>
        <tbody>
          {quarters.map((q, i) => {
            const label = text(q, "quarter") ?? "—";
            const cov = number(q, "coverage");
            const cls = [
              flag(q, "isCurrent") ? "pv-row--current" : "",
              flag(q, "isFuture") ? "pv-row--future" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <tr key={label === "—" ? i : label} className={cls || undefined}>
                <th scope="row">
                  {label}
                  {flag(q, "isCurrent") ? (
                    <Tag tone="accent" className="pv-now">
                      now
                    </Tag>
                  ) : null}
                </th>
                <td className="pv-num">{cash(number(q, "budgetGp"))}</td>
                <td className="pv-num">{cash(number(q, "wonGp"))}</td>
                <td className="pv-num">{cash(number(q, "openGp"))}</td>
                <td className="pv-num">{cash(number(q, "remainingGp"))}</td>
                <td className="pv-num">{cov === null ? "—" : multiple(cov)}</td>
                <td className="pv-num">{percent(number(q, "attainmentPct"))}</td>
                <td>
                  <Tag tone={ragTone(text(q, "totalPipelineRag"))}>
                    {text(q, "totalPipelineRag") ?? "—"}
                  </Tag>
                </td>
                <td>
                  <Tag tone={ragTone(text(q, "qualifiedPipelineRag"))}>
                    {text(q, "qualifiedPipelineRag") ?? "—"}
                  </Tag>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------------------------------------------------------- structure */

function StructureBlock({ extras }: { extras: Row }) {
  const c = rowAt(extras, "concentration");
  if (c === null) return null;
  const accounts = rowsAt(c, "accounts");
  const industries = rowsAt(c, "industries");
  const beyond = number(c, "accountsBeyondTop");

  return (
    <Block
      title="Where the book is concentrated"
      lead="Share is of gross profit, which is the basis the plan is set in."
    >
      <dl className="pv-figs">
        <div className="pv-figs__fig">
          <dt>Top account</dt>
          <dd>{percent(number(c, "topAccountShare"), 1)}</dd>
        </div>
        <div className="pv-figs__fig">
          <dt>Top five accounts</dt>
          <dd>{percent(number(c, "top5AccountShare"), 1)}</dd>
        </div>
        <div className="pv-figs__fig">
          <dt>Top industry</dt>
          <dd>{percent(number(c, "topIndustryShare"), 1)}</dd>
        </div>
        <div className="pv-figs__fig">
          <dt>
            Account HHI<span className="pv-figs__hint">10,000 = one account</span>
          </dt>
          <dd>{count(number(c, "accountHhi"))}</dd>
        </div>
        <div className="pv-figs__fig">
          <dt>
            Industry HHI<span className="pv-figs__hint">10,000 = one industry</span>
          </dt>
          <dd>{count(number(c, "industryHhi"))}</dd>
        </div>
      </dl>

      <div className="pv-split">
        <div className="pv-split__col">
          <h3 className="pv-subhead">Top accounts</h3>
          {accounts.length === 0 ? (
            <Empty>No accounts in this slice.</Empty>
          ) : (
            <div className="pv-tablewrap">
              <table className="pv-table">
                <caption className="pv-sr">Top accounts by share of gross profit</caption>
                <thead>
                  <tr>
                    <th scope="col">Account</th>
                    <th scope="col" className="pv-num">
                      Opportunities
                    </th>
                    <th scope="col" className="pv-num">
                      ACV GP
                    </th>
                    <th scope="col" className="pv-num">
                      Share
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {accounts.map((a, i) => (
                    <tr key={text(a, "account_code") ?? i}>
                      <th scope="row">{text(a, "account_name") ?? "—"}</th>
                      <td className="pv-num">{count(number(a, "opportunities"))}</td>
                      <td className="pv-num">{cash(number(a, "gp"))}</td>
                      <td className="pv-num">{percent(number(a, "share"), 1)}</td>
                    </tr>
                  ))}
                  {beyond !== null && beyond > 0 ? (
                    <tr className="pv-row--tail">
                      <th scope="row">
                        The other {count(beyond)} accounts
                      </th>
                      <td className="pv-num">—</td>
                      <td className="pv-num">{cash(number(c, "accountsBeyondTopGp"))}</td>
                      <td className="pv-num">—</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="pv-split__col">
          <h3 className="pv-subhead">Industries</h3>
          {industries.length === 0 ? (
            <Empty>No industries in this slice.</Empty>
          ) : (
            <div className="pv-tablewrap">
              <table className="pv-table">
                <caption className="pv-sr">Industries by share of gross profit</caption>
                <thead>
                  <tr>
                    <th scope="col">Industry</th>
                    <th scope="col" className="pv-num">
                      Accounts
                    </th>
                    <th scope="col" className="pv-num">
                      ACV GP
                    </th>
                    <th scope="col" className="pv-num">
                      Share
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {industries.map((r, i) => (
                    <tr key={text(r, "industry") ?? i}>
                      <th scope="row">{text(r, "industry") ?? "—"}</th>
                      <td className="pv-num">{count(number(r, "accounts"))}</td>
                      <td className="pv-num">{cash(number(r, "gp"))}</td>
                      <td className="pv-num">{percent(number(r, "share"), 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {text(c, "note") ? <p className="pv-note">{text(c, "note")}</p> : null}
    </Block>
  );
}

/* --------------------------------------------------------------------- risks */

/**
 * Findings, as a referenceable and filterable list.
 *
 * The list itself lives in components/Findings.tsx — the filter strip, the
 * numbered table, the open row with its evidence and actions. This wrapper
 * keeps the block's title and lead where every other block keeps theirs, and
 * hands the model card in as the foot of the block so it still reads, as it
 * always has, under the findings it qualifies.
 */
function RisksBlock({ extras }: { extras: Row }) {
  return (
    <Block
      title="What the detectors found"
      lead="Two detectors run over the same book. Where they agree independently, the finding says so."
    >
      <Findings extras={extras}>
        <ModelCardBlock extras={extras} />
      </Findings>
    </Block>
  );
}

/* ---------------------------------------------------------------- model card */

function modelCardOf(extras: Row): ModelCard | null {
  const mc = extras["modelCard"];
  if (!isRow(mc)) return null;
  if (!isRow(mc["riskModel"]) || !isRow(mc["primary"])) return null;
  return mc as unknown as ModelCard;
}

/**
 * The honest read, printed rather than filed.
 *
 * A platform that states its own AUC — and states that the AUC is weak — is
 * more credible than one that does not, and it is the reason the product acts
 * on deterministic risk rather than on a learned probability.
 */
function ModelCardBlock({ extras, compact }: { extras: Row; compact?: boolean }) {
  const card = modelCardOf(extras);
  if (card === null) return null;
  const primary = card.primary as Row;
  const independent = card.independent as Row;
  const honest = text(primary, "honestRead");
  const risk = card.riskModel;

  return (
    <section className={`pv-model${compact ? " pv-model--compact" : ""}`}>
      <h3 className="pv-model__title">What the models can and cannot do</h3>
      {honest ? <p className="pv-model__honest">{honest}</p> : null}
      {!primary.available ? (
        <Tag tone="warn" className="pv-model__flag">
          the data-science closure model is not loaded
        </Tag>
      ) : null}

      {compact ? null : (
        <>
          <div className="pv-model__risk">
            <p className="pv-model__risklead">
              <strong>{risk.name}</strong> — {risk.kind}
            </p>
            <p className="pv-model__why">{risk.why}</p>
            <ul className="pv-model__factors">
              {risk.factors.map((f) => (
                <li className="pv-model__factor" key={f.key}>
                  <span className="pv-model__flabel">{f.label}</span>
                  <span className="pv-model__fpts">up to {count(f.maxPoints)} pts</span>
                </li>
              ))}
            </ul>
            <ul className="pv-model__bands">
              {risk.bands.map((b) => (
                <li key={b.band}>
                  <Tag tone={bandTone(b.band)}>{b.band}</Tag>
                  <span className="pv-model__range">
                    {count(b.from)}–{count(b.to)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
          {text(independent, "target") ? (
            <p className="pv-model__indep">
              Reproduced independently on this layer's own subset —{" "}
              {text(independent, "target")}.
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}

/* ---------------------------------------------------------------------- tldr */

function TldrBlock({ extras }: { extras: Row }) {
  const quarters = rowsAt(extras, "quarters");
  return (
    <>
      <UseCasesBlock useCases={useCasesOf(extras)} />
      <Block
        title="The quarter, and what to believe about it"
        lead="Plan against actual first, then an honest account of how much the models behind this page can be trusted."
      >
        <QuartersTable quarters={quarters} />
        <ModelCardBlock extras={extras} compact />
      </Block>
    </>
  );
}

/* ----------------------------------------------------------- use cases */

const USE_CASE_KEYS = new Set<string>(["opportunities", "anomalies", "closure"]);
const LENSES = new Set<string>([
  "my-day", "my-deals", "my-accounts", "my-record",
  "pod-pulse", "rep-benchmark", "process", "calibration", "pod-whitespace",
  "tldr", "performance", "structure", "risks", "growth", "actions",
]);

/**
 * Narrowed field by field like every other reader in this file: a tile the
 * server did not fully describe is left out rather than drawn with a blank in
 * it, and a page the shell cannot open is not offered as a button.
 */
function useCasesOf(extras: Row): UseCase[] {
  return rowsAt(extras, "useCases").flatMap((r) => {
    const key = text(r, "key");
    const title = text(r, "title");
    const oneLine = text(r, "oneLine");
    const page = text(r, "page");
    const cta = text(r, "cta");
    const figures = rowsAt(r, "figures").flatMap((f) => {
      const label = text(f, "label");
      const formatted = text(f, "formatted");
      return label && formatted ? [{ label, formatted }] : [];
    });
    if (!key || !USE_CASE_KEYS.has(key) || !title || !oneLine || !page || !LENSES.has(page) || !cta) {
      return [];
    }
    return [{ key: key as UseCase["key"], title, oneLine, figures, page: page as Lens, cta }];
  });
}

/**
 * The three use cases from the 16 Sep call, stated on the landing page.
 *
 * An executive should not need to know which page proves which. Each tile
 * names the capability, gives it the figures the dedicated page draws from,
 * and opens that page — the figures arrive formatted, so this block computes
 * nothing, and the button dispatches the same page change the sidebar does.
 */
function UseCasesBlock({ useCases }: { useCases: UseCase[] }) {
  const { setPage } = useApp();
  if (useCases.length === 0) return null;
  return (
    <section className="pv-usecases" aria-labelledby="pv-usecases-title">
      <h2 className="pv-block__title" id="pv-usecases-title">
        The three things this platform watches
      </h2>
      <ol className="pv-usecases__row">
        {useCases.map((u, i) => (
          <li className={`pv-usecase pv-usecase--${u.key}`} key={u.key}>
            <p className="pv-usecase__n" aria-hidden="true">
              {i + 1}
            </p>
            <h3 className="pv-usecase__title">{u.title}</h3>
            <p className="pv-usecase__line">{u.oneLine}</p>
            <dl className="pv-usecase__figures">
              {u.figures.map((f) => (
                <div className="pv-usecase__figure" key={f.label}>
                  <dt>{f.label}</dt>
                  <dd>{f.formatted}</dd>
                </div>
              ))}
            </dl>
            <button
              type="button"
              className="pv-usecase__go"
              onClick={() => setPage(u.page)}
              aria-label={`${u.cta} — ${u.title}`}
            >
              {u.cta}
              <span aria-hidden="true"> →</span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
