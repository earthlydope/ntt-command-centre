/**
 * DealDrawer — the per-deal drill-down, as a right-hand drawer.
 *
 * Four blocks, in the order someone working the deal actually reads them:
 *
 *   HEADER    what the deal is, who owns it, what it is worth, when it lands.
 *   RISK      the score and band, then EVERY factor that fired, each with the
 *             points it contributed and the sentence that explains it. This is
 *             the visual centre of the drawer: "which variable is driving the
 *             risk" is the question the customer asked, and a score with no
 *             decomposition is a number nobody can act on.
 *   MODEL     pWin beside the rep's own confidence and the gap between them,
 *             labelled a RANKING SIGNAL rather than a forecast, then the
 *             benchmark table — with the near-noise features visibly
 *             de-emphasised instead of dressed up as verdicts.
 *   TIMELINE  the field-level change log, newest first, with the silences drawn
 *             to scale. On a stalled deal the gap IS the finding, and a list of
 *             evenly spaced rows hides exactly the thing the drawer was opened
 *             to show.
 *
 * The drawer holds no business arithmetic. Every figure is a field of
 * `DealDetail` formatted through `lib/format`; the only quantities computed
 * here are the day counts between two logged changes and the widths of the
 * bars, which are geometry over values the server already sent.
 *
 * It is a modal dialog: focus is trapped inside it, Escape closes it, a click
 * on the backdrop closes it, and focus returns to whatever opened it.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { api } from "../api/client";
import type { DealDetail, RiskBand } from "../api/types";
import { describe, type Async } from "../lenses/useView";
import { longDate, money, num, pct } from "../lib/format";
import { useApp } from "../state/AppStateProvider";

/* -------------------------------------------------------------- constants */

/** Below this, a gap between two logged changes is ordinary working rhythm and
 *  drawing a band for it would be noise. It is a RENDERING threshold — the
 *  business stall threshold lives in `meta.stallThreshold` and arrives as a
 *  prop when the caller has it. */
const GAP_BAND_DAYS = 21;

/** A silence is drawn `days / GAP_DAYS_PER_UNIT` spacing units tall, clamped,
 *  so a six-month hole reads as a hole and a one-year hole does not push the
 *  rest of the log off the screen. */
const GAP_DAYS_PER_UNIT = 30;
const GAP_UNITS_MIN = 1;
const GAP_UNITS_MAX = 6;

const BAND_MOD: Record<RiskBand, string> = {
  Low: "low",
  Watch: "watch",
  High: "high",
  Critical: "critical",
};

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/* ---------------------------------------------------------------- helpers */

const ms = (iso: string | null | undefined): number | null => {
  if (!iso) return null;
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? null : t;
};

/** Whole days between two ISO dates, or null if either will not parse. */
function daysBetween(from: string | null | undefined, to: string | null | undefined): number | null {
  const a = ms(from);
  const b = ms(to);
  if (a === null || b === null) return null;
  return Math.round((b - a) / 86_400_000);
}

/** A date the payload may not carry, rendered without pretending. */
const when = (iso: string | null | undefined): string => (iso ? longDate(iso) : "—");

/** The server sends probabilities either as a fraction or as a percentage
 *  depending on the feature; `|v| <= 1` is the only honest discriminator, and
 *  either reading formats through the one `pct` in lib/format. */
const asPct = (v: number): string => pct(Math.abs(v) <= 1 ? v * 100 : v);

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

const cssVar = (name: string, value: string): CSSProperties =>
  ({ [name]: value }) as CSSProperties;

/* ------------------------------------------------------------ small parts */

function Label({ children }: { children: ReactNode }) {
  return <div className="dd__label">{children}</div>;
}

function Fact({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={wide ? "dd__fact dd__fact--wide" : "dd__fact"}>
      <dt className="dd__fact-label">{label}</dt>
      <dd className="dd__fact-value">{value}</dd>
    </div>
  );
}

function Stat({
  label,
  value,
  note,
  tag,
}: {
  label: string;
  value: string;
  note?: string;
  tag?: string;
}) {
  return (
    <div className="dd__stat">
      <div className="dd__stat-label">
        {label}
        {tag ? <span className="dd__tag">{tag}</span> : null}
      </div>
      <div className="dd__stat-value">{value}</div>
      {note ? <div className="dd__stat-note">{note}</div> : null}
    </div>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="dd__empty">{children}</p>;
}

/* ------------------------------------------------------------------ props */

export interface DealDrawerProps {
  /** The deal, when the caller already has it. Omit it and pass `code` (or
   *  leave both off and let the drawer read `state.drawer`) and the drawer
   *  fetches it itself. */
  deal?: DealDetail | null;
  /** Opportunity code, when the caller is fetching or when the drawer must. */
  code?: string | null;
  /** The pinned business date off the payload. Never `Date.now()`. */
  asOf?: string | null;
  /** `meta.stallThreshold`, used only to mark a silence as a stall. */
  stallThreshold?: number | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onClose?: () => void;
}

/* ------------------------------------------------------------------- main */

export function DealDrawer({
  deal,
  code,
  asOf,
  stallThreshold,
  loading,
  error,
  onRetry,
  onClose,
}: DealDrawerProps) {
  const { ctx, state, closeDrawer } = useApp();

  // A drawer key of "deal:<code>" is the canonical opener; a caller that
  // already holds the payload can hand it straight over instead.
  const fromState = state.drawer?.startsWith("deal:") ? state.drawer.slice(5) : null;
  const dealCode = deal?.opportunityCode ?? code ?? fromState ?? null;

  const [attempt, setAttempt] = useState(0);
  const [own, setOwn] = useState<Async<DealDetail>>({
    data: null,
    error: null,
    loading: !deal && !!dealCode,
  });

  const fetchKey = useMemo(
    () => JSON.stringify([dealCode, ctx.persona, ctx.identity, ctx.measure, attempt]),
    [dealCode, ctx.persona, ctx.identity, ctx.measure, attempt],
  );
  const lastKey = useRef("");

  useEffect(() => {
    if (deal || !dealCode) return;
    const ac = new AbortController();
    lastKey.current = fetchKey;
    setOwn((p) => ({ ...p, loading: true, error: null }));
    api
      .deal(ctx, dealCode, ac.signal)
      // The key check is the second guard: an abort is not instantaneous and a
      // late response for another deal must never paint into this one.
      .then((d) => lastKey.current === fetchKey && setOwn({ data: d, error: null, loading: false }))
      .catch((e: unknown) => {
        if (ac.signal.aborted || lastKey.current !== fetchKey) return;
        setOwn({ data: null, error: describe(e), loading: false });
      });
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchKey]);

  const detail = deal ?? own.data;
  const isLoading = loading ?? (deal ? false : own.loading);
  const failure = error ?? (deal ? null : own.error);

  const close = useCallback(() => {
    if (onClose) onClose();
    else closeDrawer();
  }, [onClose, closeDrawer]);

  const retry = useCallback(() => {
    if (onRetry) onRetry();
    else setAttempt((n) => n + 1);
  }, [onRetry]);

  /* --- modality: Escape, focus trap, focus restore, background scroll ---- */
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const node = panelRef.current;
    node?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close();
        return;
      }
      if (e.key !== "Tab" || !node) return;
      const stops = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
        (el) => el.getClientRects().length > 0,
      );
      if (stops.length === 0) {
        e.preventDefault();
        node.focus();
        return;
      }
      const first = stops[0];
      const last = stops[stops.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || active === node)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKey, true);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = prevOverflow;
      opener?.focus?.();
    };
  }, [close]);

  const titleId = `dd-title-${dealCode ?? "deal"}`;

  return (
    <>
      <div className="dd-scrim" onClick={close} aria-hidden="true" />
      <aside
        className="dd"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-busy={isLoading || undefined}
        tabIndex={-1}
        ref={panelRef}
      >
        <header className="dd__head">
          <div className="dd__head-main">
            <div className="dd__eyebrow">
              <span className="dd__code">{dealCode ?? "Deal"}</span>
              {detail ? <span className="dd__eyebrow-sep">·</span> : null}
              {detail ? <span>{detail.account}</span> : null}
            </div>
            <h2 className="dd__title" id={titleId}>
              {detail ? detail.name : "Deal detail"}
            </h2>
            {detail ? (
              <p className="dd__sub">
                {detail.owner} · {detail.stage}
              </p>
            ) : null}
          </div>
          <button type="button" className="dd__close" onClick={close} aria-label="Close deal detail">
            ✕
          </button>
        </header>

        <p className="dd__status" role="status" aria-live="polite">
          {isLoading
            ? `Loading deal ${dealCode ?? ""} from the semantic layer…`
            : failure
              ? `Deal ${dealCode ?? ""} could not be loaded.`
              : ""}
        </p>

        <div className="dd__body">
          {failure ? (
            <div className="dd__error">
              <b>The semantic layer did not return this deal.</b>
              <p className="dd__error-detail">
                <code>GET /api/deal/{dealCode ?? "—"}</code> failed: {failure}
              </p>
              <p className="dd__error-detail">
                Nothing on this panel is computed in the browser, so there is no cached copy to show
                instead.
              </p>
              <button type="button" className="dd__retry" onClick={retry}>
                Try again
              </button>
            </div>
          ) : isLoading && !detail ? (
            <DealSkeleton />
          ) : detail ? (
            <>
              <DealFacts deal={detail} />
              <RiskBlock deal={detail} />
              <ModelBlock deal={detail} />
              <TimelineBlock deal={detail} asOf={asOf ?? null} stallThreshold={stallThreshold ?? null} />
              <p className="dd__foot">
                Risk factors, benchmarks and this change log are computed by the semantic layer from
                the opportunity's own rows. No sentence on this panel was generated by a language
                model, which is why none of them carries a lens label.
              </p>
            </>
          ) : (
            <Empty>No deal is selected.</Empty>
          )}
        </div>
      </aside>
    </>
  );
}

/* ------------------------------------------------------------ 1. H E A D E R */

function DealFacts({ deal }: { deal: DealDetail }) {
  return (
    <section className="dd__section dd__section--facts" aria-label="Deal facts">
      <dl className="dd__facts">
        <Fact label="ACV GP" value={money(deal.acvGp)} />
        <Fact label="ACV revenue" value={money(deal.acvRevenue)} />
        <Fact label="Close date" value={when(deal.closeDate)} />
        <Fact label="Stage" value={deal.stage} />
        <Fact label="LOB" value={deal.lob} />
        <Fact label="Portfolio" value={deal.portfolio} />
        {deal.stagePath ? <Fact label="Stage path" value={deal.stagePath} wide /> : null}
      </dl>
    </section>
  );
}

/* -------------------------------------------------------------- 2. R I S K */

function RiskBlock({ deal }: { deal: DealDetail }) {
  const factors = deal.riskFactors ?? [];
  const topPoints = factors.reduce((m, f) => Math.max(m, f.points), 0);
  const total = factors.reduce((s, f) => s + f.points, 0);

  return (
    <section className="dd__section dd__section--risk" aria-labelledby="dd-risk-h">
      <div className="dd__section-head">
        <h3 className="dd__section-title" id="dd-risk-h">
          What is driving the risk
        </h3>
        <span className="dd__section-note">
          {factors.length === 0
            ? "no factor fired"
            : `${num(factors.length)} of the model's factors fired`}
        </span>
      </div>

      <div className="dd__risk">
        <div className="dd__risk-score">
          <div className="dd__risk-number">{num(deal.riskScore)}</div>
          <div className={`dd__band dd__band--${BAND_MOD[deal.riskBand]}`}>{deal.riskBand}</div>
        </div>
        <div className="dd__risk-meta">
          <Stat label="Value at risk" value={money(deal.valueAtRisk)} />
          <Stat
            label="Quiet"
            value={deal.quietDays === null ? "—" : `${num(deal.quietDays)} days`}
            note={deal.quietDays === null ? "no change log for this deal" : "since the last logged change"}
          />
          {total > 0 ? (
            <Stat
              label="Points on the board"
              value={num(total)}
              note="summed across the factors below"
            />
          ) : null}
        </div>
      </div>

      {factors.length === 0 ? (
        <Empty>
          No risk factor fired on this opportunity. The score is what the model's rules leave when
          nothing trips them, not an absence of evidence.
        </Empty>
      ) : (
        <ul className="dd__factors">
          {factors.map((f) => (
            <li className="dd__factor" key={f.key}>
              <div className="dd__factor-head">
                <span className="dd__factor-label">{f.label}</span>
                <span className="dd__factor-points">+{num(f.points)} pts</span>
              </div>
              <div className="dd__factor-bar" aria-hidden="true">
                <span
                  className="dd__factor-fill"
                  style={cssVar(
                    "--dd-factor-share",
                    String(topPoints > 0 ? clamp(f.points / topPoints, 0, 1) : 0),
                  )}
                />
              </div>
              <p className="dd__factor-detail">{f.detail}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/* ------------------------------------------------------------ 3. M O D E L */

function ModelBlock({ deal }: { deal: DealDetail }) {
  const benchmarks = deal.benchmarks ?? [];
  const hasModel =
    deal.pWin !== undefined ||
    deal.repConfidence !== undefined ||
    deal.confidenceGap !== undefined;

  const lean =
    deal.pWin !== undefined && deal.repConfidence !== undefined
      ? deal.repConfidence > deal.pWin
        ? "the rep is more confident than the model's ordering"
        : deal.repConfidence < deal.pWin
          ? "the model ranks this deal above the rep's own confidence"
          : "the rep and the model agree"
      : undefined;

  return (
    <section className="dd__section dd__section--model" aria-labelledby="dd-model-h">
      <div className="dd__section-head">
        <h3 className="dd__section-title" id="dd-model-h">
          Model read
        </h3>
        <span className="dd__badge">RANKING SIGNAL — NOT A FORECAST</span>
      </div>

      <p className="dd__caveat">
        pWin orders deals against each other; it is not a probability to quote to anyone. The test
        AUC is 0.595 against 0.500 for chance — better than a coin flip at sorting a list, far short
        of a forecast for a single deal.
      </p>

      {hasModel ? (
        <div className="dd__model-stats">
          {deal.pWin !== undefined ? (
            <Stat
              label="pWin"
              value={asPct(deal.pWin)}
              tag="rank"
              note={
                deal.pWinSegment !== undefined && deal.pWinSegment !== null
                  ? `segment base rate ${asPct(deal.pWinSegment)}`
                  : undefined
              }
            />
          ) : null}
          {deal.repConfidence !== undefined ? (
            <Stat label="Rep confidence" value={asPct(deal.repConfidence)} note="as logged in SFDC" />
          ) : null}
          {deal.confidenceGap !== undefined ? (
            <Stat label="Confidence gap" value={asPct(deal.confidenceGap)} note={lean} />
          ) : null}
          {deal.expectedGp !== undefined ? (
            <Stat label="Expected GP" value={money(deal.expectedGp)} note="ACV GP weighted by pWin" />
          ) : null}
        </div>
      ) : (
        <Empty>No model read is available for this opportunity.</Empty>
      )}

      {benchmarks.length === 0 ? (
        <Empty>No feature benchmarks were returned for this opportunity.</Empty>
      ) : (
        <div className="dd__bench">
          <Label>How this deal sits against the benchmark</Label>
          <table className="dd__bench-table">
            <thead>
              <tr>
                <th scope="col">Feature</th>
                <th scope="col" className="dd__bench-num">
                  This deal
                </th>
                <th scope="col" className="dd__bench-num">
                  Benchmark
                </th>
                <th scope="col">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((b) => {
                const method = [b.benchmarkMethod, b.direction, `r = ${num(b.correlationWithWin)}`]
                  .filter(Boolean)
                  .join(" · ");
                return (
                  <tr
                    className={b.weakSignal ? "dd__bench-row dd__bench-row--weak" : "dd__bench-row"}
                    key={b.feature}
                  >
                    <th scope="row" className="dd__bench-feature">
                      {b.label}
                      {method ? <span className="dd__bench-method">{method}</span> : null}
                    </th>
                    <td className="dd__bench-num">{num(b.value)}</td>
                    <td className="dd__bench-num">{num(b.benchmark)}</td>
                    <td>
                      <span
                        className={`dd__verdict dd__verdict--${b.verdict.toLowerCase()}${
                          b.weakSignal ? " dd__verdict--weak" : ""
                        }`}
                      >
                        {b.verdict}
                      </span>
                      {b.weakSignal ? <span className="dd__tag dd__tag--weak">near noise</span> : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {benchmarks.some((b) => b.weakSignal && b.note) ? (
            <ul className="dd__weak-notes">
              {benchmarks
                .filter((b) => b.weakSignal && b.note)
                .map((b) => (
                  <li className="dd__weak-note" key={`${b.feature}-note`}>
                    <b>{b.label}</b> {b.note}
                  </li>
                ))}
            </ul>
          ) : null}
          <p className="dd__bench-caveat">
            A muted verdict is a verdict the workbook says not to lean on: its correlation with
            winning is close to zero, so the colour would read as evidence it has not got. It is
            printed rather than hidden, because a table that quietly drops its weak rows is a table
            you cannot audit.
          </p>
        </div>
      )}
    </section>
  );
}

/* ------------------------------------------------------ 4. T I M E L I N E */

function TimelineBlock({
  deal,
  asOf,
  stallThreshold,
}: {
  deal: DealDetail;
  asOf: string | null;
  stallThreshold: number | null;
}) {
  const events = useMemo(
    () =>
      [...(deal.timeline ?? [])].sort((a, b) => {
        const ta = ms(a.date);
        const tb = ms(b.date);
        return (tb ?? 0) - (ta ?? 0);
      }),
    [deal],
  );

  const newest = events[0];
  const sinceLast = deal.quietDays ?? (newest ? daysBetween(newest.date, asOf) : null);
  const stalled = stallThreshold !== null && sinceLast !== null && sinceLast >= stallThreshold;

  return (
    <section className="dd__section dd__section--timeline" aria-labelledby="dd-tl-h">
      <div className="dd__section-head">
        <h3 className="dd__section-title" id="dd-tl-h">
          Change log
        </h3>
        <span className="dd__section-note">
          {events.length === 0
            ? "nothing logged"
            : `${num(events.length)} field changes, newest first`}
        </span>
      </div>

      {events.length === 0 ? (
        <Empty>
          No field-level change has ever been logged against this opportunity. That silence is the
          whole record, not a gap in this panel.
        </Empty>
      ) : (
        <ol className="dd__timeline">
          {sinceLast !== null && sinceLast >= GAP_BAND_DAYS ? (
            <li
              className={stalled ? "dd__quiet dd__quiet--stalled" : "dd__quiet"}
              style={cssVar("--dd-gap-units", gapUnits(sinceLast))}
            >
              <span className="dd__gap-label">
                {num(sinceLast)} days quiet{asOf ? ` to ${when(asOf)}` : ""}
                {stalled ? ` — past the ${num(stallThreshold as number)}-day stall line` : ""}
              </span>
            </li>
          ) : null}

          {events.map((e, i) => {
            const next = events[i + 1];
            const gap = next ? daysBetween(next.date, e.date) : null;
            return (
              <li className="dd__event" key={`${e.date}-${e.field}-${i}`}>
                <div className="dd__event-row">
                  <span className="dd__event-dot" aria-hidden="true" />
                  <div className="dd__event-main">
                    <div className="dd__event-field">{e.field}</div>
                    <div className="dd__event-change">
                      <span className="dd__from">{e.from || "—"}</span>
                      <span className="dd__arrow" aria-label="changed to">
                        →
                      </span>
                      <span className="dd__to">{e.to || "—"}</span>
                    </div>
                    <div className="dd__event-meta">
                      {when(e.date)} · {e.by}
                    </div>
                  </div>
                </div>
                {gap !== null && gap >= GAP_BAND_DAYS ? (
                  <div className="dd__gap" style={cssVar("--dd-gap-units", gapUnits(gap))}>
                    <span className="dd__gap-label">{num(gap)} days of silence</span>
                  </div>
                ) : null}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

const gapUnits = (days: number): string =>
  clamp(days / GAP_DAYS_PER_UNIT, GAP_UNITS_MIN, GAP_UNITS_MAX).toFixed(2);

/* -------------------------------------------------------- loading skeleton */

/** The skeleton is the drawer's own layout with the text removed, so nothing
 *  jumps when the payload lands: facts grid, risk block with three factor
 *  rows, the model stats, three timeline events. */
function DealSkeleton() {
  return (
    <div className="dd__skeleton" aria-hidden="true">
      <section className="dd__section">
        <div className="dd__facts">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div className="dd__fact" key={i}>
              <span className="dd-skel dd-skel--sm" />
              <span className="dd-skel dd-skel--md" />
            </div>
          ))}
        </div>
      </section>
      <section className="dd__section dd__section--risk">
        <span className="dd-skel dd-skel--sm" />
        <div className="dd__risk">
          <div className="dd__risk-score">
            <span className="dd-skel dd-skel--score" />
          </div>
          <div className="dd__risk-meta">
            <span className="dd-skel dd-skel--md" />
            <span className="dd-skel dd-skel--md" />
          </div>
        </div>
        <ul className="dd__factors">
          {[0, 1, 2].map((i) => (
            <li className="dd__factor" key={i}>
              <span className="dd-skel dd-skel--md" />
              <span className="dd-skel dd-skel--bar" />
              <span className="dd-skel dd-skel--lg" />
            </li>
          ))}
        </ul>
      </section>
      <section className="dd__section">
        <span className="dd-skel dd-skel--sm" />
        <div className="dd__model-stats">
          {[0, 1, 2].map((i) => (
            <span className="dd-skel dd-skel--md" key={i} />
          ))}
        </div>
        <span className="dd-skel dd-skel--block" />
      </section>
      <section className="dd__section">
        <span className="dd-skel dd-skel--sm" />
        {[0, 1, 2].map((i) => (
          <div className="dd__event" key={i}>
            <span className="dd-skel dd-skel--md" />
            <span className="dd-skel dd-skel--lg" />
          </div>
        ))}
      </section>
    </div>
  );
}
