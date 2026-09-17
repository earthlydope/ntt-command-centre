/**
 * What the detectors found — as a referenceable, filterable list with actions.
 *
 * WHAT WAS WRONG. The block used to print every finding as a full card —
 * heading, three tags, a three-row definition list and a footer — stacked
 * under a category heading. Sixty of those on one page is a wall: the
 * customer's words were "too cluttered", "not very readable", and "we need to
 * make the rows referenceable, put them in a sequence with some sort of
 * filter to group them and take actions".
 *
 * WHAT IT DOES NOW. One row per finding, numbered in the order shown, with the
 * seven things a reader scans for in seven columns. Above the table a filter
 * strip narrows the list by category, priority and source and re-orders it;
 * the strip stays put while the table scrolls so the controls are never out of
 * reach. A row opens in place into the evidence, the recommended next step and
 * the three things a person can do about it — open the deal, scope the page to
 * this entity, or ask the model why it was flagged. One row is open at a time
 * and Escape closes it.
 *
 * REFERENCEABLE means two things. The ordinal ("row 7") is the number a person
 * reads out in a meeting and it follows the current sort and filter, because
 * that is the list the room is looking at. The detector's own id (ANM-00358)
 * is the stable one and it is printed in the open row for the record.
 *
 * NOTHING HERE COMPUTES A BUSINESS FIGURE. Every value at stake, severity,
 * priority and provenance arrives on the payload; the only counting is of rows
 * in the list the server already sent, which is what a filter has to say to be
 * honest about what it will show. Category stakes come from the server's own
 * summary, not from adding rows up.
 *
 * FILTERING HERE IS A VIEW OF ONE LIST, not a page filter. Nothing on the page
 * outside this table depends on it, which is why it is allowed to live in
 * component state rather than in the URL. "Filter the page to this" is the
 * real thing and goes through the reducer like every other filter.
 */
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
} from "react";
import type { DimKey, Tone } from "../api/types";
import { money, num } from "../lib/format";
import { useApp } from "../state/AppStateProvider";
import { Glyph } from "./askGlyphs";

/* ========================================================================== *
 * Reading the payload
 *
 * `extras` is `Record<string, unknown>` on the wire. These readers narrow it
 * without a cast: a field the server did not send reads as null and renders as
 * an em dash. Nothing substitutes a zero for a missing value.
 * ========================================================================== */

type Row = Record<string, unknown>;

const isRow = (v: unknown): v is Row => typeof v === "object" && v !== null && !Array.isArray(v);

function rowsAt(r: Row, key: string): Row[] {
  const v = r[key];
  return Array.isArray(v) ? v.filter(isRow) : [];
}

function rowAt(r: Row, key: string): Row | null {
  const v = r[key];
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

const flag = (r: Row, k: string): boolean => r[k] === true;

const cash = (v: number | null): string => (v === null ? "—" : money(v));
const count = (v: number): string => num(Math.round(v));

/* ========================================================================== *
 * The vocabulary
 * ========================================================================== */

type Priority = "Critical" | "High" | "Medium" | "Low";
const PRIORITIES: readonly Priority[] = ["Critical", "High", "Medium", "Low"];
const isPriority = (s: string | null): s is Priority =>
  s !== null && (PRIORITIES as readonly string[]).includes(s);

/** The tone a priority word is painted in, everywhere it appears here. */
function priorityTone(p: Priority | null): Tone {
  switch (p) {
    case "Critical":
      return "danger";
    case "High":
      return "warn";
    case "Medium":
      return "accent";
    case "Low":
      return "neutral";
    default:
      return "neutral";
  }
}

/**
 * Where a finding came from. `both` is the strongest a finding gets — the
 * data-science export and this layer's live detector fired independently on
 * the same entity for the same reason — and it is the one the table lights.
 */
type Source = "ds" | "live" | "both";
const SOURCES: readonly { key: Source; label: string }[] = [
  { key: "ds", label: "DS model" },
  { key: "live", label: "Live detector" },
  { key: "both", label: "Both agree" },
];
const sourceLabel = (s: Source): string => SOURCES.find((x) => x.key === s)?.label ?? s;

function sourceOf(r: Row): Source {
  const prov = text(r, "provenance");
  if (flag(r, "corroborated") || prov === "corroborated") return "both";
  return prov === "live" ? "live" : "ds";
}

type SortKey = "stake" | "severity" | "priority";
const SORTS: readonly { key: SortKey; label: string }[] = [
  { key: "priority", label: "Priority" },
  { key: "stake", label: "Value at stake" },
  { key: "severity", label: "Severity" },
];

/** Rows beyond this are behind "Show all" so the first paint stays a page. */
const PAGE = 60;

/**
 * Which page filter an entity maps onto. The `account` dimension filters on the
 * account NAME (see api/semantic/dimensions.py), so an account finding scopes by
 * its name rather than its 18-character code; a rep or an industry is already
 * keyed by name. An opportunity opens the deal drawer instead; a segment has no
 * single dimension and gets no scope.
 */
function scopeOf(r: Row): { dim: DimKey; value: string } | null {
  const type = text(r, "entity_type");
  const id = text(r, "entity_id");
  const label = text(r, "entity_label");
  switch (type) {
    case "Account": {
      const name = text(r, "account_name") ?? label;
      return name ? { dim: "account", value: name } : null;
    }
    case "Rep": {
      const name = id ?? text(r, "owner") ?? label;
      return name ? { dim: "rep", value: name } : null;
    }
    case "Industry": {
      const name = id ?? label;
      return name ? { dim: "industry", value: name } : null;
    }
    default:
      return null;
  }
}

/** One finding, read once off its row so the render path never re-narrows. */
interface Finding {
  key: string;
  ref: string | null;
  isOpp: boolean;
  entityId: string | null;
  entityType: string | null;
  label: string;
  category: string;
  question: string | null;
  priority: Priority | null;
  source: Source;
  severity: number | null;
  stake: number | null;
  owner: string | null;
  evidence: string | null;
  liveEvidence: string | null;
  action: string | null;
  scope: { dim: DimKey; value: string } | null;
}

function readFinding(r: Row, i: number): Finding {
  const ref = text(r, "anomaly_id");
  const entityId = text(r, "entity_id");
  const p = text(r, "priority");
  return {
    key: ref ?? `row-${i}`,
    ref,
    // Only a finding whose ENTITY is the opportunity opens the drawer: a
    // finding at opportunity grain about an account carries the account code.
    isOpp: text(r, "entity_type") === "Opportunity",
    entityId,
    entityType: text(r, "entity_type"),
    label: text(r, "entity_label") ?? entityId ?? "—",
    category: text(r, "category") ?? "Uncategorised",
    question: text(r, "category_question"),
    priority: isPriority(p) ? p : null,
    source: sourceOf(r),
    severity: number(r, "severity"),
    stake: number(r, "value_at_stake"),
    owner: text(r, "owner"),
    evidence: text(r, "evidence"),
    liveEvidence: text(r, "live_evidence"),
    action: text(r, "recommended_action"),
    scope: scopeOf(r),
  };
}

/* ========================================================================== *
 * Sorting
 *
 * Within a priority the server's order is kept: it is the triage order
 * (severity × money), which is the sequence the detectors themselves
 * recommend. Array.prototype.sort is stable, so ranking by priority alone
 * preserves it. Missing values sort last under every key, so a finding the
 * server could not price never floats above one it could.
 * ========================================================================== */

const rank = (p: Priority | null): number => (p === null ? PRIORITIES.length : PRIORITIES.indexOf(p));

function compare(a: Finding, b: Finding, key: SortKey): number {
  if (key === "priority") return rank(a.priority) - rank(b.priority);
  const av = key === "stake" ? a.stake : a.severity;
  const bv = key === "stake" ? b.stake : b.severity;
  if (av === null && bv === null) return 0;
  if (av === null) return 1;
  if (bv === null) return -1;
  return bv - av;
}

/* ========================================================================== *
 * The component
 * ========================================================================== */

export function Findings({ extras, children }: { extras: Row; children?: ReactNode }) {
  const { openDrawer, openAsk, onFilter, state } = useApp();
  const uid = useId();

  const all = useMemo(() => rowsAt(extras, "findings").map(readFinding), [extras]);
  const summary = rowAt(extras, "anomalySummary");
  const byCategory = useMemo(() => (summary ? rowsAt(summary, "byCategory") : []), [summary]);

  // The selection. Categories are a set (a reader comparing two categories
  // wants both); priority and source are one-of. Sort defaults to priority so
  // the first read is grouped Critical → Low, which is the order to act in.
  const [cats, setCats] = useState<ReadonlySet<string>>(() => new Set());
  const [priority, setPriority] = useState<Priority | "all">("all");
  const [source, setSource] = useState<Source | "all">("all");
  const [sort, setSort] = useState<SortKey>("priority");
  const [showAll, setShowAll] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [closing, setClosing] = useState<string | null>(null);
  // Bumped on every change to the selection: the body is keyed on it so it
  // remounts and plays its entrance, which is how "the list changed" is shown
  // rather than having rows silently swap under the eye.
  const [tick, setTick] = useState(0);

  // Any change to the selection is a new list: close the open row, fold the
  // long tail back, and let the body re-enter.
  const touch = useCallback(() => {
    setExpanded(null);
    setClosing(null);
    setShowAll(false);
    setTick((t) => t + 1);
  }, []);

  // A new payload (a page filter changed, the persona moved) is a new list too.
  useEffect(() => {
    setExpanded(null);
    setClosing(null);
    setShowAll(false);
  }, [all]);

  /* --------------------------------------------------------------- counts */

  const catCount = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of all) m.set(f.category, (m.get(f.category) ?? 0) + 1);
    return m;
  }, [all]);

  const priorityCount = useMemo(() => {
    const m = new Map<Priority, number>();
    for (const f of all) if (f.priority) m.set(f.priority, (m.get(f.priority) ?? 0) + 1);
    return m;
  }, [all]);

  const sourceCount = useMemo(() => {
    const m = new Map<Source, number>();
    for (const f of all) m.set(f.source, (m.get(f.source) ?? 0) + 1);
    return m;
  }, [all]);

  // The chip list is the server's taxonomy in the server's order, plus any
  // category a row carries that the summary did not name, so a finding can
  // never be un-filterable.
  const chips = useMemo(() => {
    const seen = new Set<string>();
    const out: { name: string; stake: number | null; total: number | null }[] = [];
    for (const c of byCategory) {
      const name = text(c, "category");
      if (!name || seen.has(name)) continue;
      seen.add(name);
      out.push({ name, stake: number(c, "valueAtStake"), total: number(c, "count") });
    }
    for (const name of catCount.keys()) {
      if (seen.has(name)) continue;
      seen.add(name);
      out.push({ name, stake: null, total: null });
    }
    return out;
  }, [byCategory, catCount]);

  /* ------------------------------------------------------------ the list */

  const visible = useMemo(() => {
    const kept = all.filter(
      (f) =>
        (cats.size === 0 || cats.has(f.category)) &&
        (priority === "all" || f.priority === priority) &&
        (source === "all" || f.source === source),
    );
    return [...kept].sort((a, b) => compare(a, b, sort));
  }, [all, cats, priority, source, sort]);

  const shown = showAll ? visible : visible.slice(0, PAGE);
  const hidden = visible.length - shown.length;

  /* ------------------------------------------------------------ handlers */

  const toggleCat = (name: string) => {
    setCats((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
    touch();
  };

  const toggles = useRef(new Map<string, HTMLButtonElement>());

  // One row open at a time. The row being replaced is marked closing so it
  // can play its exit while the new one enters.
  const toggleRow = (key: string) => {
    if (expanded === key) {
      setClosing(key);
      setExpanded(null);
      return;
    }
    // Re-opening a row that is still on its way out cancels the exit rather
    // than playing both animations on the same element.
    setClosing(expanded !== null ? expanded : closing === key ? null : closing);
    setExpanded(key);
  };

  // Escape closes the open row and puts focus back on the control that opened
  // it, so a keyboard user lands where they were rather than at the top of
  // the document. Scoped to this block: the Ask panel and the drawer own the
  // key when they are open, and they are outside this subtree.
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Escape" || expanded === null) return;
    e.stopPropagation();
    const key = expanded;
    setClosing(key);
    setExpanded(null);
    toggles.current.get(key)?.focus();
  };

  // A click anywhere on the row opens it, unless the click landed on a control
  // of its own — the entity button, say — which keeps its own meaning.
  const onRowClick = (key: string) => (e: MouseEvent<HTMLTableRowElement>) => {
    const t = e.target as HTMLElement;
    if (t.closest("button, a, select, input")) return;
    toggleRow(key);
  };

  /* -------------------------------------------------------------- render */

  const total = all.length;
  const groupsOn = sort === "priority";

  // Rows with their ordinal, and the group header that precedes a change of
  // priority when the list is sorted by it.
  const body: ReactNode[] = [];
  let lastGroup: string | null = null;
  shown.forEach((f, i) => {
    const groupName = f.priority ?? "Unprioritised";
    if (groupsOn && groupName !== lastGroup) {
      lastGroup = groupName;
      const n = f.priority ? (priorityCount.get(f.priority) ?? 0) : visible.filter((x) => x.priority === null).length;
      const nVisible = visible.filter((x) => (x.priority ?? "Unprioritised") === groupName).length;
      body.push(
        <tr className="fd-group" key={`g-${groupName}`}>
          <td colSpan={8}>
            <span className={`pv-tag pv-tag--${priorityTone(f.priority)}`}>{groupName}</span>
            <span className="fd-group__n">
              {count(nVisible)} {nVisible === 1 ? "finding" : "findings"}
              {nVisible !== n ? ` of ${count(n)}` : ""}
            </span>
          </td>
        </tr>,
      );
    }
    const open = expanded === f.key;
    const isClosing = closing === f.key;
    const detailId = `${uid}-d-${f.key}`;
    body.push(
      <tr
        className={`fd-row${open ? " fd-row--open" : ""}`}
        key={f.key}
        onClick={onRowClick(f.key)}
      >
        <td className="fd-cell fd-cell--n">
          <button
            type="button"
            className="fd-toggle"
            aria-expanded={open}
            aria-controls={open || isClosing ? detailId : undefined}
            aria-label={`Row ${i + 1}, ${f.label}: ${open ? "hide" : "show"} evidence and actions`}
            ref={(el) => {
              if (el) toggles.current.set(f.key, el);
              else toggles.current.delete(f.key);
            }}
            onClick={() => toggleRow(f.key)}
          >
            <span className="fd-toggle__n">{i + 1}</span>
            <span className="fd-toggle__chev" aria-hidden="true" />
          </button>
        </td>
        <td className="fd-cell fd-cell--entity">
          <EntityName f={f} onOpen={openDrawer} onScope={onFilter} />
          {f.entityType ? <span className="fd-entity__type">{f.entityType}</span> : null}
        </td>
        <td className="fd-cell fd-cell--cat">
          <span className="fd-cat" title={f.question ?? undefined}>
            {f.category}
          </span>
        </td>
        <td className="fd-cell fd-cell--priority">
          {f.priority ? (
            <span className={`pv-tag pv-tag--${priorityTone(f.priority)}`}>{f.priority}</span>
          ) : (
            <span className="fd-muted">—</span>
          )}
        </td>
        <td className="fd-cell fd-cell--source">
          <span className={`pv-tag pv-tag--${f.source === "both" ? "accent" : "neutral"}`}>
            {sourceLabel(f.source)}
          </span>
        </td>
        <td className="fd-cell fd-cell--stake pv-num">{cash(f.stake)}</td>
        <td className="fd-cell fd-cell--sev">
          <Severity value={f.severity} tone={priorityTone(f.priority)} />
        </td>
        <td className="fd-cell fd-cell--owner fd-muted">{f.owner ?? "—"}</td>
      </tr>,
    );
    if (open || isClosing) {
      body.push(
        <tr
          className={`fd-detail${isClosing ? " fd-detail--closing" : ""}`}
          key={`${f.key}-detail`}
          id={detailId}
          onAnimationEnd={
            isClosing ? () => setClosing((c) => (c === f.key ? null : c)) : undefined
          }
        >
          <td colSpan={8}>
            <div className="fd-detail__clip">
              <div className="fd-detail__body">
                <Detail
                  f={f}
                  ordinal={i + 1}
                  active={f.scope !== null && state.filters[f.scope.dim] === f.scope.value}
                  onOpen={openDrawer}
                  onScope={onFilter}
                  onAsk={openAsk}
                />
              </div>
            </div>
          </td>
        </tr>,
      );
    }
  });

  return (
    <div className="fd" onKeyDown={onKeyDown}>
      {/* The strip. Sticky under the header so the controls travel with the
          reader down a sixty-row table. */}
      <div className="fd-strip" role="region" aria-label="Filter and sort the findings">
        <div className="fd-chips" role="group" aria-label="Category">
          <button
            type="button"
            className="fd-chip"
            aria-pressed={cats.size === 0}
            onClick={() => {
              if (cats.size === 0) return;
              setCats(new Set());
              touch();
            }}
          >
            <span className="fd-chip__label">All</span>
            <span className="fd-chip__n">{count(total)}</span>
          </button>
          {chips.map((c) => {
            const n = catCount.get(c.name) ?? 0;
            const tip =
              c.total !== null && c.total !== n
                ? `${count(n)} of the ${count(c.total)} ${c.name} findings in the book are in this list`
                : undefined;
            return (
              <button
                type="button"
                className="fd-chip"
                key={c.name}
                aria-pressed={cats.has(c.name)}
                disabled={n === 0}
                title={tip}
                onClick={() => toggleCat(c.name)}
              >
                <span className="fd-chip__label">{c.name}</span>
                <span className="fd-chip__n">{count(n)}</span>
                {c.stake !== null && n > 0 ? (
                  <span className="fd-chip__stake">{money(c.stake)} at stake</span>
                ) : null}
              </button>
            );
          })}
        </div>

        <div className="fd-controls">
          <Segment<Priority | "all">
            label="Priority"
            value={priority}
            onChange={(v) => {
              setPriority(v);
              touch();
            }}
            options={[
              { key: "all", label: "All", n: total },
              ...PRIORITIES.map((p) => ({ key: p, label: p, n: priorityCount.get(p) ?? 0 })),
            ]}
          />
          <Segment<Source | "all">
            label="Source"
            value={source}
            onChange={(v) => {
              setSource(v);
              touch();
            }}
            options={[
              { key: "all", label: "All", n: total },
              ...SOURCES.map((s) => ({ key: s.key, label: s.label, n: sourceCount.get(s.key) ?? 0 })),
            ]}
          />
          <label className="fd-sort">
            <span className="fd-sort__label">Sort</span>
            <select
              className="fd-sort__select"
              value={sort}
              onChange={(e) => {
                setSort(e.target.value as SortKey);
                touch();
              }}
            >
              {SORTS.map((s) => (
                <option value={s.key} key={s.key}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <p className="fd-count" aria-live="polite">
            <strong>{count(visible.length)}</strong> of {count(total)} findings
          </p>
        </div>
      </div>

      {total === 0 ? (
        <p className="pv-empty">No findings for this persona in this slice.</p>
      ) : visible.length === 0 ? (
        <p className="pv-empty">
          <strong>Nothing matches this selection.</strong>
          Widen a filter above, or{" "}
          <button
            type="button"
            className="fd-linkbtn"
            onClick={() => {
              setCats(new Set());
              setPriority("all");
              setSource("all");
              touch();
            }}
          >
            show all {count(total)}
          </button>
          .
        </p>
      ) : (
        <div className="fd-tablewrap">
          <table className="fd-table">
            <caption className="pv-sr">
              Findings, one per row. Activate a row to read its evidence and act on it.
            </caption>
            <thead>
              <tr>
                <th scope="col" className="fd-cell--n">
                  <span className="pv-sr">Row</span>
                  <span aria-hidden="true">#</span>
                </th>
                <th scope="col">Entity</th>
                <th scope="col">Category</th>
                <th scope="col">Priority</th>
                <th scope="col">Source</th>
                <th scope="col" className="pv-num">
                  <abbr title="Gross profit at stake, not revenue">Value at stake</abbr>
                </th>
                <th scope="col">Severity</th>
                <th scope="col">Owner</th>
              </tr>
            </thead>
            <tbody className="fd-body" key={tick}>
              {body}
            </tbody>
          </table>
        </div>
      )}

      {hidden > 0 ? (
        <div className="fd-more">
          <button type="button" className="fd-btn" onClick={() => setShowAll(true)}>
            Show all {count(visible.length)}
          </button>
          <span className="fd-muted">{count(hidden)} more below the fold</span>
        </div>
      ) : null}

      {total > 0 ? (
        <p className="fd-foot">
          Value at stake is gross profit, not revenue. Rows are numbered in the order shown;
          the detector's own reference is printed in the open row.
        </p>
      ) : null}

      {children}
    </div>
  );
}

/* ========================================================================== *
 * Pieces
 * ========================================================================== */

/**
 * The entity's name, as the action it affords: an opportunity opens its deal
 * drawer, an account, rep or industry scopes the page, anything else is text.
 */
function EntityName({
  f,
  onOpen,
  onScope,
}: {
  f: Finding;
  onOpen: (drawer: string) => void;
  onScope: (dim: DimKey, value: string) => void;
}) {
  if (f.isOpp && f.entityId) {
    const id = f.entityId;
    return (
      <button
        type="button"
        className="fd-entity fd-link"
        title="Open the deal"
        onClick={() => onOpen(`deal:${id}`)}
      >
        {f.label}
      </button>
    );
  }
  if (f.scope) {
    const { dim, value } = f.scope;
    return (
      <button
        type="button"
        className="fd-entity fd-link"
        title="Filter the page to this"
        onClick={() => onScope(dim, value)}
      >
        {f.label}
      </button>
    );
  }
  return <span className="fd-entity">{f.label}</span>;
}

/** A 0–100 severity as a short track with the number beside it. */
function Severity({ value, tone }: { value: number | null; tone: Tone }) {
  if (value === null) return <span className="fd-muted">—</span>;
  const w = Math.max(0, Math.min(100, value));
  return (
    <span className="fd-sev" title={`Severity ${count(value)} of 100`}>
      <span className="fd-sev__track" aria-hidden="true">
        <span className={`fd-sev__fill fd-sev__fill--${tone}`} style={{ width: `${w}%` }} />
      </span>
      <span className="fd-sev__n">{count(value)}</span>
    </span>
  );
}

/** The open row: the question, the evidence, the next step, and the actions. */
function Detail({
  f,
  ordinal,
  active,
  onOpen,
  onScope,
  onAsk,
}: {
  f: Finding;
  ordinal: number;
  active: boolean;
  onOpen: (drawer: string) => void;
  onScope: (dim: DimKey, value: string) => void;
  onAsk: (query: string) => void;
}) {
  const question = `Why was ${f.label} flagged for ${f.category}?`;
  const scope = f.scope;
  return (
    <div className="fd-detail__inner">
      <p className="fd-detail__ref">
        Row {ordinal}
        {f.ref ? <> · Ref <code>{f.ref}</code></> : null}
        {f.entityType ? <> · {f.entityType}</> : null}
      </p>
      {f.question ? <p className="fd-detail__q">{f.question}</p> : null}
      <dl className="fd-detail__facts">
        <div className="fd-detail__fact">
          <dt>Evidence</dt>
          <dd>{f.evidence ?? "—"}</dd>
        </div>
        {f.liveEvidence ? (
          <div className="fd-detail__fact">
            <dt>Live detector adds</dt>
            <dd>{f.liveEvidence}</dd>
          </div>
        ) : null}
      </dl>
      <blockquote className="fd-detail__do">
        <span className="fd-detail__do-label">Do this</span>
        <p>{f.action ?? "—"}</p>
      </blockquote>
      <div className="fd-detail__actions">
        {f.isOpp && f.entityId ? (
          <button type="button" className="fd-btn" onClick={() => onOpen(`deal:${f.entityId}`)}>
            Open deal
          </button>
        ) : null}
        {scope ? (
          <button
            type="button"
            className="fd-btn"
            aria-pressed={active}
            onClick={() => onScope(scope.dim, scope.value)}
          >
            {active ? "Page is filtered to this" : "Filter the page to this"}
          </button>
        ) : null}
        <button type="button" className="fd-btn fd-btn--ask" onClick={() => onAsk(question)}>
          <Glyph kind="cause" className="fd-btn__glyph" size={14} />
          Ask about this
        </button>
      </div>
    </div>
  );
}

/** One-of segment, every segment a real button at the tap floor, with counts. */
function Segment<K extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: K;
  onChange: (v: K) => void;
  options: readonly { key: K; label: string; n: number }[];
}) {
  return (
    <div className="fd-seg" role="group" aria-label={label}>
      <span className="fd-seg__label">{label}</span>
      {options.map((o) => (
        <button
          type="button"
          className="fd-seg__btn"
          key={o.key}
          aria-pressed={value === o.key}
          disabled={o.n === 0 && value !== o.key}
          onClick={() => {
            if (value !== o.key) onChange(o.key);
          }}
        >
          {o.label}
          <span className="fd-seg__n">{count(o.n)}</span>
        </button>
      ))}
    </div>
  );
}
