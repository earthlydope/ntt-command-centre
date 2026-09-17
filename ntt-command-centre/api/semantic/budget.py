"""
Plan, coverage and RAG — and an honest account of where the plan does not foot.

The budget is not a separate file in this round: the extract carries it
denormalised onto every opportunity line, at four grains —

    q_budget_*        entity, per fiscal quarter          (3 distinct values)
    cell_q_budget_*   per (quarter, LOB, portfolio)       (52 of a possible 60)
    m_budget_*        entity, per month                   (9 distinct values)
    cell_m_budget_*   per (month, LOB, portfolio)         (136 of a possible 180)

**The missing cells are the important part.** A budget cell only reaches the
extract if at least one opportunity line happens to sit in that
quarter/LOB/portfolio, because that is the row it was denormalised onto. FY26-Q3
holds only 67 lines, so just 13 of its 20 cells appear and the LOB drill-down
sums to $853,594 against an entity headline of $936,244 — a 8.8% shortfall that
is an artefact of the file's shape, not a real gap in the plan. December is worse
at -23.9%.

A dashboard that silently lets a drill-down disagree with the tile above it is
worse than one that refuses to drill. So every breakdown this module returns
carries its own `residual` and `cellsPresent`, and the UI is required to print
them. `totals()` always reads the entity grain, never the sum of the cells.

**Coverage has one definition.** Open GP against the plan still to deliver,
all three measured over `forward_window` — the current quarter onward, or the
one quarter a filter names — and rolled up from `_coverage_cells` whether the
caller wants the tile, the LOB × portfolio grid or a per-line row. Two pages
showing two coverages for the same line is the failure this guards against.

RAG thresholds are the client's, carried through from the export's own
`TP_RAGStatus` / `QP_RAGStatus` / `ClosedCommit_RAGStatus` columns rather than
reinvented here — with one correction of framing: those columns read Red for
Oct-Dec because the quarter has not happened yet and has no closed business, not
because anything is wrong. That is stated wherever they are shown.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .loader import AS_OF, CUR_QUARTER, facts
from .measures import FilterState, money, slice_frame, subset
from .personas import Principal

BUDGET_CAPTION = (
    "Plan is the client's own synthetic target, carried from the extract's budget "
    "columns. Treat RAG outcomes as illustrative — the method (Country → LOB → "
    "Portfolio, revenue and GP, annual then rolled down) is how NTT says it is "
    "actually done; the figures are scaled to this sample."
)

FUTURE_QUARTER_NOTE = (
    "FY26-Q3 (Oct-Dec) has not happened yet, so it carries the full plan and almost "
    "no closed business. Its red status is a calendar position, not a performance "
    "finding."
)


@functools.lru_cache(maxsize=1)
def entity_quarters() -> pd.DataFrame:
    """Entity plan per fiscal quarter. The authoritative grain."""
    f = facts()
    q = f.groupby("quarter").agg(
        budget_revenue=("q_budget_revenue", "first"),
        budget_gp=("q_budget_gp", "first"),
        gap_gp=("q_gap_gp", "first"),
        total_pipeline_gp=("q_total_pipeline_gp", "first"),
        qualified_pipeline_gp=("q_qualified_pipeline_gp", "first"),
        tp_rag=("q_tp_rag", "first"),
        qp_rag=("q_qp_rag", "first"),
    ).reset_index().rename(columns={"quarter": "quarter"})
    return q.sort_values("quarter").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def entity_months() -> pd.DataFrame:
    f = facts()
    m = f.groupby("month").agg(
        budget_revenue=("m_budget_revenue", "first"),
        budget_gp=("m_budget_gp", "first"),
        closed_commit_gp=("m_closed_commit_gp", "first"),
        rag=("m_cc_rag", "first"),
    ).reset_index()
    return m.sort_values("month").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def cells_quarter() -> pd.DataFrame:
    """The (quarter × LOB × portfolio) plan cells that exist in the extract."""
    f = facts()
    return f.groupby(["quarter", "lob", "portfolio"]).agg(
        budget_revenue=("cell_q_budget_revenue", "first"),
        budget_gp=("cell_q_budget_gp", "first"),
    ).reset_index()


@functools.lru_cache(maxsize=1)
def cells_month() -> pd.DataFrame:
    f = facts()
    return f.groupby(["month", "lob", "portfolio"]).agg(
        budget_revenue=("cell_m_budget_revenue", "first"),
        budget_gp=("cell_m_budget_gp", "first"),
    ).reset_index()


def _residual(cell_sum: float, entity_sum: float, present: int, possible: int) -> dict:
    """The gap between a drill-down and the tile above it, named rather than hidden."""
    diff = cell_sum - entity_sum
    return {
        "cellsPresent": present,
        "cellsPossible": possible,
        "cellSum": float(cell_sum),
        "entityTotal": float(entity_sum),
        "residual": float(diff),
        "residualPct": float(100 * diff / entity_sum) if entity_sum else 0.0,
        "note": (
            f"{present} of {possible} plan cells exist in the extract. A cell only "
            f"appears where at least one opportunity line sits in that quarter, line of "
            f"business and portfolio, so the breakdown is short of the headline by "
            f"{abs(100 * diff / entity_sum):.1f}%. The headline is the authoritative "
            f"figure; this is a property of the file, not a gap in the plan."
        ) if abs(diff) > 1 else None,
    }


def forward_window(fs: FilterState) -> tuple[list[str], str]:
    """
    The quarters every coverage figure is measured over, and their label.

    Coverage is a FORWARD question — "is there enough pipeline for what is
    still to come" — so the window is the current quarter onward, or the one
    quarter a filter names. This is the ONE place that decides it: the tile,
    the LOB × portfolio grid and the per-line rows all read it from here, so
    the performance page and the actions page cannot disagree about the same
    line. Before it was shared, the grid netted the whole year's wins against
    one quarter's plan and read every line as delivered while the rows beside
    it read Networking at 1.30x.
    """
    if fs.quarter:
        return [fs.quarter], fs.quarter
    quarters = [q for q in entity_quarters()["quarter"] if q >= CUR_QUARTER]
    if not quarters:
        return [], CUR_QUARTER
    label = quarters[0] if len(quarters) == 1 else f"{quarters[0]} onward"
    return quarters, label


def _status(remaining: float, coverage: float | None) -> tuple[str, str]:
    """The one reading of a coverage ratio, so a bullet and a grid cell agree."""
    if remaining <= 0:
        return "Delivered", "good"
    if coverage >= 1.0:
        return "Covered", "good"
    if coverage >= 0.5:
        return "Thin", "warn"
    return "Uncovered", "danger"


def _coverage_cells(fs: FilterState, principal: Principal) -> tuple[pd.DataFrame, dict]:
    """
    Plan, won and open per (LOB, portfolio) over the forward window — the single
    computation behind `coverage_grid` and `coverage_by`.

    Won and open are the slice's lines whose close date falls in the window;
    the plan is the entity's cells for those quarters. A line can only sit in
    a cell that exists (that is how the cells reached the extract), so summing
    the cells to a dimension gives the same won and open as grouping the lines
    directly would.

    The second value is the footing at plan grain: how much of the entity's
    plan for the window the cells actually carry.
    """
    quarters, window = forward_window(fs)
    df = slice_frame(fs, principal)
    cells = cells_quarter()
    cells = cells[cells["quarter"].isin(quarters)]
    won_df = subset(df, "won")
    won_df = won_df[won_df["fiscal_quarter"].isin(quarters)]
    open_df = subset(df, "open")
    open_df = open_df[open_df["fiscal_quarter"].isin(quarters)]

    grid = cells.groupby(["lob", "portfolio"])["budget_gp"].sum().rename("budget_gp")
    won = won_df.groupby(["lob", "portfolio"])["acv_gp"].sum().rename("won_gp")
    open_gp = open_df.groupby(["lob", "portfolio"])["acv_gp"].sum().rename("open_gp")
    frame = pd.concat([grid, won, open_gp], axis=1).fillna(0.0)
    # Past the last planned quarter there are no cells at all, and an empty
    # concat loses the (lob, portfolio) index; give the callers the columns
    # they expect rather than a KeyError on the sort.
    if frame.empty:
        frame = pd.DataFrame(columns=["lob", "portfolio", "budget_gp", "won_gp", "open_gp"])
    else:
        frame = (frame.reset_index().sort_values(["lob", "portfolio"])
                 .reset_index(drop=True))

    eq = entity_quarters()
    entity_plan = float(eq.loc[eq["quarter"].isin(quarters), "budget_gp"].sum())
    entity_won = float(won_df["acv_gp"].sum())
    basis = {
        "quarters": quarters,
        "window": window,
        # Cells present is the number of quarter cells the extract carries for
        # the window, not the number of distinct rows they collapse to — four
        # LOB rows are not four cells, and counting them as such is what made
        # a 13-of-20 shortfall print as "4 of 20".
        "cellsPresent": int(len(cells)),
        "cellsPossible": 20 * len(quarters),
        "cellPlan": float(cells["budget_gp"].sum()),
        "entityPlan": entity_plan,
        "entityWon": entity_won,
        "entityRemaining": max(entity_plan - entity_won, 0.0),
    }
    return frame, basis


def _footing(rows: list[dict], basis: dict, what: str) -> dict:
    """
    Where a coverage breakdown stops footing to the tile above it, in words.

    Two things pull the rows away from the headline, and both are named
    rather than folded into one percentage. Plan cells the extract does not
    carry take their share of the plan with them — the 8.8% Q3 shortfall in
    the module docstring. And a row that has already beaten its plan carries a
    target of zero, not a negative one, so its surplus is not netted against
    the lines still short; the tile nets at entity grain and so sees it.
    """
    plan = _residual(basis["cellPlan"], basis["entityPlan"],
                     basis["cellsPresent"], basis["cellsPossible"])
    row_sum = float(sum(r["remainingGp"] for r in rows))
    entity_rem = float(basis["entityRemaining"])
    missing_plan = max(basis["entityPlan"] - basis["cellPlan"], 0.0)
    surplus = float(sum(max(r["wonGp"] - r["budgetGp"], 0.0) for r in rows))
    diff = row_sum - entity_rem
    reasons = []
    if missing_plan > 1:
        reasons.append(
            f"the plan is not broken down for every line ({basis['cellsPresent']} of "
            f"{basis['cellsPossible']} quarter cells exist in the extract), so "
            f"{money(missing_plan)} of it has no {what[:-1]} here")
    if surplus > 1:
        reasons.append(
            f"{money(surplus)} of surplus on lines already past their plan counts as "
            f"zero rather than offsetting the rest")
    note = None
    if abs(diff) > 1 and entity_rem > 0:
        # "of" when the rows fall short of the tile, "against" when the
        # clipped surpluses push them past it — "$456K of the $370K" reads as
        # a typo.
        joiner = "of" if row_sum < entity_rem else "against"
        note = (f"The {what} below add up to {money(row_sum)} {joiner} the "
                f"{money(entity_rem)} still to deliver from {basis['window']}"
                + (": " + "; ".join(reasons) if reasons else "")
                + ". The tile's figure is the authoritative one.")
    elif entity_rem <= 0 and row_sum > 1:
        note = (f"The plan for {basis['window']} is already delivered at entity grain; "
                f"the {money(row_sum)} of targets below are the lines still short of "
                f"their own share, which the surplus elsewhere covers.")
    return {
        **plan,
        "planNote": plan["note"],
        "remainingRowSum": row_sum,
        "remainingEntity": entity_rem,
        "remainingResidual": float(diff),
        "remainingResidualPct": float(100 * diff / entity_rem) if entity_rem else 0.0,
        "surplusClipped": surplus,
        "note": note,
    }


def totals(fs: FilterState, principal: Principal) -> dict:
    """
    Plan, actual and coverage for the current slice.

    Coverage is open GP over the plan that remains, not over the whole year's
    plan: measuring this quarter's pipeline against a target that is already
    two-thirds delivered would make every segment look catastrophically thin.
    """
    df = slice_frame(fs, principal)
    q = entity_quarters()
    if fs.quarter:
        q = q[q["quarter"] == fs.quarter]

    budget_gp = float(q["budget_gp"].sum())
    budget_rev = float(q["budget_revenue"].sum())
    won = subset(df, "won")
    open_df = subset(df, "open")
    qual_df = subset(df, "qualified")

    won_gp = float(won["acv_gp"].sum())
    open_gp = float(open_df["acv_gp"].sum())
    qual_gp = float(qual_df["acv_gp"].sum())
    remaining = max(budget_gp - won_gp, 0.0)

    # Coverage is a FORWARD question — "is there enough pipeline for what is
    # still to come" — so it is measured from the current quarter onward, not
    # across the whole year. Measured against the full-year residual it reads
    # 26.67x here, because Q1 and Q2 over-delivered and left almost nothing
    # outstanding; that number is arithmetically correct and operationally
    # meaningless, and nobody would act on it. The window comes from
    # `forward_window` so the grid and the per-line rows measure the same one.
    fwd_q, fwd_label = forward_window(fs)
    fwd_budget = float(q.loc[q["quarter"].isin(fwd_q), "budget_gp"].sum())
    fwd_won = float(won.loc[won["fiscal_quarter"].isin(fwd_q), "acv_gp"].sum())
    fwd_open = float(open_df.loc[open_df["fiscal_quarter"].isin(fwd_q), "acv_gp"].sum())
    fwd_remaining = max(fwd_budget - fwd_won, 0.0)

    # When a persona is scoped to part of the book, the entity plan is not their
    # plan — there is no rep or pod quota column anywhere in the extract, and
    # inventing one would be a fabrication. The plan is therefore reported at
    # entity grain with the principal's share stated beside it.
    scoped = bool(principal.predicate)

    return {
        "asOf": AS_OF.isoformat(),
        "quarter": fs.quarter or CUR_QUARTER,
        "budgetGp": budget_gp,
        "budgetRevenue": budget_rev,
        "wonGp": won_gp,
        "openGp": open_gp,
        "qualifiedGp": qual_gp,
        "remainingGp": remaining,
        "attainmentPct": (100 * won_gp / budget_gp) if budget_gp else 0.0,
        "coverage": (fwd_open / fwd_remaining) if fwd_remaining else None,
        "qualifiedCoverage": (qual_gp / fwd_remaining) if fwd_remaining else None,
        "coverageBasis": {
            "quarters": fwd_q,
            "window": fwd_label,
            "planGp": fwd_budget,
            "wonGp": fwd_won,
            "openGp": fwd_open,
            "remainingGp": fwd_remaining,
            "note": f"Coverage is measured over {fwd_label} — pipeline against "
                    f"the plan that is still to come, not against the year's "
                    f"residual.",
        },
        "gapGp": budget_gp - won_gp,
        "scopedToPersona": scoped,
        "planGrainNote": (
            "Plan exists at entity grain only — the extract carries no rep or pod "
            "quota column. Attainment below is this scope's contribution against the "
            "entity plan, not a personal target."
        ) if scoped else None,
        "caption": BUDGET_CAPTION,
        "futureQuarterNote": FUTURE_QUARTER_NOTE,
    }


def by_quarter(fs: FilterState, principal: Principal) -> list[dict]:
    """Plan vs won vs open per quarter, with the export's own RAG carried through."""
    df = slice_frame(fs, principal)
    q = entity_quarters()
    won = subset(df, "won").groupby("fiscal_quarter")["acv_gp"].sum()
    open_gp = subset(df, "open").groupby("fiscal_quarter")["acv_gp"].sum()
    out = []
    for r in q.itertuples(index=False):
        w = float(won.get(r.quarter, 0.0))
        o = float(open_gp.get(r.quarter, 0.0))
        remaining = max(float(r.budget_gp) - w, 0.0)
        out.append({
            "quarter": r.quarter,
            "budgetGp": float(r.budget_gp),
            "budgetRevenue": float(r.budget_revenue),
            "wonGp": w,
            "openGp": o,
            "remainingGp": remaining,
            "coverage": (o / remaining) if remaining else None,
            "attainmentPct": (100 * w / r.budget_gp) if r.budget_gp else 0.0,
            "totalPipelineRag": r.tp_rag,
            "qualifiedPipelineRag": r.qp_rag,
            "isFuture": r.quarter > CUR_QUARTER,
            "isCurrent": r.quarter == CUR_QUARTER,
        })
    return out


def by_month(fs: FilterState, principal: Principal) -> list[dict]:
    """Monthly plan against won, for the column-and-line combo."""
    df = slice_frame(fs, principal)
    m = entity_months()
    won = subset(df, "won").groupby("fiscal_month")["acv_gp"].sum()
    out = []
    for r in m.itertuples(index=False):
        w = float(won.get(r.month, 0.0))
        future = r.month > AS_OF.strftime("%Y-%m")
        out.append({
            "month": r.month,
            "budgetGp": float(r.budget_gp),
            "wonGp": w,
            "closedCommitGp": float(r.closed_commit_gp),
            "rag": r.rag,
            "attainmentPct": (100 * w / r.budget_gp) if r.budget_gp else 0.0,
            "isFuture": future,
            "annotation": "No closed history yet" if future else None,
        })
    return out


def coverage_grid(fs: FilterState, principal: Principal) -> dict:
    """
    The LOB × Portfolio coverage heat map — where a target has no pipeline behind it.

    Measured over the forward window, like every other coverage figure. Each
    cell is one (LOB, portfolio) pair with its plan, won and open summed across
    the window's quarters; `quarter` is the window's label, since a grid that
    reads "FY26-Q2" over Q2-and-Q3 numbers is the disagreement this module
    exists to prevent.

    A hole is a target still to deliver with nothing open against it. A cell
    whose plan is already won has no target left, so its empty pipeline is
    "done", not a hole. Returns the footing alongside, because these cells do
    not sum to the tile above them and the UI must say so.
    """
    frame, basis = _coverage_cells(fs, principal)
    out = []
    for r in frame.itertuples(index=False):
        b, w, o = float(r.budget_gp), float(r.won_gp), float(r.open_gp)
        remaining = max(b - w, 0.0)
        cov = (o / remaining) if remaining else None
        status, tone = _status(remaining, cov)
        out.append({
            "lob": r.lob, "portfolio": r.portfolio,
            "budgetGp": b, "openGp": o, "wonGp": w, "remainingGp": remaining,
            "coverage": cov,
            "status": status, "tone": tone,
            "isHole": bool(remaining > 0 and o <= 0),
            # Thin is "has pipeline, but under half of what is left" — it
            # excludes the holes so the two counts can be read side by side
            # without a cell being in both.
            "isThin": bool(o > 0 and cov is not None and cov < 0.5),
        })
    return {
        "quarter": basis["window"],
        "quarters": basis["quarters"],
        "window": basis["window"],
        "cells": out,
        "holes": sum(1 for c in out if c["isHole"]),
        "thin": sum(1 for c in out if c["isThin"]),
        "footing": _footing(out, basis, "cells"),
        "caption": BUDGET_CAPTION,
    }


def coverage_by(fs: FilterState, principal: Principal, dim: str) -> dict:
    """
    Plan vs pipeline for one dimension — `lob` or `portfolio` only.

    Rolled up from the same cells as `coverage_grid`, over the same forward
    window, and the remaining target is netted at THIS grain: a line's plan
    less the line's wins, floored at zero. That is why the rows do not sum to
    the tile — the tile nets at entity grain — and the footing says so.
    """
    if dim not in ("lob", "portfolio"):
        raise KeyError(
            f"the plan exists only at LOB and portfolio grain, not '{dim}'. "
            "Every other dimension would need a budget the extract does not carry."
        )
    frame, basis = _coverage_cells(fs, principal)
    g = frame.groupby(dim)[["budget_gp", "won_gp", "open_gp"]].sum()
    rows = []
    for key, r in g.iterrows():
        b, w, o = float(r.budget_gp), float(r.won_gp), float(r.open_gp)
        remaining = max(b - w, 0.0)
        cov = (o / remaining) if remaining else None
        status, tone = _status(remaining, cov)
        rows.append({
            "key": str(key), "budgetGp": b, "wonGp": w, "openGp": o,
            "remainingGp": remaining,
            "coverage": cov,
            "attainmentPct": (100 * w / b) if b else 0.0,
            "status": status, "tone": tone,
        })
    rows.sort(key=lambda r: -r["budgetGp"])
    return {
        "quarter": basis["window"],
        "quarters": basis["quarters"],
        "window": basis["window"],
        "dimension": dim, "rows": rows,
        "footing": _footing(rows, basis, "rows"),
    }


def bridge(fs: FilterState, principal: Principal) -> list[dict]:
    """
    The FY26 GP bridge: plan, what each quarter delivered, what is still open,
    and the residual gap. This is the waterfall the executive reads aloud.
    """
    df = slice_frame(fs, principal)
    q = entity_quarters()
    budget = float(q["budget_gp"].sum())
    won = subset(df, "won").groupby("fiscal_quarter")["acv_gp"].sum()
    open_gp = float(subset(df, "open")["acv_gp"].sum())

    steps = [{"key": "plan", "label": "FY26 plan", "value": budget, "type": "start"}]
    delivered = 0.0
    for quarter in q["quarter"]:
        w = float(won.get(quarter, 0.0))
        delivered += w
        steps.append({
            "key": quarter, "label": f"{quarter} won", "value": -w, "type": "delta",
            "tone": "good",
        })
    steps.append({
        "key": "open", "label": "Open pipeline", "value": -open_gp, "type": "delta",
        "tone": "warn",
        "note": "At face value, not risk-weighted — the closure model is too weak on "
                "this extract to weight with.",
    })
    # A negative residual is a surplus, and calling it "Uncovered" would invert the
    # reading. On this extract it IS a surplus, and for a reason worth stating: the
    # annual plan was rolled evenly across three quarters while every win sits in
    # Q1-Q2, because Q3 has not happened yet. The over-attainment here and the red
    # in Q3 are the same calendar artefact seen from opposite ends.
    residual = budget - delivered - open_gp
    surplus = residual < 0
    steps.append({
        "key": "gap",
        "label": "Surplus to plan" if surplus else "Uncovered",
        "value": residual,
        "type": "end",
        "tone": "good" if surplus else "danger",
        "note": (
            "Delivered plus open exceeds the full-year plan because the plan was rolled "
            "evenly across three quarters while all closed business falls in Q1-Q2. The "
            "same calendar position is what makes Q3 read red."
        ) if surplus else None,
    })
    return steps


def reset_caches() -> None:
    for fn in (entity_quarters, entity_months, cells_quarter, cells_month):
        fn.cache_clear()
