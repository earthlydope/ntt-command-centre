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
from .measures import FilterState, slice_frame, subset
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
    # meaningless, and nobody would act on it.
    fwd_q = [q for q in entity_quarters()["quarter"] if q >= CUR_QUARTER]
    if fs.quarter:
        fwd_q = [fs.quarter]
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
            "planGp": fwd_budget,
            "wonGp": fwd_won,
            "openGp": fwd_open,
            "remainingGp": fwd_remaining,
            "note": f"Coverage is measured from {CUR_QUARTER} onward — pipeline "
                    f"against the plan that is still to come, not against the "
                    f"year's residual.",
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

    Returns the residual alongside, because these cells do not sum to the
    headline and the UI must say so.
    """
    df = slice_frame(fs, principal)
    cells = cells_quarter()
    q = fs.quarter or CUR_QUARTER
    cells = cells[cells["quarter"] == q]

    open_df = subset(df, "open")
    open_df = open_df[open_df["fiscal_quarter"] == q] if fs.quarter else open_df
    won_df = subset(df, "won")
    won_df = won_df[won_df["fiscal_quarter"] == q] if fs.quarter else won_df

    open_gp = open_df.groupby(["lob", "portfolio"])["acv_gp"].sum()
    won_gp = won_df.groupby(["lob", "portfolio"])["acv_gp"].sum()

    out = []
    for r in cells.itertuples(index=False):
        k = (r.lob, r.portfolio)
        o = float(open_gp.get(k, 0.0))
        w = float(won_gp.get(k, 0.0))
        remaining = max(float(r.budget_gp) - w, 0.0)
        cov = (o / remaining) if remaining else None
        out.append({
            "lob": r.lob, "portfolio": r.portfolio,
            "budgetGp": float(r.budget_gp),
            "openGp": o, "wonGp": w, "remainingGp": remaining,
            "coverage": cov,
            "isHole": bool(o == 0 and r.budget_gp > 0),
            "isThin": bool(cov is not None and cov < 0.5),
        })

    entity_total = float(entity_quarters().loc[
        entity_quarters()["quarter"] == q, "budget_gp"].sum())
    return {
        "quarter": q,
        "cells": out,
        "holes": sum(1 for c in out if c["isHole"]),
        "thin": sum(1 for c in out if c["isThin"]),
        "footing": _residual(cells["budget_gp"].sum(), entity_total, len(cells), 20),
        "caption": BUDGET_CAPTION,
    }


def coverage_by(fs: FilterState, principal: Principal, dim: str) -> dict:
    """Plan vs pipeline for one dimension — `lob` or `portfolio` only."""
    if dim not in ("lob", "portfolio"):
        raise KeyError(
            f"the plan exists only at LOB and portfolio grain, not '{dim}'. "
            "Every other dimension would need a budget the extract does not carry."
        )
    df = slice_frame(fs, principal)
    q = fs.quarter or CUR_QUARTER
    cells = cells_quarter()
    cells = cells[cells["quarter"] == q]
    plan = cells.groupby(dim)["budget_gp"].sum()

    won = subset(df, "won").groupby(dim)["acv_gp"].sum()
    open_gp = subset(df, "open").groupby(dim)["acv_gp"].sum()

    rows = []
    for key in plan.index:
        w = float(won.get(key, 0.0))
        o = float(open_gp.get(key, 0.0))
        b = float(plan[key])
        remaining = max(b - w, 0.0)
        rows.append({
            "key": key, "budgetGp": b, "wonGp": w, "openGp": o,
            "remainingGp": remaining,
            "coverage": (o / remaining) if remaining else None,
            "attainmentPct": (100 * w / b) if b else 0.0,
        })
    rows.sort(key=lambda r: -r["budgetGp"])
    entity_total = float(entity_quarters().loc[
        entity_quarters()["quarter"] == q, "budget_gp"].sum())
    return {
        "quarter": q, "dimension": dim, "rows": rows,
        "footing": _residual(plan.sum(), entity_total,
                             int(cells.groupby(dim).ngroups), 20),
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
