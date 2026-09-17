"""
Ad-hoc questions: the model proposes a PLAN, the server validates and executes it.

There is no text-to-SQL here, and that is deliberate. The model never sees a
row, never writes a query string, and never names a column that is not in the
catalog. It emits a small JSON object describing WHAT it wants; this module
checks every field of that object against the live registries and then runs a
fixed dispatch over the semantic layer. User text never reaches pandas, there is
no `eval`, and a plan cannot widen the caller's row-level scope because
`slice_frame` applies the principal's predicate before any filter in the plan.

The chart is chosen the same way: the model may PROPOSE a repository key, but
admissibility is decided here from the shape of the RESULT, with three guards
drawn from measured properties of this dataset —

  * a raw stage count is never rendered as a funnel (it exceeds 100% partway
    down, because deals are logged straight into the middle of the ladder),
  * a treemap or bubble always gets its tail grouped (one account is 14.5% of
    gross profit and would otherwise take a seventh of the canvas),
  * anything with a negative value degrades away from treemap, mekko and funnel,
    none of which can represent one honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import charts as C
from .dimensions import REGISTRY, dim_column
from .measures import (
    FilterState,
    SUBSETS,
    by_dimension,
    by_time,
    gm,
    money,
    opps,
    slice_frame,
    subset,
)
from .personas import Principal

METRICS: tuple[str, ...] = ("gp", "rev", "revenue", "gm", "count", "winrate",
                            "coverage", "cycle", "risk", "quietdays")
INTENTS: tuple[str, ...] = ("rank", "compare", "trend", "distribution",
                            "composition", "relationship", "flow", "schedule",
                            "lookup", "explain", "refuse")


class PlanError(ValueError):
    """The plan named something that does not exist. Answered as a refusal."""


@dataclass
class Result:
    rows: list[dict]
    shape: str
    chart_key: str | None
    chart_why: str
    claim: str
    columns: list[str]
    note: str | None = None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PlanError(msg)


def validate(plan: dict, principal: Principal) -> dict:
    """Check every field against the live registries. Raises `PlanError`."""
    _require(isinstance(plan, dict), "plan must be an object")
    # `refuse` is the planner's way of declining; see the schema note in
    # api/llm/service.py for why this is an enum member and not a free-text
    # field. A plan may still carry `unanswerable` when it comes from a direct
    # POST rather than from the model.
    if str(plan.get("intent")) == "refuse" or plan.get("unanswerable"):
        return plan

    intent = str(plan.get("intent", "rank"))
    _require(intent in INTENTS, f"unknown intent '{intent}'")

    metric = str(plan.get("metric", "gp")).lower()
    _require(metric in METRICS, f"unknown metric '{metric}'. Known: {METRICS}")

    sub = str(plan.get("subset", "all")).lower()
    _require(sub in SUBSETS, f"unknown subset '{sub}'. Known: {SUBSETS}")

    breakdown = plan.get("breakdown") or []
    _require(isinstance(breakdown, list) and len(breakdown) <= 2,
             "breakdown must be a list of at most two dimensions")
    for d in breakdown:
        _require(d in REGISTRY, f"'{d}' is not a filterable dimension")

    for f in plan.get("filters") or []:
        _require(isinstance(f, dict) and "dim" in f and "value" in f,
                 "each filter needs a dim and a value")
        _require(f["dim"] in REGISTRY, f"'{f['dim']}' is not a filterable dimension")
        # RLS-aware domain check: a value outside the principal's own rows is a
        # refusal, not an empty result — the difference matters, because an
        # empty chart reads as "there is none of this" rather than "not yours".
        domain = set(REGISTRY[f["dim"]].values(slice_frame(FilterState(), principal)))
        _require(str(f["value"]) in domain,
                 f"'{f['value']}' is not a value of {f['dim']} within your scope")

    grain = str((plan.get("time") or {}).get("grain", "none")).lower()
    _require(grain in ("none", "month", "quarter"), f"unknown time grain '{grain}'")
    return plan


def claim_key(plan: dict) -> str:
    metric = {"revenue": "rev"}.get(str(plan.get("metric", "gp")), str(plan.get("metric", "gp")))
    sub = plan.get("subset", "all")
    bits = f"{metric}.{sub}"
    bd = plan.get("breakdown") or []
    if bd:
        bits += ".by:" + "+".join(bd)
    grain = (plan.get("time") or {}).get("grain", "none")
    if grain != "none":
        bits += f".t:{grain}"
    return bits


def _value_column(metric: str) -> str:
    return "acv_gp" if metric in ("gp", "risk") else "acv_revenue"


def execute(plan: dict, fs: FilterState, principal: Principal,
            charts_say: list[str]) -> Result:
    """Run a validated plan. RLS is applied by `slice_frame` before anything else."""
    plan = validate(plan, principal)
    if str(plan.get("intent")) == "refuse" or plan.get("unanswerable"):
        why = (plan.get("refuseReason") or plan.get("unanswerable")
               or "That question cannot be answered from this data.")
        return Result([], "none", None, "", "", [], note=str(why))

    metric = str(plan.get("metric", "gp")).lower()
    sub = str(plan.get("subset", "all")).lower()
    breakdown = list(plan.get("breakdown") or [])
    grain = str((plan.get("time") or {}).get("grain", "none")).lower()
    limit = int(plan.get("limit") or 15)
    limit = max(1, min(limit, 30))

    local = fs
    for f in plan.get("filters") or []:
        local = local.toggled(f["dim"], str(f["value"]))
    df = subset(slice_frame(local, principal), sub)

    if df.empty:
        return Result([], "none", None, "", claim_key(plan), [],
                      note="No rows match that slice within your scope.")

    measure_for_agg = "gp" if metric in ("gp", "risk") else "revenue"
    local = local.with_(measure=measure_for_agg)

    # --- dispatch ------------------------------------------------------- #
    if grain != "none" and not breakdown:
        g = by_time(df, local, "month" if grain == "month" else "quarter")
        rows = [{"key": r.key, "value": float(r.value), "opps": int(r.opps)}
                for r in g.itertuples(index=False)]
        shape = "temporal×measure"

    elif len(breakdown) == 1:
        dim = breakdown[0]
        if metric == "gm":
            col = dim_column(dim)
            g = df.groupby(col).agg(revenue=("acv_revenue", "sum"),
                                    gp=("acv_gp", "sum")).reset_index()
            g["value"] = np.where(g["revenue"] != 0, 100 * g["gp"] / g["revenue"], 0.0)
            rows = [{"key": str(r[col]), "value": float(r["value"]),
                     "revenue": float(r["revenue"])} for _, r in g.iterrows()]
        elif metric == "count":
            col = dim_column(dim)
            basis = REGISTRY[dim].count_basis
            g = (df.groupby(col)["opportunity_code"].nunique() if basis == "opportunities"
                 else df.groupby(col).size())
            rows = [{"key": str(k), "value": float(v)} for k, v in g.items()]
        elif metric == "winrate":
            col = dim_column(dim)
            closed = df[df["is_closed"]]
            o = closed.groupby([col, "opportunity_code"])["is_won"].first().reset_index()
            g = o.groupby(col)["is_won"].agg(["mean", "size"]).reset_index()
            rows = [{"key": str(r[col]), "value": float(100 * r["mean"]),
                     "n": int(r["size"])} for _, r in g.iterrows()]
        else:
            g = by_dimension(df, dim, local, top=limit)
            rows = [{"key": r.key, "value": float(r.value), "share": float(r.share),
                     "opps": int(r.opps), "lines": int(r.lines)}
                    for r in g.itertuples(index=False)]
        shape = "categorical×measure"

    elif len(breakdown) == 2:
        a, b = dim_column(breakdown[0]), dim_column(breakdown[1])
        vc = _value_column(metric)
        g = df.groupby([a, b])[vc].sum().reset_index()
        rows = [{"row": str(r[a]), "col": str(r[b]), "value": float(r[vc])}
                for _, r in g.iterrows()]
        shape = "categorical×categorical×measure"

    else:
        vc = _value_column(metric)
        val = (gm(df) if metric == "gm"
               else float(opps(df)) if metric == "count"
               else float(df[vc].sum()))
        rows = [{"key": "total", "value": val}]
        shape = "scalar"

    if shape == "categorical×measure":
        # The grouped tail is not a competitor. `by_dimension(top=)` appends an
        # "Other (43)" bucket whose value is the sum of everything below the cut,
        # so a value-sort puts it first and the answer to "who is biggest" reads
        # "Other". It is kept — dropping it would make the shares not add up —
        # but it is pinned to the end.
        named = [r for r in rows if not str(r.get("key", "")).startswith("Other (")]
        other = [r for r in rows if str(r.get("key", "")).startswith("Other (")]
        rows = sorted(named, key=lambda r: -abs(r.get("value", 0)))[:limit] + other

    key, why = choose_chart(plan, shape, rows, charts_say)
    return Result(rows, shape, key, why, claim_key(plan),
                  list(rows[0].keys()) if rows else [])


def choose_chart(plan: dict, shape: str, rows: list[dict],
                 charts_say: list[str]) -> tuple[str | None, str]:
    """
    The model proposes; the registry disposes.

    A scalar gets no chart at all — a single number rendered as one bar is a
    worse answer than the number on its own.
    """
    if shape == "scalar" or not rows:
        return None, "The answer is a single figure; a chart would add nothing."

    negatives = any(float(r.get("value", 0)) < 0 for r in rows)
    admissible = {shape: C.select(shape)}
    hint = plan.get("chartHint")

    if shape == "categorical×measure":
        allowed = {"bar.categorical", "table.compact"}
        if not negatives and len(rows) <= 20:
            allowed |= {"treemap.nested"}
        # The wording reaches the customer under "Why this chart", so it says
        # what the picture is for rather than which rule fired.
        if hint == "treemap.nested" and hint in allowed:
            return hint, "Tiles because the question is about share, and the sizes add up."
        if hint == "table.compact":
            return hint, "A table because the values are on different scales."
        if hint in allowed:
            return hint, "One bar per item so the biggest stands out."
        if hint and hint not in allowed:
            reason = ("a negative value cannot be drawn as an area"
                      if negatives else f"{len(rows)} items is too many for tiles")
            return "bar.categorical", (
                f"One bar per item so the biggest stands out — {reason}.")
        return "bar.categorical", "One bar per item so the biggest stands out."

    if shape == "temporal×measure":
        if hint == "combo.columnline":
            return "line.timeseries", (
                "A line because this is a change over time; there is no plan "
                "line to set against it.")
        return "line.timeseries", "A line because this is a change over time."

    if shape == "categorical×categorical×measure":
        return "heat.grid", "A grid because two things are being compared at once."

    return admissible.get(shape, "table.compact"), (
        "A table because the values are on different scales.")


def to_chart_spec(plan: dict, result: Result, fs: FilterState) -> dict | None:
    """Wrap an executed result in a spec the existing repository can render."""
    if not result.chart_key or not result.rows:
        return None
    title = _title(plan)
    metric = str(plan.get("metric", "gp"))
    fmt = ("percent" if metric in ("gm", "winrate")
           else "number" if metric in ("count", "cycle", "quietdays")
           else "currency")
    label = {"gm": "GM %", "winrate": "Win rate", "count": "Count",
             "cycle": "Cycle days", "quietdays": "Days silent"}.get(
        metric, "ACV GP" if metric in ("gp", "risk") else "ACV Revenue")

    if result.shape == "categorical×categorical×measure":
        data: Any = {
            "rows": sorted({r["row"] for r in result.rows}),
            "cols": sorted({r["col"] for r in result.rows}),
            "cells": result.rows,
        }
    else:
        data = result.rows

    bd = (plan.get("breakdown") or [None])[0]
    return C.spec(
        f"ask_{abs(hash(result.claim)) % 100000}", title, result.shape, data,
        says=[result.claim], subtitle=result.chart_why, measure_label=label,
        fmt=fmt, click_dim=bd if bd in REGISTRY else None,
        count_basis=REGISTRY[bd].count_basis if bd in REGISTRY else None,
        basis_note=REGISTRY[bd].basis_note if bd in REGISTRY else None,
        key=result.chart_key,
    )


def _title(plan: dict) -> str:
    metric = {"gp": "Gross profit", "rev": "Revenue", "revenue": "Revenue",
              "gm": "Gross margin", "count": "Count", "winrate": "Win rate",
              "coverage": "Coverage", "cycle": "Cycle time",
              "risk": "Deal risk", "quietdays": "Days silent"}.get(
        str(plan.get("metric", "gp")), "Value")
    sub = {"all": "", "open": " — open", "won": " — won", "lost": " — lost",
           "closed": " — closed", "qualified": " — qualified",
           "pastdue": " — past due", "stalled": " — stalled"}.get(
        str(plan.get("subset", "all")), "")
    bd = plan.get("breakdown") or []
    by = f" by {' and '.join(REGISTRY[d].label.lower() for d in bd if d in REGISTRY)}" if bd else ""
    grain = (plan.get("time") or {}).get("grain", "none")
    t = f" by {grain}" if grain != "none" else ""
    return f"{metric}{sub}{by}{t}".strip()
