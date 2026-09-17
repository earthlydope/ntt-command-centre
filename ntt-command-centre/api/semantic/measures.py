"""
Every calculation in the product. Nothing above this module divides anything.

Two rules hold here and are the reason the numbers on screen cannot disagree
with each other:

  1. **`slice_frame` is the only way to get rows.** It applies the principal's
     RLS predicate first and the user's filters second. No caller may reach
     `loader.facts()` directly, because a frame obtained any other way has not
     been scoped and would silently show one persona another's pipeline.

  2. **A margin is never an average of margins.** `GM% = SUM(GP) / SUM(Revenue)`,
     revenue-weighted, always. On this book the unweighted mean of the row-level
     GM_Percent column reads 23.9% against a weighted 16.2% — the difference
     between a healthy-looking portfolio and the real one, because the column is
     dominated by thousands of tiny VBR lines at 44% margin on $197K of total
     revenue. `naive_gm()` exists only so the Context lens can show the wrong
     answer beside the right one and name the trap.

Grain discipline is enforced rather than documented: `count_on()` takes the
dimension and decides for itself whether to count lines or opportunities, so no
call site has to remember that Stage is opportunity-constant while LOB is not.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from .dimensions import REGISTRY, dim_column
from .loader import (
    AS_OF,
    AS_OF_TS,
    CUR_QUARTER,
    SERVICE_PORTFOLIOS,
    facts,
)
from .personas import Principal

Measure = Literal["gp", "revenue"]
MEASURES: tuple[str, ...] = ("gp", "revenue")
DEFAULT_MEASURE: Measure = "gp"

VALUE_COLUMN: dict[str, str] = {"gp": "acv_gp", "revenue": "acv_revenue"}
MEASURE_LABEL: dict[str, str] = {"gp": "ACV GP", "revenue": "ACV Revenue"}
#: The measure as a word in a sentence — "$805K of gross profit", "of revenue".
MEASURE_WORD: dict[str, str] = {"gp": "gross profit", "revenue": "revenue"}

# The Show: Profit / Revenue toggle used to change nothing but the label,
# because every tile and most charts read `acv_gp` by name. The rule now is
# in one place: a plain money aggregate over opportunity lines follows
# `fs.value_column` and describes itself with `fs.measure_label` or
# `fs.measure_word`. Anything measured against the plan stays gross profit
# whichever way the toggle sits — the budget is set in GP and there is no
# revenue plan to read it against — and says "gross profit" on its face so
# the reader knows why it did not move.

#: The client's stated services margin target ("services GP should be around 30%").
SERVICES_GM_TARGET = 30.0


# --------------------------------------------------------------------------- #
# Formatting — the only place a number becomes a string
# --------------------------------------------------------------------------- #


def money(v: float) -> str:
    v = float(v or 0)
    a = abs(v)
    if a >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if a >= 1e3:
        return f"${v / 1e3:,.0f}K"
    return f"${v:,.0f}"


def money1(v: float) -> str:
    """Full precision, for a tile the user will read against a spreadsheet."""
    return f"${float(v or 0):,.0f}"


def pct(v: float, dp: int = 1) -> str:
    return f"{float(v or 0):.{dp}f}%"


def mult(v: float) -> str:
    return f"{float(v or 0):.2f}x"


def count(v: float) -> str:
    return f"{int(v or 0):,}"


# --------------------------------------------------------------------------- #
# Filter state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FilterState:
    """
    The user's current slice, mirrored into the URL so a view is reproducible.

    Identity is deliberately NOT in here. Who you are is the principal's
    business and is resolved at the API boundary; letting a filter carry it
    would make widening your own scope a query-string edit.
    """

    stage: str | None = None
    forecast: str | None = None
    lob: str | None = None
    portfolio: str | None = None
    industry: str | None = None
    order_type: str | None = None
    quarter: str | None = None
    rep: str | None = None
    account: str | None = None
    risk_band: str | None = None
    anomaly_category: str | None = None
    measure: Measure = DEFAULT_MEASURE

    DIMS: tuple[str, ...] = ()

    @classmethod
    def from_query(cls, q: dict) -> "FilterState":
        def g(k: str) -> str | None:
            v = q.get(k)
            v = v.strip() if isinstance(v, str) else None
            return v or None

        m = (q.get("measure") or DEFAULT_MEASURE).lower()
        return cls(
            stage=g("stage"), forecast=g("forecast"), lob=g("lob"),
            portfolio=g("portfolio"), industry=g("industry"),
            order_type=g("orderType") or g("order_type"), quarter=g("quarter"),
            rep=g("rep"), account=g("account"), risk_band=g("riskBand"),
            anomaly_category=g("anomalyCategory"),
            measure=m if m in MEASURES else DEFAULT_MEASURE,
        )

    @property
    def values(self) -> dict[str, str]:
        """Only the dimensions that are actually set."""
        raw = {
            "stage": self.stage, "forecast": self.forecast, "lob": self.lob,
            "portfolio": self.portfolio, "industry": self.industry,
            "orderType": self.order_type, "quarter": self.quarter,
            "rep": self.rep, "account": self.account,
            "riskBand": self.risk_band, "anomalyCategory": self.anomaly_category,
        }
        return {k: v for k, v in raw.items() if v}

    @property
    def is_on(self) -> bool:
        return bool(self.values)

    @property
    def value_column(self) -> str:
        return VALUE_COLUMN[self.measure]

    @property
    def measure_label(self) -> str:
        return MEASURE_LABEL[self.measure]

    @property
    def measure_word(self) -> str:
        return MEASURE_WORD[self.measure]

    def active(self) -> list[dict]:
        return [
            {"dim": k, "label": REGISTRY[k].label if k in REGISTRY else k, "value": v}
            for k, v in self.values.items()
        ]

    def scope_label(self, principal: Principal | None = None) -> str:
        base = principal.identity_label if principal else "North America"
        if not self.is_on:
            return base
        return base + " · " + " · ".join(f"{v}" for v in self.values.values())

    def with_(self, **kw) -> "FilterState":
        return replace(self, **kw)

    def toggled(self, dim: str, value: str) -> "FilterState":
        """Clicking an already-active value clears it, like a pivot slicer."""
        field = {"orderType": "order_type", "riskBand": "risk_band",
                 "anomalyCategory": "anomaly_category"}.get(dim, dim)
        cur = getattr(self, field, None)
        return replace(self, **{field: None if cur == value else value})

    def query_string(self, **override) -> str:
        q = dict(self.values)
        q["measure"] = self.measure
        q.update({k: v for k, v in override.items() if v is not None})
        return "&".join(f"{k}={v}" for k, v in q.items() if v)


# --------------------------------------------------------------------------- #
# The one way to get rows
# --------------------------------------------------------------------------- #


def slice_frame(fs: FilterState, principal: Principal) -> pd.DataFrame:
    """
    RLS first, then the user's filters. Line grain.

    The order is not cosmetic. If the filters ran first the predicate would be
    applied to an already-narrowed frame, which happens to give the same rows —
    but `rls_frame()` below depends on the predicate alone being the denominator,
    and keeping the two in one order is what makes "9.7% of the entity" a true
    statement rather than a coincidence.
    """
    df = principal.apply(facts())
    for dim, value in fs.values.items():
        col = dim_column(dim)
        if col and col in df.columns:
            df = df[df[col] == value]
    return df


def rls_frame(principal: Principal) -> pd.DataFrame:
    """The principal's whole book, unfiltered — the denominator for any share."""
    return principal.apply(facts())


def entity_frame() -> pd.DataFrame:
    """Everything, ignoring RLS. Only for "your book is X% of the entity"."""
    return facts()


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #

SUBSETS: tuple[str, ...] = (
    "all", "open", "won", "lost", "closed", "qualified", "pastdue", "stalled",
)


def subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name in (None, "all"):
        return df
    if name == "open":
        return df[df["is_open"]]
    if name == "won":
        return df[df["is_won"]]
    if name == "lost":
        return df[df["is_lost"]]
    if name == "closed":
        return df[df["is_closed"]]
    if name == "qualified":
        return df[df["is_qualified"]]
    if name == "pastdue":
        return df[df["is_past_due"]]
    if name == "stalled":
        from .movement_features import features

        stalled = set(features().loc[features()["is_stalled"], "opportunity_code"])
        return df[df["opportunity_code"].isin(stalled)]
    raise KeyError(f"unknown subset '{name}'. Known: {SUBSETS}")


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def gm(df: pd.DataFrame) -> float:
    """Revenue-weighted gross margin. The only correct way to aggregate GM."""
    rev = float(df["acv_revenue"].sum())
    return 100.0 * float(df["acv_gp"].sum()) / rev if rev else 0.0


def naive_gm(df: pd.DataFrame) -> float:
    """
    The unweighted mean of the row-level GM column — i.e. the wrong answer.

    Kept so the Context lens can put it next to the right one. On this book the
    two differ by ~7.7 points because 359 VBR lines average 44% margin on $197K
    of revenue and an unweighted mean lets each of them outvote a $1M product line.
    """
    return float(df["gm_percent_row"].mean()) if len(df) else 0.0


def opps(df: pd.DataFrame) -> int:
    return int(df["opportunity_code"].nunique())


def count_on(df: pd.DataFrame, dim: str) -> int:
    """Count on the right grain for this dimension, without the caller knowing."""
    d = REGISTRY.get(dim)
    if d is None or d.count_basis == "lines":
        return int(len(df))
    return opps(df)


def money_block(df: pd.DataFrame) -> dict:
    """Both measures at once. The keys are the measure keys, so a caller
    wanting the active one reads `block[fs.measure]`."""
    return {
        "revenue": float(df["acv_revenue"].sum()),
        "gp": float(df["acv_gp"].sum()),
        "tcvRevenue": float(df["tcv_revenue"].sum()),
        "gm": gm(df),
        "lines": int(len(df)),
        "opps": opps(df),
    }


def by_dimension(df: pd.DataFrame, dim: str, fs: FilterState,
                 top: int | None = None) -> pd.DataFrame:
    """
    Aggregate to one dimension, ordered by the active measure.

    `share` is of the frame passed in, not of the whole book, so a chart inside
    a filtered page reports shares of what the user is actually looking at.
    """
    col = dim_column(dim)
    if not col or col not in df.columns:
        raise KeyError(f"dimension '{dim}' has no column in this frame")
    g = df.groupby(col, dropna=False).agg(
        revenue=("acv_revenue", "sum"),
        gp=("acv_gp", "sum"),
        lines=("line_code", "count"),
        opps=("opportunity_code", "nunique"),
    ).reset_index().rename(columns={col: "key"})
    g["gm"] = np.where(g["revenue"] != 0, 100 * g["gp"] / g["revenue"], 0.0)
    g["value"] = g[fs.measure if fs.measure == "gp" else "revenue"]
    total = g["value"].sum()
    g["share"] = np.where(total != 0, 100 * g["value"] / total, 0.0)
    g = g.sort_values("value", ascending=False).reset_index(drop=True)
    if top and len(g) > top:
        head = g.head(top).copy()
        tail = g.tail(len(g) - top)
        other = pd.DataFrame([{
            "key": f"Other ({len(tail)})",
            "revenue": tail["revenue"].sum(), "gp": tail["gp"].sum(),
            "lines": int(tail["lines"].sum()), "opps": int(tail["opps"].sum()),
            "gm": gm_from(tail), "value": tail["value"].sum(),
            "share": tail["share"].sum(),
        }])
        g = pd.concat([head, other], ignore_index=True)
    return g


def gm_from(g: pd.DataFrame) -> float:
    rev = float(g["revenue"].sum())
    return 100.0 * float(g["gp"].sum()) / rev if rev else 0.0


def trend(df: pd.DataFrame, what: str, col: str = "acv_gp") -> list[float]:
    """
    A genuine monthly series for a KPI, or an empty list when there isn't one.

    `col` is the money column the `open_gp` and `won_gp` series sum — the
    tile passes `fs.value_column`, so a sparkline under a revenue figure is a
    revenue series rather than the gross-profit line it used to be.

    Sparklines are only drawn where a real history exists. Several of these
    measures are SNAPSHOTS — "open pipeline" is a fact about today — so the
    series is reconstructed rather than invented: at each month end, a deal was
    open if it had been created by then and had not yet closed. That is a true
    statement about the past, and it is the whole reason the extract carries
    both dates.

    Where no honest series can be built the function returns `[]` and the tile
    draws no sparkline at all. A flat or fabricated line under a number is worse
    than no line, because it implies a trend was checked.
    """
    if df.empty:
        return []
    months = pd.period_range(
        df["create_date"].min(), min(df["close_date"].max(), AS_OF_TS),
        freq="M",
    )
    if len(months) < 3:
        return []
    out: list[float] = []

    if what in ("open_gp", "open_count"):
        create = df["create_date"]
        close = df["close_date"]
        for m in months:
            edge = m.end_time
            live = (create <= edge) & (close > edge)
            out.append(
                float(df.loc[live, col].sum()) if what == "open_gp"
                else float(df.loc[live, "opportunity_code"].nunique())
            )
        return out

    # Past-due history is NOT reconstructible from this extract and is therefore
    # not offered. A deal is past due when its own committed close date has
    # passed — but that date MOVED over the deal's life (the change log records
    # 48 such slips), and the fact table keeps only the final value. Asking
    # "was it past due in June" against today's close date answers a different
    # question, and an earlier attempt to compute it here was self-contradicting
    # (a deal cannot be simultaneously still-open and already-closed at one
    # month end). The tile shows no sparkline rather than a plausible wrong one.

    if what in ("won_gp", "won_count", "win_rate", "gm"):
        g = df.groupby(df["close_date"].dt.to_period("M"))
        # For a SUM, a month with no closed business really is zero. For a RATIO
        # it is not — "no deals closed" is not "0% margin" — so ratio months with
        # no denominator are dropped rather than plotted at the axis, which would
        # draw a cliff that never happened.
        ratio = what in ("win_rate", "gm")
        for m in months:
            try:
                sub = g.get_group(m)
            except KeyError:
                if not ratio:
                    out.append(0.0)
                continue
            if what == "won_gp":
                out.append(float(sub.loc[sub["is_won"], col].sum()))
            elif what == "won_count":
                out.append(float(sub.loc[sub["is_won"], "opportunity_code"].nunique()))
            elif what == "gm":
                rev = float(sub["acv_revenue"].sum())
                if rev:
                    out.append(100 * float(sub["acv_gp"].sum()) / rev)
            else:
                closed = sub[sub["is_closed"]]
                n = closed["opportunity_code"].nunique()
                if n:
                    w = closed.loc[closed["is_won"], "opportunity_code"].nunique()
                    out.append(100 * w / n)
        # Two points is not a trend.
        return out if len(out) >= 3 else []

    return []


def by_time(df: pd.DataFrame, fs: FilterState, grain: str = "month",
            date_col: str = "close_date") -> pd.DataFrame:
    key = "fiscal_month" if grain == "month" else "fiscal_quarter"
    if date_col != "close_date":
        key = "create_month" if grain == "month" else "create_quarter"
    g = df.groupby(key).agg(
        revenue=("acv_revenue", "sum"), gp=("acv_gp", "sum"),
        lines=("line_code", "count"), opps=("opportunity_code", "nunique"),
    ).reset_index().rename(columns={key: "key"})
    g["value"] = g["gp"] if fs.measure == "gp" else g["revenue"]
    return g.sort_values("key").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The measure bundle every view starts from
# --------------------------------------------------------------------------- #


def measures(fs: FilterState, principal: Principal) -> dict:
    """
    One computation of everything a page might need, for one slice.

    Assembling this once and handing the same dict to the KPI tiles, the
    narrative and the action engine is what guarantees the headline and the
    prose cannot contradict each other — they are reading the same numbers, not
    recomputing them.
    """
    from .movement_features import features

    df = slice_frame(fs, principal)
    book = rls_frame(principal)
    entity = entity_frame()

    open_df = subset(df, "open")
    won_df = subset(df, "won")
    closed_df = subset(df, "closed")
    past_df = subset(df, "pastdue")
    qual_df = subset(df, "qualified")

    mv = features().set_index("opportunity_code")
    open_codes = open_df["opportunity_code"].unique()
    in_scope = mv.reindex(open_codes)
    stalled_codes = in_scope.index[in_scope["is_stalled"].fillna(False)]
    stalled_df = open_df[open_df["opportunity_code"].isin(stalled_codes)]

    svc = df[df["portfolio"].isin(SERVICE_PORTFOLIOS)]
    open_value = float(open_df[fs.value_column].sum())
    book_open_value = float(subset(book, "open")[fs.value_column].sum())
    entity_open_value = float(subset(entity, "open")[fs.value_column].sum())

    won_opps = opps(won_df)
    lost_opps = opps(subset(df, "lost"))

    return {
        "asOf": AS_OF.isoformat(),
        "quarter": CUR_QUARTER,
        "measure": fs.measure,
        "measureLabel": fs.measure_label,
        "total": money_block(df),
        "open": money_block(open_df),
        "won": money_block(won_df),
        "closed": money_block(closed_df),
        "qualified": money_block(qual_df),
        "pastDue": money_block(past_df),
        "stalled": money_block(stalled_df),
        "pastDueShare": (100 * float(past_df[fs.value_column].sum()) / open_value)
        if open_value else 0.0,
        "stalledShare": (100 * float(stalled_df[fs.value_column].sum()) / open_value)
        if open_value else 0.0,
        "winRate": (100 * won_opps / (won_opps + lost_opps)) if (won_opps + lost_opps) else 0.0,
        "winRateBasis": {"won": won_opps, "lost": lost_opps},
        "servicesGm": gm(svc),
        "servicesGmTarget": SERVICES_GM_TARGET,
        "servicesGmGap": gm(svc) - SERVICES_GM_TARGET,
        "blendedGm": gm(df),
        "naiveGm": naive_gm(df),
        "medianQuietDays": float(in_scope["quiet_days"].median())
        if in_scope["quiet_days"].notna().any() else 0.0,
        "shareOfBook": (100 * open_value / book_open_value) if book_open_value else 100.0,
        "shareOfEntity": (100 * open_value / entity_open_value) if entity_open_value else 0.0,
        "entityOpen": entity_open_value,
        "scope": {
            "label": fs.scope_label(principal),
            "predicate": principal.predicate_sql,
            "persona": principal.key,
            "identity": principal.identity_label,
        },
        "filters": fs.values,
    }
