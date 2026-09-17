"""
The transactional layer: read the three client datasets once, canonicalise them,
and derive the columns every downstream module is allowed to assume.

Three datasets, three grains — and keeping them straight is the whole job:

  FACT      one row per **opportunity line** (3,034). Money, LOB and Portfolio
            live here and vary within an opportunity, so anything monetary or
            product-shaped must count lines.
  MOVEMENT  one row per **field change** (36,631) across 2,050 opportunities.
            Stage history, value drift, stall age and every behavioural pattern
            come from here and nowhere else.
  ANOMALY   one row per **finding** (485) from the data-science team, at mixed
            entity grain (opportunity / account / rep / segment / industry).

Opportunity-level attributes (stage, forecast category, dates, owner, account)
repeat across all lines of the same opportunity. Summing a line-grain frame by
stage therefore double-counts opportunities, and counting opportunities by LOB
under-counts lines. Every module downstream states which grain it is on.

Nothing here computes a business measure. This module's contract is: a clean
frame with canonical names, correct dtypes, and derivations that are pure
restatements of what the source already says.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..config import (
    ANOMALIES_CSV,
    AS_OF,
    CROSS_SELL_CSV,
    DISTRIBUTION_CSV,
    FY_START_MONTH,
    MOVEMENT_CSV,
    OPPORTUNITIES_CSV,
)

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: Raw SFDC export name -> the canonical name used everywhere above this module.
#: `ProductBusinessUnit` really is the Line of Business and `ServiceCategory`
#: really is the Portfolio — the client confirmed both on the 11 Sep walkthrough,
#: and carrying the export's names upward would make every downstream module read
#: as though it were about something else.
FACT_COLUMNS: dict[str, str] = {
    "OpportunityGlobalRegionName": "region",
    "OpportunityCountry": "country",
    "AccountCode": "account_code",
    "AccountName": "account_name",
    "AccountGroup": "account_group",
    "AccountIndustry": "industry",
    "OpportunityCode": "opportunity_code",
    "OpportunityLineCode": "line_code",
    "OpportunityName": "opportunity_name",
    "OpportunityStage": "stage",
    "Confidence": "confidence",
    "ForecastCategory": "forecast_category",
    "OpportunityCreateDate": "create_date",
    "OpportunityCloseDate": "close_date",
    "AccountOwnerFullName": "account_owner",
    "AccountOwnerEmailAdress": "account_owner_email",
    "OpportunityOwnerFullName": "owner",
    "OpportunityOwnerEmailAdress": "owner_email",
    "ProductBusinessUnit": "lob",
    "ServiceCategory": "portfolio",
    "OrderType": "order_type",
    "ContractTermMonths": "term_months",
    "SFDC_ACV_Revenue": "acv_revenue",
    "SFDC_ACV_Gp": "acv_gp",
    "SFDC_TCV_Revenue": "tcv_revenue",
    "SFDC_TCV_Gp": "tcv_gp",
    "GM_Percent": "gm_percent_row",
    "Quarter": "quarter",
    "Month": "month",
    "QuarterlyBudget_ACVRevenue": "q_budget_revenue",
    "QuarterlyBudget_ACVGp": "q_budget_gp",
    "QuarterlyGap_ACVGP": "q_gap_gp",
    "QuarterlyTotalPipeline_ACVGP": "q_total_pipeline_gp",
    "TP_RAGStatus": "q_tp_rag",
    "QuarterlyQualifiedPipeline_ACVGP": "q_qualified_pipeline_gp",
    "QP_RAGStatus": "q_qp_rag",
    "LOBPortfolioQuarterlyBudget_ACVRevenue": "cell_q_budget_revenue",
    "LOBPortfolioQuarterlyBudget_ACVGp": "cell_q_budget_gp",
    "MonthlyBudget_ACVRevenue": "m_budget_revenue",
    "MonthlyBudget_ACVGp": "m_budget_gp",
    "MonthlyClosedPlusCommit_ACVGP": "m_closed_commit_gp",
    "ClosedCommit_RAGStatus": "m_cc_rag",
    "LOBPortfolioMonthlyBudget_ACVRevenue": "cell_m_budget_revenue",
    "LOBPortfolioMonthlyBudget_ACVGp": "cell_m_budget_gp",
}

MOVEMENT_COLUMNS: dict[str, str] = {
    "ChangeId": "change_id",
    "OpportunityCode": "opportunity_code",
    "AccountName": "account_name",
    "OpportunityName": "opportunity_name",
    "FieldChanged": "field",
    "OldValue": "old_value",
    "NewValue": "new_value",
    "ChangeDate": "change_date",
    "ChangedByFullName": "changed_by",
    "ChangedByEmail": "changed_by_email",
}

#: The cross-sell export. Unlike the fact table this arrives already analysed,
#: so the canonical names describe FINDINGS rather than transactions.
CROSS_SELL_COLUMNS: dict[str, str] = {
    "RecommendationId": "recommendation_id",
    "AccountCode": "account_code",
    "AccountName": "account_name",
    "Industry": "industry",
    "CurrentProducts": "current_products",
    "Recommendation": "recommendation",
    "Confidence": "confidence",
    "Why": "why",
    "Methods": "methods",
    "NumMethods": "num_methods",
    "AvgWonRevenue": "peer_won_revenue",
    "AvgWonGp": "peer_won_gp",
    "CurrentWonGp": "account_won_gp",
}

ANOMALY_COLUMNS: dict[str, str] = {
    "AnomalyId": "anomaly_id",
    "EntityType": "entity_type",
    "DealId": "entity_id",
    "Deal": "entity_label",
    "RepName": "rep",
    "DealValue": "value_at_stake",
    "AnomalyCategory": "category",
    "AnomalyType": "anomaly_type",
    "Evidence": "evidence",
    "RecommendedAction": "recommended_action",
    "Severity": "severity",
    "DemoPriority": "demo_priority",
}

#: The stage ladder, in the order the client walked through it. The index is the
#: stage's ordinal — `stage_skip` detection, funnel ordering and "moved backwards"
#: all depend on this list and on nothing else.
STAGE_ORDER: tuple[str, ...] = (
    "Identification",
    "Requirements Definition",
    "Qualification",
    "Proposal",
    "Proposal Evaluation",
    "Finalist",
    "Deal Won",
    "Deal Lost",
)
STAGE_RANK: dict[str, int] = {s: i for i, s in enumerate(STAGE_ORDER)}
#: Deal Lost is a terminal state, not a rung above Finalist. Ranking it 7 would
#: make every loss read as forward progress.
STAGE_RANK["Deal Lost"] = STAGE_RANK["Deal Won"]

CLOSED_STAGES = ("Deal Won", "Deal Lost")
OPEN_STAGES = tuple(s for s in STAGE_ORDER if s not in CLOSED_STAGES)

#: Forecast category, ordered by how close to booked the rep is claiming it is.
FORECAST_ORDER: tuple[str, ...] = ("Omitted", "Pipeline", "Best Case", "Commit", "Closed")
FORECAST_RANK: dict[str, int] = {f: i for i, f in enumerate(FORECAST_ORDER)}

LOB_ORDER: tuple[str, ...] = ("Networking", "Security", "Data Center", "Digital Workplace")
PORTFOLIO_ORDER: tuple[str, ...] = (
    "Product",
    "Technical Services",
    "SDIS",
    "VBR",
    "Consulting Services",
)
#: "Services" for margin purposes is everything that is not a one-time product
#: sale. VBR is a reseller margin line, not a service.
SERVICE_PORTFOLIOS: tuple[str, ...] = ("Technical Services", "SDIS", "Consulting Services")

ORDER_TYPES: tuple[str, ...] = ("New Business", "Renewal", "Expansion")


# --------------------------------------------------------------------------- #
# Fiscal calendar
# --------------------------------------------------------------------------- #


def fiscal_year(d: date | pd.Timestamp) -> int:
    """FY26 is Apr 2026 - Mar 2027, so Jan-Mar belongs to the year before."""
    return d.year if d.month >= FY_START_MONTH else d.year - 1


def fiscal_quarter(d: date | pd.Timestamp) -> str:
    fy = fiscal_year(d)
    q = ((d.month - FY_START_MONTH) % 12) // 3 + 1
    return f"FY{fy % 100:02d}-Q{q}"


def fy_label(fy: int) -> str:
    return f"FY{fy % 100:02d}"


AS_OF_TS = pd.Timestamp(AS_OF)
CUR_FY = fiscal_year(AS_OF)
CUR_QUARTER = fiscal_quarter(AS_OF)


# --------------------------------------------------------------------------- #
# Load + derive
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoadReport:
    """What actually came off disk. Surfaced on /healthz and the Context lens."""

    lines: int
    opportunities: int
    accounts: int
    reps: int
    movement_rows: int
    movement_opportunities: int
    anomalies: int
    cross_sell_recommendations: int
    orphan_cross_sell_accounts: int
    orphan_movement_opps: int
    orphan_anomaly_opps: int
    close_before_create: int
    fact_path: str
    movement_path: str
    anomaly_path: str
    cross_sell_path: str


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed")


@functools.lru_cache(maxsize=1)
def facts() -> pd.DataFrame:
    """
    The opportunity-LINE fact table, canonicalised and derived.

    Derivations are restatements, never new business rules: an `is_open` column
    is the client's own "not Won and not Lost" said once, in one place, so that
    twenty call sites cannot each get it subtly different.
    """
    raw = pd.read_csv(OPPORTUNITIES_CSV)
    missing = set(FACT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(
            f"{OPPORTUNITIES_CSV.name} is missing expected columns: {sorted(missing)}. "
            "The export schema changed — update FACT_COLUMNS rather than coercing here."
        )
    df = raw.rename(columns=FACT_COLUMNS)[list(FACT_COLUMNS.values())].copy()

    df["create_date"] = _to_date(df["create_date"])
    df["close_date"] = _to_date(df["close_date"])

    # Stage-derived flags. One definition, one place.
    df["is_won"] = df["stage"].eq("Deal Won")
    df["is_lost"] = df["stage"].eq("Deal Lost")
    df["is_closed"] = df["stage"].isin(CLOSED_STAGES)
    df["is_open"] = ~df["is_closed"]
    # "Qualified" excludes Identification: the client treats an unqualified
    # sniff as pipeline you may not count towards coverage.
    df["is_qualified"] = df["is_open"] & df["stage"].ne("Identification")
    df["stage_rank"] = df["stage"].map(STAGE_RANK).astype("int16")
    df["forecast_rank"] = df["forecast_category"].map(FORECAST_RANK).astype("int16")

    # Ageing, measured from AS_OF rather than today() so a rehearsal and the
    # demo show the same numbers.
    df["age_days"] = (AS_OF_TS - df["create_date"]).dt.days
    df["days_to_close"] = (df["close_date"] - AS_OF_TS).dt.days
    df["cycle_days"] = (df["close_date"] - df["create_date"]).dt.days
    df["is_past_due"] = df["is_open"] & (df["close_date"] < AS_OF_TS)
    df["days_past_due"] = np.where(df["is_past_due"], -df["days_to_close"], 0)

    # Fiscal placement. The export ships `quarter`/`month` already; these are
    # recomputed so a downstream slice never has to trust a denormalised string,
    # and any disagreement shows up in verify rather than on stage.
    df["fiscal_quarter"] = df["close_date"].map(fiscal_quarter)
    df["fiscal_month"] = df["close_date"].dt.strftime("%Y-%m")
    df["fiscal_year"] = df["close_date"].map(fiscal_year)
    df["create_quarter"] = df["create_date"].map(fiscal_quarter)
    df["create_month"] = df["create_date"].dt.strftime("%Y-%m")

    df["is_service"] = df["portfolio"].isin(SERVICE_PORTFOLIOS)
    # GM% is carried per row by the export but is only ever legitimate
    # row-by-row; every aggregate must divide summed GP by summed revenue.
    df["gm_percent_row"] = pd.to_numeric(df["gm_percent_row"], errors="coerce")

    # AccountGroup is blank ~65% of the time by design (the client says the
    # sibling-account de-dup simply is not always done). Name the blank rather
    # than leaving NaN to leak into a group-by as a silent bucket.
    df["account_group"] = df["account_group"].fillna("(ungrouped)")
    df["term_months"] = pd.to_numeric(df["term_months"], errors="coerce")
    df["has_term"] = df["term_months"].notna()

    for col in ("q_tp_rag", "q_qp_rag", "m_cc_rag"):
        df[col] = df[col].astype("string")

    return df


@functools.lru_cache(maxsize=1)
def movement() -> pd.DataFrame:
    """
    The field-level change log, sorted into a replayable timeline.

    `seq` is the within-opportunity ordinal. The export's own ChangeId ordering
    is the generation order and already agrees with ChangeDate, but several
    changes share a date (a rep touches four fields in one save), so ordering by
    date alone is ambiguous and would make stage paths non-deterministic.
    """
    raw = pd.read_csv(MOVEMENT_CSV)
    missing = set(MOVEMENT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{MOVEMENT_CSV.name} is missing columns: {sorted(missing)}")
    df = raw.rename(columns=MOVEMENT_COLUMNS)[list(MOVEMENT_COLUMNS.values())].copy()
    df["change_date"] = _to_date(df["change_date"])
    df["field"] = df["field"].map(
        {
            "OpportunityStage": "stage",
            "ForecastCategory": "forecast_category",
            "Confidence": "confidence",
            "SFDC_ACV_Revenue": "acv_revenue",
            "SFDC_ACV_Gp": "acv_gp",
            "OpportunityCloseDate": "close_date",
        }
    ).fillna(df["field"])
    df = df.sort_values(["opportunity_code", "change_id"], kind="stable").reset_index(drop=True)
    df["seq"] = df.groupby("opportunity_code").cumcount()
    df["days_ago"] = (AS_OF_TS - df["change_date"]).dt.days

    # Numeric view of the value changes, so drift arithmetic never reparses.
    num = df["field"].isin(("acv_revenue", "acv_gp", "confidence"))
    df["old_num"] = np.where(num, pd.to_numeric(df["old_value"], errors="coerce"), np.nan)
    df["new_num"] = np.where(num, pd.to_numeric(df["new_value"], errors="coerce"), np.nan)
    return df


@functools.lru_cache(maxsize=1)
def anomalies_raw() -> pd.DataFrame:
    """The data-science team's findings, canonicalised but not yet enriched."""
    raw = pd.read_csv(ANOMALIES_CSV)
    missing = set(ANOMALY_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{ANOMALIES_CSV.name} is missing columns: {sorted(missing)}")
    df = raw.rename(columns=ANOMALY_COLUMNS)[list(ANOMALY_COLUMNS.values())].copy()
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0).astype(int)
    df["value_at_stake"] = pd.to_numeric(df["value_at_stake"], errors="coerce").fillna(0.0)
    df["demo_priority"] = df["demo_priority"].astype(str).str.lower().isin(("true", "1", "yes"))
    return df


@functools.lru_cache(maxsize=1)
def cross_sell_raw() -> pd.DataFrame:
    """
    The data-science cross-sell export, canonicalised but not yet fused.

    Returned as it arrived apart from the column names and two numeric coercions.
    Everything interpretive — splitting `Add Security / VBR` into its LOB and
    portfolio, ranking the three confidence words, cutting `Why` into one string
    per method — happens in `semantic.crosssell`, so this stays a reader.
    """
    raw = pd.read_csv(CROSS_SELL_CSV)
    missing = set(CROSS_SELL_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{CROSS_SELL_CSV.name} is missing columns: {sorted(missing)}")
    df = raw.rename(columns=CROSS_SELL_COLUMNS)[list(CROSS_SELL_COLUMNS.values())].copy()
    for col in ("num_methods",):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("peer_won_revenue", "peer_won_gp", "account_won_gp"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


@functools.lru_cache(maxsize=1)
def distribution_check() -> pd.DataFrame:
    """
    The generator's own categorical-mix validation, carried through to the UI.

    This is the honest provenance panel: the synthetic data claims to match the
    client's distribution workbook to within 0.1pp, and this is the sheet that
    says so. Showing it is cheaper than being asked whether the data is real.
    """
    # Row 0 is the sheet title and row 1 the methodology note; the real header
    # is row 2. Read it positionally rather than sniffing, so a reworded note
    # cannot silently shift the frame.
    raw = pd.read_csv(DISTRIBUTION_CSV, header=2)
    raw = raw.dropna(how="all", axis=1).dropna(how="all")
    raw.columns = [str(c).strip() for c in raw.columns]
    rename = {
        "Column": "column",
        "Category": "category",
        "Given % (input workbook)": "given_pct",
        "Synthetic % (generated data)": "synthetic_pct",
        "Difference (pp)": "difference_pp",
        "Status": "status",
    }
    raw = raw.rename(columns=rename)
    keep = [c for c in rename.values() if c in raw.columns]
    out = raw[keep].dropna(subset=["category"]).copy()
    for c in ("given_pct", "synthetic_pct", "difference_pp"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["column"] = out["column"].replace(
        {"ProductBusinessUnit": "LOB", "ServiceCategory": "Portfolio",
         "OpportunityStage": "Stage", "ForecastCategory": "Forecast Category"}
    )
    return out.reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def opportunities() -> pd.DataFrame:
    """
    The OPPORTUNITY-grain frame: one row per opportunity code.

    Line-varying fields are aggregated (money summed, LOB/portfolio reduced to a
    primary plus a count), opportunity-constant fields taken once. Anything that
    counts deals — win rate, cycle time, the funnel, the prediction model —
    belongs on this frame, not on `facts()`.
    """
    f = facts()
    const = [
        "region", "country", "account_code", "account_name", "account_group",
        "industry", "opportunity_name", "stage", "confidence", "forecast_category",
        "create_date", "close_date", "account_owner", "owner", "owner_email",
        "order_type", "is_won", "is_lost", "is_closed", "is_open", "is_qualified",
        "stage_rank", "forecast_rank", "age_days", "days_to_close", "cycle_days",
        "is_past_due", "days_past_due", "fiscal_quarter", "fiscal_month",
        "create_quarter", "fiscal_year",
    ]
    agg = {c: "first" for c in const}
    agg.update(
        acv_revenue=("acv_revenue", "sum"),
        acv_gp=("acv_gp", "sum"),
        tcv_revenue=("tcv_revenue", "sum"),
        tcv_gp=("tcv_gp", "sum"),
        line_count=("line_code", "count"),
        lob_count=("lob", "nunique"),
        portfolio_count=("portfolio", "nunique"),
    )
    g = f.groupby("opportunity_code", sort=False)
    out = g.agg(
        **{k: (k, v) for k, v in agg.items() if isinstance(v, str)},
        **{k: v for k, v in agg.items() if isinstance(v, tuple)},
    )

    # Primary LOB/portfolio = the one carrying the most revenue on the deal. A
    # mode would pick by line count and hand a $2k VBR line the same weight as a
    # $400k product line.
    def _primary(col: str) -> pd.Series:
        idx = f.groupby(["opportunity_code", col])["acv_revenue"].sum()
        return idx.groupby(level=0).idxmax().map(lambda t: t[1])

    out["lob"] = _primary("lob")
    out["portfolio"] = _primary("portfolio")
    out["is_multi_lob"] = out["lob_count"] > 1
    out["gm_percent"] = np.where(
        out["acv_revenue"] != 0, 100 * out["acv_gp"] / out["acv_revenue"], np.nan
    )
    return out.reset_index()


@functools.lru_cache(maxsize=1)
def report() -> LoadReport:
    f, m, a = facts(), movement(), anomalies_raw()
    xs = cross_sell_raw()
    opp_codes = set(f["opportunity_code"])
    acct_codes = set(f["account_code"])
    anom_opp = a[a["entity_type"] == "Opportunity"]["entity_id"]
    return LoadReport(
        lines=len(f),
        opportunities=f["opportunity_code"].nunique(),
        accounts=f["account_code"].nunique(),
        reps=f["owner"].nunique(),
        movement_rows=len(m),
        movement_opportunities=m["opportunity_code"].nunique(),
        anomalies=len(a),
        cross_sell_recommendations=len(xs),
        orphan_cross_sell_accounts=int(
            (~xs["account_code"].isin(acct_codes)).groupby(xs["account_code"]).first().sum()),
        orphan_movement_opps=int((~m["opportunity_code"].isin(opp_codes)).groupby(
            m["opportunity_code"]).first().sum()),
        orphan_anomaly_opps=int((~anom_opp.isin(opp_codes)).sum()),
        close_before_create=int((f["close_date"] < f["create_date"]).sum()),
        fact_path=str(OPPORTUNITIES_CSV),
        movement_path=str(MOVEMENT_CSV),
        anomaly_path=str(ANOMALIES_CSV),
        cross_sell_path=str(CROSS_SELL_CSV),
    )


def reset_caches() -> None:
    """Drop every memoised frame. Used by the test harness, never by a request."""
    for fn in (facts, movement, anomalies_raw, cross_sell_raw, opportunities,
               report, distribution_check):
        fn.cache_clear()
