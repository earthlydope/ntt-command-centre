"""
THE CONTEXT LAYER — the machine-readable description of this business that an
LLM is given in place of the data itself.

The client's own architecture, drawn on the 11 Sep call: *"transactional data on
SQL → semantic data where you have your rules, your definitions, your
calculations → an API interface… → your front end."* This module is the part of
the semantic layer that faces the model rather than the screen. It answers, in
one compact document: what entities exist, what every column means in business
terms, what the legal values are, how each measure is defined, which joins are
valid, what the business rules are, and — the part most catalogs omit and most
demos die on — what is NOT true about the data.

Three design decisions, each of which is the reason a naive version fails:

1. **The model never sees rows.** It sees this catalog and a set of pre-computed,
   pre-formatted figures. It answers by emitting a constrained query PLAN, which
   the server validates against this catalog and executes in pandas. There is no
   text-to-SQL, no `eval`, and no path by which user text reaches the dataframe.
   A plan naming a column or a value that is not in here fails validation.

2. **Enumerated values are inlined where the cardinality is small.** Eight stages
   and five portfolios cost a few dozen tokens and remove an entire class of
   failure — the model inventing "Closed Won" when the value is "Deal Won". High
   cardinality columns (357 accounts, 70 reps) carry only their count and a
   handful of examples, and are resolved by lookup at validation time instead.

3. **The caveats are first-class, not a footnote.** `ForecastCategory` perfectly
   predicts the outcome in this extract; the plan does not foot at LOB grain in
   Q3; the anomaly report's `DealValue` is gross profit rather than revenue.
   A model that is not told these things will produce answers that are locally
   correct and globally wrong, and it will produce them confidently.

Budget: the rendered catalog is kept under roughly 6,000 tokens so it fits
comfortably alongside a facts pack in every request without crowding the answer.
"""

from __future__ import annotations

import functools
import json

from . import anomalies as A
from . import crosssell as X
from . import ds_model
from .dimensions import REGISTRY
from .loader import (
    AS_OF,
    CUR_QUARTER,
    FORECAST_ORDER,
    LOB_ORDER,
    ORDER_TYPES,
    PORTFOLIO_ORDER,
    SERVICE_PORTFOLIOS,
    STAGE_ORDER,
    facts,
    opportunities,
    report,
)
from .measures import MEASURES, SERVICES_GM_TARGET, SUBSETS
from .movement_features import STALL_DAYS

CATALOG_VERSION = "2026-09-16.1"


def _examples(col: str, n: int = 4) -> list[str]:
    vals = facts()[col].dropna().unique()
    return [str(v) for v in vals[:n]]


@functools.lru_cache(maxsize=1)
def entities() -> list[dict]:
    """The four tables, their grain, and their keys."""
    r = report()
    return [
        {
            "name": "opportunity_line",
            "grain": "one row per opportunity LINE",
            "rows": r.lines,
            "key": "line_code",
            "purpose": (
                "The fact table. Money, line of business and portfolio live here and "
                "vary between the lines of one opportunity, so anything monetary or "
                "product-shaped must count LINES."
            ),
        },
        {
            "name": "opportunity",
            "grain": "one row per OPPORTUNITY (lines rolled up)",
            "rows": r.opportunities,
            "key": "opportunity_code",
            "purpose": (
                "Stage, forecast category, dates, owner and account are constant across "
                "the lines of an opportunity, so anything that counts DEALS — win rate, "
                "cycle time, the funnel, risk — must use this grain. Money is summed; "
                "`lob` and `portfolio` here mean the one carrying the most revenue."
            ),
        },
        {
            "name": "movement",
            "grain": "one row per FIELD CHANGE",
            "rows": r.movement_rows,
            "key": "change_id",
            "purpose": (
                "The audit trail: how every deal reached its current state. Stage "
                "history, value revisions, forecast changes, close-date slips and "
                "silence. Every behavioural finding in the product comes from here."
            ),
        },
        {
            "name": "anomaly",
            "grain": "one row per FINDING (mixed entity grain)",
            "rows": r.anomalies,
            "key": "anomaly_id",
            "purpose": (
                "The data-science team's detections, plus the ones this layer computes "
                "live from `movement`. Entities are opportunities, accounts, reps, "
                "segments or industries — check `entity_type` before joining."
            ),
        },
        {
            "name": "cross_sell",
            "grain": "one row per ACCOUNT and recommended OFFERING",
            "rows": len(X.recommendations()),
            "key": "recommendation_id",
            "purpose": (
                "The data-science team's growth recommendations. An offering is a line "
                "of business AND a service category together ('Security / Technical "
                "Services'), which is one level finer than the whitespace this layer "
                "computes itself. Upside, never risk: these are reasons to open a "
                "conversation, not deals."
            ),
        },
        {
            "name": "closure_model",
            "grain": "one prediction per opportunity",
            "rows": len(ds_model.predictions()) if ds_model.available() else 0,
            "key": "opportunity_code",
            "purpose": (
                "The data-science team's calibrated win probability, its risk bucket, "
                "and a per-deal SHAP attribution naming the feature that moved that "
                "specific prediction most."
            ),
        },
    ]


@functools.lru_cache(maxsize=1)
def columns() -> list[dict]:
    """Business meaning for every column a query plan may name."""
    f = facts()
    o = opportunities()

    def enum(values: tuple[str, ...]) -> dict:
        return {"values": list(values)}

    def card(col: str, frame=None) -> dict:
        fr = f if frame is None else frame
        return {"distinct": int(fr[col].nunique()), "examples": _examples(col)}

    return [
        {"name": "opportunity_code", "type": "id", "entity": "opportunity",
         "means": "The deal. One opportunity can carry several lines."},
        {"name": "line_code", "type": "id", "entity": "opportunity_line",
         "means": "The offering line within a deal. Primary key of the fact table."},
        {"name": "account_name", "type": "category", "entity": "opportunity",
         "means": "The customer.", **card("account_name")},
        {"name": "account_code", "type": "id", "entity": "opportunity",
         "means": "Customer key. 363 codes against 357 names — group by code, not name."},
        {"name": "industry", "type": "category", "entity": "opportunity",
         "means": "The account's vertical.", **card("industry")},
        {"name": "country", "type": "category", "entity": "opportunity",
         "means": "United States or Canada. Region is always North America.",
         **enum(("United States", "Canada"))},
        {"name": "owner", "type": "category", "entity": "opportunity",
         "means": "The rep who owns the deal.", **card("owner")},
        {"name": "stage", "type": "ordinal", "entity": "opportunity",
         "means": "Position on the sales ladder, in the order listed.",
         **enum(STAGE_ORDER)},
        {"name": "forecast_category", "type": "ordinal", "entity": "opportunity",
         "means": "The rep's stated commitment level.", **enum(FORECAST_ORDER),
         "caveat": "Encodes the outcome in this extract. Safe as a filter, never as a predictor."},
        {"name": "confidence", "type": "number", "entity": "opportunity",
         "means": "The rep's own 0-1 confidence. An assertion, not a model output.",
         "caveat": "Also encodes the outcome: <=0.19 never won, >=0.97 always won."},
        {"name": "lob", "type": "category", "entity": "opportunity_line",
         "means": "Line of business. Varies within an opportunity.", **enum(LOB_ORDER)},
        {"name": "portfolio", "type": "category", "entity": "opportunity_line",
         "means": "Offering type. Varies within an opportunity.", **enum(PORTFOLIO_ORDER)},
        {"name": "order_type", "type": "category", "entity": "opportunity",
         "means": "How the business was won.", **enum(ORDER_TYPES)},
        {"name": "acv_revenue", "type": "money", "entity": "opportunity_line",
         "means": "Annual contract value, revenue. The top line."},
        {"name": "acv_gp", "type": "money", "entity": "opportunity_line",
         "means": "Annual contract value, gross profit. THE DEFAULT MEASURE — the plan "
                  "is set in GP and every target is a GP target."},
        {"name": "tcv_revenue", "type": "money", "entity": "opportunity_line",
         "means": "Total contract value. Equals ACV except on SDIS lines, which are the "
                  "only ones carrying a contract term."},
        {"name": "term_months", "type": "number", "entity": "opportunity_line",
         "means": "Contract length. Null for every portfolio except SDIS — product and "
                  "technical services are one-time sales, which is why TCV = ACV there."},
        {"name": "create_date", "type": "date", "entity": "opportunity",
         "means": "When the deal was logged. Range 2026-01-01 to 2026-08-31."},
        {"name": "close_date", "type": "date", "entity": "opportunity",
         "means": "Expected or actual close. Range 2026-04-01 to 2026-12-31. Always on "
                  "or after create_date."},
        {"name": "fiscal_quarter", "type": "ordinal", "entity": "opportunity",
         "means": "NTT fiscal quarter of the close date. The year starts in April.",
         **enum(("FY26-Q1", "FY26-Q2", "FY26-Q3"))},
        {"name": "fiscal_month", "type": "ordinal", "entity": "opportunity",
         "means": "Calendar month of the close date, YYYY-MM. 2026-04 to 2026-12."},
        {"name": "quiet_days", "type": "number", "entity": "movement",
         "means": f"Days since this deal's last logged change. >= {STALL_DAYS} is stalled."},
        {"name": "value_drift", "type": "number", "entity": "movement",
         "means": "Last logged value over first non-zero logged value, minus one. "
                  "Negative means the deal shrank since it was logged."},
        {"name": "skipped_stages", "type": "number", "entity": "movement",
         "means": "How many ladder rungs the recorded history jumped over."},
        {"name": "risk_score", "type": "number", "entity": "opportunity",
         "means": "0-100, computed from observable facts (stall, past due, slip, "
                  "shrinkage, skipped stages, owner pattern). Open deals only."},
        {"name": "p_win", "type": "number", "entity": "closure_model",
         "means": "Calibrated win probability from the data-science model.",
         "caveat": "Test AUC 0.595. A ranking signal, not a forecast."},
        {"name": "severity", "type": "number", "entity": "anomaly",
         "means": "0-100 from the detection model."},
        {"name": "value_at_stake", "type": "money", "entity": "anomaly",
         "means": "The money attached to the finding.",
         "caveat": "This is GROSS PROFIT, not revenue. Comparing it against a revenue "
                   "chart will look roughly six times off."},
    ]


@functools.lru_cache(maxsize=1)
def measures_catalog() -> list[dict]:
    """Every named measure, its formula in words, and its formula in pandas."""
    return [
        {"name": "gp", "label": "ACV GP", "formula": "SUM(acv_gp)",
         "pandas": "df['acv_gp'].sum()", "default": True,
         "note": "The default measure. Budget, targets and severity are all in GP."},
        {"name": "revenue", "label": "ACV Revenue", "formula": "SUM(acv_revenue)",
         "pandas": "df['acv_revenue'].sum()"},
        {"name": "gm", "label": "Gross margin %", "formula": "100 * SUM(acv_gp) / SUM(acv_revenue)",
         "pandas": "100 * df['acv_gp'].sum() / df['acv_revenue'].sum()",
         "note": "REVENUE-WEIGHTED, never the average of a margin column. The unweighted "
                 "mean reads about 4 points higher on this book because hundreds of tiny "
                 "high-margin VBR lines outvote the large product lines."},
        {"name": "count", "label": "Count", "formula": "COUNT of rows at the dimension's own grain",
         "pandas": "len(df) for line-grain dims; df['opportunity_code'].nunique() for opportunity-grain",
         "note": "Grain is decided by the dimension, not by the caller. Stage and forecast "
                 "category count opportunities; LOB and portfolio count lines."},
        {"name": "winrate", "label": "Win rate",
         "formula": "won opportunities / (won + lost) opportunities",
         "pandas": "closed.groupby(k)['is_won'].mean()",
         "note": "Opportunity grain only. Entity rate is 37.6% on 1,815 closed deals."},
        {"name": "coverage", "label": "Coverage",
         "formula": "open GP / (plan GP - won GP)",
         "pandas": "open_gp / max(budget_gp - won_gp, 0)",
         "note": "Against REMAINING plan, not the full-year plan."},
        {"name": "cycle", "label": "Cycle days", "formula": "close_date - create_date",
         "pandas": "(df['close_date'] - df['create_date']).dt.days",
         "note": "Median 60 days on closed deals."},
        {"name": "risk", "label": "Deal risk score", "formula": "sum of firing risk factor weights, capped at 100",
         "pandas": "predict.risk_table()['risk_score']", "note": "Open deals only."},
        {"name": "quietdays", "label": "Days since last change",
         "formula": "AS_OF - MAX(movement.change_date) for that opportunity",
         "pandas": "movement_features.features()['quiet_days']"},
    ]


@functools.lru_cache(maxsize=1)
def rules() -> list[dict]:
    """The business definitions everything downstream depends on."""
    return [
        {"rule": "as_of", "statement": f"Every figure is computed as of {AS_OF.isoformat()}, "
                                       f"never today(). Current quarter is {CUR_QUARTER}."},
        {"rule": "fiscal_year", "statement": "NTT's fiscal year starts in April. FY26 is "
                                             "Apr 2026 to Mar 2027, so FY26-Q1 is Apr-Jun."},
        {"rule": "open", "statement": "open = stage NOT IN ('Deal Won', 'Deal Lost')."},
        {"rule": "qualified", "statement": "qualified = open AND stage <> 'Identification'."},
        {"rule": "past_due", "statement": "past_due = open AND close_date < as_of."},
        {"rule": "stalled", "statement": f"stalled = open AND no logged field change in "
                                         f"{STALL_DAYS}+ days."},
        {"rule": "services", "statement": f"Services = {', '.join(SERVICE_PORTFOLIOS)}. "
                                          f"Product and VBR are not services — VBR is a "
                                          f"reseller margin line. Services GM target is "
                                          f"{SERVICES_GM_TARGET:.0f}%."},
        {"rule": "gm_weighting", "statement": "A margin is ALWAYS SUM(GP)/SUM(revenue). Never "
                                              "average a percentage column."},
        {"rule": "grain", "statement": "Stage, forecast category, order type, dates, owner and "
                                       "account are constant within an opportunity and count "
                                       "opportunities. LOB and portfolio vary within one and "
                                       "count lines."},
        {"rule": "default_measure", "statement": "GP is the default. Revenue is the toggle."},
    ]


@functools.lru_cache(maxsize=1)
def caveats() -> list[dict]:
    """
    What is NOT true about this data.

    These exist because each one has already produced a confidently wrong answer
    in testing. A model told none of them will reproduce all of them.
    """
    return [
        {
            "id": "cross_sell_resolution",
            "severity": "important",
            "statement": (
                "There are TWO cross-sell views and they are not the same number. The "
                "data-science export works at line-of-business AND service-category "
                "resolution; this layer's own whitespace works at line-of-business "
                "resolution only. An account can have zero whitespace and several "
                "recommendations at the same time — that is correct, not a "
                "contradiction. Never add the two estimates together."
            ),
        },
        {
            "id": "cross_sell_peer_value",
            "severity": "important",
            "statement": (
                "The money on a cross-sell recommendation is what PEER accounts earn on "
                "that offering, not what this account would pay. Describe it as an order "
                "of magnitude or as peer context. Never call it pipeline, forecast, or "
                "expected revenue."
            ),
        },
        {
            "id": "outcome_leakage",
            "severity": "critical",
            "statement": (
                "forecast_category and confidence encode the outcome. Every 'Closed' "
                "opportunity won and every 'Omitted' one lost. Never describe either as "
                "predictive, and never explain a win rate by them — it is a tautology."
            ),
        },
        {
            "id": "model_is_weak",
            "severity": "high",
            "statement": (
                "The closure model's test AUC is 0.595 against 0.50 for chance. Present "
                "p_win as a ranking signal. Do not say a deal 'will' or 'will not' close."
            ),
        },
        {
            "id": "future_quarter",
            "severity": "high",
            "statement": (
                "FY26-Q3 (Oct-Dec) has not happened yet. Its zero attainment and red RAG "
                "are a calendar position, not a performance finding. The mirror of this "
                "is that Q1 and Q2 read 149% and 170% of plan, because the annual plan "
                "was rolled evenly across three quarters while every win sits in Q1-Q2."
            ),
        },
        {
            "id": "budget_footing",
            "severity": "medium",
            "statement": (
                "Plan cells are denormalised onto opportunity rows, so a cell exists only "
                "where a line happens to sit in that quarter/LOB/portfolio. 52 of 60 "
                "quarter cells and 136 of 180 month cells are present; the Q3 LOB "
                "breakdown is 8.8% short of the Q3 headline and December is 23.9% short. "
                "The entity figure is authoritative; a breakdown must state its residual."
            ),
        },
        {
            "id": "anomaly_value_basis",
            "severity": "medium",
            "statement": (
                "value_at_stake in the anomaly table is gross profit, not revenue. Do not "
                "compare it to a revenue chart."
            ),
        },
        {
            "id": "concentration",
            "severity": "medium",
            "statement": (
                "One account holds 14.5% of all gross profit and the top five hold 39%. "
                "Any 'typical account' statement must use a median, and any share-of-total "
                "chart must group the tail."
            ),
        },
        {
            "id": "account_group_blank",
            "severity": "low",
            "statement": (
                "account_group is blank for about 65% of rows by design — the client says "
                "the sibling-account de-duplication is simply not always done. Shown as "
                "'(ungrouped)'. It is not a data quality failure."
            ),
        },
        {
            "id": "synthetic",
            "severity": "low",
            "statement": (
                "This is synthetic data built to match the client's own distribution "
                "workbook (every category within 0.1 percentage points). Company names "
                "are invented. Never present a finding as though it were about a real "
                "customer relationship."
            ),
        },
        {
            "id": "rep_skill",
            "severity": "medium",
            "statement": (
                "Win rate does not vary meaningfully by rep, LOB, portfolio or order type "
                "in this extract — the spread is 31% to 44% and the outcome was assigned "
                "independently of these attributes. Behavioural patterns (shrinkage, "
                "forecast reversal, stalling) ARE real and repeat by person; outcome "
                "skill is not. Coach on behaviour, never on win rate."
            ),
        },
    ]


@functools.lru_cache(maxsize=1)
def joins() -> list[dict]:
    return [
        {"from": "opportunity_line", "to": "opportunity", "on": "opportunity_code",
         "kind": "many-to-one"},
        {"from": "movement", "to": "opportunity", "on": "opportunity_code",
         "kind": "many-to-one", "note": "All 2,050 opportunities are covered; no orphans."},
        {"from": "anomaly", "to": "opportunity", "on": "entity_id = opportunity_code",
         "kind": "many-to-one", "note": "Only where entity_type = 'Opportunity'. Account, "
                                        "Rep, Segment and Industry findings resolve through "
                                        "their own keys."},
        {"from": "closure_model", "to": "opportunity", "on": "opportunity_code",
         "kind": "one-to-one", "note": "Complete coverage of this book's 2,050 opportunities."},
    ]


@functools.lru_cache(maxsize=1)
def build() -> dict:
    """The whole catalog, as one JSON-serialisable object."""
    r = report()
    return {
        "version": CATALOG_VERSION,
        "subject": (
            "NTT DATA Global — North America sales pipeline. Synthetic extract built to "
            "the client's own distribution workbook."
        ),
        "asOf": AS_OF.isoformat(),
        "currentQuarter": CUR_QUARTER,
        "scale": {
            "lines": r.lines, "opportunities": r.opportunities,
            "accounts": r.accounts, "reps": r.reps,
            "movementRows": r.movement_rows, "anomalies": r.anomalies,
        },
        "entities": entities(),
        "columns": columns(),
        "measures": measures_catalog(),
        "subsets": list(SUBSETS),
        "dimensions": [
            {"key": d.key, "label": d.label, "column": d.column,
             "countBasis": d.count_basis, "validated": d.validated}
            for d in REGISTRY.values()
        ],
        "rules": rules(),
        "joins": joins(),
        "caveats": caveats(),
        "anomalyTaxonomy": {
            "categories": [
                {"name": c, "question": A.CATEGORY_BLURB[c]} for c in A.CATEGORY_ORDER
            ],
            "types": [
                {"type": t, "means": m,
                 "upside": t in A.OPPORTUNITY_TYPES,
                 "personas": list(A.ROUTING.get(t, ()))}
                for t, m in A.TYPE_MEANING.items()
            ],
        },
        "crossSell": {
            "recommendations": len(X.recommendations()),
            "accounts": int(X.recommendations()["account_code"].nunique()),
            "confidence": [
                {"value": c, "means": m, "rank": X.CONFIDENCE_RANK[c]}
                for c, m in X.CONFIDENCE_MEANING.items()
            ],
            "methods": [{"name": k, "means": v} for k, v in X.METHOD_MEANING.items()],
            "framing": X.CAVEAT,
        },
        "measureVocabulary": list(MEASURES),
    }


@functools.lru_cache(maxsize=1)
def render() -> str:
    """
    The catalog as the string handed to the model.

    JSON rather than prose: it is denser per token, it is unambiguous about
    structure, and the model is being asked to emit JSON back.
    """
    return json.dumps(build(), separators=(",", ":"), default=str)


def size_report() -> dict:
    """Token budget check. Roughly four characters per token."""
    s = render()
    return {
        "characters": len(s),
        "approxTokens": len(s) // 4,
        "budget": 6000,
        "withinBudget": len(s) // 4 <= 6000,
    }


def reset_caches() -> None:
    for fn in (entities, columns, measures_catalog, rules, caveats, joins,
               build, render):
        fn.cache_clear()
