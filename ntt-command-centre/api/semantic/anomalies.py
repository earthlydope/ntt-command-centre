"""
Anomaly intelligence: the data-science team's findings, enriched, triaged and
routed — plus the subset this layer recomputes itself from the movement log.

Two provenances live side by side here, and the distinction is shown on screen
rather than blurred:

  `ds-model`  the 485 findings in Client_Anomaly_Report.csv, produced by the
              data-science team's own detectors (rule, adaptive-threshold and a
              multivariate ML pass). Replayed, not recomputed — several of them
              (the ML outlier, the peer-IQR fences) are the DS team's model and
              re-deriving them here would be inventing a second answer to a
              question they already own.
  `live`      types this layer derives directly from the 36,631-row change log
              at request time. These exist so the capability is demonstrably
              live rather than a CSV being read back, and so the platform keeps
              working the moment the movement data changes.

Where both fire on the same opportunity they are reconciled rather than listed
twice, and the card says both agreed — which is the strongest form the finding
can take.

The card contract is the client's own, from the 15 Sep call: *what* it is, *why*
it fired (evidence), *how severe*, *what value* is at stake, *who* owns it, and
the *recommended action*.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .loader import anomalies_raw, facts, opportunities
from .movement_features import STALL_DAYS, features, rep_behaviour

# --------------------------------------------------------------------------- #
# The taxonomy — Anomaly_Reference_Guide.docx, carried verbatim in meaning
# --------------------------------------------------------------------------- #

CATEGORY_ORDER: tuple[str, ...] = (
    "Sequence Integrity",
    "Pipeline Coverage",
    "Value Integrity",
    "Deal Governance",
    "Rep Behavior",
    "Concentration & Cross-Sell",
    "Multivariate ML",
)

CATEGORY_BLURB: dict[str, str] = {
    "Sequence Integrity": "Does the deal's own history make logical sense?",
    "Pipeline Coverage": "Is the pipeline healthy, active and large enough?",
    "Value Integrity": "Can the dollar figures on the deal be trusted?",
    "Deal Governance": "Was the number arrived at properly?",
    "Rep Behavior": "Does one person repeat a pattern their peers do not?",
    "Concentration & Cross-Sell": "Where are we over-exposed, and where is there room to sell more?",
    "Multivariate ML": "What looks wrong only when many signals are read together?",
}

#: Not every finding is a problem. The guide is explicit that Cross-Sell items
#: are upside, and presenting them in a red band alongside a stalled deal would
#: misread them to leadership.
OPPORTUNITY_TYPES: frozenset[str] = frozenset(
    {"whitespace_single_lob_account", "portfolio_mix_imbalance", "rep_sandbagging_rate"}
)

TYPE_MEANING: dict[str, str] = {
    "stage_skip": "The recorded history jumps from an early stage to a much later one, missing the steps between.",
    "forecast_regression": "Confidence reached a high point and later fell back, rather than only building.",
    "stalled_pipeline": "An open deal with no activity logged in a long time, still counted as active pipeline.",
    "sales_cycle_outlier": "The deal has taken considerably longer than the standard expected time.",
    "coverage_hole": "A business line and offering combination with a target but very little pipeline behind it.",
    "win_rate_outlier": "A rep, line or offering winning at a rate well below what is typical elsewhere.",
    "top_deal_dependency": "A large share of the quarter's target rests on a handful of very large deals.",
    "value_shrinkage": "The deal was logged considerably higher than what it eventually settled at.",
    "value_inflation": "The deal grew significantly in value compared with when it was first logged.",
    "deal_size_outlier": "The value is unusually large or small compared with similar deals of the same type.",
    "margin_outlier": "The margin is unusually high or low compared with similar deals of the same type.",
    "won_loss_making_deal": "The deal was won but generates zero or negative profit.",
    "quarter_end_clustering": "More deals close at the very end of a quarter than a steady pattern would produce.",
    "flash_close": "The deal moved from open to won implausibly fast for its type.",
    "rep_forecast_reliability": "This rep's deals show the confidence-reversal pattern more often than their peers.",
    "rep_value_shrink_rate": "This rep's deals shrink in value more often than their peers.",
    "rep_sandbagging_rate": "This rep's deals grow well beyond the initial estimate more often than their peers.",
    "rep_concentration_risk": "One rep holds an unusually large share of total open pipeline.",
    "new_business_mix_risk": "This rep's deals are almost entirely renewals, with very little new business.",
    "industry_concentration_risk": "An unusually large share of the business sits in a single industry.",
    "account_concentration_risk": "An unusually large share of the business sits in a single account.",
    "whitespace_single_lob_account": "A large account buys from only one business line, where similar accounts buy several.",
    "portfolio_mix_imbalance": "A large account's spend is skewed to one offering type against the target mix.",
    "ml_multivariate_outlier": "The deal does not resemble its peers once many signals are read together.",
}

#: Which persona each type is FOR. A stalled deal is the owner's to chase; a rep
#: pattern is the manager's to coach; a concentration risk is leadership's to
#: decide on. Routing every finding to everyone is the same as routing it to
#: nobody, and is why anomaly consoles get ignored.
ROUTING: dict[str, tuple[str, ...]] = {
    "stage_skip": ("ae", "manager"),
    "forecast_regression": ("ae", "manager"),
    "stalled_pipeline": ("ae", "manager"),
    "sales_cycle_outlier": ("ae", "manager"),
    "coverage_hole": ("manager", "executive"),
    "win_rate_outlier": ("manager", "executive"),
    "top_deal_dependency": ("manager", "executive"),
    "value_shrinkage": ("ae", "manager"),
    "value_inflation": ("ae", "manager"),
    "deal_size_outlier": ("ae", "manager"),
    "margin_outlier": ("ae", "manager"),
    "won_loss_making_deal": ("manager", "executive"),
    "quarter_end_clustering": ("manager", "executive"),
    "flash_close": ("manager",),
    "rep_forecast_reliability": ("manager",),
    "rep_value_shrink_rate": ("manager",),
    "rep_sandbagging_rate": ("manager",),
    "rep_concentration_risk": ("manager", "executive"),
    "new_business_mix_risk": ("manager",),
    "industry_concentration_risk": ("executive",),
    "account_concentration_risk": ("executive", "manager"),
    "whitespace_single_lob_account": ("ae", "manager", "executive"),
    "portfolio_mix_imbalance": ("ae", "manager", "executive"),
    "ml_multivariate_outlier": ("manager",),
}


# --------------------------------------------------------------------------- #
# Enrichment
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def enriched() -> pd.DataFrame:
    """
    Every DS finding joined back to the book, so a card can carry the deal's own
    LOB, portfolio, owner, stage and value rather than just an id and a sentence.

    Findings at account, rep, segment and industry grain cannot join on an
    opportunity code, so each grain is resolved through its own key and the ones
    that resolve to nothing are kept and marked — a finding the product cannot
    contextualise is still a finding, and silently dropping it would understate
    the count the DS team reported.
    """
    a = anomalies_raw().copy()
    o = opportunities().set_index("opportunity_code")

    opp_cols = ["account_name", "account_code", "industry", "lob", "portfolio",
                "owner", "stage", "acv_revenue", "acv_gp", "close_date",
                "is_open", "order_type", "opportunity_name"]
    # Build each enrichment column as its own object-dtype Series and assign
    # once. Pre-seeding with NaN would fix the column to float64, and pandas 3
    # refuses the later string write rather than silently upcasting.
    is_opp = a["entity_type"].eq("Opportunity")
    is_acc = a["entity_type"].eq("Account")
    is_rep = a["entity_type"].eq("Rep")

    acc = facts().groupby("account_code").agg(
        account_name=("account_name", "first"), industry=("industry", "first"),
        acv_gp=("acv_gp", "sum"), acv_revenue=("acv_revenue", "sum"),
        owner=("owner", "first"),
    )

    def _lookup(mask: pd.Series, table: pd.DataFrame, col: str) -> pd.Series:
        if col not in table.columns:
            return pd.Series(None, index=a.index, dtype=object)
        ids = a["entity_id"].where(mask)
        return ids.map(
            lambda c: table.at[c, col] if isinstance(c, str) and c in table.index else None
        ).astype(object)

    for col in opp_cols:
        vals = _lookup(is_opp, o, col)
        if col in acc.columns:
            vals = vals.combine_first(_lookup(is_acc, acc, col))
        a[col] = vals

    # Rep-grain findings carry the rep in `entity_label` already.
    a["owner"] = a["owner"].where(~is_rep, a["entity_label"])

    a["resolved"] = a[["lob", "owner", "account_name"]].notna().any(axis=1)
    a["provenance"] = "ds-model"
    a["is_opportunity_grain"] = is_opp
    a["meaning"] = a["anomaly_type"].map(TYPE_MEANING).fillna("")
    a["framing"] = np.where(a["anomaly_type"].isin(OPPORTUNITY_TYPES), "opportunity", "risk")
    a["personas"] = a["anomaly_type"].map(lambda t: list(ROUTING.get(t, ("manager",))))
    a["category_question"] = a["category"].map(CATEGORY_BLURB).fillna("")

    # --- triage ---------------------------------------------------------- #
    # Severity alone ranks a $200 finding above a $200k one whenever its reading
    # is more extreme. Triage multiplies the DS severity by how much money moves
    # with it, so the top of the list is where attention actually pays.
    v = a["value_at_stake"].abs()
    a["value_rank"] = v.rank(pct=True)
    a["triage"] = (0.6 * a["severity"] / 100 + 0.4 * a["value_rank"]) * 100
    # An upside finding competes on its own list, not against a stalled deal.
    a.loc[a["framing"] == "opportunity", "triage"] *= 0.9
    a["priority"] = pd.cut(
        a["triage"], [-1, 40, 60, 80, 101], labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)
    return a.sort_values("triage", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Live detection — recomputed from the movement log on every load
# --------------------------------------------------------------------------- #

#: The types this layer derives itself. Everything not listed is replayed from
#: the DS export, and the card says which.
LIVE_TYPES: tuple[str, ...] = (
    "stalled_pipeline", "stage_skip", "forecast_regression",
    "value_shrinkage", "value_inflation", "rep_value_shrink_rate",
    "rep_sandbagging_rate", "rep_forecast_reliability", "rep_concentration_risk",
    "new_business_mix_risk", "close_date_slip",
)


@functools.lru_cache(maxsize=1)
def live() -> pd.DataFrame:
    """
    Anomalies computed here, now, from the change log.

    Thresholds are the client's own where they gave one (60 days quiet), and a
    peer z-score where they did not — because the guide is explicit that most
    checks should compare each deal or rep "against what is typical for its own
    peer group" rather than against a fixed rule.
    """
    o = opportunities().set_index("opportunity_code")
    # `features()` carries its own `is_open` (resolved against the fact table);
    # drop it before the join so the fact table stays the single source of truth
    # for what is open, rather than two columns disagreeing under a suffix.
    mv = features().set_index("opportunity_code").drop(columns=["is_open"], errors="ignore")
    j = mv.join(o[["opportunity_name", "account_name", "owner", "lob", "portfolio",
                   "stage", "acv_gp", "acv_revenue", "is_open", "order_type"]], how="left")
    rows: list[dict] = []

    def add(code, typ, cat, evidence, action, severity, value, **extra):
        r = j.loc[code] if code in j.index else None
        rows.append({
            "anomaly_id": f"LIVE-{typ}-{code}",
            "entity_type": extra.pop("entity_type", "Opportunity"),
            "entity_id": code,
            "entity_label": (r["opportunity_name"] if r is not None else code),
            "anomaly_type": typ, "category": cat,
            "evidence": evidence, "recommended_action": action,
            "severity": int(severity), "value_at_stake": float(value),
            "owner": (r["owner"] if r is not None else extra.pop("owner", None)),
            "lob": (r["lob"] if r is not None else None),
            "portfolio": (r["portfolio"] if r is not None else None),
            "account_name": (r["account_name"] if r is not None else None),
            "stage": (r["stage"] if r is not None else None),
            "acv_gp": (float(r["acv_gp"]) if r is not None else 0.0),
            **extra,
        })

    open_j = j.loc[j["is_open"].fillna(False)]

    for code, r in open_j.loc[open_j["quiet_days"] >= STALL_DAYS].iterrows():
        add(code, "stalled_pipeline", "Pipeline Coverage",
            f"No field changed in {int(r['quiet_days'])} days; still counted as open pipeline "
            f"at {r['stage']}.",
            "The owner should confirm the real status, update it, or close it out.",
            min(40 + r["quiet_days"] / 4, 95), r["acv_gp"])

    for code, r in j.loc[j["skipped_stages"] > 0].iterrows():
        add(code, "stage_skip", "Sequence Integrity",
            f"{int(r['skipped_stages'])} stage(s) skipped — logged path was {r['stage_path_str']}.",
            "Confirm the deal was genuinely worked through the stages it skipped.",
            min(40 + 12 * r["skipped_stages"], 90), r["acv_gp"])

    for code, r in open_j.loc[open_j["forecast_regressions"] > 0].iterrows():
        add(code, "forecast_regression", "Sequence Integrity",
            f"Forecast category walked backwards {int(r['forecast_regressions'])}x; "
            f"confidence is {r['confidence_drawdown']:.2f} below its peak.",
            "Treat the current forecast on this deal with reduced confidence and ask for a "
            "re-qualification.",
            min(45 + 15 * r["forecast_regressions"], 88), r["acv_gp"])

    for code, r in open_j.loc[open_j["value_drift"] < -0.30].iterrows():
        add(code, "value_shrinkage", "Value Integrity",
            f"Logged at {r['value_first']:,.0f} and now {r['value_last']:,.0f} — "
            f"down {abs(r['value_drift']) * 100:.0f}% over {int(r['value_changes'])} revisions.",
            "Review how this deal was sized and qualified at the outset.",
            min(40 + abs(r["value_drift"]) * 60, 90), r["acv_gp"])

    for code, r in open_j.loc[open_j["value_drift"] > 0.40].iterrows():
        add(code, "value_inflation", "Value Integrity",
            f"Grown {r['value_drift'] * 100:.0f}% from {r['value_first']:,.0f} to "
            f"{r['value_last']:,.0f}.",
            "Confirm the growth is a genuine scope increase rather than an under-scoped start.",
            min(38 + r["value_drift"] * 40, 85), r["acv_gp"])

    for code, r in open_j.loc[open_j["close_date_slips"] > 0].iterrows():
        add(code, "close_date_slip", "Deal Governance",
            f"Close date moved {int(r['close_date_slips'])}x, {int(r['slip_days'])} days later "
            f"in total.",
            "Confirm the new date is committed rather than the deal drifting.",
            min(40 + r["slip_days"] / 3, 88), r["acv_gp"])

    # --- rep-level patterns ------------------------------------------------ #
    reps = rep_behaviour()
    credible = reps.loc[~reps["thin"]]
    open_gp_total = j.loc[j["is_open"].fillna(False), "acv_gp"].sum()
    rep_specs = [
        ("shrink_rate", "rep_value_shrink_rate",
         "Review this rep's opportunity-qualification and sizing process."),
        ("inflate_rate", "rep_sandbagging_rate",
         "Coach on early sizing accuracy — this rep may be under-forecasting consistently."),
        ("regression_rate", "rep_forecast_reliability",
         "Discount this rep's confidence ratings and review their forecasting habits directly."),
    ]
    for col, typ, action in rep_specs:
        for _, r in credible.loc[credible[f"{col}_z"] >= 1.5].iterrows():
            rows.append({
                "anomaly_id": f"LIVE-{typ}-{r['rep']}", "entity_type": "Rep",
                "entity_id": r["rep"], "entity_label": r["rep"],
                "anomaly_type": typ, "category": "Rep Behavior",
                "evidence": (f"{r[col] * 100:.0f}% of {int(r['deals'])} owned deals, against a "
                             f"{r[f'{col}_peer'] * 100:.0f}% peer norm "
                             f"({r[f'{col}_z']:+.1f} SD)."),
                "recommended_action": action,
                "severity": int(min(60 + 10 * r[f"{col}_z"], 96)),
                "value_at_stake": float(r["open_gp"]),
                "owner": r["rep"], "lob": None, "portfolio": None,
                "account_name": None, "stage": None, "acv_gp": float(r["open_gp"]),
            })

    for _, r in credible.loc[credible["open_gp"] > 0.12 * max(open_gp_total, 1)].iterrows():
        rows.append({
            "anomaly_id": f"LIVE-rep_concentration_risk-{r['rep']}", "entity_type": "Rep",
            "entity_id": r["rep"], "entity_label": r["rep"],
            "anomaly_type": "rep_concentration_risk", "category": "Rep Behavior",
            "evidence": (f"Holds {100 * r['open_gp'] / open_gp_total:.0f}% of North America's "
                         f"total open pipeline GP across {int(r['open_deals'])} deals."),
            "recommended_action": "Succession and coverage planning — key-person risk if this rep "
                                  "is unavailable.",
            "severity": int(min(80 + 100 * r["open_gp"] / open_gp_total, 98)),
            "value_at_stake": float(r["open_gp"]), "owner": r["rep"],
            "lob": None, "portfolio": None, "account_name": None, "stage": None,
            "acv_gp": float(r["open_gp"]),
        })

    for _, r in credible.loc[credible["new_business_share"] <= 0.22].iterrows():
        rows.append({
            "anomaly_id": f"LIVE-new_business_mix_risk-{r['rep']}", "entity_type": "Rep",
            "entity_id": r["rep"], "entity_label": r["rep"],
            "anomaly_type": "new_business_mix_risk", "category": "Rep Behavior",
            "evidence": (f"Only {r['new_business_share'] * 100:.0f}% of {int(r['deals'])} owned "
                         f"opportunities are New Business (peer norm "
                         f"{r['new_business_share_peer'] * 100:.0f}%)."),
            "recommended_action": "Check pipeline-generation activity — over-reliance on renewals "
                                  "is a growth risk that surfaces too late to correct.",
            "severity": int(min(70 + (0.22 - r["new_business_share"]) * 100, 95)),
            "value_at_stake": float(r["open_gp"]), "owner": r["rep"],
            "lob": None, "portfolio": None, "account_name": None, "stage": None,
            "acv_gp": float(r["open_gp"]),
        })

    if not rows:
        return pd.DataFrame(columns=["anomaly_id", "anomaly_type", "category"])
    df = pd.DataFrame(rows)
    df["provenance"] = "live"
    df["meaning"] = df["anomaly_type"].map(TYPE_MEANING).fillna(
        "The close date has moved materially since the deal was logged.")
    df["framing"] = np.where(df["anomaly_type"].isin(OPPORTUNITY_TYPES), "opportunity", "risk")
    df["personas"] = df["anomaly_type"].map(lambda t: list(ROUTING.get(t, ("manager",))))
    df["category_question"] = df["category"].map(CATEGORY_BLURB).fillna("")
    df["resolved"] = True
    df["is_opportunity_grain"] = df["entity_type"].eq("Opportunity")
    df["demo_priority"] = False
    v = df["value_at_stake"].abs()
    df["value_rank"] = v.rank(pct=True)
    df["triage"] = (0.6 * df["severity"] / 100 + 0.4 * df["value_rank"]) * 100
    df["priority"] = pd.cut(
        df["triage"], [-1, 40, 60, 80, 101], labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)
    return df.sort_values("triage", ascending=False).reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def unified() -> pd.DataFrame:
    """
    Both provenances in one frame, with agreement between them marked.

    When the DS export and this layer independently flag the same opportunity for
    the same reason, that is the strongest a finding gets, and the card says so —
    `corroborated` is the flag the UI reads to promote it.
    """
    ds, lv = enriched(), live()
    cols = ["anomaly_id", "entity_type", "entity_id", "entity_label", "anomaly_type",
            "category", "evidence", "recommended_action", "severity", "value_at_stake",
            "owner", "lob", "portfolio", "account_name", "stage", "acv_gp",
            "provenance", "meaning", "framing", "personas", "category_question",
            "resolved", "is_opportunity_grain", "demo_priority", "triage", "priority"]
    both = pd.concat(
        [ds.reindex(columns=cols), lv.reindex(columns=cols)], ignore_index=True
    )
    key = both["entity_id"].astype(str) + "|" + both["anomaly_type"].astype(str)
    dupes = key[key.duplicated(keep=False)]
    both["corroborated"] = key.isin(set(dupes))
    # Keep the DS row where both exist — it carries the team's own severity —
    # but the live row's evidence is richer, so it is folded in.
    live_evidence = dict(zip(key[both["provenance"] == "live"],
                             both.loc[both["provenance"] == "live", "evidence"]))
    ds_mask = both["provenance"] == "ds-model"
    both.loc[ds_mask, "live_evidence"] = key[ds_mask].map(live_evidence)
    out = both.loc[~(both["corroborated"] & (both["provenance"] == "live"))].copy()
    out["triage"] = np.where(out["corroborated"], np.minimum(out["triage"] + 8, 100),
                             out["triage"])
    return out.sort_values("triage", ascending=False).reset_index(drop=True)


def for_persona(persona: str) -> pd.DataFrame:
    """The findings this persona is the right person to act on."""
    u = unified()
    return u.loc[u["personas"].map(lambda p: persona in p)].reset_index(drop=True)


def summary() -> dict:
    """Counts by category and provenance, for the Risks lens header."""
    u = unified()
    return {
        "total": int(len(u)),
        "dsModel": int((u["provenance"] == "ds-model").sum()),
        "live": int((u["provenance"] == "live").sum()),
        "corroborated": int(u["corroborated"].sum()),
        "byCategory": [
            {
                "category": c,
                "question": CATEGORY_BLURB.get(c, ""),
                "count": int((u["category"] == c).sum()),
                "valueAtStake": float(u.loc[u["category"] == c, "value_at_stake"].sum()),
                "critical": int(((u["category"] == c) & (u["priority"] == "Critical")).sum()),
            }
            for c in CATEGORY_ORDER
            if (u["category"] == c).any()
        ],
        "byPriority": {
            p: int((u["priority"] == p).sum())
            for p in ("Critical", "High", "Medium", "Low")
        },
        "upside": int((u["framing"] == "opportunity").sum()),
    }


def evidence_rows(entity_id: str, limit: int = 40) -> list[dict]:
    """
    The actual change-log rows behind a finding.

    The client asked to be able to drill from a card into the raw data; this is
    that drill, and it is the reason a flag reads as a fact rather than an
    assertion.
    """
    from .loader import movement

    m = movement()
    sub = m.loc[m["opportunity_code"] == entity_id].tail(limit)
    return [
        {
            "date": r.change_date.date().isoformat() if pd.notna(r.change_date) else None,
            "field": r.field,
            "from": r.old_value,
            "to": r.new_value,
            "by": r.changed_by,
        }
        for r in sub.itertuples(index=False)
    ]


def reset_caches() -> None:
    for fn in (enriched, live, unified):
        fn.cache_clear()
