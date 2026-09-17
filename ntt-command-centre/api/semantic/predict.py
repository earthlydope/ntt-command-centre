"""
Deal risk, closure probability, and the honest account of how well each works.

The client asked for two things on the call: "what's the likelihood that it will
close", and — more emphatically — "if a deal is at higher risk, what is the
variable that is driving it". Those are different questions and this module
answers them with different machinery, because only one of them is answerable
well from this data.

**What the data actually supports.** Measured over the 1,815 closed opportunities
in this extract:

  * `ForecastCategory` and `Confidence` are outcome ENCODINGS, not predictors.
    P(Won | Closed) = 1.000, P(Won | Omitted) = 0.000; every deal with confidence
    <= 0.19 lost and every deal >= 0.97 won. The generator writes the terminal
    confidence and category at close, so any model given them scores ~1.0 AUC by
    reading the label back. Movement-derived `confidence_drawdown` (r = -0.94),
    `forecast_regressions` (r = -0.89) and `peak_forecast_rank` (r = +0.80)
    inherit the same leak.
  * With every leaking feature removed, nothing else is strong: no remaining
    feature exceeds |r| = 0.06 against the outcome, every closed deal reaches
    Finalist so the stage path does not discriminate, and segment win rates
    spread only 31% to 44%. A leakage-free model scores **AUC 0.57 under 5-fold
    CV and 0.53 on a time-aware holdout** — barely above chance.

So `closure_probability()` is built, validated, and reports its own measured AUC
on screen; it is never presented as though it were sharp. The product leads
instead with `risk_score()`, which is not a model at all: it is a transparent sum
of observable facts, each of which is independently defensible to the rep whose
deal it flags, and each of which names itself as the driver. That is what the
client actually asked for, and unlike a 0.53-AUC classifier it is true.

None of this is a defect in the detection method — it is a property of a
synthetic extract whose outcome was assigned independently of its own features.
The platform says so out loud (see `model_card()`), because the people in the
room built this data and would spot a model living off its label immediately.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import AS_OF_TS, facts, opportunities
from .movement_features import STALL_DAYS, features, rep_behaviour

#: Features that encode the outcome. Quarantined from the model, and named here
#: so the exclusion is auditable rather than implicit in a column list.
LEAKING_FEATURES: dict[str, str] = {
    "forecast_category": "P(Won|Closed)=1.000, P(Won|Omitted)=0.000 — written at close",
    "confidence": "<=0.19 never won, >=0.97 always won — the rep's terminal entry",
    "forecast_rank": "ordinal restatement of forecast_category",
    "confidence_peak": "r=+0.79 — peaks at the terminal value",
    "confidence_now": "the terminal value itself",
    "confidence_drawdown": "r=-0.94 — the single strongest label echo in the data",
    "forecast_regressions": "r=-0.89 — a loss is logged as a walk back to Omitted",
    "peak_forecast_rank": "r=+0.80 — reaches Closed only on a win",
}

NUMERIC_FEATURES: tuple[str, ...] = (
    "acv_revenue", "acv_gp", "gm_percent", "line_count", "lob_count",
    "portfolio_count", "cycle_days", "touches", "touch_rate", "stage_changes",
    "max_stage_jump", "skipped_stages", "quiet_days", "stage_velocity",
    "value_drift", "value_changes", "close_date_slips", "slip_days", "active_days",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "lob", "portfolio", "order_type", "industry", "country", "create_quarter",
)


# --------------------------------------------------------------------------- #
# Risk — the primary surface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RiskFactor:
    """One named, observable reason a deal is risky."""

    key: str
    label: str
    #: Points added to the 0-100 risk score when this factor fires at full weight.
    weight: int
    #: Shown to a salesperson, so it states the fact and its consequence.
    template: str


#: The risk model, in full. Weights are a judgement about how much each fact
#: should move a review decision — they are deliberately round numbers, visible
#: in the API, and not fitted to the outcome, because a fitted weight on this
#: data would be fitted to noise (see the module docstring).
RISK_FACTORS: tuple[RiskFactor, ...] = (
    RiskFactor("stalled", "No activity", 26,
               "No field has changed in {quiet_days} days — past the {stall} day stall threshold."),
    RiskFactor("past_due", "Past its own close date", 24,
               "Close date was {days_past_due} days ago and the deal is still open."),
    RiskFactor("slipped", "Close date pushed", 14,
               "Close date has been moved {close_date_slips}x, {slip_days} days later in total."),
    RiskFactor("shrinking", "Value falling", 14,
               "Booked value is down {drift_pct} since it was first logged."),
    RiskFactor("stage_skip", "Stages skipped", 10,
               "Jumped {skipped_stages} stage(s) — the qualification steps have no record."),
    RiskFactor("early_stage_late_date", "Early stage, late in the cycle", 12,
               "Still at {stage} with {days_to_close} days to close."),
    RiskFactor("rep_pattern", "Owner's pattern", 10,
               "{owner}'s deals shrink {shrink_rate} of the time against a {peer} peer norm."),
    RiskFactor("single_thread", "Single line, large value", 6,
               "One line carrying {value} — no second offering attached to the deal."),
    RiskFactor("thin_margin", "Margin below its peer group", 8,
               "{gm}% GM against a {peer_gm}% norm for {portfolio} in {lob}."),
)
RISK_BY_KEY = {f.key: f for f in RISK_FACTORS}


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


@functools.lru_cache(maxsize=1)
def risk_table() -> pd.DataFrame:
    """
    Every OPEN opportunity scored 0-100, with its firing factors named.

    Closed deals are excluded on purpose: a risk score on a deal that already
    won or lost is a number nobody can act on, and including them would let the
    headline "N deals at risk" be inflated by history.
    """
    o = opportunities()
    mv = features()
    reps = rep_behaviour().set_index("rep")
    df = o.loc[o["is_open"]].merge(mv, on="opportunity_code", how="left", suffixes=("", "_mv"))
    if df.empty:
        return df.assign(risk_score=[], risk_band=[], risk_factors=[])

    # Peer margin norm per (lob, portfolio), measured on the full book.
    f = facts()
    peer_gm = (
        f.groupby(["lob", "portfolio"])
        .apply(lambda d: 100 * d["acv_gp"].sum() / d["acv_revenue"].sum()
               if d["acv_revenue"].sum() else np.nan, include_groups=False)
        .rename("peer_gm")
    )
    df = df.merge(peer_gm, left_on=["lob", "portfolio"], right_index=True, how="left")

    scores: list[int] = []
    drivers: list[list[dict]] = []
    for r in df.itertuples(index=False):
        hits: list[dict] = []

        def fire(key: str, intensity: float, **fmt) -> None:
            """`intensity` in 0..1 scales the factor's weight by how bad it is."""
            fac = RISK_BY_KEY[key]
            pts = int(round(fac.weight * min(max(intensity, 0.0), 1.0)))
            if pts <= 0:
                return
            hits.append({
                "key": key, "label": fac.label, "points": pts,
                "detail": fac.template.format(**fmt),
            })

        quiet = float(getattr(r, "quiet_days", np.nan) or 0)
        if quiet >= STALL_DAYS:
            # Saturates at 180 days: past six months of silence, more silence
            # tells you nothing new.
            fire("stalled", (quiet - STALL_DAYS) / (180 - STALL_DAYS) * 0.6 + 0.4,
                 quiet_days=int(quiet), stall=STALL_DAYS)
        if r.is_past_due:
            fire("past_due", min(r.days_past_due / 120, 1.0) * 0.7 + 0.3,
                 days_past_due=int(r.days_past_due))
        if getattr(r, "close_date_slips", 0):
            fire("slipped", min(r.slip_days / 90, 1.0),
                 close_date_slips=int(r.close_date_slips), slip_days=int(r.slip_days))
        drift = getattr(r, "value_drift", np.nan)
        if pd.notna(drift) and drift < -0.15:
            fire("shrinking", min(abs(drift) / 0.6, 1.0), drift_pct=_pct(abs(drift)))
        if getattr(r, "skipped_stages", 0):
            fire("stage_skip", min(r.skipped_stages / 3, 1.0),
                 skipped_stages=int(r.skipped_stages))
        # An Identification-stage deal due inside six weeks is not a forecast.
        # Gated on a close date that has not passed yet: once it has, `past_due`
        # is the truer statement of the same problem and saying both would
        # double-count one fact and print "-144 days to close".
        if (r.stage_rank <= 2 and pd.notna(r.days_to_close)
                and 0 <= r.days_to_close < 45):
            fire("early_stage_late_date", 1.0 - r.days_to_close / 45,
                 stage=r.stage, days_to_close=int(r.days_to_close))
        rep = reps.loc[r.owner] if r.owner in reps.index else None
        if rep is not None and not rep["thin"] and rep["shrink_rate_z"] > 1.5:
            fire("rep_pattern", min((rep["shrink_rate_z"] - 1.5) / 2, 1.0),
                 owner=r.owner, shrink_rate=_pct(rep["shrink_rate"]),
                 peer=_pct(rep["shrink_rate_peer"]))
        if r.line_count == 1 and r.acv_revenue > 150_000:
            fire("single_thread", min(r.acv_revenue / 500_000, 1.0), value=_money(r.acv_revenue))
        if pd.notna(r.gm_percent) and pd.notna(r.peer_gm) and r.gm_percent < r.peer_gm - 5:
            fire("thin_margin", min((r.peer_gm - r.gm_percent) / 15, 1.0),
                 gm=f"{r.gm_percent:.1f}", peer_gm=f"{r.peer_gm:.1f}",
                 portfolio=r.portfolio, lob=r.lob)

        hits.sort(key=lambda h: -h["points"])
        scores.append(min(sum(h["points"] for h in hits), 100))
        drivers.append(hits)

    df["risk_score"] = scores
    df["risk_factors"] = drivers
    df["top_driver"] = [d[0]["label"] if d else "No risk factor firing" for d in drivers]
    df["risk_band"] = pd.cut(
        df["risk_score"], [-1, 24, 49, 74, 100],
        labels=["Low", "Watch", "High", "Critical"],
    ).astype(str)
    # Value at risk is GP, not revenue: it is the number the budget is set in.
    df["value_at_risk"] = df["acv_gp"] * df["risk_score"] / 100
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Closure probability — the secondary, honestly-reported surface
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _trained() -> dict:
    """
    Train once, at first use, on closed opportunities with a time-aware split.

    The split is by close date rather than at random, because a random split
    lets the model see the back half of a quarter while predicting its front
    half, which flatters the score in exactly the way a pipeline product must
    not be flattered.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    o = opportunities()
    mv = features()
    df = o.merge(mv, on="opportunity_code", how="left", suffixes=("", "_mv"))
    closed = df.loc[df["is_closed"]].sort_values("close_date").reset_index(drop=True)

    X_all = _design_matrix(df)
    X = X_all.loc[closed.index] if len(X_all) == len(df) else _design_matrix(closed)
    X = _design_matrix(closed)
    y = closed["is_won"].astype(int).to_numpy()

    # Logistic regression, not a forest: on features this weak a boosted tree
    # scores no better (0.564 vs 0.571 under CV) and costs the per-deal
    # coefficient reading that makes an explanation possible.
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    cv_pred = cross_val_predict(base, X, y, cv=5, method="predict_proba")[:, 1]
    cv_auc = float(roc_auc_score(y, cv_pred))

    cut = int(len(closed) * 0.7)
    base.fit(X.iloc[:cut], y[:cut])
    holdout_auc = float(roc_auc_score(y[cut:], base.predict_proba(X.iloc[cut:])[:, 1]))

    model = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        cv=5, method="isotonic",
    ).fit(X, y)

    # Decile lift, measured out-of-fold so it reports generalisation.
    lift = pd.DataFrame({"p": cv_pred, "y": y})
    lift["decile"] = pd.qcut(lift["p"].rank(method="first"), 10, labels=False)
    deciles = (
        lift.groupby("decile")
        .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
        .reset_index()
        .to_dict("records")
    )

    return {
        "model": model,
        "columns": list(X.columns),
        "base_rate": float(y.mean()),
        "cv_auc": cv_auc,
        "holdout_auc": holdout_auc,
        "train_n": int(len(closed)),
        "deciles": deciles,
    }


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the categoricals against a fixed vocabulary so open and closed
    frames always produce identical columns in identical order."""
    from .loader import LOB_ORDER, ORDER_TYPES, PORTFOLIO_ORDER

    num = df.reindex(columns=list(NUMERIC_FEATURES))
    num = num.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    parts = [num]
    vocab = {
        "lob": LOB_ORDER,
        "portfolio": PORTFOLIO_ORDER,
        "order_type": ORDER_TYPES,
        "industry": tuple(sorted(facts()["industry"].dropna().unique())),
        "country": tuple(sorted(facts()["country"].dropna().unique())),
        "create_quarter": tuple(sorted(opportunities()["create_quarter"].dropna().unique())),
    }
    for col, values in vocab.items():
        src = df[col] if col in df.columns else pd.Series(index=df.index, dtype=object)
        for v in values:
            parts.append(src.eq(v).astype(float).rename(f"{col}={v}"))
    return pd.concat(parts, axis=1)


@functools.lru_cache(maxsize=1)
def closure_probability() -> pd.DataFrame:
    """
    P(win) for every OPEN opportunity, beside the rep's own call.

    The headline probability is the **data-science team's model** (see
    `ds_model`), because that model is theirs, it was validated on a wider
    extract than this book, and consuming it is the division of labour the
    client set out. This layer's own classifier is still trained and carried
    alongside as `p_win_independent`, so the two can be compared on screen —
    they agree closely, which is the point of showing both.

    If the DS workbook has not been ingested, the independent model becomes the
    headline rather than the surface going blank.
    """
    from . import ds_model

    t = _trained()
    o = opportunities()
    mv = features()
    df = o.loc[o["is_open"]].merge(mv, on="opportunity_code", how="left", suffixes=("", "_mv"))
    if df.empty:
        return df
    X = _design_matrix(df)[t["columns"]]
    df["p_win_independent"] = t["model"].predict_proba(X)[:, 1]

    if ds_model.available():
        ds = ds_model.predictions().set_index("opportunity_code")
        cols = ["p_win", "risk_bucket", "risk_bucket_label", "risk_bucket_relative",
                "risk_bucket_relative_label", "driving_force", "driver_feature",
                "driver_direction", "stage_at_cutoff"]
        for c in cols:
            df[c] = df["opportunity_code"].map(ds[c]) if c in ds.columns else None
        df["p_win_source"] = "ds-model"
        # A deal the DS drop does not cover falls back rather than going null.
        gap = df["p_win"].isna()
        df.loc[gap, "p_win"] = df.loc[gap, "p_win_independent"]
        df.loc[gap, "p_win_source"] = "independent"
    else:
        df["p_win"] = df["p_win_independent"]
        df["p_win_source"] = "independent"
        for c in ("risk_bucket", "risk_bucket_label", "risk_bucket_relative",
                  "risk_bucket_relative_label", "driving_force", "driver_feature",
                  "driver_direction", "stage_at_cutoff"):
            df[c] = None

    # The SHAP driver in the workbook's own words for the feature, and a
    # one-line form for a table cell. The full sentence ("Stage progression
    # (Identification) decreased win probability the most") is what the drawer
    # prints; a deal list has room for "Stage reached · lowers pWin" and no
    # more. Both are None where the DS drop does not cover the deal, so the UI
    # can say so rather than print a driver that belongs to no model.
    df["driver_label"] = df["driver_feature"].map(ds_model.FEATURE_LABEL)
    df["ds_driver"] = [
        f"{label} · {'lifts' if direction == 'up' else 'lowers'} pWin"
        if isinstance(label, str) and isinstance(direction, str) else None
        for label, direction in zip(df["driver_label"], df["driver_direction"])
    ]

    df["p_win_segment"] = df.set_index(["lob", "order_type"]).index.map(
        segment_base_rates().set_index(["lob", "order_type"])["win_rate"]
    )
    # The tension metric: the rep's own confidence against the statistical prior.
    # This is the one place `confidence` is legitimately used — as a claim to be
    # compared with, never as a feature to predict from.
    df["rep_confidence"] = df["confidence"]
    df["confidence_gap"] = df["rep_confidence"] - df["p_win"]
    df["expected_gp"] = df["acv_gp"] * df["p_win"]
    return df


@functools.lru_cache(maxsize=1)
def segment_base_rates() -> pd.DataFrame:
    """
    Historical win rate by (LOB x order type), with its sample size.

    Sample size travels with the rate everywhere it is shown. A 44% win rate on
    27 Digital Workplace expansions and a 42% on 621 Networking new-business
    deals are not the same fact, and a bare percentage hides that.
    """
    o = opportunities()
    closed = o.loc[o["is_closed"]]
    g = closed.groupby(["lob", "order_type"]).agg(
        n=("is_won", "size"), wins=("is_won", "sum"), gp=("acv_gp", "sum"),
    ).reset_index()
    g["win_rate"] = g["wins"] / g["n"]
    overall = closed["is_won"].mean()
    # Shrink small cells toward the book-wide rate: 27 deals is not enough to
    # claim a segment beats the average by 7 points.
    k = 30
    g["win_rate_smoothed"] = (g["wins"] + k * overall) / (g["n"] + k)
    g["overall"] = overall
    return g


def model_card() -> dict:
    """
    The measured truth about the models, for display on screen.

    Both are reported: the data-science team's model (the headline) and this
    layer's independent reproduction (the corroboration). This is shown, not
    buried — the people who built the data are in the room, and a platform that
    states its AUC is more credible than one that does not.
    """
    from . import ds_model

    return {
        "primary": ds_model.model_card() if ds_model.available()
        else ds_model.unavailable_card(),
        "independent": _independent_card(),
        "riskModel": {
            "name": "Deal Risk Score",
            "kind": "deterministic, not learned",
            "why": (
                "Both closure models are weak on this extract, for the same reason and "
                "by the same measure. Risk is therefore the surface the product acts "
                "on: every point of it comes from an observable fact about the deal — "
                "days since the last change, days past its own close date, how far the "
                "close date has moved, how much the value has fallen, stages with no "
                "record — so each score can be defended to the rep whose deal it flags."
            ),
            "factors": [
                {"key": f.key, "label": f.label, "maxPoints": f.weight}
                for f in RISK_FACTORS
            ],
            "bands": [
                {"band": "Low", "from": 0, "to": 24},
                {"band": "Watch", "from": 25, "to": 49},
                {"band": "High", "from": 50, "to": 74},
                {"band": "Critical", "from": 75, "to": 100},
            ],
        },
    }


def _independent_card() -> dict:
    t = _trained()
    return {
        "target": "P(Deal Won) at opportunity grain",
        "trainedOn": t["train_n"],
        "baseRate": round(t["base_rate"], 4),
        "cvAuc": round(t["cv_auc"], 4),
        "holdoutAuc": round(t["holdout_auc"], 4),
        "split": "time-aware, 70% earliest close dates train / 30% latest test",
        "algorithm": "logistic regression, isotonic-calibrated, 5-fold",
        "deciles": t["deciles"],
        "featuresUsed": len(t["columns"]),
        "excluded": [{"feature": k, "reason": v} for k, v in LEAKING_FEATURES.items()],
        "verdict": (
            f"Discrimination is weak on this extract (holdout AUC "
            f"{t['holdout_auc']:.2f} against 0.50 for chance). Once the columns that "
            f"encode the outcome are removed, no remaining feature correlates above "
            f"0.06 with win or loss, every closed deal reaches Finalist, and segment "
            f"win rates span only 31-44%. That is a property of how this synthetic "
            f"extract was generated, not of the method — the outcome was assigned "
            f"independently of the features. Deal risk, which is computed from "
            f"observable facts rather than learned, is the surface to act on here; "
            f"this probability is carried as a secondary signal and is labelled as "
            f"such wherever it appears."
        ),
    }


def explain(opportunity_code: str) -> dict | None:
    """Everything the intelligence layer knows about one deal, for the drill-down."""
    r = risk_table()
    row = r.loc[r["opportunity_code"] == opportunity_code]
    if row.empty:
        return None
    row = row.iloc[0]
    p = closure_probability()
    prow = p.loc[p["opportunity_code"] == opportunity_code]
    out = {
        "opportunityCode": opportunity_code,
        "name": row["opportunity_name"],
        "account": row["account_name"],
        "owner": row["owner"],
        "stage": row["stage"],
        "lob": row["lob"],
        "portfolio": row["portfolio"],
        "acvRevenue": float(row["acv_revenue"]),
        "acvGp": float(row["acv_gp"]),
        "closeDate": row["close_date"].date().isoformat() if pd.notna(row["close_date"]) else None,
        "riskScore": int(row["risk_score"]),
        "riskBand": row["risk_band"],
        "riskFactors": row["risk_factors"],
        "valueAtRisk": float(row["value_at_risk"]),
        "quietDays": int(row["quiet_days"]) if pd.notna(row["quiet_days"]) else None,
        "stagePath": row["stage_path_str"],
    }
    if not prow.empty:
        pr = prow.iloc[0]

        def text(col: str) -> str | None:
            # A deal the DS drop lacks carries NaN in every mapped column, and
            # NaN would reach the page as the string "nan" or as JSON null by
            # accident of serialisation. Only a real string is a driver.
            v = pr[col]
            return v if isinstance(v, str) and v else None

        out |= {
            "pWin": float(pr["p_win"]),
            "pWinSegment": float(pr["p_win_segment"]) if pd.notna(pr["p_win_segment"]) else None,
            "repConfidence": float(pr["rep_confidence"]),
            "confidenceGap": float(pr["confidence_gap"]),
            "expectedGp": float(pr["expected_gp"]),
            # The DS model's own account of its number. pWin without the
            # variable that moved it is exactly the "likelihood" surface the
            # client said was not enough; the SHAP sentence is their answer
            # to "what is driving it", carried verbatim beside this layer's
            # own risk factors rather than merged into them.
            "pWinSource": pr["p_win_source"],
            "drivingForce": text("driving_force"),
            "driverFeature": text("driver_feature"),
            "driverLabel": text("driver_label"),
            "driverDirection": text("driver_direction"),
            "dsDriver": text("ds_driver"),
            "riskBucket": int(pr["risk_bucket"]) if pd.notna(pr["risk_bucket"]) else None,
            "riskBucketLabel": text("risk_bucket_label"),
            "riskBucketRelativeLabel": text("risk_bucket_relative_label"),
        }
    return out


def reset_caches() -> None:
    for fn in (risk_table, closure_probability, segment_base_rates, _trained):
        fn.cache_clear()
