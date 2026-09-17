"""
Everything the change log knows about an opportunity, reduced to one row each.

This is the module the previous build could not write, because the movement
dataset did not exist yet. It now does, and it is the difference between a
dashboard that reports the present state of the pipeline and a product that can
say *how the pipeline got here* — which deals stopped moving, whose forecasts
walk backwards, which values were inflated at logging and quietly cut before
close, and how long a deal really takes to cross each stage.

Every feature below is computed by replaying the log in `seq` order. Nothing
here reads the fact table's current state, so these features remain an
independent account of the deal and can be compared against it.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .loader import AS_OF_TS, FORECAST_RANK, STAGE_RANK, facts, movement

#: A value revision of less than this fraction is a correction, not a behaviour.
#: 15% is wide enough to ignore a rounding or a small re-scope and narrow enough
#: to catch the 3-5x inflate-then-shrink pattern the client described.
MATERIAL_DRIFT = 0.15

#: An open deal with no logged change in this long is treated as stalled. The
#: client's own example was "an opportunity contributing 20% of a region's
#: target that hasn't moved for 60 days".
STALL_DAYS = 60


def _stage_path(g: pd.DataFrame) -> list[str]:
    """The ordered list of stages this opportunity was recorded in."""
    s = g.loc[g["field"] == "stage"]
    if s.empty:
        return []
    path = [str(s["old_value"].iloc[0])] + [str(v) for v in s["new_value"]]
    return [p for p in path if p and p not in ("nan", "Not Set")]


@functools.lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    """
    One row per opportunity: the replayed history as model-ready features.

    Returned columns are grouped by what they describe —

      timing      first_change, last_change, quiet_days, is_stalled, touches,
                  touch_rate, days_in_current_stage
      progression stage_path, stage_changes, max_stage_jump, skipped_stages,
                  reached_stage_rank, went_backwards
      forecast    forecast_changes, forecast_regressions, peak_forecast_rank,
                  confidence_peak, confidence_now, confidence_drawdown
      value       value_first, value_peak, value_last, value_drift,
                  value_changes, shrank, inflated
      slippage    close_date_slips, slip_days
    """
    m = movement()
    out: dict[str, list] = {k: [] for k in (
        "opportunity_code", "first_change", "last_change", "touches",
        "stage_changes", "stage_path_str", "max_stage_jump", "skipped_stages",
        "reached_stage_rank", "went_backwards", "days_in_current_stage",
        "forecast_changes", "forecast_regressions", "peak_forecast_rank",
        "confidence_peak", "confidence_now", "confidence_drawdown",
        "value_first", "value_peak", "value_last", "value_changes",
        "close_date_slips", "slip_days", "changed_by",
    )}

    for code, g in m.groupby("opportunity_code", sort=False):
        out["opportunity_code"].append(code)
        out["first_change"].append(g["change_date"].min())
        out["last_change"].append(g["change_date"].max())
        out["touches"].append(len(g))
        out["changed_by"].append(g["changed_by"].iloc[-1])

        # --- progression -------------------------------------------------- #
        path = _stage_path(g)
        ranks = [STAGE_RANK[p] for p in path if p in STAGE_RANK]
        jumps = [b - a for a, b in zip(ranks, ranks[1:])] if len(ranks) > 1 else []
        stage_rows = g.loc[g["field"] == "stage"]
        out["stage_changes"].append(len(stage_rows))
        out["stage_path_str"].append(" > ".join(path))
        out["max_stage_jump"].append(max(jumps) if jumps else 0)
        # A jump of n rungs skips n-1 stages. Won/Lost share a rank, so a close
        # out of Finalist reads as +1 and contributes no skip, which is correct.
        out["skipped_stages"].append(max((j - 1 for j in jumps), default=0))
        out["reached_stage_rank"].append(max(ranks) if ranks else 0)
        out["went_backwards"].append(any(j < 0 for j in jumps))
        last_stage_change = stage_rows["change_date"].max() if len(stage_rows) else pd.NaT
        anchor = last_stage_change if pd.notna(last_stage_change) else g["change_date"].min()
        out["days_in_current_stage"].append(int((AS_OF_TS - anchor).days))

        # --- forecast + confidence ---------------------------------------- #
        fc = g.loc[g["field"] == "forecast_category"]
        fr = [FORECAST_RANK.get(str(v), -1) for v in fc["new_value"]]
        fr = [r for r in fr if r >= 0]
        # A regression is the rep walking a call back: Commit -> Best Case.
        # Anything landing on Closed is the deal finishing, not a retreat.
        regress = 0
        for a, b in zip(fr, fr[1:]):
            if b < a and FORECAST_RANK["Closed"] not in (a, b):
                regress += 1
        out["forecast_changes"].append(len(fc))
        out["forecast_regressions"].append(regress)
        out["peak_forecast_rank"].append(max(fr) if fr else 0)

        cf = g.loc[g["field"] == "confidence", "new_num"].dropna()
        out["confidence_peak"].append(float(cf.max()) if len(cf) else np.nan)
        out["confidence_now"].append(float(cf.iloc[-1]) if len(cf) else np.nan)
        out["confidence_drawdown"].append(
            float(cf.max() - cf.iloc[-1]) if len(cf) else np.nan
        )

        # --- value --------------------------------------------------------- #
        # The log opens every deal at 0 -> first estimate, mirroring the client's
        # own example. That leading zero is the creation event, not a shrink, so
        # the baseline is the first NON-ZERO value logged.
        rv = g.loc[g["field"] == "acv_revenue", "new_num"].dropna()
        rv = rv[rv > 0]
        out["value_first"].append(float(rv.iloc[0]) if len(rv) else np.nan)
        out["value_peak"].append(float(rv.max()) if len(rv) else np.nan)
        out["value_last"].append(float(rv.iloc[-1]) if len(rv) else np.nan)
        out["value_changes"].append(int(len(rv)))

        # --- slippage ------------------------------------------------------ #
        cd = g.loc[g["field"] == "close_date"]
        slip = 0
        if len(cd):
            old = pd.to_datetime(cd["old_value"], errors="coerce")
            new = pd.to_datetime(cd["new_value"], errors="coerce")
            slip = int((new - old).dt.days.fillna(0).sum())
        out["close_date_slips"].append(len(cd))
        out["slip_days"].append(slip)

    df = pd.DataFrame(out)

    # --- derived ratios, vectorised ---------------------------------------- #
    df["quiet_days"] = (AS_OF_TS - df["last_change"]).dt.days
    df["active_days"] = (df["last_change"] - df["first_change"]).dt.days.clip(lower=1)
    df["touch_rate"] = df["touches"] / df["active_days"]
    # Drift is last vs first: the client's pattern is "logged really large, closed
    # at a third". Peak-vs-last would flag a deal that grew then settled back to
    # where it started, which is not the behaviour they described.
    df["value_drift"] = np.where(
        df["value_first"] > 0, df["value_last"] / df["value_first"] - 1.0, np.nan
    )
    df["shrank"] = df["value_drift"] < -MATERIAL_DRIFT
    df["inflated"] = df["value_drift"] > MATERIAL_DRIFT
    df["stage_velocity"] = np.where(
        df["stage_changes"] > 0, df["active_days"] / df["stage_changes"], np.nan
    )

    # `is_stalled` is only meaningful for a deal that is still open, so it is
    # resolved against the fact table's current stage rather than the log.
    open_codes = set(facts().loc[facts()["is_open"], "opportunity_code"])
    df["is_open"] = df["opportunity_code"].isin(open_codes)
    df["is_stalled"] = df["is_open"] & (df["quiet_days"] >= STALL_DAYS)
    return df


@functools.lru_cache(maxsize=1)
def rep_behaviour() -> pd.DataFrame:
    """
    Per-rep behavioural rates, with the peer norm attached to every row.

    The client was explicit that one shrunk deal is noise and the same rep
    shrinking repeatedly is a finding — so every rate here is carried next to the
    all-rep mean and expressed as a z-score, and nothing downstream is allowed to
    call a rep an outlier off the raw rate alone. Reps below `min_deals` are
    scored but flagged `thin`, because a 100% shrink rate on two deals is a
    sample-size artefact and presenting it as a coaching finding would burn the
    credibility of every other flag on the page.
    """
    min_deals = 8
    f = features()
    opp = facts().groupby("opportunity_code").agg(
        owner=("owner", "first"),
        acv_gp=("acv_gp", "sum"),
        acv_revenue=("acv_revenue", "sum"),
        is_won=("is_won", "first"),
        is_lost=("is_lost", "first"),
        is_closed=("is_closed", "first"),
        is_open=("is_open", "first"),
        order_type=("order_type", "first"),
        cycle_days=("cycle_days", "first"),
    ).reset_index()
    j = opp.merge(f, on="opportunity_code", how="left", suffixes=("", "_mv"))

    g = j.groupby("owner")
    out = pd.DataFrame({
        "deals": g.size(),
        "open_deals": g["is_open"].sum(),
        "closed_deals": g["is_closed"].sum(),
        "won": g["is_won"].sum(),
        "open_gp": g.apply(lambda d: d.loc[d["is_open"], "acv_gp"].sum(), include_groups=False),
        "won_gp": g.apply(lambda d: d.loc[d["is_won"], "acv_gp"].sum(), include_groups=False),
        "shrink_rate": g["shrank"].mean(),
        "inflate_rate": g["inflated"].mean(),
        "regression_rate": g.apply(
            lambda d: (d["forecast_regressions"] > 0).mean(), include_groups=False),
        "skip_rate": g.apply(lambda d: (d["skipped_stages"] > 0).mean(), include_groups=False),
        "stall_rate": g.apply(
            lambda d: d.loc[d["is_open"], "is_stalled"].mean() if d["is_open"].any() else 0.0,
            include_groups=False),
        "slip_days": g["slip_days"].mean(),
        "touch_rate": g["touch_rate"].mean(),
        "avg_cycle_days": g.apply(
            lambda d: d.loc[d["is_closed"], "cycle_days"].mean(), include_groups=False),
        "new_business_share": g.apply(
            lambda d: (d["order_type"] == "New Business").mean(), include_groups=False),
    })
    out["win_rate"] = np.where(out["closed_deals"] > 0, out["won"] / out["closed_deals"], np.nan)
    out["thin"] = out["deals"] < min_deals

    # Peer norms computed on the credible reps only, so a handful of two-deal
    # reps cannot drag the mean the rest are judged against.
    base = out.loc[~out["thin"]]
    for col in ("shrink_rate", "inflate_rate", "regression_rate", "skip_rate",
                "stall_rate", "win_rate", "new_business_share", "avg_cycle_days",
                "touch_rate"):
        mu, sd = base[col].mean(), base[col].std(ddof=0)
        out[f"{col}_peer"] = mu
        out[f"{col}_z"] = (out[col] - mu) / sd if sd and sd > 0 else 0.0
    return out.reset_index().rename(columns={"owner": "rep"})


def reset_caches() -> None:
    features.cache_clear()
    rep_behaviour.cache_clear()
