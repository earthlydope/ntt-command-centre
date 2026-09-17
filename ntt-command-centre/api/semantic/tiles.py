"""
KPI tiles, chosen per PAGE rather than per persona.

This module exists because of a real defect. The tiles used to be picked by
persona alone, so all five of a manager's pages showed the same six numbers:
you clicked "Compare reps", then "Whose forecast can I trust", then "Where we
lose", and the entire top half of the screen — six tiles and the brief — did not
move. The charts below did change, but they were under the fold, so the product
looked broken in exactly the way the customer reported: *"the application
doesn't change values, charts and information on the right as the feature
changes."*

Every page states one question. Its tiles are now the measures that answer THAT
question and nothing else — a page about where deals fall out shows conversion
and cycle time, not cross-sell upside. Where two pages legitimately share a
measure it is the same arithmetic, not a second definition.

Nothing here computes a number from scratch: every figure comes from
`measures.measures()`, `predict`, `accounts`, `budget` or `movement_features`,
which is what keeps the tiles and the charts below them in agreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import accounts as ACC
from . import anomalies as ANOM
from . import budget as B
from . import crosssell as XS
from . import predict as P
from .loader import CUR_QUARTER, STAGE_ORDER, opportunities
from .measures import (
    FilterState,
    count,
    money,
    mult,
    pct,
    slice_frame,
    subset,
    trend,
)
from .movement_features import STALL_DAYS, features, rep_behaviour
from .personas import Principal


def tile(key: str, label: str, value: float, formatted: str, sub: str,
         tone: str = "accent", direction: str = "neutral",
         icon: str = "metric", spark: list[float] | None = None) -> dict:
    """
    One tile.

    `spark` is omitted rather than faked: several of these are snapshots with no
    honest history, and a flat line under a number implies a trend was checked
    when it was not.
    """
    return {"key": key, "label": label, "value": float(value), "formatted": formatted,
            "sub": sub, "tone": tone, "direction": direction, "icon": icon,
            "spark": spark or []}


def _ctx(fs: FilterState, principal: Principal) -> dict:
    """Everything the tile sets draw on, computed once per request."""
    frame = slice_frame(fs, principal)
    codes = set(frame["opportunity_code"])
    risk = P.risk_table()
    risk = risk[risk["opportunity_code"].isin(codes)]
    mv = features()
    mv = mv[mv["opportunity_code"].isin(codes)]
    opp = opportunities()
    opp = opp[opp["opportunity_code"].isin(codes)]
    return {"frame": frame, "risk": risk, "mv": mv, "opp": opp}


def _tone_for(share: float, warn: float, danger: float) -> str:
    return "danger" if share >= danger else "warn" if share >= warn else "good"


# --------------------------------------------------------------------------- #
# Account Executive
# --------------------------------------------------------------------------- #


def _ae(page: str, fs: FilterState, p: Principal, m: dict, c: dict) -> list[dict]:
    frame, risk, mv, opp = c["frame"], c["risk"], c["mv"], c["opp"]
    open_b, past, stalled = m["open"], m["pastDue"], m["stalled"]
    at_risk = float(risk.loc[risk["risk_band"].isin(("High", "Critical")), "acv_gp"].sum())

    if page == "my-deals":
        slipped = risk[risk["close_date_slips"] > 0]
        shrunk = risk[risk["value_drift"] < -0.30]
        worst = int(risk["risk_score"].max()) if len(risk) else 0
        return [
            tile("past_due", "Past close date", past["gp"], money(past["gp"]),
                 f"{past['opps']} deals the date has already gone by",
                 "danger" if m["pastDueShare"] > 50 else "warn", "up-bad", "clock"),
            tile("stalled", "Stopped moving", stalled["gp"], money(stalled["gp"]),
                 f"{stalled['opps']} with nothing logged in {STALL_DAYS}+ days",
                 "danger" if m["stalledShare"] > 50 else "warn", "up-bad", "clock"),
            tile("slipped", "Date pushed back", len(slipped), count(len(slipped)),
                 f"{int(slipped['slip_days'].sum()):,} days later in total"
                 if len(slipped) else "No close date has moved",
                 "warn" if len(slipped) else "good", "up-bad", "calendar"),
            tile("shrunk", "Value dropped", len(shrunk), count(len(shrunk)),
                 "Worth less now than when first logged"
                 if len(shrunk) else "No deal has shrunk materially",
                 "warn" if len(shrunk) else "good", "up-bad", "down"),
            tile("worst", "Worst deal", worst, f"{worst}",
                 "Highest risk score in your book",
                 "danger" if worst >= 70 else "warn" if worst >= 50 else "good",
                 "up-bad", "risk"),
            tile("open", "Still open", open_b["gp"], money(open_b["gp"]),
                 f"{open_b['opps']} deals", "accent", "up-good",
                 "pipeline", trend(frame, "open_gp")),
        ]

    if page == "my-accounts":
        ws = ACC.whitespace(fs, p, limit=999)
        xs = XS.summary(fs, p)
        upside = sum(w["estimatedGp"] for w in ws)
        # Built from THIS rep's own rows, not from the entity account profile.
        # Reading the global profile made "biggest customer" show Cobalt's whole
        # $1.39M when the rep in front of the screen owns a fraction of it —
        # a true number about somebody else's business.
        mine = frame.groupby(["account_code", "account_name"]).agg(
            gp=("acv_gp", "sum"),
            opportunities=("opportunity_code", "nunique"),
            lob_count=("lob", "nunique"),
        ).reset_index()
        one_lob = int((mine["lob_count"] == 1).sum())
        top = mine.nlargest(1, "gp") if len(mine) else mine
        return [
            tile("customers", "My customers", len(mine), count(len(mine)),
                 f"{int(mine['opportunities'].sum())} deals between them",
                 "accent", "up-good", "account"),
            tile("one_lob", "Buying one thing", one_lob, count(one_lob),
                 "Customers taking only one of our four lines",
                 "warn" if one_lob else "good", "up-bad", "pipeline"),
            tile("biggest", "Biggest customer",
                 float(top["gp"].iloc[0]) if len(top) else 0,
                 money(float(top["gp"].iloc[0])) if len(top) else "—",
                 f"{top['account_name'].iloc[0]} — your share"
                 if len(top) else "No customers in scope",
                 "accent", "neutral", "account"),
            tile("ideas", "Growth ideas", xs["recommendations"],
                 count(xs["recommendations"]),
                 f"{xs['strong']} we are confident about" if xs["recommendations"]
                 else "Nothing suggested for your accounts",
                 "good" if xs["strong"] else "neutral", "up-good", "growth"),
            tile("best_idea", "Best idea",
                 xs["topRecommendation"]["peerGp"] if xs["topRecommendation"] else 0,
                 xs["topRecommendation"]["offering"] if xs["topRecommendation"] else "—",
                 f"{xs['topRecommendation']['accountName']} · "
                 f"{xs['topRecommendation']['confidence'].lower()} confidence"
                 if xs["topRecommendation"] else "No suggestion in scope",
                 "accent", "neutral", "growth"),
        ]

    if page == "my-record":
        closed = opp[opp["is_closed"]]
        cycle = float(closed["cycle_days"].median()) if len(closed) else 0
        avg = float(closed.loc[closed["is_won"], "acv_gp"].mean()) if closed["is_won"].any() else 0
        allreps = rep_behaviour()
        peer = float(allreps["win_rate"].median() * 100)
        lost_stage = "—"
        if closed["is_lost"].any():
            paths = features().set_index("opportunity_code")
            lost = closed.loc[closed["is_lost"], "opportunity_code"]
            last = paths.reindex(lost)["stage_path_str"].dropna()
            if len(last):
                ends = last.map(lambda s: [x.strip() for x in str(s).split(">")][-2]
                                if len(str(s).split(">")) > 1 else "—")
                lost_stage = ends.mode().iloc[0] if len(ends.mode()) else "—"
        return [
            tile("winrate", "My win rate", m["winRate"], pct(m["winRate"]),
                 f"Team middle is {peer:.0f}%",
                 "good" if m["winRate"] >= peer else "warn", "up-good",
                 "target", trend(frame, "win_rate")),
            tile("won", "Won this year", m["won"]["gp"], money(m["won"]["gp"]),
                 f"{m['won']['opps']} deals", "good", "up-good",
                 "won", trend(frame, "won_gp")),
            tile("closed", "Deals closed", m["closed"]["opps"], count(m["closed"]["opps"]),
                 f"{m['winRateBasis']['won']} won, {m['winRateBasis']['lost']} lost",
                 "neutral", "neutral", "metric"),
            tile("avg", "Average win", avg, money(avg),
                 "Profit on a typical won deal", "neutral", "up-good", "metric"),
            tile("cycle", "Time to close", cycle, f"{cycle:.0f}d",
                 "From first logged to closed", "neutral", "up-bad", "clock"),
            tile("lost_at", "Usually lost at", 0, lost_stage,
                 "The stage most of your losses reach", "warn", "neutral", "pipeline"),
        ]

    # my-day — what to do first
    needs = int(len(risk[risk["quiet_days"] >= STALL_DAYS])) if len(risk) else 0
    accounts_flagged = int(risk["account_name"].nunique()) if len(risk) else 0
    return [
        tile("needs_call", "Needs a call", needs, count(needs),
             f"Nothing logged in {STALL_DAYS}+ days",
             "danger" if needs > 5 else "warn" if needs else "good", "up-bad", "clock"),
        tile("at_risk", "At risk", at_risk, money(at_risk),
             f"{int(risk['risk_band'].isin(('High', 'Critical')).sum())} deals scoring high",
             "danger" if at_risk > open_b["gp"] * 0.4 else "warn", "up-bad", "risk"),
        tile("calls", "Calls to make", accounts_flagged, count(accounts_flagged),
             "Customers with something to chase — one call each",
             "warn" if accounts_flagged else "good", "up-bad", "account"),
        tile("past_due", "Past close date", past["gp"], money(past["gp"]),
             f"{past['opps']} deals", "warn", "up-bad", "calendar"),
        tile("open", "Open pipeline", open_b["gp"], money(open_b["gp"]),
             f"{open_b['opps']} deals · {pct(m['shareOfEntity'])} of the region",
             "accent", "up-good", "pipeline", trend(frame, "open_gp")),
        tile("quiet", "Typical silence", m["medianQuietDays"],
             f"{m['medianQuietDays']:.0f}d", "Across your open deals",
             "warn" if m["medianQuietDays"] > STALL_DAYS else "good", "up-bad", "clock"),
    ]


# --------------------------------------------------------------------------- #
# Sales Manager
# --------------------------------------------------------------------------- #


def _manager(page: str, fs: FilterState, p: Principal, m: dict, c: dict) -> list[dict]:
    frame, risk, mv, opp = c["frame"], c["risk"], c["mv"], c["opp"]
    open_b, stalled = m["open"], m["stalled"]
    reps = rep_behaviour()
    members = p.predicate.get("owner", ())
    pod = reps[reps["rep"].isin(members)] if members else reps
    credible = pod[~pod["thin"]]
    zcols = ["shrink_rate_z", "inflate_rate_z", "regression_rate_z", "stall_rate_z"]
    off = int(((credible[zcols].abs() >= 1.5).any(axis=1)).sum())

    if page == "rep-benchmark":
        worst = credible.assign(m=credible[zcols].abs().max(axis=1)).nlargest(1, "m") \
            if len(credible) else credible
        spread = (float(credible["win_rate"].max() - credible["win_rate"].min()) * 100
                  if len(credible) else 0.0)
        big = pod.nlargest(1, "open_gp") if len(pod) else pod
        return [
            tile("off", "Reps off the pattern", off, count(off),
                 f"of {len(pod)} · more than 1.5 SD from the team",
                 "warn" if off else "good", "up-bad", "people"),
            tile("worst", "Furthest out", 0,
                 str(worst["rep"].iloc[0]) if len(worst) else "—",
                 "Largest distance from the team norm", "danger", "neutral", "people"),
            tile("spread", "Win-rate spread", spread, pct(spread, 0),
                 f"Best to worst, across the {len(credible)} reps with enough deals",
                 "neutral", "up-bad", "target"),
            tile("team_win", "Team win rate", m["winRate"], pct(m["winRate"]),
                 f"{m['winRateBasis']['won']} won of "
                 f"{m['winRateBasis']['won'] + m['winRateBasis']['lost']}",
                 "neutral", "up-good", "target", trend(frame, "win_rate")),
            tile("biggest_book", "Biggest book", 0,
                 str(big["rep"].iloc[0]) if len(big) else "—",
                 f"{money(float(big['open_gp'].iloc[0]))} still open" if len(big) else "—",
                 "accent", "neutral", "pipeline"),
            tile("reps", "Reps", len(pod), count(len(pod)),
                 f"{int(pod['deals'].sum())} deals between them",
                 "neutral", "neutral", "people"),
        ]

    if page == "process":
        entered = mv[mv["stage_path_str"].str.startswith("Identification", na=False)]
        reached = entered[entered["stage_path_str"].str.contains("Finalist", regex=False, na=False)]
        conv = 100 * len(reached) / len(entered) if len(entered) else 0
        skipped = int((mv["skipped_stages"] > 0).sum())
        closed = opp[opp["is_closed"]]
        cycle = float(closed["cycle_days"].median()) if len(closed) else 0
        return [
            tile("entered", "Deals started", len(entered), count(len(entered)),
                 "Logged from the very first stage", "accent", "up-good", "pipeline"),
            tile("reached", "Got to the last stage", conv, pct(conv, 0),
                 f"{len(reached)} of {len(entered)} that started at the beginning",
                 "good" if conv > 85 else "warn", "up-good", "target"),
            tile("winrate", "Then won", m["winRate"], pct(m["winRate"]),
                 f"{m['winRateBasis']['won']} of "
                 f"{m['winRateBasis']['won'] + m['winRateBasis']['lost']} closed",
                 "neutral", "up-good", "won", trend(frame, "win_rate")),
            tile("skipped", "Skipped a step", skipped, count(skipped),
                 "No record of the stages in between",
                 "warn" if skipped else "good", "up-bad", "risk"),
            tile("cycle", "Time to close", cycle, f"{cycle:.0f}d",
                 "First logged to closed, typical deal", "neutral", "up-bad", "clock"),
            tile("lost", "Deals lost", m["winRateBasis"]["lost"],
                 count(m["winRateBasis"]["lost"]), "This year", "danger", "up-bad", "down"),
        ]

    if page == "calibration":
        shrinkers = int((credible["shrink_rate_z"] >= 1.5).sum())
        sandbaggers = int((credible["inflate_rate_z"] >= 1.5).sum())
        reversers = int((credible["regression_rate_z"] >= 1.5).sum())
        shrunk = risk[risk["value_drift"] < -0.30]
        return [
            tile("shrinkers", "Oversize their deals", shrinkers, count(shrinkers),
                 "Reps whose deals shrink more than their peers'",
                 "warn" if shrinkers else "good", "up-bad", "down"),
            tile("sandbaggers", "Undersize their deals", sandbaggers, count(sandbaggers),
                 "Reps whose deals grow well past the first estimate",
                 "accent" if sandbaggers else "good", "up-bad", "growth"),
            tile("reversers", "Change their mind", reversers, count(reversers),
                 "Reps who walk a forecast backwards more often",
                 "warn" if reversers else "good", "up-bad", "people"),
            tile("shrunk_deals", "Deals worth less", len(shrunk), count(len(shrunk)),
                 "Down more than 30% since first logged",
                 "warn" if len(shrunk) else "good", "up-bad", "down"),
            tile("stalled", "Stopped moving", m["stalledShare"], pct(m["stalledShare"]),
                 f"{stalled['opps']} deals silent {STALL_DAYS}+ days",
                 "danger" if m["stalledShare"] > 50 else "warn", "up-bad", "clock"),
            tile("team_win", "Team win rate", m["winRate"], pct(m["winRate"]),
                 "The one outcome that survives averaging",
                 "neutral", "up-good", "target", trend(frame, "win_rate")),
        ]

    if page == "pod-whitespace":
        ws = ACC.whitespace(fs, p, limit=999)
        xs = XS.summary(fs, p)
        upside = sum(w["estimatedGp"] for w in ws)
        lv = ACC.lob_count_value()
        one = lv[lv["lob_count"] == 1]
        four = lv[lv["lob_count"] == 4]
        lift = (float(four["medianGp"].iloc[0]) / float(one["medianGp"].iloc[0])
                if len(one) and len(four) and float(one["medianGp"].iloc[0]) else 0)
        best = ws[0] if ws else None
        prof = ACC.account_profile()
        mine = prof[prof["account_code"].isin(set(frame["account_code"]))]
        return [
            tile("upside", "Missing a whole line", upside, money(upside),
                 f"{len(ws)} customers do not buy one of our four lines",
                 "good", "up-good", "growth"),
            tile("best", "Best opening", 0,
                 str(best["recommendedLob"]) if best else "—",
                 f"at {best['accountName']}" if best else "No gap in scope",
                 "good", "neutral", "target"),
            tile("lift", "Worth doing", lift, f"{lift:.0f}x" if lift else "—",
                 "A four-line customer against a one-line one",
                 "good", "up-good", "growth"),
            tile("ideas", "Growth ideas to assign", xs["recommendations"],
                 count(xs["recommendations"]),
                 f"{xs['strong']} we are confident about · {xs['accounts']} customers",
                 "good" if xs["strong"] else "neutral", "up-good", "growth"),
            tile("plays", "Ideas that repeat", xs["themes"], count(xs["themes"]),
                 f"Top: {xs['topTheme']} at {xs['topThemeAccounts']} customers"
                 if xs["topTheme"] else "Nothing repeats across customers",
                 "accent", "neutral", "target"),
        ]

    # pod-pulse — who needs me this week
    needs = int(len(risk[risk["quiet_days"] >= STALL_DAYS])) if len(risk) else 0
    at_risk = float(risk.loc[risk["risk_band"].isin(("High", "Critical")), "acv_gp"].sum())
    return [
        tile("off", "Reps to coach", off, count(off),
             f"of {len(pod)} · off the team pattern",
             "warn" if off else "good", "up-bad", "people"),
        tile("needs_call", "Deals to chase", needs, count(needs),
             f"Nothing logged in {STALL_DAYS}+ days",
             "danger" if needs > 20 else "warn", "up-bad", "clock"),
        tile("at_risk", "At risk", at_risk, money(at_risk),
             "In deals scoring high or critical", "danger", "up-bad", "risk"),
        tile("open", "Team pipeline", open_b["gp"], money(open_b["gp"]),
             f"{open_b['opps']} deals · {pct(m['shareOfEntity'])} of the region",
             "accent", "up-good", "pipeline", trend(frame, "open_gp")),
        tile("stalled", "Stopped moving", m["stalledShare"], pct(m["stalledShare"]),
             f"{stalled['opps']} of {open_b['opps']} open deals",
             "danger" if m["stalledShare"] > 50 else "warn", "up-bad", "clock"),
        tile("team_win", "Team win rate", m["winRate"], pct(m["winRate"]),
             f"{m['winRateBasis']['won']} won of "
             f"{m['winRateBasis']['won'] + m['winRateBasis']['lost']}",
             "neutral", "up-good", "target", trend(frame, "win_rate")),
    ]


# --------------------------------------------------------------------------- #
# Executive
# --------------------------------------------------------------------------- #


def _exec(page: str, fs: FilterState, p: Principal, m: dict, c: dict) -> list[dict]:
    frame, risk = c["frame"], c["risk"]
    open_b, stalled = m["open"], m["stalled"]
    t = B.totals(fs, p)
    conc = ACC.concentration(fs, p)
    at_risk = float(risk.loc[risk["risk_band"].isin(("High", "Critical")), "acv_gp"].sum())

    if page == "performance":
        qs = B.by_quarter(fs, p)
        cur = next((q for q in qs if q["isCurrent"]), None)
        fwd = next((q for q in qs if q["isFuture"]), None)
        months = B.by_month(fs, p)
        past_months = [x for x in months if not x["isFuture"]]
        best = max(past_months, key=lambda x: x["wonGp"]) if past_months else None
        beat = sum(1 for x in past_months if x["attainmentPct"] >= 100)
        return [
            tile("won", "Won this year", t["wonGp"], money(t["wonGp"]),
                 f"Against a {money(t['budgetGp'])} plan", "good", "up-good",
                 "won", trend(frame, "won_gp")),
            tile("attain", "Of the plan", t["attainmentPct"], pct(t["attainmentPct"], 0),
                 "Delivered so far this year",
                 "good" if t["attainmentPct"] >= 100 else "warn", "up-good", "target"),
            tile("cover", "Pipeline cover", t["coverage"] or 0,
                 mult(t["coverage"]) if t["coverage"] else "plan met",
                 f"Open pipeline against what is left from {CUR_QUARTER}",
                 "good" if (t["coverage"] or 9) >= 1.5 else "danger", "up-good", "pipeline"),
            tile("this_q", "This quarter", cur["wonGp"] if cur else 0,
                 money(cur["wonGp"]) if cur else "—",
                 f"{pct(cur['attainmentPct'], 0)} of its plan" if cur else "—",
                 "good" if cur and cur["attainmentPct"] >= 100 else "warn",
                 "up-good", "calendar"),
            tile("next_q", "Next quarter", fwd["remainingGp"] if fwd else 0,
                 money(fwd["remainingGp"]) if fwd else "—",
                 f"{money(fwd['openGp'])} of pipeline behind it" if fwd else "Year is done",
                 "danger" if fwd and (fwd["coverage"] or 0) < 1 else "warn",
                 "up-bad", "calendar"),
            tile("best", "Best month", best["wonGp"] if best else 0,
                 money(best["wonGp"]) if best else "—",
                 best["month"] if best else "—", "accent", "neutral", "metric"),
        ]

    if page == "structure":
        lv = ACC.lob_count_value()
        by_lob = frame.groupby("lob")["acv_gp"].sum().sort_values(ascending=False)
        return [
            tile("revenue", "Revenue", m["total"]["revenue"], money(m["total"]["revenue"]),
                 f"{m['total']['lines']:,} deal lines", "accent", "up-good", "metric"),
            tile("margin", "Margin", m["blendedGm"], pct(m["blendedGm"]),
                 f"Services at {pct(m['servicesGm'])} against a "
                 f"{pct(m['servicesGmTarget'], 0)} target",
                 "warn" if m["servicesGmGap"] < 0 else "good", "up-good",
                 "margin", trend(frame, "gm")),
            tile("top_account", "Biggest customer", conc["topAccountShare"],
                 pct(conc["topAccountShare"]),
                 conc["accounts"][0]["account_name"] if conc["accounts"] else "—",
                 "danger" if conc["topAccountShare"] >= 12 else "warn", "up-bad", "account"),
            tile("top_five", "Top five together", conc["top5AccountShare"],
                 pct(conc["top5AccountShare"]), "Of all profit",
                 "danger" if conc["top5AccountShare"] >= 40 else "warn", "up-bad", "account"),
            tile("top_industry", "Biggest industry", conc["topIndustryShare"],
                 pct(conc["topIndustryShare"]),
                 conc["industries"][0]["industry"] if conc["industries"] else "—",
                 "warn", "up-bad", "pipeline"),
            tile("top_lob", "Biggest line", float(by_lob.iloc[0]) if len(by_lob) else 0,
                 money(float(by_lob.iloc[0])) if len(by_lob) else "—",
                 str(by_lob.index[0]) if len(by_lob) else "—",
                 "accent", "neutral", "pipeline"),
        ]

    if page == "risks":
        s = ANOM.summary()
        a = ANOM.for_persona("executive")
        stake = float(a["value_at_stake"].sum()) if len(a) else 0
        reps_flagged = int(a.loc[a["entity_type"] == "Rep", "entity_id"].nunique()) if len(a) else 0
        return [
            tile("findings", "Things to look at", s["total"], count(s["total"]),
                 f"Across {len(s['byCategory'])} kinds of problem",
                 "warn", "up-bad", "risk"),
            tile("critical", "Most serious", s["byPriority"].get("Critical", 0),
                 count(s["byPriority"].get("Critical", 0)),
                 "Worth looking at first", "danger", "up-bad", "risk"),
            tile("stake", "Money involved", stake, money(stake),
                 "Profit attached to these findings", "danger", "up-bad", "metric"),
            tile("at_risk", "Pipeline at risk", at_risk, money(at_risk),
                 f"{pct(m['stalledShare'])} of open value has stopped moving",
                 "danger", "up-bad", "clock"),
            tile("upside", "Good news", s["upside"], count(s["upside"]),
                 "Findings that are chances to sell, not problems",
                 "good", "up-good", "growth"),
            tile("reps", "People flagged", reps_flagged, count(reps_flagged),
                 "Patterns that repeat for one person",
                 "warn" if reps_flagged else "good", "up-bad", "people"),
        ]

    if page == "growth":
        xs = XS.summary(fs, p)
        th = XS.themes(fs, p)
        big = th[0] if th else None
        owners = len({o for t in th for o in t["owners"]})
        return [
            tile("plays", "Plays worth running", len(th), count(len(th)),
                 "Same offering missing at two or more customers",
                 "good", "up-good", "target"),
            tile("biggest", "Biggest play", big["accounts"] if big else 0,
                 big["offering"] if big else "—",
                 f"{big['accounts']} customers, {big['ownerCount']} owners"
                 if big else "Nothing repeats across customers",
                 "accent", "neutral", "growth"),
            tile("ideas", "Growth ideas", xs["recommendations"],
                 count(xs["recommendations"]),
                 f"Across {xs['accounts']} customers", "good", "up-good", "account"),
            tile("sure", "We are confident", xs["strong"], count(xs["strong"]),
                 f"{xs['veryHigh']} found by more than one method",
                 "good", "up-good", "metric"),
            tile("worth", "Peer value", xs["estimatedGp"], money(xs["estimatedGp"]),
                 "What peers earn on the same offerings — an order of magnitude, "
                 "not a forecast", "neutral", "neutral", "growth"),
            tile("owners", "People to brief", owners, count(owners),
                 "Account owners who would run these plays",
                 "accent", "neutral", "people"),
        ]

    if page == "actions":
        from . import actions as ACT

        cards = ACT.build(fs, p, limit=40)
        stake = sum(x["valueAtStake"] for x in cards)
        crit = sum(1 for x in cards if x["urgencyLabel"] == "Critical")
        upside = sum(1 for x in cards if x["framing"] == "opportunity")
        grid = B.coverage_grid(fs, p)
        return [
            tile("decisions", "Decisions waiting", len(cards), count(len(cards)),
                 "Each one names who owns it", "accent", "up-bad", "target"),
            tile("critical", "Cannot wait", crit, count(crit),
                 "Needs a call this week", "danger", "up-bad", "risk"),
            tile("stake", "Money involved", stake, money(stake),
                 "Across every decision on this page", "danger", "up-bad", "metric"),
            tile("holes", "Targets with no pipeline", grid["holes"], count(grid["holes"]),
                 f"of {len(grid['cells'])} areas in {grid['quarter']}",
                 "danger" if grid["holes"] else "good", "up-bad", "pipeline"),
            tile("cover", "Pipeline cover", t["coverage"] or 0,
                 mult(t["coverage"]) if t["coverage"] else "plan met",
                 "Against what is left of the plan",
                 "good" if (t["coverage"] or 9) >= 1.5 else "danger", "up-good", "target"),
            tile("upside", "Chances to sell", upside, count(upside),
                 "Decisions that are opportunities", "good", "up-good", "growth"),
        ]

    # tldr — the read
    return [
        tile("won", "Won this year", t["wonGp"], money(t["wonGp"]),
             f"{pct(t['attainmentPct'], 0)} of the {money(t['budgetGp'])} plan",
             "good" if t["attainmentPct"] >= 100 else "warn", "up-good",
             "won", trend(frame, "won_gp")),
        tile("open", "Open pipeline", open_b["gp"], money(open_b["gp"]),
             f"{open_b['opps']} deals still in play", "accent", "up-good",
             "pipeline", trend(frame, "open_gp")),
        tile("cover", "Pipeline cover", t["coverage"] or 0,
             mult(t["coverage"]) if t["coverage"] else "plan met",
             "Against what is left of the plan",
             "good" if (t["coverage"] or 9) >= 1.5 else "danger", "up-good", "target"),
        tile("at_risk", "At risk", at_risk, money(at_risk),
             f"{pct(m['stalledShare'])} of open value has stopped moving",
             "danger", "up-bad", "risk"),
        tile("top_account", "Biggest customer", conc["topAccountShare"],
             pct(conc["topAccountShare"]),
             conc["accounts"][0]["account_name"] if conc["accounts"] else "—",
             "danger" if conc["topAccountShare"] >= 12 else "warn", "up-bad", "account"),
        tile("margin", "Margin", m["blendedGm"], pct(m["blendedGm"]),
             f"Services at {pct(m['servicesGm'])} against {pct(m['servicesGmTarget'], 0)}",
             "warn" if m["servicesGmGap"] < 0 else "good", "up-good",
             "margin", trend(frame, "gm")),
    ]


def build(page: str, fs: FilterState, principal: Principal, m: dict) -> list[dict]:
    """The six tiles that answer THIS page's question."""
    c = _ctx(fs, principal)
    if principal.key == "ae":
        return _ae(page, fs, principal, m, c)
    if principal.key == "manager":
        return _manager(page, fs, principal, m, c)
    return _exec(page, fs, principal, m, c)
