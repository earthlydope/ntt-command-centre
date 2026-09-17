"""
Deterministic narrative — the words the product says when no model is involved.

This module exists for three reasons, in order of importance:

1. **The demo never shows an error.** If every LLM provider is down, rate-limited
   or slow, the page still reads as a finished product. Everything here is
   computed, so it cannot be wrong and cannot be late.
2. **It is the grounding source.** The figures the model is allowed to use are
   the figures this module has already formatted into strings. A model that
   never receives a raw number cannot divide one.
3. **It sets the register.** Professional, specific, and free of the hedging
   that makes generated prose recognisable. The client is presenting this to
   entity leadership at a Japanese-owned global IT services firm; "it appears
   that performance may be trending positively" is worse than silence.

Every sentence carries a `lens` tag — `cause`, `norm`, `delta`, `action`,
`answer` or `state`. The AI layer may only add a `state` sentence whose claim is
NOT already made by a chart on screen. These templates are written in complement
lenses wherever possible for the same reason.
"""

from __future__ import annotations

from . import accounts as ACC
from . import anomalies as ANOM
from . import budget as B
from . import crosssell as XS
from . import predict as P
from .loader import AS_OF, CUR_QUARTER
from .measures import FilterState, money, pct, slice_frame, subset
from .movement_features import STALL_DAYS
from .personas import Principal


def _s(text: str, lens: str, claim: str | None = None, bold: list[str] | None = None) -> dict:
    return {"text": text, "lens": lens, "claim": claim, "bold": bold or []}


def brief(fs: FilterState, principal: Principal, m: dict, page: str = "") -> dict:
    """
    The situation in three or four sentences, computed.

    Structured as: where we stand, what is wrong with it, what it is worth, and
    the one thing to decide. Each sentence states a fact and its consequence —
    a summary that only restates the KPI tiles above it has taken space and
    given nothing.
    """
    open_b = m["open"]
    past = m["pastDue"]
    stalled = m["stalled"]
    sentences: list[dict] = []

    # The brief answers THIS page's question. It used to be keyed on persona
    # alone, so the same four sentences sat above five different pages and the
    # top of the screen never changed as you moved down the navigation.
    page = page or principal.persona.home
    specific = _page_lead(page, fs, principal, m)
    if specific:
        return {"headline": specific["headline"], "sentences": specific["sentences"],
                "provider": "computed", "degraded": False}

    if principal.key == "ae":
        sentences.append(_s(
            f"You have {money(open_b['gp'])} of gross profit open across "
            f"{open_b['opps']} opportunities, which is "
            f"{pct(m['shareOfEntity'])} of North America's open pipeline.",
            "state", "gp.open.by:rep",
            [money(open_b["gp"]), pct(m["shareOfEntity"])]))
        if past["opps"]:
            sentences.append(_s(
                f"{past['opps']} of them are past their own close date, carrying "
                f"{money(past['gp'])} — {pct(m['pastDueShare'])} of everything you "
                f"have open.",
                "cause", "gp.pastdue.by:rep", [money(past["gp"])]))
        if stalled["opps"]:
            sentences.append(_s(
                f"{stalled['opps']} have had nothing logged for {STALL_DAYS} days or "
                f"more; the median silence across your open book is "
                f"{m['medianQuietDays']:.0f} days.",
                "norm", "quietdays.open.by:rep"))
        sentences.append(_s(
            "Start with the accounts carrying more than one flagged deal — those are "
            "one conversation each, not one per record.",
            "action"))

    elif principal.key == "manager":
        sentences.append(_s(
            f"The pod holds {money(open_b['gp'])} of open gross profit across "
            f"{open_b['opps']} opportunities, {pct(m['shareOfEntity'])} of the entity.",
            "state", "gp.open.by:rep", [money(open_b["gp"])]))
        sentences.append(_s(
            f"Win rate is {pct(m['winRate'])} on {m['winRateBasis']['won']} won and "
            f"{m['winRateBasis']['lost']} lost — but win rate does not vary "
            f"meaningfully by rep in this data, so it is a coverage input, not a "
            f"coaching one.",
            "norm", "winrate.closed.by:rep", [pct(m["winRate"])]))
        sentences.append(_s(
            f"{pct(m['stalledShare'])} of the pod's open value has stopped moving. "
            f"Behaviour is where the repeatable differences are: shrinkage, forecast "
            f"reversal and stalling all cluster by person.",
            "cause", "gp.stalled.by:rep"))
        sentences.append(_s(
            "Coach the behaviour that repeats, not the outcome that does not.",
            "action"))

    else:
        t = B.totals(fs, principal)
        fwd = [q for q in B.by_quarter(fs, principal) if q["isFuture"]]
        sentences.append(_s(
            f"FY26 gross profit stands at {money(t['wonGp'])} delivered against a "
            f"{money(t['budgetGp'])} plan, with {money(open_b['gp'])} still open.",
            "state", "gp.won.by:quarter", [money(t["wonGp"]), money(t["budgetGp"])]))
        if fwd:
            q = fwd[0]
            cov = q["coverage"]
            sentences.append(_s(
                f"{q['quarter']} carries {money(q['remainingGp'])} of plan against "
                f"{money(q['openGp'])} of open pipeline"
                + (f" — {cov:.2f}x coverage." if cov else ".")
                + " The quarter has not started, so this is a build decision rather "
                  "than a performance finding.",
                "cause", "coverage.open.by:quarter"))
        conc = ACC.concentration(fs, principal)
        if conc["accounts"]:
            sentences.append(_s(
                f"{conc['accounts'][0]['account_name']} alone is "
                f"{pct(conc['topAccountShare'])} of gross profit and the top five are "
                f"{pct(conc['top5AccountShare'])}; "
                f"{conc['industries'][0]['industry']} is "
                f"{pct(conc['topIndustryShare'])} of it.",
                "norm", "gp.all.by:account"))
        sentences.append(_s(
            f"{pct(m['stalledShare'])} of open value has not moved in {STALL_DAYS}+ "
            f"days — the coverage number above is softer than it looks.",
            "cause", "gp.stalled.by:quarter"))

    return {
        "headline": _headline(principal, m),
        "sentences": sentences,
        "provider": "computed",
        "degraded": False,
    }


def _page_lead(page: str, fs: FilterState, principal: Principal,
               m: dict) -> dict | None:
    """
    The computed brief for one page, in that page's own terms.

    Returns None for a persona's home page, which keeps the wider situation
    summary below — that IS the question the home page asks.
    """
    from .movement_features import features, rep_behaviour

    frame = slice_frame(fs, principal)
    codes = set(frame["opportunity_code"])
    risk = P.risk_table()
    risk = risk[risk["opportunity_code"].isin(codes)]

    if page == "my-deals":
        slipped = risk[risk["close_date_slips"] > 0]
        worst = risk.nlargest(1, "risk_score")
        out = [_s(f"{m['pastDue']['opps']} of your {m['open']['opps']} open deals are past the "
                  f"close date you committed to, carrying {money(m['pastDue']['gp'])}.",
                  "state", "gp.pastdue.by:rep")]
        if len(slipped):
            out.append(_s(f"{len(slipped)} have had the date moved already, "
                          f"{int(slipped['slip_days'].sum())} days later in total — a date that "
                          f"moves twice usually moves again.", "cause"))
        if len(worst):
            w = worst.iloc[0]
            out.append(_s(f"The worst is {w['opportunity_name']} at {w['account_name']}: "
                          f"{w['top_driver'].lower()}, scoring {int(w['risk_score'])}.", "norm"))
        out.append(_s("Re-date what is real and close what is not — a date nobody believes is "
                      "worse than no date.", "action"))
        return {"headline": "Which deals are slipping", "sentences": out}

    if page == "my-accounts":
        ws = ACC.whitespace(fs, principal, limit=999)
        xs = XS.summary(fs, principal)
        top = xs["topRecommendation"]
        if not top and not ws:
            return {"headline": "Nothing left to sell into", "sentences": [
                _s("Neither the whole-line view nor the recommendation engine finds a gap "
                   "at any of your customers.", "state"),
                _s("That is a good position and a limit: growth here comes from new "
                   "customers or bigger deals, not from selling more to these ones.",
                   "norm"),
            ]}
        out = []
        if top:
            out.append(_s(
                f"{xs['recommendations']} growth ideas across {xs['accounts']} of your "
                f"customers, {xs['strong']} of them ones we are confident about.",
                "state", "xsell.rec.by:account"))
            first = top["reasons"][0]["text"] if top["reasons"] else ""
            out.append(_s(
                f"The strongest is {top['offering']} at {top['accountName']} — "
                + (first[:180].rstrip(". ") + "." if first else
                   f"{top['confidence'].lower()} confidence."), "cause"))
        if ws:
            best = ws[0]
            out.append(_s(
                f"{len(ws)} of them do not buy one of our four lines at all; the clearest "
                f"is {best['recommendedLob']} at {best['accountName']}, which "
                f"{best['peerAttachRate'] * 100:.0f}% of similar customers take.", "norm"))
        else:
            out.append(_s(
                "None of them is missing a whole line of business — these are services "
                "to add inside lines they already buy, which is a shorter conversation.",
                "norm"))
        out.append(_s(
            f"Open the {top['offering'] if top else ws[0]['recommendedLob']} conversation "
            f"at {(top['accountName'] if top else ws[0]['accountName']).rstrip('. ')} "
            f"first. "
            + XS.CAVEAT, "action"))
        return {"headline": "Where you can sell more", "sentences": out}

    if page == "my-record":
        reps = rep_behaviour()
        peer = float(reps["win_rate"].median() * 100)
        return {"headline": "How you are doing", "sentences": [
            _s(f"You win {pct(m['winRate'])} of the deals you close, against a team middle of "
               f"{peer:.0f}%.", "norm", "winrate.closed.by:rep"),
            _s(f"That is {m['winRateBasis']['won']} won and {m['winRateBasis']['lost']} lost "
               f"this year, worth {money(m['won']['gp'])}.", "state"),
            _s("Win rate barely varies between reps in this data, so it is not the thing to "
               "judge yourself on. How long deals sit untouched is.", "cause"),
            _s(f"Your open deals have gone {m['medianQuietDays']:.0f} days without a change on "
               f"average.", "action"),
        ]}

    if page == "rep-benchmark":
        reps = rep_behaviour()
        members = principal.predicate.get("owner", ())
        pod = reps[reps["rep"].isin(members)] if members else reps
        credible = pod[~pod["thin"]]
        zc = ["shrink_rate_z", "inflate_rate_z", "regression_rate_z", "stall_rate_z"]
        off = credible[(credible[zc].abs() >= 1.5).any(axis=1)] if len(credible) else credible
        names = ", ".join(off["rep"].head(3)) if len(off) else ""
        return {"headline": "Who is off the pattern", "sentences": [
            _s(f"{len(off)} of your {len(pod)} reps sit more than one and a half standard "
               f"deviations from the team on at least one behaviour"
               + (f": {names}." if names else "."), "state"),
            _s("Each rate is judged against the team's own norm, not a fixed target, so "
               "'unusual' means unusual here rather than unusual in general.", "norm"),
            _s("Win rate is not one of the behaviours worth coaching — it barely moves between "
               "reps in this data. Shrinking, sandbagging, walking a forecast back and letting "
               "deals sit do.", "cause"),
            _s("Start with whoever is furthest out on the behaviour that costs you most.",
               "action"),
        ]}

    if page == "process":
        mv = features()
        mv = mv[mv["opportunity_code"].isin(codes)]
        entered = mv[mv["stage_path_str"].str.startswith("Identification", na=False)]
        reached = entered[entered["stage_path_str"].str.contains("Finalist", regex=False, na=False)]
        conv = 100 * len(reached) / len(entered) if len(entered) else 0
        skipped = int((mv["skipped_stages"] > 0).sum())
        return {"headline": "Where deals fall out", "sentences": [
            _s(f"{pct(conv, 0)} of the deals that start at the beginning reach the last stage — "
               f"{len(reached)} of {len(entered)}.", "state", "count.all.by:stage"),
            _s("Almost nothing is lost in the middle of the process. Deals are lost at the end, "
               "which means qualification is not the problem here — closing is.", "cause"),
            _s(f"{skipped} deals have no record of the stages they passed through, so their "
               f"history cannot be checked.", "norm"),
            _s("Look at what happens between the final stage and the decision, not at the "
               "early funnel.", "action"),
        ]}

    if page == "calibration":
        reps = rep_behaviour()
        members = principal.predicate.get("owner", ())
        pod = reps[reps["rep"].isin(members)] if members else reps
        credible = pod[~pod["thin"]]
        sh = credible[credible["shrink_rate_z"] >= 1.5]
        return {"headline": "Whose numbers to trust", "sentences": [
            _s(f"{len(sh)} of your reps shrink their deals materially more often than the rest "
               f"of the team" + (f" — {', '.join(sh['rep'].head(2))}." if len(sh) else "."),
               "state"),
            _s("One deal shrinking is normal. The same person shrinking repeatedly is a sizing "
               "habit, and it is the difference between a forecast you can add up and one you "
               "cannot.", "cause"),
            _s(f"{pct(m['stalledShare'])} of the team's open value has not moved in 60 days, so "
               f"the pipeline total is softer than it looks whoever owns it.", "norm"),
            _s("Discount the numbers from the reps above before you roll the team up.",
               "action"),
        ]}

    if page == "pod-whitespace":
        ws = ACC.whitespace(fs, principal, limit=999)
        xs = XS.summary(fs, principal)
        th = XS.themes(fs, principal)
        if not ws and not th and not xs["recommendations"]:
            return None
        out = [_s(
            f"{xs['recommendations']} growth ideas sit with your team across "
            f"{xs['accounts']} customers, {xs['strong']} of them ones we are confident "
            f"about.", "state", "xsell.rec.by:account")]
        if th:
            t = th[0]
            out.append(_s(
                f"{t['accounts']} of those customers are missing the same thing — "
                f"{t['offering']} — spread across {t['ownerCount']} owners. That is one "
                f"briefing, not {t['accounts']} conversations.", "cause"))
        if ws:
            best = ws[0]
            out.append(_s(
                f"Separately, {len(ws)} customers do not buy one of our four lines at "
                f"all; the clearest is {best['recommendedLob']} at "
                f"{best['accountName']}, owned by {best['owner']}.", "norm"))
        out.append(_s("Assign each with a name and a date. " + XS.CAVEAT, "action"))
        return {"headline": "What the team should sell next", "sentences": out}

    if page == "growth":
        th = XS.themes(fs, principal)
        xs = XS.summary(fs, principal)
        if not th:
            return {"headline": "No repeating growth theme", "sentences": [
                _s(f"{xs['recommendations']} growth ideas exist, but no offering is missing "
                   f"at two or more customers.", "state"),
                _s("That makes these account conversations rather than a campaign — hand "
                   "them to owners individually.", "action"),
            ]}
        t = th[0]
        second = th[1] if len(th) > 1 else None
        return {"headline": "Where growth repeats", "sentences": [
            _s(f"{t['accounts']} customers are missing the same offering — {t['offering']} "
               f"— which is the largest of {len(th)} repeating themes.",
               "state", "xsell.theme.by:offering"),
            _s(f"They sit with {t['ownerCount']} different owners, so it is a briefing and "
               f"a target list rather than {t['accounts']} separate conversations."
               + (f" The next largest is {second['offering']} at {second['accounts']} "
                  f"customers." if second else ""), "cause"),
            _s(f"{xs['veryHigh']} of all {xs['recommendations']} ideas were found "
               f"independently by more than one method, which is the strongest the engine "
               f"gets. {xs['strong']} clear the confidence bar.", "norm"),
            _s(f"Launch {t['offering']} as a named play and pilot it with "
               f"{t['strongest']['accountName'].rstrip('. ')}. " + XS.CAVEAT, "action"),
        ]}

    if page == "performance":
        qs = B.by_quarter(fs, principal)
        fwd = [q for q in qs if q["isFuture"]]
        t = B.totals(fs, principal)
        out = [_s(f"{money(t['wonGp'])} delivered against a {money(t['budgetGp'])} plan — "
                  f"{pct(t['attainmentPct'], 0)} of the year.", "state",
                  "gp.won.by:quarter")]
        if fwd:
            q = fwd[0]
            out.append(_s(f"{q['quarter']} has {money(q['remainingGp'])} of plan and "
                          f"{money(q['openGp'])} of pipeline behind it. The quarter has not "
                          f"started, so this is a build decision, not a miss.", "cause"))
        out.append(_s("The plan was spread evenly across three quarters while every win so far "
                      "sits in the first two, which is why the early quarters read above "
                      "target and the last one reads below.", "norm"))
        out.append(_s("Decide now between funding pipeline and re-phasing the target.",
                      "action"))
        return {"headline": "Against the plan", "sentences": out}

    if page == "structure":
        conc = ACC.concentration(fs, principal)
        return {"headline": "Where the money sits", "sentences": [
            _s(f"{conc['accounts'][0]['account_name']} is {pct(conc['topAccountShare'])} of all "
               f"profit and the top five are {pct(conc['top5AccountShare'])}.", "state",
               "gp.all.by:account"),
            _s(f"{conc['industries'][0]['industry']} alone is "
               f"{pct(conc['topIndustryShare'])}, so the book is exposed to one sector's "
               f"budget cycle as well as to a handful of customers.", "norm"),
            _s(f"Margin is {pct(m['blendedGm'])} blended, and the largest line of business is "
               f"also the thinnest — volume is not where the profit is.", "cause"),
            _s("Set an attach target on the biggest line rather than chasing more of it.",
               "action"),
        ]}

    if page == "risks":
        summ = ANOM.summary()
        top = summ["byCategory"][0] if summ["byCategory"] else None
        return {"headline": "What needs fixing", "sentences": [
            _s(f"{summ['total']} findings, {summ['byPriority'].get('Critical', 0)} of them "
               f"serious enough to look at first.", "state", "sev.flagged.by:anomcat"),
            _s(f"The largest group is {top['category'].lower()} — {top['question']}"
               if top else "", "cause"),
            _s(f"{summ['corroborated']} were flagged independently by both the data-science "
               f"model and this layer's own checks, which is the strongest a finding gets.",
               "norm"),
            _s(f"{summ['upside']} of these are chances to sell rather than problems — read them "
               f"that way.", "action"),
        ]}

    if page == "actions":
        from . import actions as ACT

        cards = ACT.build(fs, principal, limit=40)
        crit = [c for c in cards if c["urgencyLabel"] == "Critical"]
        return {"headline": "What needs deciding", "sentences": [
            _s(f"{len(cards)} decisions are waiting, {len(crit)} of which cannot wait a week.",
               "state"),
            _s(crit[0]["headline"] + " — " + crit[0]["why"][0] if crit
               else "Nothing is urgent enough to interrupt the quarter.", "cause"),
            _s("Every one names who owns it and the rule that raised it, so none of them needs "
               "a meeting to establish the facts.", "norm"),
            _s("Take the top three; the rest can wait for the review.", "action"),
        ]}

    return None


def _headline(principal: Principal, m: dict) -> str:
    if principal.key == "ae":
        return f"Good morning, {principal.identity.split()[0]}"
    if principal.key == "manager":
        return principal.identity_label
    return "North America — FY26 position"


def explain_card(card: dict) -> dict:
    """The deterministic version of 'why is this flagged'."""
    return {
        "headline": card.get("headline", ""),
        "sentences": [
            _s(w, "cause") for w in card.get("why", [])
        ] + [
            _s(f"This fired because {card.get('predicate', 'a rule matched')}.", "cause"),
            _s(card.get("nextStep", ""), "action"),
        ],
        "provider": "computed",
        "degraded": False,
    }


def facts_pack(fs: FilterState, principal: Principal, m: dict) -> dict:
    """
    The pre-formatted figures the LLM is allowed to use, as STRINGS.

    Raw floats never cross into a prompt. A model that cannot see 810234.55
    cannot divide it by anything, and a model that writes "$805K" is copying a
    value this layer computed rather than producing one of its own.
    """
    t = B.totals(fs, principal)
    conc = ACC.concentration(fs, principal)
    anom = ANOM.summary()
    _xs = XS.summary(fs, principal)
    risk = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    risk = risk[risk["opportunity_code"].isin(codes)]

    # Keys are HUMAN LABELS, not field names. A model handed `pastDueShare`
    # writes "pastDueShare has climbed to 73.8%", which reads as machine output
    # in front of entity leadership. Handed "share of open pipeline past its
    # close date" it writes the sentence a person would.
    return {
        "as of date": AS_OF.isoformat(),
        "current fiscal quarter": CUR_QUARTER,
        "scope shown": m["scope"]["label"],
        "viewing as": principal.persona.label,
        "open pipeline gross profit": money(m["open"]["gp"]),
        "open pipeline revenue": money(m["open"]["revenue"]),
        "number of open opportunities": f"{m['open']['opps']:,}",
        "gross profit won this year": money(m["won"]["gp"]),
        "number of opportunities won": f"{m['won']['opps']:,}",
        "total gross profit in scope": money(m["total"]["gp"]),
        "total revenue in scope": money(m["total"]["revenue"]),
        "gross profit past its close date": money(m["pastDue"]["gp"]),
        "number of deals past their close date": f"{m['pastDue']['opps']:,}",
        "share of open pipeline past its close date": pct(m["pastDueShare"]),
        "gross profit that has stopped moving": money(m["stalled"]["gp"]),
        "number of deals that have stopped moving": f"{m['stalled']['opps']:,}",
        "share of open pipeline that has stopped moving": pct(m["stalledShare"]),
        "median days since a deal was last touched": f"{m['medianQuietDays']:.0f} days",
        "win rate": pct(m["winRate"]),
        "deals won": f"{m['winRateBasis']['won']:,}",
        "deals lost": f"{m['winRateBasis']['lost']:,}",
        "blended gross margin": pct(m["blendedGm"]),
        "services gross margin": pct(m["servicesGm"]),
        "services gross margin target": pct(m["servicesGmTarget"]),
        "this scope as a share of North America": pct(m["shareOfEntity"]),
        "largest account": conc["accounts"][0]["account_name"] if conc["accounts"] else "n/a",
        "largest account share of gross profit": pct(conc["topAccountShare"]),
        "top five accounts share of gross profit": pct(conc["top5AccountShare"]),
        "largest industry": conc["industries"][0]["industry"] if conc["industries"] else "n/a",
        "largest industry share of gross profit": pct(conc["topIndustryShare"]),
        "number of findings": f"{anom['total']:,}",
        "number of critical findings": f"{anom['byPriority'].get('Critical', 0):,}",
        "number of findings framed as upside": f"{anom['upside']:,}",
        "deals scoring critical on risk": f"{int((risk['risk_band'] == 'Critical').sum()):,}",
        "deals scoring high on risk": f"{int((risk['risk_band'] == 'High').sum()):,}",
        "risk weighted value of open pipeline": money(float(risk["value_at_risk"].sum())),
        "days of silence that counts as stalled": f"{STALL_DAYS} days",
        # Growth. Peer value is labelled as PEER value in the key itself, so a
        # model cannot quote it as pipeline without contradicting the label it
        # was handed.
        **({
            "number of cross-sell recommendations": f"{_xs['recommendations']:,}",
            "customers with a cross-sell recommendation": f"{_xs['accounts']:,}",
            "cross-sell recommendations we are confident about": f"{_xs['strong']:,}",
            "cross-sell recommendations found by more than one method":
                f"{_xs['veryHigh']:,}",
            "number of repeating growth plays": f"{_xs['themes']:,}",
            "largest growth play": _xs["topTheme"] or "none",
            "customers in the largest growth play": f"{_xs['topThemeAccounts']:,}",
            "what peer accounts earn on these offerings": money(_xs["estimatedGp"]),
        } if _xs["recommendations"] else {}),
        # Plan figures reach the model ONLY for an unscoped principal. The
        # extract carries no rep or pod quota, so a pod's "attainment" is its
        # own contribution measured against the whole entity's target — a number
        # that is arithmetically fine and editorially false, and a model handed
        # it will write "the pod has delivered 26% of plan", which is not a
        # thing that is true of anybody.
        **({
            "full year plan gross profit": money(t["budgetGp"]),
            "attainment against plan": pct(t["attainmentPct"]),
            "plan gross profit still to deliver": money(t["remainingGp"]),
            "pipeline coverage of remaining plan": (
                f"{t['coverage']:.2f}x" if t["coverage"] else "plan already met"),
            # The forward quarter is the decision on this page, so its own two
            # figures belong in the pack rather than only inside an action card.
            **{
                f"{q['quarter']} plan gross profit remaining": money(q["remainingGp"])
                for q in B.by_quarter(fs, principal) if q["isFuture"]
            },
            **{
                f"{q['quarter']} open pipeline gross profit": money(q["openGp"])
                for q in B.by_quarter(fs, principal) if q["isFuture"]
            },
        } if not principal.predicate else {
            "note on plan": (
                "There is no quota for this scope in the source data — plan exists "
                "only for the whole entity, so do not describe this scope's "
                "attainment or coverage."),
        }),
    }
