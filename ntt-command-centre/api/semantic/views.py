"""
Page assembly — fourteen pages across three personas, one payload shape.

Every page answers exactly ONE stated question and is built from the same parts:
a question, KPI tiles, a narrative, action cards, charts, and the evidence
behind them. That repetition is what makes fourteen pages tractable; each page
then adds only the block that answers its own question.

**The personas do not share pages.** An AE's "My Day" and an executive's "TLDR"
are not the same screen with a different filter — they have different questions,
different tiles, different charts and different verbs. A persona that reaches a
page it does not own gets redirected to its own home rather than shown a page
built for someone else's job.

`chartsSay` at the payload root is the union of every chart's claim keys. The AI
layer reads it and is forbidden from restating any of them. That is the
enforcement point for "the AI must not repeat the charts".
"""

from __future__ import annotations

from . import accounts as ACC
from . import actions as ACT
from . import anomalies as ANOM
from . import budget as B
from . import charts as C
from . import crosssell as XS
from . import ds_model
from . import narrative as N
from . import predict as P
from .dimensions import describe as describe_dims
from .loader import AS_OF, CUR_QUARTER, distribution_check, fy_label, report
from .measures import (
    FilterState,
    count,
    measures,
    money,
    mult,
    pct,
    rls_frame,
    slice_frame,
    subset,
)
from .movement_features import STALL_DAYS
from .personas import PERSONAS, Principal

#: page -> (persona, question). The single source of truth for navigation.
PAGES: dict[str, tuple[str, str, str]] = {
    # AE
    "my-day": ("ae", "Today", "What should I do first?"),
    "my-deals": ("ae", "My deals", "Which deals are slipping?"),
    "my-accounts": ("ae", "My customers", "Where can I sell more?"),
    "my-record": ("ae", "My record", "How am I doing?"),
    # Manager
    "pod-pulse": ("manager", "My team", "Who needs me this week?"),
    "rep-benchmark": ("manager", "Compare reps", "Who is off the pattern?"),
    "process": ("manager", "Where we lose", "Where do deals fall out?"),
    "calibration": ("manager", "Whose numbers", "Whose forecast can I trust?"),
    "pod-whitespace": ("manager", "Grow accounts", "What should the team sell next?"),
    # Executive
    "tldr": ("executive", "Brief", "What do I need to know?"),
    "performance": ("executive", "Against plan", "Are we on track?"),
    "structure": ("executive", "The business", "Where does the money sit?"),
    "risks": ("executive", "What is wrong", "What needs fixing, and what is it worth?"),
    "growth": ("executive", "Where to grow", "Which growth ideas repeat often enough to run as a play?"),
    "actions": ("executive", "Decisions", "What needs deciding now?"),
}


def pages_for(persona: str) -> list[dict]:
    return [
        {"key": k, "label": v[1], "question": v[2]}
        for k, v in PAGES.items() if v[0] == persona
    ]


def resolve_page(page: str, principal: Principal) -> str:
    """A persona that asks for someone else's page lands on its own home."""
    if page in PAGES and PAGES[page][0] == principal.key:
        return page
    return principal.persona.home


# --------------------------------------------------------------------------- #
# KPI tiles
# --------------------------------------------------------------------------- #


def kpis(page: str, fs: FilterState, principal: Principal, m: dict) -> list[dict]:
    """
    The six tiles that answer THIS page's question.

    Delegated to `tiles.py`. These used to be chosen by persona alone, which
    meant every page of a persona showed the same six numbers and moving down
    the navigation changed nothing above the fold — the charts changed, but they
    were below it. A page states one question; its tiles answer that question.
    """
    from . import tiles

    return tiles.build(page, fs, principal, m)


# --------------------------------------------------------------------------- #
# Charts per page
# --------------------------------------------------------------------------- #


def charts_for(page: str, fs: FilterState, principal: Principal) -> list[dict]:
    """
    Which charts answer THIS page's question.

    Two rules, both checked by `verify.py`:

    **Within a persona, a chart id appears on exactly one page.** The customer's
    words were that the same chart on two of a persona's pages reads as the
    application not changing when the navigation does. Where one chart was the
    right answer to two questions it stays on the page whose question it
    answers best, and the other page gets a chart that answers ITS question —
    the executive's brief keeps the plan-to-position bridge and the risk bands
    (the landing page must show the closure use case as well as the plan); the
    risks page keeps the anomaly categories and takes the overdue stack, which
    is what "what is it worth" looks like deal by deal; the decisions page
    carries the coverage bullets, because a decision is made a row at a time
    and the heat grid on the performance page is made for finding the hole,
    not for funding it. The manager's pod pulse hands the fingerprint grid to
    the benchmark page, whose question it is, and takes the stalled-by-rep bar
    instead — the pulse asks who needs me, and a band has no phone number.

    **Every page keeps two or three charts.** Across personas the same chart may
    repeat only when the rows and the question both differ: an AE's risk bands
    are their own deals, a manager's the pod's, and the same rows never appear
    twice anywhere.
    """
    p = principal
    if page == "my-day":
        return [C.deal_triage_bubble(fs, p), C.risk_by_band(fs, p)]
    if page == "my-deals":
        return [C.deal_gantt(fs, p), C.ageing_stack(fs, p), C.open_by_dimension(fs, p, "stage")]
    if page == "my-accounts":
        return [C.cross_sell_list(fs, p), C.whitespace_table(fs, p),
                C.account_treemap(fs, p)]
    if page == "my-record":
        return [C.won_by_month(fs, p), C.stage_funnel(fs, p),
                C.open_by_dimension(fs, p, "portfolio")]

    if page == "pod-pulse":
        return [C.risk_by_band(fs, p), C.stalled_by_rep(fs, p)]
    if page == "rep-benchmark":
        return [C.rep_benchmark_heat(fs, p), C.open_by_dimension(fs, p, "rep")]
    if page == "process":
        return [C.stage_funnel(fs, p), C.stage_path_flow(fs, p)]
    if page == "calibration":
        return [C.deal_triage_bubble(fs, p), C.ageing_stack(fs, p)]
    if page == "pod-whitespace":
        return [C.cross_sell_list(fs, p), C.whitespace_table(fs, p),
                C.margin_mekko(fs, p)]

    if page == "tldr":
        return [C.gp_bridge(fs, p), C.risk_by_band(fs, p)]
    if page == "performance":
        return [C.month_vs_plan(fs, p), C.coverage_heat(fs, p), C.won_by_month(fs, p)]
    if page == "structure":
        return [C.margin_mekko(fs, p), C.account_treemap(fs, p), C.industry_flow(fs, p)]
    if page == "risks":
        return [C.anomaly_by_category(fs, p), C.ageing_stack(fs, p), C.stage_path_flow(fs, p)]
    if page == "growth":
        return [C.cross_sell_themes(fs, p), C.cross_sell_matrix(fs, p)]
    if page == "actions":
        return [C.coverage_bullet(fs, p, "lob"), C.coverage_bullet(fs, p, "portfolio")]
    return []


# --------------------------------------------------------------------------- #
# Page-specific blocks
# --------------------------------------------------------------------------- #


def extras_for(page: str, fs: FilterState, principal: Principal) -> dict:
    """The one block that makes this page itself rather than a generic lens."""
    p = principal
    if page == "my-deals":
        r = P.risk_table()
        codes = set(slice_frame(fs, p)["opportunity_code"])
        r = r[r["opportunity_code"].isin(codes)].head(25)
        return {"deals": [
            {"opportunityCode": row.opportunity_code, "name": row.opportunity_name,
             "account": row.account_name, "stage": row.stage, "lob": row.lob,
             "gp": float(row.acv_gp), "revenue": float(row.acv_revenue),
             "riskScore": int(row.risk_score), "riskBand": row.risk_band,
             "topDriver": row.top_driver,
             "quietDays": int(row.quiet_days) if row.quiet_days == row.quiet_days else None,
             "closeDate": row.close_date.date().isoformat() if row.close_date == row.close_date else None,
             "factors": row.risk_factors}
            for row in r.itertuples(index=False)]}

    if page in ("my-accounts", "pod-whitespace"):
        return {"whitespace": ACC.whitespace(fs, p, limit=25),
                "lobValue": ACC.lob_count_value().to_dict("records"),
                "attach": ACC.attach_matrix().to_dict("records"),
                # The data-science recommendations sit BESIDE the computed
                # whitespace rather than replacing it: they are finer-grained
                # (a service inside a line of business, not the whole line) and
                # they carry written reasons this layer cannot produce.
                "crossSell": XS.unified(fs, p, limit=25),
                "crossSellSummary": XS.summary(fs, p)}

    if page == "growth":
        return {"themes": XS.themes(fs, p),
                "crossSell": XS.unified(fs, p, limit=40),
                "crossSellSummary": XS.summary(fs, p)}

    if page == "my-record":
        return {"segmentRates": P.segment_base_rates().to_dict("records")}

    if page in ("rep-benchmark", "calibration", "pod-pulse"):
        from .movement_features import rep_behaviour

        reps = rep_behaviour()
        members = p.predicate.get("owner", ())
        pod = reps[reps["rep"].isin(members)] if members else reps
        return {"reps": pod.round(4).to_dict("records"),
                "peerNorms": {c: float(reps[c].median()) for c in
                              ("win_rate", "shrink_rate", "inflate_rate",
                               "regression_rate", "stall_rate", "new_business_share")}}

    if page == "process":
        return {"funnelNote": (
            "Anchored on the entry cohort. Counting every opportunity at every stage "
            "reads above 100% partway down, because many deals are logged straight "
            "into the middle of the ladder.")}

    if page == "performance":
        return {"quarters": B.by_quarter(fs, p), "months": B.by_month(fs, p),
                "totals": B.totals(fs, p), "coverageByLob": B.coverage_by(fs, p, "lob"),
                "coverageByPortfolio": B.coverage_by(fs, p, "portfolio")}

    if page == "structure":
        return {"concentration": ACC.concentration(fs, p),
                "lobValue": ACC.lob_count_value().to_dict("records")}

    if page == "risks":
        a = ANOM.for_persona(p.key)
        return {"anomalySummary": ANOM.summary(),
                "findings": a.head(60).replace({float("nan"): None}).to_dict("records"),
                "modelCard": P.model_card()}

    if page == "tldr":
        return {"modelCard": P.model_card(),
                "quarters": B.by_quarter(fs, p),
                "anomalySummary": ANOM.summary(),
                "useCases": use_cases(fs, p)}
    return {}


def use_cases(fs: FilterState, principal: Principal) -> list[dict]:
    """
    The three things this platform watches, stated on the landing page.

    The client asked to verify that all three use cases from the 16 Sep call
    are implemented. An executive should not have to know which page proves
    which; the brief names each one, gives it two or three figures that come
    from the same functions the dedicated pages draw from, and opens the page.
    Figures arrive formatted, because the front end computes nothing.
    """
    xs = XS.summary(fs, principal)
    an = ANOM.summary()
    card = P.model_card()["primary"]
    risk = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    risk = risk[risk["opportunity_code"].isin(codes)]
    hot = risk[risk["risk_band"].isin(("High", "Critical"))]
    auc = card.get("testAuc")
    return [
        {
            "key": "opportunities",
            "title": "Cross-sell and upsell",
            "oneLine": "Where an account buys less than accounts like it, "
                       "sized against size-matched peers.",
            "figures": [
                {"label": "Recommendations", "formatted": count(xs["recommendations"])},
                {"label": "Accounts", "formatted": count(xs["accounts"])},
                {"label": "Repeating plays", "formatted": count(xs["themes"])},
            ],
            "page": "growth",
            "cta": "Open the growth plays",
        },
        {
            "key": "anomalies",
            "title": "Anomaly detection",
            "oneLine": "The data-science findings, re-detected live from the change "
                       "log and routed to whoever can act.",
            "figures": [
                {"label": "Findings", "formatted": count(an["total"])},
                {"label": "Critical", "formatted": count(an["byPriority"]["Critical"])},
                {"label": "Corroborated by both detectors",
                 "formatted": count(an["corroborated"])},
            ],
            "page": "risks",
            "cta": "Open the findings",
        },
        {
            "key": "closure",
            "title": "Deal closure likelihood",
            "oneLine": "The closure model's probability, its honest strength, and the "
                       "deal-risk score the product acts on.",
            "figures": [
                {"label": "Open deals scored", "formatted": count(risk["opportunity_code"].nunique())},
                {"label": "High or critical risk",
                 "formatted": f"{count(hot['opportunity_code'].nunique())} · "
                              f"{money(float(hot['acv_gp'].sum()))}"},
                {"label": "Model AUC",
                 "formatted": f"{auc:.3f} against 0.50 for chance"
                 if isinstance(auc, (int, float)) else "not loaded"},
            ],
            "page": "risks",
            "cta": "Open the risk view",
        },
    ]


# --------------------------------------------------------------------------- #
# The payload
# --------------------------------------------------------------------------- #


def view(page: str, fs: FilterState, principal: Principal) -> dict:
    page = resolve_page(page, principal)
    persona, label, question = PAGES[page]
    m = measures(fs, principal)
    ch = charts_for(page, fs, principal)
    cards = ACT.build(fs, principal,
                      limit=8 if page in ("my-day", "pod-pulse", "tldr", "actions") else 4,
                      page=page)

    says: list[str] = []
    for c in ch:
        for s in c.get("says", []):
            if s not in says:
                says.append(s)

    return {
        "page": page,
        "label": label,
        "question": question,
        "persona": persona,
        "asOf": AS_OF.isoformat(),
        "fy": fy_label(2026),
        "quarter": CUR_QUARTER,
        "scope": m["scope"],
        "filters": fs.active(),
        "measure": fs.measure,
        "kpis": kpis(page, fs, principal, m),
        "narrative": N.brief(fs, principal, m, page),
        "actions": cards,
        "charts": ch,
        "chartsSay": says,
        "extras": extras_for(page, fs, principal),
        "measures": m,
    }


def meta(principal: Principal) -> dict:
    """Everything the shell needs once: navigation, dimensions, provenance."""
    from . import personas as PR

    r = report()
    return {
        "asOf": AS_OF.isoformat(),
        "fy": fy_label(2026),
        "quarter": CUR_QUARTER,
        "persona": PR.describe(principal),
        "pages": pages_for(principal.key),
        "allPages": [{"key": k, "persona": v[0], "label": v[1], "question": v[2]}
                     for k, v in PAGES.items()],
        # Dimension values are listed from the caller's OWN rows. Described
        # off the whole fact table, a manager's rep list said "All 70" for a
        # pod of eleven and an AE's account list offered 357 accounts of which
        # they own a dozen — every other value was a filter to an empty page.
        "dimensions": describe_dims(rls_frame(principal)),
        "measures": [{"key": "gp", "label": "ACV GP", "default": True},
                     {"key": "revenue", "label": "ACV Revenue", "default": False}],
        "data": {
            "lines": r.lines, "opportunities": r.opportunities,
            "accounts": r.accounts, "reps": r.reps,
            "movementRows": r.movement_rows, "anomalies": r.anomalies,
            "orphanMovement": r.orphan_movement_opps,
            "orphanAnomalies": r.orphan_anomaly_opps,
            "closeBeforeCreate": r.close_before_create,
        },
        "provenance": {
            "distribution": distribution_check().to_dict("records"),
            "note": (
                "Synthetic data generated against the client's own distribution "
                "workbook. Every categorical mix matches the given distribution to "
                "within 0.1 percentage points — the table above is the generator's "
                "own validation, carried through rather than summarised."
            ),
        },
        "modelCard": P.model_card(),
        "dsModel": ds_model.model_card() if ds_model.available()
        else ds_model.unavailable_card(),
        "anomalyTaxonomy": {
            "categories": [{"name": c, "question": ANOM.CATEGORY_BLURB[c]}
                           for c in ANOM.CATEGORY_ORDER],
        },
        "stallThreshold": STALL_DAYS,
    }
