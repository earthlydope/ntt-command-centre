"""
The regression harness. `./run.sh verify`.

Three things are asserted here and nowhere else:

1. **The figures.** Every headline number the demo will say out loud is pinned.
   If a refactor moves one, this fails rather than the customer noticing.
2. **The three use cases** the client asked for on the 16 Sep call — anomaly
   detection, deal-closure likelihood with red/green feature benchmarking, and
   cross-sell/upsell identification — are each checked end to end, from the
   source file through to a rendered payload.
3. **The guarantees**, rather than just the values: that row-level security
   actually narrows, that a persona cannot reach another persona's page, that
   every chart declares a claim key, that the AI cannot restate a chart, and
   that nothing in the product silently computes a margin as an average.

Exit code is the number of failures, so CI can gate on it.
"""

from __future__ import annotations

import os

# The harness pins the business date BEFORE the semantic layer is imported:
# every figure it asserts was taken on 2026-09-15, and the product otherwise
# follows the calendar (see config.AS_OF).
os.environ.setdefault("NTT_AS_OF", "2026-09-15")

import sys

from ..semantic import accounts as ACC
from ..semantic import actions as ACT
from ..semantic import anomalies as ANOM
from ..semantic import budget as B
from ..semantic import catalog as CAT
from ..semantic import charts as C
from ..semantic import crosssell as XS
from ..semantic import ds_model as DS
from ..semantic import measures as M
from ..semantic import personas as PR
from ..semantic import predict as P
from ..semantic import views as V
from ..semantic.loader import AS_OF, facts, movement, opportunities, report
from ..semantic.movement_features import features, rep_behaviour

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def check(name: str, got, want, tol: float | None = None) -> None:
    if tol is not None:
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    print(f"[{PASS if ok else FAIL}] {name:52s} got {got!r}"
          + ("" if ok else f"  want {want!r}"))
    if not ok:
        _failures.append(name)


def assert_true(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{PASS if cond else FAIL}] {name:52s} {detail}")
    if not cond:
        _failures.append(name)


def main() -> int:
    exec_p = PR.resolve("executive")
    fs = M.FilterState()

    print("\n── DATA ─────────────────────────────────────────────────────────────")
    r = report()
    check("as-of is pinned, never today()", AS_OF.isoformat(), "2026-09-15")
    check("opportunity lines", r.lines, 3034)
    check("opportunities", r.opportunities, 2050)
    check("accounts", r.accounts, 363)
    check("reps", r.reps, 70)
    check("movement rows", r.movement_rows, 36631)
    check("anomaly findings", r.anomalies, 485)
    check("movement orphans", r.orphan_movement_opps, 0)
    check("anomaly orphans", r.orphan_anomaly_opps, 0)
    check("close date before create date", r.close_before_create, 0)

    f = facts()
    check("recomputed fiscal quarter matches the export",
          int((f["fiscal_quarter"] != f["quarter"]).sum()), 0)
    check("recomputed fiscal month matches the export",
          int((f["fiscal_month"] != f["month"]).sum()), 0)

    print("\n── MEASURES ─────────────────────────────────────────────────────────")
    m = M.measures(fs, exec_p)
    check("total ACV revenue ($M)", round(m["total"]["revenue"] / 1e6, 2), 58.85)
    check("total ACV GP ($M)", round(m["total"]["gp"] / 1e6, 2), 9.55)
    check("blended GM %", round(m["blendedGm"], 2), 16.23)
    check("open lines", m["open"]["lines"], 350)
    check("open opportunities", m["open"]["opps"], 235)
    check("won opportunities", m["won"]["opps"], 682)
    check("win rate %", round(m["winRate"], 1), 37.6)
    check("past-due lines", m["pastDue"]["lines"], 256)
    check("stalled opportunities", m["stalled"]["opps"], 113)

    # The margin rule, stated as a difference rather than a value: an average of
    # percentages and a revenue-weighted margin must NOT agree, or the guard has
    # stopped guarding anything.
    assert_true("weighted GM differs from the average of GM%",
                abs(m["blendedGm"] - m["naiveGm"]) > 1.0,
                f"weighted {m['blendedGm']:.2f}% vs naive {m['naiveGm']:.2f}%")

    print("\n── ROW-LEVEL SECURITY ───────────────────────────────────────────────")
    ae = PR.resolve("ae", "Brian Thompson")
    mgr = PR.resolve("manager", "pod-A")
    ae_m = M.measures(fs, ae)
    mgr_m = M.measures(fs, mgr)
    check("AE sees only their own opportunities", ae_m["total"]["opps"], 283)
    assert_true("AE book is a strict subset of the entity",
                ae_m["total"]["gp"] < m["total"]["gp"],
                f"{M.money(ae_m['total']['gp'])} of {M.money(m['total']['gp'])}")
    assert_true("pod is a strict subset of the entity",
                0 < mgr_m["total"]["opps"] < m["total"]["opps"],
                f"{mgr_m['total']['opps']} of {m['total']['opps']} opportunities")
    assert_true("AE share of entity is reported, not assumed",
                0 < ae_m["shareOfEntity"] < 100,
                f"{ae_m['shareOfEntity']:.1f}% of open pipeline")
    # RLS must apply BEFORE aggregation, or the denominator leaks.
    ae_rows = set(M.slice_frame(fs, ae)["owner"])
    assert_true("AE frame contains exactly one owner", ae_rows == {"Brian Thompson"},
                str(sorted(ae_rows))[:60])

    print("\n── USE CASE 1 · ANOMALY DETECTION ───────────────────────────────────")
    summ = ANOM.summary()
    check("findings ingested from the DS export", summ["dsModel"], 485)
    assert_true("live detectors run on the movement log", summ["live"] > 50,
                f"{summ['live']} computed here from 36,631 logged changes")
    assert_true("both detectors agree on some findings", summ["corroborated"] > 0,
                f"{summ['corroborated']} corroborated independently")
    check("anomaly categories covered", len(summ["byCategory"]), 7)
    assert_true("cross-sell findings are framed as upside", summ["upside"] > 0,
                f"{summ['upside']} opportunity-framed findings")
    # Every type in the reference guide has a meaning and a persona.
    missing = [t for t in ANOM.TYPE_MEANING if t not in ANOM.ROUTING]
    assert_true("every anomaly type routes to a persona", not missing, str(missing))
    ev = ANOM.evidence_rows(
        ANOM.unified().query("is_opportunity_grain").iloc[0]["entity_id"], 5)
    assert_true("a finding drills to its own change-log rows", len(ev) > 0,
                f"{len(ev)} rows behind the top finding")

    print("\n── USE CASE 2 · DEAL CLOSURE LIKELIHOOD ─────────────────────────────")
    assert_true("the DS closure model is ingested", DS.available())
    card = P.model_card()
    check("DS model test AUC", card["primary"]["testAuc"], 0.5952)
    check("DS model covers this book's open deals",
          card["primary"]["coverage"]["openOpportunities"], 235)
    cp = P.closure_probability()
    check("every open deal is scored", len(cp), 235)
    assert_true("probability comes from the DS model",
                (cp["p_win_source"] == "ds-model").all())
    assert_true("an independent reproduction is reported beside it",
                0.5 < card["independent"]["holdoutAuc"] < 0.75,
                f"independent holdout AUC {card['independent']['holdoutAuc']}")
    # The honest read is the point: a weak model must SAY it is weak.
    assert_true("the model card states the model is weak",
                "0.595" in card["primary"]["honestRead"],
                "AUC quoted in the on-screen text")
    bench = DS.benchmark_card(cp.iloc[0]["opportunity_code"])
    check("benchmarked features per deal", len(bench), 9)
    assert_true("benchmark verdicts are direction-aware",
                any(b["direction"] == "lower_is_better" for b in bench),
                "revenue and GM% are coded lower-is-better in this data")
    assert_true("near-noise features are flagged as weak",
                sum(1 for b in bench if b["weakSignal"]) == 4,
                "4 of 9 features carry |r| < 0.02 and say so")
    # Leakage quarantine.
    leaky = set(P.LEAKING_FEATURES)
    used = set(P.NUMERIC_FEATURES) | set(P.CATEGORICAL_FEATURES)
    assert_true("outcome-encoding features are excluded from the model",
                not (leaky & used), str(sorted(leaky & used)))

    print("\n── USE CASE 3 · CROSS-SELL / UPSELL ─────────────────────────────────")
    ws = ACC.whitespace(fs, exec_p, limit=999)
    assert_true("whitespace accounts identified", len(ws) > 10,
                f"{len(ws)} accounts buying fewer than four lines of business")
    assert_true("each carries a peer-attach rate and a sized estimate",
                all(w["peerAttachRate"] > 0 and w["estimatedGp"] > 0 for w in ws))
    assert_true("each states its method", all(w["method"] for w in ws))
    assert_true("all are framed as opportunity, never risk",
                all(w["framing"] == "opportunity" for w in ws))
    att = ACC.attach_matrix()
    assert_true("attach is directional, not symmetric",
                len(att) == 12 and att["attachRate"].std() > 0.01,
                "P(B|A) differs from P(A|B) — that asymmetry is the play")

    # ...and the data-science export of the same use case, fused with it.
    recs = XS.unified(fs, exec_p, limit=999)
    check("cross-sell recommendations ingested", len(recs), 72)
    assert_true("every recommendation names an offering at service resolution",
                all(r["lob"] and r["portfolio"] for r in recs),
                "line of business AND service category, not just the line")
    assert_true("every recommendation carries a written reason",
                all(r["reasons"] and r["reasons"][0]["text"] for r in recs))
    assert_true("every recommendation names its owner",
                all(r["owner"] and r["owner"] != "Unassigned" for r in recs))
    assert_true("confidence is explained, not just asserted",
                all(r["confidenceMeaning"] for r in recs))
    assert_true("all are framed as opportunity, never risk",
                all(r["framing"] == "opportunity" for r in recs))
    assert_true("the caveat travels with every row",
                all("not a forecast" in r["caveat"] for r in recs))
    corr = [r for r in recs if r["corroborated"]]
    assert_true("some recommendations are corroborated by this layer's own view",
                0 < len(corr) < len(recs),
                f"{len(corr)} of {len(recs)} are a whole line of business the account "
                f"does not buy at all")
    th = XS.themes(fs, exec_p)
    assert_true("recommendations group into repeating plays", len(th) >= 5,
                f"{len(th)} offerings missing at two or more accounts")
    assert_true("a theme of one is not a theme",
                all(t["accounts"] >= 2 for t in th))
    assert_true("the biggest play spans more than one owner",
                th[0]["ownerCount"] > 1,
                f"{th[0]['offering']} — {th[0]['accounts']} accounts, "
                f"{th[0]['ownerCount']} owners")

    # Cross-sell obeys the same row-level rule as everything else.
    ae_p = PR.resolve("ae")
    ae_recs = XS.unified(fs, ae_p, limit=999)
    assert_true("a rep sees only their own accounts' recommendations",
                0 < len(ae_recs) < len(recs),
                f"{len(ae_recs)} of {len(recs)}")
    ae_accounts = set(M.slice_frame(fs, ae_p)["account_code"])
    assert_true("and never one for an account outside their book",
                all(r["accountCode"] in ae_accounts for r in ae_recs))

    print("\n── BUDGET AND COVERAGE ──────────────────────────────────────────────")
    t = B.totals(fs, exec_p)
    check("FY26 plan GP ($M)", round(t["budgetGp"] / 1e6, 3), 2.442)
    assert_true("coverage is measured forward, not against the year's residual",
                1.0 < t["coverage"] < 3.0, f"{t['coverage']:.2f}x")
    grid = B.coverage_grid(fs, exec_p)
    assert_true("the plan's footing gap is disclosed, not hidden",
                "cellsPresent" in grid["footing"],
                f"{grid['footing']['cellsPresent']} of {grid['footing']['cellsPossible']} cells")

    print("\n── PERSONAS, PAGES AND CHARTS ───────────────────────────────────────")
    check("pages", len(V.PAGES), 15)
    for key in PR.PERSONA_KEYS:
        p = PR.resolve(key)
        pages = [k for k, v in V.PAGES.items() if v[0] == key]
        assert_true(f"{key}: has its own pages", len(pages) >= 4, ", ".join(pages))
        # A persona asking for someone else's page lands on its own home.
        other = next(k for k, v in V.PAGES.items() if v[0] != key)
        check(f"{key}: cannot reach another persona's page",
              V.resolve_page(other, p), p.persona.home)

    used_keys: set[str] = set()
    claims_ok = True
    for key in PR.PERSONA_KEYS:
        p = PR.resolve(key)
        for page in [k for k, v in V.PAGES.items() if v[0] == key]:
            payload = V.view(page, fs, p)
            for ch in payload["charts"]:
                used_keys.add(ch["repositoryKey"])
                if not ch.get("says"):
                    claims_ok = False
            if not payload["kpis"]:
                claims_ok = False
    assert_true("every chart declares what it says", claims_ok)
    check("distinct chart types in use", len(used_keys), 14)

    # One chart, one page. Within a persona a chart id may appear on exactly
    # one page — the customer's words were that a chart repeated across a
    # persona's pages reads as the application not changing when the
    # navigation does. Across personas a repeat is allowed only where the rows
    # and the question both differ, which the RLS predicate guarantees.
    for key in PR.PERSONA_KEYS:
        p = PR.resolve(key)
        placed: dict[str, list[str]] = {}
        thin: list[str] = []
        for page in [k for k, v in V.PAGES.items() if v[0] == key]:
            ids = [c["id"] for c in V.charts_for(page, fs, p)]
            if len(ids) < 2:
                thin.append(page)
            for cid in ids:
                placed.setdefault(cid, []).append(page)
        dupes = {cid: pages for cid, pages in placed.items() if len(pages) > 1}
        assert_true(f"{key}: no chart appears on two pages", not dupes,
                    "; ".join(f"{c} on {', '.join(pg)}" for c, pg in dupes.items())
                    or f"{len(placed)} distinct charts")
        assert_true(f"{key}: every page carries two or more charts", not thin,
                    ", ".join(thin) or "all pages")

    # The executive brief names the three use cases and opens their pages.
    uc = V.view("tldr", fs, exec_p)["extras"].get("useCases") or []
    check("the brief states the three use cases",
          [u["key"] for u in uc], ["opportunities", "anomalies", "closure"])
    assert_true("each use case carries formatted figures and a page",
                all(len(u["figures"]) >= 2 and all(f["formatted"] for f in u["figures"])
                    and V.PAGES[u["page"]][0] == "executive" for u in uc))

    # The overdue stack: a deal whose close date has not arrived sits in "Not
    # yet due", not in "1-30 days over" — the loader writes 0, not a negative
    # count, for those rows.
    stack = C.ageing_stack(fs, exec_p)["data"]
    undue = sum(s["values"][stack["categories"].index("Not yet due")]
                for s in stack["series"])
    open_undue = M.subset(M.slice_frame(fs, exec_p), "open")
    open_undue = float(open_undue.loc[~open_undue["is_past_due"], "acv_gp"].sum())
    check("undue open deals fill the 'Not yet due' band", round(undue), round(open_undue))

    print("\n── FILTERS ARE SCOPED TO THE PERSONA ─────────────────────────────────")
    dims = {d["key"]: d["values"] for d in V.meta(mgr)["dimensions"]}
    assert_true("a manager's rep list holds only the pod",
                0 < len(dims["rep"]) < 20, f"{len(dims['rep'])} reps")
    ae_dims = {d["key"]: d["values"] for d in V.meta(ae)["dimensions"]}
    check("an AE's rep list is themselves alone", ae_dims["rep"], ["Brian Thompson"])
    assert_true("an AE's account list is their own accounts",
                set(ae_dims["account"]) == set(M.slice_frame(fs, ae)["account_name"]),
                f"{len(ae_dims['account'])} accounts")
    assert_true("the advanced chart set is actually used",
                {"waterfall.bridge", "combo.columnline", "funnel.stage",
                 "bubble.scatter", "mekko.marimekko", "treemap.nested",
                 "sankey.flow", "gantt.timeline"} <= used_keys,
                ", ".join(sorted(used_keys)))

    print("\n── ACTIONS ──────────────────────────────────────────────────────────")
    for key, verb in (("ae", "call"), ("manager", "coach"), ("executive", "coverage")):
        p = PR.resolve(key)
        cards = ACT.build(fs, p, limit=8)
        assert_true(f"{key}: has ranked actions", len(cards) > 0, f"{len(cards)} cards")
        assert_true(f"{key}: every card names an owner",
                    all(c["owner"] for c in cards))
        assert_true(f"{key}: every card shows the rule that fired it",
                    all(c["predicate"] for c in cards))
        assert_true(f"{key}: every card is deduplicated by entity",
                    len({f"{c['entity']['type']}:{c['entity']['id']}" for c in cards})
                    == len(cards))

    # The rail answers the page. Every page of a persona used to get the same
    # cards; now the cards that answer the page's question lead, and nothing is
    # lost — the same deduplicated set, re-ordered.
    lead = {page: [c["headline"] for c in V.view(page, fs, exec_p)["actions"]][:3]
            for page in ("tldr", "performance", "structure", "risks", "growth")}
    assert_true("executive pages lead with different cards",
                len({tuple(v) for v in lead.values()}) == len(lead),
                "; ".join(f"{k}: {v[0][:28]}" for k, v in lead.items()))
    assert_true("the risks page leads with risk-framed cards",
                all(c["framing"] == "risk"
                    for c in V.view("risks", fs, exec_p)["actions"][:3]))
    assert_true("the growth page leads with opportunity-framed cards",
                all(c["framing"] == "opportunity"
                    for c in V.view("growth", fs, exec_p)["actions"][:2]))
    assert_true("the structure page leads with concentration",
                all(c["kind"] == "concentration"
                    for c in V.view("structure", fs, exec_p)["actions"][:3]))
    for key in PR.PERSONA_KEYS:
        pr = PR.resolve(key)
        few = [page for page, v in V.PAGES.items() if v[0] == key
               and len(V.view(page, fs, pr)["actions"]) < 3]
        assert_true(f"{key}: every page carries three or more cards", not few,
                    ", ".join(few) or "all pages")

    print("\n── THE DECISION CARD CONTRACT ───────────────────────────────────────")
    for key in ("ae", "manager", "executive"):
        pr = PR.resolve(key)
        cards = ACT.build(fs, pr, limit=40)
        assert_true(f"{key}: every card offers two or more real choices",
                    all(len(c["options"]) >= 3 for c in cards),
                    "beyond watch and dismiss")
        assert_true(f"{key}: every choice explains what it does",
                    all(o.get("meaning") for c in cards for o in c["options"]))
        assert_true(f"{key}: every choice lands in a known status",
                    all(o["status"] in ACT.STATUSES
                        for c in cards for o in c["options"]))
        assert_true(f"{key}: every card carries a due date and a severity word",
                    all(c["dueDate"] and c["confidenceLabel"] for c in cards))
        assert_true(f"{key}: dismissing always demands a reason",
                    all(any(o["key"] == "dismiss" and o["needsReason"]
                            for o in c["options"]) for c in cards))

    print("\n── THE AI CANNOT REPEAT THE CHARTS ──────────────────────────────────")
    from ..llm import grounding as G

    payload = V.view("tldr", fs, exec_p)
    says = payload["chartsSay"]
    assert_true("the page publishes its claim keys", len(says) > 0, ", ".join(says))
    kept, dropped = G.strip_redundant(
        [{"text": "x", "lens": "state", "claim": says[0]},
         {"text": "y", "lens": "cause", "claim": says[0]}], says)
    assert_true("a state sentence repeating a chart is dropped", len(dropped) == 1)
    assert_true("a cause sentence about the same claim is kept", len(kept) == 1)
    ok, rejected = G.verify(
        [{"text": "Open pipeline is $805K."}, {"text": "Open pipeline is $999M."}],
        {"open pipeline gross profit": "$805K"})
    assert_true("an ungrounded number is rejected", len(rejected) == 1)
    assert_true("a grounded number survives", len(ok) == 1)

    print("\n── THE CONTEXT LAYER ────────────────────────────────────────────────")
    size = CAT.size_report()
    assert_true("the catalog fits in a prompt", size["withinBudget"],
                f"{size['approxTokens']} tokens of a {size['budget']} budget")
    cat = CAT.build()
    assert_true("every caveat carries a severity",
                all(c.get("severity") for c in cat["caveats"]))
    assert_true("the leakage caveat is critical",
                any(c["id"] == "outcome_leakage" and c["severity"] == "critical"
                    for c in cat["caveats"]))

    from ..semantic import query as Q

    plan = {"intent": "rank", "metric": "gp", "subset": "open", "breakdown": ["lob"]}
    res = Q.execute(plan, fs, exec_p, [])
    assert_true("a valid plan executes over the semantic layer", len(res.rows) == 4,
                f"{len(res.rows)} rows, rendered as {res.chart_key}")
    try:
        Q.execute({"intent": "rank", "metric": "gp", "subset": "open",
                   "breakdown": ["ssn"]}, fs, exec_p, [])
        assert_true("an invalid plan is refused", False, "it was not")
    except Q.PlanError:
        assert_true("an invalid plan is refused", True, "PlanError raised")
    # A plan cannot widen the caller's scope.
    try:
        Q.execute({"intent": "rank", "metric": "gp", "subset": "all",
                   "breakdown": ["lob"],
                   "filters": [{"dim": "rep", "value": "Karen Phillips"}]},
                  fs, ae, [])
        assert_true("a plan cannot reach outside the caller's rows", False, "it did")
    except Q.PlanError:
        assert_true("a plan cannot reach outside the caller's rows", True,
                    "PlanError raised")

    print("\n─────────────────────────────────────────────────────────────────────")
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
    else:
        print("all checks passed")
    return len(_failures)


if __name__ == "__main__":
    sys.exit(main())
