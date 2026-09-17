"""
The six AI surfaces, each with an explicit job and an explicit prohibition.

The governing rule, from the user's own brief: *the AI features must be unique
and must not feel redundant with the charts; we shall not share the same info
again and again.* So every surface here is defined by what it says that the
charts on the same screen CANNOT, and the server enforces that mechanically via
`grounding.strip_redundant` rather than trusting a sentence in a prompt.

  brief        the situation in four sentences — mechanism and decision, not levels
  explain      why one card fired, from the recorded history behind it
  ask          free text in; a validated query plan, a chart, and a narration out
  nextAction   the specific next move on one deal, written to the rep
  digest       what changed in the movement log since a given date
  modelRead    how much to trust the closure model, in plain words

Every surface degrades to a computed template rather than to an error. If every
provider is down the page still reads as a finished product, and the response
says `degraded: true` so nobody mistakes a template for a model.
"""

from __future__ import annotations

import hashlib
import json
import time

from ..semantic import anomalies as ANOM
from ..semantic import catalog as CAT
from ..semantic import measures as M
from ..semantic import narrative as N
from ..semantic import predict as P
from ..semantic import query as Q
from ..semantic.personas import Principal
from . import grounding as G
from . import prompts, providers

_CACHE: dict[str, tuple[float, dict]] = {}

# A generated surface is keyed on persona, identity, page and filters — and
# AS_OF is pinned, so the data behind that key CANNOT change while the process
# lives. A short TTL therefore buys nothing and costs real quota: the free
# Gemini tier is a per-minute allowance (its 429 carries `RetryInfo: 53s`), and
# re-asking for a brief nobody's inputs have changed is what exhausts it. The
# cache is held for the process lifetime and refreshed explicitly.
CACHE_TTL_S = float(__import__("os").environ.get("NTT_LLM_CACHE_TTL", 86_400))

SENTENCES_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "lens": {"type": "string",
                             "enum": list(G.LENSES)},
                    "claim": {"type": "string"},
                },
                "required": ["text", "lens"],
            },
        },
    },
    "required": ["headline", "sentences"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(Q.INTENTS)},
        "metric": {"type": "string", "enum": list(Q.METRICS)},
        "subset": {"type": "string", "enum": list(M.SUBSETS)},
        "breakdown": {"type": "array", "items": {"type": "string"}},
        "time": {"type": "object",
                 "properties": {"grain": {"type": "string",
                                          "enum": ["none", "month", "quarter"]}}},
        "filters": {"type": "array", "items": {
            "type": "object",
            "properties": {"dim": {"type": "string"}, "op": {"type": "string"},
                           "value": {"type": "string"}},
            "required": ["dim", "value"]}},
        "compareTo": {"type": "string"},
        "sort": {"type": "object", "properties": {
            "by": {"type": "string"}, "dir": {"type": "string"}}},
        "limit": {"type": "integer"},
        "chartHint": {"type": "string"},
        "needsChart": {"type": "boolean"},
        "restatesClaim": {"type": "string"},
        # NOTE: there is deliberately no `unanswerable` field. Gemini's
        # structured output populates every declared property whether or not it
        # is required, so an optional free-text "why not" field came back filled
        # on every single call — including calls whose plan was perfectly good,
        # where it simply echoed the question. Refusal is carried by the
        # `intent` enum instead, which cannot be spuriously populated because
        # every one of its members is a real instruction.
        "refuseReason": {"type": "string"},
    },
    "required": ["intent", "metric", "subset"],
}


def _cache_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


def _cached(key: str) -> dict | None:
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return {**hit[1], "cached": True}
    return None


def _store(key: str, value: dict) -> dict:
    _CACHE[key] = (time.time(), value)
    return value


def _entities(fs: M.FilterState, principal: Principal) -> list[str]:
    df = M.slice_frame(fs, principal)
    out = list(df["lob"].dropna().unique()[:6]) + list(df["portfolio"].dropna().unique()[:6])
    out += list(df["account_name"].dropna().unique()[:10])
    out += [principal.identity_label]
    return [str(x) for x in out]


# --------------------------------------------------------------------------- #
# Surface 1 — the situation brief
# --------------------------------------------------------------------------- #


def brief(fs: M.FilterState, principal: Principal, charts_say: list[str],
          page: str = "") -> dict:
    m = M.measures(fs, principal)
    fallback = N.brief(fs, principal, m, page)
    pack = G.pack_figures(N.facts_pack(fs, principal, m))

    key = _cache_key("brief", principal.key, principal.identity, page,
                     json.dumps(fs.values, sort_keys=True), fs.measure)
    if (hit := _cached(key)):
        return hit

    cards = []
    from ..semantic import actions as ACT

    for c in ACT.build(fs, principal, limit=4):
        cards.append({"id": c["key"], "headline": c["headline"],
                      "owner": c["owner"], "why": c["why"][:2]})

    task = (
        f"Write the situation brief for a {principal.persona.label} "
        f"({principal.persona.jtbd}) looking at the '{page or principal.persona.home}' "
        f"page, scoped to {m['scope']['label']}. "
        f"The four highest-priority action cards are in EVIDENCE; use them to ground "
        f"what to decide, but do not simply list them."
    )
    user = G.render(pack, entities=_entities(fs, principal), evidence=cards,
                    charts_say=charts_say,
                    untrusted=[c["headline"] for c in cards], task=task)
    try:
        res = providers.complete(prompts.BRIEF_V1, user, SENTENCES_SCHEMA,
                                 max_tokens=700, deadline_s=12)
    except providers.LLMUnavailable as e:
        return {**fallback, "degraded": True, "reason": str(e),
                "attempts": getattr(e, "attempts", [])}

    # The allowed-token set must cover EVERY string the server put in front of
    # the model, not just the figures pack. The action cards' `why` bullets
    # carry server-computed figures ("$936K of plan against $168K of open
    # pipeline"), and without them the validator rejected the model for quoting
    # numbers this layer had handed it — dropping true sentences as if they
    # were hallucinations.
    clean = G.sanitise(res.obj.get("sentences", []), pack, charts_say,
                       extra=[c["headline"] for c in cards]
                       + [w for c in cards for w in c["why"]])
    if not clean["sentences"]:
        return {**fallback, "degraded": True,
                "reason": "every generated sentence was redundant or ungrounded",
                "rejected": clean["rejected"], "dropped": clean["dropped"]}

    return _store(key, {
        "headline": res.obj.get("headline") or fallback["headline"],
        "sentences": clean["sentences"],
        "provider": res.provider, "model": res.model, "latencyMs": res.latency_ms,
        "degraded": False, "cached": False,
        "rejected": clean["rejected"], "dropped": clean["dropped"],
    })


# --------------------------------------------------------------------------- #
# Surface 2 — why this card fired
# --------------------------------------------------------------------------- #


def explain(card: dict, fs: M.FilterState, principal: Principal,
            charts_say: list[str]) -> dict:
    fallback = N.explain_card(card)
    m = M.measures(fs, principal)
    pack = G.pack_figures({
        **N.facts_pack(fs, principal, m),
        "cardValue": M.money(card.get("valueAtStake", 0)),
        "cardSeverity": f"{card.get('severity', 0):.0f}",
    })

    entity = card.get("entity", {})
    evidence: list[dict] = []
    if entity.get("type") == "Opportunity":
        evidence = [{"id": f"chg{i}", **r}
                    for i, r in enumerate(ANOM.evidence_rows(entity.get("id", ""), 14))]
        detail = P.explain(entity.get("id", ""))
        if detail:
            evidence.append({"id": "risk", "factors": detail.get("riskFactors", []),
                             "stagePath": detail.get("stagePath")})

    key = _cache_key("explain", card.get("key"), principal.key)
    if (hit := _cached(key)):
        return hit

    task = (
        f"Explain to {card.get('owner')} why this fired and what to do: "
        f"\"{card.get('headline')}\". The rule that fired it is: "
        f"{card.get('predicate')}. EVIDENCE holds the actual change-log rows behind it."
    )
    user = G.render(pack, entities=[str(entity.get("label", ""))], evidence=evidence,
                    charts_say=charts_say,
                    untrusted=[card.get("headline", ""), *card.get("why", [])],
                    task=task)
    try:
        res = providers.complete(prompts.EXPLAIN_V1, user, SENTENCES_SCHEMA,
                                 max_tokens=550, deadline_s=10)
    except providers.LLMUnavailable as e:
        return {**fallback, "degraded": True, "reason": str(e),
                "attempts": getattr(e, "attempts", []), "evidence": evidence}

    clean = G.sanitise(res.obj.get("sentences", []), pack, charts_say,
                       extra=[*card.get("why", []), card.get("predicate", "")])
    if not clean["sentences"]:
        return {**fallback, "degraded": True, "evidence": evidence}
    return _store(key, {
        "headline": res.obj.get("headline") or card.get("headline", ""),
        "sentences": clean["sentences"], "evidence": evidence,
        "provider": res.provider, "model": res.model, "latencyMs": res.latency_ms,
        "degraded": False, "cached": False,
        "rejected": clean["rejected"], "dropped": clean["dropped"],
    })


# --------------------------------------------------------------------------- #
# Surface 3 — free-text ad-hoc question
# --------------------------------------------------------------------------- #


def _refusal(q: str, principal: Principal, reason: str, plan: dict | None = None,
             headline: str = "The data can't answer that as asked") -> dict:
    """
    The calm decline. One plain sentence saying why, and two questions for this
    persona that are known to work, so a dead end always has a way out.
    """
    from ..semantic import answers as A

    out = {"question": q, "refused": True, "degraded": False, "provider": "computed",
           "answer": {"headline": headline,
                      "sentences": [{"text": reason, "lens": "answer", "claim": None}]},
           "suggestions": A.suggestions_for(principal.key, 2)}
    if plan:
        out["plan"] = plan
    return out


def _plan_text(q: str, fs: M.FilterState, principal: Principal,
               charts_say: list[str]) -> tuple[dict, str]:
    """
    Turn the question into a query plan: the model when one will answer, the
    keyword rules when none will. Raises `providers.LLMUnavailable` only when
    neither produced a plan, so the caller can refuse with alternatives.
    """
    from ..semantic.dimensions import REGISTRY
    from . import planner_fallback

    catalog = CAT.render()
    # The legal vocabulary is stated ABOVE the catalog as well as inside it.
    # With it only embedded in a 19KB JSON blob the planner invented limits that
    # do not exist — it refused "open pipeline by rep" on the grounds that rep
    # was not a valid breakdown, when rep is in the registry. A short explicit
    # list in front of the question removes that failure entirely.
    planner_user = (
        f"LEGAL BREAKDOWN DIMENSIONS (use these exact keys, nothing else):\n"
        f"{', '.join(REGISTRY)}\n\n"
        f"LEGAL METRICS: {', '.join(Q.METRICS)}\n"
        f"LEGAL SUBSETS: {', '.join(M.SUBSETS)}\n"
        f"LEGAL INTENTS: {', '.join(Q.INTENTS)}\n\n"
        f"Do not set intent=refuse because a dimension looks unfamiliar — if it is "
        f"in the list above it is available. Use intent=refuse only when the "
        f"question genuinely cannot be expressed with this vocabulary, or when the "
        f"catalog's caveats say the data cannot support the claim.\n\n"
        f"CATALOG\n{catalog}\n\n"
        f"CURRENT SCOPE\npersona={principal.persona.label}; "
        f"rows={principal.predicate_sql}; filters={json.dumps(fs.values)}\n\n"
        f"ALREADY ON SCREEN\n{json.dumps(charts_say)}\n\n"
        f"QUESTION\n{q}\n\n"
        f"Emit the query plan as JSON."
    )
    try:
        planned = providers.complete(prompts.ASK_PLANNER_V1, planner_user,
                                     PLAN_SCHEMA, max_tokens=500, deadline_s=12,
                                     temperature=0.0)
        return planned.obj, planned.provider
    except providers.LLMUnavailable as e:
        plan = planner_fallback.plan_from_text(q)
        if plan is None:
            raise e
        return plan, "keyword rules"


def ask(question: str, fs: M.FilterState, principal: Principal,
        charts_say: list[str], chart_id: str | None = None,
        page: str | None = None) -> dict:
    """
    Natural language in; a validated plan, a chart and a narration out.

    Four rungs, each of which produces a finished answer rather than an error:

      1. a question the product itself suggests has a computed answer in
         `semantic.answers` — correct, chart-backed and independent of any
         provider, so a suggestion chip never fails during a demo;
      2. free text is planned by the model, executed here, and narrated by a
         second model call, as before;
      3. when no model will plan, keyword rules produce the plan for the
         breakdown and trend questions people actually ask;
      4. when no model will narrate, the executed result is written up by the
         same computed prose the suggested answers use.

    The execution is always the server's, so the answer's numbers are the
    server's too — the model never sees a row and never produces a figure.
    Nothing here returns "cannot answer right now": a question the data cannot
    answer as asked gets a calm refusal with two questions that work.
    """
    from ..semantic import answers as A

    q = (question or "").strip()
    if not q:
        return {"error": "empty question"}

    key = _cache_key("ask", q.lower(), principal.key, principal.identity,
                     json.dumps(fs.values, sort_keys=True), chart_id or "", page or "")
    if (hit := _cached(key)):
        return hit

    # Rung 0 — a question typed beside a chart is about THAT chart. It is
    # answered from the chart's own rows and comes back as words only; see
    # chart_ask.py. A chart id the caller cannot reach falls through to the
    # ordinary path rather than to a refusal.
    if chart_id:
        from . import chart_ask

        scoped = chart_ask.answer(q, chart_id, page or "", fs, principal, charts_say)
        if scoped is not None:
            return _store(key, A.jsonable(scoped))

    # Rung 1 — a suggested question has a computed answer. It goes through
    # the cache too: the risk table and the cross-sell recommendations are
    # cached, but the join and the prose are not free, and the same chip is
    # clicked many times in a demo.
    computed = A.answer(q, fs, principal, charts_say)
    if computed is not None:
        return _store(key, computed)

    # Rung 2 and 3 — a plan, from the model or from the keyword rules.
    try:
        plan, planner = _plan_text(q, fs, principal, charts_say)
    except providers.LLMUnavailable:
        return _refusal(
            q, principal,
            "No language service is available right now and the words in the "
            "question do not name a measure or a breakdown the data holds. Try one "
            "of the questions below, or ask for a figure by rep, account, line of "
            "business, industry, portfolio, stage or month.")

    try:
        result = Q.execute(plan, fs, principal, charts_say)
    except Q.PlanError as e:
        return _store(key, _refusal(
            q, principal, f"{str(e).rstrip('.')}. The figures the page is built from "
                          f"are unchanged.", plan=plan,
            headline="That question cannot be answered from this data"))

    if result.note and not result.rows:
        return _store(key, _refusal(q, principal, result.note, plan=plan,
                                    headline="Not answerable from this extract"))

    chart = Q.to_chart_spec(plan, result, fs)

    # The chart we just drew is now on screen, so the narrator is forbidden from
    # describing it — the whole point of the second hop.
    say = list(charts_say) + ([result.claim] if chart else [])
    m = M.measures(fs, principal)
    pack = G.pack_figures({
        **N.facts_pack(fs, principal, m),
        **{f"row{i}_{r.get('key', r.get('row', i))}":
           (M.money(r["value"]) if str(plan.get("metric")) in ("gp", "rev", "revenue")
            else f"{r['value']:,.1f}")
           for i, r in enumerate(result.rows[:12])},
    })
    task = (
        f"The user asked: \"{q}\". The server executed the plan and the result is "
        f"charted beside your text. Say what the result MEANS, how it sits against "
        f"the rest of the book, and what produced it. Do not describe the chart."
    )
    user = G.render(pack, entities=[str(r.get("key", r.get("row", ""))) for r in result.rows[:12]],
                    evidence=result.rows[:12], charts_say=say, untrusted=[q], task=task)
    try:
        narrated = providers.complete(prompts.ASK_NARRATOR_V1, user, SENTENCES_SCHEMA,
                                      max_tokens=600, deadline_s=12)
        clean = G.sanitise(narrated.obj.get("sentences", []), pack, say)
        answer = {"headline": narrated.obj.get("headline", Q._title(plan)),
                  "sentences": clean["sentences"],
                  "rejected": clean["rejected"], "dropped": clean["dropped"]}
        provider = narrated.provider
        if not clean["sentences"]:
            raise providers.LLMUnavailable("every narrated sentence was redundant or ungrounded")
    except providers.LLMUnavailable:
        # Rung 4 — the computed prose. It reads as an answer, not a log line:
        # the top item named, its share of the slice, a peer figure where one
        # is cheap, and what to do — the same writer the suggested questions
        # use, so the two paths sound alike.
        answer = A.narrate_result(plan, result, fs, principal)
        provider = "computed"

    return _store(key, A.jsonable({
        "question": q, "plan": plan, "answer": answer, "chart": chart,
        "rows": result.rows, "shape": result.shape, "chartWhy": result.chart_why,
        "claim": result.claim, "provider": provider, "planner": planner,
        # "Written from the computed figures" is information, not a warning,
        # and the client prints it quietly. `degraded` is reserved for the case
        # where nothing was computed at all, which no longer happens here.
        "degraded": False, "deterministic": provider == "computed",
        "cached": False,
    }))



# --------------------------------------------------------------------------- #
# Surface 4 — the next move on one deal
# --------------------------------------------------------------------------- #


def next_action(opportunity_code: str, fs: M.FilterState,
                principal: Principal) -> dict:
    detail = P.explain(opportunity_code)
    if not detail:
        return {"error": "unknown or closed opportunity"}
    key = _cache_key("nba", opportunity_code, principal.key)
    if (hit := _cached(key)):
        return hit

    rows = ANOM.evidence_rows(opportunity_code, 16)
    pack = G.pack_figures({
        "gp": M.money(detail["acvGp"]),
        "revenue": M.money(detail["acvRevenue"]),
        "riskScore": f"{detail['riskScore']}",
        "quietDays": f"{detail.get('quietDays') or 0} days",
        "pWin": f"{detail.get('pWin', 0) * 100:.0f}%" if detail.get("pWin") else "n/a",
        "repConfidence": f"{detail.get('repConfidence', 0) * 100:.0f}%",
        "closeDate": str(detail.get("closeDate")),
    })
    task = (
        f"Write the next move on '{detail['name']}' at {detail['account']}, owned by "
        f"{detail['owner']}, currently at {detail['stage']}. The risk factors and the "
        f"change-log rows are in EVIDENCE."
    )
    user = G.render(pack, entities=[detail["account"], detail["owner"], detail["name"]],
                    evidence=[{"id": "risk", "factors": detail["riskFactors"]},
                              *[{"id": f"chg{i}", **r} for i, r in enumerate(rows)]],
                    charts_say=[], untrusted=[detail["name"]], task=task)
    try:
        res = providers.complete(prompts.NEXT_ACTION_V1, user, SENTENCES_SCHEMA,
                                 max_tokens=450, deadline_s=10)
        clean = G.sanitise(res.obj.get("sentences", []), pack, [])
        if clean["sentences"]:
            return _store(key, {"deal": detail, "headline": res.obj.get("headline", ""),
                                "sentences": clean["sentences"],
                                "provider": res.provider, "degraded": False,
                                "evidence": rows})
    except providers.LLMUnavailable:
        pass

    return {"deal": detail, "headline": detail["riskBand"] + " risk",
            "sentences": [{"text": f["detail"], "lens": "cause", "claim": None}
                          for f in detail["riskFactors"]],
            "provider": "computed", "degraded": True, "evidence": rows}


# --------------------------------------------------------------------------- #
# Surface 5 — what changed
# --------------------------------------------------------------------------- #


def digest(fs: M.FilterState, principal: Principal, days: int = 7) -> dict:
    """
    What moved in the change log in the last N days, within this scope.

    This is the one surface that is purely `delta`, which is exactly why it can
    never collide with a chart: no chart on any page draws a window this narrow.
    """
    from ..semantic.loader import AS_OF_TS, movement

    df = M.slice_frame(fs, principal)
    codes = set(df["opportunity_code"])
    mv = movement()
    window = mv[(mv["opportunity_code"].isin(codes))
                & (mv["days_ago"] >= 0) & (mv["days_ago"] <= days)]
    by_field = window["field"].value_counts().to_dict()
    movers = (window.groupby(["opportunity_code", "opportunity_name"]).size()
              .sort_values(ascending=False).head(8).reset_index(name="changes"))
    stage_moves = window[window["field"] == "stage"]
    return {
        "windowDays": days,
        "asOf": AS_OF_TS.date().isoformat(),
        "totalChanges": int(len(window)),
        "opportunitiesTouched": int(window["opportunity_code"].nunique()),
        "byField": by_field,
        "stageMoves": [
            {"opportunity": r.opportunity_name, "from": r.old_value, "to": r.new_value,
             "by": r.changed_by, "date": r.change_date.date().isoformat()}
            for r in stage_moves.head(10).itertuples(index=False)
        ],
        "busiest": movers.to_dict("records"),
        "note": (
            f"The change log ends {movement()['change_date'].max().date().isoformat()} and this "
            f"window is the {days} days before {AS_OF_TS.date().isoformat()}. Nothing on any chart covers a window this "
            f"narrow — this is the only view of what actually moved."
        ),
    }


def health() -> dict:
    return {**providers.health(), "cacheEntries": len(_CACHE),
            "cacheTtlSeconds": CACHE_TTL_S}
