"""
The API boundary.

The client drew this layer himself on the 11 Sep call: *"transactional data on
SQL → semantic data where you have your rules, your definitions, your
calculations → an API interface… **that interface is where the authentication,
low level security all sits** → your front end."*

So two rules hold in this file and nowhere else:

  1. **Row-level security attaches here.** `X-User-UPN` (or the demo's
     `?persona=` / `?identity=`) resolves to a Principal, and that principal's
     predicate is applied to the frame before any aggregation happens
     downstream. No endpoint reaches a dataframe except through
     `measures.slice_frame`, which cannot be called without a principal.

  2. **No business logic.** Every number returned below was computed in
     `api/semantic/`. This module parses query strings, attaches identity, and
     serialises. If a calculation appears here, it is in the wrong file.

In front of both sits the access gate (`api/auth.py`): one shared password for
the whole platform, exchanged for a bearer token that every request must
carry. It is a door, not an identity — it decides whether a caller is in the
demo at all, and rule 1 still decides what they can see once inside.

`GET /api/v1/measures` and `GET /api/v1/catalog` exist to prove point 2: the
first returns the raw measure dictionary with no view model at all, the second
the machine-readable semantic catalog. A second consumer — a notebook, Power BI,
another team's agent — reads those and gets the same numbers the screen shows,
by construction.
"""

from __future__ import annotations

import json

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from . import auth
from .config import ALLOWED_ORIGINS, AS_OF
from .llm import service as AI
from .semantic import accounts as ACC
from .semantic import actions as ACT
from .semantic import anomalies as ANOM
from .semantic import budget as B
from .semantic import catalog as CAT
from .semantic import crosssell as XS
from .semantic import ds_model as DS
from .semantic import measures as M
from .semantic import personas as PR
from .semantic import predict as P
from .semantic import query as Q
from .semantic import views as V
from .semantic.loader import CUR_QUARTER, fy_label, report
from .semantic.measures import FilterState
from .semantic.personas import Principal

app = FastAPI(
    title="NTT Deal Intelligence — semantic API",
    version="2.0.0",
    description=(
        "The API boundary. Authentication and row-level security attach here; "
        "every figure is computed in api/semantic/ from the client's three "
        "datasets at request time, pinned to AS_OF."
    ),
)

# Order matters and is the reverse of how it reads: Starlette makes the LAST
# middleware added the OUTERMOST. The gate goes on first so that CORS wraps
# it — a 401 from the gate then still carries the CORS headers, and the
# browser hands the page a readable status instead of an opaque network error.
app.add_middleware(auth.AccessGate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth.router)


@app.on_event("startup")
def _warm() -> None:
    """
    Build the expensive caches before the first person arrives.

    The first view of a process loads the four files, derives the movement
    features, scores every open deal and fits the independent closure model —
    about twenty seconds on a fresh Cloud Run instance. Left to the first
    request, that is twenty seconds of a stakeholder looking at a skeleton.
    Done here, in a thread so the health probe answers immediately, the same
    work happens while the instance is still being routed to.
    """
    import threading

    def run() -> None:
        try:
            for persona in PR.PERSONAS:
                p = PR.resolve(persona)
                V.view(p.persona.home, FilterState(), p)
        except Exception as e:  # noqa: BLE001 — warming must never take the process down
            print(f"warm-up skipped: {type(e).__name__}: {e}")

    threading.Thread(target=run, name="warm-up", daemon=True).start()


# --------------------------------------------------------------------------- #
# Identity — the one place a principal is created
# --------------------------------------------------------------------------- #


def _principal(x_user_upn: str | None = None, persona: str | None = None,
               identity: str | None = None) -> Principal:
    """
    Resolve the caller.

    In production the UPN arrives inside an Entra token and `persona`/`identity`
    do not exist. For the demo the header selector drives them, which is the
    client's own diagram made visible rather than a way around it.
    """
    return PR.resolve(persona=persona, identity=identity, upn=x_user_upn)


def _filters(request: Request) -> FilterState:
    return FilterState.from_query(dict(request.query_params))


def _charts_say(request: Request) -> list[str]:
    """
    What the client says is on screen — union'd server-side, never trusted alone.

    A stale or edited client list must not be able to unlock a redundant answer,
    so the caller's list is combined with the set the server recomputes for the
    same page.
    """
    raw = request.query_params.get("chartsSay")
    if not raw:
        return []
    try:
        return [str(s) for s in json.loads(raw)][:40]
    except (ValueError, TypeError):
        return [s.strip() for s in raw.split(",") if s.strip()][:40]


# --------------------------------------------------------------------------- #
# Health and provenance
# --------------------------------------------------------------------------- #


# Both paths serve the same probe. `/healthz` is what run.sh and the local
# proxy use; on Cloud Run that path is answered by Google's front end before
# the container sees it, so production monitoring reads `/api/health`.
@app.get("/healthz")
@app.get("/api/health")
def healthz() -> dict:
    r = report()
    return {
        "ok": True,
        "asOf": AS_OF.isoformat(),
        "quarter": CUR_QUARTER,
        "fy": fy_label(2026),
        "data": {
            "lines": r.lines, "opportunities": r.opportunities,
            "accounts": r.accounts, "reps": r.reps,
            "movementRows": r.movement_rows, "anomalies": r.anomalies,
            "orphanMovement": r.orphan_movement_opps,
            "orphanAnomalies": r.orphan_anomaly_opps,
            "closeBeforeCreate": r.close_before_create,
        },
        "dsModel": DS.available(),
        "llm": AI.health(),
        "catalog": CAT.size_report(),
    }


@app.get("/api/meta")
def api_meta(x_user_upn: str | None = Header(default=None),
             persona: str | None = Query(default=None),
             identity: str | None = Query(default=None)) -> dict:
    return V.meta(_principal(x_user_upn, persona, identity))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


@app.get("/api/view")
def api_view(request: Request, page: str = Query(default=""),
             x_user_upn: str | None = Header(default=None),
             persona: str | None = Query(default=None),
             identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    return V.view(page or p.persona.home, _filters(request), p)


@app.get("/api/actions")
def api_actions(request: Request, limit: int = Query(default=12, le=40),
                x_user_upn: str | None = Header(default=None),
                persona: str | None = Query(default=None),
                identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    return {"actions": ACT.build(_filters(request), p, limit=limit),
            "persona": p.key, "scope": p.identity_label}


# --------------------------------------------------------------------------- #
# Intelligence
# --------------------------------------------------------------------------- #


@app.get("/api/risk")
def api_risk(request: Request, limit: int = Query(default=50, le=300),
             x_user_upn: str | None = Header(default=None),
             persona: str | None = Query(default=None),
             identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    r = P.risk_table()
    codes = set(M.slice_frame(fs, p)["opportunity_code"])
    r = r[r["opportunity_code"].isin(codes)].head(limit)
    return {
        "asOf": AS_OF.isoformat(),
        "model": P.model_card(),
        "deals": [
            {"opportunityCode": row.opportunity_code, "name": row.opportunity_name,
             "account": row.account_name, "owner": row.owner, "stage": row.stage,
             "lob": row.lob, "portfolio": row.portfolio,
             "gp": float(row.acv_gp), "revenue": float(row.acv_revenue),
             "riskScore": int(row.risk_score), "riskBand": row.risk_band,
             "topDriver": row.top_driver, "factors": row.risk_factors,
             "valueAtRisk": float(row.value_at_risk)}
            for row in r.itertuples(index=False)
        ],
    }


@app.get("/api/deal/{opportunity_code}")
def api_deal(opportunity_code: str,
             x_user_upn: str | None = Header(default=None),
             persona: str | None = Query(default=None),
             identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    # RLS on a single record too: a deal outside the principal's scope is a 404,
    # not a redacted 200 — the existence of the record is itself information.
    scoped = set(M.slice_frame(FilterState(), p)["opportunity_code"])
    if opportunity_code not in scoped:
        raise HTTPException(status_code=404, detail="no such opportunity in your scope")
    detail = P.explain(opportunity_code)
    if not detail:
        raise HTTPException(status_code=404, detail="opportunity is not open")
    return {
        **detail,
        "timeline": ANOM.evidence_rows(opportunity_code, 60),
        "benchmarks": DS.benchmark_card(opportunity_code) if DS.available() else [],
        "findings": ANOM.unified().query(
            "entity_id == @opportunity_code").to_dict("records"),
    }


@app.get("/api/anomalies")
def api_anomalies(request: Request, limit: int = Query(default=80, le=500),
                  x_user_upn: str | None = Header(default=None),
                  persona: str | None = Query(default=None),
                  identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    a = ANOM.for_persona(p.key)
    if fs.anomaly_category:
        a = a[a["category"] == fs.anomaly_category]
    # Routing decides which findings a role can act on; row-level security
    # decides which of those this caller may see. Every grain is narrowed —
    # an account finding to the caller's own accounts, a rep finding to the
    # reps in their predicate — not only the opportunity ones. The rule lives
    # with the findings so the risks page applies the same one.
    a = ANOM.scoped(a, fs, p)
    return {
        "summary": ANOM.summary(),
        "taxonomy": {"categories": [{"name": c, "question": ANOM.CATEGORY_BLURB[c]}
                                    for c in ANOM.CATEGORY_ORDER]},
        "findings": a.head(limit).replace({float("nan"): None}).to_dict("records"),
        "total": int(len(a)),
    }


@app.get("/api/accounts")
def api_accounts(request: Request,
                 x_user_upn: str | None = Header(default=None),
                 persona: str | None = Query(default=None),
                 identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    return {
        "whitespace": ACC.whitespace(fs, p, limit=40),
        "concentration": ACC.concentration(fs, p),
        "lobValue": ACC.lob_count_value().to_dict("records"),
        "attach": ACC.attach_matrix().to_dict("records"),
    }


@app.get("/api/account/{account_code}")
def api_account(account_code: str) -> dict:
    d = ACC.account_detail(account_code)
    if not d:
        raise HTTPException(status_code=404, detail="no such account")
    # The cross-sell recommendations for this account travel with it, so the
    # drawer never has to make a second call to answer "what should we sell
    # them next".
    d["crossSell"] = XS.for_account(account_code)
    return d


@app.get("/api/crosssell")
def api_crosssell(request: Request,
                  x_user_upn: str | None = Header(default=None),
                  persona: str | None = Query(default=None),
                  identity: str | None = Query(default=None)) -> dict:
    """
    The growth surface: recommendations, the plays they group into, and counts.

    Row-level security applies before anything is grouped, so a rep's themes
    are themes across their own accounts rather than the entity's themes with
    other people's accounts hidden — a filtered aggregate is a different
    number, not a smaller view of the same one.
    """
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    return {
        "recommendations": XS.unified(fs, p, limit=100),
        "themes": XS.themes(fs, p),
        "summary": XS.summary(fs, p),
        "confidenceMeaning": XS.CONFIDENCE_MEANING,
        "methodMeaning": XS.METHOD_MEANING,
        "caveat": XS.CAVEAT,
    }


@app.get("/api/budget")
def api_budget(request: Request,
               x_user_upn: str | None = Header(default=None),
               persona: str | None = Query(default=None),
               identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    return {
        "totals": B.totals(fs, p),
        "quarters": B.by_quarter(fs, p),
        "months": B.by_month(fs, p),
        "grid": B.coverage_grid(fs, p),
        "byLob": B.coverage_by(fs, p, "lob"),
        "byPortfolio": B.coverage_by(fs, p, "portfolio"),
        "bridge": B.bridge(fs, p),
    }


# --------------------------------------------------------------------------- #
# The AI surfaces
# --------------------------------------------------------------------------- #


@app.get("/api/ai/brief")
def api_brief(request: Request, page: str = Query(default=""),
              x_user_upn: str | None = Header(default=None),
              persona: str | None = Query(default=None),
              identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    page = V.resolve_page(page or p.persona.home, p)
    # The server recomputes what is on screen and unions it with the client's
    # claim, so a stale client cannot unlock a redundant answer.
    server_says = [s for c in V.charts_for(page, fs, p) for s in c["says"]]
    say = list(dict.fromkeys(_charts_say(request) + server_says))
    return AI.brief(fs, p, say, page)


@app.get("/api/ai/ask")
def api_ask(request: Request, q: str = Query(..., min_length=2, max_length=400),
            chart: str | None = Query(default=None, max_length=80),
            page: str | None = Query(default=None, max_length=40),
            x_user_upn: str | None = Header(default=None),
            persona: str | None = Query(default=None),
            identity: str | None = Query(default=None)) -> dict:
    """
    `chart` and `page` name the chart a question was typed beside. The chart
    is rebuilt server-side from the id for THIS principal and slice — the
    client's copy of it is never trusted — and the answer is words only.
    """
    p = _principal(x_user_upn, persona, identity)
    return AI.ask(q, _filters(request), p, _charts_say(request), chart_id=chart, page=page)


@app.post("/api/ai/explain")
def api_explain(request: Request, card: dict = Body(...),
                x_user_upn: str | None = Header(default=None),
                persona: str | None = Query(default=None),
                identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    return AI.explain(card, _filters(request), p, _charts_say(request))


@app.get("/api/ai/next-action/{opportunity_code}")
def api_next_action(opportunity_code: str, request: Request,
                    x_user_upn: str | None = Header(default=None),
                    persona: str | None = Query(default=None),
                    identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    scoped = set(M.slice_frame(FilterState(), p)["opportunity_code"])
    if opportunity_code not in scoped:
        raise HTTPException(status_code=404, detail="no such opportunity in your scope")
    return AI.next_action(opportunity_code, _filters(request), p)


@app.get("/api/ai/digest")
def api_digest(request: Request, days: int = Query(default=7, ge=1, le=90),
               x_user_upn: str | None = Header(default=None),
               persona: str | None = Query(default=None),
               identity: str | None = Query(default=None)) -> dict:
    p = _principal(x_user_upn, persona, identity)
    return AI.digest(_filters(request), p, days)


# --------------------------------------------------------------------------- #
# The proof the semantic layer is not welded to this UI
# --------------------------------------------------------------------------- #


@app.get("/api/v1/measures")
def api_measures(request: Request,
                 x_user_upn: str | None = Header(default=None),
                 persona: str | None = Query(default=None),
                 identity: str | None = Query(default=None)) -> dict:
    """
    The raw measure dictionary, unwrapped.

    No view model, no chart specs, no prose. If the semantic layer were welded
    to the front end, this endpoint could not exist.
    """
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    return {
        "asOf": AS_OF.isoformat(),
        "principal": {"persona": p.key, "identity": p.identity,
                      "predicate": p.predicate_sql},
        "filters": fs.values,
        "measure": fs.measure,
        "measures": M.measures(fs, p),
    }


@app.get("/api/v1/catalog")
def api_catalog() -> dict:
    """
    The machine-readable semantic catalog — the context layer the LLM is given.

    Published deliberately: any other consumer that wants to reason over this
    business reads the same description of it that our own model does, and can
    check that the definitions match what the screen shows.
    """
    return {**CAT.build(), "size": CAT.size_report()}


@app.post("/api/v1/query")
def api_query(request: Request, plan: dict = Body(...),
              x_user_upn: str | None = Header(default=None),
              persona: str | None = Query(default=None),
              identity: str | None = Query(default=None)) -> dict:
    """
    Execute a query plan directly, without a model in the loop.

    The same validator and the same dispatch the AI path uses. It exists so the
    plan contract can be tested — and demonstrated — independently of whether a
    language model is available or behaving.
    """
    p = _principal(x_user_upn, persona, identity)
    fs = _filters(request)
    try:
        r = Q.execute(plan, fs, p, _charts_say(request))
    except Q.PlanError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"rows": r.rows, "shape": r.shape, "chartKey": r.chart_key,
            "chartWhy": r.chart_why, "claim": r.claim, "note": r.note,
            "chart": Q.to_chart_spec(plan, r, fs)}
