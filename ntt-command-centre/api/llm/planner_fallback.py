"""
A keyword planner for free text, used only when no language model can plan.

The model-driven path turns a question into a small query plan — metric,
subset, breakdown, time grain — and the server executes that plan. When every
provider is rate-limited the plan still has to come from somewhere, and for
the shapes of question this product actually gets ("open pipeline by rep",
"win rate by industry", "closed-won by month") a handful of word rules produce
the same plan the model would.

It is deliberately narrow. It returns None whenever it cannot name at least one
of a metric, a subset or a breakdown from the words in the question, and the
caller turns that into a calm refusal with two questions that are known to
work. Guessing a plan for "why did Cobalt slip" and drawing a bar chart of
gross profit would be worse than saying the question is not one the data can
answer as asked.

The plan it emits goes through the same `query.validate` as a model's, so a
word that happens to match a dimension label cannot widen the caller's scope
or name a column outside the registry.
"""

from __future__ import annotations

import re

from ..semantic.dimensions import REGISTRY

#: Words that name a breakdown, in the order they are tested. The first hit
#: wins, so the more specific phrase ("line of business") is listed before a
#: word it contains ("business" never appears alone here for that reason).
_DIMENSION_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rep", ("rep", "reps", "owner", "owners", "salesperson", "sales person",
             "seller", "sellers", "who owns", "who holds", "which people", "by person")),
    ("lob", ("lob", "lobs", "line of business", "lines of business", "business line",
             "business lines")),
    ("account", ("account", "accounts", "customer", "customers", "client", "clients")),
    ("industry", ("industry", "industries", "sector", "sectors", "vertical", "verticals")),
    ("portfolio", ("portfolio", "portfolios")),
    ("stage", ("stage", "stages")),
    ("country", ("country", "countries", "region", "regions", "geography")),
    ("quarter", ("quarter", "quarters", "quarterly", "by q")),
    ("forecast", ("forecast category", "forecast categories", "commit", "best case")),
    ("orderType", ("order type", "order types", "new business", "renewal", "renewals")),
)


def _norm(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9%$]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _has(s: str, *phrases: str) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", s) for p in phrases)


def _metric(s: str) -> tuple[str | None, bool]:
    """The metric named in the text, and whether it was named at all."""
    if _has(s, "win rate", "winrate", "conversion", "hit rate", "close rate"):
        return "winrate", True
    if _has(s, "margin", "gm", "gross margin", "profitability"):
        return "gm", True
    if _has(s, "how many", "number of", "count", "deals count"):
        return "count", True
    if _has(s, "revenue", "sales", "bookings", "acv"):
        return "rev", True
    if _has(s, "gross profit", "gp", "profit"):
        return "gp", True
    return None, False


def _subset(s: str) -> tuple[str | None, bool]:
    if _has(s, "quiet", "stalled", "stopped moving", "silent", "no activity", "stuck"):
        return "stalled", True
    if _has(s, "overdue", "past due", "past its date", "past their close",
            "slipped", "late"):
        return "pastdue", True
    # "win" on its own is a subset word here; when the metric is win rate the
    # caller overrides the subset to closed anyway, so the two never collide.
    if _has(s, "closed won", "won", "win", "wins", "winning", "booked", "closed"):
        if _has(s, "closed") and not _has(s, "closed won", "won", "win", "wins", "winning"):
            return "closed", True
        return "won", True
    if _has(s, "lost", "lose", "losses", "losing"):
        return "lost", True
    if _has(s, "qualified", "commit"):
        return "qualified", True
    if _has(s, "open", "pipeline", "in flight", "outstanding"):
        return "open", True
    return None, False


def _breakdown(s: str) -> str | None:
    for dim, words in _DIMENSION_WORDS:
        if dim in REGISTRY and _has(s, *words):
            return dim
    # A registry label that the word table does not cover still counts, so a
    # dimension added later is reachable without touching this file.
    for dim, d in REGISTRY.items():
        if _has(s, d.label.lower(), d.label.lower() + "s"):
            return dim
    return None


def _grain(s: str) -> str | None:
    if _has(s, "month", "months", "monthly", "trend", "over time", "by month",
            "month by month", "each month"):
        return "month"
    if _has(s, "quarter", "quarters", "quarterly", "by quarter"):
        return "quarter"
    return None


def plan_from_text(question: str) -> dict | None:
    """
    A query plan for the question, or None when the words name nothing.

    A trend question that also names a breakdown keeps the breakdown and drops
    the grain, because the executor draws one or the other and a bar per rep
    answers "which rep" better than a line per month would.
    """
    s = _norm(question)
    if not s:
        return None

    metric, named_metric = _metric(s)
    sub, named_subset = _subset(s)
    dim = _breakdown(s)
    grain = _grain(s)

    # "By quarter" is a time question, not a breakdown, unless the rest of the
    # question names another dimension too.
    if dim == "quarter" and grain == "quarter":
        dim = None

    if not (named_metric or named_subset or dim or grain):
        return None

    # Win rate is only defined on closed deals. Everything else defaults to
    # the whole book when the question names no subset: "pipeline" and "open"
    # already narrow it, so a bare "gross profit by account" means all of it.
    if metric == "winrate":
        sub = "closed"
    else:
        sub = sub or "all"
    metric = metric or "gp"

    if dim and grain:
        grain = None

    intent = "trend" if grain else "compare" if metric in ("winrate", "gm") else "rank"
    plan: dict = {"intent": intent, "metric": metric, "subset": sub,
                  "breakdown": [dim] if dim else [],
                  "time": {"grain": grain or "none"},
                  "plannedBy": "keyword rules"}
    if _has(s, "top", "biggest", "largest", "most", "highest", "best") and dim:
        plan["sort"] = {"by": "value", "dir": "desc"}
    if _has(s, "smallest", "least", "lowest", "weakest", "worst") and dim:
        plan["sort"] = {"by": "value", "dir": "asc"}
    m = re.search(r"\btop (\d{1,2})\b", s)
    if m:
        plan["limit"] = max(1, min(int(m.group(1)), 30))
    return plan
