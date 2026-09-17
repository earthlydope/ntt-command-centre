"""
Deterministic answers for every question the product itself suggests.

The Ask panel offers a handful of questions per persona, and every chart offers
two or three of its own. Those strings are a promise: a reader who clicks one
must get a correct, chart-backed answer in plain English, and must get it even
when every language provider is rate-limited — which, on a free tier, is the
normal state during a demo.

So each suggested question is registered here against a function that computes
the answer from the semantic layer. Where the question is a straightforward
breakdown or trend the function builds a query plan and hands it to
`query.execute`, so the chart is chosen by the same shape rule the model-driven
path uses. Where it needs a derived table — deal risk, quiet deals, rep
behaviour, findings, whitespace, plan against actual — it reads the module that
owns that table and draws the chart with `charts.spec`.

Three rules hold for every answer:

  * every number in a sentence goes through `measures.money`, `pct` or `count`;
    a raw float never reaches the page,
  * the sentences follow one shape — what the answer is, how it compares or
    what produced it, and what to do — each in one plain sentence,
  * the row set is the caller's: `slice_frame` applies the persona's predicate
    and the page's filters before anything is counted, so an account executive
    asking a chart question sees their own book and nobody else's.

The lookup key is the normalised question text (lower case, punctuation and
extra whitespace stripped, a few synonyms folded), so a trailing question mark
or "LOB" for "line of business" still matches.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Callable

import numpy as np
import pandas as pd

from . import accounts as ACC
from . import actions as ACT
from . import anomalies as ANOM
from . import budget as B
from . import charts as C
from . import crosssell as XS
from . import predict as P
from . import query as Q
from .dimensions import REGISTRY
from .loader import CUR_QUARTER, STAGE_ORDER, facts, movement
from .measures import (
    FilterState,
    by_dimension,
    count,
    measures,
    money,
    pct,
    slice_frame,
    subset,
)
from .movement_features import STALL_DAYS, features, rep_behaviour
from .personas import Principal

Answerer = Callable[[FilterState, Principal, list[str]], dict]

#: normalised question -> (canonical wording, answerer)
_REGISTRY: dict[str, tuple[str, Answerer]] = {}

#: The persona chips, mirrored from web/src/components/askSuggestions.ts. The
#: client matches on text, so a change to either side must be made on both.
SUGGESTIONS: dict[str, list[str]] = {
    "ae": [
        "Which of my accounts has the most at stake?",
        "How does my win rate compare with the book?",
        "Where is my pipeline concentrated by portfolio?",
        "Which of my deals have gone quiet?",
        "What should I do first today?",
    ],
    "manager": [
        "Which reps in my pod hold the most open pipeline?",
        "How does win rate vary across my pod?",
        "Where is the pod's pipeline by line of business?",
        "Who should I coach first?",
        "How much of the pod's pipeline has stopped moving?",
    ],
    "executive": [
        "Which line of business has the weakest margin?",
        "Which accounts are biggest by gross profit?",
        "How has closed-won tracked by month?",
        "Where is open pipeline concentrated by industry?",
        "Are we on track against plan this quarter?",
        "What needs fixing, and what is it worth?",
    ],
}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_SYNONYMS: tuple[tuple[str, str], ...] = (
    (r"\blobs\b", "lines of business"),
    (r"\blob\b", "line of business"),
    (r"\bline of businesses\b", "lines of business"),
    (r"\bgp\b", "gross profit"),
    (r"\bwinrate\b", "win rate"),
    (r"\bsales rep(s)?\b", r"rep\1"),
    (r"\bteam s\b", "pod s"),
    (r"\bmy team\b", "my pod"),
    (r"\bthe team\b", "the pod"),
)


def normalise(question: str) -> str:
    """Lower case, punctuation to spaces, whitespace collapsed, synonyms folded."""
    s = (question or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pattern, rep in _SYNONYMS:
        s = re.sub(pattern, rep, s)
    return s


def register(*questions: str) -> Callable[[Answerer], Answerer]:
    def deco(fn: Answerer) -> Answerer:
        for q in questions:
            _REGISTRY[normalise(q)] = (q, fn)
        return fn
    return deco


def register_template(template: str, fn: Callable[..., dict]) -> None:
    """One question per registry dimension, e.g. "Which {label} has ..."."""
    for dim, d in REGISTRY.items():
        q = template.format(label=d.label.lower())
        _REGISTRY[normalise(q)] = (q, functools.partial(fn, dim=dim))


def lookup(question: str) -> Answerer | None:
    hit = _REGISTRY.get(normalise(question))
    return hit[1] if hit else None


def answer(question: str, fs: FilterState, principal: Principal,
           charts_say: list[str]) -> dict | None:
    """The computed answer for a registered question, or None if it is not one."""
    fn = lookup(question)
    if fn is None:
        return None
    out = fn(fs, principal, charts_say)
    out["question"] = question
    return jsonable(out)


def registered_questions() -> list[str]:
    return [v[0] for v in _REGISTRY.values()]


def suggestions_for(persona: str, n: int = 2) -> list[str]:
    return SUGGESTIONS.get(persona, SUGGESTIONS["executive"])[:n]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def jsonable(obj):
    """numpy scalars to Python, NaN to None, Timestamps to ISO — recursively."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, pd.Timestamp):
        return obj.date().isoformat()
    if obj is pd.NaT:
        return None
    return obj


def _s(text: str, lens: str, claim: str | None = None,
       bold: list[str] | None = None) -> dict:
    return {"text": text, "lens": lens, "claim": claim,
            "bold": [b for b in (bold or []) if b]}


def _envelope(*, plan: dict, headline: str, sentences: list[dict],
              chart: dict | None, rows: list[dict], shape: str, claim: str,
              chart_why: str) -> dict:
    return {
        "plan": plan,
        "answer": {"headline": headline, "sentences": sentences},
        "chart": chart, "chartWhy": chart_why, "rows": rows, "shape": shape,
        "claim": claim, "provider": "computed", "degraded": False,
        "deterministic": True, "refused": False, "cached": False,
    }


def _voice(principal: Principal) -> dict:
    """The words each persona uses for their own scope."""
    if principal.key == "ae":
        return {"scope": "your book", "poss": "your", "you": "you",
                "who": "you", "own": "you own"}
    if principal.key == "manager":
        return {"scope": "the pod", "poss": "the pod's", "you": "the pod",
                "who": "the pod", "own": "the pod holds"}
    return {"scope": "North America", "poss": "North America's",
            "you": "North America", "who": "the business", "own": "the book holds"}


def _persona_dim(principal: Principal) -> str:
    """The breakdown each persona naturally asks for when the question says 'who'."""
    return {"ae": "account", "manager": "rep"}.get(principal.key, "lob")


def _month_label(key: str) -> str:
    try:
        return pd.Period(str(key), freq="M").strftime("%b %Y")
    except (ValueError, TypeError):
        return str(key)


def _label(key: str, grain: str | None = None) -> str:
    return _month_label(key) if grain == "month" else str(key)


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def _bar(chart_id: str, title: str, rows: list[dict], *, says: list[str],
         label: str = "ACV GP", fmt: str = "currency", subtitle: str = "",
         click_dim: str | None = None, footnote: str | None = None) -> dict:
    d = REGISTRY.get(click_dim) if click_dim else None
    return C.spec(chart_id, title, "categorical×measure", rows, says=says,
                  subtitle=subtitle, measure_label=label, fmt=fmt,
                  click_dim=click_dim, count_basis=d.count_basis if d else None,
                  basis_note=d.basis_note if d else None, footnote=footnote)


def _table(chart_id: str, title: str, columns: list[dict], rows: list[dict], *,
           says: list[str], fmt: str = "currency", label: str = "ACV GP",
           subtitle: str = "", footnote: str | None = None) -> dict:
    # The compact table draws its first column as the label, the first numeric
    # column as the ranked value and up to two more NUMERIC columns beside it,
    # so anything textual is folded into the label rather than given a column.
    return C.spec(chart_id, title, "table", {"columns": columns, "rows": rows},
                  says=says, subtitle=subtitle, measure_label=label, fmt=fmt,
                  key="table.compact", footnote=footnote)


def _heat(chart_id: str, title: str, cells: list[dict], *, says: list[str],
          label: str, fmt: str, subtitle: str = "", footnote: str | None = None) -> dict:
    rows = sorted({c["row"] for c in cells})
    cols = sorted({c["col"] for c in cells})
    return C.spec(chart_id, title, "categorical×categorical×measure",
                  {"rows": rows, "cols": cols, "cells": cells}, says=says,
                  subtitle=subtitle, measure_label=label, fmt=fmt, footnote=footnote)


def _empty(plan: dict, what: str, principal: Principal,
           headline: str = "Nothing to show for that here") -> dict:
    v = _voice(principal)
    return _envelope(
        plan=plan, headline=headline,
        sentences=[
            _s(f"There are no {what} in {v['scope']} with the filters that are on.",
               "answer"),
            _s("Clear a filter, or widen the scope, to see whether the wider book has any.",
               "action"),
        ],
        chart=None, rows=[], shape="none", claim="",
        chart_why="Nothing matched, so there is nothing to draw.")


def _risk_in_scope(fs: FilterState, principal: Principal) -> pd.DataFrame:
    r = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    return r[r["opportunity_code"].isin(codes)]


def _features_in_scope(fs: FilterState, principal: Principal) -> pd.DataFrame:
    f = features()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    return f[f["opportunity_code"].isin(codes)]


def _findings_in_scope(fs: FilterState, principal: Principal) -> pd.DataFrame:
    """
    The findings routed to this role AND inside this caller's rows.

    Routing alone answered "what is the most serious finding?" for a rep with
    another rep's account, because a role-level list is the entity's list
    with the wrong types removed. The same grain-by-grain rule the findings
    endpoint applies narrows it here.
    """
    return ANOM.scoped(ANOM.for_persona(principal.key), fs, principal)


def _opp_win_rate(df: pd.DataFrame) -> tuple[float, int, int]:
    closed = df[df["is_closed"]]
    o = closed.groupby("opportunity_code")["is_won"].first()
    won = int(o.sum())
    lost = int(len(o) - won)
    return (100.0 * won / len(o) if len(o) else 0.0), won, lost


def _pod_members(principal: Principal) -> tuple[str, ...]:
    from . import personas as PR

    if principal.key == "manager":
        return tuple(principal.predicate.get("owner", ()))
    if principal.key == "ae":
        pod = PR.rep_pod(principal.identity)
        return PR._pod_members().get(pod, ()) if pod else ()
    return ()


# --------------------------------------------------------------------------- #
# Narration of an executed plan — shared with the free-text path
# --------------------------------------------------------------------------- #

_METRIC_NOUN = {
    "gp": "gross profit", "rev": "revenue", "revenue": "revenue",
    "gm": "gross margin", "count": "opportunities", "winrate": "win rate",
    "risk": "gross profit", "coverage": "coverage", "cycle": "cycle time",
    "quietdays": "days of silence",
}
_SUBSET_ADJ = {
    "all": "", "open": "open ", "won": "won ", "lost": "lost ", "closed": "closed ",
    "qualified": "qualified ", "pastdue": "past-due ", "stalled": "stalled ",
}
_SUBSET_MEANING = {
    "stalled": f"stalled means no field has changed in {STALL_DAYS} days or more",
    "pastdue": "past-due means the deal is still open after the close date the rep committed to",
    "qualified": "qualified means the rep has put the deal at Best Case or Commit",
}


def _fmt_for(metric: str) -> Callable[[float], str]:
    if metric in ("gm", "winrate"):
        return pct
    if metric == "count":
        return lambda v: count(v)
    if metric in ("cycle", "quietdays"):
        return lambda v: f"{int(round(float(v or 0)))} days"
    return money


def _dim_label(dim: str | None, plural: bool = False) -> str:
    d = REGISTRY.get(dim or "")
    base = d.label.lower() if d else "group"
    if base == "opportunity owner":
        base = "rep"
    if base == "fiscal quarter":
        base = "quarter"
    if base == "forecast category":
        base = "forecast category"
    if plural:
        return {"line of business": "lines of business",
                "industry": "industries"}.get(base, base + "s")
    return base


def _action_for(metric: str, sub: str, top: str, low: str, dim_label: str,
                principal: Principal) -> str:
    v = _voice(principal)
    if metric == "gm":
        return (f"Look at pricing and mix in {low} before chasing more volume there — "
                f"more of a thin-margin line does not fix the margin.")
    if metric == "winrate":
        return (f"Win rate barely varies by {dim_label} in this data, so do not read "
                f"the spread as skill; look at how long deals sit untouched and how "
                f"often close dates move instead.")
    if sub == "stalled":
        return (f"Start with {top}: it holds the most value that has stopped moving, "
                f"and one logged update either revives a deal or lets it be closed out.")
    if sub == "pastdue":
        return (f"Re-date what is real in {top} and close what is not — a close date "
                f"nobody believes is worse than no date.")
    if sub == "open":
        return (f"Put {v['poss']} attention on {top} first; it is the biggest lever "
                f"in this slice, so a change there moves the total most.")
    if sub == "won":
        return f"Look at what {top} did differently and repeat it where the pipeline is thinnest."
    if sub == "lost":
        return f"Review the lost deals in {top} for a reason that repeats."
    return f"Start with {top}: it carries the most weight in this view."


def _full_distribution(plan: dict, fs: FilterState, principal: Principal,
                       dim: str, metric: str, sub: str) -> dict | None:
    """
    The whole grouped distribution behind a top-N breakdown.

    The rows a plan returns are cut at `limit` with the tail folded into one
    "Other" row, which is right for a chart and wrong for a median: taken over
    the ten accounts shown, "the middle account holds $411K and the smallest
    $214K" when the middle of all 357 holds about $5K. So the breakdown is
    recomputed here without the cut, over the same scoped and filtered rows
    the plan ran on.

    Where the metric is money, members of the scope that hold none of the
    subset count as zero: an account with nothing open still exists, and a
    median of open pipeline that skips it overstates what a typical account
    holds. `zero` is how many such members there are, so the sentence can say
    so rather than quietly folding them in.
    """
    local = fs
    for f in plan.get("filters") or []:
        if f.get("dim") and f.get("value"):
            local = local.toggled(str(f["dim"]), str(f["value"]))
    scope = slice_frame(local, principal)
    df = subset(scope, sub)
    col = REGISTRY[dim].column if dim in REGISTRY else None
    if df.empty or not col or col not in df.columns:
        return None
    money_metric = metric in ("gp", "rev", "revenue", "risk")
    if money_metric:
        g = by_dimension(df, dim, local.with_(measure="gp" if metric in ("gp", "risk")
                                                else "revenue"))
        held = {str(k): float(v) for k, v in zip(g["key"], g["value"])}
        members = {str(k) for k in scope[col].dropna().unique()}
        values = [held.get(m, 0.0) for m in members | set(held)]
    elif metric == "count":
        basis = REGISTRY[dim].count_basis
        g = (df.groupby(col)["opportunity_code"].nunique() if basis == "opportunities"
             else df.groupby(col).size())
        held = {str(k): float(v) for k, v in g.items()}
        values = list(held.values())
    else:
        return None
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    # The same middle element the on-screen rows used, so the two agree when
    # nothing was cut; a true median would average the two centre values.
    median = ordered[n // 2]
    with_any = {k: v for k, v in held.items() if v > 0}
    lo_key = min(with_any, key=with_any.get) if with_any else None
    return {"n": n, "median": float(median), "zero": int(sum(1 for v in ordered if v <= 0)),
            "lo_key": lo_key, "lo_val": float(with_any[lo_key]) if lo_key else 0.0}


def narrate_result(plan: dict, result: Q.Result, fs: FilterState,
                   principal: Principal, lead: list[dict] | None = None,
                   headline: str | None = None) -> dict:
    """
    Plain-English sentences for an executed plan: answer, norm, action.

    Used by the registered answers and by `service.ask` when the narrator model
    is unavailable, so the free-text path degrades to the same prose rather
    than to a log line.
    """
    metric = str(plan.get("metric", "gp")).lower()
    sub = str(plan.get("subset", "all")).lower()
    grain = str((plan.get("time") or {}).get("grain", "none")).lower()
    bd = list(plan.get("breakdown") or [])
    fmt = _fmt_for(metric)
    noun = _METRIC_NOUN.get(metric, "value")
    adj = _SUBSET_ADJ.get(sub, "")
    v = dict(_voice(principal))
    # A plan that narrows the slice ("Networking only") must say so in the
    # scope words, or "the most gross profit in North America" reads as the
    # whole book when it is one line of it.
    narrowed = [str(f.get("value")) for f in (plan.get("filters") or []) if f.get("value")]
    if narrowed:
        v["scope"] = f"{v['poss']} {' and '.join(narrowed)} business"
    claim = result.claim
    rows = result.rows
    sentences: list[dict] = list(lead or [])

    def val(r: dict) -> float:
        try:
            return float(r.get("value", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    if result.shape == "temporal×measure" and rows:
        first, last = rows[0], rows[-1]
        peak = max(rows, key=val)
        avg = sum(val(r) for r in rows) / len(rows)
        above = sum(1 for r in rows if val(r) > avg)
        lp, ll, lf = (_label(peak["key"], "month"), _label(last["key"], "month"),
                      _label(first["key"], "month"))
        sentences.append(_s(
            f"{lp} was the strongest month for {adj}{noun} in {v['scope']} at "
            f"{fmt(val(peak))}; the run started at {fmt(val(first))} in {lf} and the "
            f"latest month, {ll}, came in at {fmt(val(last))}.",
            "answer", claim, [lp, fmt(val(peak))]))
        sentences.append(_s(
            f"The average month is {fmt(avg)}, and {above} of the {len(rows)} months "
            f"came in above it.", "norm", None, [fmt(avg)]))
        if len(rows) >= 2:
            prev = rows[-2]
            diff = val(last) - val(prev)
            word = "up" if diff > 0 else "down" if diff < 0 else "flat"
            sentences.append(_s(
                f"{ll} is {word}"
                + (f" by {fmt(abs(diff))}" if diff else "")
                + f" against {_label(prev['key'], 'month')}.", "delta"))
        sentences.append(_s(
            "Plan the coming months against the average, not the peak — one strong "
            "month is not a run rate.", "action"))
        return {"headline": headline or f"{lp} was the strongest month at {fmt(val(peak))}",
                "sentences": sentences}

    if result.shape == "categorical×categorical×measure" and rows:
        top = max(rows, key=val)
        total = sum(val(r) for r in rows)
        share = 100 * val(top) / total if total else 0.0
        by_row: dict[str, float] = {}
        for r in rows:
            by_row[r["row"]] = by_row.get(r["row"], 0.0) + val(r)
        big_row = max(by_row, key=by_row.get) if by_row else ""
        a, b = _dim_label(bd[0] if bd else None), _dim_label(bd[1] if len(bd) > 1 else None)
        sentences.append(_s(
            f"The largest cell is {top['row']} with {top['col']}, at {fmt(val(top))} of "
            f"{adj}{noun}.", "answer", claim, [f"{top['row']} with {top['col']}", fmt(val(top))]))
        if metric in ("gp", "rev", "revenue", "count"):
            sentences.append(_s(
                f"That one {a}-and-{b} pair is {pct(share)} of the {fmt(total)} in view; "
                f"{big_row} is the biggest {a} overall at {fmt(by_row[big_row])}.",
                "norm", None, [pct(share)]))
        sentences.append(_s(
            f"Read the grid by row: where a {a} is strong in one {b} and absent in "
            f"another is where the mix is uneven, and that gap is the conversation.",
            "action"))
        return {"headline": headline or f"{top['row']} with {top['col']} leads at {fmt(val(top))}",
                "sentences": sentences}

    if result.shape == "categorical×measure" and rows:
        dim = bd[0] if bd else None
        dl, dlp = _dim_label(dim), _dim_label(dim, plural=True)
        named = [r for r in rows if not str(r.get("key", "")).startswith("Other (")]
        other = [r for r in rows if str(r.get("key", "")).startswith("Other (")]
        if metric == "gm":
            # A cell with a sliver of revenue can print a margin of 100% or -300%.
            # The weakest and strongest are chosen among groups that carry at least
            # one percent of the revenue in view, so the answer names something real.
            rev_total = sum(float(r.get("revenue", 0) or 0) for r in named) or 1.0
            material = [r for r in named if float(r.get("revenue", 0) or 0) >= 0.01 * rev_total] or named
            lo, hi = min(material, key=val), max(material, key=val)
            blended = (sum(val(r) * float(r.get("revenue", 0) or 0) for r in named) / rev_total)
            sentences.append(_s(
                f"{lo['key']} has the weakest gross margin in {v['scope']} at {fmt(val(lo))}; "
                f"{hi['key']} is the strongest at {fmt(val(hi))}.",
                "answer", claim, [str(lo["key"]), fmt(val(lo))]))
            sentences.append(_s(
                f"The blended rate across everything in view is {pct(blended)}, so "
                f"{lo['key']} sits {abs(val(lo) - blended):.1f} points "
                f"{'below' if val(lo) < blended else 'above'} it. Margin here is total "
                f"gross profit over total revenue, never an average of margins.",
                "norm", None, [pct(blended)]))
            sentences.append(_s(_action_for(metric, sub, str(hi["key"]), str(lo["key"]), dl,
                                            principal), "action"))
            return {"headline": headline or f"{lo['key']} has the weakest margin at {fmt(val(lo))}",
                    "sentences": sentences}
        if metric == "winrate":
            material = [r for r in named if int(r.get("n", 0) or 0) >= 5] or named
            lo, hi = min(material, key=val), max(material, key=val)
            n_total = sum(int(r.get("n", 0) or 0) for r in named)
            overall = (sum(val(r) * int(r.get("n", 0) or 0) for r in named) / n_total
                       if n_total else 0.0)
            sentences.append(_s(
                f"Win rate runs from {fmt(val(lo))} at {lo['key']} to {fmt(val(hi))} at "
                f"{hi['key']} across {len(named)} {dlp} in {v['scope']}.",
                "answer", claim, [str(hi["key"]), fmt(val(hi))]))
            sentences.append(_s(
                f"Taken together the rate is {pct(overall)} on {count(n_total)} closed "
                f"opportunities; a {dl} with fewer than five closed deals is left out of "
                f"the high and low.", "norm", None, [pct(overall)]))
            sentences.append(_s(_action_for(metric, sub, str(hi["key"]), str(lo["key"]), dl,
                                            principal), "action"))
            return {"headline": headline or f"Win rate runs from {fmt(val(lo))} to {fmt(val(hi))}",
                    "sentences": sentences}

        hi = max(named, key=val) if named else rows[0]
        lo = min(named, key=val) if named else rows[0]
        total = sum(val(r) for r in rows)
        share = 100 * val(hi) / total if total else 0.0
        opps = int(hi.get("opps", 0) or 0)
        sentences.append(_s(
            f"{hi['key']} has the most {adj}{noun} in {v['scope']}: {fmt(val(hi))}"
            + (f" across {count(opps)} {_plural(opps, 'opportunity', 'opportunities')}"
               if opps and metric != "count" else "")
            + ".", "answer", claim, [str(hi["key"]), fmt(val(hi))]))
        # The middle and the smallest come from the WHOLE distribution, never
        # from the rows that survived the cut: over the ten shown the middle
        # account read $411K when the middle of all 357 holds about $5K. If
        # the full recount is not available the sentence says it is speaking
        # of the rows shown, and names no smallest at all.
        full = None
        try:
            full = _full_distribution(plan, fs, principal, dim, metric, sub) if dim else None
        except (KeyError, ValueError, TypeError):
            full = None
        norm = f"That is {pct(share)} of the {fmt(total)} of {adj}{noun} in view"
        if full and full["n"] >= 3:
            n_all, zero = count(full["n"]), full["zero"]
            with_any = "with any" if zero else ""
            if zero and full["median"] <= 0:
                norm += (f"; {count(zero)} of the {n_all} {dlp} "
                         f"{'holds' if zero == 1 else 'hold'} no {adj}{noun} at all, "
                         f"so the middle one holds nothing")
            else:
                norm += f"; the middle of the {n_all} {dlp} holds {fmt(full['median'])}"
                if zero:
                    norm += f" ({count(zero)} of them {'holds' if zero == 1 else 'hold'} none)"
            if full["lo_key"]:
                norm += (f", and the smallest {with_any}".rstrip()
                         + f", {full['lo_key']}, {fmt(full['lo_val'])}")
            norm += "."
        elif other and len(named) >= 3:
            shown = sorted(val(r) for r in named)
            norm += f"; the middle of the {count(len(named))} shown holds {fmt(shown[len(shown) // 2])}."
        else:
            shown = sorted(val(r) for r in named)
            norm += ((f"; the middle {dl} holds {fmt(shown[len(shown) // 2])}" if len(named) >= 3 else "")
                     + (f", and the smallest, {lo['key']}, {fmt(val(lo))}" if len(named) >= 2 else "")
                     + ".")
        if other:
            o = other[0]
            norm += (f" The remaining {str(o['key'])[7:-1]} {dlp} together hold "
                     f"{fmt(val(o))}.")
        sentences.append(_s(norm, "norm", None, [pct(share)]))
        if sub in _SUBSET_MEANING:
            sentences.append(_s(f"Here {_SUBSET_MEANING[sub]}.", "cause"))
        sentences.append(_s(_action_for(metric, sub, str(hi["key"]), str(lo["key"]), dl,
                                        principal), "action"))
        return {"headline": headline or f"{hi['key']} leads with {fmt(val(hi))}",
                "sentences": sentences}

    if rows:
        r0 = rows[0]
        sentences.append(_s(
            f"{adj.capitalize() or 'Total '}{noun} in {v['scope']} is {fmt(val(r0))}.",
            "answer", claim, [fmt(val(r0))]))
        sentences.append(_s(
            "That is the whole slice in one figure; add a breakdown to see where it sits.",
            "action"))
        return {"headline": headline or f"{fmt(val(r0))} of {adj}{noun}", "sentences": sentences}

    sentences.append(_s("No rows matched that question within the current scope.", "answer"))
    sentences.append(_s("Clear a filter or widen the scope and ask again.", "action"))
    return {"headline": headline or "Nothing matched", "sentences": sentences}


def _via_plan(plan: dict, fs: FilterState, principal: Principal,
              charts_say: list[str], *, lead: list[dict] | None = None,
              headline: str | None = None, extra: list[dict] | None = None,
              empty_what: str = "rows") -> dict:
    """Execute a plan and narrate it — the path for every plain breakdown or trend."""
    try:
        result = Q.execute(plan, fs, principal, charts_say)
    except Q.PlanError as e:
        return _empty(plan, f"{empty_what} ({e})", principal)
    if not result.rows:
        return _empty(plan, empty_what, principal)
    chart = Q.to_chart_spec(plan, result, fs)
    narrated = narrate_result(plan, result, fs, principal, lead=lead, headline=headline)
    sentences = narrated["sentences"]
    if extra:
        # The action sentence stays last: the extra context slots in before it.
        sentences = sentences[:-1] + list(extra) + sentences[-1:]
    return _envelope(plan=plan, headline=narrated["headline"], sentences=sentences,
                     chart=chart, rows=result.rows, shape=result.shape,
                     claim=result.claim, chart_why=result.chart_why)


def _plan(metric: str, sub: str, breakdown: list[str] | None = None,
          grain: str | None = None, limit: int | None = None,
          filters: list[dict] | None = None, intent: str | None = None) -> dict:
    p: dict = {"intent": intent or ("trend" if grain else "rank"),
               "metric": metric, "subset": sub}
    if breakdown:
        p["breakdown"] = breakdown
    if grain:
        p["time"] = {"grain": grain}
    if limit:
        p["limit"] = limit
    if filters:
        p["filters"] = filters
    return p


def _derived_plan(computed: str, **fields) -> dict:
    """A synthetic plan for the Details disclosure when no query plan was run."""
    return {"intent": "explain", "computedFrom": computed, **fields}


# --------------------------------------------------------------------------- #
# Templated chart questions — the basic breakdowns
# --------------------------------------------------------------------------- #


def _most_open(fs: FilterState, principal: Principal, say: list[str], *, dim: str) -> dict:
    limit = 10 if dim in ("account", "rep") else 15
    return _via_plan(_plan("gp", "open", [dim], limit=limit), fs, principal, say,
                     empty_what="open deals")


def _winrate_across(fs: FilterState, principal: Principal, say: list[str], *, dim: str) -> dict:
    return _via_plan(_plan("winrate", "closed", [dim], intent="compare"), fs, principal,
                     say, empty_what="closed deals")


def _weakest_margin(fs: FilterState, principal: Principal, say: list[str], *, dim: str) -> dict:
    return _via_plan(_plan("gm", "all", [dim], intent="compare"), fs, principal, say,
                     empty_what="lines with revenue")


register_template("Which {label} has the most open pipeline?", _most_open)
register_template("How does the win rate compare across {label}?", _winrate_across)
register_template("Which {label} has the weakest margin?", _weakest_margin)
register_template("How does win rate compare across {label}?", _winrate_across)
register_template("Which {label} holds the most open pipeline?", _most_open)


# --------------------------------------------------------------------------- #
# Persona chips
# --------------------------------------------------------------------------- #


@register("Which of my accounts has the most at stake?",
          "Which account has the most at stake?")
def _accounts_at_stake(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("risk_table", metric="risk", subset="open", breakdown=["account"])
    r = _risk_in_scope(fs, principal)
    if r.empty:
        return _empty(plan, "open deals", principal)
    v = _voice(principal)
    g = r.groupby("account_name").agg(
        gp=("acv_gp", "sum"), at_risk=("value_at_risk", "sum"),
        deals=("opportunity_code", "count"),
        flagged=("risk_band", lambda s: int(s.isin(["High", "Critical"]).sum())),
    ).sort_values("at_risk", ascending=False)
    total = float(g["at_risk"].sum())
    rows = [{"key": str(k), "value": float(x.at_risk), "opps": int(x.deals),
             "share": (100 * float(x.at_risk) / total) if total else 0.0}
            for k, x in g.iterrows()]
    top = rows[0]
    top_deal = r[r["account_name"] == top["key"]].iloc[0]
    flagged = int(g.iloc[0]["flagged"])
    chart = _bar("ask_at_stake", "Gross profit at risk, by account", rows[:12],
                 says=["risk.open.by:account"], label="GP at risk", click_dim="account",
                 subtitle="Each deal's gross profit weighted by its 0-100 risk score",
                 footnote="Risk is a transparent sum of observable facts — silence, a passed "
                          "close date, slips, shrinkage — not a learned probability.")
    sentences = [
        _s(f"{top['key']} has the most at stake: {money(top['value'])} of risk-weighted "
           f"gross profit across {count(top['opps'])} open "
           f"{_plural(top['opps'], 'deal')}, {flagged} of them scored High or Critical.",
           "answer", "risk.open.by:account", [top["key"], money(top["value"])]),
        _s(f"That is {pct(top['share'])} of everything at risk in {v['scope']} — "
           f"{money(total)} across {count(len(rows))} accounts. Risk-weighted means each "
           f"deal's gross profit multiplied by its risk score, so a big deal with a small "
           f"problem and a small deal with a big one count alike.",
           "norm", None, [pct(top["share"])]),
        _s(f"The biggest single reason there is {str(top_deal['top_driver']).lower()} on "
           f"{top_deal['opportunity_name']}, which scores {int(top_deal['risk_score'])} "
           f"out of 100.", "cause"),
        _s(f"Call {top['key']} first — {flagged or top['opps']} flagged "
           f"{_plural(flagged or top['opps'], 'deal')} at one customer is one conversation, "
           f"not {flagged or top['opps']} records.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} has {money(top['value'])} at stake",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="risk.open.by:account",
                     chart_why="One bar per account, ranked by risk-weighted gross profit.")


@register("How does my win rate compare with the book?",
          "How does my win rate compare with the team?")
def _my_win_rate(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("win rate against peers", metric="winrate", subset="closed",
                         compareTo="pod and North America")
    mine, won, lost = _opp_win_rate(slice_frame(fs, principal))
    ent, ent_w, ent_l = _opp_win_rate(facts())
    reps = rep_behaviour()
    credible = reps[~reps["thin"]]
    peer = float(credible["win_rate"].median() * 100) if len(credible) else 0.0
    spread_lo = float(credible["win_rate"].min() * 100) if len(credible) else 0.0
    spread_hi = float(credible["win_rate"].max() * 100) if len(credible) else 0.0
    v = _voice(principal)
    rows = [{"key": "You" if principal.key == "ae" else principal.identity_label,
             "value": mine, "n": won + lost}]
    members = _pod_members(principal)
    if members and principal.key == "ae":
        pod_rate, pw, pl = _opp_win_rate(facts()[facts()["owner"].isin(members)])
        rows.append({"key": "Your pod", "value": pod_rate, "n": pw + pl})
    rows.append({"key": "Typical rep (median)", "value": peer, "n": int(len(credible))})
    rows.append({"key": "North America", "value": ent, "n": ent_w + ent_l})
    chart = _bar("ask_winrate_vs_book", "Win rate against the book", rows,
                 says=["winrate.closed.by:rep"], label="Win rate", fmt="percent",
                 subtitle="Won as a share of closed opportunities",
                 footnote="Win rate is counted per opportunity, on closed deals only.")
    if not (won + lost):
        return _empty(plan, "closed deals to measure a win rate on", principal)
    diff = mine - ent
    sentences = [
        _s(f"{'You win' if principal.key == 'ae' else v['you'].capitalize() + ' wins'} "
           f"{pct(mine)} of the deals {'you close' if principal.key == 'ae' else 'it closes'} "
           f"— {won} won and {lost} lost.", "answer", "winrate.closed.by:rep",
           [pct(mine)]),
        _s(f"North America closes at {pct(ent)} and the typical rep at {pct(peer)}, so "
           f"{v['you']} {'are' if principal.key == 'ae' else 'is'} "
           f"{abs(diff):.1f} points {'above' if diff >= 0 else 'below'} the book.",
           "norm", None, [pct(ent)]),
        _s(f"Win rate barely varies between reps in this data — the credible reps run "
           f"from {pct(spread_lo, 0)} to {pct(spread_hi, 0)} — so it says little about "
           f"skill. Behaviour is where people differ.", "cause"),
        _s("Judge the book on what you control: how long deals sit untouched and how "
           "often a close date moves.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{pct(mine)} against {pct(ent)} for North America",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="winrate.closed.by:rep",
                     chart_why="Four bars: the scope, its pod, the typical rep and the entity.")


@register("Where is my pipeline concentrated by portfolio?",
          "Where is the pipeline concentrated by portfolio?")
def _my_by_portfolio(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "open", ["portfolio"]), fs, principal, say,
                     empty_what="open deals")


def _quiet_table(r: pd.DataFrame, chart_id: str, title: str) -> tuple[dict, list[dict]]:
    rows = [{"key": f"{x.opportunity_name} · {x.account_name}", "value": float(x.acv_gp),
             "quietDays": int(x.quiet_days), "riskScore": int(x.risk_score),
             "id": x.opportunity_code}
            for x in r.itertuples(index=False)]
    chart = _table(chart_id, title, [
        {"key": "key", "label": "Deal", "align": "left"},
        {"key": "value", "label": "ACV GP", "format": "currency", "align": "right"},
        {"key": "quietDays", "label": "Days quiet", "align": "right"},
        {"key": "riskScore", "label": "Risk 0-100", "align": "right"},
    ], rows, says=["quietdays.open.by:account"],
        subtitle=f"No field changed in {STALL_DAYS}+ days · biggest first",
        footnote="Days quiet is measured from the last row in the change log to the "
                 "pinned as-of date.")
    return chart, rows


@register("Which of my deals have gone quiet?", "Which deals have gone quiet?",
          "Which deals have gone quiet and are still big?")
def _quiet_deals(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("risk_table", metric="quietdays", subset="stalled",
                         breakdown=["opportunity"])
    r = _risk_in_scope(fs, principal)
    open_n = int(len(r))
    open_gp = float(r["acv_gp"].sum()) if open_n else 0.0
    q = r[r["quiet_days"].fillna(0) >= STALL_DAYS].sort_values("acv_gp", ascending=False)
    if q.empty:
        v = _voice(principal)
        return _envelope(
            plan=plan, headline="Nothing has gone quiet",
            sentences=[
                _s(f"None of the {count(open_n)} open deals in {v['scope']} has gone "
                   f"{STALL_DAYS} days without a logged change.", "answer"),
                _s("That is unusual for this book — the typical open deal has been "
                   "silent for weeks — so keep the cadence that produced it.", "norm"),
            ], chart=None, rows=[], shape="none", claim="quietdays.open.by:account",
            chart_why="Nothing is stalled, so there is nothing to list.")
    v = _voice(principal)
    chart, rows = _quiet_table(q.head(15), "ask_quiet", "Open deals that have stopped moving")
    quiet_gp = float(q["acv_gp"].sum())
    top = q.iloc[0]
    longest = q.sort_values("quiet_days", ascending=False).iloc[0]
    sentences = [
        _s(f"{count(len(q))} of {v['poss']} {count(open_n)} open deals have had nothing "
           f"logged for {STALL_DAYS} days or more, carrying {money(quiet_gp)} of gross "
           f"profit.", "answer", "quietdays.open.by:account",
           [count(len(q)), money(quiet_gp)]),
        _s(f"That is {pct(100 * quiet_gp / open_gp if open_gp else 0)} of "
           f"{v['poss']} open gross profit. The biggest is {top['opportunity_name']} at "
           f"{top['account_name']} ({money(top['acv_gp'])}, {int(top['quiet_days'])} days "
           f"quiet); the longest silence is {longest['opportunity_name']} at "
           f"{int(longest['quiet_days'])} days.", "norm"),
        _s("A deal nobody has touched in two months is usually a deal the customer has "
           "stopped talking about — the silence is the signal, whatever the stage says.",
           "cause"),
        _s(f"Start with {top['opportunity_name']} at {top['account_name']}: it is the "
           f"largest, and one logged update either revives it or lets you close it out "
           f"honestly.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{count(len(q))} deals quiet, worth {money(quiet_gp)}",
                     sentences=sentences, chart=chart, rows=rows, shape="table",
                     claim="quietdays.open.by:account",
                     chart_why="A table, because each deal needs its days quiet and its "
                               "risk score beside its value.")


@register("What should I do first today?", "What should I do first?")
def _do_first(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("actions.build", ranked_by="priority", limit=6)
    cards = ACT.build(fs, principal, limit=6)
    if not cards:
        return _empty(plan, "actions waiting", principal, "Nothing is waiting")
    rows = [{"key": c["headline"], "value": float(c["valueAtStake"]),
             "priority": int(round(c["priority"])), "dueInDays": int(c["dueInDays"])}
            for c in cards]
    chart = _table("ask_do_first", "The next moves, in order", [
        {"key": "key", "label": "Action", "align": "left"},
        {"key": "value", "label": "GP at stake", "format": "currency", "align": "right"},
        {"key": "priority", "label": "Priority", "align": "right"},
        {"key": "dueInDays", "label": "Due in days", "align": "right"},
    ], rows, says=["actions.ranked.by:priority"],
        subtitle="Ranked by what it is worth and how soon it stops being useful")
    c0 = cards[0]
    total = float(sum(c["valueAtStake"] for c in cards))
    crit = sum(1 for c in cards if c["urgencyLabel"] == "Critical")
    sentences = [
        _s(f"First: {c0['headline']}.", "answer", "actions.ranked.by:priority",
           [c0["headline"]]),
        _s(c0["why"][0] if c0["why"] else f"It fired because {c0['predicate']}.", "cause"),
        _s(f"{count(len(cards))} moves are waiting in all, worth {money(total)} of gross "
           f"profit between them; {crit} cannot wait a week.", "norm", None, [money(total)]),
        _s(c0["nextStep"] or "Take the top three; the rest can wait for the review.",
           "action"),
    ]
    return _envelope(plan=plan, headline=c0["headline"], sentences=sentences, chart=chart,
                     rows=rows, shape="table", claim="actions.ranked.by:priority",
                     chart_why="A ranked table, because the order is the answer.")


@register("Which reps in my pod hold the most open pipeline?",
          "Which reps hold the most open pipeline?")
def _pod_open_by_rep(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "open", ["rep"], limit=12), fs, principal, say,
                     empty_what="open deals")


@register("How does win rate vary across my pod?", "How does win rate vary across the pod?",
          "How does the team compare on win rate?")
def _pod_win_rate(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    reps = rep_behaviour()
    credible = reps[~reps["thin"]]
    peer = float(credible["win_rate"].median() * 100) if len(credible) else 0.0
    pod_rate, w, l = _opp_win_rate(slice_frame(fs, principal))
    extra = [_s(f"The typical rep across all of North America wins {pct(peer)}; "
                f"{_voice(principal)['scope']} as a whole wins {pct(pod_rate)} on "
                f"{count(w + l)} closed deals.", "norm", None, [pct(peer)])]
    return _via_plan(_plan("winrate", "closed", ["rep"], intent="compare"), fs, principal,
                     say, extra=extra, empty_what="closed deals")


@register("Where is the pod's pipeline by line of business?",
          "Where is the pod s pipeline by line of business?",
          "Where is the pipeline by line of business?")
def _pod_by_lob(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "open", ["lob"]), fs, principal, say,
                     empty_what="open deals")


_BEHAVIOURS: tuple[tuple[str, str, str], ...] = (
    ("shrink_rate", "value shrink", "shrink after they are logged"),
    ("inflate_rate", "sandbagging", "are logged small and grow later"),
    ("regression_rate", "forecast reversal", "get walked back from a stronger forecast call"),
    ("stall_rate", "stalled book", f"sit {STALL_DAYS}+ days without a change"),
)


def _pod_behaviour(fs: FilterState, principal: Principal) -> pd.DataFrame:
    reps = rep_behaviour()
    members = principal.predicate.get("owner")
    if members:
        reps = reps[reps["rep"].isin(members)]
    elif fs.rep:
        reps = reps[reps["rep"] == fs.rep]
    return reps


@register("Who should I coach first?", "Who to coach first?")
def _coach_first(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("rep_behaviour", metric="behaviour z-score", breakdown=["rep"])
    reps = _pod_behaviour(fs, principal)
    credible = reps[~reps["thin"]]
    if credible.empty:
        return _empty(plan, "reps with enough deals to judge", principal)
    rows = []
    for x in credible.itertuples(index=False):
        best_col, best_z = max(((c, float(getattr(x, f"{c}_z") or 0.0)) for c, _, _ in _BEHAVIOURS),
                               key=lambda t: t[1])
        label = next(lbl for c, lbl, _ in _BEHAVIOURS if c == best_col)
        rows.append({"key": x.rep, "value": round(best_z, 2), "behaviour": label,
                     "col": best_col, "rate": float(getattr(x, best_col) or 0.0),
                     "peer": float(getattr(x, f"{best_col}_peer") or 0.0),
                     "openGp": float(x.open_gp), "openDeals": int(x.open_deals)})
    rows.sort(key=lambda r: -r["value"])
    top = rows[0]
    off = sum(1 for r in rows if r["value"] >= 1.5)
    thin = int(reps["thin"].sum())
    verb = next(vb for c, _, vb in _BEHAVIOURS if c == top["col"])
    chart = _bar("ask_coach_first", "How far each rep sits from the team norm",
                 [{"key": r["key"], "value": r["value"], "opps": r["openDeals"]} for r in rows],
                 says=["risk.open.by:rep"], label="SD from norm, worst behaviour",
                 fmt="number", click_dim="rep",
                 subtitle="Standard deviations above the all-rep norm on their worst behaviour",
                 footnote="Reps with fewer than eight deals are left out — a rate on two "
                          "deals is a sample-size artefact, not a habit.")
    sentences = [
        _s(f"Coach {top['key']} first: {pct(100 * top['rate'], 0)} of their deals "
           f"{verb}, against a team norm of {pct(100 * top['peer'], 0)} — "
           f"{top['value']:.1f} standard deviations out on {top['behaviour']}.",
           "answer", "risk.open.by:rep", [top["key"], pct(100 * top["rate"], 0)]),
        _s(f"{off} of the {count(len(rows))} reps with enough deals to judge sit more "
           f"than one and a half standard deviations from the norm on at least one "
           f"behaviour"
           + (f"; {thin} more have too few deals to score." if thin else "."),
           "norm"),
        _s("These are habits, not luck: the same person shrinks, sandbags, walks a "
           "forecast back or lets deals sit again and again, and every one of them makes "
           "the forecast harder to add up. Win rate is not on the list because it barely "
           "moves between reps here.", "cause"),
        _s(f"Open with {top['key']}'s {count(top['openDeals'])} open deals, worth "
           f"{money(top['openGp'])}, and walk through the ones that {verb}.", "action"),
    ]
    return _envelope(plan=plan, headline=f"Coach {top['key']} first — {top['behaviour']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="risk.open.by:rep",
                     chart_why="One bar per rep, ranked by how far their worst behaviour sits "
                               "from the team norm.")


@register("Who is furthest from the team norm?", "Who is furthest from the norm?")
def _furthest_from_norm(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("rep_behaviour", metric="largest |z| across six behaviours",
                         breakdown=["rep"])
    reps = _pod_behaviour(fs, principal)
    credible = reps[~reps["thin"]]
    if credible.empty:
        return _empty(plan, "reps with enough deals to judge", principal)
    cols = [("win_rate", "win rate"), ("shrink_rate", "value shrink"),
            ("inflate_rate", "sandbagging"), ("regression_rate", "forecast reversal"),
            ("stall_rate", "stalled book"), ("new_business_share", "new-business mix")]
    rows = []
    for x in credible.itertuples(index=False):
        col, z = max(((c, abs(float(getattr(x, f"{c}_z") or 0.0))) for c, _ in cols),
                     key=lambda t: t[1])
        rows.append({"key": x.rep, "value": round(z, 2),
                     "behaviour": dict(cols)[col],
                     "rate": float(getattr(x, col) or 0.0),
                     "peer": float(getattr(x, f"{col}_peer") or 0.0),
                     "direction": "above" if float(getattr(x, f"{col}_z") or 0) >= 0 else "below"})
    rows.sort(key=lambda r: -r["value"])
    top = rows[0]
    chart = _bar("ask_furthest_norm", "Distance from the team norm, by rep",
                 [{"key": r["key"], "value": r["value"]} for r in rows],
                 says=["risk.open.by:rep"], label="SD from norm", fmt="number",
                 click_dim="rep", subtitle="Largest distance across six behaviours, either direction")
    sentences = [
        _s(f"{top['key']} is furthest from the norm: {top['value']:.1f} standard deviations "
           f"{top['direction']} the team on {top['behaviour']} "
           f"({pct(100 * top['rate'], 0)} against {pct(100 * top['peer'], 0)}).",
           "answer", "risk.open.by:rep", [top["key"]]),
        _s("Distance is measured against the norm of all credible reps in North America, "
           "not just this pod, so unusual means unusual for the whole company.", "norm"),
        _s(f"Being far out is not always bad — a rep far above the norm on new-business "
           f"mix is doing something worth copying. Read the direction before the number.",
           "cause"),
        _s(f"Sit down with {top['key']} and look at the deals behind the rate; a pattern "
           f"this clear is coachable.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} is {top['value']:.1f} SD out on {top['behaviour']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="risk.open.by:rep",
                     chart_why="One bar per rep, ranked by distance from the norm.")


def _stalled_answer(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    m = measures(fs, principal)
    v = _voice(principal)
    dim = _persona_dim(principal)
    st, op = m["stalled"], m["open"]
    ent = measures(fs.with_(rep=None) if principal.key == "ae" else fs,
                   _entity_principal())
    lead = [_s(
        f"{money(st['gp'])} of {v['poss']} {money(op['gp'])} open gross profit has not "
        f"changed in {STALL_DAYS} days or more — {pct(m['stalledShare'])} of it, across "
        f"{count(st['opps'])} {_plural(st['opps'], 'deal')}.",
        "answer", "gp.stalled.by:" + dim, [money(st["gp"]), pct(m["stalledShare"])])]
    extra = [_s(
        f"Across all of North America {pct(ent['stalledShare'])} of open value has stopped "
        f"moving, so {v['scope']} is "
        f"{'better' if m['stalledShare'] < ent['stalledShare'] else 'worse'} than the book.",
        "norm", None, [pct(ent["stalledShare"])])]
    if not st["opps"]:
        return _envelope(plan=_plan("gp", "stalled", [dim]),
                         headline="Nothing has stopped moving",
                         sentences=lead + extra, chart=None, rows=[], shape="none",
                         claim="gp.stalled.by:" + dim,
                         chart_why="Nothing is stalled, so there is nothing to draw.")
    return _via_plan(_plan("gp", "stalled", [dim], limit=12), fs, principal, say, lead=lead,
                     headline=f"{money(st['gp'])} has stopped moving — {pct(m['stalledShare'])} of open",
                     extra=extra, empty_what="stalled deals")


@functools.lru_cache(maxsize=1)
def _entity_principal() -> Principal:
    from . import personas as PR

    return PR.resolve("executive")


register("How much of the pod's pipeline has stopped moving?",
         "How much of the pod s pipeline has stopped moving?",
         "How much value has stopped moving?",
         "How much of the pipeline has stopped moving?")(_stalled_answer)


@register("Which accounts are biggest by gross profit?", "Who are the biggest customers?",
          "Which accounts are biggest?")
def _biggest_accounts(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "all", ["account"], limit=10), fs, principal, say,
                     empty_what="accounts")


@register("How has closed-won tracked by month?", "How much did we win each month?",
          "Which month was strongest?", "How has closed won tracked by month?")
def _won_by_month(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "won", grain="month"), fs, principal, say,
                     empty_what="won deals")


@register("Where is open pipeline concentrated by industry?",
          "Where is the pipeline concentrated by industry?")
def _open_by_industry(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "open", ["industry"]), fs, principal, say,
                     empty_what="open deals")


def _plan_note(principal: Principal) -> list[dict]:
    if not principal.predicate:
        return []
    return [_s("The plan exists only for North America as a whole — the extract carries no "
               "rep or pod quota — so the figures here are this scope's contribution "
               "against the entity plan, not a personal target.", "norm")]


@register("Are we on track against plan this quarter?", "Are we on track this quarter?",
          "Are we on track against plan?")
def _on_track(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("budget.by_quarter", quarter=CUR_QUARTER,
                         compare="plan vs won vs open")
    qs = B.by_quarter(fs, principal)
    cur = next((q for q in qs if q["isCurrent"]), None) or (qs[0] if qs else None)
    if not cur:
        return _empty(plan, "plan cells", principal)
    rows = [{"key": f"{cur['quarter']} plan", "value": cur["budgetGp"], "tone": "neutral"},
            {"key": "Won so far", "value": cur["wonGp"],
             "tone": "good" if cur["attainmentPct"] >= 100 else "warn"},
            {"key": "Still open, due this quarter", "value": cur["openGp"], "tone": "accent"}]
    chart = _bar("ask_on_track", f"{cur['quarter']} — plan, won and open", rows,
                 says=["gp.won.by:quarter", "gap.all.by:quarter"], label="ACV GP",
                 subtitle="Gross profit; open is at face value, not risk-weighted",
                 footnote="The plan was rolled evenly across three quarters while every win "
                          "so far sits in Q1 and Q2.")
    ahead = cur["attainmentPct"] >= 100
    nxt = next((q for q in qs if q["isFuture"]), None)
    v = _voice(principal)
    sentences = [
        _s((f"Yes — {cur['quarter']} is already ahead of plan: {money(cur['wonGp'])} won "
            f"against a {money(cur['budgetGp'])} plan, {pct(cur['attainmentPct'], 0)} of it."
            if ahead else
            f"Not yet — {cur['quarter']} has {money(cur['wonGp'])} won against a "
            f"{money(cur['budgetGp'])} plan, {pct(cur['attainmentPct'], 0)} of it, with "
            f"{money(cur['remainingGp'])} still to deliver."),
           "answer", "gp.won.by:quarter", [money(cur["wonGp"]), money(cur["budgetGp"])]),
        _s(f"There is {money(cur['openGp'])} of open pipeline due in the quarter"
           + (f", {cur['coverage']:.1f}x what is left to deliver." if cur["coverage"]
              else ", and nothing is left to deliver.")
           + (f" {nxt['quarter']} carries {money(nxt['remainingGp'])} of plan against "
              f"{money(nxt['openGp'])} of open pipeline." if nxt else ""),
           "norm", None, [money(cur["openGp"])]),
        *_plan_note(principal),
        _s("The plan was spread evenly across three quarters while every win so far sits "
           "in the first two, which is why the early quarters read above target and the "
           "last one reads below — a calendar position, not a performance finding.",
           "cause"),
        _s(("Decide now between funding pipeline for the next quarter and re-phasing its "
            "target." if nxt else "Keep the pace; the next quarter is where the plan bites."),
           "action"),
    ]
    return _envelope(plan=plan, headline=f"{cur['quarter']}: {pct(cur['attainmentPct'], 0)} of plan won",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.won.by:quarter",
                     chart_why="Three bars — plan, won and open — for the quarter asked about.")


@register("What needs fixing, and what is it worth?", "What needs fixing and what is it worth?",
          "What needs fixing?")
def _needs_fixing(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("anomalies.for_persona", filter="priority in (Critical, High)",
                         breakdown=["category"])
    a = _findings_in_scope(fs, principal)
    serious = a[a["priority"].isin(["Critical", "High"])]
    if serious.empty:
        return _empty(plan, "serious findings", principal)
    g = (serious.groupby("category").agg(value=("value_at_stake", "sum"),
                                         n=("anomaly_id", "count"))
         .sort_values("value", ascending=False))
    rows = [{"key": str(k), "value": float(x.value), "opps": int(x.n)} for k, x in g.iterrows()]
    total = float(g["value"].sum())
    top = rows[0]
    summ = ANOM.summary()
    blurb = ANOM.CATEGORY_BLURB.get(top["key"], "")
    chart = _bar("ask_needs_fixing", "Serious findings — gross profit at stake by category",
                 rows, says=["gp.flagged.by:anomcat"], label="ACV GP at stake",
                 click_dim="anomalyCategory",
                 subtitle="Critical and High priority only · value is gross profit",
                 footnote="Value is gross profit, not revenue; a concentration finding "
                          "counts the exposure, not a loss.")
    sentences = [
        _s(f"{count(len(serious))} findings are serious enough to fix now, with "
           f"{money(total)} of gross profit behind them; the biggest group is "
           f"{top['key']} — {count(top['opps'])} findings worth {money(top['value'])}.",
           "answer", "gp.flagged.by:anomcat", [top["key"], money(top["value"])]),
        _s(f"{count(summ['total'])} findings exist in all; {count(summ['corroborated'])} "
           f"were flagged independently by both the data-science model and this layer's "
           f"own checks, which is the strongest a finding gets.", "norm"),
        _s(f"{top['key']} asks: {blurb}" if blurb else
           f"{top['key']} is where the money sits.", "cause"),
        _s(f"{count(summ['upside'])} of the findings are chances to sell rather than "
           f"problems — take those to the account owners, and put the rest in front of "
           f"the managers who own the reps.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{count(len(serious))} serious findings worth {money(total)}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.flagged.by:anomcat",
                     chart_why="One bar per category, serious findings only, ranked by value.")


# --------------------------------------------------------------------------- #
# Closed-won by month and the plan
# --------------------------------------------------------------------------- #


def _months_vs_plan(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("budget.by_month", compare="won vs plan, months with history")
    months = [m for m in B.by_month(fs, principal) if not m["isFuture"]]
    if principal.predicate:
        # There is no plan for a rep or a pod, so "against the plan" is answered
        # honestly with the nearest true comparison: the scope's own months.
        lead_plan = _plan("gp", "won", grain="month")
        return _via_plan(lead_plan, fs, principal, say, lead=_plan_note(principal),
                         empty_what="won deals")
    if not months:
        return _empty(plan, "months with closed history", principal)
    rows = [{"key": _month_label(m["month"]), "value": float(m["attainmentPct"]),
             "tone": "good" if m["attainmentPct"] >= 100 else "danger"} for m in months]
    beat = [m for m in months if m["attainmentPct"] >= 100]
    best = max(months, key=lambda m: m["attainmentPct"])
    worst = min(months, key=lambda m: m["attainmentPct"])
    t = B.totals(fs, principal)
    chart = _bar("ask_month_vs_plan", "Won gross profit as a share of the monthly plan", rows,
                 says=["gp.won.t:month", "budget.all.t:month"], label="Attainment",
                 fmt="percent", subtitle="Months with closed history only",
                 footnote="October to December carry a plan but no closed history yet.")
    sentences = [
        _s(f"{count(len(beat))} of the {count(len(months))} months so far beat the plan"
           + (f": {', '.join(_month_label(m['month']) for m in beat)}." if beat else ".")
           + f" The best was {_month_label(best['month'])} at {pct(best['attainmentPct'], 0)} "
           f"of plan and the weakest {_month_label(worst['month'])} at "
           f"{pct(worst['attainmentPct'], 0)}.",
           "answer", "budget.all.t:month", [_month_label(best["month"]), pct(best["attainmentPct"], 0)]),
        _s(f"For the year so far that is {money(t['wonGp'])} won against a "
           f"{money(t['budgetGp'])} full-year plan — {pct(t['attainmentPct'], 0)}.",
           "norm", None, [pct(t["attainmentPct"], 0)]),
        _s("The monthly plan is the annual figure spread evenly, so a month reads red or "
           "green against a twelfth of the year rather than against a seasonal target.",
           "cause"),
        _s("Judge the year on the cumulative line, not on any one month's colour.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{count(len(beat))} of {count(len(months))} months beat the plan",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="budget.all.t:month",
                     chart_why="One bar per month, as a percentage of that month's plan.")


register("How does this compare with the plan?", "Which months beat the plan?",
         "How does closed-won compare with the plan?")(_months_vs_plan)


@register("Which month was weakest?")
def _weakest_month(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    df = subset(slice_frame(fs, principal), "won")
    if df.empty:
        return _empty(_plan("gp", "won", grain="month"), "won deals", principal)
    g = df.groupby("fiscal_month")["acv_gp"].sum().sort_values()
    lo_key, lo_val = str(g.index[0]), float(g.iloc[0])
    lead = [_s(f"{_month_label(lo_key)} was the weakest month for won gross profit in "
               f"{_voice(principal)['scope']}, at {money(lo_val)}.", "answer",
               "gp.won.t:month", [_month_label(lo_key), money(lo_val)])]
    return _via_plan(_plan("gp", "won", grain="month"), fs, principal, say, lead=lead,
                     headline=f"{_month_label(lo_key)} was weakest at {money(lo_val)}",
                     empty_what="won deals")


@register("What closed in the best month?")
def _best_month_deals(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("won deals in the strongest month", metric="gp", subset="won",
                         breakdown=["opportunity"])
    df = subset(slice_frame(fs, principal), "won")
    if df.empty:
        return _empty(plan, "won deals", principal)
    g = df.groupby("fiscal_month")["acv_gp"].sum().sort_values(ascending=False)
    best, best_val = str(g.index[0]), float(g.iloc[0])
    d = df[df["fiscal_month"] == best]
    deals = (d.groupby(["opportunity_code", "opportunity_name", "account_name"])["acv_gp"]
             .sum().reset_index().sort_values("acv_gp", ascending=False))
    rows = [{"key": f"{x.opportunity_name} · {x.account_name}", "value": float(x.acv_gp),
             "id": x.opportunity_code} for x in deals.itertuples(index=False)]
    chart = _bar("ask_best_month_deals", f"What closed in {_month_label(best)}", rows[:12],
                 says=["gp.won.by:account"], label="ACV GP",
                 subtitle=f"{count(len(deals))} deals won · biggest first")
    top = rows[0]
    top_share = 100 * top["value"] / best_val if best_val else 0.0
    sentences = [
        _s(f"{_month_label(best)} was the best month at {money(best_val)}, and the biggest "
           f"deal in it was {top['key']} at {money(top['value'])}.",
           "answer", "gp.won.by:account", [_month_label(best), top["key"]]),
        _s(f"That one deal was {pct(top_share)} of the month; {count(len(deals))} deals "
           f"closed in all.", "norm", None, [pct(top_share)]),
        _s("A month made by one or two large deals is not a run rate — the next month "
           "needs its own large deals to match it.", "cause"),
        _s("Look at who sold the top two and what they did in the last stage; that is "
           "the playbook worth repeating.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{_month_label(best)}: {money(best_val)} across {count(len(deals))} deals",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.won.by:account",
                     chart_why="One bar per deal won in the strongest month.")


# --------------------------------------------------------------------------- #
# Overdue and ageing
# --------------------------------------------------------------------------- #

_AGE_BANDS = [(-10_000, -1, "Not yet due"), (0, 30, "1-30 days over"),
              (31, 90, "31-90 days over"), (91, 10_000, "90+ days over")]


@register("How much is more than 90 days overdue?", "How much is over 90 days overdue?")
def _over_90(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("open deals by days past close date", metric="gp", subset="open",
                         breakdown=["age band"])
    df = subset(slice_frame(fs, principal), "open")
    if df.empty:
        return _empty(plan, "open deals", principal)
    v = _voice(principal)
    rows = []
    # `days_past_due` is zero, not negative, for a deal whose date has not
    # arrived, so the not-yet-due band is the past-due flag inverted rather
    # than a range on the day count.
    for lo, hi, name in _AGE_BANDS:
        d = (df[~df["is_past_due"]] if lo < 0
             else df[df["is_past_due"] & df["days_past_due"].between(lo, hi)])
        rows.append({"key": name, "value": float(d["acv_gp"].sum()),
                     "opps": int(d["opportunity_code"].nunique()),
                     "tone": "danger" if lo >= 91 else "warn" if lo >= 0 else "neutral"})
    over = rows[-1]
    open_gp = float(df["acv_gp"].sum())
    past = sum(r["value"] for r in rows[1:])
    worst = df[df["days_past_due"] >= 91].sort_values("days_past_due", ascending=False)
    chart = _bar("ask_over_90", "Open gross profit by how overdue it is", rows,
                 says=["gp.open.by:agebucket"], label="ACV GP",
                 subtitle="Days past the close date the rep committed to")
    sentences = [
        _s(f"{money(over['value'])} of {v['poss']} open gross profit is more than 90 days "
           f"past its close date — {count(over['opps'])} "
           f"{_plural(over['opps'], 'deal')}.", "answer", "gp.open.by:agebucket",
           [money(over["value"]), count(over["opps"])]),
        _s(f"That is {pct(100 * over['value'] / open_gp if open_gp else 0)} of everything "
           f"open, and {pct(100 * past / open_gp if open_gp else 0)} is past its date by "
           f"any amount.", "norm", None, [pct(100 * over["value"] / open_gp if open_gp else 0)]),
        *([_s(f"The furthest gone is {worst.iloc[0]['opportunity_name']} at "
              f"{worst.iloc[0]['account_name']}, {int(worst.iloc[0]['days_past_due'])} days "
              f"past its date and still at {worst.iloc[0]['stage']}.", "cause")]
          if len(worst) else []),
        _s("A deal three months past its own date is either real with a wrong date or "
           "not real at all; re-date the first kind and close the second.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{money(over['value'])} is more than 90 days overdue",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.open.by:agebucket",
                     chart_why="One bar per ageing band, with the 90+ band the answer.")


@register("Which line of business is worst for overdue deals?")
def _overdue_by_lob(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "pastdue", ["lob"]), fs, principal, say,
                     empty_what="past-due deals")


@register("Who owns the most overdue value?", "Who owns the most overdue deals?")
def _overdue_owner(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    dim = "account" if principal.key == "ae" else "rep"
    return _via_plan(_plan("gp", "pastdue", [dim], limit=12), fs, principal, say,
                     empty_what="past-due deals")


@register("How much value is overdue?", "How much is overdue?")
def _overdue_total(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    m = measures(fs, principal)
    v = _voice(principal)
    dim = _persona_dim(principal)
    pd_, op = m["pastDue"], m["open"]
    lead = [_s(f"{money(pd_['gp'])} of {v['poss']} {money(op['gp'])} open gross profit is "
               f"past its close date — {pct(m['pastDueShare'])}, across {count(pd_['opps'])} "
               f"{_plural(pd_['opps'], 'deal')}.", "answer", "gp.pastdue.by:" + dim,
               [money(pd_["gp"]), pct(m["pastDueShare"])])]
    if not pd_["opps"]:
        return _envelope(plan=_plan("gp", "pastdue", [dim]), headline="Nothing is overdue",
                         sentences=lead + [_s("Keep the dates honest as they approach; a "
                                              "clean book is easier to keep clean.", "action")],
                         chart=None, rows=[], shape="none", claim="gp.pastdue.by:" + dim,
                         chart_why="Nothing is past due, so there is nothing to draw.")
    return _via_plan(_plan("gp", "pastdue", [dim], limit=12), fs, principal, say, lead=lead,
                     headline=f"{money(pd_['gp'])} is overdue — {pct(m['pastDueShare'])} of open",
                     empty_what="past-due deals")


@register("Which deals are furthest past their close date?",
          "Which deals are furthest past the close date?")
def _furthest_past_due(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("risk_table", subset="pastdue", sort="days past due")
    r = _risk_in_scope(fs, principal)
    p = r[r["is_past_due"]].sort_values("days_past_due", ascending=False)
    if p.empty:
        return _empty(plan, "past-due deals", principal)
    rows = [{"key": f"{x.opportunity_name} · {x.account_name}", "value": float(x.acv_gp),
             "daysPastDue": int(x.days_past_due), "riskScore": int(x.risk_score),
             "id": x.opportunity_code} for x in p.head(15).itertuples(index=False)]
    chart = _table("ask_furthest_past_due", "Open deals furthest past their close date", [
        {"key": "key", "label": "Deal", "align": "left"},
        {"key": "value", "label": "ACV GP", "format": "currency", "align": "right"},
        {"key": "daysPastDue", "label": "Days past due", "align": "right"},
        {"key": "riskScore", "label": "Risk 0-100", "align": "right"},
    ], rows, says=["risk.open.by:account"], subtitle="Longest overdue first")
    top = p.iloc[0]
    big = p.sort_values("acv_gp", ascending=False).iloc[0]
    sentences = [
        _s(f"{top['opportunity_name']} at {top['account_name']} is furthest past its date: "
           f"{int(top['days_past_due'])} days, still at {top['stage']}, worth "
           f"{money(top['acv_gp'])}.", "answer", "risk.open.by:account",
           [top["opportunity_name"], f"{int(top['days_past_due'])} days"]),
        _s(f"{count(len(p))} deals are past their date in all, carrying "
           f"{money(float(p['acv_gp'].sum()))}; the largest of them is "
           f"{big['opportunity_name']} at {money(big['acv_gp'])}, "
           f"{int(big['days_past_due'])} days over.", "norm"),
        _s("The close date is the rep's own commitment, so a deal months past it is a "
           "forecast that was wrong and has not been corrected.", "cause"),
        _s(f"Re-date {top['opportunity_name']} if the customer is still engaged; if not, "
           f"close it out so the pipeline total stops carrying it.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['opportunity_name']}: {int(top['days_past_due'])} days past due",
                     sentences=sentences, chart=chart, rows=rows, shape="table",
                     claim="risk.open.by:account",
                     chart_why="A table, so each deal carries its days overdue and risk score.")


@register("Which of these has gone quiet?", "Which of these deals has gone quiet?")
def _gantt_quiet(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("risk_table", subset="30 highest-risk open deals",
                         filter=f"quiet_days >= {STALL_DAYS}")
    r = _risk_in_scope(fs, principal).head(30)
    q = r[r["quiet_days"].fillna(0) >= STALL_DAYS].sort_values("quiet_days", ascending=False)
    if q.empty:
        return _empty(plan, f"highest-risk deals that have been quiet {STALL_DAYS}+ days",
                      principal, "None of these has gone quiet")
    chart, rows = _quiet_table(q, "ask_gantt_quiet",
                               "Of the highest-risk deals, the ones that have gone quiet")
    top = q.iloc[0]
    sentences = [
        _s(f"{count(len(q))} of the {count(len(r))} highest-risk deals have had nothing "
           f"logged for {STALL_DAYS} days or more; the longest silence is "
           f"{top['opportunity_name']} at {top['account_name']}, {int(top['quiet_days'])} "
           f"days.", "answer", "quietdays.open.by:account",
           [count(len(q)), top["opportunity_name"]]),
        _s(f"Together they carry {money(float(q['acv_gp'].sum()))} of gross profit.",
           "norm"),
        _s("Silence and a passed close date usually arrive together — the date slipped "
           "because nobody was working the deal, not the other way round.", "cause"),
        _s(f"Log one real update on {top['opportunity_name']} this week, or close it.",
           "action"),
    ]
    return _envelope(plan=plan, headline=f"{count(len(q))} of the riskiest deals have gone quiet",
                     sentences=sentences, chart=chart, rows=rows, shape="table",
                     claim="quietdays.open.by:account",
                     chart_why="A table of the quiet ones, longest silence first.")


# --------------------------------------------------------------------------- #
# Quarters and the plan
# --------------------------------------------------------------------------- #


@register("Which quarter delivered the most?")
def _quarter_most(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "won", ["quarter"]), fs, principal, say,
                     empty_what="won deals")


@register("How much is still open?", "How much pipeline is still open?")
def _still_open(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    m = measures(fs, principal)
    v = _voice(principal)
    op = m["open"]
    lead = [_s(f"{money(op['gp'])} of gross profit is still open in {v['scope']} — "
               f"{count(op['opps'])} {_plural(op['opps'], 'opportunity', 'opportunities')} "
               f"worth {money(op['revenue'])} of revenue.", "answer", "gp.open.by:quarter",
               [money(op["gp"]), count(op["opps"])])]
    return _via_plan(_plan("gp", "open", ["quarter"]), fs, principal, say, lead=lead,
                     headline=f"{money(op['gp'])} still open", empty_what="open deals")


@register("What is the plan by quarter?", "What is the plan for each quarter?")
def _plan_by_quarter(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("budget.by_quarter", metric="plan GP", breakdown=["quarter"])
    qs = B.by_quarter(fs, principal)
    if not qs:
        return _empty(plan, "plan quarters", principal)
    rows = [{"key": q["quarter"], "value": q["budgetGp"], "won": q["wonGp"],
             "tone": "good" if q["attainmentPct"] >= 100 else "neutral" if q["isFuture"] else "warn"}
            for q in qs]
    t = B.totals(fs, principal)
    chart = _bar("ask_plan_by_quarter", "Plan gross profit by quarter", rows,
                 says=["budget.all.by:quarter"], label="Plan GP", click_dim="quarter",
                 subtitle="The entity plan, as exported")
    done = [q for q in qs if not q["isFuture"]]
    sentences = [
        _s(f"The FY26 plan is {money(t['budgetGp'])} of gross profit, split "
           + ", ".join(f"{q['quarter']} {money(q['budgetGp'])}" for q in qs) + ".",
           "answer", "budget.all.by:quarter", [money(t["budgetGp"])]),
        _s(f"{money(t['wonGp'])} has been won so far — {pct(t['attainmentPct'], 0)} of the "
           f"year — with " + ", ".join(f"{q['quarter']} at {pct(q['attainmentPct'], 0)}"
                                       for q in done) + ".", "norm", None,
           [pct(t["attainmentPct"], 0)]),
        *_plan_note(principal),
        _s("The plan was rolled evenly across the three quarters, so it does not follow "
           "the seasonality the wins actually show.", "cause"),
        _s("Read a quarter's colour against its calendar position before reading it as "
           "performance.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{money(t['budgetGp'])} plan across {count(len(qs))} quarters",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="budget.all.by:quarter", chart_why="One bar per quarter of plan.")


@register("How much plan is left this quarter?", "How much plan is left?")
def _plan_left(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("budget.by_quarter", metric="remaining plan GP", breakdown=["quarter"])
    qs = B.by_quarter(fs, principal)
    cur = next((q for q in qs if q["isCurrent"]), None)
    if not cur:
        return _empty(plan, "plan quarters", principal)
    rows = [{"key": q["quarter"], "value": q["remainingGp"], "open": q["openGp"],
             "tone": "good" if q["remainingGp"] == 0 else "warn"} for q in qs]
    chart = _bar("ask_plan_left", "Plan still to deliver, by quarter", rows,
                 says=["gap.all.by:quarter"], label="Remaining plan GP", click_dim="quarter",
                 subtitle="Plan minus won; zero means the quarter is already met")
    sentences = [
        _s((f"Nothing — {cur['quarter']} has already delivered {money(cur['wonGp'])} against "
            f"a {money(cur['budgetGp'])} plan." if cur["remainingGp"] == 0 else
            f"{money(cur['remainingGp'])} of the {cur['quarter']} plan is still to deliver: "
            f"{money(cur['wonGp'])} is won against {money(cur['budgetGp'])}."),
           "answer", "gap.all.by:quarter", [money(cur["remainingGp"])]),
        _s(f"{money(cur['openGp'])} of open pipeline is due in the quarter"
           + (f", {cur['coverage']:.1f}x what is left." if cur["coverage"] else "."),
           "norm", None, [money(cur["openGp"])]),
        *_plan_note(principal),
        _s("Coverage is measured against what is left to deliver, not the whole year's "
           "plan, so it answers the forward question rather than flattering the past.",
           "cause"),
        _s("Where a coming quarter shows plan left and thin pipeline, the choice is to "
           "build pipeline now or re-phase the target — and it is better made early.",
           "action"),
    ]
    return _envelope(plan=plan, headline=f"{money(cur['remainingGp'])} of {cur['quarter']} plan left",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gap.all.by:quarter", chart_why="One bar per quarter of remaining plan.")


@register("Where is there a target with no pipeline?", "Where is there a target but no pipeline?")
def _coverage_holes(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("budget.coverage_grid", filter="open GP = 0 and plan > 0")
    g = B.coverage_grid(fs, principal)
    holes = sorted((c for c in g["cells"] if c["isHole"]), key=lambda c: -c["budgetGp"])
    if not holes:
        return _empty(plan, f"plan cells without pipeline in {g['window']}", principal,
                      "Every plan cell has some pipeline")
    rows = [{"key": f"{c['lob']} · {c['portfolio']}", "value": c["budgetGp"], "tone": "danger"}
            for c in holes]
    total = sum(c["budgetGp"] for c in holes)
    chart = _bar("ask_coverage_holes", f"Plan cells with no open pipeline — {g['window']}",
                 rows, says=["coverage.open.by:lob+portfolio"], label="Plan GP",
                 subtitle="Line of business and portfolio cells with a target and nothing open",
                 footnote=g["footing"].get("note"))
    sentences = [
        _s(f"{count(len(holes))} {_plural(len(holes), 'cell')} of the plan for {g['window']} "
           f"{'has' if len(holes) == 1 else 'have'} a target and no open pipeline at all, "
           f"worth {money(total)} of plan; the largest is {holes[0]['lob']} in "
           f"{holes[0]['portfolio']} at {money(holes[0]['budgetGp'])}.",
           "answer", "coverage.open.by:lob+portfolio", [holes[0]["lob"], money(total)]),
        _s(f"{count(g['thin'])} more {_plural(g['thin'], 'cell')} "
           f"{'has' if g['thin'] == 1 else 'have'} pipeline but less than half of what is "
           f"left to deliver.", "norm"),
        *_plan_note(principal),
        _s("A hole is a target nobody is working, which is different from a target being "
           "missed — nothing has been lost yet, but nothing is coming either.", "cause"),
        _s(f"Assign new-pipeline work for {holes[0]['lob']} {holes[0]['portfolio']} to an "
           f"owner with a date, or decide the target is wrong and re-phase it.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{count(len(holes))} plan cells have no pipeline",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="coverage.open.by:lob+portfolio",
                     chart_why="One bar per empty cell, sized by the plan it carries.")


@register("Which area is furthest behind?", "Which area is furthest behind plan?")
def _furthest_behind(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("budget.coverage_grid", metric="open GP as % of remaining plan")
    g = B.coverage_grid(fs, principal)
    cells = [c for c in g["cells"] if c["remainingGp"] > 0]
    if not cells:
        return _empty(plan, "plan cells with anything left to deliver", principal,
                      "Nothing is behind")
    cells.sort(key=lambda c: ((c["coverage"] or 0.0), -c["remainingGp"]))
    rows = [{"key": f"{c['lob']} · {c['portfolio']}", "value": 100 * (c["coverage"] or 0.0),
             "tone": "danger" if (c["coverage"] or 0) < 0.5 else "warn" if (c["coverage"] or 0) < 1 else "good"}
            for c in cells[:12]]
    w = cells[0]
    chart = _bar("ask_furthest_behind", f"Pipeline as a share of remaining plan — {g['window']}",
                 rows, says=["coverage.open.by:lob+portfolio"], label="Coverage", fmt="percent",
                 subtitle="Lowest first · 100% means the open pipeline equals what is left")
    sentences = [
        _s(f"{w['lob']} in {w['portfolio']} is furthest behind: {money(w['remainingGp'])} "
           f"of plan still to deliver with {money(w['openGp'])} of pipeline behind it — "
           f"{pct(100 * (w['coverage'] or 0), 0)} coverage.",
           "answer", "coverage.open.by:lob+portfolio", [f"{w['lob']} in {w['portfolio']}"]),
        _s(f"{count(g['holes'])} {_plural(g['holes'], 'cell')} "
           f"{'has' if g['holes'] == 1 else 'have'} no pipeline at all and "
           f"{count(g['thin'])} {'has' if g['thin'] == 1 else 'have'} less than half of "
           f"what is left.", "norm"),
        *_plan_note(principal),
        _s("Coverage below 100% with the quarter under way means the target will not be "
           "met from what exists — only from deals not yet logged.", "cause"),
        _s(f"Re-qualify what is in {w['lob']} {w['portfolio']} before adding to it; pipeline "
           f"that exists but is not real is worse than a visible gap.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{w['lob']} · {w['portfolio']} at {pct(100 * (w['coverage'] or 0), 0)} coverage",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="coverage.open.by:lob+portfolio",
                     chart_why="One bar per plan cell, lowest coverage first.")


# --------------------------------------------------------------------------- #
# The funnel and the stage path
# --------------------------------------------------------------------------- #


def _funnel_rows(fs: FilterState, principal: Principal) -> list[dict]:
    return list(C.stage_funnel(fs, principal)["data"])


@register("Where do most deals fall out?", "Where do deals fall out?")
def _fall_out(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("charts.stage_funnel", metric="deals lost between stages",
                         cohort="entered at Identification")
    fr = _funnel_rows(fs, principal)
    cohort = fr[0]["count"] if fr else 0
    if not cohort:
        return _empty(plan, "deals whose history starts at Identification", principal)
    steps = [{"key": f"{fr[i - 1]['key']} to {r['key']}", "value": float(r["lostHere"]),
              "opps": int(r["lostHere"]), "tone": "danger"} for i, r in enumerate(fr) if i]
    lost_total = sum(s["value"] for s in steps)
    last = fr[-1]
    steps_sorted = sorted(steps, key=lambda s: -s["value"])
    top = steps_sorted[0]
    chart = _bar("ask_fall_out", "Deals lost between each stage and the next", steps,
                 says=["count.all.by:stage"], label="Deals", fmt="number",
                 subtitle=f"Cohort of {count(cohort)} deals whose history starts at Identification",
                 footnote="Anchored on the entry cohort so the count can only fall.")
    end_lost = cohort - last["count"]
    sentences = [
        _s((f"Almost nothing falls out between stages: only {count(int(lost_total))} of "
            f"{count(cohort)} deals stop short of the last stage, the most — "
            f"{count(top['opps'])} — between {top['key'].replace(' to ', ' and ')}."
            if lost_total <= 0.1 * cohort else
            f"Most deals fall out between {top['key'].replace(' to ', ' and ')}: "
            f"{count(top['opps'])} of the {count(cohort)} that started stop there."),
           "answer", "count.all.by:stage", [top["key"]]),
        _s(f"{pct(last['cohortPct'], 0)} of the cohort reaches {last['key']}, so the deals "
           f"are decided at the end, not in the middle.", "norm",
           None, [pct(last["cohortPct"], 0)]),
        _s("Deals are lost at the decision, which means qualification is not the problem "
           "in this data — closing is.", "cause"),
        _s(f"Look at what happens between {last['key']} and the decision rather than at "
           f"the early funnel.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{pct(last['cohortPct'], 0)} reach {last['key']}; {count(end_lost)} do not",
                     sentences=sentences, chart=chart, rows=steps, shape="categorical×measure",
                     claim="count.all.by:stage",
                     chart_why="One bar per step of the ladder, counting the deals that stop there.")


@register("How many deals reach the last stage?", "How many deals reach the final stage?")
def _reach_last(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("charts.stage_funnel", metric="deals reaching each stage",
                         cohort="entered at Identification")
    fr = _funnel_rows(fs, principal)
    cohort = fr[0]["count"] if fr else 0
    if not cohort:
        return _empty(plan, "deals whose history starts at Identification", principal)
    rows = [{"key": r["key"], "value": float(r["count"]), "opps": int(r["count"])} for r in fr]
    last = fr[-1]
    chart = _bar("ask_reach_last", "Deals that reached each stage", rows,
                 says=["count.all.by:stage"], label="Deals", fmt="number", click_dim="stage",
                 subtitle=f"Cohort of {count(cohort)} deals whose history starts at Identification")
    sentences = [
        _s(f"{count(last['count'])} of the {count(cohort)} deals that started at "
           f"Identification reached {last['key']} — {pct(last['cohortPct'], 0)}.",
           "answer", "count.all.by:stage", [count(last["count"]), pct(last["cohortPct"], 0)]),
        _s(f"They carry {money(last['value'])} of gross profit between them.", "norm"),
        _s("Reaching the last stage is not winning: the decision comes after it, and that "
           "is where this book's deals are lost.", "cause"),
        _s("Count the ones at the last stage whose close date has passed — those are the "
           "decisions nobody has recorded.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{pct(last['cohortPct'], 0)} of deals reach {last['key']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="count.all.by:stage", chart_why="One bar per stage of the ladder.")


@register("How much open value sits at each stage?", "How much is open at each stage?")
def _open_by_stage(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "open", ["stage"]), fs, principal, say,
                     empty_what="open deals")


@register("What is the win rate by stage?", "How does the win rate compare across stage?",
          "How does win rate compare across stage?", "What is the win rate at each stage?")
def _winrate_by_stage(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    """
    A closed deal's stage IS its outcome — Deal Won or Deal Lost — so grouping
    closed deals by stage gives 100% and 0% and nothing else. The question a
    reader means is "of the deals that got as far as each rung, how many were
    eventually won", which is read off each opportunity's recorded stage path.
    """
    plan = _derived_plan("movement_features", metric="winrate", subset="closed",
                         breakdown=["stage reached"])
    df = slice_frame(fs, principal)
    closed = df[df["is_closed"]].groupby("opportunity_code")["is_won"].first()
    if closed.empty:
        return _empty(plan, "closed deals", principal)
    f = features().set_index("opportunity_code")["stage_path_str"]
    ladder = [s for s in STAGE_ORDER if s not in ("Deal Won", "Deal Lost")]
    rank = {s: i for i, s in enumerate(ladder)}

    def furthest(code: str) -> int:
        path = str(f.get(code, "") or "")
        return max((rank.get(tok.strip(), -1) for tok in path.split(">")), default=-1)

    reach = pd.Series({c: furthest(c) for c in closed.index})
    rows = []
    for i, stage in enumerate(ladder):
        got_here = closed[reach >= i]
        if len(got_here) == 0:
            continue
        rows.append({"key": stage, "value": 100.0 * float(got_here.mean()),
                     "n": int(len(got_here)), "won": int(got_here.sum())})
    if not rows:
        return _empty(plan, "closed deals with a recorded stage path", principal)
    overall = 100.0 * float(closed.mean())
    hi = max(rows, key=lambda r: r["value"])
    lo = min(rows, key=lambda r: r["value"])
    last = rows[-1]
    spread = hi["value"] - lo["value"]
    v = _voice(principal)
    chart = _bar("ask_winrate_by_stage", "Win rate of closed deals, by the furthest stage they reached",
                 rows, says=["winrate.closed.by:stage"], label="Win rate", fmt="percent",
                 click_dim="stage",
                 subtitle="Share of closed deals that got at least this far and were then won",
                 footnote="Read from each opportunity's recorded stage path; a deal logged "
                          "straight into the middle of the ladder counts from where it appears.")
    sentences = [
        _s(f"Of the closed deals in {v['scope']} that reached {last['key']}, "
           f"{pct(last['value'])} were won — {count(last['won'])} of {count(last['n'])}; "
           f"the rate runs from {pct(lo['value'])} at {lo['key']} to {pct(hi['value'])} at "
           f"{hi['key']}.", "answer", "winrate.closed.by:stage", [last["key"], pct(last["value"])]),
        _s(f"The overall win rate on {count(len(closed))} closed deals is {pct(overall)}, so "
           f"getting further down the ladder "
           f"{'changes' if spread < 10 else 'lifts'} the odds by "
           f"{'only ' if spread < 10 else ''}{int(round(spread))} "
           f"{_plural(int(round(spread)), 'point')}.",
           "norm", None, [pct(overall)]),
        _s("Almost every closed deal in this extract reaches the last stage before it is "
           "decided, so the stage a deal is at says little about whether it will win — "
           "the decision happens after the ladder ends.", "cause"),
        _s("Do not forecast from stage alone; weight a deal by how long it has been quiet "
           "and whether its close date has moved.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{pct(last['value'])} of deals reaching {last['key']} are won",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="winrate.closed.by:stage",
                     chart_why="One bar per stage, showing the eventual win rate of deals that got that far.")


def _stage_moves(fs: FilterState, principal: Principal) -> pd.DataFrame:
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    m = movement()
    return m[(m["field"] == "stage") & (m["opportunity_code"].isin(codes))]


@register("How many deals skipped a stage?", "How many deals have skipped a stage?")
def _skipped(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    dim = "account" if principal.key == "ae" else "rep"
    plan = _derived_plan("movement_features", filter="skipped_stages > 0", breakdown=[dim])
    f = _features_in_scope(fs, principal)
    sk = f[f["skipped_stages"] > 0]
    total = int(len(f))
    if sk.empty:
        return _empty(plan, "deals that skipped a stage", principal, "No deal skipped a stage")
    o = facts().groupby("opportunity_code").agg(owner=("owner", "first"),
                                                 account=("account_name", "first"),
                                                 gp=("acv_gp", "sum"))
    j = sk.merge(o, left_on="opportunity_code", right_index=True, how="left")
    col = "account" if dim == "account" else "owner"
    g = j.groupby(col).agg(n=("opportunity_code", "count"), gp=("gp", "sum")).sort_values("n", ascending=False)
    rows = [{"key": str(k), "value": float(x.n), "opps": int(x.n), "gp": float(x.gp)}
            for k, x in g.iterrows()]
    top = rows[0]
    v = _voice(principal)
    chart = _bar("ask_skipped", f"Deals that skipped a stage, by {_dim_label(dim)}", rows[:12],
                 says=["count.all.by:stage"], label="Deals", fmt="number", click_dim=dim,
                 subtitle="A skip is a stage change that jumps at least one rung")
    sentences = [
        _s(f"{count(len(sk))} of the {count(total)} deals in {v['scope']} with a change "
           f"history skipped at least one stage — {pct(100 * len(sk) / total if total else 0)}; "
           f"{top['key']} accounts for {count(top['opps'])} of them.",
           "answer", "count.all.by:stage", [count(len(sk)), top["key"]]),
        _s(f"Those deals carry {money(float(j['gp'].sum()))} of gross profit; "
           f"{count(int((sk['skipped_stages'] >= 2).sum()))} jumped two rungs or more.",
           "norm"),
        _s("A skipped stage means the qualification steps between have no record, so the "
           "deal's own history cannot be checked — sometimes a data-entry habit, sometimes "
           "a deal logged late.", "cause"),
        _s(f"Ask {top['key']} which it is; a late-logged deal is fine, an unqualified one "
           f"is not.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{count(len(sk))} deals skipped a stage",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="count.all.by:stage",
                     chart_why=f"One bar per {_dim_label(dim)}, counting deals that skipped.")


@register("Where do deals go after Proposal?", "Where do deals go after proposal?")
def _after_proposal(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("movement log", filter="stage changes from Proposal",
                         breakdown=["next stage"])
    st = _stage_moves(fs, principal)
    frm = st[st["old_value"] == "Proposal"]
    if frm.empty:
        return _empty(plan, "recorded moves out of Proposal", principal)
    g = frm["new_value"].value_counts()
    rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    rows = [{"key": str(k), "value": float(n), "opps": int(n),
             "tone": "good" if k == "Deal Won" else "danger" if k == "Deal Lost"
             else "warn" if rank.get(str(k), 0) > rank["Proposal"] + 1 else "neutral"}
            for k, n in g.items()]
    total = int(g.sum())
    top = rows[0]
    nxt = int(g.get("Proposal Evaluation", 0))
    chart = _bar("ask_after_proposal", "Where deals go after Proposal", rows,
                 says=["count.all.by:stage"], label="Transitions", fmt="number",
                 subtitle=f"{count(total)} recorded moves out of Proposal",
                 footnote="Amber is a jump past the next rung; the steps between have no record.")
    sentences = [
        _s(f"Most go to {top['key']}: {count(top['opps'])} of the {count(total)} recorded "
           f"moves out of Proposal, {pct(100 * top['opps'] / total)}.",
           "answer", "count.all.by:stage", [top["key"], pct(100 * top["opps"] / total)]),
        _s(f"{count(nxt)} followed the ladder to Proposal Evaluation and "
           f"{count(total - nxt)} went somewhere else — forward past a rung, backwards, "
           f"or straight to a decision.", "norm"),
        _s("A move that jumps a rung leaves the evaluation step with no record, so the "
           "deal's history cannot show whether the customer actually evaluated anything.",
           "cause"),
        _s("Treat a jump from Proposal straight to Finalist as a question for the owner, "
           "not as progress.", "action"),
    ]
    return _envelope(plan=plan, headline=f"After Proposal, {pct(100 * top['opps'] / total, 0)} go to {top['key']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="count.all.by:stage", chart_why="One bar per destination stage.")


@register("Which stage loses the most deals?", "Which stage loses most deals?")
def _stage_loses(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("movement log", filter="stage changes into Deal Lost",
                         breakdown=["stage before the loss"])
    st = _stage_moves(fs, principal)
    lost = st[st["new_value"] == "Deal Lost"]
    if lost.empty:
        return _empty(plan, "recorded losses", principal, "No recorded losses")
    g = lost["old_value"].value_counts()
    rows = [{"key": str(k), "value": float(n), "opps": int(n), "tone": "danger"}
            for k, n in g.items()]
    total = int(g.sum())
    top = rows[0]
    chart = _bar("ask_stage_loses", "The stage a deal was at when it was lost", rows,
                 says=["count.all.by:stage"], label="Deals lost", fmt="number", click_dim="stage",
                 subtitle=f"{count(total)} recorded losses")
    sentences = [
        _s(f"{top['key']} loses the most: {count(top['opps'])} of the {count(total)} "
           f"recorded losses — {pct(100 * top['opps'] / total)} — were lost from there.",
           "answer", "count.all.by:stage", [top["key"], count(top["opps"])]),
        _s(f"{count(len(rows))} stages appear as the last stop before a loss"
           + (f"; the next most common is {rows[1]['key']} with {count(rows[1]['opps'])}."
              if len(rows) > 1 else "."), "norm"),
        _s("Losing at the final stage means the deal was fully worked and then lost on the "
           "decision — price, competition or a project that stalled — not on qualification.",
           "cause"),
        _s(f"Read the loss reasons on the {top['key']} losses before changing anything "
           f"earlier in the process.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} loses the most deals ({count(top['opps'])})",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="count.all.by:stage", chart_why="One bar per stage a loss came from.")


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #


@register("Who owns the deals at risk?", "Who owns the most risk?")
def _risk_owners(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    dim = "account" if principal.key == "ae" else "rep"
    col = "account_name" if dim == "account" else "owner"
    plan = _derived_plan("risk_table", filter="risk band High or Critical", breakdown=[dim])
    r = _risk_in_scope(fs, principal)
    hi = r[r["risk_band"].isin(["High", "Critical"])]
    if hi.empty:
        return _empty(plan, "deals scored High or Critical", principal, "Nothing is at high risk")
    g = hi.groupby(col).agg(gp=("acv_gp", "sum"), n=("opportunity_code", "count")).sort_values("gp", ascending=False)
    total = float(g["gp"].sum())
    rows = [{"key": str(k), "value": float(x.gp), "opps": int(x.n),
             "share": 100 * float(x.gp) / total if total else 0.0} for k, x in g.iterrows()]
    top = rows[0]
    v = _voice(principal)
    chart = _bar("ask_risk_owners", f"High and Critical risk, by {_dim_label(dim)}", rows[:12],
                 says=["risk.open.by:" + dim], label="ACV GP at risk", click_dim=dim,
                 subtitle="Gross profit on deals scoring 50 or more out of 100")
    sentences = [
        _s(f"{top['key']} {'owns' if dim == 'rep' else 'holds'} the most: {money(top['value'])} "
           f"across {count(top['opps'])} {_plural(top['opps'], 'deal')} scored High or "
           f"Critical.", "answer", "risk.open.by:" + dim, [top["key"], money(top["value"])]),
        _s(f"That is {pct(top['share'])} of the {money(total)} at high risk in "
           f"{v['scope']}, spread over {count(len(rows))} {_dim_label(dim, plural=True)}.",
           "norm", None, [pct(top["share"])]),
        _s("Risk here is a sum of facts — silence, a passed date, slips, shrinkage, skipped "
           "stages — so each flagged deal names the reason it is on the list.", "cause"),
        _s(f"{'Review' if dim == 'rep' else 'Call'} {top['key']} first and walk the "
           f"{count(top['opps'])} flagged {_plural(top['opps'], 'deal')} one by one.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} holds {money(top['value'])} at high risk",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="risk.open.by:" + dim, chart_why=f"One bar per {_dim_label(dim)}.")


@register("What makes a deal high risk?", "What makes a deal risky?")
def _what_makes_risky(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("risk_table", metric="how often each risk factor fires",
                         filter="risk band High or Critical")
    r = _risk_in_scope(fs, principal)
    hi = r[r["risk_band"].isin(["High", "Critical"])]
    if hi.empty:
        return _empty(plan, "deals scored High or Critical", principal, "Nothing is at high risk")
    counts: dict[str, int] = {}
    points: dict[str, int] = {}
    for factors in hi["risk_factors"]:
        for f in factors:
            counts[f["label"]] = counts.get(f["label"], 0) + 1
            points[f["label"]] = points.get(f["label"], 0) + int(f["points"])
    rows = sorted(({"key": k, "value": float(n), "opps": n, "points": points[k]}
                   for k, n in counts.items()), key=lambda x: -x["value"])
    top = rows[0]
    n = int(len(hi))
    chart = _bar("ask_risk_factors", "What is firing on the high-risk deals", rows,
                 says=["risk.open.by:riskband"], label="Deals", fmt="number",
                 subtitle=f"How many of the {count(n)} High and Critical deals each factor fires on",
                 footnote="Weights are fixed and visible, not fitted — a fitted weight on this "
                          "data would be fitted to noise.")
    weights = ", ".join(f"{f.label.lower()} ({f.weight})" for f in P.RISK_FACTORS[:4])
    sentences = [
        _s(f"The most common reason is {top['key'].lower()}: it fires on {count(top['opps'])} "
           f"of the {count(n)} high-risk deals, {pct(100 * top['opps'] / n)}.",
           "answer", "risk.open.by:riskband", [top["key"].lower(), pct(100 * top["opps"] / n)]),
        _s("A deal scores 0 to 100 by adding fixed points for each observable fact — the "
           f"biggest are {weights} — and anything at 50 or more is High.", "norm"),
        _s("Nothing here is learned from the outcome, because the closure model on this "
           "extract is barely better than chance; each factor is a fact the rep can be "
           "shown.", "cause"),
        _s(f"Fix the {top['key'].lower()} deals first: it is the reason that repeats, and "
           f"it is the cheapest to change.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} fires on {pct(100 * top['opps'] / n, 0)} of high-risk deals",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="risk.open.by:riskband",
                     chart_why="One bar per risk factor, counting the deals it fires on.")


@register("How much is in the riskiest band?", "How much is in the riskiest risk band?")
def _riskiest_band(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("risk_table", metric="gp", breakdown=["risk band"])
    r = _risk_in_scope(fs, principal)
    if r.empty:
        return _empty(plan, "open deals", principal)
    order = ["Critical", "High", "Watch", "Low"]
    tones = {"Critical": "danger", "High": "danger", "Watch": "warn", "Low": "good"}
    g = r.groupby("risk_band").agg(gp=("acv_gp", "sum"), n=("opportunity_code", "count"))
    rows = [{"key": b, "value": float(g["gp"].get(b, 0.0)), "opps": int(g["n"].get(b, 0)),
             "tone": tones[b]} for b in order if b in g.index]
    total = float(r["acv_gp"].sum())
    top = rows[0]
    v = _voice(principal)
    chart = _bar("ask_riskiest_band", "Open gross profit by risk band", rows,
                 says=["gp.open.by:riskband"], label="ACV GP", click_dim="riskBand",
                 subtitle="Critical is 75 or more out of 100; High is 50 to 74")
    sentences = [
        _s(f"The riskiest band with anything in it is {top['key']}: {money(top['value'])} "
           f"across {count(top['opps'])} {_plural(top['opps'], 'deal')}.",
           "answer", "gp.open.by:riskband", [top["key"], money(top["value"])]),
        _s(f"That is {pct(100 * top['value'] / total if total else 0)} of the "
           f"{money(total)} open in {v['scope']}.", "norm", None,
           [pct(100 * top["value"] / total if total else 0)]),
        _s("A Critical score needs several facts to fire at once — typically a deal that "
           "has gone quiet, passed its close date and slipped before — so it is rarely a "
           "surprise to the owner.", "cause"),
        _s(f"Work the {top['key']} band deal by deal; each one names its own reason.",
           "action"),
    ]
    return _envelope(plan=plan, headline=f"{money(top['value'])} sits in the {top['key']} band",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.open.by:riskband", chart_why="One bar per risk band.")


@register("Which deals are worst?", "Which deals are the worst?")
def _worst_deals(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("risk_table", sort="risk score", limit=15)
    r = _risk_in_scope(fs, principal).head(15)
    if r.empty:
        return _empty(plan, "open deals", principal)
    rows = [{"key": f"{x.opportunity_name} · {x.account_name}", "value": float(x.acv_gp),
             "riskScore": int(x.risk_score),
             "quietDays": int(x.quiet_days) if pd.notna(x.quiet_days) else 0,
             "id": x.opportunity_code} for x in r.itertuples(index=False)]
    chart = _table("ask_worst_deals", "The highest-risk open deals", [
        {"key": "key", "label": "Deal", "align": "left"},
        {"key": "value", "label": "ACV GP", "format": "currency", "align": "right"},
        {"key": "riskScore", "label": "Risk 0-100", "align": "right"},
        {"key": "quietDays", "label": "Days quiet", "align": "right"},
    ], rows, says=["risk.open.by:account"], subtitle="Highest risk score first")
    top = r.iloc[0]
    factors = ", ".join(f["label"].lower() for f in top["risk_factors"][:3]) or "no factor"
    sentences = [
        _s(f"{top['opportunity_name']} at {top['account_name']} is the worst, scoring "
           f"{int(top['risk_score'])} out of 100 on {money(top['acv_gp'])} of gross profit.",
           "answer", "risk.open.by:account", [top["opportunity_name"], f"{int(top['risk_score'])} out of 100"]),
        _s(f"Together these {count(len(r))} deals carry {money(float(r['acv_gp'].sum()))}; "
           f"{count(int((r['risk_band'] == 'Critical').sum()))} of them are Critical.",
           "norm"),
        _s(f"On the worst one the score comes from {factors}.", "cause"),
        _s(f"Start at the top of the list — {top['opportunity_name']} — and either log a "
           f"real next step or close it out.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['opportunity_name']} scores {int(top['risk_score'])}",
                     sentences=sentences, chart=chart, rows=rows, shape="table",
                     claim="risk.open.by:account",
                     chart_why="A table, so each deal shows its score and its silence.")


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


@register("Which kind of problem is biggest?", "Which kind of finding is biggest?")
def _biggest_problem(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("anomalies.for_persona", metric="value at stake", breakdown=["category"])
    a = _findings_in_scope(fs, principal)
    if a.empty:
        return _empty(plan, "findings", principal)
    g = (a.groupby("category").agg(value=("value_at_stake", "sum"), n=("anomaly_id", "count"))
         .sort_values("value", ascending=False))
    rows = [{"key": str(k), "value": float(x.value), "opps": int(x.n)} for k, x in g.iterrows()]
    by_n = max(rows, key=lambda r: r["opps"])
    top = rows[0]
    total = float(g["value"].sum())
    chart = _bar("ask_biggest_problem", "Findings by category — gross profit at stake", rows,
                 says=["gp.flagged.by:anomcat"], label="ACV GP at stake",
                 click_dim="anomalyCategory",
                 subtitle=f"{count(len(a))} findings routed to this profile",
                 footnote="Value is gross profit, not revenue.")
    sentences = [
        _s(f"By money, {top['key']}: {count(top['opps'])} findings with {money(top['value'])} "
           f"behind them — {pct(100 * top['value'] / total if total else 0)} of everything "
           f"flagged.", "answer", "gp.flagged.by:anomcat", [top["key"], money(top["value"])]),
        _s(f"By count the biggest is {by_n['key']} with {count(by_n['opps'])} findings"
           + (" — the same group." if by_n["key"] == top["key"] else
              f", but they are worth {money(by_n['value'])} between them."), "norm"),
        _s(f"{top['key']} asks: {ANOM.CATEGORY_BLURB.get(top['key'], '')}", "cause"),
        _s("Sort by what it is worth, not by how many there are; one finding on a large "
           "account outweighs a dozen small ones.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} is biggest at {money(top['value'])}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.flagged.by:anomcat", chart_why="One bar per category, by value at stake.")


@register("What is the most serious finding?", "What is the most serious problem?")
def _most_serious(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("anomalies.for_persona", sort="triage score", limit=8)
    a = _findings_in_scope(fs, principal).head(8)
    if a.empty:
        return _empty(plan, "findings", principal)
    rows = [{"key": f"{x.entity_label} · {str(x.anomaly_type).replace('_', ' ')}",
             "value": float(x.value_at_stake or 0), "severity": int(x.severity),
             "triage": int(x.triage), "id": x.anomaly_id} for x in a.itertuples(index=False)]
    chart = _table("ask_most_serious", "The most serious findings", [
        {"key": "key", "label": "Finding", "align": "left"},
        {"key": "value", "label": "GP at stake", "format": "currency", "align": "right"},
        {"key": "severity", "label": "Severity", "align": "right"},
        {"key": "triage", "label": "Triage", "align": "right"},
    ], rows, says=["sev.flagged.by:anomcat"], subtitle="Ranked by severity and money together")
    top = a.iloc[0]
    evidence = str(top["evidence"] or "").strip().rstrip(".")
    sentences = [
        _s(f"The most serious is {str(top['anomaly_type']).replace('_', ' ')} on "
           f"{top['entity_label']}: {money(float(top['value_at_stake'] or 0))} at stake, "
           f"severity {int(top['severity'])} of 100.", "answer", "sev.flagged.by:anomcat",
           [top["entity_label"]]),
        _s((evidence[:220] + ".") if evidence else f"It sits in {top['category']}.", "cause"),
        _s(("Both the data-science model and this layer's own checks flagged it "
            "independently, which is the strongest a finding gets."
            if bool(top["corroborated"]) else
            f"It was raised by {'the data-science model' if top['provenance'] == 'ds-model' else 'this layer'}; "
            f"{count(int(a['corroborated'].sum()))} of the top {count(len(a))} are confirmed "
            f"by both."), "norm"),
        _s(str(top["recommended_action"] or "Take it to the owner named on the card.")
           .strip().rstrip(".") + ".", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['entity_label']}: {str(top['anomaly_type']).replace('_', ' ')}",
                     sentences=sentences, chart=chart, rows=rows, shape="table",
                     claim="sev.flagged.by:anomcat",
                     chart_why="A table, so each finding carries its severity and its money.")


@register("How much money is affected?", "How much money is at stake?")
def _money_affected(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("anomalies.for_persona", metric="value at stake", breakdown=["priority"])
    a = _findings_in_scope(fs, principal)
    if a.empty:
        return _empty(plan, "findings", principal)
    order = ["Critical", "High", "Medium", "Low"]
    tones = {"Critical": "danger", "High": "danger", "Medium": "warn", "Low": "neutral"}
    g = a.groupby("priority").agg(value=("value_at_stake", "sum"), n=("anomaly_id", "count"))
    rows = [{"key": p, "value": float(g["value"].get(p, 0.0)), "opps": int(g["n"].get(p, 0)),
             "tone": tones[p]} for p in order if p in g.index]
    total = float(a["value_at_stake"].sum())
    crit = next((r for r in rows if r["key"] == "Critical"), rows[0])
    up = a[a["framing"] == "opportunity"]
    chart = _bar("ask_money_affected", "Gross profit at stake by priority", rows,
                 says=["gp.flagged.by:anomcat"], label="ACV GP at stake",
                 subtitle=f"{count(len(a))} findings · value is gross profit",
                 footnote="Value is gross profit, not revenue — roughly six times smaller "
                          "than the same deals on a revenue chart.")
    sentences = [
        _s(f"{money(total)} of gross profit sits behind the {count(len(a))} findings routed "
           f"to this profile; {money(crit['value'])} of it is on the {count(crit['opps'])} "
           f"{crit['key']} ones.", "answer", "gp.flagged.by:anomcat", [money(total)]),
        _s(f"{money(float(up['value_at_stake'].sum()))} of that is upside — "
           f"{count(len(up))} findings framed as room to sell rather than as a problem — "
           f"so the money at risk is {money(total - float(up['value_at_stake'].sum()))}.",
           "norm"),
        _s("A concentration finding counts the whole exposure, not a loss, which is why "
           "the total is larger than the pipeline it refers to.", "cause"),
        _s(f"Start with the {crit['key']} findings; they are few and they carry the most.",
           "action"),
    ]
    return _envelope(plan=plan, headline=f"{money(total)} at stake across {count(len(a))} findings",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.flagged.by:anomcat", chart_why="One bar per priority level.")


# --------------------------------------------------------------------------- #
# Margin and mix
# --------------------------------------------------------------------------- #


@register("Where is the margin weakest?", "Where is margin weakest?")
def _margin_weakest_cell(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("gross margin by line of business and portfolio", metric="gm",
                         breakdown=["lob", "portfolio"])
    df = slice_frame(fs, principal)
    if df.empty:
        return _empty(plan, "lines", principal)
    rev_total = float(df["acv_revenue"].sum())
    blended = 100 * float(df["acv_gp"].sum()) / rev_total if rev_total else 0.0
    g = df.groupby(["lob", "portfolio"]).agg(revenue=("acv_revenue", "sum"),
                                             gp=("acv_gp", "sum")).reset_index()
    g["gm"] = np.where(g["revenue"] != 0, 100 * g["gp"] / g["revenue"], 0.0)
    cells = [{"row": x.lob, "col": x.portfolio, "value": float(x.gm),
              "secondary": float(x.revenue), "secondaryLabel": "revenue"}
             for x in g.itertuples(index=False) if x.revenue > 0]
    material = [c for c in cells if c["secondary"] >= 0.01 * rev_total] or cells
    lo = min(material, key=lambda c: c["value"])
    hi = max(material, key=lambda c: c["value"])
    chart = _heat("ask_margin_weakest", "Gross margin by line of business and portfolio", cells,
                  says=["gm.all.by:lob+portfolio"], label="GM %", fmt="percent",
                  subtitle=f"Blended rate {pct(blended)} · cells under 1% of revenue ignored for the answer",
                  footnote="Margin is total gross profit over total revenue per cell.")
    sentences = [
        _s(f"The weakest margin is {lo['row']} {lo['col']} at {pct(lo['value'])}, on "
           f"{money(lo['secondary'])} of revenue.", "answer", "gm.all.by:lob+portfolio",
           [f"{lo['row']} {lo['col']}", pct(lo["value"])]),
        _s(f"The blended rate across {_voice(principal)['scope']} is {pct(blended)}; the "
           f"strongest cell is {hi['row']} {hi['col']} at {pct(hi['value'])}.",
           "norm", None, [pct(blended)]),
        _s("Product resale carries thin margin by nature and services carry thick, so a "
           "line that is mostly product will read cold however well it is priced.",
           "cause"),
        _s(f"Attach services to the {lo['row']} product deals rather than trying to price "
           f"the product up.", "action"),
    ]
    return _envelope(plan=plan, headline=f"{lo['row']} {lo['col']} is weakest at {pct(lo['value'])}",
                     sentences=sentences, chart=chart, rows=cells,
                     shape="categorical×categorical×measure", claim="gm.all.by:lob+portfolio",
                     chart_why="A grid of margin by line of business and portfolio.")


@register("Which line of business is biggest?", "Which line of business is the biggest?")
def _biggest_lob(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("rev", "all", ["lob"]), fs, principal, say, empty_what="lines")


@register("How does the mix differ by line of business?", "How does the mix differ by lob?")
def _mix_by_lob(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("rev", "all", ["lob", "portfolio"], intent="composition"),
                     fs, principal, say, empty_what="lines")


# --------------------------------------------------------------------------- #
# Accounts and industries
# --------------------------------------------------------------------------- #


@register("How much do the top five carry?", "How much do the top five accounts carry?")
def _top_five(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("accounts.concentration", metric="gp", breakdown=["account"], limit=5)
    c = ACC.concentration(fs, principal, top=5)
    if not c["accounts"]:
        return _empty(plan, "accounts", principal)
    rows = [{"key": a["account_name"], "value": float(a["gp"]), "share": float(a["share"]),
             "opps": int(a["opportunities"])} for a in c["accounts"]]
    if c["accountsBeyondTop"]:
        rows.append({"key": f"Other ({c['accountsBeyondTop']})", "value": float(c["accountsBeyondTopGp"]),
                     "share": 100 - c["top5AccountShare"], "tone": "neutral"})
    chart = _bar("ask_top_five", "Gross profit — the top five accounts and everyone else", rows,
                 says=["gp.all.by:account"], label="ACV GP", click_dim="account",
                 subtitle=f"{count(len(c['accounts']) + c['accountsBeyondTop'])} accounts in scope")
    top = c["accounts"][0]
    v = _voice(principal)
    sentences = [
        _s(f"The top five carry {pct(c['top5AccountShare'])} of {v['poss']} gross profit — "
           f"{money(sum(a['gp'] for a in c['accounts']))} of {money(c['totalGp'])}.",
           "answer", "gp.all.by:account", [pct(c["top5AccountShare"])]),
        _s(f"{top['account_name']} alone is {pct(c['topAccountShare'])}, and the other "
           f"{count(c['accountsBeyondTop'])} accounts share the remaining "
           f"{pct(100 - c['top5AccountShare'])}.", "norm", None, [pct(c["topAccountShare"])]),
        _s("Concentration this high means one customer's budget cycle moves the whole "
           "book, which is a retention question before it is a growth one.", "cause"),
        _s(f"Name a retention owner and a cadence for {top['account_name']}, and build "
           f"value outside the top five on purpose.", "action"),
    ]
    return _envelope(plan=plan, headline=f"Top five carry {pct(c['top5AccountShare'])} of gross profit",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.all.by:account", chart_why="Five named bars and one grouped tail.")


@register("Which industries are they in?", "Which industries are the biggest customers in?")
def _top_industries(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("accounts.concentration", metric="gp", breakdown=["industry"],
                         filter="top 10 accounts")
    c = ACC.concentration(fs, principal, top=10)
    if not c["accounts"]:
        return _empty(plan, "accounts", principal)
    df = slice_frame(fs, principal)
    codes = [a["account_code"] for a in c["accounts"]]
    ind = df[df["account_code"].isin(codes)].groupby("industry").agg(
        gp=("acv_gp", "sum"), n=("account_code", "nunique")).sort_values("gp", ascending=False)
    rows = [{"key": str(k), "value": float(x.gp), "opps": int(x.n)} for k, x in ind.iterrows()]
    top = rows[0]
    names = "; ".join(f"{a['account_name']} is in "
                      f"{df.loc[df['account_code'] == a['account_code'], 'industry'].iloc[0]}"
                      for a in c["accounts"][:3])
    chart = _bar("ask_top_industries", "The top ten accounts, grouped by industry", rows,
                 says=["gp.all.by:industry"], label="ACV GP", click_dim="industry",
                 subtitle="Gross profit of the ten largest accounts, by their industry")
    all_ind = c["industries"][0] if c["industries"] else None
    sentences = [
        _s(f"{names}.", "answer", "gp.all.by:industry", [top["key"]]),
        _s(f"Across the top ten, {top['key']} is the largest industry with "
           f"{count(top['opps'])} of them and {money(top['value'])}"
           + (f"; across the whole book {all_ind['industry']} is "
              f"{pct(c['topIndustryShare'])} of gross profit." if all_ind else "."),
           "norm", None, [top["key"]]),
        _s("When the largest accounts share an industry, the book is exposed to one "
           "sector's budget cycle as well as to a handful of customers.", "cause"),
        _s(f"Look for the next {top['key']} account that does not buy from us yet — the "
           f"references are already in the book.", "action"),
    ]
    return _envelope(plan=plan, headline=f"The biggest customers are mostly {top['key']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.all.by:industry", chart_why="One bar per industry, top ten accounts only.")


@register("Which industry brings the most?", "Which industry brings the most gross profit?")
def _industry_most(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("gp", "all", ["industry"]), fs, principal, say, empty_what="lines")


@register("Where does Networking business come from?", "Where does networking business come from?")
def _networking_from(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _plan("gp", "all", ["industry"], filters=[{"dim": "lob", "op": "eq", "value": "Networking"}])
    return _via_plan(plan, fs, principal, say, empty_what="Networking lines",
                     headline=None)


@register("Which industry wins most often?", "Which industry wins the most often?")
def _industry_wins(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    return _via_plan(_plan("winrate", "closed", ["industry"], intent="compare"), fs,
                     principal, say, empty_what="closed deals")


# --------------------------------------------------------------------------- #
# Whitespace and cross-sell
# --------------------------------------------------------------------------- #


def _ws_rows(ws: list[dict]) -> list[dict]:
    return [{"key": w["accountName"], "value": float(w["estimatedGp"]),
             "share": w["peerAttachRate"] * 100, "lob": w["recommendedLob"]} for w in ws]


@register("Which customer is the best opportunity?", "Which account is the best opportunity?")
def _best_ws(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("accounts.whitespace", sort="attach rate x peer median GP")
    ws = ACC.whitespace(fs, principal, limit=12)
    if not ws:
        return _empty(plan, "accounts missing a whole line of business", principal,
                      "No whole-line gap in scope")
    rows = _ws_rows(ws)
    b = ws[0]
    chart = _bar("ask_best_ws", "Accounts with room to grow — what the gap is worth", rows,
                 says=["gp.all.by:account"], label="Peer median GP", click_dim="account",
                 subtitle="Median gross profit that size-matched accounts earn from the missing line",
                 footnote=XS.CAVEAT)
    sentences = [
        _s(f"{b['accountName']} is the best opportunity: it buys {', '.join(b['holds'])} but "
           f"not {b['recommendedLob']}, which {pct(b['peerAttachRate'] * 100, 0)} of similar "
           f"customers take.", "answer", "gp.all.by:account", [b["accountName"], b["recommendedLob"]]),
        _s(f"Size-matched accounts that hold {b['recommendedLob']} earn a median "
           f"{money(b['estimatedGp'])} from it; {b['accountName']} is a "
           f"{money(b['gp'])} account today.", "norm", None, [money(b["estimatedGp"])]),
        _s("The estimate is a median of comparable accounts, never the biggest one, so "
           "it is a number the owner can defend in the room.", "cause"),
        _s(f"Open the {b['recommendedLob']} conversation at {b['accountName']} with "
           f"{b['owner']}. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"{b['accountName']}: add {b['recommendedLob']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.all.by:account", chart_why="One bar per account, by what the gap is worth.")


@register("What should we sell them?", "What should I sell them?")
def _sell_them(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("accounts.whitespace", breakdown=["recommended line of business"])
    ws = ACC.whitespace(fs, principal, limit=999)
    if not ws:
        return _empty(plan, "accounts missing a whole line of business", principal,
                      "No whole-line gap in scope")
    g: dict[str, dict] = {}
    for w in ws:
        e = g.setdefault(w["recommendedLob"], {"n": 0, "gp": 0.0})
        e["n"] += 1
        e["gp"] += w["estimatedGp"]
    rows = sorted(({"key": k, "value": float(e["n"]), "opps": e["n"], "gp": e["gp"]}
                   for k, e in g.items()), key=lambda r: -r["value"])
    top = rows[0]
    b = ws[0]
    chart = _bar("ask_sell_them", "What the accounts with room to grow are missing", rows,
                 says=["xsell.theme.by:offering"], label="Accounts", fmt="number", click_dim="lob",
                 subtitle="Number of accounts whose best gap is each line of business")
    sentences = [
        _s(f"{top['key']}, most often: it is the best gap at {count(top['opps'])} of the "
           f"{count(len(ws))} accounts with room to grow, worth a median "
           f"{money(top['gp'])} between them.", "answer", "xsell.theme.by:offering",
           [top["key"], count(top["opps"])]),
        _s(f"The single clearest case is {b['recommendedLob']} at {b['accountName']}, which "
           f"{pct(b['peerAttachRate'] * 100, 0)} of similar customers already take.", "norm"),
        _s("The direction matters: customers who buy Security almost always buy Networking, "
           "but Networking customers mostly do not buy Security — so the play runs one way.",
           "cause"),
        _s(f"Brief the owners on {top['key']} as one play rather than {count(top['opps'])} "
           f"separate ideas. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"Sell {top['key']} — {count(top['opps'])} accounts are missing it",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.theme.by:offering", chart_why="One bar per missing line of business.")


@register("How much is this worth?", "How much is the whitespace worth?")
def _ws_worth(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("accounts.whitespace", metric="sum of peer median GP")
    ws = ACC.whitespace(fs, principal, limit=999)
    if not ws:
        return _empty(plan, "accounts missing a whole line of business", principal,
                      "No whole-line gap in scope")
    total = sum(w["estimatedGp"] for w in ws)
    current = sum(w["gp"] for w in ws)
    rows = _ws_rows(ws[:12])
    chart = _bar("ask_ws_worth", "What each gap is worth, by account", rows,
                 says=["gp.all.by:account"], label="Peer median GP", click_dim="account",
                 subtitle="Median gross profit comparable accounts earn from the missing line",
                 footnote=XS.CAVEAT)
    sentences = [
        _s(f"Around {money(total)} of gross profit a year, if every one of the "
           f"{count(len(ws))} accounts bought its missing line at the rate comparable "
           f"accounts do.", "answer", "gp.all.by:account", [money(total)]),
        _s(f"Those accounts are worth {money(current)} today, so the gaps add "
           f"{pct(100 * total / current if current else 0, 0)} on top of what they already "
           f"buy; the largest single gap is {money(ws[0]['estimatedGp'])} at "
           f"{ws[0]['accountName']}.", "norm", None, [money(current)]),
        _s("Each figure is a median of size-matched accounts that hold the line, not a "
           "forecast — some will never buy it.", "cause"),
        _s(f"Take the top three by value to their owners this week. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"About {money(total)} of room to grow across {count(len(ws))} accounts",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="gp.all.by:account", chart_why="One bar per account, by what the gap is worth.")


def _xs_rows(recs: list[dict]) -> list[dict]:
    return [{"key": f"{r['accountName']} · {r['offering']}", "value": float(r["peerGp"]),
             "opps": int(r["methodCount"]), "confidence": r["confidence"]} for r in recs]


@register("Which of these should I start with?", "Which of these should we start with?")
def _xs_start(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.unified", sort="methods agreeing, then confidence")
    recs = XS.unified(fs, principal, limit=10)
    if not recs:
        return _empty(plan, "cross-sell recommendations", principal, "No recommendation in scope")
    rows = _xs_rows(recs)
    r0 = recs[0]
    chart = _bar("ask_xs_start", "Growth ideas, in the order to take them", rows,
                 says=["xsell.rec.by:account"], label="Peer GP",
                 subtitle="Ranked by how many independent methods agree, not by size",
                 footnote=XS.CAVEAT)
    reason = r0["reasons"][0]["text"].rstrip(".") if r0["reasons"] else r0["confidenceMeaning"].rstrip(".")
    sentences = [
        _s(f"Start with {r0['offering']} at {r0['accountName']}: {r0['confidence'].lower()} "
           f"confidence, found by {count(r0['methodCount'])} "
           f"{_plural(r0['methodCount'], 'method')}.", "answer", "xsell.rec.by:account",
           [r0["accountName"], r0["offering"]]),
        _s(reason[:240] + ".", "cause"),
        _s(f"A peer account earns {money(r0['peerGp'])} of gross profit on this offering; "
           f"{r0['accountName']} is a {money(r0['accountGp'])} account today.", "norm",
           None, [money(r0["peerGp"])]),
        _s(f"Ask {r0['owner']} to open it this week. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"Start with {r0['offering']} at {r0['accountName']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.rec.by:account", chart_why="One bar per idea, in ranked order.")


@register("Why is this account a good fit?", "Why is this a good fit?")
def _xs_why(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.unified", detail="reasons behind the top recommendation")
    recs = XS.unified(fs, principal, limit=1)
    if not recs:
        return _empty(plan, "cross-sell recommendations", principal, "No recommendation in scope")
    r0 = recs[0]
    mine = XS.for_account(r0["accountCode"])
    # `for_account` carries the method list rather than a count, so the count is
    # taken from its length here.
    rows = [{"key": r["offering"], "value": float(r["peerGp"]),
             "opps": int(len(r.get("methods") or []))}
            for r in mine] or _xs_rows([r0])
    chart = _bar("ask_xs_why", f"Everything suggested for {r0['accountName']}", rows,
                 says=["xsell.rec.by:account"], label="Peer GP",
                 subtitle="One bar per suggested offering at this account", footnote=XS.CAVEAT)
    reasons = [_s(x["text"].rstrip(".")[:240] + ".", "cause") for x in r0["reasons"][:2]]
    sentences = [
        _s(f"{r0['accountName']} fits {r0['offering']} because {r0['confidenceMeaning'].rstrip('.').lower()}"
           f" — {r0['confidence'].lower()} confidence.", "answer", "xsell.rec.by:account",
           [r0["accountName"], r0["offering"]]),
        *reasons,
        _s(f"It already buys {count(len(r0['holds']))} offerings across "
           f"{', '.join(r0['holdsLobs'])}, so this is {r0['kind']}.", "norm"),
        _s(f"Lead with the peer examples when {r0['owner']} raises it. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"{r0['accountName']}: {r0['confidence'].lower()} confidence on {r0['offering']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.rec.by:account", chart_why="Every suggestion at this one account.")


@register("What is a peer account worth on this?", "What is a peer account worth?")
def _xs_peer_worth(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.unified", metric="peer GP", limit=10)
    recs = XS.unified(fs, principal, limit=10)
    if not recs:
        return _empty(plan, "cross-sell recommendations", principal, "No recommendation in scope")
    rows = sorted(_xs_rows(recs), key=lambda r: -r["value"])
    r0 = recs[0]
    top = rows[0]
    chart = _bar("ask_xs_peer_worth", "What a peer account earns on each idea", rows,
                 says=["xsell.rec.by:account"], label="Peer GP",
                 subtitle="Average gross profit that accounts holding the offering earn from it",
                 footnote=XS.CAVEAT)
    sentences = [
        _s(f"On the top idea — {r0['offering']} at {r0['accountName']} — a peer account "
           f"earns {money(r0['peerGp'])} of gross profit on {money(r0['peerRevenue'])} of "
           f"revenue.", "answer", "xsell.rec.by:account", [money(r0["peerGp"])]),
        _s(f"Across the top ten ideas the richest is {top['key']} at {money(top['value'])}; "
           f"the ten together are worth {money(sum(r['value'] for r in rows))} at peer rates.",
           "norm", None, [money(top["value"])]),
        _s("Peer value is what comparable accounts already pay, not a quote — the actual "
           "deal will be sized to this customer.", "cause"),
        _s(f"Use the peer figure to size the first conversation, not the forecast. "
           f"{XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"A peer earns {money(r0['peerGp'])} on {r0['offering']}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.rec.by:account", chart_why="One bar per idea, by peer value.")


@register("Which play should we launch first?", "Which play should be launched first?")
def _play_first(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.themes", sort="accounts missing the same offering")
    th = XS.themes(fs, principal)
    if not th:
        return _empty(plan, "offerings missing at two or more accounts", principal,
                      "No repeating play in scope")
    rows = [{"key": t["offering"], "value": float(t["accounts"]), "opps": int(t["accounts"]),
             "gp": t["estimatedGp"]} for t in th]
    t0 = th[0]
    chart = _bar("ask_play_first", "Growth plays, by how many accounts are missing the offering",
                 rows, says=["xsell.theme.by:offering"], label="Accounts", fmt="number",
                 subtitle="Height is accounts, not money — the count is a fact, the money an estimate",
                 footnote=XS.CAVEAT)
    sentences = [
        _s(f"Launch {t0['offering']} first: {count(t0['accounts'])} accounts are missing it, "
           f"spread across {count(t0['ownerCount'])} owners.", "answer",
           "xsell.theme.by:offering", [t0["offering"], count(t0["accounts"])]),
        _s(f"At peer rates it is worth about {money(t0['estimatedGp'])}"
           + (f"; the next largest play is {th[1]['offering']} at {count(th[1]['accounts'])} "
              f"accounts." if len(th) > 1 else "."), "norm", None, [money(t0["estimatedGp"])]),
        _s(f"{count(t0['corroborated'])} of its accounts were flagged by more than one "
           f"method independently, and {t0['topIndustry']} is the industry it repeats in "
           f"most.", "cause"),
        _s(f"Pilot it with {t0['strongest']['accountName']} ({t0['strongest']['owner']}): "
           f"{t0['strongest']['reason'].rstrip('.')}. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"Launch {t0['offering']} — {count(t0['accounts'])} accounts",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.theme.by:offering", chart_why="One bar per play, by account count.")


@register("Who owns the accounts in the biggest play?", "Who owns the biggest play?")
def _play_owners(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.unified", filter="offering = the largest play", breakdown=["rep"])
    th = XS.themes(fs, principal)
    if not th:
        return _empty(plan, "offerings missing at two or more accounts", principal,
                      "No repeating play in scope")
    t0 = th[0]
    recs = [r for r in XS.unified(fs, principal, limit=10_000) if r["offering"] == t0["offering"]]
    g: dict[str, dict] = {}
    for r in recs:
        e = g.setdefault(r["owner"], {"n": 0, "gp": 0.0})
        e["n"] += 1
        e["gp"] += r["peerGp"]
    rows = sorted(({"key": k, "value": float(e["n"]), "opps": e["n"], "gp": e["gp"]}
                   for k, e in g.items()), key=lambda r: (-r["value"], r["key"]))
    top = rows[0]
    chart = _bar("ask_play_owners", f"Who owns the accounts missing {t0['offering']}", rows,
                 says=["xsell.rec.by:account"], label="Accounts", fmt="number", click_dim="rep",
                 subtitle=f"{count(t0['accounts'])} accounts across {count(t0['ownerCount'])} owners",
                 footnote=XS.CAVEAT)
    sentences = [
        _s(f"{count(t0['ownerCount'])} owners between them: {top['key']} has the most with "
           f"{count(top['opps'])}, and most of the rest hold one each.", "answer",
           "xsell.rec.by:account", [top["key"], count(top["opps"])]),
        _s(f"The play covers {count(t0['accounts'])} accounts, so no single owner can run "
           f"it alone — it is a briefing and a target list.", "norm"),
        _s("Spread this wide, the gap is a coverage pattern rather than one rep's blind "
           "spot.", "cause"),
        _s(f"Brief all {count(t0['ownerCount'])} at once and ask {top['key']} to go first. "
           f"{XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"{count(t0['ownerCount'])} owners share the {t0['offering']} play",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.rec.by:account", chart_why="One bar per owner in the biggest play.")


@register("How much is the biggest play worth?", "What is the biggest play worth?")
def _play_worth(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.themes", metric="estimated GP", breakdown=["offering"])
    th = XS.themes(fs, principal)
    if not th:
        return _empty(plan, "offerings missing at two or more accounts", principal,
                      "No repeating play in scope")
    rows = [{"key": t["offering"], "value": float(t["estimatedGp"]), "opps": int(t["accounts"])}
            for t in th]
    t0 = th[0]
    richest = max(th, key=lambda t: t["estimatedGp"])
    chart = _bar("ask_play_worth", "What each play is worth at peer rates", rows,
                 says=["xsell.theme.by:offering"], label="Estimated GP",
                 subtitle="Sum of peer gross profit across the accounts in each play",
                 footnote=XS.CAVEAT)
    sentences = [
        _s(f"About {money(t0['estimatedGp'])} of gross profit a year for {t0['offering']}, "
           f"across its {count(t0['accounts'])} accounts.", "answer",
           "xsell.theme.by:offering", [money(t0["estimatedGp"])]),
        _s((f"That is also the richest play." if richest is t0 else
            f"By money the richest play is actually {richest['offering']} at "
            f"{money(richest['estimatedGp'])} across {count(richest['accounts'])} accounts."),
           "norm"),
        _s("The figure is what peer accounts already earn on the offering, added up — a "
           "ceiling for the play, not a forecast.", "cause"),
        _s(f"Size the campaign to a third of it and treat the rest as upside. {XS.CAVEAT}",
           "action"),
    ]
    return _envelope(plan=plan, headline=f"{t0['offering']} is worth about {money(t0['estimatedGp'])}",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.theme.by:offering", chart_why="One bar per play, by estimated value.")


@register("Which industry has the most room to grow?", "Which industry has most room to grow?")
def _industry_room(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.unified", breakdown=["industry"], metric="recommendations")
    recs = [r for r in XS.unified(fs, principal, limit=10_000) if r["industry"]]
    if not recs:
        return _empty(plan, "recommendations with an industry", principal)
    g: dict[str, dict] = {}
    for r in recs:
        e = g.setdefault(r["industry"], {"n": 0, "accounts": set(), "gp": 0.0})
        e["n"] += 1
        e["accounts"].add(r["accountCode"])
        e["gp"] += r["peerGp"]
    rows = sorted(({"key": k, "value": float(e["n"]), "opps": len(e["accounts"]), "gp": e["gp"]}
                   for k, e in g.items()), key=lambda r: -r["value"])
    top = rows[0]
    chart = _bar("ask_industry_room", "Growth ideas by industry", rows,
                 says=["xsell.rec.by:industry"], label="Ideas", fmt="number", click_dim="industry",
                 subtitle="Number of recommendations, by the account's industry", footnote=XS.CAVEAT)
    sentences = [
        _s(f"{top['key']}: {count(top['opps'])} accounts with {count(int(top['value']))} "
           f"growth ideas between them, worth about {money(top['gp'])} at peer rates.",
           "answer", "xsell.rec.by:industry", [top["key"], count(int(top["value"]))]),
        _s(f"{count(len(rows))} industries carry at least one idea; {top['key']} holds "
           f"{pct(100 * top['value'] / len(recs))} of all of them.", "norm"),
        _s("An industry with many ideas is one where our reference customers already "
           "buy the missing offering, so the sale is easier to evidence.", "cause"),
        _s(f"Run the {top['key']} accounts as one campaign with shared references. "
           f"{XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} has the most room to grow",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.rec.by:industry", chart_why="One bar per industry.")


@register("What is the strongest industry and offering pair?",
          "What is the strongest industry offering pair?")
def _strongest_pair(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("charts.cross_sell_matrix", metric="accounts x mean confidence")
    spec = C.cross_sell_matrix(fs, principal)
    cells = list(spec["data"]["cells"])
    if not cells:
        return _empty(plan, "recommendations with an industry", principal)
    top = max(cells, key=lambda c: c["value"])
    chart = C.spec("ask_strongest_pair", "Where growth repeats — industry against offering",
                   "categorical×categorical×measure", spec["data"],
                   says=["xsell.rec.by:industry"], subtitle=spec["subtitle"],
                   measure_label="Strength", fmt="number", footnote=XS.CAVEAT)
    sentences = [
        _s(f"{top['row']} with {top['col']}: {count(top['accounts'])} accounts missing the "
           f"same thing, which is the strongest pairing in scope.", "answer",
           "xsell.rec.by:industry", [f"{top['row']} with {top['col']}"]),
        _s(f"Strength is accounts multiplied by average confidence, so {count(len(cells))} "
           f"pairs are ranked by how many and how sure together.", "norm"),
        _s("One very confident idea does not outrank five credible ones, and five weak "
           "ones do not outrank two strong — that is the point of the score.", "cause"),
        _s(f"Build the {top['col']} pitch for {top['row']} once and run it at all "
           f"{count(top['accounts'])}. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['row']} × {top['col']} is the strongest pair",
                     sentences=sentences, chart=chart, rows=cells,
                     shape="categorical×categorical×measure", claim="xsell.rec.by:industry",
                     chart_why="The industry-by-offering grid, shaded by strength.")


@register("Which offering is missing across the most industries?",
          "Which offering is missing in the most industries?")
def _offering_industries(fs: FilterState, principal: Principal, say: list[str]) -> dict:
    plan = _derived_plan("crosssell.unified", breakdown=["offering"], metric="distinct industries")
    recs = [r for r in XS.unified(fs, principal, limit=10_000) if r["industry"]]
    if not recs:
        return _empty(plan, "recommendations with an industry", principal)
    g: dict[str, dict] = {}
    for r in recs:
        e = g.setdefault(r["offering"], {"ind": set(), "n": 0})
        e["ind"].add(r["industry"])
        e["n"] += 1
    rows = sorted(({"key": k, "value": float(len(e["ind"])), "opps": e["n"]} for k, e in g.items()),
                  key=lambda r: (-r["value"], -r["opps"]))
    top = rows[0]
    chart = _bar("ask_offering_industries", "How many industries each offering is missing in", rows,
                 says=["xsell.rec.by:industry"], label="Industries", fmt="number",
                 subtitle="Distinct industries with at least one account missing the offering",
                 footnote=XS.CAVEAT)
    sentences = [
        _s(f"{top['key']} is missing across {count(int(top['value']))} industries — "
           f"{count(top['opps'])} accounts in all.", "answer", "xsell.rec.by:industry",
           [top["key"], count(int(top["value"]))]),
        _s(f"{count(len(rows))} offerings are recommended somewhere"
           + (f"; the next widest is {rows[1]['key']} across {count(int(rows[1]['value']))}."
              if len(rows) > 1 else "."), "norm"),
        _s("An offering missing everywhere is a coverage gap in how we sell, not a "
           "quirk of one sector.", "cause"),
        _s(f"Make {top['key']} a standing attach in every proposal rather than an "
           f"industry play. {XS.CAVEAT}", "action"),
    ]
    return _envelope(plan=plan, headline=f"{top['key']} is missing across {count(int(top['value']))} industries",
                     sentences=sentences, chart=chart, rows=rows, shape="categorical×measure",
                     claim="xsell.rec.by:industry", chart_why="One bar per offering.")
