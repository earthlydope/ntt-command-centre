"""
A question about ONE chart, answered from that chart's own rows.

The per-chart Ask is a blank box beside a chart. Whatever is typed into it is
about the marks the person is looking at, so the answer is worked out from the
same payload the chart was drawn from — rebuilt server-side from the chart id,
never trusted from the client — and comes back as words only. The chart is
already on screen; drawing it again beside the sentence about it would be the
redundancy this product exists to avoid.

Two paths, one shape of answer:

  * a language model, when one will answer, narrates the question against the
    chart's rows with every figure pre-formatted by the server, and the same
    grounding checks the page brief goes through reject any number the server
    did not hand over;
  * otherwise the question is read for its shape — the biggest, the smallest,
    the total, how many, what share, a named mark, one mark against another —
    and answered by computed prose over the rows. That is the path a demo
    exercises whenever the free tiers are exhausted, so it has to read as an
    answer and not as a fallback.

Every payload in the chart contract (docs/CHART_CONTRACT.md) is flattened to
the same `(label, value)` list first, so one narrator serves fourteen chart
types. Structured payloads keep their secondary figures where the question is
likely to be about them: a gantt bar keeps its quiet days, a funnel stage its
cohort percentage, a mekko segment its margin.
"""

from __future__ import annotations

import re

from ..semantic import measures as M
from ..semantic import views as V
from ..semantic.measures import FilterState
from ..semantic.personas import Principal
from . import grounding as G
from . import prompts, providers

# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def flatten(spec: dict) -> tuple[list[dict], str]:
    """
    Every chart payload as `[{label, value, extra{...}}]`, plus the noun the
    values are counted in ("gross profit", "deals", ...).

    `extra` carries the figures a person might reasonably ask about beyond the
    main value, under plain names, so the narrator can answer "which of these
    has gone quiet" from a gantt or "where is the margin weakest" from a mekko
    without a second query.
    """
    data = spec.get("data")
    key = spec.get("repositoryKey") or ""
    fmt = spec.get("format", "currency")
    noun = {
        "currency": (spec.get("measureLabel") or "value").replace("ACV ", "").replace("GP", "gross profit").lower(),
        "percent": "percent",
        "number": "count",
        "days": "days",
    }.get(fmt, "value")
    out: list[dict] = []

    def add(label, value, **extra):
        v = _num(value)
        if label is None or v is None:
            return
        out.append({"label": str(label), "value": v,
                    "extra": {k: x for k, x in extra.items() if x is not None and x == x}})

    if isinstance(data, list):
        for r in data:
            if not isinstance(r, dict):
                continue
            if key == "funnel.stage" or "cohortPct" in r:
                add(r.get("label") or r.get("key"), r.get("count"),
                    value_at_stage=r.get("value"), share_of_entry_cohort=r.get("cohortPct"),
                    share_of_previous_stage=r.get("stepPct"), lost_here=r.get("lostHere"))
                noun = "deals"
            elif key == "bullet.target" or "target" in r:
                add(r.get("label") or r.get("key"), r.get("actual"), target=r.get("target"))
            elif key == "waterfall.bridge" or "type" in r:
                add(r.get("label") or r.get("key"), r.get("value"), step=r.get("type"),
                    note=r.get("note"))
            else:
                add(r.get("label") or r.get("key"), r.get("value"), share_pct=r.get("share"),
                    deals=r.get("opps"), lines=r.get("lines"), margin_pct=r.get("gm"))
        return out, noun

    if not isinstance(data, dict):
        return out, noun

    if "cells" in data:
        for c in data.get("cells") or []:
            add(f"{c.get('row')} / {c.get('col')}", c.get("value"),
                secondary=c.get("secondary"))
    elif "bars" in data:
        for b in data.get("bars") or []:
            add(b.get("label"), b.get("value"), account=b.get("sublabel"),
                close_date=b.get("end"), past_due=b.get("pastDue"),
                stalled=b.get("stalled"), quiet_days=b.get("quietDays"))
    elif "points" in data and "xLabel" in data:
        for p in data.get("points") or []:
            add(p.get("label"), p.get("size"), **{
                str(data.get("xLabel") or "x").lower(): p.get("x"),
                str(data.get("yLabel") or "y").lower(): p.get("y"),
                "group": p.get("category")})
        noun = str(data.get("sizeLabel") or noun).lower()
    elif "points" in data:
        for p in data.get("points") or []:
            add(p.get("label") or p.get("key"), p.get("bar"),
                **{str(data.get("lineLabel") or "line").lower(): p.get("line")})
        noun = str(data.get("barLabel") or noun).lower()
    elif "columns" in data and "fillLabel" in data:
        for col in data.get("columns") or []:
            add(col.get("key"), col.get("total"))
            for seg in col.get("segments") or []:
                add(f"{col.get('key')} / {seg.get('key')}", seg.get("value"),
                    **{str(data.get("fillLabel") or "fill").lower(): seg.get("fill")})
    elif "columns" in data and "rows" in data:
        cols = [c for c in data.get("columns") or [] if isinstance(c, dict)]
        label_col = next((c["key"] for c in cols if c.get("format") is None), cols[0]["key"] if cols else "key")
        num_cols = [c for c in cols if c.get("format")]
        first_num = num_cols[0]["key"] if num_cols else None
        for r in data.get("rows") or []:
            extra = {str(c.get("label") or c["key"]).lower(): r.get(c["key"])
                     for c in num_cols[1:]}
            add(r.get(label_col), r.get(first_num) if first_num else None, **extra)
        if num_cols:
            noun = str(num_cols[0].get("label") or noun).lower()
    elif "nodes" in data and "links" in data:
        for l in data.get("links") or []:
            add(f"{l.get('source')} → {l.get('target')}", l.get("value"), note=l.get("note"))
    elif "nodes" in data:
        for n in data.get("nodes") or []:
            if n.get("parent") is not None:
                add(n.get("label"), n.get("value"), secondary=n.get("secondary"))
    elif "categories" in data and "series" in data:
        cats = data.get("categories") or []
        for s in data.get("series") or []:
            for i, v in enumerate(s.get("values") or []):
                if i < len(cats):
                    add(f"{cats[i]} / {s.get('name')}", v)
    return out, noun


def find_chart(chart_id: str, page: str, fs: FilterState, principal: Principal) -> dict | None:
    """Rebuild the page's charts and pick the one asked about."""
    page = V.resolve_page(page or principal.persona.home, principal)
    for c in V.charts_for(page, fs, principal):
        if c.get("id") == chart_id:
            return c
    return None


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _fmt_for(spec: dict):
    f = spec.get("format", "currency")
    if f == "currency":
        return M.money
    if f == "percent":
        return lambda v: M.pct(v)
    if f == "days":
        return lambda v: f"{int(round(v))} days"
    return lambda v: M.count(v)


def _fmt_extra(k: str, v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    n = _num(v)
    if n is None:
        return str(v)
    if k.endswith("_pct") or "share" in k or "margin" in k or "rate" in k:
        return M.pct(n)
    if "days" in k:
        return f"{int(round(n))} days"
    if k in ("deals", "lines", "lost_here", "count"):
        return M.count(n)
    return M.money(n) if abs(n) >= 1000 else f"{n:,.1f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- #
# The computed narrator
# --------------------------------------------------------------------------- #

_WORDS_TOP = re.compile(r"\b(biggest|largest|highest|most|top|best|strongest|greatest|max(imum)?|leads?|first)\b")
_WORDS_LOW = re.compile(r"\b(smallest|lowest|least|fewest|weakest|worst|bottom|min(imum)?|last)\b")
_WORDS_TOTAL = re.compile(r"\b(total|altogether|in all|sum|overall|combined|how much)\b")
_WORDS_COUNT = re.compile(r"\b(how many|number of|count)\b")
_WORDS_SHARE = re.compile(r"\b(share|percent(age)?|proportion|fraction|%)\b")
_WORDS_AVG = re.compile(r"\b(average|mean|typical|median)\b")
_WORDS_COMPARE = re.compile(r"\b(vs\.?|versus|compared? (to|with)|against|difference between)\b")
_WORDS_QUIET = re.compile(r"\b(quiet|stalled|silent|stopped|stuck)\b")
_WORDS_OVERDUE = re.compile(r"\b(overdue|past due|late|slipped)\b")


def _s(text: str, lens: str, bold: list[str] | None = None) -> dict:
    return {"text": text, "lens": lens, "claim": None, "bold": bold or []}


def _head(label: str) -> str:
    """The part of a label a person actually types: 'Concentration & Cross-Sell'
    is asked about as 'Concentration', 'Cobalt Holdings Co.' as 'Cobalt'."""
    return re.split(r"\s*[&/(·—-]\s*|\s+(?:Co|Inc|Ltd|LLC|Corp)\.?$", label, maxsplit=1)[0].strip().lower()


def _named(rows: list[dict], q: str) -> list[dict]:
    """Rows the question names, in the order it names them, whole label first
    and then the label's head word, so 'Networking / Product' beats 'Networking'
    and 'Concentration vs Rep Behavior' finds both."""
    ql = q.lower()
    hits: list[tuple[int, dict]] = []
    for r in rows:
        if r["label"].startswith("Other ("):
            continue
        full = r["label"].lower()
        head = _head(r["label"])
        pos = ql.find(full) if len(full) >= 3 else -1
        if pos < 0 and len(head) >= 4:
            m = re.search(r"\b" + re.escape(head) + r"\b", ql)
            pos = m.start() if m else -1
        if pos >= 0:
            hits.append((pos, r))
    # Order by where they appear in the question, so "A vs B" keeps A first.
    hits.sort(key=lambda t: t[0])
    seen: set[str] = set()
    out = []
    for _, r in hits:
        if r["label"] not in seen:
            seen.add(r["label"])
            out.append(r)
    return out


def computed(question: str, spec: dict, rows: list[dict], noun: str) -> dict:
    fmt = _fmt_for(spec)
    q = question.lower()
    title = spec.get("title", "this chart")
    named = [r for r in rows if not r["label"].startswith("Other (")] or rows
    total = sum(r["value"] for r in rows)
    n = len(named)
    top = max(named, key=lambda r: r["value"]) if named else None
    low = min(named, key=lambda r: r["value"]) if named else None
    sentences: list[dict] = []
    headline = title

    def share(r: dict) -> str:
        return M.pct(100 * r["value"] / total) if total else "—"

    def extras(r: dict, limit: int = 3) -> str:
        bits = []
        for k, v in list(r["extra"].items())[:limit]:
            word = k.replace("_", " ")
            shown = _fmt_extra(k, v)
            # "10 deals" reads; "deals 10" does not.
            bits.append(f"{shown} {word}" if k in ("deals", "lines", "lost_here") else f"{word} {shown}")
        return f" — {', '.join(bits)}" if bits else ""

    # What one mark on this chart is called, for "how many": the noun after
    # "by" in the title ("Findings by category" -> categories), else "items".
    m_by = re.search(r"\bby\s+([a-z][a-z ]+?)\s*$", title.lower())
    item = (m_by.group(1).strip() if m_by else "item")
    items = item if item.endswith("s") else (item[:-1] + "ies" if item.endswith("y") else item + "s")

    hits = _named(rows, question)

    if _WORDS_COMPARE.search(q) and len(hits) >= 2:
        a, b = hits[0], hits[1]
        diff = a["value"] - b["value"]
        bigger, smaller = (a, b) if diff >= 0 else (b, a)
        headline = f"{bigger['label']} is ahead of {smaller['label']} by {fmt(abs(diff))}"
        sentences.append(_s(
            f"{a['label']} stands at {fmt(a['value'])} and {b['label']} at {fmt(b['value'])} "
            f"on {title.lower()}.", "answer", [fmt(a['value']), fmt(b['value'])]))
        if total:
            sentences.append(_s(
                f"That is {share(a)} against {share(b)} of the {fmt(total)} in view.", "norm"))
        sentences.append(_s(
            f"The gap is {fmt(abs(diff))}; start with {smaller['label']} if the question is "
            f"where the room is, and with {bigger['label']} if it is where the weight is.", "action"))

    elif hits and not (_WORDS_TOP.search(q) or _WORDS_LOW.search(q)):
        r = hits[0]
        rank = sorted(named, key=lambda x: -x["value"]).index(r) + 1 if r in named else None
        headline = f"{r['label']}: {fmt(r['value'])}"
        sentences.append(_s(
            f"{r['label']} shows {fmt(r['value'])} of {noun} on {title.lower()}{extras(r)}.",
            "answer", [r['label'], fmt(r['value'])]))
        if rank and n > 1:
            sentences.append(_s(
                f"That ranks it {_ordinal(rank)} of {n}" + (f", at {share(r)} of the total" if total else "")
                + (f"; the largest is {top['label']} at {fmt(top['value'])}." if top and top is not r else "."),
                "norm"))
        sentences.append(_s(_action(spec, r, top), "action"))

    elif _WORDS_QUIET.search(q) and any("quiet_days" in r["extra"] or "stalled" in r["extra"] for r in rows):
        quiet = [r for r in rows if r["extra"].get("stalled") or _num(r["extra"].get("quiet_days")) and _num(r["extra"].get("quiet_days")) >= 60]
        quiet.sort(key=lambda r: -(_num(r["extra"].get("quiet_days")) or 0))
        if quiet:
            headline = f"{len(quiet)} of {len(rows)} have gone quiet"
            lead = quiet[0]
            sentences.append(_s(
                f"{len(quiet)} of the {len(rows)} deals on {title.lower()} have not moved in "
                f"60 days or more, worth {fmt(sum(r['value'] for r in quiet))} between them.",
                "answer", [str(len(quiet))]))
            sentences.append(_s(
                f"The quietest is {lead['label']} at {_fmt_extra('quiet_days', lead['extra'].get('quiet_days'))} "
                f"of silence, worth {fmt(lead['value'])}.", "cause", [lead['label']]))
            sentences.append(_s(
                "Silence is the strongest single signal of a deal that will slip; call the "
                "quietest large one before the one that is merely late.", "action"))
        else:
            headline = "Nothing here has gone quiet"
            sentences.append(_s(f"None of the deals on {title.lower()} has been silent for 60 days.", "answer"))

    elif _WORDS_OVERDUE.search(q) and any("past_due" in r["extra"] for r in rows):
        late = [r for r in rows if r["extra"].get("past_due")]
        headline = f"{len(late)} of {len(rows)} are past their close date"
        sentences.append(_s(
            f"{len(late)} of the {len(rows)} deals on {title.lower()} are past the close date "
            f"they carry, worth {fmt(sum(r['value'] for r in late))}.", "answer", [str(len(late))]))
        if late:
            big = max(late, key=lambda r: r["value"])
            sentences.append(_s(f"The largest overdue one is {big['label']} at {fmt(big['value'])}.", "norm", [big['label']]))
        sentences.append(_s("A close date that has passed is a forecast that is already wrong; re-date or re-qualify each one.", "action"))

    elif _WORDS_COUNT.search(q):
        headline = f"{n} {items} on this chart"
        sentences.append(_s(
            f"{title} shows {n} {items}"
            + (f", worth {fmt(total)} of {noun} together." if total and noun not in ("count", "percent", "days") else "."),
            "answer", [str(n)]))
        if top:
            sentences.append(_s(f"{top['label']} is the largest at {fmt(top['value'])} ({share(top)} of the total).", "norm", [top['label']]))
        sentences.append(_s(_action(spec, top, top), "action"))

    elif _WORDS_AVG.search(q) and named:
        avg = total / len(named)
        above = [r for r in named if r["value"] > avg]
        headline = f"The average is {fmt(avg)}"
        sentences.append(_s(
            f"Across the {n} items on {title.lower()} the average is {fmt(avg)}; {len(above)} sit above it.",
            "answer", [fmt(avg)]))
        if top and low:
            sentences.append(_s(f"The spread runs from {low['label']} at {fmt(low['value'])} to {top['label']} at {fmt(top['value'])}.", "norm"))
        sentences.append(_s("Judge each item against the average, not the leader — the leader is usually the outlier.", "action"))

    elif _WORDS_LOW.search(q) and low:
        headline = f"{low['label']} is the smallest at {fmt(low['value'])}"
        sentences.append(_s(
            f"{low['label']} is the smallest on {title.lower()} at {fmt(low['value'])}{extras(low)}.",
            "answer", [low['label'], fmt(low['value'])]))
        if top and total:
            sentences.append(_s(
                f"It is {share(low)} of the {fmt(total)} in view; the largest, {top['label']}, "
                f"is {fmt(top['value'])}.", "norm"))
        sentences.append(_s(_action(spec, low, top), "action"))

    elif (_WORDS_TOTAL.search(q) or _WORDS_SHARE.search(q)) and not _WORDS_TOP.search(q) and total:
        headline = f"{fmt(total)} in total"
        sentences.append(_s(
            f"The {n} items on {title.lower()} add up to {fmt(total)} of {noun}.", "answer", [fmt(total)]))
        if top:
            sentences.append(_s(
                f"{top['label']} alone is {share(top)} of that, at {fmt(top['value'])}"
                + (f"; the top three together are {M.pct(100 * sum(r['value'] for r in sorted(named, key=lambda x: -x['value'])[:3]) / total)}." if n > 3 else "."),
                "norm", [top['label']]))
        sentences.append(_s(_action(spec, top, top), "action"))

    elif top:
        headline = f"{top['label']} is the largest at {fmt(top['value'])}"
        sentences.append(_s(
            f"{top['label']} is the biggest on {title.lower()} at {fmt(top['value'])}{extras(top)}.",
            "answer", [top['label'], fmt(top['value'])]))
        second = sorted(named, key=lambda r: -r["value"])[1] if n > 1 else None
        if total and second:
            sentences.append(_s(
                f"That is {share(top)} of the {fmt(total)} in view; next is {second['label']} at "
                f"{fmt(second['value'])}.", "norm", [share(top)]))
        sentences.append(_s(_action(spec, top, top), "action"))

    if spec.get("footnote"):
        sentences.append(_s(str(spec["footnote"]), "state"))

    return {"headline": headline, "sentences": sentences}


def _ordinal(i: int) -> str:
    return f"{i}{'th' if 11 <= i % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')}"


def _action(spec: dict, r: dict | None, top: dict | None) -> str:
    """One plain next step, keyed on what the chart is about."""
    t = (spec.get("title") or "").lower()
    who = r["label"] if r else "the leader"
    if "risk" in t or "quiet" in t or "stopped" in t:
        return f"Open {who} and read the factors that put it there before deciding anything."
    if "margin" in t or "mix" in t:
        return f"Ask why {who} carries the margin it does — price, mix or cost — before treating it as a target."
    if "plan" in t or "coverage" in t or "target" in t:
        return f"Compare {who} with what is still open against it; a gap with no pipeline is the one to fund."
    if "finding" in t or "anomal" in t or "problem" in t:
        return f"Sort by what it is worth, not how many there are; one finding on a large account outweighs a dozen small ones."
    if "rep" in t or "team" in t or "coach" in t:
        return f"Look at {who}'s pattern over several deals, not one — a single deal is noise, a repeat is a finding."
    if "customer" in t or "account" in t or "sell" in t:
        return f"Start with {who}: it carries the most weight in this view, and one conversation there moves the total."
    if top is not None and r is not None and r is not top:
        return f"Set {who} against {top['label']} — the gap between them is the question worth asking next."
    return f"Start with {who}: it is where this chart says the weight is."


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def answer(question: str, chart_id: str, page: str, fs: FilterState,
           principal: Principal, charts_say: list[str]) -> dict | None:
    """
    The ask envelope for a question about one chart, or None when the chart
    cannot be rebuilt for this caller (a stale id, or a chart that belongs to
    a page this persona cannot reach — in which case the ordinary Ask path is
    the right answer rather than a refusal).
    """
    spec = find_chart(chart_id, page, fs, principal)
    if spec is None:
        return None
    rows, noun = flatten(spec)
    if not rows:
        return {"question": question, "chart": None, "chartId": chart_id, "refused": True,
                "degraded": False, "provider": "computed",
                "answer": {"headline": "This chart has nothing to ask about",
                           "sentences": [_s(spec.get("emptyMessage") or
                                            "There are no rows behind this chart in the current slice.", "answer")]}}

    fmt = _fmt_for(spec)
    say = list(dict.fromkeys(list(charts_say) + list(spec.get("says") or [])))

    # The model gets the rows as pre-formatted strings under plain labels and
    # nothing else, so it cannot do arithmetic; anything it says with a number
    # the server did not supply is rejected by the same verifier the brief uses.
    pack = G.pack_figures({
        **{f"{r['label']}": fmt(r["value"]) for r in rows[:40]},
        **{f"{r['label']} {k.replace('_', ' ')}": _fmt_extra(k, v)
           for r in rows[:40] for k, v in r["extra"].items()},
        "total": fmt(sum(r["value"] for r in rows)),
        "items": str(len(rows)),
    })
    task = (
        f"The user is looking at the chart \"{spec.get('title')}\" ({spec.get('subtitle') or ''}) "
        f"and asked: \"{question}\". Answer in two to four short sentences from the figures "
        f"given, in plain words, and end with one thing to do. Do not describe what the "
        f"chart shows; say what it means."
        + (f" The chart carries this caveat, which you must respect: {spec['footnote']}" if spec.get("footnote") else "")
    )
    user = G.render(pack, entities=[r["label"] for r in rows[:40]],
                    evidence=[{"id": f"r{i}", "label": r["label"], "value": fmt(r["value"]),
                               **{k: _fmt_extra(k, v) for k, v in r["extra"].items()}}
                              for i, r in enumerate(rows[:40])],
                    charts_say=say, untrusted=[question], task=task)
    provider = "computed"
    try:
        res = providers.complete(prompts.ASK_NARRATOR_V1, user, _SCHEMA,
                                 max_tokens=500, deadline_s=12)
        clean = G.sanitise(res.obj.get("sentences", []), pack, say)
        if clean["sentences"]:
            ans = {"headline": res.obj.get("headline") or spec.get("title"),
                   "sentences": clean["sentences"],
                   "rejected": clean["rejected"], "dropped": clean["dropped"]}
            provider = res.provider
        else:
            ans = computed(question, spec, rows, noun)
    except providers.LLMUnavailable:
        ans = computed(question, spec, rows, noun)

    return {
        "question": question, "chartId": chart_id, "chart": None,
        "answer": ans, "provider": provider, "degraded": False,
        "deterministic": provider == "computed", "refused": False, "cached": False,
        "plan": {"computed": "narration over the chart's own rows", "chart": chart_id,
                 "rows": len(rows)},
    }


_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "lens": {"type": "string", "enum": list(G.LENSES)},
                    "claim": {"type": "string"},
                },
                "required": ["text", "lens"],
            },
        },
    },
    "required": ["headline", "sentences"],
}
