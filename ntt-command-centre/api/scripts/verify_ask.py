"""
Every question the product suggests must answer with no language model at all.

`python -m api.scripts.verify_ask`

The Ask panel offers a handful of questions per persona and every chart offers
two or three of its own. On a free tier the providers are rate-limited for
most of any demo, so this harness forces them unavailable and asks every one
of those questions, for every persona, on every page — then checks that what
comes back is a finished answer: not refused, a headline, at least two plain
sentences, a chart wherever the answer is a breakdown or a trend, no raw
float anywhere in the prose, and a payload that serialises. It also runs a
few free-text questions through the keyword planner so that path is covered.

Exit code is the number of failures, so CI can gate on it.
"""

from __future__ import annotations

import os

# The harness pins the business date BEFORE the semantic layer is imported:
# every figure it asserts was taken on 2026-09-15, and the product otherwise
# follows the calendar (see config.AS_OF).
os.environ.setdefault("NTT_AS_OF", "2026-09-15")

import json
import re
import sys
import time

from ..llm import providers
from ..llm import service
from ..semantic import answers as A
from ..semantic import personas as PR
from ..semantic import views as V
from ..semantic.measures import FilterState

#: The persona chips, exactly as the client offers them.
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

#: Free text the keyword planner must handle, and one it must refuse politely.
FREE_TEXT: list[tuple[str, bool]] = [
    ("open pipeline by rep", True),
    ("How much revenue have we won by month?", True),
    ("Win rate by industry", True),
    ("Which portfolio has the best margin?", True),
    ("how many deals are overdue by stage", True),
    ("what is the meaning of life", False),
]

#: A number with three or more decimals is a raw float that escaped formatting.
_RAW_FLOAT = re.compile(r"\d\.\d{3,}")
#: Shapes for which the answer must carry a chart.
_CHARTED = {"categorical×measure", "temporal×measure",
            "categorical×categorical×measure", "table"}


def _unavailable(*_args, **_kwargs):
    raise providers.LLMUnavailable("forced unavailable by verify_ask")


def _problems(out: dict, expect_refusal: bool = False) -> list[str]:
    bad: list[str] = []
    if expect_refusal:
        if not out.get("refused"):
            bad.append("expected a refusal")
        if len(out.get("suggestions") or []) < 2:
            bad.append("refusal carries fewer than two suggestions")
        return bad
    if out.get("refused"):
        bad.append("refused: " + " ".join(s["text"] for s in out["answer"]["sentences"])[:80])
        return bad
    if out.get("error"):
        bad.append(f"error: {out['error']}")
        return bad
    ans = out.get("answer") or {}
    if not str(ans.get("headline") or "").strip():
        bad.append("empty headline")
    sentences = ans.get("sentences") or []
    if len(sentences) < 2:
        bad.append(f"only {len(sentences)} sentence(s)")
    for s in sentences:
        if _RAW_FLOAT.search(str(s.get("text", ""))):
            bad.append("raw float in: " + str(s.get("text"))[:70])
            break
        if not str(s.get("text", "")).strip():
            bad.append("empty sentence")
            break
    if out.get("provider") != "computed":
        bad.append(f"provider is {out.get('provider')!r}, not 'computed'")
    if out.get("shape") in _CHARTED and not out.get("chart"):
        bad.append(f"no chart for a {out.get('shape')} answer")
    if out.get("chart") and not out["chart"].get("says"):
        bad.append("chart declares no claim")
    if "cannot answer right now" in str(ans.get("headline", "")).lower():
        bad.append("the forbidden headline")
    try:
        json.dumps(out, allow_nan=False)
    except (TypeError, ValueError) as e:
        bad.append(f"not JSON: {e}")
    return bad


def main() -> int:
    providers.complete = _unavailable  # type: ignore[assignment]
    service.providers.complete = _unavailable  # type: ignore[attr-defined]
    fs = FilterState()
    failures: list[str] = []
    asked: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, str, str, float]] = []

    def run(persona: str, page: str, q: str, expect_refusal: bool = False) -> None:
        if (persona, q) in asked:
            return
        asked.add((persona, q))
        principal = PR.resolve(persona)
        t0 = time.time()
        try:
            out = service.ask(q, fs, principal, [])
            problems = _problems(out, expect_refusal)
        except Exception as e:  # noqa: BLE001 — the harness reports, it does not crash
            problems = [f"{type(e).__name__}: {e}"]
            out = {}
        ms = (time.time() - t0) * 1000
        chart = (out.get("chart") or {}).get("repositoryKey", "-") if out else "-"
        rows.append((persona, page, q, "; ".join(problems) if problems else chart, ms))
        if problems:
            failures.append(f"[{persona}/{page}] {q} — {'; '.join(problems)}")

    for persona, qs in SUGGESTIONS.items():
        for q in qs:
            run(persona, "chips", q)

    for page, (persona, _label, _question) in V.PAGES.items():
        principal = PR.resolve(persona)
        for chart in V.charts_for(page, fs, principal):
            for q in chart.get("questions") or []:
                run(persona, page, q)

    for persona in SUGGESTIONS:
        for q, ok in FREE_TEXT:
            run(persona, "free-text", q, expect_refusal=not ok)

    width = max(len(r[2]) for r in rows)
    print(f"\n{'persona':10s} {'page':15s} {'question':{width}s}  result")
    print("-" * (width + 60))
    for persona, page, q, result, ms in rows:
        mark = " FAIL " if any(f.startswith(f"[{persona}/{page}] {q} ") for f in failures) else "  ok  "
        print(f"[{mark}] {persona:10s} {page:15s} {q:{width}s}  {result[:70]}  {ms:5.0f}ms")

    print(f"\nregistered questions: {len(A.registered_questions())}")
    print(f"asked: {len(rows)}   failed: {len(failures)}")
    for f in failures:
        print("  " + f)
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
