"""
Chart specs — the server decides WHAT to draw and WHICH module draws it.

Two things make this more than a formatting layer.

**Shape resolution happens here.** The client re-runs the same rule as a check
and prints `shape → key` in every card footer, so the claim that the repository
is indexed by data shape is verifiable on screen rather than asserted in a
README.

**Every spec declares what it `says`.** A claim key like `gp.open.by:lob` is a
canonical statement of "gross profit, open subset, broken down by line of
business". The view payload unions the claim keys of every chart on screen, and
the AI layer is forbidden from making a `state`-lens sentence whose claim is
already in that set. That is how "the AI must not repeat the charts" becomes a
machine-checked property instead of a line in a prompt that a model may or may
not honour.

Chart choice is a judgement about the QUESTION, not about available variety. A
waterfall is here because "what moved the number" cannot be read off a bar
chart; a mekko because size, mix and margin are three facts that belong in one
picture; a gantt because "has this deal run out of road" is a statement about a
bar and a date. Where a type has no honest use on this data it is not built —
there is no radar chart and no pie.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import accounts as ACC
from . import anomalies as ANOM
from . import budget as B
from . import crosssell as XS
from . import predict as P
from .dimensions import REGISTRY
from .loader import AS_OF, STAGE_ORDER, facts, opportunities
from .measures import (
    FilterState,
    by_dimension,
    by_time,
    money,
    slice_frame,
    subset,
)
from .movement_features import STALL_DAYS, features
from .personas import Principal

#: shape -> repository key. The client holds the same table.
SHAPE_TO_KEY: dict[str, str] = {
    "categorical×measure": "bar.categorical",
    "temporal×measure": "line.timeseries",
    "categorical×categorical×measure": "heat.grid",
    "categorical×series×measure": "bar.stacked",
    "target×actual": "bullet.target",
    "bridge": "waterfall.bridge",
    "temporal×measure×measure": "combo.columnline",
    "cohort×stage": "funnel.stage",
    "x×y×size": "bubble.scatter",
    "categorical×measure×width": "mekko.marimekko",
    "hierarchy×measure": "treemap.nested",
    "source×target×measure": "sankey.flow",
    "entity×start×end": "gantt.timeline",
}
FALLBACK_KEY = "table.compact"


def select(shape: str) -> str:
    return SHAPE_TO_KEY.get(shape, FALLBACK_KEY)


def spec(
    chart_id: str,
    title: str,
    shape: str,
    data: Any,
    *,
    says: list[str],
    subtitle: str = "",
    measure_label: str = "ACV GP",
    fmt: str = "currency",
    click_dim: str | None = None,
    count_basis: str | None = None,
    basis_note: str | None = None,
    footnote: str | None = None,
    height: int | None = None,
    key: str | None = None,
    encoding: dict | None = None,
    empty_message: str | None = None,
    questions: list[str] | None = None,
) -> dict:
    """Assemble one spec. `says` is required — a chart with no declared claim
    cannot participate in the non-redundancy check and would silently let the
    AI restate it."""
    if not says:
        raise ValueError(
            f"chart '{chart_id}' declares no claim keys. Every spec must say what it "
            "puts on screen, or the AI layer cannot avoid repeating it."
        )
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "shape": shape,
        "repositoryKey": key or select(shape),
        "data": data,
        # Encoding names which FIELD of a row carries x and y, and is only
        # meaningful for row-shaped payloads. The structured types name their
        # own fields in the payload, and stamping a default on them is actively
        # harmful: `bubble.scatter` reads `encoding.x` in preference to its own
        # "x" default, so a blanket `{"x": "key"}` made it look for a field its
        # points do not have and render an empty state over perfectly good data.
        **(
            {"encoding": encoding}
            if encoding
            else {"encoding": {"x": "key", "y": "value"}}
            if isinstance(data, list)
            else {}
        ),
        "measureLabel": measure_label,
        "format": fmt,
        "clickDim": click_dim,
        "countBasis": count_basis,
        "basisNote": basis_note,
        "says": says,
        "footnote": footnote,
        "height": height,
        # What "no rows" MEANS for this particular chart. A module's generic
        # "No rows in this slice" is true and useless: an account executive
        # whose accounts already buy all four lines of business has no
        # cross-sell gap, which is a finding worth stating rather than a blank
        # card. Absent, the module's own wording stands.
        "emptyMessage": empty_message,
        # Two or three questions THIS chart invites. They seed the per-chart Ask
        # panel, so the first thing a reader sees is a question about the marks
        # in front of them rather than a blank box and a cursor.
        "questions": questions or [],
    }


def _basis(dim: str) -> tuple[str, str]:
    d = REGISTRY.get(dim)
    if not d:
        return "lines", ""
    return d.count_basis, d.basis_note


def _measure_key(fs: FilterState) -> str:
    return "gp" if fs.measure == "gp" else "rev"


# --------------------------------------------------------------------------- #
# The basic set
# --------------------------------------------------------------------------- #


def open_by_dimension(fs: FilterState, principal: Principal, dim: str) -> dict:
    df = subset(slice_frame(fs, principal), "open")
    g = by_dimension(df, dim, fs, top=12)
    basis, note = _basis(dim)
    label = REGISTRY[dim].label
    return spec(
        f"open_by_{dim}", f"Open pipeline by {label.lower()}",
        "categorical×measure",
        [{"key": r.key, "value": float(r.value), "share": float(r.share),
          "lines": int(r.lines), "opps": int(r.opps), "gm": float(r.gm)}
         for r in g.itertuples(index=False)],
        says=[f"{_measure_key(fs)}.open.by:{dim}"],
        subtitle=f"{fs.measure_label} · {len(df):,} lines",
        measure_label=fs.measure_label, click_dim=dim,
        count_basis=basis, basis_note=note,
        questions=[
            f"Which {label.lower()} has the most open pipeline?",
            f"How does the win rate compare across {label.lower()}?",
            f"Which {label.lower()} has the weakest margin?",
        ],
    )


def won_by_month(fs: FilterState, principal: Principal) -> dict:
    df = subset(slice_frame(fs, principal), "won")
    g = by_time(df, fs, "month")
    return spec(
        "won_by_month", "Closed-won by month", "temporal×measure",
        [{"key": r.key, "value": float(r.value)} for r in g.itertuples(index=False)],
        says=[f"{_measure_key(fs)}.won.t:month"],
        subtitle=f"{fs.measure_label} · close date",
        measure_label=fs.measure_label,
        count_basis="lines", basis_note="Money is line-grain.",
        footnote="Close dates stop at 31 Dec 2026; there is no January onward in this extract.",
        questions=["Which month was strongest?",
                   "How does this compare with the plan?",
                   "What closed in the best month?"],
    )


def ageing_stack(fs: FilterState, principal: Principal) -> dict:
    """Open pipeline by how overdue it is, split by forecast category."""
    df = subset(slice_frame(fs, principal), "open").copy()
    # The loader writes `days_past_due = 0`, not a negative count, for a deal
    # whose close date has not arrived. So "not yet due" is the past-due flag
    # inverted rather than a range on the day count: banding on the number
    # alone put every undue deal in "1-30 days over" and left the first band
    # empty on every page that drew this. `answers._AGE_BANDS` reads the same
    # flag, so the chart and the Ask answer agree band for band.
    bands = [(-10_000, -1, "Not yet due"), (0, 30, "1-30 days over"),
             (31, 90, "31-90 days over"), (91, 10_000, "90+ days over")]
    df["band"] = "Not yet due"
    past = df["is_past_due"].astype(bool)
    for lo, hi, name in bands:
        if lo < 0:
            continue
        df.loc[past & df["days_past_due"].between(lo, hi), "band"] = name
    cats = [b[2] for b in bands]
    series_names = [c for c in ("Commit", "Best Case", "Pipeline", "Omitted")
                    if (df["forecast_category"] == c).any()]
    piv = df.pivot_table(index="band", columns="forecast_category",
                         values="acv_gp" if fs.measure == "gp" else "acv_revenue",
                         aggfunc="sum", fill_value=0.0)
    return spec(
        "ageing_stack", "Open pipeline by how overdue it is",
        "categorical×series×measure",
        {"categories": cats,
         "series": [{"name": s, "values": [float(piv.get(s, {}).get(c, 0.0))
                                           if s in piv.columns else 0.0 for c in cats]}
                    for s in series_names]},
        says=[f"{_measure_key(fs)}.open.by:agebucket+forecast"],
        subtitle=f"{fs.measure_label} · split by the rep's own forecast call",
        measure_label=fs.measure_label,
        count_basis="lines", basis_note="Money is line-grain.",
        footnote="A deal still in Commit weeks after its own close date is the "
                 "combination worth looking at.",
        questions=["How much is more than 90 days overdue?",
                   "Which line of business is worst for overdue deals?",
                   "Who owns the most overdue value?"],
    )


# --------------------------------------------------------------------------- #
# The advanced set
# --------------------------------------------------------------------------- #


def gp_bridge(fs: FilterState, principal: Principal) -> dict:
    """What moved the number — the one question a bar chart cannot answer."""
    steps = B.bridge(fs, principal)
    return spec(
        "gp_bridge", "FY26 gross profit — plan to position", "bridge", steps,
        says=["gap.all.by:quarter", "gp.won.by:quarter"],
        subtitle="Plan, what each quarter delivered, what is still open",
        measure_label="ACV GP",
        footnote=next((s.get("note") for s in steps if s.get("note")), None),
        height=300,
        questions=["Which quarter delivered the most?",
                   "How much is still open?",
                   "What is the plan by quarter?"],
    )


def month_vs_plan(fs: FilterState, principal: Principal) -> dict:
    rows = B.by_month(fs, principal)
    return spec(
        "month_vs_plan", "Closed-won against monthly plan",
        "temporal×measure×measure",
        {"points": [{"key": r["month"], "bar": r["wonGp"], "line": r["budgetGp"],
                     "tone": ("good" if r["attainmentPct"] >= 100
                              else "warn" if r["attainmentPct"] >= 80
                              else "neutral" if r["isFuture"] else "danger"),
                     "annotation": r["annotation"]}
                    for r in rows],
         "barLabel": "Won GP", "lineLabel": "Plan GP", "lineFormat": "currency"},
        says=["gp.won.t:month", "budget.all.t:month"],
        subtitle="Bars are delivered gross profit; the line is plan",
        measure_label="ACV GP",
        footnote="October to December carry a plan but no closed history — the year "
                 "has not reached them. Their red is a calendar position.",
        height=280,
        questions=["Which months beat the plan?",
                   "How much did we win each month?",
                   "Which month was weakest?"],
    )


def stage_funnel(fs: FilterState, principal: Principal) -> dict:
    """
    A COHORT-ANCHORED funnel.

    A naive stage funnel on this data exceeds 100% partway down, because deals
    are born at different rungs — 456 opportunities start at Qualification and
    never appear in the Identification count. So the cohort is defined as
    opportunities whose recorded history STARTS at Identification, and each
    stage counts how many of that cohort ever reached it. That is monotone by
    construction, and it is the only version of this chart that is true.
    """
    df = slice_frame(fs, principal)
    codes = set(df["opportunity_code"])
    mv = features()
    mv = mv[mv["opportunity_code"].isin(codes)]
    entry = mv[mv["stage_path_str"].str.startswith("Identification", na=False)]
    cohort = len(entry)
    ladder = [s for s in STAGE_ORDER if s not in ("Deal Won", "Deal Lost")]

    o = opportunities().set_index("opportunity_code")
    value_col = "acv_gp" if fs.measure == "gp" else "acv_revenue"

    # "Reached this stage" means got AT LEAST this far, which is the only
    # definition that is monotone in the presence of stage skipping — and 73
    # opportunities in this book do skip. Testing for the stage's own name
    # instead would read 1,431 at Proposal against 1,418 at Qualification, a
    # funnel that widens, because a deal can be logged straight from
    # Qualification into Proposal Evaluation.
    #
    # The path is tokenised on the separator rather than substring-matched:
    # "Proposal" is a substring of "Proposal Evaluation", so `contains` counts
    # every Proposal-Evaluation deal twice over and inflates the rung above it.
    rank = {s: i for i, s in enumerate(ladder)}
    furthest = entry["stage_path_str"].map(
        lambda s: max((rank.get(tok.strip(), -1) for tok in str(s).split(">")),
                      default=-1)
    )
    entry = entry.assign(_furthest=furthest)

    rows: list[dict] = []
    prev = cohort
    for i, stage in enumerate(ladder):
        reached = entry[entry["_furthest"] >= i]
        n = len(reached)
        v = float(o.reindex(reached["opportunity_code"])[value_col].fillna(0).sum())
        rows.append({
            "key": stage, "label": stage, "count": n, "value": v,
            "cohortPct": (100 * n / cohort) if cohort else 0.0,
            "stepPct": (100 * n / prev) if prev else 0.0,
            "lostHere": max(prev - n, 0),
        })
        prev = n
    return spec(
        "stage_funnel", "Stage funnel — opportunities that entered at Identification",
        "cohort×stage", rows,
        says=["count.all.by:stage"],
        subtitle=f"Cohort of {cohort:,} opportunities whose logged history starts at "
                 f"Identification",
        measure_label="Opportunities", fmt="number", click_dim="stage",
        count_basis="opportunities",
        basis_note="Stage is constant within an opportunity.",
        footnote="Anchored on the entry cohort, not a running ratio. Counting every "
                 "opportunity at every stage reads above 100% partway down, because "
                 "many deals are logged straight into the middle of the ladder.",
        height=320,
        questions=["Where do most deals fall out?",
                   "How many deals reach the last stage?",
                   "What is the win rate by stage?"],
    )


def deal_triage_bubble(fs: FilterState, principal: Principal) -> dict:
    """What to touch first: silence against value, sized by deal, coloured by risk."""
    r = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    r = r[r["opportunity_code"].isin(codes)]
    tone = {"Critical": "danger", "High": "danger", "Watch": "warn", "Low": "good"}
    return spec(
        "deal_triage", "Open deals — silence against value",
        "x×y×size",
        {"points": [{"id": row.opportunity_code, "label": row.opportunity_name,
                     "x": float(row.quiet_days or 0), "y": float(row.acv_gp),
                     "size": float(row.acv_revenue),
                     "category": row.risk_band,
                     "tone": tone.get(row.risk_band, "neutral")}
                    for row in r.itertuples(index=False)],
         "xLabel": "Days since last logged change", "yLabel": "ACV GP",
         "sizeLabel": "ACV revenue", "xFormat": "days", "yFormat": "currency"},
        says=["risk.open.by:riskband", "quietdays.open.by:account"],
        subtitle=f"{len(r):,} open opportunities · bubble area is revenue",
        measure_label="ACV GP",
        count_basis="opportunities",
        basis_note="One bubble per opportunity, not per line.",
        footnote=f"Median lines mark the quadrants. Anything right of the {STALL_DAYS}-day "
                 f"line has stopped moving.",
        height=360,
        questions=["Which deals have gone quiet and are still big?",
                   "How much value has stopped moving?",
                   "Who owns the deals at risk?"],
    )


def margin_mekko(fs: FilterState, principal: Principal) -> dict:
    """
    Size, mix and margin in one picture.

    Column width is each LOB's revenue, segment height its portfolio mix, and
    fill the margin against the blended rate. Three facts that are usually three
    charts, and the relationship between them is the finding: the widest column
    is also the one whose fill is coldest.
    """
    df = slice_frame(fs, principal)
    blended = 100 * float(df["acv_gp"].sum()) / float(df["acv_revenue"].sum() or 1)
    g = df.groupby(["lob", "portfolio"]).agg(
        revenue=("acv_revenue", "sum"), gp=("acv_gp", "sum")).reset_index()
    g["gm"] = np.where(g["revenue"] != 0, 100 * g["gp"] / g["revenue"], 0.0)
    cols = []
    for lob, sub in g.groupby("lob"):
        total = float(sub["revenue"].sum())
        if total <= 0:
            continue
        cols.append({
            "key": lob, "total": total,
            "segments": [{"key": r.portfolio, "value": float(r.revenue),
                          "fill": float(r.gm)}
                         for r in sub.sort_values("revenue", ascending=False)
                         .itertuples(index=False) if r.revenue > 0],
        })
    cols.sort(key=lambda c: -c["total"])
    return spec(
        "margin_mekko", "Revenue by line of business and portfolio, shaded by margin",
        "categorical×measure×width",
        {"columns": cols, "fillLabel": "GM %", "fillMid": round(blended, 2)},
        says=["rev.all.by:lob+portfolio", "gm.all.by:lob+portfolio"],
        subtitle=f"Column width is revenue · fill is margin against the {blended:.1f}% "
                 f"blended rate",
        measure_label="ACV Revenue",
        count_basis="lines",
        basis_note="LOB and portfolio vary within an opportunity, so this counts lines.",
        footnote="Margin here is SUM(GP)/SUM(revenue) per cell — never the average of "
                 "a margin column, which reads several points higher on this book.",
        height=360,
        questions=["Where is the margin weakest?",
                   "Which line of business is biggest?",
                   "How does the mix differ by line of business?"],
    )


def account_treemap(fs: FilterState, principal: Principal) -> dict:
    """How concentrated the book is. Tail grouped, because one account is 14.5%."""
    c = ACC.concentration(fs, principal, top=20)
    nodes = [{"id": "root", "label": "North America", "parent": None, "value": 0.0}]
    for a in c["accounts"]:
        nodes.append({"id": a["account_code"], "label": a["account_name"],
                      "parent": "root", "value": float(a["gp"]),
                      "secondary": float(a["share"])})
    if c["accountsBeyondTop"] > 0:
        nodes.append({"id": "__other__",
                      "label": f"Other ({c['accountsBeyondTop']} accounts)",
                      "parent": "root", "value": float(c["accountsBeyondTopGp"]),
                      "tone": "neutral"})
    return spec(
        "account_treemap", "Gross profit by account", "hierarchy×measure",
        {"nodes": nodes},
        says=["gp.all.by:account"],
        subtitle=f"Top 20 named · largest account is {c['topAccountShare']:.1f}% of the book",
        measure_label="ACV GP", click_dim="account",
        count_basis="lines", basis_note="Money is line-grain.",
        footnote="The tail is grouped deliberately. Drawn ungrouped, one tile would "
                 "take a seventh of the area and the rest would be unreadable.",
        height=340,
        questions=["Who are the biggest customers?",
                   "How much do the top five carry?",
                   "Which industries are they in?"],
    )


def industry_flow(fs: FilterState, principal: Principal) -> dict:
    """Which verticals feed which lines of business, and how they end."""
    df = slice_frame(fs, principal)
    top_ind = (df.groupby("industry")["acv_gp"].sum()
               .sort_values(ascending=False).head(6).index.tolist())
    d = df.copy()
    d["ind"] = np.where(d["industry"].isin(top_ind), d["industry"], "Other industries")
    d["outcome"] = np.where(d["is_won"], "Won",
                            np.where(d["is_lost"], "Lost", "Open"))

    nodes: list[dict] = []
    seen: set[str] = set()

    def node(nid: str, label: str, depth: int, tone: str | None = None) -> None:
        if nid in seen:
            return
        seen.add(nid)
        n = {"id": nid, "label": label, "depth": depth}
        if tone:
            n["tone"] = tone
        nodes.append(n)

    links: list[dict] = []
    a = d.groupby(["ind", "lob"])["acv_gp"].sum().reset_index()
    for r in a.itertuples(index=False):
        if r.acv_gp <= 0:
            continue
        node(f"i:{r.ind}", r.ind, 0)
        node(f"l:{r.lob}", r.lob, 1)
        links.append({"source": f"i:{r.ind}", "target": f"l:{r.lob}",
                      "value": float(r.acv_gp)})
    b = d.groupby(["lob", "outcome"])["acv_gp"].sum().reset_index()
    tones = {"Won": "good", "Lost": "danger", "Open": "warn"}
    for r in b.itertuples(index=False):
        if r.acv_gp <= 0:
            continue
        node(f"l:{r.lob}", r.lob, 1)
        node(f"o:{r.outcome}", r.outcome, 2, tones.get(r.outcome))
        links.append({"source": f"l:{r.lob}", "target": f"o:{r.outcome}",
                      "value": float(r.acv_gp), "tone": tones.get(r.outcome)})
    return spec(
        "industry_flow", "Industry to line of business to outcome",
        "source×target×measure", {"nodes": nodes, "links": links},
        says=["gp.all.by:industry+lob", "gp.all.by:lob+stage"],
        subtitle="Ribbon width is gross profit · top six industries named",
        measure_label="ACV GP",
        count_basis="lines", basis_note="Money is line-grain.",
        footnote="Won and Lost are settled; Open is still in play and is not a result.",
        height=400,
        questions=["Which industry brings the most?",
                   "Where does Networking business come from?",
                   "Which industry wins most often?"],
    )


def stage_path_flow(fs: FilterState, principal: Principal) -> dict:
    """Do the deals follow the process? Every recorded stage transition."""
    from .loader import movement

    df = slice_frame(fs, principal)
    codes = set(df["opportunity_code"])
    m = movement()
    st = m[(m["field"] == "stage") & (m["opportunity_code"].isin(codes))]
    pairs = st.groupby(["old_value", "new_value"]).size().reset_index(name="n")
    pairs = pairs[pairs["n"] > 0]

    rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    nodes: list[dict] = []
    seen: set[str] = set()
    links: list[dict] = []
    for r in pairs.itertuples(index=False):
        a, b = str(r.old_value), str(r.new_value)
        if a not in rank or b not in rank:
            continue
        da, db = rank[a], min(rank[b], 6)
        if db <= da:
            db = da + 1
        for nid, lbl, dep in ((a, a, da), (b, b, db)):
            if nid not in seen:
                seen.add(nid)
                tone = ("good" if nid == "Deal Won" else
                        "danger" if nid == "Deal Lost" else None)
                n = {"id": nid, "label": lbl, "depth": dep}
                if tone:
                    n["tone"] = tone
                nodes.append(n)
        skipped = rank[b] - rank[a] - 1
        links.append({
            "source": a, "target": b, "value": int(r.n),
            "tone": "warn" if skipped > 0 and b not in ("Deal Won", "Deal Lost") else None,
            "note": f"{skipped} stage(s) skipped" if skipped > 0 else None,
        })
    return spec(
        "stage_path_flow", "Recorded stage transitions", "source×target×measure",
        {"nodes": nodes, "links": links},
        says=["count.all.by:stage"],
        subtitle=f"{int(pairs['n'].sum()):,} logged stage changes",
        measure_label="Transitions", fmt="number",
        count_basis="opportunities",
        basis_note="One ribbon per recorded transition, from the change log.",
        footnote="Amber ribbons jump a rung — the qualification steps between have "
                 "no record.",
        height=400,
        questions=["How many deals skipped a stage?",
                   "Where do deals go after Proposal?",
                   "Which stage loses the most deals?"],
    )


def deal_gantt(fs: FilterState, principal: Principal, limit: int = 30) -> dict:
    """Which deals have run out of road."""
    r = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    r = r[r["opportunity_code"].isin(codes)].head(limit)
    tone = {"Critical": "danger", "High": "danger", "Watch": "warn", "Low": "accent"}
    bars = []
    for row in r.itertuples(index=False):
        if pd.isna(row.create_date) or pd.isna(row.close_date):
            continue
        bars.append({
            "id": row.opportunity_code,
            "label": row.opportunity_name,
            "sublabel": f"{row.stage} · {row.owner}",
            "start": row.create_date.date().isoformat(),
            "end": row.close_date.date().isoformat(),
            "value": float(row.acv_gp),
            "tone": tone.get(row.risk_band, "accent"),
            "pastDue": bool(row.is_past_due),
            "stalled": bool(row.quiet_days >= STALL_DAYS) if pd.notna(row.quiet_days) else False,
            "quietDays": int(row.quiet_days) if pd.notna(row.quiet_days) else None,
        })
    starts = [b["start"] for b in bars] or [AS_OF.isoformat()]
    ends = [b["end"] for b in bars] or [AS_OF.isoformat()]
    return spec(
        "deal_gantt", "Open deals against the calendar", "entity×start×end",
        {"bars": bars, "asOf": AS_OF.isoformat(),
         "rangeStart": min(starts), "rangeEnd": max(max(ends), AS_OF.isoformat())},
        says=["risk.open.by:account"],
        subtitle=f"{len(bars)} highest-risk open deals · create date to close date",
        measure_label="ACV GP",
        count_basis="opportunities",
        basis_note="One bar per opportunity.",
        footnote="The red tail is time already overrun. Hatching means no field has "
                 f"changed in {STALL_DAYS}+ days.",
        questions=["Which deals are furthest past their close date?",
                   "How much value is overdue?",
                   "Which of these has gone quiet?"],
        height=min(120 + len(bars) * 22, 460),
    )


def coverage_heat(fs: FilterState, principal: Principal) -> dict:
    g = B.coverage_grid(fs, principal)
    cells = [{"row": c["lob"], "col": c["portfolio"],
              "value": float(c["coverage"]) if c["coverage"] is not None else 0.0,
              "secondary": float(c["budgetGp"]), "secondaryLabel": "plan GP",
              "zero": bool(c["isHole"])}
             for c in g["cells"]]
    rows = sorted({c["row"] for c in cells})
    cols = sorted({c["col"] for c in cells})
    foot = g["footing"]
    return spec(
        "coverage_heat", f"Coverage by line of business and portfolio — {g['quarter']}",
        "categorical×categorical×measure",
        {"rows": rows, "cols": cols, "cells": cells},
        says=["coverage.open.by:lob+portfolio"],
        subtitle=f"Open GP against remaining plan · {g['holes']} cells have a target and "
                 f"no pipeline at all",
        measure_label="Coverage", fmt="number", click_dim="lob",
        count_basis="lines", basis_note="Money is line-grain.",
        footnote=foot.get("note"),
        height=260,
        questions=["Where is there a target with no pipeline?",
                   "Which area is furthest behind?",
                   "How much plan is left this quarter?"],
    )


def risk_by_band(fs: FilterState, principal: Principal) -> dict:
    r = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    r = r[r["opportunity_code"].isin(codes)]
    order = ["Critical", "High", "Watch", "Low"]
    tones = {"Critical": "danger", "High": "danger", "Watch": "warn", "Low": "good"}
    g = r.groupby("risk_band").agg(value=("acv_gp", "sum"), opps=("opportunity_code", "count"))
    return spec(
        "risk_by_band", "Open pipeline by risk band", "categorical×measure",
        [{"key": b, "value": float(g["value"].get(b, 0.0)),
          "opps": int(g["opps"].get(b, 0)), "tone": tones[b]}
         for b in order if b in g.index],
        says=["gp.open.by:riskband"],
        subtitle="Risk is computed from observable facts, not learned",
        measure_label="ACV GP", click_dim="riskBand",
        count_basis="opportunities", basis_note="One count per opportunity.",
        height=240,
        questions=["What makes a deal high risk?",
                   "How much is in the riskiest band?",
                   "Which deals are worst?"],
    )


def stalled_by_rep(fs: FilterState, principal: Principal) -> dict:
    """
    The manager's Monday-morning bar: whose open book has stopped moving.

    `risk_by_band` says how much of the pod is at risk; this says WHO is
    sitting on it, which is the only version a manager can act on — a band has
    no phone number. Stalled is the product's one definition, `quiet_days >=
    STALL_DAYS`, the same threshold the tiles and the risk score use.
    """
    r = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    r = r[r["opportunity_code"].isin(codes) & (r["quiet_days"] >= STALL_DAYS)]
    g = (r.groupby("owner")
         .agg(value=("acv_gp", "sum"), opps=("opportunity_code", "count"),
              quiet=("quiet_days", "median"))
         .sort_values("value", ascending=False).head(12))
    return spec(
        "stalled_by_rep", "Stalled open pipeline by rep", "categorical×measure",
        [{"key": str(rep), "value": float(row.value), "opps": int(row.opps),
          "quietDays": int(row.quiet) if row.quiet == row.quiet else None,
          "tone": "danger" if row.opps >= 5 else "warn"}
         for rep, row in g.iterrows()],
        says=["gp.stalled.by:rep"],
        subtitle=f"Open gross profit with no logged change in {STALL_DAYS}+ days · "
                 f"{len(r):,} deals",
        measure_label="ACV GP", click_dim="rep",
        count_basis="opportunities", basis_note="One count per opportunity.",
        empty_message="Nothing in this scope has been silent for "
                      f"{STALL_DAYS} days or more.",
        footnote="A rep with several stalled deals is a scrub conversation, not "
                 "a chase; the count decides the colour.",
        height=240,
        questions=["How much of the pipeline has stopped moving?",
                   "Which reps hold the most open pipeline?",
                   "Who should I coach first?"],
    )


def coverage_bullet(fs: FilterState, principal: Principal, dim: str) -> dict:
    """
    Open pipeline against what the quarter still needs, one bullet per line of
    business or portfolio.

    The heat grid on the performance page answers "where is the hole" cell by
    cell. A decision is made a row at a time — fund pipeline generation in
    THIS line, re-phase THAT target — and a bullet reads as exactly that: the
    bar is what is there, the marker is what is needed, and the gap between
    them is the decision. Nothing is rescaled so a bar reaches its marker.

    Coverage is the FORWARD question, measured the way `budget.totals` measures
    it: plan, won and open from the current quarter onward, or from the one
    quarter a filter names. Netting the whole year's wins against the plan
    reads every line as delivered, because Q1 and Q2 over-delivered — true of
    the year, and useless for a decision about the quarters still to come.
    """
    df = slice_frame(fs, principal)
    cells = B.cells_quarter()
    quarters = ([fs.quarter] if fs.quarter else
                sorted(q for q in cells["quarter"].unique() if q >= B.CUR_QUARTER))
    plan = cells[cells["quarter"].isin(quarters)].groupby(dim)["budget_gp"].sum()
    won_df = subset(df, "won")
    open_df = subset(df, "open")
    won = won_df[won_df["fiscal_quarter"].isin(quarters)].groupby(dim)["acv_gp"].sum()
    open_gp = open_df[open_df["fiscal_quarter"].isin(quarters)].groupby(dim)["acv_gp"].sum()
    window = quarters[0] if len(quarters) == 1 else f"{quarters[0]} onward"
    label = REGISTRY[dim].label
    rows = []
    for key in plan.sort_values(ascending=False).index:
        b = float(plan[key])
        w = float(won.get(key, 0.0))
        o = float(open_gp.get(key, 0.0))
        remaining = max(b - w, 0.0)
        c = (o / remaining) if remaining else None
        if remaining <= 0:
            status, tone = "Delivered", "good"
        elif c >= 1.0:
            status, tone = "Covered", "good"
        elif c >= 0.5:
            status, tone = "Thin", "warn"
        else:
            status, tone = "Uncovered", "danger"
        rows.append({"key": str(key), "value": o, "target": remaining,
                     "coverage": c, "budgetGp": b, "wonGp": w,
                     "status": status, "tone": tone})
    uncovered = sum(1 for r in rows if r["status"] == "Uncovered")
    return spec(
        f"coverage_{dim}",
        f"Open pipeline against remaining plan by {label.lower()} — {window}",
        "target×actual", rows,
        says=[f"coverage.open.by:{dim}"],
        subtitle=f"The bar is open gross profit from {window}; the marker is the "
                 f"plan still to deliver · {uncovered} of {len(rows)} below half cover",
        measure_label="ACV GP", click_dim=dim,
        encoding={"x": "key", "y": "value", "target": "target"},
        count_basis="lines", basis_note="Money is line-grain.",
        footnote="Plan exists at entity grain only; a scoped persona sees its "
                 "own pipeline against the whole entity's target.",
        questions=["Where is there a target with no pipeline?",
                   "Which area is furthest behind?",
                   "How much plan is left this quarter?"],
    )


def anomaly_by_category(fs: FilterState, principal: Principal) -> dict:
    a = ANOM.for_persona(principal.key)
    g = a.groupby("category").agg(value=("value_at_stake", "sum"),
                                  n=("anomaly_id", "count")).reset_index()
    g = g.sort_values("value", ascending=False)
    return spec(
        "anomaly_by_category", "Findings by category", "categorical×measure",
        [{"key": r.category, "value": float(r.value), "opps": int(r.n)}
         for r in g.itertuples(index=False)],
        says=["sev.flagged.by:anomcat", "gp.flagged.by:anomcat"],
        subtitle=f"{len(a):,} findings routed to this profile · value is gross profit",
        measure_label="ACV GP at stake", click_dim="anomalyCategory",
        count_basis="opportunities",
        basis_note="Findings sit at mixed grain — opportunity, account, rep or segment.",
        footnote="Value here is gross profit, not revenue — comparing it against a "
                 "revenue chart will look roughly six times off.",
        height=260,
        questions=["Which kind of problem is biggest?",
                   "What is the most serious finding?",
                   "How much money is affected?"],
    )


def rep_benchmark_heat(fs: FilterState, principal: Principal) -> dict:
    """The manager's fingerprint grid: every rep against the peer distribution."""
    from .movement_features import rep_behaviour

    r = rep_behaviour()
    members = principal.predicate.get("owner")
    if members:
        r = r[r["rep"].isin(members)]
    metrics = [
        ("win_rate", "Win rate", False),
        ("shrink_rate", "Value shrink", True),
        ("inflate_rate", "Sandbagging", True),
        ("regression_rate", "Forecast reversal", True),
        ("stall_rate", "Stalled book", True),
        ("new_business_share", "New-business mix", False),
    ]
    cells = []
    for row in r.itertuples(index=False):
        for col, label, bad_high in metrics:
            z = float(getattr(row, f"{col}_z", 0.0) or 0.0)
            raw = float(getattr(row, col, 0.0) or 0.0)
            # Normalise so "worse" is always the high end of the ramp, whichever
            # direction the underlying metric runs.
            cells.append({
                "row": row.rep, "col": label,
                "value": z if bad_high else -z,
                "secondary": raw * 100, "secondaryLabel": "%",
            })
    return spec(
        "rep_benchmark", "Rep behaviour against the peer distribution",
        "categorical×categorical×measure",
        {"rows": sorted(r["rep"].tolist()),
         "cols": [m[1] for m in metrics], "cells": cells},
        says=["winrate.closed.by:rep", "risk.open.by:rep"],
        subtitle="Standard deviations from the all-rep norm · high is worse in every column",
        measure_label="SD from peer norm", fmt="number", click_dim="rep",
        count_basis="opportunities", basis_note="Rates are per owned opportunity.",
        footnote="Behaviour repeats by person and is coachable. Win rate does not vary "
                 "meaningfully by rep in this extract — coach the behaviour, not the "
                 "outcome.",
        questions=["Who is furthest from the team norm?",
                   "Who should I coach first?",
                   "How does the team compare on win rate?"],
        height=max(200, 40 + len(r) * 22),
    )


def whitespace_table(fs: FilterState, principal: Principal) -> dict:
    rows = ACC.whitespace(fs, principal, limit=25)
    return spec(
        "whitespace", "Accounts with room to grow", "table",
        {"columns": [
            {"key": "accountName", "label": "Account", "align": "left"},
            {"key": "gp", "label": "Current GP", "format": "currency", "align": "right"},
            {"key": "holdsLabel", "label": "Buys today", "align": "left"},
            {"key": "recommendedLob", "label": "Not buying", "align": "left"},
            {"key": "peerAttachRatePct", "label": "Peer attach", "format": "percent",
             "align": "right"},
            {"key": "estimatedGp", "label": "Peer median GP", "format": "currency",
             "align": "right"},
        ],
         "rows": [{**r, "holdsLabel": ", ".join(r["holds"]),
                   "peerAttachRatePct": r["peerAttachRate"] * 100} for r in rows]},
        says=["gp.all.by:account"],
        subtitle="Upside, not risk — framed as the reference guide requires",
        measure_label="ACV GP", key="table.compact",
        empty_message=(
            "No cross-sell gap in scope — every account above "
            f"{money(ACC.MATERIAL_ACCOUNT_GP)} of gross profit already buys across "
            "all four lines of business."
        ),
        count_basis="lines", basis_note="Account totals are line-grain.",
        footnote="Estimate is the median gross profit that size-matched accounts earn "
                 "from the missing line of business.",
        questions=["Which customer is the best opportunity?",
                   "What should we sell them?",
                   "How much is this worth?"],
    )


# --------------------------------------------------------------------------- #
# Cross-sell — the data-science export, at three resolutions
# --------------------------------------------------------------------------- #
#
# The same 72 recommendations, asked three different questions. This is the rule
# the product holds itself to: a chart repeats only when the ROWS and the
# QUESTION both differ. A rep needs one account and one conversation; a manager
# needs to know who to hand it to; a leader needs to know whether twenty-nine
# accounts missing the same thing is a campaign. None of those is readable off
# either of the others.


def cross_sell_list(fs: FilterState, principal: Principal) -> dict:
    """AE and manager: the ranked worklist, one row per conversation."""
    rows = XS.unified(fs, principal, limit=25)
    return spec(
        "xsell-list", "Growth ideas for your accounts", "table",
        {"columns": [
            {"key": "accountName", "label": "Account", "align": "left"},
            {"key": "offering", "label": "Suggest they add", "align": "left"},
            {"key": "confidence", "label": "How sure", "align": "left"},
            {"key": "methodLabel", "label": "Why we think so", "align": "left"},
            {"key": "peerGp", "label": "Peer GP", "format": "currency", "align": "right"},
            {"key": "owner", "label": "Owner", "align": "left"},
        ],
         "rows": [{**r, "methodLabel": " + ".join(
             XS.METHOD_MEANING.get(m, m).rstrip(".") for m in r["methods"])}
             for r in rows]},
        says=["xsell.rec.by:account"],
        subtitle="Ranked by how many independent methods agree, not by size",
        measure_label="Peer ACV GP", key="table.compact",
        empty_message=(
            "No cross-sell recommendation for your accounts in this scope. The "
            "engine keeps a deliberately short, high-confidence list."),
        count_basis="lines",
        basis_note="One row per account and suggested offering.",
        footnote=XS.CAVEAT,
        questions=["Which of these should I start with?",
                   "Why is this account a good fit?",
                   "What is a peer account worth on this?"],
    )


def cross_sell_themes(fs: FilterState, principal: Principal) -> dict:
    """
    Executive: the same rows grouped into plays, sized by how many accounts.

    A bar of accounts rather than of money, deliberately. The money on a
    cross-sell theme is an estimate built on peer averages; the account COUNT is
    a fact, and the fact is what decides whether something is a campaign.
    """
    th = XS.themes(fs, principal)
    return spec(
        "xsell-themes", "Growth plays worth running", "categorical×measure",
        [{"key": t["offering"], "value": t["accounts"],
          "meta": {"estimatedGp": t["estimatedGp"], "owners": t["ownerCount"],
                   "topIndustry": t["topIndustry"],
                   "strongest": t["strongest"]["accountName"]}}
         for t in th],
        says=["xsell.theme.by:offering"],
        subtitle="Number of accounts missing the same thing",
        measure_label="Accounts", fmt="number",
        empty_message=("No offering is missing at two or more accounts in scope, so "
                       "there is no play here — only individual conversations."),
        count_basis="lines",
        basis_note="One bar per recommended offering; height is accounts, not money.",
        footnote="An offering recommended at only one account is left out — a theme "
                 "of one is an account action wearing the wrong label.",
        questions=["Which play should we launch first?",
                   "Who owns the accounts in the biggest play?",
                   "How much is the biggest play worth?"],
    )


def cross_sell_matrix(fs: FilterState, principal: Principal) -> dict:
    """
    Executive: industry × offering, coloured by how many accounts and how sure.

    The SOW's "Growth Whitespace Matrix": cell strength uses confidence. The
    score is `accounts × mean confidence rank`, so one very-high-confidence gap
    does not outrank five credible ones, and five weak ones do not outrank two
    strong.
    """
    recs = XS.unified(fs, principal, limit=10_000)
    agg: dict[tuple[str, str], list[int]] = {}
    for r in recs:
        if not r["industry"]:
            continue
        agg.setdefault((r["industry"], r["offering"]), []).append(r["confidenceRank"])
    cells = [{"row": ind, "col": off, "value": round(len(v) * (sum(v) / len(v)), 2),
              "accounts": len(v)}
             for (ind, off), v in agg.items()]
    rows = sorted({c["row"] for c in cells})
    cols = sorted({c["col"] for c in cells},
                  key=lambda o: -sum(c["accounts"] for c in cells if c["col"] == o))
    return spec(
        "xsell-matrix", "Where growth repeats", "categorical×categorical×measure",
        {"rows": rows, "cols": cols, "cells": cells},
        says=["xsell.rec.by:industry"],
        subtitle="Industry against suggested offering — darker means more accounts, "
                 "more sure",
        measure_label="Strength", fmt="number",
        empty_message="No recommendation in scope carries an industry, so there is "
                      "nothing to compare across industries.",
        count_basis="lines",
        basis_note="Strength is accounts multiplied by average confidence.",
        footnote=XS.CAVEAT,
        questions=["Which industry has the most room to grow?",
                   "What is the strongest industry and offering pair?",
                   "Which offering is missing across the most industries?"],
    )


def reset_caches() -> None:
    pass
