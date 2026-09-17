"""
Cross-sell: the data-science export, fused with what this layer computes itself.

Two engines answer the same question — *what is this account missing that
accounts like it buy?* — and they answer it at different resolutions:

  * **The DS export** (`data/source/cross_sell.csv`, 72 rows over 46 accounts)
    works at **LOB / portfolio** resolution: not "sell them Security" but "sell
    them Security / Technical Services". It carries three independent methods
    (market-basket association, industry bundle, similar-account matching), a
    plain-word confidence, and a written reason per method.
  * **`accounts.whitespace()`** works at **LOB** resolution and estimates what
    the gap is worth using size-matched comparables, which the DS export does
    not do (it quotes a flat peer average).

Fusing them is worth more than either alone, for the same reason the anomaly
layer fuses its two sources: **agreement between methods that cannot see each
other is the strongest signal either can produce.** A recommendation both
engines reach is `corroborated`; one only the DS export reaches is usually an
*upsell inside an LOB the account already buys*, which the LOB-level view is
blind to by construction; one only this layer reaches is a whole missing LOB the
DS export's stricter bar rejected.

The reference guide that came with the export is explicit that these are
**opportunities, not risks**, and that a recommendation is "a starting point,
not a guaranteed sale". Both statements are carried onto the cards rather than
left in a document nobody opens — see `CAVEAT`.
"""

from __future__ import annotations

import functools
import re

import pandas as pd

from .loader import LOB_ORDER, cross_sell_raw, facts
from .measures import FilterState, slice_frame
from .personas import Principal

#: The export's own words for how many independent methods agreed. Ranked so a
#: list can be sorted without the UI knowing the vocabulary.
CONFIDENCE_RANK: dict[str, int] = {"Very High": 3, "High": 2, "Medium": 1}

#: What each confidence word actually means, from the reference guide. Shown on
#: the card, because "High" on its own is a word, not evidence.
CONFIDENCE_MEANING: dict[str, str] = {
    "Very High": "More than one method found this independently.",
    "High": "One method found it, and it stood out among that method's own findings.",
    "Medium": "One method found it and it cleared that method's bar.",
}

#: The three engines, in the guide's own plain terms. The UI shows these instead
#: of the method names, which mean nothing to an account owner.
METHOD_MEANING: dict[str, str] = {
    "Market Basket Rule": "Accounts that buy one of these usually buy the other.",
    "Peer/Industry Bundle": "Most accounts in this industry already have it.",
    "Similar Account": "Accounts with near-identical buying patterns have it.",
}

#: Carried onto every card. The guide asks for both sentences explicitly.
CAVEAT = (
    "A growth idea, not a forecast. It is a reason to open a specific "
    "conversation, not a deal that will close."
)

_REC = re.compile(r"^\s*Add\s+(?P<lob>[^/]+?)\s*/\s*(?P<portfolio>.+?)\s*$")


def _split_recommendation(text: str) -> tuple[str, str]:
    """`Add Security / VBR` -> `("Security", "VBR")`.

    A row whose wording does not match is kept with an empty portfolio rather
    than dropped: losing a recommendation silently is worse than showing one
    that only names its LOB.
    """
    m = _REC.match(str(text or ""))
    if not m:
        return str(text or "").replace("Add ", "").strip(), ""
    return m.group("lob").strip(), m.group("portfolio").strip()


def _reasons(why: str, methods: list[str]) -> list[dict]:
    """
    Cut the `Why` cell into one reason per method, paired with its method name.

    The export joins reasons with ` | ` in the same order the methods are
    listed. When the counts disagree the reasons are returned unlabelled — an
    unlabelled true sentence beats a labelled wrong one.
    """
    # The export writes "--" for a dash. Left alone it reads as a typo on a
    # card the client will look at closely.
    parts = [p.strip().replace(" -- ", " — ")
             for p in str(why or "").split("|") if p.strip()]
    if len(parts) == len(methods):
        return [{"method": m, "meaning": METHOD_MEANING.get(m, ""), "text": p}
                for m, p in zip(methods, parts)]
    return [{"method": "", "meaning": "", "text": p} for p in parts]


@functools.lru_cache(maxsize=1)
def recommendations() -> pd.DataFrame:
    """
    The export, parsed: one row per account/offering pair, ready to rank.

    Adds the owner from the fact table, because the export names the account but
    not who works it, and every action in the SOW needs an owner.
    """
    df = cross_sell_raw().copy()
    split = df["recommendation"].map(_split_recommendation)
    df["rec_lob"] = [s[0] for s in split]
    df["rec_portfolio"] = [s[1] for s in split]
    df["offering"] = df["rec_lob"] + " / " + df["rec_portfolio"]
    df["offering"] = df["offering"].str.rstrip(" /")
    df["confidence_rank"] = df["confidence"].map(CONFIDENCE_RANK).fillna(0).astype(int)
    df["method_list"] = df["methods"].fillna("").map(
        lambda s: [m.strip() for m in str(s).split(",") if m.strip()])
    df["holds"] = df["current_products"].fillna("").map(
        lambda s: [p.strip() for p in str(s).split(",") if p.strip()])
    df["holds_lobs"] = df["holds"].map(
        lambda ps: sorted({p.split("/")[0].strip() for p in ps}))

    # The owner of the account's largest line, which is how every other surface
    # in this layer decides who "owns" an account.
    f = facts()
    owner = (f.groupby(["account_code", "owner"])["acv_gp"].sum()
             .reset_index().sort_values("acv_gp", ascending=False)
             .drop_duplicates("account_code").set_index("account_code")["owner"])
    df["owner"] = df["account_code"].map(owner).fillna("Unassigned")
    return df


@functools.lru_cache(maxsize=1)
def _native_missing() -> dict[str, set[str]]:
    """Per account, the whole LOBs this layer's own view says are absent."""
    f = facts()
    held = f.groupby("account_code")["lob"].agg(lambda s: set(s.dropna()))
    return {code: set(LOB_ORDER) - lobs for code, lobs in held.items()}


def unified(fs: FilterState, principal: Principal, limit: int = 60) -> list[dict]:
    """
    Every recommendation the signed-in user is entitled to see, ranked.

    Scoped by the SAME row-level rule as everything else: an account appears
    only if the principal can see at least one of its lines. A rep therefore
    sees growth ideas for their own accounts and nobody else's, and the ranking
    they see is the ranking of their own book, not a filtered view of the
    entity's.

    Ranked by confidence first and peer gross profit second. Confidence leads
    because the guide is explicit that method agreement is the strongest signal
    the engine produces — a bigger number found by one method is a worse bet
    than a smaller one found by three.
    """
    scoped = slice_frame(fs, principal)
    codes = set(scoped["account_code"])
    df = recommendations()
    df = df[df["account_code"].isin(codes)]
    if df.empty:
        return []

    native = _native_missing()
    out: list[dict] = []
    for r in df.itertuples(index=False):
        missing_lobs = native.get(r.account_code, set())
        corroborated = r.rec_lob in missing_lobs
        out.append({
            "id": r.recommendation_id,
            "accountCode": r.account_code,
            "accountName": r.account_name,
            "industry": r.industry,
            "owner": r.owner,
            "offering": r.offering,
            "lob": r.rec_lob,
            "portfolio": r.rec_portfolio,
            "confidence": r.confidence,
            "confidenceRank": int(r.confidence_rank),
            "confidenceMeaning": CONFIDENCE_MEANING.get(r.confidence, ""),
            "methods": list(r.method_list),
            "methodCount": int(r.num_methods),
            "reasons": _reasons(r.why, list(r.method_list)),
            "holds": list(r.holds),
            "holdsLobs": list(r.holds_lobs),
            "peerRevenue": float(r.peer_won_revenue),
            "peerGp": float(r.peer_won_gp),
            "accountGp": float(r.account_won_gp),
            # Whole LOB the account does not buy at all, versus a new service
            # inside an LOB it already buys. Different conversation, different
            # person, and the card says which.
            "kind": "new line of business" if corroborated else "more of what they buy",
            "corroborated": bool(corroborated),
            "source": "both" if corroborated else "data science",
            "caveat": CAVEAT,
            "framing": "opportunity",
        })

    out.sort(key=lambda r: (-r["confidenceRank"], -r["methodCount"], -r["peerGp"]))
    return out[:limit]


def themes(fs: FilterState, principal: Principal, min_accounts: int = 2) -> list[dict]:
    """
    The same recommendations grouped by offering — a play, not a to-do list.

    This is the difference between the executive view and the rep view of the
    same data. A rep needs 'call Westfall about Security'; a commercial head
    needs 'twenty-nine accounts are missing Networking / Product' — one is an
    action, the other is a campaign, and a leader cannot see the campaign in a
    list sorted by account.

    Offerings recommended at only one account are dropped: a theme of one is an
    account action that has been mislabelled.
    """
    recs = unified(fs, principal, limit=10_000)
    if not recs:
        return []
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(r["offering"], []).append(r)

    out = []
    for offering, rows in by.items():
        if len(rows) < min_accounts:
            continue
        industries = sorted({r["industry"] for r in rows if r["industry"]})
        owners = sorted({r["owner"] for r in rows})
        best = max(rows, key=lambda r: (r["confidenceRank"], r["peerGp"]))
        out.append({
            "offering": offering,
            "lob": rows[0]["lob"],
            "portfolio": rows[0]["portfolio"],
            "accounts": len(rows),
            "owners": owners,
            "ownerCount": len(owners),
            "industries": industries,
            "topIndustry": max(industries, key=lambda i: sum(
                1 for r in rows if r["industry"] == i)) if industries else "",
            # What the play is plausibly worth: peer gross profit per account,
            # summed. Stated as an order of magnitude on the card, never as a
            # forecast — see CAVEAT.
            "estimatedGp": float(sum(r["peerGp"] for r in rows)),
            "confidenceMix": {
                c: sum(1 for r in rows if r["confidence"] == c)
                for c in ("Very High", "High", "Medium")
                if any(r["confidence"] == c for r in rows)
            },
            "strongest": {
                "accountName": best["accountName"],
                "owner": best["owner"],
                "confidence": best["confidence"],
                "reason": best["reasons"][0]["text"] if best["reasons"] else "",
            },
            "corroborated": sum(1 for r in rows if r["corroborated"]),
            "caveat": CAVEAT,
            "framing": "opportunity",
        })

    out.sort(key=lambda t: (-t["accounts"], -t["estimatedGp"]))
    return out


def for_account(account_code: str) -> list[dict]:
    """Every recommendation for one account, for the account drawer."""
    df = recommendations()
    rows = df[df["account_code"] == account_code]
    native = _native_missing().get(account_code, set())
    out = []
    for r in rows.itertuples(index=False):
        out.append({
            "id": r.recommendation_id,
            "offering": r.offering,
            "lob": r.rec_lob,
            "portfolio": r.rec_portfolio,
            "confidence": r.confidence,
            "confidenceMeaning": CONFIDENCE_MEANING.get(r.confidence, ""),
            "methods": list(r.method_list),
            "reasons": _reasons(r.why, list(r.method_list)),
            "peerGp": float(r.peer_won_gp),
            "peerRevenue": float(r.peer_won_revenue),
            "accountGp": float(r.account_won_gp),
            "corroborated": r.rec_lob in native,
            "caveat": CAVEAT,
        })
    out.sort(key=lambda r: -CONFIDENCE_RANK.get(r["confidence"], 0))
    return out


def summary(fs: FilterState, principal: Principal) -> dict:
    """Counts for the tiles and for the AI facts pack."""
    recs = unified(fs, principal, limit=10_000)
    th = themes(fs, principal)
    strong = [r for r in recs if r["confidenceRank"] >= 2]
    return {
        "recommendations": len(recs),
        "accounts": len({r["accountCode"] for r in recs}),
        "strong": len(strong),
        "veryHigh": sum(1 for r in recs if r["confidence"] == "Very High"),
        "corroborated": sum(1 for r in recs if r["corroborated"]),
        "themes": len(th),
        "estimatedGp": float(sum(r["peerGp"] for r in recs)),
        "topTheme": th[0]["offering"] if th else "",
        "topThemeAccounts": th[0]["accounts"] if th else 0,
        "topRecommendation": recs[0] if recs else None,
        "caveat": CAVEAT,
    }


def reset_caches() -> None:
    for fn in (recommendations, _native_missing):
        fn.cache_clear()
