"""
Account structure: concentration on one side, cross-sell whitespace on the other.

These are the same measurement read two ways, which is why they live together.
An account that buys one line of business is a concentration risk for the entity
and a growth opportunity for whoever owns it; the Anomaly Reference Guide is
explicit that the whitespace findings must be presented as upside, not as
problems, and this module carries that framing through in the data rather than
leaving it to the UI to remember.

The attach analysis is the part with real teeth. It is not "this account has
gaps" — every account has gaps. It is: *given what this account already buys,
what do comparable accounts buy that this one does not*, and how much is that
worth. Measured on this book, the asymmetry is the finding: accounts that buy
Security almost always also buy Networking, while accounts that buy Networking
usually do not buy Security. That direction tells you which play to run.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .loader import LOB_ORDER, PORTFOLIO_ORDER, facts
from .measures import FilterState, slice_frame
from .personas import Principal

#: Below this an account is too small for a cross-sell motion to be worth a play.
MATERIAL_ACCOUNT_GP = 25_000


@functools.lru_cache(maxsize=1)
def account_profile() -> pd.DataFrame:
    """One row per account: what they buy, from whom, and how much."""
    f = facts()
    g = f.groupby("account_code")
    base = pd.DataFrame({
        "account_name": g["account_name"].first(),
        "industry": g["industry"].first(),
        "country": g["country"].first(),
        "account_group": g["account_group"].first(),
        "owner": g["account_owner"].first(),
        "revenue": g["acv_revenue"].sum(),
        "gp": g["acv_gp"].sum(),
        "lines": g["line_code"].count(),
        "opportunities": g["opportunity_code"].nunique(),
        "lob_count": g["lob"].nunique(),
        "portfolio_count": g["portfolio"].nunique(),
        "won_gp": g.apply(lambda d: d.loc[d["is_won"], "acv_gp"].sum(), include_groups=False),
        "open_gp": g.apply(lambda d: d.loc[d["is_open"], "acv_gp"].sum(), include_groups=False),
    })
    base["gm"] = np.where(base["revenue"] != 0, 100 * base["gp"] / base["revenue"], np.nan)

    # Which LOBs and portfolios each account actually holds, as sets.
    lobs = f.groupby("account_code")["lob"].apply(lambda s: frozenset(s.unique()))
    ports = f.groupby("account_code")["portfolio"].apply(lambda s: frozenset(s.unique()))
    base["lobs"] = lobs
    base["portfolios"] = ports
    base["missing_lobs"] = base["lobs"].map(lambda s: tuple(l for l in LOB_ORDER if l not in s))
    base["missing_portfolios"] = base["portfolios"].map(
        lambda s: tuple(p for p in PORTFOLIO_ORDER if p not in s))

    total_gp = float(base["gp"].sum())
    base["gp_share"] = 100 * base["gp"] / total_gp if total_gp else 0.0
    return base.sort_values("gp", ascending=False).reset_index()


@functools.lru_cache(maxsize=1)
def attach_matrix() -> pd.DataFrame:
    """
    P(account buys B | account buys A) for every ordered pair of LOBs.

    Directional on purpose. A symmetric co-occurrence count would hide the only
    actionable fact in here: if Security buyers nearly all hold Networking but
    Networking buyers mostly do not hold Security, then the play is to attach
    Security to the Networking base — not the reverse, which is already done.
    """
    p = account_profile()
    rows = []
    for a in LOB_ORDER:
        has_a = p[p["lobs"].map(lambda s, _a=a: _a in s)]
        for b in LOB_ORDER:
            if a == b:
                continue
            both = has_a["lobs"].map(lambda s, _b=b: _b in s).sum()
            rows.append({
                "given": a, "then": b,
                "accountsWithGiven": int(len(has_a)),
                "accountsWithBoth": int(both),
                "attachRate": float(both / len(has_a)) if len(has_a) else 0.0,
                "gapAccounts": int(len(has_a) - both),
                "gapGp": float(has_a.loc[~has_a["lobs"].map(
                    lambda s, _b=b: _b in s), "gp"].sum()),
            })
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def lob_count_value() -> pd.DataFrame:
    """
    Median account value by how many LOBs that account buys.

    This is the single number that turns a cross-sell slide into a business case,
    and it must be reported as a MEDIAN with its sample size — the mean is
    wrecked by one account holding 14.5% of all gross profit.
    """
    p = account_profile()
    g = p.groupby("lob_count").agg(
        accounts=("gp", "size"),
        medianGp=("gp", "median"),
        meanGp=("gp", "mean"),
        totalGp=("gp", "sum"),
    ).reset_index()
    total = float(p["gp"].sum())
    g["shareOfGp"] = 100 * g["totalGp"] / total if total else 0.0
    g["shareOfAccounts"] = 100 * g["accounts"] / len(p)
    return g


def whitespace(fs: FilterState, principal: Principal, limit: int = 40) -> list[dict]:
    """
    Accounts with room to grow, ranked by what the gap is plausibly worth.

    The estimate is deliberately conservative and its method is stated on the
    card: for each missing LOB, the MEDIAN gross profit that comparable accounts
    (same LOB count band) earn from that LOB — not the mean, and never the
    biggest comparable, which is how cross-sell decks end up promising numbers
    nobody can find later.
    """
    scoped = slice_frame(fs, principal)
    codes = set(scoped["account_code"])
    p = account_profile()
    p = p[p["account_code"].isin(codes)]
    p = p[(p["gp"] >= MATERIAL_ACCOUNT_GP) & (p["lob_count"] < len(LOB_ORDER))]

    f = facts()
    per_lob = f.groupby(["account_code", "lob"])["acv_gp"].sum().reset_index()

    # Size-matched comparables. A flat median across all accounts answers "what
    # does a typical account spend on Data Center", which is the wrong question:
    # a $200K account will not behave like the 357-account median. The benchmark
    # is therefore the median spend on the missing LOB among accounts in the SAME
    # total-GP quartile that actually hold it — a comparison the account owner
    # can defend in the room.
    prof = account_profile().set_index("account_code")
    quartile = pd.qcut(prof["gp"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    per_lob["size_band"] = per_lob["account_code"].map(quartile).astype(object)
    banded = (
        per_lob.groupby(["size_band", "lob"])["acv_gp"]
        .agg(peer_gp="median", peer_n="size").reset_index()
    )
    benchmark_band = {
        (r.size_band, r.lob): (float(r.peer_gp), int(r.peer_n))
        for r in banded.itertuples(index=False)
    }
    benchmark_all = per_lob.groupby("lob")["acv_gp"].median().to_dict()
    attach = attach_matrix().set_index(["given", "then"])["attachRate"].to_dict()

    out = []
    for r in p.itertuples(index=False):
        held = r.lobs
        band = quartile.get(r.account_code, "Q4")
        candidates = []
        for miss in r.missing_lobs:
            # Strongest signal first: how often accounts that hold what THIS
            # account holds go on to hold the missing one.
            rates = [attach.get((h, miss), 0.0) for h in held]
            peer_gp, peer_n = benchmark_band.get(
                (band, miss), (float(benchmark_all.get(miss, 0.0)), 0))
            # Too few size-matched comparables to quote one; fall back and say so.
            if peer_n < 5:
                peer_gp, peer_n = float(benchmark_all.get(miss, 0.0)), 0
            candidates.append({
                "lob": miss,
                "peerAttachRate": float(max(rates) if rates else 0.0),
                "medianPeerGp": peer_gp,
                "comparables": peer_n,
                "sizeBand": str(band),
            })
        candidates.sort(key=lambda c: (-(c["peerAttachRate"] * c["medianPeerGp"]),
                                       -c["peerAttachRate"]))
        if not candidates:
            continue
        best = candidates[0]
        out.append({
            "accountCode": r.account_code,
            "accountName": r.account_name,
            "industry": r.industry,
            "owner": r.owner,
            "gp": float(r.gp),
            "revenue": float(r.revenue),
            "lobCount": int(r.lob_count),
            "holds": sorted(held),
            "missing": list(r.missing_lobs),
            "recommendedLob": best["lob"],
            "peerAttachRate": best["peerAttachRate"],
            "estimatedGp": best["medianPeerGp"],
            "allCandidates": candidates,
            "framing": "opportunity",
            "comparables": best["comparables"],
            "sizeBand": best["sizeBand"],
            "method": (
                f"{best['peerAttachRate'] * 100:.0f}% of accounts holding "
                f"{', '.join(sorted(held))} also hold {best['lob']}. The estimate is the "
                f"median {best['lob']} gross profit among "
                + (f"the {best['comparables']} accounts of comparable size "
                   f"(same total-GP quartile) that hold it"
                   if best["comparables"] >= 5
                   else f"all accounts that hold it — too few size-matched comparables "
                        f"to quote a banded figure")
                + ". Median, not mean: one account carries 14.5% of the book."
            ),
        })
    out.sort(key=lambda r: -(r["peerAttachRate"] * r["estimatedGp"]))
    return out[:limit]


def concentration(fs: FilterState, principal: Principal, top: int = 20) -> dict:
    """Where the book is over-exposed — by account and by industry."""
    df = slice_frame(fs, principal)
    total_gp = float(df["acv_gp"].sum())

    acc = df.groupby(["account_code", "account_name"]).agg(
        gp=("acv_gp", "sum"), revenue=("acv_revenue", "sum"),
        opportunities=("opportunity_code", "nunique"),
    ).reset_index().sort_values("gp", ascending=False)
    acc["share"] = 100 * acc["gp"] / total_gp if total_gp else 0.0

    ind = df.groupby("industry").agg(
        gp=("acv_gp", "sum"), revenue=("acv_revenue", "sum"),
        accounts=("account_code", "nunique"),
    ).reset_index().sort_values("gp", ascending=False)
    ind["share"] = 100 * ind["gp"] / total_gp if total_gp else 0.0

    def hhi(shares: pd.Series) -> float:
        """Herfindahl index on percentage shares; 10,000 is a monopoly."""
        return float((shares ** 2).sum())

    return {
        "totalGp": total_gp,
        "accounts": acc.head(top).to_dict("records"),
        "accountsBeyondTop": int(max(len(acc) - top, 0)),
        "accountsBeyondTopGp": float(acc["gp"].iloc[top:].sum()) if len(acc) > top else 0.0,
        "industries": ind.to_dict("records"),
        "topAccountShare": float(acc["share"].iloc[0]) if len(acc) else 0.0,
        "top5AccountShare": float(acc["share"].head(5).sum()) if len(acc) else 0.0,
        "topIndustryShare": float(ind["share"].iloc[0]) if len(ind) else 0.0,
        "accountHhi": hhi(acc["share"]),
        "industryHhi": hhi(ind["share"]),
        "note": (
            "Share is of gross profit, which is the basis the plan is set in. The top "
            "account alone carries a double-digit share, so any treemap or bubble of "
            "this data collapses to one tile unless the tail is grouped — the server "
            "groups it rather than letting the chart mislead."
        ),
    }


def account_detail(account_code: str) -> dict | None:
    """The drill-down behind a whitespace or concentration card."""
    p = account_profile()
    row = p[p["account_code"] == account_code]
    if row.empty:
        return None
    row = row.iloc[0]
    f = facts()
    lines = f[f["account_code"] == account_code]
    grid = lines.groupby(["lob", "portfolio"])["acv_gp"].sum().reset_index()
    return {
        "accountCode": account_code,
        "accountName": row["account_name"],
        "industry": row["industry"],
        "country": row["country"],
        "owner": row["owner"],
        "gp": float(row["gp"]),
        "revenue": float(row["revenue"]),
        "gm": float(row["gm"]) if pd.notna(row["gm"]) else None,
        "gpShare": float(row["gp_share"]),
        "opportunities": int(row["opportunities"]),
        "lobCount": int(row["lob_count"]),
        "holds": sorted(row["lobs"]),
        "missing": list(row["missing_lobs"]),
        "missingPortfolios": list(row["missing_portfolios"]),
        "grid": grid.to_dict("records"),
        "openGp": float(row["open_gp"]),
        "wonGp": float(row["won_gp"]),
    }


def reset_caches() -> None:
    for fn in (account_profile, attach_matrix, lob_count_value):
        fn.cache_clear()
