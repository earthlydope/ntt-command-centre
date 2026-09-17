"""
Smart actions — one card shape, three vocabularies.

The client's own card format, from the 15 Sep call: *what* it is, *why* it fired,
*how severe*, *what value* is at stake, *who* owns it, and the *recommended
action*. Every card below carries all six, plus the literal predicate that fired
it, shown behind "Why this fired" — because a recommendation a user cannot audit
is a recommendation they will stop trusting the second one of them is wrong.

**The verbs differ by role, and so does the ranking.** An AE is told to call an
account today; a manager to coach a named rep; an executive to move coverage.
Ranking follows the same logic:

    AE         urgency dominates       — the question is what to do in the next
                                         two hours, so days silent and days past
                                         the close date drive the sort.
    manager    persistence dominates   — one bad deal is noise; the same pattern
                                         from the same person is the signal, so
                                         the peer z-score drives the sort.
    executive  money and structure     — quarter-relative, deduplicated to one
                                         card per entity, capped at what fits on
                                         a screen someone reads aloud.

A card that cannot name an owner is not emitted. "Someone should look at this"
is how an action list becomes wallpaper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from . import accounts as ACC
from . import anomalies as ANOM
from . import budget as B
from . import crosssell as XS
from . import predict as P
from .loader import AS_OF, CUR_QUARTER
from .measures import FilterState, money, pct, slice_frame
from .movement_features import STALL_DAYS, rep_behaviour
from .personas import Principal


#: The status a decision can be in, in the order it moves through them. The
#: client's functional spec names these six and no others; the UI stores the
#: chosen one per card for the session, which is what a read-only demo can
#: honestly do — there is no Salesforce to write back to.
STATUSES: tuple[str, ...] = ("New", "In Review", "Actioned", "Monitoring",
                             "Closed", "Dismissed")

#: The two choices every card offers on top of its own specific ones. Keeping
#: them last and identical everywhere means a user never has to hunt for "not
#: now" — and `dismiss` always demands a reason, which is the point of it.
UNIVERSAL_OPTIONS: tuple[dict, ...] = (
    {"key": "monitor", "label": "Watch it", "status": "Monitoring",
     "needsReason": False,
     "meaning": "No change today. It stays on the list until something moves."},
    {"key": "dismiss", "label": "Not worth doing", "status": "Dismissed",
     "needsReason": True,
     "meaning": "Say why, so next time the system can tell 'not pursued' from "
                "'never seen'."},
)


#: The choices a card can offer, by kind. Written once so two cards of the same
#: kind offer the same words — a worklist where every card invents its own verbs
#: is a worklist nobody learns.
OPTIONS: dict[str, tuple[dict, ...]] = {
    "deal_review": (
        {"key": "review", "label": "Review the deal", "status": "In Review",
         "needsReason": False,
         "meaning": "Open it, check stage, close date and value, and log what you find."},
        {"key": "recovery", "label": "Start a recovery step", "status": "Actioned",
         "needsReason": False,
         "meaning": "Create a dated next step against the thing that is blocking it."},
        {"key": "help", "label": "Ask for help", "status": "Actioned",
         "needsReason": True,
         "meaning": "Send it up with what you need — pricing, a specialist, or a sponsor."},
    ),
    "forecast": (
        {"key": "keep", "label": "Keep the forecast", "status": "Closed",
         "needsReason": True,
         "meaning": "Say what makes you confident, so the call is on the record."},
        {"key": "change_date", "label": "Move the close date", "status": "Actioned",
         "needsReason": True, "meaning": "Set a date you believe and say why it moved."},
        {"key": "change_category", "label": "Change the category", "status": "Actioned",
         "needsReason": True, "meaning": "Move it out of commit, or into it, with a reason."},
    ),
    "coach": (
        {"key": "coach", "label": "Have the conversation", "status": "Actioned",
         "needsReason": False,
         "meaning": "Book the coaching session with the example deals attached."},
        {"key": "pattern", "label": "Watch the pattern", "status": "Monitoring",
         "needsReason": False,
         "meaning": "One quarter is not a habit. Keep it visible until it repeats."},
        {"key": "ops", "label": "Ask Sales Ops to check the data", "status": "Actioned",
         "needsReason": False,
         "meaning": "Some of these are process or entry problems, not behaviour."},
    ),
    "coverage": (
        {"key": "generate", "label": "Build pipeline here", "status": "Actioned",
         "needsReason": False,
         "meaning": "Assign new-pipeline work to this area with an owner and a date."},
        {"key": "requalify", "label": "Re-qualify what is there", "status": "Actioned",
         "needsReason": False,
         "meaning": "The pipeline may exist but not be real. Check it before adding more."},
        {"key": "rephase", "label": "Re-phase the target", "status": "Actioned",
         "needsReason": True,
         "meaning": "Decide the plan is wrong rather than the pipeline, and say why."},
    ),
    "concentration": (
        {"key": "diversify", "label": "Spread the risk", "status": "Actioned",
         "needsReason": False,
         "meaning": "Commission a plan to build value outside this exposure."},
        {"key": "protect", "label": "Protect it", "status": "Actioned",
         "needsReason": False,
         "meaning": "Name a retention owner and a cadence for the exposure itself."},
        {"key": "plan", "label": "Ask for a coverage plan", "status": "Actioned",
         "needsReason": False,
         "meaning": "Send it to the manager who owns the area, with a due date."},
    ),
    "growth": (
        {"key": "pursue", "label": "Go after it", "status": "Actioned",
         "needsReason": False,
         "meaning": "Assign it to the account owner with a target date."},
        {"key": "play", "label": "Make it a play", "status": "Actioned",
         "needsReason": False,
         "meaning": "Group every account missing the same thing into one campaign."},
        {"key": "later", "label": "Save for later", "status": "Monitoring",
         "needsReason": False,
         "meaning": "Keep it on the list for the next account planning round."},
    ),
}


def _due(days: int) -> str:
    """A due date, as a date rather than as 'in 5 days'."""
    return (AS_OF + timedelta(days=max(days, 1))).isoformat()


@dataclass(frozen=True)
class Card:
    key: str
    persona: str
    entity_type: str
    entity_id: str
    entity_label: str
    headline: str
    why: list[str]
    predicate: str
    owner: str
    next_step: str
    severity: float          # 0-100, what it is worth
    urgency: float           # 0-1, how soon
    value_at_stake: float
    framing: str             # "risk" | "opportunity"
    #: What sort of decision this is, stable across slices: "deal" (one deal
    #: or one account's deals), "rep_pattern" (a person's habit), "coverage"
    #: (plan against pipeline), "concentration" (one exposure too large), or
    #: "growth" (an attach or cross-sell play). A page ranks by it — see
    #: `PAGE_FOCUS` — so the rail answers the page's question rather than
    #: showing every page the same top eight.
    kind: str = "deal"
    scope_to: dict | None = None
    drill: str | None = None
    #: The choices this card offers BEYOND watch/dismiss. Two or more, each with
    #: the status it moves the card into. The client's spec requires that a
    #: decision card never be a dead end.
    options: tuple[dict, ...] = ()
    #: Days from AS_OF by which the decision stops being useful.
    due_days: int = 7

    def as_dict(self, priority: float) -> dict:
        urgency_label = ("Critical" if priority >= 80 else "High" if priority >= 60
                         else "Medium" if priority >= 40 else "Low")
        return {
            "key": self.key,
            "persona": self.persona,
            "entity": {"type": self.entity_type, "id": self.entity_id,
                       "label": self.entity_label},
            "headline": self.headline,
            "why": self.why,
            "predicate": self.predicate,
            "owner": self.owner,
            "nextStep": self.next_step,
            "severity": round(self.severity, 1),
            "urgency": round(self.urgency, 3),
            "priority": round(priority, 1),
            "urgencyLabel": urgency_label,
            "valueAtStake": round(self.value_at_stake, 2),
            "framing": self.framing,
            "kind": self.kind,
            "scopeTo": self.scope_to,
            "drill": self.drill,
            # --- the fields the client's decision-card spec requires ---------
            # Every card must offer two or more real choices, name an owner and
            # a date, and start in a known status. Without these a "decision
            # card" is a notification with a longer sentence.
            "options": [*self.options, *UNIVERSAL_OPTIONS],
            "dueDate": _due(self.due_days),
            "dueInDays": max(self.due_days, 1),
            "status": "New",
            "statuses": list(STATUSES),
            "confidenceLabel": urgency_label if self.framing == "risk" else "Opportunity",
        }


def _rank(cards: list[Card], persona: str) -> list[dict]:
    """Persona-specific priority, then a stable sort."""
    out = []
    for c in cards:
        if persona == "ae":
            p = 0.35 * c.severity + 0.65 * (c.urgency * 100)
        elif persona == "manager":
            p = 0.60 * c.severity + 0.40 * (c.urgency * 100)
        else:
            p = 0.70 * c.severity + 0.30 * (c.urgency * 100)
        out.append(c.as_dict(min(p, 100)))
    out.sort(key=lambda d: -d["priority"])
    return out


# --------------------------------------------------------------------------- #
# Account Executive
# --------------------------------------------------------------------------- #


def _ae_cards(fs: FilterState, principal: Principal) -> list[Card]:
    me = principal.identity
    risk = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    risk = risk[risk["opportunity_code"].isin(codes)]
    cards: list[Card] = []

    # Group first, because the unit of work is a CALL, not a record. Brian
    # Thompson's book is 37 open deals across a handful of accounts and eight of
    # his top nine flagged deals sit at one customer — nine separate cards there
    # is one phone call written out nine times. Where an account carries several
    # flagged deals the rep gets one card for the account and the deal list
    # inside it; single-deal accounts still get their own card.
    multi = (
        risk.groupby("account_name")
        .filter(lambda g: len(g) >= 2)
        .groupby("account_name")
    )
    grouped_accounts: set[str] = set()
    for account, g in multi:
        g = g.sort_values("risk_score", ascending=False)
        grouped_accounts.add(str(account))
        gp = float(g["acv_gp"].sum())
        worst = g.iloc[0]
        quiet = int(g["quiet_days"].max()) if g["quiet_days"].notna().any() else 0
        overdue = int(g["is_past_due"].sum())
        cards.append(Card(
            key=f"ae_account_{account}", persona="ae", kind="deal",
            options=OPTIONS["deal_review"], due_days=3,
            entity_type="Account", entity_id=str(account), entity_label=str(account),
            headline=f"{account} — {len(g)} deals need a decision",
            why=[f"{money(gp)} of gross profit across {len(g)} open opportunities, "
                 f"{overdue} of them past their own close date.",
                 f"The longest has had nothing logged for {quiet} days; the worst "
                 f"scores {int(worst['risk_score'])} on deal risk.",
                 "One conversation covers all of them."],
            predicate=f"account has >= 2 open opportunities with a firing risk factor",
            owner=me,
            next_step=f"Book one call with {account} and walk the whole list — "
                      f"re-date what is real, close what is not.",
            severity=float(g["risk_score"].max()),
            urgency=min((quiet / (STALL_DAYS * 2)) * 0.5 + (overdue / len(g)) * 0.5, 1.0),
            value_at_stake=gp, framing="risk",
            scope_to={"dim": "account", "value": str(account)},
            drill=f"account:{worst['account_code'] if 'account_code' in g.columns else account}",
        ))

    singles = risk[~risk["account_name"].isin(grouped_accounts)]
    for r in singles.head(20).itertuples(index=False):
        drivers = r.risk_factors or []
        keys = {d["key"] for d in drivers}
        gp = float(r.acv_gp)

        # The headline carries the DISTINGUISHING fact, not the category. A
        # worklist where nine cards all read "Confirm or close this deal" is a
        # worklist the rep scrolls past; the number is what makes each one a
        # different job.
        quiet = int(r.quiet_days) if pd.notna(r.quiet_days) else 0
        overdue = int(r.days_past_due or 0)
        who = str(r.account_name)

        if "stalled" in keys and "past_due" in keys:
            head = f"{who} — silent {quiet} days and {overdue} days past close"
            step = ("Call the account today. Either move the stage with a new close "
                    "date and a reason, or mark it Closed Lost.")
        elif "stalled" in keys:
            head = f"{who} — nothing logged in {quiet} days"
            step = "Call the account and log what you learn, even if the answer is no."
        elif "past_due" in keys:
            head = f"{who} — {overdue} days past its own close date"
            step = "Re-date it with a reason, or close it."
        elif "shrinking" in keys:
            drift = next((d for d in drivers if d["key"] == "shrinking"), None)
            head = f"{who} — value has fallen since you logged it"
            step = "Re-qualify the scope before it reaches the quarterly roll-up."
            if drift:
                head = f"{who} — {drift['detail'].split('down ')[-1].rstrip('.')} down"
        elif "stage_skip" in keys:
            head = f"{who} — {int(r.skipped_stages)} stage(s) have no record"
            step = "Backfill the qualification you skipped, or expect a surprise late."
        elif "thin_margin" in keys:
            head = f"{who} — margin below the floor for {r.portfolio}"
            step = "Price check with the deal desk before you submit."
        elif "slipped" in keys:
            head = f"{who} — close date pushed {int(r.slip_days)} days"
            step = "Confirm the new date is committed rather than the deal drifting."
        elif drivers:
            head = f"{who} — {drivers[0]['label'].lower()}"
            step = "Review this deal before the weekly pipeline call."
        else:
            continue

        urgency = min(
            (float(r.quiet_days or 0) / (STALL_DAYS * 2)) * 0.5
            + (float(r.days_past_due or 0) / 60) * 0.5, 1.0)
        cards.append(Card(
            key=f"ae_deal_{r.opportunity_code}", persona="ae", kind="deal",
            options=OPTIONS["deal_review"], due_days=3,
            entity_type="Opportunity", entity_id=r.opportunity_code,
            entity_label=r.opportunity_name,
            headline=head,
            why=[d["detail"] for d in drivers[:3]],
            predicate=f"risk_score = {int(r.risk_score)} from "
                      f"{', '.join(sorted(keys))}",
            owner=me, next_step=step,
            severity=float(r.risk_score), urgency=urgency,
            value_at_stake=gp, framing="risk",
            scope_to={"dim": "account", "value": r.account_name},
            drill=f"deal:{r.opportunity_code}",
        ))

    # Cross-sell on accounts this rep already owns — framed as upside, in green.
    #
    # The data-science recommendations come FIRST and the computed whitespace
    # fills in behind them, because a recommendation carrying a written reason
    # and a named peer account is a better conversation opener than "they buy
    # three of four lines". Accounts already covered by a recommendation are
    # skipped below rather than carded twice.
    recommended: set[str] = set()
    for x in XS.unified(fs, principal, limit=6):
        recommended.add(x["accountCode"])
        reason = x["reasons"][0]["text"] if x["reasons"] else ""
        cards.append(Card(
            key=f"ae_xsell_{x['id']}", persona="ae", kind="growth",
            options=OPTIONS["growth"], due_days=21,
            entity_type="Account", entity_id=x["accountCode"],
            entity_label=x["accountName"],
            headline=f"{x['accountName']} could take {x['offering']}",
            why=[reason or x["confidenceMeaning"],
                 f"{x['confidence']} confidence — {x['confidenceMeaning']}",
                 f"Peer accounts earn about {money(x['peerGp'])} of gross profit on "
                 f"this; they already spend {money(x['accountGp'])} with us.",
                 XS.CAVEAT],
            predicate=f"cross-sell engine: {', '.join(x['methods'])}",
            owner=me,
            next_step=f"Open a {x['offering']} conversation at {x['accountName']}.",
            severity=min(45 + x["confidenceRank"] * 12, 85),
            urgency=0.25, value_at_stake=float(x["peerGp"]),
            framing="opportunity",
            scope_to={"dim": "account", "value": x["accountName"]},
            drill=f"account:{x['accountCode']}",
        ))

    for w in ACC.whitespace(fs, principal, limit=6):
        if w["accountCode"] in recommended:
            continue
        cards.append(Card(
            key=f"ae_whitespace_{w['accountCode']}", persona="ae", kind="growth",
            options=OPTIONS["growth"], due_days=21,
            entity_type="Account", entity_id=w["accountCode"],
            entity_label=w["accountName"],
            headline=f"{w['accountName']} buys only {len(w['holds'])} of four lines",
            why=[w["method"],
                 f"Currently {money(w['gp'])} of gross profit across "
                 f"{', '.join(w['holds'])}."],
            predicate=f"account holds {len(w['holds'])} of 4 LOBs and "
                      f"GP >= {money(ACC.MATERIAL_ACCOUNT_GP)}",
            owner=me,
            next_step=f"Open a {w['recommendedLob']} conversation at this account.",
            severity=min(40 + w["peerAttachRate"] * 50, 85),
            urgency=0.25, value_at_stake=float(w["estimatedGp"]),
            framing="opportunity",
            scope_to={"dim": "account", "value": w["accountName"]},
            drill=f"account:{w['accountCode']}",
        ))
    return cards


# --------------------------------------------------------------------------- #
# Sales Manager
# --------------------------------------------------------------------------- #


def _manager_cards(fs: FilterState, principal: Principal) -> list[Card]:
    members = set(principal.predicate.get("owner", ()))
    reps = rep_behaviour()
    pod = reps[reps["rep"].isin(members)] if members else reps
    manager = "You"
    cards: list[Card] = []

    patterns = [
        ("shrink_rate", "value shrinkage",
         "Review how this rep sizes and qualifies at the outset.",
         "Their deals shrink more often than their peers'"),
        ("inflate_rate", "sandbagging",
         "Coach on early sizing — they may be under-forecasting consistently.",
         "Their deals grow well beyond the first estimate more often than peers'"),
        ("regression_rate", "forecast reversal",
         "Discount their confidence calls and review forecasting habits directly.",
         "They walk forecasts backwards more often than their peers"),
        ("stall_rate", "stalled book",
         "Work their open list together and close out what is not real.",
         "More of their open book has stopped moving than their peers'"),
    ]
    for col, label, step, why_lead in patterns:
        z_col = f"{col}_z"
        for r in pod[(~pod["thin"]) & (pod[z_col] >= 1.3)].itertuples(index=False):
            rate = float(getattr(r, col) or 0)
            peer = float(getattr(r, f"{col}_peer") or 0)
            z = float(getattr(r, z_col))
            cards.append(Card(
                key=f"mgr_{col}_{r.rep}", persona="manager", kind="rep_pattern",
                options=OPTIONS["coach"], due_days=7,
                entity_type="Rep", entity_id=r.rep, entity_label=r.rep,
                headline=f"Coach {r.rep} on {label}",
                why=[f"{why_lead}: {pct(rate * 100, 0)} of {int(r.deals)} owned "
                     f"opportunities against a {pct(peer * 100, 0)} peer norm "
                     f"({z:+.1f} SD).",
                     f"{int(r.open_deals)} open deals carrying {money(r.open_gp)} "
                     f"of gross profit."],
                predicate=f"{col} z-score >= 1.3 over >= 8 owned opportunities",
                owner=manager, next_step=step,
                severity=min(55 + 12 * z, 96), urgency=min(z / 4, 0.7),
                value_at_stake=float(r.open_gp), framing="risk",
                scope_to={"dim": "rep", "value": r.rep},
                drill=f"rep:{r.rep}",
            ))

    # Sandbagging is framed as upside in the reference guide, not as a fault.
    for c in cards:
        if c.key.startswith("mgr_inflate_rate"):
            object.__setattr__(c, "framing", "opportunity")

    # Pod-level pipeline-generation risk.
    for r in pod[(~pod["thin"]) & (pod["new_business_share"] <= 0.25)].itertuples(index=False):
        cards.append(Card(
            key=f"mgr_newbiz_{r.rep}", persona="manager", kind="rep_pattern",
            options=OPTIONS["coach"], due_days=7,
            entity_type="Rep", entity_id=r.rep, entity_label=r.rep,
            headline=f"{r.rep}'s book is almost all renewals",
            why=[f"Only {pct(r.new_business_share * 100, 0)} of "
                 f"{int(r.deals)} owned opportunities are New Business, against a "
                 f"{pct(r.new_business_share_peer * 100, 0)} peer norm."],
            predicate="new_business_share <= 25% over >= 8 owned opportunities",
            owner=manager,
            next_step="Check their pipeline-generation activity — this surfaces too "
                      "late to correct if it is left.",
            severity=78, urgency=0.35,
            value_at_stake=float(r.open_gp), framing="risk",
            scope_to={"dim": "rep", "value": r.rep}, drill=f"rep:{r.rep}",
        ))

    # The pod's own stalled worklist, as one card rather than forty.
    risk = P.risk_table()
    codes = set(slice_frame(fs, principal)["opportunity_code"])
    stalled = risk[(risk["opportunity_code"].isin(codes))
                   & (risk["quiet_days"] >= STALL_DAYS)]
    if len(stalled):
        cards.append(Card(
            key="mgr_stalled_book", persona="manager", kind="deal",
            options=OPTIONS["forecast"], due_days=5,
            entity_type="Segment", entity_id="stalled", entity_label="Stalled pipeline",
            headline=f"{len(stalled)} open deals in this pod have stopped moving",
            why=[f"{money(stalled['acv_gp'].sum())} of gross profit with no logged "
                 f"change in {STALL_DAYS}+ days.",
                 f"Median silence is {int(stalled['quiet_days'].median())} days; the "
                 f"worst is {int(stalled['quiet_days'].max())}."],
            predicate=f"open AND quiet_days >= {STALL_DAYS}",
            owner=manager,
            next_step="Run these as one scrub in the weekly pipeline call rather than "
                      "chasing them individually.",
            severity=72, urgency=0.5,
            value_at_stake=float(stalled["acv_gp"].sum()), framing="risk",
            scope_to={"dim": "riskBand", "value": "High"},
        ))

    # A theme is a briefing; a list of accounts is homework. The manager card
    # exists because the same offering missing at five accounts owned by five
    # people is one decision, not five.
    for t in XS.themes(fs, principal)[:3]:
        cards.append(Card(
            key=f"mgr_xsell_{t['offering']}", persona="manager", kind="growth",
            options=OPTIONS["growth"], due_days=21,
            entity_type="Segment", entity_id=t["offering"], entity_label=t["offering"],
            headline=f"{t['accounts']} customers could take {t['offering']}",
            why=[f"Spread across {t['ownerCount']} owners, so it is one briefing "
                 f"rather than {t['accounts']} separate conversations.",
                 f"Strongest is {t['strongest']['accountName']} "
                 f"({t['strongest']['confidence'].lower()} confidence), owned by "
                 f"{t['strongest']['owner']}.",
                 f"Peer accounts earn about {money(t['estimatedGp'])} of gross profit "
                 f"across these.",
                 XS.CAVEAT],
            predicate="same offering recommended at 2 or more accounts in this pod",
            owner=principal.identity_label.replace("'s team", ""),
            next_step=f"Brief the {t['ownerCount']} owners together and give each a date.",
            severity=min(48 + t["accounts"] * 3, 82), urgency=0.25,
            value_at_stake=float(t["estimatedGp"]),
            framing="opportunity",
        ))
    return cards


# --------------------------------------------------------------------------- #
# Executive
# --------------------------------------------------------------------------- #


def _exec_cards(fs: FilterState, principal: Principal) -> list[Card]:
    cards: list[Card] = []
    totals = B.totals(fs, principal)

    # Coverage holes — a target with nothing behind it.
    #
    # Measured FORWARD, over the window `budget.forward_window` decides (the
    # current quarter onward, or the one quarter a filter names), and read
    # from the same LOB × portfolio cells the performance grid draws, so this
    # card, the grid and the actions-page bullets cannot disagree about a
    # line. Read against the current quarter alone this found nothing,
    # because that quarter's plan is already delivered — while the next one
    # sat at 0.18x cover with no cell-level card to say where. A decision
    # about pipeline generation is about the quarters still to come.
    #
    # Two exclusions, both of which produced a wrong card before they were
    # added. A cell whose plan is already fully delivered has NO remaining
    # target, so its status is "Delivered", not "Uncovered" — without this
    # test a cell sitting on $50K of open pipeline was being reported as having
    # none. And a cell is only worth an executive's attention if the money is
    # material; a $3K shortfall on a $2.4M plan is a rounding line, not a
    # decision.
    MATERIAL_CELL_GP = 25_000
    grid = B.coverage_grid(fs, principal)
    window = grid["window"]
    holes = [{"lob": c["lob"], "portfolio": c["portfolio"], "remainingGp": c["remainingGp"],
              "openGp": c["openGp"], "coverage": c["coverage"]}
             for c in grid["cells"]
             if c["remainingGp"] >= MATERIAL_CELL_GP and c["status"] == "Uncovered"]
    holes.sort(key=lambda c: -c["remainingGp"])
    for c in holes[:4]:
        cov = c["coverage"]
        cards.append(Card(
            key=f"exec_cov_{c['lob']}_{c['portfolio']}", persona="executive", kind="coverage",
            options=OPTIONS["coverage"], due_days=14,
            entity_type="Segment",
            entity_id=f"{c['lob']}|{c['portfolio']}",
            entity_label=f"{c['lob']} · {c['portfolio']}",
            headline=f"Move coverage into {c['lob']} {c['portfolio']}",
            why=[f"{money(c['remainingGp'])} of plan remains from {window} "
                 f"against {money(c['openGp'])} of open pipeline"
                 + (" — no pipeline at all." if c["openGp"] <= 0
                    else f" — {cov:.2f}x coverage."),
                 "A target with no pipeline behind it is a decision, not a forecast."],
            predicate=f"open GP / remaining plan GP < 0.5 for this cell, {window}",
            owner=f"{c['lob']} line-of-business lead",
            next_step="Direct pipeline-generation into this cell this quarter, or "
                      "re-phase the target.",
            severity=min(60 + 40 * (c["remainingGp"] / max(totals["budgetGp"], 1)), 98),
            urgency=0.6, value_at_stake=float(c["remainingGp"]), framing="risk",
            scope_to={"dim": "lob", "value": c["lob"]},
        ))

    # Whole-quarter coverage.
    q = next((r for r in B.by_quarter(fs, principal) if r["quarter"] == CUR_QUARTER), None)
    fwd = [r for r in B.by_quarter(fs, principal) if r["isFuture"]]
    for r in fwd:
        if r["coverage"] is not None and r["coverage"] < 1.0:
            cards.append(Card(
                key=f"exec_quarter_{r['quarter']}", persona="executive", kind="coverage",
                options=OPTIONS["coverage"], due_days=14,
                entity_type="Segment", entity_id=r["quarter"], entity_label=r["quarter"],
                headline=f"{r['quarter']} is covered {r['coverage']:.2f}x",
                why=[f"{money(r['remainingGp'])} of plan against {money(r['openGp'])} "
                     f"of open pipeline.",
                     "The quarter has not started, so this is a build decision rather "
                     "than a performance finding — but the build has to start now."],
                predicate="open GP / remaining plan GP < 1.0 for a future quarter",
                owner="Entity leader",
                next_step="Choose now between funding pipeline generation and "
                          "re-phasing the target.",
                severity=92, urgency=0.55,
                value_at_stake=float(r["remainingGp"]), framing="risk",
                scope_to={"dim": "quarter", "value": r["quarter"]},
            ))

    # Concentration, from the DS findings where they exist.
    conc = ACC.concentration(fs, principal)
    if conc["accounts"] and conc["topAccountShare"] >= 10:
        a = conc["accounts"][0]
        cards.append(Card(
            key="exec_whale", persona="executive", kind="concentration", entity_type="Account",
            options=OPTIONS["concentration"], due_days=30,
            entity_id=a["account_code"], entity_label=a["account_name"],
            headline=f"{a['account_name']} is {conc['topAccountShare']:.1f}% of the book",
            why=[f"{money(a['gp'])} of gross profit across {int(a['opportunities'])} "
                 f"opportunities.",
                 f"The top five accounts together carry {conc['top5AccountShare']:.0f}% "
                 f"of gross profit."],
            predicate="account GP share >= 10% of total GP",
            owner="Account leadership",
            next_step="Run a retention and diversification review on this relationship.",
            severity=96, urgency=0.3, value_at_stake=float(a["gp"]),
            framing="risk", scope_to={"dim": "account", "value": a["account_name"]},
            drill=f"account:{a['account_code']}",
        ))
    if conc["industries"] and conc["topIndustryShare"] >= 20:
        i = conc["industries"][0]
        cards.append(Card(
            key="exec_vertical", persona="executive", kind="concentration", entity_type="Industry",
            options=OPTIONS["concentration"], due_days=30,
            entity_id=i["industry"], entity_label=i["industry"],
            headline=f"{i['industry']} is {conc['topIndustryShare']:.1f}% of the book",
            why=[f"{money(i['gp'])} of gross profit across {int(i['accounts'])} accounts.",
                 "Concentrated exposure to one vertical's budget cycle."],
            predicate="industry GP share >= 20% of total GP",
            owner="Strategy",
            next_step="Assess exposure to this vertical's budget cycle and set a "
                      "diversification target.",
            severity=90, urgency=0.25, value_at_stake=float(i["gp"]),
            framing="risk", scope_to={"dim": "industry", "value": i["industry"]},
        ))

    # Key-person risk, computed rather than replayed.
    reps = rep_behaviour()
    open_total = float(reps["open_gp"].sum())
    for r in reps.nlargest(1, "open_gp").itertuples(index=False):
        share = 100 * float(r.open_gp) / open_total if open_total else 0
        if share >= 15:
            cards.append(Card(
                key=f"exec_keyperson_{r.rep}", persona="executive", kind="concentration", entity_type="Rep",
                options=OPTIONS["concentration"], due_days=30,
                entity_id=r.rep, entity_label=r.rep,
                headline=f"{r.rep} holds {share:.0f}% of open pipeline",
                why=[f"{money(r.open_gp)} of open gross profit across "
                     f"{int(r.open_deals)} deals.",
                     "Single-point dependency on one person's relationships."],
                predicate="rep open-pipeline GP share >= 15%",
                owner="Entity leader and pod manager",
                next_step="Succession and coverage planning for this book.",
                severity=94, urgency=0.3, value_at_stake=float(r.open_gp),
                framing="risk", scope_to={"dim": "rep", "value": r.rep},
            ))

    # The cross-sell programme, as one funded decision rather than 39 cards.
    ws = ACC.whitespace(fs, principal, limit=999)
    if ws:
        lv = ACC.lob_count_value()
        one = lv[lv["lob_count"] == 1]
        four = lv[lv["lob_count"] == 4]
        lift = ""
        if len(one) and len(four) and float(one["medianGp"].iloc[0]) > 0:
            x = float(four["medianGp"].iloc[0]) / float(one["medianGp"].iloc[0])
            lift = (f"Median account value is {money(four['medianGp'].iloc[0])} at four "
                    f"lines of business against {money(one['medianGp'].iloc[0])} at one "
                    f"— {x:.0f} times.")
        cards.append(Card(
            key="exec_whitespace_programme", persona="executive", kind="growth",
            options=OPTIONS["growth"], due_days=30,
            entity_type="Segment", entity_id="whitespace",
            entity_label="Cross-sell whitespace",
            headline=f"Fund an attach play across {len(ws)} accounts",
            why=[f"{len(ws)} accounts above {money(ACC.MATERIAL_ACCOUNT_GP)} of gross "
                 f"profit buy fewer than four lines of business.", lift or
                 "Accounts holding more lines of business are worth materially more."],
            predicate=f"account GP >= {money(ACC.MATERIAL_ACCOUNT_GP)} AND LOB count < 4",
            owner="Sales leadership",
            next_step="Stand up a named attach play with an owner and a quarterly target.",
            severity=74, urgency=0.2,
            value_at_stake=float(sum(w["estimatedGp"] for w in ws)),
            framing="opportunity",
        ))

    th = XS.themes(fs, principal)
    if th:
        t = th[0]
        cards.append(Card(
            key="exec_xsell_theme", persona="executive", kind="growth",
            options=OPTIONS["growth"], due_days=30,
            entity_type="Segment", entity_id=t["offering"],
            entity_label=t["offering"],
            headline=f"Launch a {t['offering']} play across {t['accounts']} accounts",
            why=[f"The same offering is missing at {t['accounts']} accounts across "
                 f"{len(t['industries'])} industries — the largest of {len(th)} "
                 f"repeating themes.",
                 f"It sits with {t['ownerCount']} owners today, so it needs one brief "
                 f"and one target list rather than {t['accounts']} account plans.",
                 f"Peer accounts earn about {money(t['estimatedGp'])} of gross profit "
                 f"on it.",
                 XS.CAVEAT],
            predicate="same offering recommended at 2 or more accounts entity-wide",
            owner="Sales leadership",
            next_step=f"Name a play owner and pilot with "
                      f"{t['strongest']['accountName'].rstrip('. ')}.",
            severity=76, urgency=0.22,
            value_at_stake=float(t["estimatedGp"]),
            framing="opportunity",
        ))
    return cards


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #



#: page -> which cards lead. A predicate over the card as sent, so the rule is
#: readable and testable rather than buried in a sort key.
#:
#: The home pages (my-day, pod-pulse, tldr) and the pages not listed keep the
#: mixed ranking: their question is "what first", and the answer to that is
#: whatever scores highest. Every other page asks something narrower, and the
#: rail is re-ordered — never re-invented — so the cards that answer it come
#: first and the rest back-fill in their usual order. A page therefore never
#: has fewer cards than it did; it has the same cards in the order its
#: question implies.
PAGE_FOCUS: dict[str, tuple[str, str]] = {
    # the executive's "what is wrong" and "what is it worth"
    "risks": ("framing", "risk"),
    # the three growth pages, one per persona
    "growth": ("framing", "opportunity"),
    "pod-whitespace": ("framing", "opportunity"),
    "my-accounts": ("framing", "opportunity"),
    # plan against pipeline
    "performance": ("kind", "coverage"),
    "actions": ("kind", "coverage"),
    # where the money sits, and where too much of it sits in one place
    "structure": ("kind", "concentration"),
    # one deal at a time
    "my-deals": ("kind", "deal"),
    # a person's habit, not a deal's
    "process": ("kind", "rep_pattern"),
    "calibration": ("kind", "rep_pattern"),
    "rep-benchmark": ("kind", "rep_pattern"),
}


def _focus(cards: list[dict], page: str | None) -> list[dict]:
    """Stable partition: the cards that answer the page first, then the rest."""
    rule = PAGE_FOCUS.get(page or "")
    if not rule:
        return cards
    field, want = rule
    lead = [c for c in cards if c.get(field) == want]
    rest = [c for c in cards if c.get(field) != want]
    if page == "risks":
        # "What needs fixing, AND WHAT IS IT WORTH" — the second half of the
        # question is the sort. The executive's mixed ranking is already
        # risk-first, so without this the risks page would repeat the brief.
        lead.sort(key=lambda c: -c["valueAtStake"])
    return lead + rest


def build(fs: FilterState, principal: Principal, limit: int = 12,
          page: str | None = None) -> list[dict]:
    """
    The ranked action list for this persona and slice.

    Executive cards are deduplicated to one per entity: 54 opportunities in this
    book carry two or more anomalies and the ML outlier restates a rule hit in 25
    of its 41 rows, so without a dedupe the same deal arrives three times and the
    list reads as noise.

    `page` re-orders the deduplicated list so the cards that answer that page's
    question lead (see `PAGE_FOCUS`); with no page the mixed ranking stands.
    """
    if principal.key == "ae":
        cards = _ae_cards(fs, principal)
    elif principal.key == "manager":
        cards = _manager_cards(fs, principal)
    else:
        cards = _exec_cards(fs, principal)

    ranked = _rank(cards, principal.key)
    seen: set[str] = set()
    out: list[dict] = []
    for c in ranked:
        eid = f"{c['entity']['type']}:{c['entity']['id']}"
        if eid in seen:
            continue
        seen.add(eid)
        out.append(c)
    out = _focus(out, page)[:limit]
    for i, c in enumerate(out, 1):
        c["n"] = i
    return out
