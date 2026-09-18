from typing import TypedDict, Any


class CaseState(TypedDict, total=False):
    """State that flows through the graph. Each node reads what it needs and
    writes back a partial update. total=False so nodes don't have to populate
    every key up front."""

    txn_id: str
    txn: dict

    # planner output - the ordered steps the orchestrator decided to run
    plan: list

    # triage output
    risk_score: float
    risk_tier: str          # low / medium / high
    score_drivers: list

    # investigation findings
    evidence: list          # list of {source, detail} the agents gathered
    watchlist_hits: list
    linked_accounts: list

    # write-up + review
    narrative: str
    review_notes: str
    recommendation: str     # e.g. "file SAR", "clear", "escalate"
    needs_human: bool

    # append-only decision ledger - every node drops a line here so the whole
    # case is replayable after the fact. this is the governance backbone.
    audit: list
