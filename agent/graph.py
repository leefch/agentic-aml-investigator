"""
The multi-agent investigation graph.

Flow:
    planner -> triage -> (fast path) narrative
                      -> (deep path) investigator -> narrative -> reviewer -> END

Each node is one "agent" with a narrow job. The orchestration - who runs when,
and whether we take the fast or deep path - is the graph itself, which is the
whole reason for using LangGraph over a pile of if-statements: the routing is
explicit and inspectable.

Every node appends to state["audit"], so after a run you can read exactly what
happened and why. That ledger is what makes this defensible in a review.
"""
import datetime as dt

from langgraph.graph import StateGraph, END

from config import HUMAN_REVIEW_FLOOR, MED_RISK
from agent.state import CaseState
from agent import tools
from agent.llm import ask, ask_json


def _log(state: CaseState, actor: str, action: str, detail=""):
    state.setdefault("audit", []).append({
        "ts": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "actor": actor,
        "action": action,
        "detail": detail,
    })


# --- node 1: planner ---------------------------------------------------------
def planner_node(state: CaseState) -> dict:
    """Orchestrator. Looks at the raw transaction and decides which steps this
    case needs. We let the LLM plan, but fall back to a sane default plan if it
    misbehaves - the graph downstream doesn't actually depend on the plan text,
    it's there to show reasoning and to log intent."""
    txn = state["txn"]
    default_plan = ["triage", "investigate", "narrate", "review"]

    plan = ask_json(
        system=(
            "You are the orchestrator of an AML investigation. Given one "
            "transaction, list the steps to investigate it. Choose from: "
            "triage, investigate, narrate, review."
        ),
        user=f"Transaction: {txn}\nReturn a JSON list of step names in order.",
        fallback=default_plan,
    )
    if not isinstance(plan, list) or not plan:
        plan = default_plan

    _log(state, "planner", "built plan", plan)
    return {"plan": plan}


# --- node 2: triage ----------------------------------------------------------
def triage_node(state: CaseState) -> dict:
    """Calls the ML scorer tool and bands the risk. No LLM here - this is a
    deterministic model call, and we want it that way for auditability."""
    result = tools.score_transaction(state["txn"])
    _log(state, "triage", "scored txn",
         f"score={result['score']} tier={result['tier']}")
    return {
        "risk_score": result["score"],
        "risk_tier": result["tier"],
        "score_drivers": result["drivers"],
    }


def route_after_triage(state: CaseState) -> str:
    """Low-risk cases skip the deep dive - that's the 'plan' actually changing
    the path taken, not just decoration. Medium/high get the full workup."""
    if state["risk_score"] < MED_RISK:
        return "fast"
    return "deep"


# --- node 3: investigator ----------------------------------------------------
def investigator_node(state: CaseState) -> dict:
    """The tool user. Pulls history, checks the watchlist on both parties, and
    walks one hop of the transaction network. Collects everything into evidence
    so the narrative agent has real facts to write from."""
    txn = state["txn"]
    sender, receiver = txn["sender_id"], txn["receiver_id"]

    hits = tools.check_watchlist([sender, receiver])
    links = tools.linked_accounts(sender)
    hist = tools.account_history(sender, limit=5)

    evidence = []
    evidence.append({"source": "score_drivers", "detail": state.get("score_drivers", [])})
    evidence.append({"source": "sender_recent_activity", "detail": hist})
    if hits:
        evidence.append({"source": "watchlist", "detail": hits})
    if links:
        evidence.append({"source": "linked_accounts", "detail": links})

    _log(state, "investigator", "gathered evidence",
         f"watchlist_hits={len(hits)} linked={len(links)}")
    return {
        "evidence": evidence,
        "watchlist_hits": hits,
        "linked_accounts": links,
    }


# --- node 4: narrative -------------------------------------------------------
def narrative_node(state: CaseState) -> dict:
    """Writes the case summary in plain English, grounded ONLY in the evidence
    collected. We tell the model in no uncertain terms not to invent facts -
    the reviewer node checks that it listened."""
    txn = state["txn"]
    evidence = state.get("evidence", [{"source": "score", "detail": state.get("score_drivers")}])

    narrative = ask(
        system=(
            "You are a financial-crime analyst writing a concise case note. "
            "Use ONLY the evidence provided. Do not speculate or add facts that "
            "aren't in the evidence. 4-6 sentences. State what happened, why it "
            "was flagged, and what the evidence does and does not show."
        ),
        user=f"Transaction: {txn}\nRisk tier: {state.get('risk_tier')}\n"
             f"Evidence: {evidence}",
    )
    _log(state, "narrative", "wrote case note", f"{len(narrative)} chars")
    return {"narrative": narrative}


# --- node 5: reviewer + human-review rail ------------------------------------
def reviewer_node(state: CaseState) -> dict:
    """Critic agent. Checks the narrative against the evidence for unsupported
    claims and proposes a recommendation. Then the hard rail runs: anything at
    or above the human-review floor is forced to a person no matter what the
    reviewer decided. The agents can escalate but can never auto-clear a
    high-risk case."""
    review = ask_json(
        system=(
            "You review an AML case note for an unsupported-claims problem and "
            "propose an action. Actions: 'clear', 'escalate', 'file_sar'. "
            "Judge only whether the narrative is supported by the evidence."
        ),
        user=f"Narrative: {state.get('narrative')}\nEvidence: {state.get('evidence')}",
        fallback={"supported": True, "notes": "auto-review unavailable",
                  "recommendation": "escalate"},
    )

    notes = review.get("notes", "")
    rec = review.get("recommendation", "escalate")

    # the rail: high score always needs a human, full stop.
    needs_human = state["risk_score"] >= HUMAN_REVIEW_FLOOR
    if needs_human and rec == "clear":
        # reviewer wanted to clear a high-risk case - override and record it
        rec = "escalate"
        notes = (notes + " | OVERRIDE: score above human-review floor, "
                          "cannot auto-clear.").strip(" |")
        _log(state, "guardrail", "blocked auto-clear",
             f"score={state['risk_score']} >= floor={HUMAN_REVIEW_FLOOR}")

    _log(state, "reviewer", "reviewed case",
         f"recommendation={rec} needs_human={needs_human}")
    return {"review_notes": notes, "recommendation": rec, "needs_human": needs_human}


def build_graph():
    g = StateGraph(CaseState)
    g.add_node("planner", planner_node)
    g.add_node("triage", triage_node)
    g.add_node("investigator", investigator_node)
    g.add_node("narrative", narrative_node)
    g.add_node("reviewer", reviewer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "triage")
    # the branch: fast path clears low-risk quickly, deep path does the workup
    g.add_conditional_edges("triage", route_after_triage,
                            {"fast": "narrative", "deep": "investigator"})
    g.add_edge("investigator", "narrative")
    g.add_edge("narrative", "reviewer")
    g.add_edge("reviewer", END)
    return g.compile()


def investigate(txn_id: str) -> CaseState:
    """Convenience entry point used by the CLI and the UI."""
    txn = tools.get_transaction(txn_id)
    if not txn:
        raise ValueError(f"no such transaction: {txn_id}")

    app = build_graph()
    init: CaseState = {"txn_id": txn_id, "txn": txn, "audit": []}
    return app.invoke(init)
