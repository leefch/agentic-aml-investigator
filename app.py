"""
Streamlit front end for the investigator. The review queue is scored up front
and shown highest-risk first, so the cases that matter are at the top. Pick one,
run the agents, and see the risk tier, evidence, case note, the reviewer's call,
and the full audit trail.

    streamlit run app.py
"""
import streamlit as st

from agent import tools
from agent.graph import investigate

st.set_page_config(page_title="Transaction Investigator", layout="wide")

st.title("Agentic Transaction Investigator")
st.caption("Highest-risk cases first. Investigate to run the multi-agent workflow. "
           "Synthetic data only - nothing here is real.")

TIER_TAG = {"high": "[HIGH]", "medium": "[MED] ", "low": "[LOW] "}


@st.cache_data(show_spinner="Scoring the review queue...")
def ranked_queue(n: int = 25):
    """Score every queued transaction once and return them worst-first. Cached
    so we don't re-score on every interaction - clear the cache (or restart) if
    you regenerate the data."""
    scored = []
    for tid in tools.sample_queue(60):
        r = tools.score_transaction(tools.get_transaction(tid))
        scored.append({"txn_id": tid, "score": r["score"], "tier": r["tier"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n]


ranked = ranked_queue()
if not ranked:
    st.error("No data yet. Run `python data/generate_data.py` then "
             "`python models/train_scorer.py`.")
    st.stop()

# label each option with its tier + score so risk is visible before picking.
# dict preserves order, so the first (riskiest) case is selected by default.
options = {
    f"{TIER_TAG[r['tier']]}  {r['txn_id']}   score {r['score']:.2f}": r["txn_id"]
    for r in ranked
}

col_pick, col_btn = st.columns([3, 1])
with col_pick:
    label = st.selectbox("Cases for review (highest risk first)", list(options.keys()))
    txn_id = options[label]
with col_btn:
    st.write("")
    st.write("")
    go = st.button("Investigate", type="primary", use_container_width=True)

# show the raw txn so the analyst has context before running anything
txn = tools.get_transaction(txn_id)
with st.expander("Raw transaction", expanded=False):
    st.json(txn)

if go:
    with st.spinner("Agents working the case..."):
        result = investigate(txn_id)

    tier = result["risk_tier"]
    color = {"high": "red", "medium": "orange", "low": "green"}.get(tier, "gray")

    top = st.columns(3)
    top[0].metric("Risk score", f"{result['risk_score']:.2f}")
    top[1].markdown(f"**Tier**  :{color}[{tier.upper()}]")
    top[2].markdown(f"**Recommendation**  {result['recommendation']}")

    # make the human-review outcome impossible to miss - it's the whole point
    if result["needs_human"]:
        st.error("Needs human review - this case cannot be auto-cleared.")
    else:
        st.success("Cleared by the agents - no human review required.")

    st.subheader("Case note")
    st.write(result.get("narrative", ""))

    if result.get("review_notes"):
        st.subheader("Reviewer notes")
        st.info(result["review_notes"])

    left, right = st.columns(2)
    with left:
        st.subheader("Evidence")
        for e in result.get("evidence", []):
            st.markdown(f"**{e['source']}**")
            st.write(e["detail"])
    with right:
        st.subheader("Audit trail")
        for step in result["audit"]:
            st.text(f"[{step['actor']}] {step['action']}\n   {step['detail']}")