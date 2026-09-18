A multi-agent AI system that triages, investigates, and writes up suspicious financial transactions — combining classic machine learning with an agentic LLM workflow, human-in-the-loop guardrails, and a full decision audit trail.
Built with Python, LangGraph, Claude, and scikit-learn. All data is synthetic — nothing here uses real customer information.
What it does? Given a flagged transaction, a LangGraph orchestrator coordinates five specialist agents that reason about the case, call tools (including a trained ML model), gather evidence, write a plain-English case note, and decide whether it can be cleared or must go to a human reviewer.

The design deliberately mixes two kinds of components:
Deterministic ML / rules for the parts that must be reproducible and auditable (risk scoring, evidence gathering, the human-review guardrail).
LLM agents for judgment and language (planning, narrative writing, reviewing).
Agent	Role	Uses
Planner	Reasons about the investigation approach and logs intent	(Claude)
Triage	Scores risk and bands it low / medium / high	(ML risk scorer)
Investigate	Gathers evidence (watchlist, history, linked accounts)	(Investigation tools)
Narrate	Writes a grounded, evidence-only case note	(Claude)
Review	Checks the note for unsupported claims; enforces the guardrail	(Claude + rule)

Conditional branch: low-risk cases take a fast path and skip the investigator; medium/high-risk cases take the full deep-path workup.

Guardrail: any case scoring at or above the human-review floor cannot be auto-cleared — the LLM reviewer can recommend, but a deterministic rule makes the final call, and every override is written to the audit ledger.
Agent	Role	Uses
Planner--	Reasons about the investigation approach and logs intent	Claude
Triage--	Scores risk and bands it low / medium / high	ML risk scorer
Investigate--	Gathers evidence (watchlist, history, linked accounts)	Investigation tools
Narrate--	Writes a grounded, evidence-only case note	Claude
Review--	Checks the note for unsupported claims; enforces the guardrail	Claude + rule

Key features
Multi-agent orchestration with LangGraph (explicit, inspectable routing)
Trained ML anomaly scorer — Isolation Forest + logistic-regression ensemble, exposed as a tool the agents call, with driver-level explainability
Tool use — agents act on data through well-defined, testable tool functions
Human-in-the-loop guardrail — high-risk cases are forced to human review
Full audit trail — every agent action is logged for replay and governance
Streamlit UI — pick a case, run the workflow, see tier, evidence, note, and ledger
Tech stack

Python · LangGraph · LangChain · Claude (Anthropic) · scikit-learn · pandas · Streamlit

Setup

Requires Python 3.12 and an Anthropic API key.
