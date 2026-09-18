"""
The tools the agents are allowed to call. Each one is a plain function over
the synthetic data - the point is that the LLM never touches raw data directly,
it asks these for exactly what it needs and gets structured results back.

Data and model load once at import. If they're missing we fail loudly with a
hint, because the usual cause is "you forgot to run generate_data / train".
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from config import DATA_DIR, MODEL_PATH, HIGH_RISK, MED_RISK
from agent.features import build_features


@lru_cache(maxsize=1)
def _load():
    if not (DATA_DIR / "transactions.csv").exists():
        raise FileNotFoundError(
            "no data found - run `python data/generate_data.py` first"
        )
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            "no model found - run `python models/train_scorer.py` first"
        )
    txns = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    watch = pd.read_csv(DATA_DIR / "watchlist.csv")
    scorer = joblib.load(MODEL_PATH)
    return txns, accounts, watch, scorer


def get_transaction(txn_id: str) -> dict:
    txns, _, _, _ = _load()
    row = txns.loc[txns.txn_id == txn_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def sample_queue(n: int = 15):
    """Return a few txn_ids sitting in the review queue - what the UI offers up
    to investigate."""
    txns, _, _, _ = _load()
    q = txns[txns.in_review_queue == 1]["txn_id"].tolist()
    return q[:n]


def score_transaction(txn: dict) -> dict:
    """Run the ML scorer and band the result. This is the tool the triage
    agent leans on rather than eyeballing the amount."""
    txns, accounts, _, scorer = _load()
    feats = build_features(txn, txns, accounts)
    out = scorer.score(feats)
    score = out["score"]
    tier = "high" if score >= HIGH_RISK else "medium" if score >= MED_RISK else "low"
    return {"score": score, "tier": tier, "drivers": out["drivers"]}


def account_history(account_id: str, limit: int = 10) -> list:
    """Recent movement for an account - context for the investigator."""
    txns, _, _, _ = _load()
    hist = txns[(txns.sender_id == account_id) | (txns.receiver_id == account_id)]
    hist = hist.sort_values("timestamp", ascending=False).head(limit)
    keep = ["txn_id", "timestamp", "sender_id", "receiver_id", "amount", "dest_country"]
    return hist[keep].astype({"timestamp": str}).to_dict("records")


def check_watchlist(account_ids: list) -> list:
    """Are any of these parties on the sanctions/high-risk list?"""
    _, _, watch, _ = _load()
    hits = watch[watch.account_id.isin(account_ids)]
    return hits.to_dict("records")


def linked_accounts(account_id: str) -> list:
    """Very light 'network' lookup: other accounts this one has transacted with.
    A real system would resolve shared device/IP/beneficial-owner; here we just
    walk the transaction graph one hop."""
    txns, _, _, _ = _load()
    out = txns[txns.sender_id == account_id]["receiver_id"].value_counts().head(5)
    return [{"account_id": a, "shared_txns": int(c)} for a, c in out.items()]


def ranked_queue(n: int = 25) -> list:
    """Score the review queue and return items worst-first, each with its
    score and tier, so the UI can surface the riskiest cases at the top."""
    txns, _, _, _ = _load()
    queue_ids = txns[txns.in_review_queue == 1]["txn_id"].tolist()

    scored = []
    for tid in queue_ids:
        r = score_transaction(get_transaction(tid))
        scored.append({"txn_id": tid, "score": r["score"], "tier": r["tier"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n]