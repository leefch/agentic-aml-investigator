"""
One place to turn a raw transaction into model features. Both the training
script and the live scoring tool import from here on purpose - if the feature
logic lived in two places it would drift and training/serving would silently
disagree.
"""
import pandas as pd

HIGH_RISK_COUNTRIES = {"Ruritania", "Kolechia", "Arstotzka"}

FEATURE_COLS = [
    "amount",
    "amount_log",
    "cross_border",
    "to_high_risk",
    "account_age_days",
    "amt_vs_sender_avg",
    "sender_txns_24h",
]


def _sender_context(txn, history, accounts):
    """Pull the few things we need about the sender: how old the account is,
    their typical amount, and how busy they've been in the last day."""
    sender = txn["sender_id"]

    age = 999999
    row = accounts.loc[accounts.account_id == sender]
    if not row.empty:
        age = int(row.iloc[0]["opened_days_ago"])

    sent = history[history.sender_id == sender]
    avg_amt = float(sent["amount"].mean()) if len(sent) else txn["amount"]

    # count sender activity in the 24h window before this txn
    ts = pd.Timestamp(txn["timestamp"])
    recent = sent[(sent.timestamp <= ts) & (sent.timestamp > ts - pd.Timedelta("24h"))]
    return age, avg_amt, len(recent)


def build_features(txn, history, accounts):
    """txn is a dict/Series for a single transaction. history is the full txn
    frame (used for sender averages and velocity). Returns a plain dict."""
    import numpy as np

    age, avg_amt, n_24h = _sender_context(txn, history, accounts)
    amount = float(txn["amount"])
    dest = txn["dest_country"]
    home = "US"
    row = accounts.loc[accounts.account_id == txn["sender_id"]]
    if not row.empty:
        home = row.iloc[0]["home_country"]

    return {
        "amount": amount,
        "amount_log": float(np.log1p(amount)),
        "cross_border": int(dest != home),
        "to_high_risk": int(dest in HIGH_RISK_COUNTRIES),
        "account_age_days": age,
        "amt_vs_sender_avg": amount / avg_amt if avg_amt else 1.0,
        "sender_txns_24h": n_24h,
    }
