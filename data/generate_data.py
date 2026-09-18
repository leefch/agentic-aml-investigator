"""
Builds a small synthetic dataset so the whole pipeline runs without ever
touching real customer data. Everything here is invented - the countries are
made up, the amounts are exaggerated, and the "suspicious" patterns are
planted on purpose so the model and the agents actually have something to
catch.

Writes three CSVs into data/generated:
    accounts.csv       - the party master
    transactions.csv   - the movement, with a hidden is_suspicious label
    watchlist.csv      - counterparties we treat as sanctioned / high risk
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR

rng = np.random.default_rng(42)

N_ACCOUNTS = 300
N_TXNS = 6000

# obviously fake so nobody mistakes this for a real sanctions list
HIGH_RISK_COUNTRIES = ["Ruritania", "Kolechia", "Arstotzka"]
NORMAL_COUNTRIES = ["US", "US", "US", "UK", "Canada", "Germany", "Japan"]


def make_accounts():
    rows = []
    for i in range(N_ACCOUNTS):
        rows.append({
            "account_id": f"ACC{i:05d}",
            "opened_days_ago": int(rng.integers(5, 3650)),
            "home_country": rng.choice(NORMAL_COUNTRIES),
            "segment": rng.choice(["retail", "retail", "retail", "wealth", "business"]),
        })
    return pd.DataFrame(rows)


def make_transactions(accounts):
    acc_ids = accounts["account_id"].to_numpy()
    age_lookup = dict(zip(accounts["account_id"], accounts["opened_days_ago"]))

    rows = []
    base_ts = pd.Timestamp("2026-09-01")

    for i in range(N_TXNS):
        sender = rng.choice(acc_ids)
        receiver = rng.choice(acc_ids)
        while receiver == sender:
            receiver = rng.choice(acc_ids)

        # normal traffic: modest amounts, mostly domestic, spread over 90 days
        amount = float(np.round(rng.lognormal(mean=6.2, sigma=1.0), 2))
        country = rng.choice(NORMAL_COUNTRIES)
        ts = base_ts + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 90)))
        suspicious = 0

        # plant ~2.5% clearly dodgy movement: big round-ish amount, cross border
        # into a high-risk country, often from a freshly opened account.
        if rng.random() < 0.025:
            amount = float(np.round(rng.uniform(9000, 60000), 2))
            country = rng.choice(HIGH_RISK_COUNTRIES)
            suspicious = 1
            # bias these toward newer accounts so account age becomes a signal
            young = [a for a in acc_ids if age_lookup[a] < 120]
            if young:
                sender = rng.choice(young)

        rows.append({
            "txn_id": f"TXN{i:06d}",
            "timestamp": ts,
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "dest_country": country,
            "is_suspicious": suspicious,
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # a review queue: everything suspicious plus a chunk of normal traffic, so
    # the agents get realistic false positives to clear, not just guilty cases.
    flagged = df[df.is_suspicious == 1].copy()
    noise = df[df.is_suspicious == 0].sample(n=len(flagged) * 3, random_state=1)
    queue = pd.concat([flagged, noise]).sample(frac=1, random_state=2)
    df["in_review_queue"] = df.txn_id.isin(queue.txn_id).astype(int)
    return df


def make_watchlist(accounts):
    # a handful of counterparties flagged as sanctioned. keep it small.
    picks = accounts.sample(n=15, random_state=7)["account_id"].tolist()
    return pd.DataFrame({
        "account_id": picks,
        "reason": rng.choice(
            ["OFAC match", "adverse media", "PEP", "prior SAR"], size=len(picks)
        ),
    })


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    accounts = make_accounts()
    txns = make_transactions(accounts)
    watch = make_watchlist(accounts)

    accounts.to_csv(DATA_DIR / "accounts.csv", index=False)
    txns.to_csv(DATA_DIR / "transactions.csv", index=False)
    watch.to_csv(DATA_DIR / "watchlist.csv", index=False)

    print(f"accounts:     {len(accounts)}")
    print(f"transactions: {len(txns)}  (suspicious={int(txns.is_suspicious.sum())}, "
          f"in queue={int(txns.in_review_queue.sum())})")
    print(f"watchlist:    {len(watch)}")
    print(f"written to {DATA_DIR}")


if __name__ == "__main__":
    main()
