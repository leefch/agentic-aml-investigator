"""
Trains the risk scorer the triage agent calls as a tool.

Two models, on purpose:
  - IsolationForest: unsupervised, catches "weird" even with no label
  - LogisticRegression: supervised on the planted label, gives a probability
    and interpretable coefficients

We blend them. The blend is kept deliberately simple so the score stays
explainable - in a regulated setting "the model said 0.9" is not an answer,
"large cross-border amount from a 40-day-old account" is.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODEL_PATH
from agent.features import build_features, FEATURE_COLS
from agent.scorer import Scorer


def main():
    txns = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")

    # build the feature table row by row. small data, so a plain loop is fine and
    # it reuses the exact function serving will use.
    rows = [build_features(t, txns, accounts) for _, t in txns.iterrows()]
    X = pd.DataFrame(rows)[FEATURE_COLS]
    y = txns["is_suspicious"].to_numpy()

    # fit on the raw ndarray so the scaler doesn't memorize feature names -
    # serving passes a plain array and we don't want a warning on every call.
    X_np = X.to_numpy()
    scaler = StandardScaler().fit(X_np)
    Xs = scaler.transform(X_np)

    iforest = IsolationForest(n_estimators=200, contamination=0.03, random_state=0).fit(Xs)
    logit = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs, y)

    scorer = Scorer(
        scaler=scaler,
        iforest=iforest,
        logit=logit,
        means=X.mean().to_numpy(),
        stds=X.std(ddof=0).replace(0, 1).to_numpy(),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, MODEL_PATH)

    # quick sanity readout so we know training didn't produce garbage
    probs = np.array([scorer.score(r)["score"] for r in rows])
    print(f"trained on {len(y)} txns")
    print(f"mean score  suspicious={probs[y == 1].mean():.3f}  normal={probs[y == 0].mean():.3f}")
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
