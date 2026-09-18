"""
The scorer object that gets pickled by training and loaded at serve time.

It lives here (in the package) rather than in the training script on purpose:
joblib pickles by fully-qualified class name, so if this class were defined in
train_scorer.py it would pickle as __main__.Scorer and fail to load anywhere
else. Learned that the hard way.
"""
import numpy as np

from agent.features import FEATURE_COLS


class Scorer:
    def __init__(self, scaler, iforest, logit, means, stds):
        self.scaler = scaler
        self.iforest = iforest
        self.logit = logit
        self.means = means          # per-feature mean, used only for the explanation
        self.stds = stds

    def score(self, feats: dict):
        x = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)
        xs = self.scaler.transform(x)

        # logistic gives a probability directly
        p_logit = float(self.logit.predict_proba(xs)[0, 1])

        # isolation forest score_samples: higher = more normal. flip and squash
        # into 0..1 so it reads the same direction as the logistic prob.
        raw = float(self.iforest.score_samples(xs)[0])
        p_iso = 1.0 / (1.0 + np.exp(4.0 * raw))

        blended = 0.6 * p_logit + 0.4 * p_iso
        # plain float, not np.float64 - keeps it JSON-serializable downstream
        return {"score": float(round(blended, 4)), "drivers": self._drivers(feats)}

    def _drivers(self, feats):
        """Cheap explanation: which features sit unusually high vs the training
        mean, filtered to ones the logistic model treats as risk-increasing.
        Good enough for a case note; not a SHAP replacement."""
        out = []
        for i, c in enumerate(FEATURE_COLS):
            if self.stds[i] == 0:
                continue
            z = (feats[c] - self.means[i]) / self.stds[i]
            if z > 1.0 and self.logit.coef_[0][i] > 0:
                out.append((c, round(float(z), 2)))
        out.sort(key=lambda t: t[1], reverse=True)
        return [f"{name} (z={z})" for name, z in out[:4]]
