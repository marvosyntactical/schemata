"""Schema prototype memory + attention controller (dsa_native_ssm.md §8.2).

Stores gauge-free schema embeddings and makes the continual-learning decision:
reuse a known schema (assimilate) or allocate a new one (accommodate). The
decision is the differentiable §4.4 rule -- distance to the nearest prototype vs
a threshold tau -- with softmax(-beta*d^2) reported as soft schema membership
(beta is the boundary-sharpness knob; beta->inf recovers the hard argmin).

Comparison is done in a running z-scored feature space so the heterogeneous
channels (Koopman / symbolic / dynamical) are commensurate.
"""
import numpy as np


class SchemaMemory:
    def __init__(self, tau=2.5, beta=1.0, eps=1e-6, slices=None, weights=None):
        self.tau = tau          # assimilate if nearest normalised distance < tau
        self.beta = beta        # attention temperature
        self.eps = eps
        self.slices = slices    # {channel: (lo,hi)} for channel-weighted distance
        self.weights = weights or {}   # {channel: w}; default 1.0
        self._seen = []         # all raw embeddings, for running normalisation
        self.protos = []        # list of dicts: {mean(raw), count, name}

    def _wdist(self, zq, zp):
        """Channel-weighted Euclidean distance in z-scored space. Down-weights
        noisy/degenerate channels (Koopman, symbolic) relative to the coordinate-
        invariant dynamical channel."""
        if not self.slices:
            return np.linalg.norm(zq - zp)
        d2 = 0.0
        for ch, (lo, hi) in self.slices.items():
            w = self.weights.get(ch, 1.0)
            d2 += w * np.sum((zq[lo:hi] - zp[lo:hi]) ** 2)
        return np.sqrt(d2)

    # --- running normalisation over everything seen ---
    def observe(self, emb):
        self._seen.append(np.asarray(emb, float))

    def _stats(self):
        X = np.array(self._seen)
        mu = X.mean(0)
        sd = X.std(0)
        sd = np.where(sd < self.eps, 1.0, sd)
        return mu, sd

    def _norm(self, v, mu, sd):
        return (np.asarray(v, float) - mu) / sd

    # --- retrieval / decision ---
    def query(self, emb):
        """Return (decision, k_star, distances, attention). decision in
        {'accommodate','assimilate'}."""
        self.observe(emb)
        if not self.protos:
            return "accommodate", None, np.array([]), np.array([])
        mu, sd = self._stats()
        q = self._norm(emb, mu, sd)
        dists = np.array([self._wdist(self._norm(p["mean"], mu, sd), q)
                          for p in self.protos])
        attn = self._softmax(-self.beta * dists ** 2)
        k = int(np.argmin(dists))
        decision = "assimilate" if dists[k] < self.tau else "accommodate"
        return decision, k, dists, attn

    def _softmax(self, x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    # --- updates ---
    def add_prototype(self, emb, name):
        self.protos.append({"mean": np.asarray(emb, float), "count": 1,
                            "name": name})
        return len(self.protos) - 1

    def update_prototype(self, k, emb):
        p = self.protos[k]
        p["count"] += 1
        p["mean"] += (np.asarray(emb, float) - p["mean"]) / p["count"]
