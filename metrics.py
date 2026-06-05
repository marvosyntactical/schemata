"""Invariant-based DSR metrics (primer §3.5). Judge by invariants, not forecasts.

- d_stsp:  divergence between binned invariant densities (attractor geometry).
- ps_distance:  Welch power-spectrum distance (temporal structure).
- lyapunov_spectrum / kaplan_yorke:  from the model's exact PWL Jacobian.
"""
import numpy as np
from scipy.signal import welch


def d_stsp(gen, true, bins=20, eps=1e-6, pad=0.1):
    """Symmetric KL between coarse 3D histograms of two point clouds.
    Bin range is fixed from the TRUE attractor (plus padding); generated points
    outside it are clipped into the edge bins. This is deliberate: a divergent
    free-run must score *worse*, not collapse all mass into one shared bin and
    look spuriously good (the failure mode of an adaptive shared range)."""
    span = true.max(0) - true.min(0)
    lo, hi = true.min(0) - pad * span, true.max(0) + pad * span
    edges = [np.linspace(lo[k], hi[k], bins + 1) for k in range(true.shape[1])]
    g = np.clip(gen, lo, hi)
    p, _ = np.histogramdd(g, bins=edges)
    q, _ = np.histogramdd(true, bins=edges)
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def ps_distance(gen, true, nperseg=512):
    """Mean over dims of L1 distance between normalised log power spectra."""
    dists = []
    for k in range(gen.shape[1]):
        _, pg = welch(gen[:, k], nperseg=nperseg)
        _, pt = welch(true[:, k], nperseg=nperseg)
        pg = pg / pg.sum()
        pt = pt / pt.sum()
        dists.append(np.abs(pg - pt).sum())
    return float(np.mean(dists))


def lyapunov_spectrum(model, seed_obs, n=4000, warmup=500):
    """Full Lyapunov spectrum via QR along a model free-run (exact PWL Jacobians).
    Returns exponents in nats per step (descending)."""
    import torch
    z = torch.zeros(model.M, dtype=model.A.dtype)
    z[: model.d] = torch.as_tensor(seed_obs, dtype=model.A.dtype)
    Q = torch.eye(model.M, dtype=model.A.dtype)
    lsum = np.zeros(model.M)
    count = 0
    with torch.no_grad():
        for t in range(n + warmup):
            J = model.jacobian(z)
            Q, R = torch.linalg.qr(J @ Q)
            if t >= warmup:
                lsum += np.log(np.abs(np.diag(R.numpy())) + 1e-300)
                count += 1
            z = model.step(z)
    return np.sort(lsum / count)[::-1]


def kaplan_yorke(spectrum):
    """Lyapunov (Kaplan–Yorke) dimension from a descending spectrum."""
    s = np.sort(spectrum)[::-1]
    csum = np.cumsum(s)
    j = np.searchsorted(-csum, 0.0)  # last index with positive cumulative sum
    if j == 0:
        return 0.0
    if j >= len(s):
        return float(len(s))
    return float(j + csum[j - 1] / abs(s[j]))
