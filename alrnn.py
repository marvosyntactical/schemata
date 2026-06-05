"""Almost-Linear RNN: ReLU on only P units, linear on the rest (Brenner et al.
2024). The P switching boundaries induce a parsimonious symbolic dynamics: the
activation pattern over the P nonlinear units is the symbol at each step.

    g(z)_i = relu(z_i)  for i < P   (nonlinear units, carry a switching boundary)
    g(z)_i = z_i        for i >= P  (linear units, the Koopman-ish core)
    z_t = A z_{t-1} + W g(z_{t-1}) + h        (A diagonal), clipped for boundedness

Extraction methods give the two gauge-free signatures used downstream:
  - linear_core_spectrum(): eigenvalues of the linear-unit block (Koopman channel)
  - itinerary(): activation-pattern sequence -> transition graph (symbolic channel)
"""
import numpy as np
import torch
import torch.nn as nn


class ALRNN(nn.Module):
    def __init__(self, latent_dim=16, obs_dim=3, P=3, clip=8.0):
        super().__init__()
        self.M, self.d, self.P = latent_dim, obs_dim, P
        self.A = nn.Parameter(torch.full((latent_dim,), 0.8))
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.05)
        self.h = nn.Parameter(torch.zeros(latent_dim))
        self.clip = clip

    def _g(self, z):
        out = z.clone()
        out[..., : self.P] = torch.relu(z[..., : self.P])
        return out

    def step(self, z):
        z = self.A * z + self._g(z) @ self.W.t() + self.h
        if self.clip is not None:
            z = torch.clamp(z, -self.clip, self.clip)
        return z

    def forced_rollout(self, x, alpha, return_latents=False):
        B, T, d = x.shape
        z = torch.zeros(B, self.M, device=x.device, dtype=x.dtype)
        z[:, :d] = x[:, 0]
        preds, lats = [], []
        for t in range(1, T):
            z = self.step(z)
            preds.append(z[:, :d])
            if return_latents:
                lats.append(z[:, : self.P])     # pre-injection nonlinear-unit states
            inj = z.clone()
            inj[:, :d] = (1 - alpha) * z[:, :d] + alpha * x[:, t]
            z = inj
        preds = torch.stack(preds, dim=1)
        if return_latents:
            return preds, torch.stack(lats, dim=1)
        return preds

    def region_reg(self, lats):
        """Penalise degenerate (always on/off) ReLU units: maximise per-unit
        activation entropy so the units flip and the symbolic partition is used.
        lats: (B,T,P) pre-activations. Returns -mean binary entropy (minimising it
        pushes each unit's active-rate toward 0.5)."""
        p = torch.sigmoid(lats).mean(dim=(0, 1)).clamp(1e-4, 1 - 1e-4)   # (P,)
        H = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return -H.mean()

    @torch.no_grad()
    def free_run(self, z0_obs, n, return_patterns=False):
        z = torch.zeros(self.M, dtype=self.A.dtype)
        z[: self.d] = torch.as_tensor(z0_obs, dtype=self.A.dtype)
        out = np.empty((n, self.d), dtype=np.float64)
        pats = np.empty((n, self.P), dtype=np.int8) if return_patterns else None
        for t in range(n):
            z = self.step(z)
            out[t] = z[: self.d].numpy()
            if return_patterns:
                pats[t] = (z[: self.P] > 0).numpy().astype(np.int8)
        return (out, pats) if return_patterns else out

    @torch.no_grad()
    def jacobian(self, z):
        dg = torch.ones(self.M, dtype=self.A.dtype)
        dg[: self.P] = (z[: self.P] > 0).to(self.A.dtype)
        return torch.diag(self.A) + self.W * dg

    @torch.no_grad()
    def linear_core_spectrum(self):
        """Eigenvalues of the linear-unit sub-block A+W restricted to the linear
        units -- the Koopman-ish operator that acts without switching. Gauge
        info (basis) is quotiented by taking eigenvalues only."""
        P, M = self.P, self.M
        block = torch.diag(self.A)[P:, P:] + self.W[P:, P:]
        ev = torch.linalg.eigvals(block).numpy()
        return ev[np.argsort(-np.abs(ev))]      # sorted by magnitude

    @torch.no_grad()
    def itinerary(self, z0_obs, n=4000, warmup=500):
        """Free-run, return the activation-pattern sequence (as integer symbols)
        and the empirical transition matrix over visited symbols."""
        _, pats = self.free_run(z0_obs, n + warmup, return_patterns=True)
        pats = pats[warmup:]
        # encode each P-bit pattern as an integer symbol
        powers = (1 << np.arange(self.P))
        syms = (pats * powers).sum(1)
        uniq = np.unique(syms)
        idx = {s: i for i, s in enumerate(uniq)}
        k = len(uniq)
        T = np.zeros((k, k))
        for a, b in zip(syms[:-1], syms[1:]):
            T[idx[a], idx[b]] += 1
        row = T.sum(1, keepdims=True)
        Tn = T / np.clip(row, 1, None)
        return syms, uniq, Tn
