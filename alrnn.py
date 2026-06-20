"""Almost-Linear RNN with SLAO factorization of the full coupling matrix W.

Architecture:
    g(z)_i = relu(z_i)  for i < P   (nonlinear switching units)
    g(z)_i = z_i        for i >= P  (linear Koopman-ish units)

    z_t = A_diag z_{t-1} + g(z_{t-1}) @ W^T + h
    W = W_B @ W_A  ∈ R^{M×M}          (rank-r factorization of the full W)
    W_B ∈ R^{M×r}  — SLAO "B": up-projection   (EMA-merged across tasks)
    W_A ∈ R^{r×M}  — SLAO "A": down-projection (QR-init, replaced per task)

The step is therefore:
    z_t = A_diag z_{t-1} + g(z_{t-1}) @ W_A^T @ W_B^T + h

which is identical in output to the unfactored version with W = W_B @ W_A.
"""
import numpy as np
import torch
import torch.nn as nn


class ALRNN(nn.Module):
    def __init__(self, latent_dim=16, obs_dim=3, P=3, rank=None, clip=8.0):
        super().__init__()
        self.M, self.d, self.P = latent_dim, obs_dim, P
        self.rank = rank if rank is not None else P   # LoRA rank r, default r=P
        r = self.rank

        self.A    = nn.Parameter(torch.full((latent_dim,), 0.8))
        self.W_B  = nn.Parameter(torch.randn(latent_dim, r) * 0.05)   # M×r
        self.W_A  = nn.Parameter(torch.randn(r, latent_dim) * 0.05)   # r×M
        self.h    = nn.Parameter(torch.zeros(latent_dim))
        self.clip = clip

    # ── W as composed product ─────────────────────────────────────────────────

    def W(self):
        """Effective full coupling: W_B @ W_A ∈ R^{M×M}."""
        return self.W_B @ self.W_A

    # ── forward ──────────────────────────────────────────────────────────────

    def _g(self, z):
        out = z.clone()
        out[..., :self.P] = torch.relu(z[..., :self.P])
        return out

    def step(self, z):
        # g(z) @ W^T = g(z) @ W_A^T @ W_B^T
        z = self.A * z + self._g(z) @ self.W_A.t() @ self.W_B.t() + self.h
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
                lats.append(z[:, :self.P])
            inj = z.clone()
            inj[:, :d] = (1 - alpha) * z[:, :d] + alpha * x[:, t]
            z = inj
        preds = torch.stack(preds, dim=1)
        if return_latents:
            return preds, torch.stack(lats, dim=1)
        return preds

    def region_reg(self, lats):
        p = torch.sigmoid(lats).mean(dim=(0, 1)).clamp(1e-4, 1 - 1e-4)
        H = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return -H.mean()

    # ── analysis ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def free_run(self, z0_obs, n, return_patterns=False):
        z = torch.zeros(self.M, dtype=self.A.dtype)
        z[:self.d] = torch.as_tensor(z0_obs, dtype=self.A.dtype)
        out  = np.empty((n, self.d), dtype=np.float64)
        pats = np.empty((n, self.P), dtype=np.int8) if return_patterns else None
        for t in range(n):
            z = self.step(z)
            out[t] = z[:self.d].numpy()
            if return_patterns:
                pats[t] = (z[:self.P] > 0).numpy().astype(np.int8)
        return (out, pats) if return_patterns else out

    @torch.no_grad()
    def jacobian(self, z):
        dg = torch.ones(self.M, dtype=self.A.dtype)
        dg[:self.P] = (z[:self.P] > 0).to(self.A.dtype)
        return torch.diag(self.A) + self.W() * dg

    @torch.no_grad()
    def linear_core_spectrum(self):
        """Eigenvalues of the linear-unit sub-block of the effective W."""
        P = self.P
        W_full = self.W()
        block  = torch.diag(self.A[P:]) + W_full[P:, P:]
        ev = torch.linalg.eigvals(block).numpy()
        return ev[np.argsort(-np.abs(ev))]

    @torch.no_grad()
    def region_matrix(self, pattern):
        dg = torch.ones(self.M, dtype=self.A.dtype)
        dg[:self.P] = torch.as_tensor(pattern, dtype=self.A.dtype)
        return (torch.diag(self.A) + self.W() * dg).numpy().astype(np.float64)

    @torch.no_grad()
    def enumerate_visited_regions(self, z0_obs, n=4000, warmup=500):
        z = torch.zeros(self.M, dtype=self.A.dtype)
        z[:self.d] = torch.as_tensor(z0_obs, dtype=self.A.dtype)
        powers = (1 << np.arange(self.P))
        syms, regions, patterns = [], {}, {}
        clip_hits = 0
        for t in range(n + warmup):
            pat    = (z[:self.P] > 0).numpy().astype(np.int8)
            z_next = self.A * z + self._g(z) @ self.W_A.t() @ self.W_B.t() + self.h
            if self.clip is not None:
                z_clipped = torch.clamp(z_next, -self.clip, self.clip)
                if t >= warmup and bool((z_next != z_clipped).any()):
                    clip_hits += 1
                z_next = z_clipped
            if t >= warmup:
                s = int((pat * powers).sum())
                syms.append(s)
                if s not in regions:
                    regions[s] = self.region_matrix(pat)
                    patterns[s] = pat
            z = z_next
        syms  = np.asarray(syms, dtype=np.int64)
        uniq  = np.unique(syms)
        idx   = {int(s): i for i, s in enumerate(uniq)}
        B_mat = np.zeros((len(uniq), len(uniq)))
        for a, b in zip(syms[:-1], syms[1:]):
            B_mat[idx[int(a)], idx[int(b)]] += 1
        return {
            "symbols":   syms,
            "uniq":      uniq,
            "index":     idx,
            "regions":   regions,
            "patterns":  patterns,
            "trans":     B_mat,
            "h":         self.h.detach().numpy().astype(np.float64),
            "clip_rate": clip_hits / max(len(syms), 1),
        }

    @torch.no_grad()
    def itinerary(self, z0_obs, n=4000, warmup=500):
        _, pats = self.free_run(z0_obs, n + warmup, return_patterns=True)
        pats  = pats[warmup:]
        powers = (1 << np.arange(self.P))
        syms   = (pats * powers).sum(1)
        uniq   = np.unique(syms)
        idx    = {s: i for i, s in enumerate(uniq)}
        k      = len(uniq)
        T      = np.zeros((k, k))
        for a, b in zip(syms[:-1], syms[1:]):
            T[idx[a], idx[b]] += 1
        row = T.sum(1, keepdims=True)
        return syms, uniq, T / np.clip(row, 1, None)
