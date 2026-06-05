"""Minimal PLRNN for DSR (primer §3.2).

    z_t = A z_{t-1} + W relu(z_{t-1}) + h    (A diagonal)

Observation reads the first d latent dims (identity decoder), which makes
sparse *identity teacher forcing* trivial: every tau steps overwrite the
observed latent dims with data. The piecewise-linear form gives an exact
Jacobian J(z) = diag(A) + W diag(1[z>0]) used for the Lyapunov spectrum and
(later, E1) analytic fixed points.
"""
import numpy as np
import torch
import torch.nn as nn


class PLRNN(nn.Module):
    def __init__(self, latent_dim=20, obs_dim=3, clip=8.0):
        super().__init__()
        self.M, self.d = latent_dim, obs_dim
        # Boundedness wall: vanilla PLRNNs give no bounded-orbit guarantee, so a
        # correct chaotic transient can still escape to infinity. clip >> the
        # attractor's extent (data lives within ~3 std) prevents escape without
        # distorting the attractor. clip=None recovers the unbounded PLRNN.
        self.clip = clip
        # Contractive init: forward must stay finite so a gradient exists.
        # Untrained unforced latent dims otherwise explode over long rollouts.
        self.A = nn.Parameter(torch.full((latent_dim,), 0.8))
        W = torch.randn(latent_dim, latent_dim) * 0.05
        self.W = nn.Parameter(W)
        self.h = nn.Parameter(torch.zeros(latent_dim))

    def step(self, z):
        z = self.A * z + torch.relu(z) @ self.W.t() + self.h
        if self.clip is not None:
            z = torch.clamp(z, -self.clip, self.clip)
        return z

    def forced_rollout(self, x, alpha):
        """Generalized-TF rollout over a data batch x: (B, T, d) (Hess et al. 2023).
        Every step the observed latent dims are pulled toward data by a convex
        mix z <- (1-alpha) z_model + alpha z_data. alpha=0 is pure BPTT, alpha=1
        is hard identity forcing. Returns one-step predictions (B, T-1, d) taken
        BEFORE injection, so the loss measures the model, not the data."""
        B, T, d = x.shape
        z = torch.zeros(B, self.M, device=x.device, dtype=x.dtype)
        z[:, :d] = x[:, 0]
        preds = []
        for t in range(1, T):
            z = self.step(z)
            preds.append(z[:, :d])
            inj = z.clone()
            inj[:, :d] = (1 - alpha) * z[:, :d] + alpha * x[:, t]
            z = inj
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def free_run(self, z0_obs, n):
        """Autonomous generation (no forcing) from an observed seed z0_obs (d,)."""
        z = torch.zeros(self.M, dtype=self.A.dtype)
        z[: self.d] = torch.as_tensor(z0_obs, dtype=self.A.dtype)
        out = np.empty((n, self.d), dtype=np.float64)
        for t in range(n):
            z = self.step(z)
            out[t] = z[: self.d].numpy()
        return out

    @torch.no_grad()
    def jacobian(self, z):
        D = (z > 0).to(self.A.dtype)
        return torch.diag(self.A) + self.W * D            # (M,M), broadcasts cols


def train(model, data, alpha=0.3, alpha_end=None, seq_len=100, epochs=40,
          batch=64, lr=1e-3, reg_lambda=0.0, device="cpu", log=print):
    """Generalized-TF BPTT. data: (N, d) numpy. Returns loss history.
    If alpha_end is set, anneal the GTF mix linearly alpha -> alpha_end across
    epochs: start strongly forced (learn the global map over both lobes), end
    weakly forced (force self-sustained free-running dynamics)."""
    model.to(device)
    x = torch.as_tensor(data, dtype=torch.float32, device=device)
    N, d = x.shape
    # non-overlapping chunks
    n_chunks = (N - 1) // seq_len
    chunks = x[: n_chunks * seq_len].reshape(n_chunks, seq_len, d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        a = alpha if alpha_end is None else alpha + (alpha_end - alpha) * ep / max(1, epochs - 1)
        perm = torch.randperm(n_chunks)
        tot = 0.0
        for i in range(0, n_chunks, batch):
            idx = perm[i:i + batch]
            xb = chunks[idx]
            if reg_lambda > 0:
                pred, lat = model.forced_rollout(xb, a, return_latents=True)
                loss = ((pred - xb[:, 1:]) ** 2).mean() + reg_lambda * model.region_reg(lat)
            else:
                pred = model.forced_rollout(xb, a)
                loss = ((pred - xb[:, 1:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / n_chunks)
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"  epoch {ep+1:3d}/{epochs}  loss {hist[-1]:.5f}")
    return hist
