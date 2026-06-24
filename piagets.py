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
import copy
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
                 use_class_ewc: bool = False,
                 use_class_b: bool = False,
                 use_ortho: bool = False,
                 ortho_k: int = 2,
                 no_slao: bool = False,
                 invert_fisher: bool = False):
        self.lam_ewc         = lam_ewc
        self.lam_assim       = lam_assim if lam_assim is not None else 2.0 * lam_ewc
        self.lam_accom       = lam_accom if lam_accom is not None else 0.4 * lam_ewc
        self.schema_sigma    = schema_sigma
        self.assim_mse_ratio = assim_mse_ratio
        self.sig_kw          = sig_fisher_kwargs or {}
        self.no_slao         = no_slao   # EWC on all params (incl W_A), no QR-init/merge
        self.invert_fisher   = invert_fisher  # use (1-F) instead of F as EWC weights
        # Option 3: φ-functional regularization
        self.lam_phi      = lam_phi
        self._phi_reg_kw  = phi_reg_kwargs or dict(T_warmup=100, T_track=100, beta=10.0)
        self._phi_stars:  dict = {}   # class → 24-dim np.array (EMA of trained φ)
        # Option 4: dual LoRA block split
        self.r_lin        = r_lin
        # Per-class EWC anchors
        self.use_class_ewc       = use_class_ewc
        self._F_B_by_class:      dict = {}
        self._W_B_star_by_class: dict = {}
        self._class_task_counts: dict = {}

        # Per-class (W_B, W_A) modules — replaces B_merged_by_class.
        # Each class stores an EMA of both factors, eliminating the A_merge mismatch
        # that caused large negative BWT when using (W_B^class, W_A^global) at eval.
        self.use_class_b     = use_class_b
        self._class_modules: dict = {}   # class → {'W_B': tensor, 'W_A': tensor}

        # Orthogonal subspace — joint growing basis with budget cap.
        # Replaces per-class _W_B_bases (which exhausted the rank after n_classes × k dims).
        # _U_joint grows incrementally: at most ortho_budget = r//2 total directions.
        self.use_ortho   = use_ortho
        self._ortho_k    = ortho_k       # max new directions extracted per task
        self._U_joint: torch.Tensor | None = None   # joint orthonormal basis (M × n_dirs)

        self.current_mode: str = "accommodation"

        # SLAO merge state
        self.B_merged: torch.Tensor | None = None
        self.A_merge:  torch.Tensor | None = None
        self._B_prev:  torch.Tensor | None = None
        self._A_prev:  torch.Tensor | None = None

        # EWC state — W_B, A, h always; W_A only when no_slao
        self._F_B:      torch.Tensor | None = None
        self._W_B_star: torch.Tensor | None = None
        self._F_A_diag: torch.Tensor | None = None
        self._A_star:   torch.Tensor | None = None
        self._F_h:      torch.Tensor | None = None
        self._h_star:   torch.Tensor | None = None
        self._F_W_A:    torch.Tensor | None = None   # W_A Fisher (no_slao only)
        self._W_A_star: torch.Tensor | None = None   # W_A anchor (no_slao only)

        # Schema tracking
        self._task_phis:    list  = []
        self._n_tasks:      int   = 0
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

    # ── Option B: φ-probe via brief λ=0 fine-tune ──────────────────────────

    def probe_from_phi_finetune(self, model, data,
                                n_probe_epochs: int = 30,
                                seq_len: int = 100, batch: int = 64,
                                probe_alpha: float = 0.5, probe_lr: float = 1e-3,
                                device: str = "cpu"):
        """Brief λ=0 fine-tune on new data → compute φ → compare to stored centroids.

        Two-stage classification:
        1. Chaos shortcut: if φ_act_std.mean() < CHAOS_THRESHOLD after fine-tuning,
           classify as the lowest-φ_act_std centroid (the "chaos" class, i.e. Lorenz).
           φ_act_std drops toward 0 within ~50 epochs for chaotic attractors even when
           MSE hasn't fully converged, making this robust to partial probe convergence.
        2. Full 24D φ comparison (fallback): for periodic/quasiperiodic schemas
           (Torus/VdP) which converge quickly and are well-separated in full φ-space.
        """
        if not self._phi_stars or not self._task_phis:
            return self.lam_accom, "accommodation", None

        # Store M and P for φ_act_std slice before deepcopy
        M_lat = model.M   # total latent units
        P_nl  = model.P   # nonlinear (piecewise-linear) units
        # φ layout: [phi_lin(M-P), phi_acts(P), phi_act_std(P), phi_topo(1), phi_lyap(1)]
        # phi_act_std occupies indices [M_lat : M_lat + P_nl]
        act_std_sl = slice(M_lat, M_lat + P_nl)

        # Deep-copy so the probe doesn't touch the actual model weights
        m_probe = copy.deepcopy(model).to(device)
        x = torch.as_tensor(data, dtype=torch.float32, device=device)
        n_chunks = (x.shape[0] - 1) // seq_len
        if n_chunks < 1:
            del m_probe
            return self.lam_accom, "accommodation", None
        chunks = x[:n_chunks * seq_len].reshape(n_chunks, seq_len, x.shape[1])
        opt = torch.optim.Adam(m_probe.parameters(), lr=probe_lr)
        for _ in range(n_probe_epochs):
            perm = torch.randperm(n_chunks, device=device)
            for i in range(0, n_chunks, batch):
                xb   = chunks[perm[i : i + batch]]
                pred = m_probe.forced_rollout(xb, probe_alpha)
                loss = ((pred - xb[:, 1:]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m_probe.parameters(), 10.0)
                opt.step()

        phi_probe = diff_phi(m_probe,
                             T_warmup=self.sig_kw.get("T_warmup", 100),
                             T_track =self.sig_kw.get("T_track",  200),
                             beta    =self.sig_kw.get("beta",     10.0)).detach().numpy()
        del m_probe, opt

        classes   = list(self._phi_stars.keys())
        centroids = self._phi_stars

        # ── Stage 1: chaos shortcut (detects Lorenz even from partial probe) ──
        # Chaotic attractors drive φ_act_std → 0 (all gates freeze in 1-region mode).
        # This happens within ~50–100 probe epochs even before MSE converges.
        # Periodic/quasiperiodic attractors keep φ_act_std >> 0 at any convergence level.
        CHAOS_THRESHOLD    = 0.20   # probe φ_act_std mean below this → chaos regime
        CHAOS_CENTROID_MAX = 0.05   # centroid φ_act_std mean below this → confirmed chaos class
        probe_act_std_mean = float(phi_probe[act_std_sl].mean())
        centroid_act_std   = {c: float(centroids[c][act_std_sl].mean()) for c in classes}
        chaos_cls = min(centroid_act_std, key=centroid_act_std.__getitem__)
        if (probe_act_std_mean < CHAOS_THRESHOLD
                and centroid_act_std[chaos_cls] < CHAOS_CENTROID_MAX):
            lam  = self.lam_assim
            mode = "assimilation"
            print(f"  [probe_φ] φ_act_std={probe_act_std_mean:.3f} < {CHAOS_THRESHOLD}"
                  f"  chaos_cls={chaos_cls}  → {mode.upper()} (chaos shortcut)  λ={lam:.1f}")
            return lam, mode, chaos_cls

        # ── Stage 2: full 24D φ comparison (Torus / VdP / new schema) ─────────
        all_phis = np.stack([p for _, p in self._task_phis])
        scale    = np.std(all_phis, axis=0)
        scale[scale < 1e-9] = 1.0

        intra     = [np.linalg.norm((p - centroids[c]) / (scale + 1e-9))
                     for c, p in self._task_phis]
        intra_mean = float(np.mean(intra)) if intra else 1.0

        dists    = {c: float(np.linalg.norm((phi_probe - centroids[c]) / (scale + 1e-9)))
                    for c in classes}
        near_c, near_d = min(dists.items(), key=lambda x: x[1])
        threshold  = self.schema_sigma * max(intra_mean, 0.1)
        recognised = near_c if near_d < threshold else None
        mode = "assimilation" if recognised is not None else "accommodation"
        lam  = self.lam_assim if mode == "assimilation" else self.lam_accom
        print(f"  [probe_φ] φ_act_std={probe_act_std_mean:.3f}  near_c={near_c}"
              f"  d={near_d:.3f}  thr={threshold:.3f}  → {mode.upper()}  λ={lam:.1f}")
        return lam, mode, recognised

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

    def _ew(self, F: torch.Tensor) -> torch.Tensor:
        """Return EWC weight tensor: F normally, (1-F) when invert_fisher=True."""
        return (1.0 - F) if self.invert_fisher else F

    def ewc_loss(self, model, current_class=None):
        """L_EWC = λ · Σ_{ij} w_{ij} · (θ_{ij} − θ̂_{ij})²
        where w = F̂  (standard) or (1−F̂) (inverted).
        Inverted: high-sensitivity parameters are released; low-sensitivity are frozen.
        If use_class_ewc: W_B anchored per-class; A and h via global EWC.
        If no_slao: W_A also penalized (since it is not re-initialized by QR)."""
        dev = model.W_B.device
        if self.use_class_ewc and current_class is not None:
            if current_class not in self._F_B_by_class:
                # New class: no per-class W_B anchor yet, but still protect A and h
                loss = torch.tensor(0.0)
                if self._F_A_diag is not None:
                    loss = loss + self.lam_ewc * (self._ew(self._F_A_diag.to(dev)) * (model.A - self._A_star.to(dev)) ** 2).sum()
                if self._F_h is not None:
                    loss = loss + self.lam_ewc * (self._ew(self._F_h.to(dev)) * (model.h - self._h_star.to(dev)) ** 2).sum()
                return loss
            F_c     = self._ew(self._F_B_by_class[current_class].to(dev))
            Wstar_c = self._W_B_star_by_class[current_class].to(dev)
            loss = self.lam_ewc * (F_c * (model.W_B - Wstar_c) ** 2).sum()
            if self._F_A_diag is not None:
                loss = loss + self.lam_ewc * (self._ew(self._F_A_diag.to(dev)) * (model.A - self._A_star.to(dev)) ** 2).sum()
            if self._F_h is not None:
                loss = loss + self.lam_ewc * (self._ew(self._F_h.to(dev)) * (model.h - self._h_star.to(dev)) ** 2).sum()
            return loss
        if self._F_B is None:
            return torch.tensor(0.0)
        loss = self.lam_ewc * (self._ew(self._F_B.to(dev)) * (model.W_B - self._W_B_star.to(dev)) ** 2).sum()
        if self._F_A_diag is not None:
            loss = loss + self.lam_ewc * (self._ew(self._F_A_diag.to(dev)) * (model.A - self._A_star.to(dev)) ** 2).sum()
        if self._F_h is not None:
            loss = loss + self.lam_ewc * (self._ew(self._F_h.to(dev)) * (model.h - self._h_star.to(dev)) ** 2).sum()
        if self.no_slao and self._F_W_A is not None:
            loss = loss + self.lam_ewc * (self._ew(self._F_W_A.to(dev)) * (model.W_A - self._W_A_star.to(dev)) ** 2).sum()
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

        # ── Fisher on all named parameters ──────────────────────────────────
        label = "sig" if use_sig_fisher else "pred"
        print(f"  [SLAO] computing {label}-Fisher (task {t}) …")
        raw   = signature_fisher(model, **self.sig_kw) if use_sig_fisher \
                else pred_fisher(model, data, device=device)
        F_B   = _normalize(raw.get("W_B", torch.zeros_like(model.W_B)))
        F_W_A = _normalize(raw.get("W_A", torch.zeros_like(model.W_A)))
        F_A   = _normalize(raw.get("A",   torch.zeros_like(model.A)))
        F_h   = _normalize(raw.get("h",   torch.zeros_like(model.h)))

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

        # W_A EWC — only for no_slao (W_A is not QR-overwritten in this mode)
        if self.no_slao:
            if self._F_W_A is None:
                self._F_W_A    = F_W_A.clone()
                self._W_A_star = model.W_A.data.clone()
            else:
                self._F_W_A    = self._F_W_A    + lam_t * (F_W_A          - self._F_W_A)
                self._W_A_star = self._W_A_star + lam_t * (model.W_A.data - self._W_A_star)

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

        # ── Per-class module store (piagets_class_b) ─────────────────────────
        # Store (W_B, W_A) per class — avoids the A_merge mismatch where global
        # W_A (from most recent task) is mismatched with class-specific W_B at eval.
        if self.use_class_b and true_class is not None:
            alpha_c = lam_t * (1.0 - self._F_B.clamp(0.0, 1.0))
            if true_class not in self._class_modules:
                self._class_modules[true_class] = {
                    'W_B': model.W_B.data.clone(),
                    'W_A': model.W_A.data.clone(),
                }
            else:
                old = self._class_modules[true_class]
                self._class_modules[true_class] = {
                    'W_B': old['W_B'] + alpha_c * (model.W_B.data - old['W_B']),
                    'W_A': old['W_A'] + lam_t   * (model.W_A.data - old['W_A']),
                }

        # ── Orthogonal joint basis update (piagets_ortho) ────────────────────
        # Adds top Fisher-weighted singular directions of this task's W_B to a
        # shared joint basis, capped at r//2 total dirs. Replaces per-class bases
        # which exhausted the full rank after n_classes × ortho_k > r.
        if self.use_ortho:
            self._update_joint_basis(model.W_B.data, self._F_B)

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

    # ── Inference adapters (SLAO line 8) ─────────────────────────────────────

    def restore_merged(self, model):
        """Set W_B = B_merged (global EMA), W_A = A_merge."""
        if self.B_merged is not None and self.A_merge is not None:
            with torch.no_grad():
                model.W_B.data.copy_(self.B_merged)
                model.W_A.data.copy_(self.A_merge)

    def restore_merged_class(self, model, class_id: int):
        """Set (W_B, W_A) from the per-class module for class_id.
        Both factors are class-specific — no A_merge mismatch.
        Falls back to global restore_merged if class not yet seen."""
        if self.use_class_b and class_id in self._class_modules:
            mod = self._class_modules[class_id]
            with torch.no_grad():
                model.W_B.data.copy_(mod['W_B'])
                model.W_A.data.copy_(mod['W_A'])
        else:
            self.restore_merged(model)

    # ── Class-B probe ────────────────────────────────────────────────────────

    @torch.no_grad()
    def probe_from_class(self, model, data, true_class=None,
                         seq_len: int = 100, n_batches: int = 20,
                         alpha: float = 0.05):
        """For class_b: evaluate the class-specific module on the data.
        If true_class is already known, checks that module's MSE directly.
        Falls back to probe_from_task_start for unseen classes."""
        if true_class in self._class_modules and self._mse_max_seen is not None:
            mod = self._class_modules[true_class]
            orig_WB = model.W_B.data.clone()
            orig_WA = model.W_A.data.clone()
            model.W_B.data.copy_(mod['W_B'])
            model.W_A.data.copy_(mod['W_A'])
            mse = _eval_mse_quick(model, data, seq_len=seq_len,
                                  n_batches=n_batches, alpha=alpha)
            model.W_B.data.copy_(orig_WB)
            model.W_A.data.copy_(orig_WA)
            ratio = mse / (self._mse_max_seen + 1e-8)
            mode  = "assimilation" if ratio < self.assim_mse_ratio else "accommodation"
            lam   = self.lam_assim if mode == "assimilation" else self.lam_accom
            print(f"  [probe_class] cls={true_class}  mse={mse:.4f}"
                  f"  ratio={ratio:.2f}  → {mode.upper()}  λ={lam:.1f}")
            return lam, mode, true_class
        return self.probe_from_task_start(model, data, seq_len=seq_len,
                                          n_batches=n_batches, alpha=alpha)

    # ── Orthogonal subspace helpers ───────────────────────────────────────────

    def _update_joint_basis(self, W_B: torch.Tensor, F_B: torch.Tensor):
        """Add Fisher-weighted singular directions of W_B to the joint protected basis.

        Uses Gram-Schmidt to orthogonalize new directions against existing ones,
        capped at budget = r // 2 total directions. This avoids exhausting the
        full LoRA rank when n_classes × ortho_k >= r.
        """
        r      = W_B.shape[1]
        budget = max(r // 2, 1)
        k      = min(self._ortho_k, r)

        col_norms = (F_B.sum(dim=0) + 1e-8).sqrt()
        W_sc  = (W_B * col_norms.unsqueeze(0)).float()
        U_new, _, _ = torch.linalg.svd(W_sc, full_matrices=False)
        U_new = U_new[:, :k].detach()           # (M, k) candidate directions

        if self._U_joint is None:
            n_take = min(k, budget)
            self._U_joint = U_new[:, :n_take].clone()
        else:
            for j in range(U_new.shape[1]):
                if self._U_joint.shape[1] >= budget:
                    break
                v    = U_new[:, j].to(self._U_joint.device)
                v    = v - self._U_joint @ (self._U_joint.t() @ v)
                norm = v.norm()
                if norm > 1e-4:                 # only add truly new direction
                    self._U_joint = torch.cat(
                        [self._U_joint, (v / norm).unsqueeze(1)], dim=1)

    def apply_ortho_grad_mask(self, model):
        """Project W_B.grad onto the orthogonal complement of the joint schema basis.

        grad ← grad − U (U^T grad)

        Removes gradient components in the directions most important for all
        previously learned schemas, making accommodation structurally interference-free.
        Budget-capped at r//2 directions so new schemas always have room to learn.

        Call AFTER loss.backward(), BEFORE opt.step(), during accommodation only.
        """
        if not self.use_ortho or self._U_joint is None or model.W_B.grad is None:
            return
        U    = self._U_joint.to(model.W_B.grad.device)
        grad = model.W_B.grad
        model.W_B.grad.data.copy_(grad - U @ (U.t() @ grad))


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_with_ewc(model, data, cl=None, epochs: int = 200, seq_len: int = 100,
                   batch: int = 64, lr: float = 1e-3, alpha: float = 0.5,
                   alpha_end: float = 0.05, reg_lambda: float = 0.0,
                   device: str = "cpu", epoch_callback=None, log=print,
                   regime: str | None = None, current_class=None,
                   accommodation: bool = False,
                   epoch_log: list | None = None,
                   n_warmup: int = 0,
                   anneal_frac: float = 0.0):
    """GTF-BPTT with optional EWC + φ-reg penalties, dual-LoRA masking, ortho masking.

    accommodation=True activates orthogonal gradient masking (if cl.use_ortho).
    epoch_log: if a list is passed, per-epoch reconstruction MSE is appended to it.

    n_warmup: first N epochs train with λ=0 and W_B frozen (W_A warm-start after QR-init).
    anneal_frac: fraction of post-warmup epochs in which λ decays linearly to 0 (final phase).
      This lets the model reach near-baseline MSE while still having received EWC protection
      for the majority of training.  The anchor for the NEXT task is stored from the final
      (annealed) model, which has lower MSE but slightly less EWC-constrained parameters.
    """
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
    use_ortho   = (cl is not None and cl.use_ortho and accommodation)

    # Compute epoch boundaries for the three-phase schedule:
    #   [0, n_warmup)      — warmup: λ=0, W_B frozen
    #   [n_warmup, ewc_end) — full EWC at λ_target
    #   [ewc_end, epochs)   — anneal: λ linearly → 0
    n_post    = epochs - n_warmup
    n_anneal  = int(round(n_post * anneal_frac))
    ewc_end   = epochs - n_anneal   # absolute epoch where anneal begins

    for ep in range(epochs):
        # --- λ schedule ---
        if ep < n_warmup:
            lam_scale = 0.0
            in_warmup = True
        elif ep < ewc_end:
            lam_scale = 1.0
            in_warmup = False
        else:
            steps_in  = ep - ewc_end
            lam_scale = 1.0 - (steps_in + 1) / n_anneal if n_anneal > 0 else 0.0
            in_warmup = False

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
            if cl is not None and lam_scale > 0.0:
                loss_e = cl.ewc_loss(model, current_class) * lam_scale
            else:
                loss_e = torch.tensor(0.0)
            loss_p = (cl.phi_reg_loss(model, current_class)
                      if use_phi_reg else torch.tensor(0.0))
            loss   = loss_r + loss_e + loss_p
            opt.zero_grad()
            loss.backward()
            if in_warmup and model.W_B.grad is not None:
                model.W_B.grad.zero_()   # freeze W_B during warm-start
            if use_block:
                cl.apply_block_mask(model, regime)
            if use_ortho:
                cl.apply_ortho_grad_mask(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot_r += loss_r.item() * n
            tot_e += loss_e.item() * n
            tot_p += loss_p.item() * n
            n_seen += n
        tot_r /= n_seen
        tot_e /= n_seen
        tot_p /= n_seen
        if epoch_log is not None:
            epoch_log.append(tot_r)
        if epoch_callback is not None:
            epoch_callback(ep, tot_r, tot_e)
        if (ep + 1) % 50 == 0 or ep == 0:
            log(f"  ep {ep + 1:3d}/{epochs}  recon={tot_r:.5f}  ewc={tot_e:.5f}"
                + (f"  φ={lam_scale:.2f}" if (n_warmup > 0 or anneal_frac > 0) else "")
                + (f"  phi_reg={tot_p:.5f}" if use_phi_reg else ""))
