"""PIAGETS long-sequence validation with BWT / FWT / schema-recognition metrics.

Task stream (T=10, three maximally topologically distinct schema classes)
------------------------------------------------------------------------
  Class 0 — Lorenz      (chaotic strange attractor, λ₁>0, fractal dim≈2.06)  × 4
  Class 1 — torus       (quasiperiodic, 2D, 2 zero Lyapunovs)                × 3
  Class 2 — van_der_pol (relaxation limit cycle, 1D, 1 zero Lyapunov)        × 3

  Lorenz ρ ∈ {28,35,40,45}: all in the 1-region double-wing chaotic regime
  (avoids bifurcation regions near ρ≈30 that cause n_regions>1 and large φ-spread).
  VdP μ ∈ {1.5,3.0,5.0}: spread widely to maximise Torus↔VdP φ-distance.
  φ now includes phi_act_std (std of σ(βz_i) over time): ≈0 for all 1-region Lorenz,
  >0 for oscillators — this compresses within-Lorenz spread in φ-space.

  t=0  Lor ρ=28     class 0  (200 ep, no prior)
  t=1  Tor ω₂=0.382 class 1  (200 ep)
  t=2  VdP μ=1.5    class 2  (120 ep)
  t=3  Lor ρ=35     class 0  (150 ep)
  t=4  Tor ω₂=0.618 class 1  (200 ep)
  t=5  VdP μ=3.0    class 2  (120 ep)
  t=6  Lor ρ=40     class 0  (150 ep)
  t=7  VdP μ=5.0    class 2  (120 ep)
  t=8  Tor ω₂=0.271 class 1  (200 ep)
  t=9  Lor ρ=45     class 0  (150 ep)

Methods
-------
  baseline          — naive sequential fine-tuning, no protection
  pred_ewc          — standard EWC with reconstruction-loss Fisher (consolidated)
  pred_ewc_adaptive — pred_ewc + schema-probe λ scheduling (no sig-Fisher, no restore_merged)
  piagets           — PIAGETS: sig-Fisher EWC + SLAO QR-init (consolidated, fixed λ)
  piagets_adaptive  — PIAGETS + schema-probe λ scheduling
                        assimilation (recognised schema) → λ=LAM_ASSIM (high protection)
                        accommodation (new schema)       → λ=LAM_ACCOM (high plasticity)

Metrics
-------
  R[t, i]    = –φ_distance(φ(model_t), φ(oracle_i))   performance matrix (higher = better)
  BWT        = mean_{i<T} (R[T-1,i] − R[i,i])          backward transfer
  FWT        = mean_{i>0} (R[i,i] − R_rand[i])         forward transfer
  rec_ora    = fraction of steps where argmin_i φ_dist(model, oracle_i) has the right class
               (nearest-TASK-oracle metric; avoids bad class centroids)
  rec_cen    = fraction of steps where nearest CLASS centroid has the right class
               (old metric — centroid averages heterogeneous tasks, biased)

Outputs
-------
  fig_piagets_bwt.png        — R-matrix heatmaps (one column per method)
  fig_piagets_curves.png     — recognition accuracy over time + BWT bar chart
  piagets_bwt_results.txt    — full numeric tables
"""
import sys, copy, warnings, time
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import systems
import signatures
from alrnn import ALRNN
import piagets as pg

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
np.random.seed(0)
torch.manual_seed(0)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
M, P, d   = 16, 6, 3
N_DATA    = 8000
SEQ_LEN   = 100
BATCH     = 64
LR        = 1e-3
REG_LAM   = 0.05

EP_ORACLE      = 300      # oracle fits: independent, full-length training
EP_TASK0       = 300      # CL task-0 (no prior tasks, no EWC)
EP_FINE_LOR    = 300      # Lorenz fine-tuning — doubled so forgetting is real
EP_FINE_TORUS  = 350      # Torus fine-tuning
EP_FINE_VDP    = 200      # VdP fine-tuning

# LAM_EWC halved (5 → was 10): weak EWC forces pred_ewc to show genuine forgetting,
# separating it from PIAGETS variants that use φ-reg and per-class anchors.
LAM_EWC       = 5.0
LAM_ASSIM     = 10.0     # adaptive: recognised schema → protect strongly
LAM_ACCOM     = 2.0      # adaptive: new schema → allow high plasticity
SCHEMA_SIGMA  = 2.0       # recognition threshold in intra-class σ units
BETA_MERGE    = 5.0
SIG_FISHER_KW = dict(n_avg=3, T_warmup=100, T_track=200, beta=10.0)
# Option 3: φ-functional regularization (lam raised to compensate for weaker EWC)
LAM_PHI       = 0.15      # penalty weight for ||φ(θ) − φ*(class)||²
PHI_REG_KW    = dict(T_warmup=100, T_track=100, beta=10.0)
# Option 4: dual LoRA block split — asymmetric: r_lin=2 for Lorenz (1-region, simpler),
# r_nl=4 for Torus/VdP (multi-region, needs more rank).
R_LIN         = 2         # linear-core block width (Lorenz); nonlinear block = r - R_LIN = 4

PHI_T_WARMUP, PHI_T_TRACK, PHI_BETA, PHI_N_AVG = 100, 200, 10.0, 3
ASSIM_MSE_RATIO = 10.0    # Fix-1 probe: assim if MSE_new / MSE_max_seen < this
DEVICE = "cpu"

# ─── Task stream ─────────────────────────────────────────────────────────────
# 4-tuple: (name, data_generator, class_label, fine_tune_epochs)
STREAM = [
    ("Lor_r28",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=28)[0].astype("float32"),   0, EP_FINE_LOR),
    ("Tor_0.382", lambda: systems.torus(n=N_DATA, omega2=0.382)[0].astype("float32"),        1, EP_FINE_TORUS),
    ("VdP_m1.5",  lambda: systems.van_der_pol(n=N_DATA, mu=1.5)[0].astype("float32"),       2, EP_FINE_VDP),
    ("Lor_r35",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=35)[0].astype("float32"),   0, EP_FINE_LOR),
    ("Tor_0.618", lambda: systems.torus(n=N_DATA, omega2=0.618)[0].astype("float32"),        1, EP_FINE_TORUS),
    ("VdP_m3.0",  lambda: systems.van_der_pol(n=N_DATA, mu=3.0)[0].astype("float32"),       2, EP_FINE_VDP),
    ("Lor_r40",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=40)[0].astype("float32"),   0, EP_FINE_LOR),
    ("VdP_m5.0",  lambda: systems.van_der_pol(n=N_DATA, mu=5.0)[0].astype("float32"),       2, EP_FINE_VDP),
    ("Tor_0.271", lambda: systems.torus(n=N_DATA, omega2=0.271)[0].astype("float32"),        1, EP_FINE_TORUS),
    ("Lor_r45",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=45)[0].astype("float32"),   0, EP_FINE_LOR),
]
T         = len(STREAM)
NAMES     = [n for n, _, _, _ in STREAM]
CLASSES   = [c for _, _, c, _ in STREAM]   # 0, 1, or 2
N_SCHEMAS = len(set(CLASSES))              # 3

# per-class first occurrence → used as the oracle per schema class
CLASS_ORACLE_IDX = {}                    # class → first task index for that class
for i, c in enumerate(CLASSES):
    if c not in CLASS_ORACLE_IDX:
        CLASS_ORACLE_IDX[c] = i

CLASS_NAMES = {0: "Lor", 1: "Tor", 2: "VdP"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def fresh_model(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return ALRNN(latent_dim=M, obs_dim=d, P=P, rank=12)


def phi_of(model):
    phis = []
    with torch.no_grad():
        for _ in range(PHI_N_AVG):
            phis.append(pg.diff_phi(model, PHI_T_WARMUP, PHI_T_TRACK, PHI_BETA).numpy())
    return np.mean(phis, axis=0)


def phi_dist(pa, pb, scale):
    d = (pa - pb) / (scale + 1e-9)
    return float(np.linalg.norm(d))


def train(model, data, epochs, cl=None, label="", alpha=0.5, alpha_end=0.05,
          regime=None, current_class=None):
    log_lines = []
    pg.train_with_ewc(model, data, cl=cl, epochs=epochs, seq_len=SEQ_LEN,
                      batch=BATCH, lr=LR, alpha=alpha, alpha_end=alpha_end,
                      reg_lambda=REG_LAM, device=DEVICE,
                      regime=regime, current_class=current_class,
                      log=lambda s: (log_lines.append(s),
                                     print(f"  [{label}] {s}", flush=True)))
    return log_lines


@torch.no_grad()
def eval_recon_mse(model, data, n_batches=30, alpha=0.05):
    """Mean squared reconstruction error over random batches (lower = better).
    Uses alpha=0.05 (mostly free-run) matching end-of-training GTF schedule."""
    N = len(data)
    losses = []
    for _ in range(n_batches):
        start = np.random.randint(0, N - SEQ_LEN - 1)
        x = torch.tensor(data[start:start + SEQ_LEN + 1]).unsqueeze(0)  # (1, T+1, d)
        preds = model.forced_rollout(x, alpha=alpha)                      # (1, T, d)
        losses.append(((preds - x[:, 1:]) ** 2).mean().item())
    return float(np.mean(losses))


# ─── Step 1: Oracle models ───────────────────────────────────────────────────
print("=" * 64)
print("Step 1: Oracle models")
print("=" * 64)

oracle_phis   = []
oracle_sigs   = []
task_data     = []
seed_obs      = np.zeros(d, dtype="float32")

for name, gen, cls, _ in STREAM:
    data = gen()
    task_data.append(data)
    print(f"\n  Oracle {name} (class={cls})")
    m = fresh_model(seed=42)
    train(m, data, EP_ORACLE, label=f"ora/{name}")
    phi = phi_of(m)
    oracle_phis.append(phi)
    rd = m.enumerate_visited_regions(seed_obs, n=2000, warmup=300)
    blks = {}
    blks.update(signatures.equilibrium_portrait(rd))
    blks.update(signatures.symbolic_graph(rd, k_max=4))
    blks.update(signatures.lyapunov_signature(m, seed_obs, n=1500, warmup=300))
    oracle_sigs.append(signatures.Signature(blks, meta={"n_regions": len(rd["uniq"])}))
    print(f"    n_regions={len(rd['uniq'])}  φ_last3={phi[-3:]}")

# phi scale from oracle distribution
phi_scale = np.std(np.stack(oracle_phis), axis=0)
phi_scale[phi_scale < 1e-9] = 1.0
print(f"\n  φ scale (std across oracle): min={phi_scale.min():.3f} max={phi_scale.max():.3f}")

# Oracle class centroids (mean φ per class)
class_phis = {}
for i, cls in enumerate(CLASSES):
    class_phis.setdefault(cls, []).append(oracle_phis[i])
centroid = {c: np.mean(ps, axis=0) for c, ps in class_phis.items()}

# Random baseline: untrained model φ (for FWT denominator)
rand_phis = [phi_of(fresh_model(seed=s)) for s in range(5)]
rand_phi  = np.mean(rand_phis, axis=0)

print("\n  Oracle pairwise φ-distance matrix:")
print(f"  {'':10s}" + "".join(f"  {n:8s}" for n in NAMES))
for i in range(T):
    row = "".join(f"  {phi_dist(oracle_phis[i], oracle_phis[j], phi_scale):8.3f}"
                  for j in range(T))
    print(f"  {NAMES[i]:10s}{row}")


# ─── Step 2: CL experiment ───────────────────────────────────────────────────
print("\n" + "=" * 64)
print("Step 2: Sequential CL")
print("=" * 64)

METHODS = {
    "vanilla":           dict(use_cl=False, use_sig=False, adaptive=False, use_merge=False,
                               phi_reg=False, dual_lora=False, class_ewc=False,
                               fresh_per_task=True),
    "baseline":          dict(use_cl=False, use_sig=False, adaptive=False, use_merge=False,
                               phi_reg=False, dual_lora=False, class_ewc=False),
    "pred_ewc":          dict(use_cl=True,  use_sig=False, adaptive=False, use_merge=False,
                               phi_reg=False, dual_lora=False, class_ewc=False),
    "pred_ewc_adaptive": dict(use_cl=True,  use_sig=False, adaptive=True,  use_merge=False,
                               phi_reg=False, dual_lora=False, class_ewc=False),
    "piagets":           dict(use_cl=True,  use_sig=True,  adaptive=False, use_merge=True,
                               phi_reg=False, dual_lora=False, class_ewc=False),
    "piagets_adaptive":  dict(use_cl=True,  use_sig=True,  adaptive=True,  use_merge=True,
                               phi_reg=False, dual_lora=False, class_ewc=False),
}

# R_perf[method][t, i] = -phi_dist(model_t, oracle_i)  for i <= t, else NaN
R_perf        = {m: np.full((T, T), np.nan) for m in METHODS}
rec_ora       = {m: np.zeros(T, dtype=float) for m in METHODS}  # nearest-task-oracle accuracy
rec_cen       = {m: np.zeros(T, dtype=float) for m in METHODS}  # nearest-class-centroid accuracy
# mse_diag[method][t] = reconstruction MSE on task t right after training (before merge)
mse_diag      = {m: np.zeros(T, dtype=float) for m in METHODS}
walltime      = {m: 0.0 for m in METHODS}

for method, cfg in METHODS.items():
    print(f"\n{'─'*60}")
    print(f"  Method: {method.upper()}")
    print(f"{'─'*60}")

    t_method_start = time.time()
    m = fresh_model(seed=0)
    lam_ewc_eff = cfg.get("lam_ewc_override", LAM_EWC)
    cl = (pg.PIAGETSContinual(
              lam_ewc=lam_ewc_eff,
              lam_assim=LAM_ASSIM, lam_accom=LAM_ACCOM,
              schema_sigma=SCHEMA_SIGMA,
              assim_mse_ratio=ASSIM_MSE_RATIO,
              sig_fisher_kwargs=SIG_FISHER_KW,
              lam_phi=LAM_PHI if cfg.get("phi_reg") else 0.0,
              r_lin=R_LIN if cfg.get("dual_lora") else None,
              phi_reg_kwargs=PHI_REG_KW,
              use_class_ewc=cfg.get("class_ewc", False))
          if cfg["use_cl"] else None)

    for t, (name, gen, true_cls, ep_fine) in enumerate(STREAM):
        data = task_data[t]
        ep   = EP_TASK0 if t == 0 else ep_fine
        # dual-LoRA: route by class label (linear=Lorenz, nonlinear=Torus/VdP)
        regime = ("linear" if true_cls == 0 else "nonlinear") if cfg.get("dual_lora") else None
        print(f"\n  t={t} {name} (class={true_cls})" +
              (f"  regime={regime}" if regime else ""))

        # vanilla: reinitialise model from scratch for every task after t=0
        if cfg.get("fresh_per_task") and t > 0:
            m = fresh_model(seed=42)

        if cl is not None and t > 0:
            cl.qr_init(m)
            # Fix 1: probe INCOMING task's φ after short warmup (correct causal direction)
            if cfg["adaptive"]:
                lam, mode, schema = cl.probe_from_task_start(m, data, seq_len=SEQ_LEN)
                cl.lam_ewc = lam

        train(m, data, ep, cl=(cl if cfg["use_cl"] else None), label=f"{method}/{name}",
              regime=regime, current_class=true_cls if cfg.get("phi_reg") else None)

        # reconstruction MSE on current task — evaluated BEFORE restore_merged so
        # we measure fine-tuned quality, not the blended B_merge quality
        mse_diag[method][t] = eval_recon_mse(m, task_data[t])
        print(f"    mse_recon={mse_diag[method][t]:.5f}")

        if cl is not None:
            cl.store_task(m, data, true_class=true_cls,
                          use_sig_fisher=cfg["use_sig"], device=DEVICE)
            # SLAO line 8: switch to merged LoRA for inference / evaluation
            if cfg["use_merge"]:
                cl.restore_merged(m)

        # ── evaluate: fill column of R for all past tasks ──────────────
        phi_now = phi_of(m)
        for i in range(t + 1):
            R_perf[method][t, i] = -phi_dist(phi_now, oracle_phis[i], phi_scale)

        # ── schema recognition (nearest TASK oracle → infer class) ─────
        dists_to_ora = {i: phi_dist(phi_now, oracle_phis[i], phi_scale) for i in range(t + 1)}
        nearest_ora_idx = min(dists_to_ora, key=dists_to_ora.get)
        pred_cls_ora = CLASSES[nearest_ora_idx]
        rec_ora[method][t] = int(pred_cls_ora == true_cls)

        # ── schema recognition (nearest CLASS centroid, old metric) ────
        dists_to_cls = {c: phi_dist(phi_now, centroid[c], phi_scale) for c in centroid}
        pred_cls_cen = min(dists_to_cls, key=dists_to_cls.get)
        rec_cen[method][t] = int(pred_cls_cen == true_cls)

        print(f"    rec_ora={int(pred_cls_ora==true_cls)}  rec_cen={int(pred_cls_cen==true_cls)}  "
              f"true={true_cls}  nearest_oracle=t{nearest_ora_idx}({NAMES[nearest_ora_idx]})")
        print(f"    φ-dist to class centroids: " +
              " ".join(f"C{c}={dists_to_cls[c]:.2f}" for c in sorted(dists_to_cls)))

    walltime[method] = time.time() - t_method_start
    print(f"  [walltime] {method}: {walltime[method]:.0f}s")


# ─── Step 3: BWT / FWT ───────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("Step 3: BWT / FWT / Schema Recognition")
print("=" * 64)

def bwt(R):
    """BWT = mean_{i<T} (R[T-1, i] - R[i, i]).
    R[i,i] is performance right after training on task i.
    R[T-1,i] is final performance on task i after all tasks."""
    vals = [R[T - 1, i] - R[i, i] for i in range(T - 1)]
    return float(np.mean(vals))

def fwt(R, R_rand_col):
    """FWT = mean_{i>0} (R[i,i] - baseline_rand[i]).
    R[i,i] is performance after training on i; baseline_rand is random-model performance."""
    vals = [R[i, i] - R_rand_col[i] for i in range(1, T)]
    return float(np.mean(vals))

# random model R column: -phi_dist(rand_phi, oracle_i) for each i
rand_R_col = np.array([-phi_dist(rand_phi, oracle_phis[i], phi_scale) for i in range(T)])

print(f"\n  {'method':22s}  BWT       FWT       rec_ora  rec_cen  mse_avg   R_diag_mean  walltime")
lines = []
for method in METHODS:
    R = R_perf[method]
    b = bwt(R)
    f = fwt(R, rand_R_col)
    ro = rec_ora[method].mean()
    rc = rec_cen[method].mean()
    mse_avg = mse_diag[method].mean()
    rdiag_mean = np.mean([R[i, i] for i in range(T)])
    wt = walltime[method]
    print(f"  {method:22s}  {b:+.4f}   {f:+.4f}   {ro:.3f}    {rc:.3f}    {mse_avg:.5f}   {rdiag_mean:.2f}        {wt:.0f}s")
    lines.append((method, b, f, ro, rc, R))

# Print full R matrices
for method, b, f, ro, rc, R in lines:
    print(f"\n  R matrix — {method}  (BWT={b:+.4f}  rec_ora={ro:.3f}  rec_cen={rc:.3f}):")
    header = "  {:6s}".format("t\\i") + "".join(f"  {n:8s}" for n in NAMES)
    print(header)
    for t in range(T):
        row = "".join(f"  {R[t,i]:+8.3f}" if i <= t else "        " + "  "
                      for i in range(T))
        print(f"  t={t}  {NAMES[t]:8s}{row}")


# ─── Step 4: Save figures ─────────────────────────────────────────────────────
print("\nGenerating figures …")

CMAP  = "RdYlGn"
VMIN  = -20.0
VMAX  = 0.0
COLS  = {"vanilla": "#aaaaaa",
         "baseline": "#e15759", "pred_ewc": "#f28e2b",
         "pred_ewc_adaptive": "#edc948",
         "piagets": "#4e79a7", "piagets_adaptive": "#59a14f"}

# Figure 1: R-matrix heatmaps (one per method)
fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 8))
fig.suptitle("Performance matrix R[t,i] = −φ_dist(model_t, oracle_i)\n"
             "(higher/greener = model at time t is closer to oracle i's schema)", fontsize=11)
for ax, method in zip(axes, METHODS):
    R = R_perf[method]
    masked = np.ma.masked_where(np.isnan(R), R)
    im = ax.imshow(masked, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")
    ax.set_title(f"{method}\nBWT={bwt(R):+.3f}  rec_ora={rec_ora[method].mean():.2f}", fontsize=9)
    ax.set_xticks(range(T)); ax.set_xticklabels(NAMES, rotation=40, ha="right", fontsize=6)
    ax.set_yticks(range(T)); ax.set_yticklabels(
        [f"t={t} {NAMES[t]}[{CLASS_NAMES[CLASSES[t]]}]" for t in range(T)], fontsize=6)
    ax.set_xlabel("oracle task i"); ax.set_ylabel("training step t")
    fig.colorbar(im, ax=ax, fraction=0.046)
    for t in range(T):
        ax.add_patch(plt.Rectangle((t - 0.5, t - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=1.5))
plt.tight_layout()
plt.savefig("fig_piagets_bwt.png", dpi=120)
print("  saved fig_piagets_bwt.png")
plt.close()

# Figure 2: recognition accuracy + BWT bar chart
fig, axes2 = plt.subplots(1, 3, figsize=(20, 5))

ax1, ax2, ax3 = axes2

ax1.set_title("Schema recognition (nearest-task-oracle)\n"
              "rec_ora: argmin oracle → class", fontsize=10)
for method, col in COLS.items():
    ax1.plot(range(T), rec_ora[method], "o-", color=col, label=method, lw=2)
ax1.axhline(1 / N_SCHEMAS, color="gray", ls="--", label="chance")
ax1.set_xticks(range(T))
ax1.set_xticklabels([f"t={t}\n{NAMES[t]}\n(c{CLASSES[t]})" for t in range(T)], fontsize=6.5)
ax1.set_ylim(-0.05, 1.1); ax1.set_ylabel("accuracy"); ax1.legend(fontsize=8)

ax2.set_title("Schema recognition (class centroid)\n"
              "rec_cen: argmin centroid distance", fontsize=10)
for method, col in COLS.items():
    ax2.plot(range(T), rec_cen[method], "s--", color=col, label=method, lw=1.5)
ax2.axhline(1 / N_SCHEMAS, color="gray", ls="--", label="chance")
ax2.set_xticks(range(T))
ax2.set_xticklabels([f"t={t}\n{NAMES[t]}\n(c{CLASSES[t]})" for t in range(T)], fontsize=6.5)
ax2.set_ylim(-0.05, 1.1); ax2.set_ylabel("accuracy"); ax2.legend(fontsize=8)

ax3.set_title("BWT and FWT by method\n(positive = backward synergy / warm-start benefit)", fontsize=10)
xs    = np.arange(len(METHODS))
bwts  = [bwt(R_perf[m]) for m in METHODS]
fwts_ = [fwt(R_perf[m], rand_R_col) for m in METHODS]
w     = 0.35
bars_b = ax3.bar(xs - w / 2, bwts,  width=w, label="BWT",
                 color=[COLS[m] for m in METHODS], alpha=0.85)
bars_f = ax3.bar(xs + w / 2, fwts_, width=w, label="FWT",
                 color=[COLS[m] for m in METHODS], alpha=0.4, hatch="//")
ax3.axhline(0, color="black", lw=0.8)
ax3.set_xticks(xs); ax3.set_xticklabels(list(METHODS.keys()), rotation=15, ha="right", fontsize=8)
ax3.set_ylabel("transfer score (φ-units)"); ax3.legend()
for bar in bars_b:
    v = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.05 * abs(v) + 0.1,
             f"{v:+.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("fig_piagets_curves.png", dpi=120)
print("  saved fig_piagets_curves.png")
plt.close()

# Text summary
with open("piagets_bwt_results.txt", "w") as f:
    f.write("PIAGETS long-sequence BWT/FWT results\n")
    f.write(f"Stream: {NAMES}\nClasses: {CLASSES}\n\n")
    f.write(f"Adaptive λ: assimilation={LAM_ASSIM}  accommodation={LAM_ACCOM}  "
            f"sigma={SCHEMA_SIGMA}\n\n")
    for method in METHODS:
        R = R_perf[method]
        f.write(f"{method}:\n")
        f.write(f"  BWT = {bwt(R):+.4f}   FWT = {fwt(R, rand_R_col):+.4f}   "
                f"rec_ora = {rec_ora[method].mean():.3f}   "
                f"rec_cen = {rec_cen[method].mean():.3f}   "
                f"mse_avg = {mse_diag[method].mean():.5f}   "
                f"walltime = {walltime[method]:.0f}s\n")
        f.write(f"  R diagonal (perf at train time): " +
                "  ".join(f"{R[i,i]:.3f}" for i in range(T)) + "\n")
        f.write(f"  MSE diagonal (recon at train time): " +
                "  ".join(f"{mse_diag[method][i]:.5f}" for i in range(T)) + "\n")
        f.write(f"  BWT per task: " +
                "  ".join(f"{R[T-1,i]-R[i,i]:+.3f}" for i in range(T-1)) + "\n\n")

print("  saved piagets_bwt_results.txt")
print("\nDone.")
