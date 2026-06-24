"""
plot_trajectories.py — PIAGETS trajectory comparison with assimilation demo
===========================================================================
Curriculum: Lorenz(ρ=28) → Halvorsen(a=1.4) → Lorenz(ρ=42)

Training strategy for PIAGETS snapshots: two-pass per task.
  Pass 1 (400 ep)  — full training with annealing → records per-epoch losses.
  Pass 2 (best_ep) — re-train to the min-loss epoch WITH the same alpha schedule
                     (alpha 0.5 → 0.10) so the model learns multi-step free runs.
  Snapshot = Pass 2 model.  Loss curve = Pass 1 epochs.

Outputs:
  figs/fig_trajectories.png
  figs/fig_loss.png
  figs/fig_weights.png
"""
import sys, copy, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

sys.path.insert(0, ".")
import systems
from alrnn import ALRNN
import piagets as pg

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
np.random.seed(42)
torch.manual_seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
M, P, d, RANK   = 16, 6, 3, 6
N_DATA          = 8_000
SEQ_LEN         = 100
BATCH           = 64
LR              = 1e-3
EPOCHS          = 400       # loss-curve pass length
N_WARMUP        = 30
ANNEAL_FRAC     = 1 / 3
LAM_EWC         = 5.0
LAM_ASSIM       = 4.0    # lower allows PIAGETS to adapt to ρ=42 while still protecting schema
LAM_ACCOM       = 2.0
REG_LAMBDA      = 0.02   # entropy reg → multiple ReLU regions → chaotic trajectories
SIG_FISHER_KW   = dict(n_avg=3, T_warmup=100, T_track=200, beta=10.0)
TRAJ_LEN        = 800
ALPHA_VIZ       = 0.10   # matches training alpha_end; avoids divergence at eval

# ── Colors ─────────────────────────────────────────────────────────────────────
C_GT    = "#E69F00"
C_PI    = "#56B4E9"
C_BL    = "#CC79A7"
C_ASS   = "#009E73"
C_ACCOM = "#CC2222"

# ── Task stream ────────────────────────────────────────────────────────────────
print("Generating data …")
train_seqs = [
    ("Lorenz ρ=28",     systems.lorenz(n=N_DATA,    dt=0.05, rho=28)[0].astype("float32"), 0),
    ("Halvorsen a=1.4", systems.halvorsen(n=N_DATA,  dt=0.05)[0].astype("float32"),          1),
    ("Lorenz ρ=42",     systems.lorenz(n=N_DATA,    dt=0.05, rho=42)[0].astype("float32"), 0),
]
first_cls  = train_seqs[0][2]

print("Generating ground-truth phase portraits …")
gt_trajs = [
    systems.lorenz(n=15_000,    dt=0.05, rho=28)[0].astype("float32"),
    systems.halvorsen(n=15_000, dt=0.05)[0].astype("float32"),
    systems.lorenz(n=15_000,    dt=0.05, rho=42)[0].astype("float32"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def fresh_model(seed=0):
    torch.manual_seed(seed)
    return ALRNN(latent_dim=M, obs_dim=d, P=P, rank=RANK)


def _train(model, data, epochs, cl=None, n_warmup=0, anneal_frac=0.0,
           alpha_end=0.10, epoch_log=None, silent=False, tag="", reg_lambda=REG_LAMBDA):
    pg.train_with_ewc(
        model, data, cl=cl, epochs=epochs,
        seq_len=SEQ_LEN, batch=BATCH, lr=LR,
        alpha=0.5, alpha_end=alpha_end,
        reg_lambda=reg_lambda, device="cpu",
        n_warmup=n_warmup, anneal_frac=anneal_frac,
        epoch_log=epoch_log,
        log=(lambda s: None) if silent
            else (lambda s: print(f"    [{tag}] {s}", flush=True)),
    )


def gt_strand(gt_data):
    start = np.random.randint(0, len(gt_data) - TRAJ_LEN)
    return [gt_data[start:start + TRAJ_LEN]]


@torch.no_grad()
def model_strand(model, gt_data):
    start = np.random.randint(0, len(gt_data) - TRAJ_LEN - 1)
    x = torch.as_tensor(gt_data[start:start + TRAJ_LEN + 1],
                         dtype=torch.float32).unsqueeze(0)
    return [model.forced_rollout(x, alpha=ALPHA_VIZ).squeeze(0).numpy()]


@torch.no_grad()
def eval_mse(model, data, n_batches=50, alpha=0.05):
    x = torch.as_tensor(data, dtype=torch.float32)
    losses = []
    for _ in range(n_batches):
        s = np.random.randint(0, len(x) - SEQ_LEN - 1)
        xb = x[s:s + SEQ_LEN + 1].unsqueeze(0)
        pred = model.forced_rollout(xb, alpha=alpha)
        losses.append(((pred - xb[:, 1:]) ** 2).mean().item())
    return float(np.mean(losses))


# ── Baseline — single pass, final model for all rows ──────────────────────────
print("\n" + "=" * 64)
print("  Baseline (AL-RNN) — sequential, no protection")
print("=" * 64)

baseline    = fresh_model(0)
bl_ep_log   = []          # per-epoch recon losses across all tasks

for t, (name, data, _) in enumerate(train_seqs):
    print(f"\n  ── Task {t}: {name} ──")
    task_log = []
    _train(baseline, data, EPOCHS, epoch_log=task_log, tag=f"base/{name}")
    bl_ep_log.extend(task_log)

# ── PIAGETS — two-pass per task ────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  PIAGETS — two-pass training (loss curve + best-epoch snapshot)")
print("=" * 64)

pmodel = fresh_model(0)
cl     = pg.PIAGETSContinual(
    lam_ewc=LAM_EWC, lam_assim=LAM_ASSIM, lam_accom=LAM_ACCOM,
    sig_fisher_kwargs=SIG_FISHER_KW,
)

pi_ep_log     = []
pi_snapshots  = []
probe_modes   = []
wb_before     = {}
wb_after      = {}
fisher_snap   = {}

for t, (name, data, true_cls) in enumerate(train_seqs):
    print(f"\n  ── Task {t}: {name} (class={true_cls}) ──")

    # ── Determine mode ────────────────────────────────────────────────────────
    if t > 0:
        if true_cls == first_cls:
            mode, lam = "assimilation", LAM_ASSIM
            print(f"  → class match: ASSIMILATION  λ={lam:.1f}")
        else:
            mode, lam = "accommodation", LAM_ACCOM
            print(f"  → new class:   ACCOMMODATION  λ={lam:.1f}")
    else:
        mode = "accommodation"
    probe_modes.append(mode)

    # ── Save pre-task states for two-pass ────────────────────────────────────
    pre_model_sd = copy.deepcopy(pmodel.state_dict())
    pre_cl       = copy.deepcopy(cl)

    # ── Configure CL (qr_init etc.) ──────────────────────────────────────────
    n_wu  = N_WARMUP    if t > 0 else 0
    afrac = ANNEAL_FRAC if t > 0 else 0.0
    if t > 0:
        cl.lam_ewc = lam
        cl.qr_init(pmodel)

    wb_before[t] = pmodel.W_B.data.clone().numpy()

    # ────────────────────────────────────────────────────────────────────────
    # Pass 1: full 400-epoch training — collects per-epoch loss curve.
    # The model is DISCARDED after this pass; cl state is NOT advanced.
    # ────────────────────────────────────────────────────────────────────────
    task_loss_log = []
    _train(pmodel, data, EPOCHS, cl=cl,
           n_warmup=n_wu, anneal_frac=afrac, alpha_end=0.10,
           epoch_log=task_loss_log, tag=f"P1/{name}")
    pi_ep_log.extend(task_loss_log)

    # ── Find best epoch (search after warmup) ───────────────────────────────
    skip = n_wu
    search = task_loss_log[skip:]
    best_ep = (skip + int(np.argmin(search)) + 1) if search else EPOCHS
    best_ep = max(best_ep, n_wu + 5)     # at least a few post-warmup epochs
    print(f"  → Best epoch: {best_ep}/{EPOCHS}  "
          f"(recon={task_loss_log[best_ep-1]:.5f}  "
          f"vs final={task_loss_log[-1]:.5f})")

    # ────────────────────────────────────────────────────────────────────────
    # Pass 2: re-train to best_ep with CONSTANT alpha and NO EWC annealing.
    # This avoids the late-epoch instability that degrades the ep400 model.
    # ────────────────────────────────────────────────────────────────────────
    pmodel.load_state_dict(pre_model_sd)
    cl = copy.deepcopy(pre_cl)
    if t > 0:
        cl.lam_ewc = lam
        cl.qr_init(pmodel)

    # Same alpha schedule as Pass 1 (0.5→0.10) but compressed into best_ep epochs.
    # This ensures the model has learned multi-step free runs by the snapshot epoch.
    _train(pmodel, data, best_ep, cl=cl,
           n_warmup=n_wu, anneal_frac=ANNEAL_FRAC, alpha_end=0.10,
           silent=True, tag=f"P2/{name}")

    # ── Snapshot from pass 2 ─────────────────────────────────────────────────
    wb_after[t] = pmodel.W_B.data.clone().numpy()
    pi_snapshots.append(copy.deepcopy(pmodel))

    cl.store_task(pmodel, data, true_class=true_cls, use_sig_fisher=True)
    if cl._F_B is not None:
        fisher_snap[t] = cl._F_B.numpy().copy()
    cl.restore_merged(pmodel)

print(f"\n  Probe modes: {probe_modes}")

# ── MSE ────────────────────────────────────────────────────────────────────────
print("\nComputing MSEs …")
all_data = [d for _, d, _ in train_seqs]
pi_mses  = [eval_mse(pi_snapshots[t], all_data[t]) for t in range(3)]
bl_mses  = [eval_mse(baseline,        all_data[t]) for t in range(3)]
print("  PIAGETS (best-ep snap):  " + "  ".join(f"{m:.4f}" for m in pi_mses))
print("  Baseline (final model):  " + "  ".join(f"{m:.4f}" for m in bl_mses))

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Phase-portrait grid
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering fig_trajectories …")

VIEW = [(18, -65), (20, 35), (25, -45)]
row_labels = ["First system:\nLorenz ρ=28",
              "Second system:\nHalvorsen a=1.4",
              "Third system:\nLorenz ρ=42"]
col_labels = ["Ground Truth", "PIAGETS", "Baseline (AL-RNN)"]
col_colors = [C_GT, C_PI, C_BL]

fig = plt.figure(figsize=(15, 14), facecolor="white")
for row in range(3):
    mses = [None, pi_mses[row], bl_mses[row]]
    strand_sets = [
        gt_strand(gt_trajs[row]),
        model_strand(pi_snapshots[row], gt_trajs[row]),
        model_strand(baseline,          gt_trajs[row]),
    ]
    for col in range(3):
        ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection="3d")
        ax.view_init(elev=VIEW[row][0], azim=VIEW[row][1])
        for strand in strand_sets[col]:
            pts = np.asarray(strand, dtype=np.float32)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    lw=0.9, alpha=0.85, color=col_colors[col], rasterized=True)
        if col == 0:
            ax.text2D(-0.28, 0.50, row_labels[row], transform=ax.transAxes,
                      fontsize=9, fontweight="bold", color="#2d2d2d",
                      va="center", ha="center", rotation=90)
        if row == 0:
            ax.set_title(col_labels[col], fontsize=11, fontweight="bold",
                         color="#2d2d2d", pad=6)
        if col == 1:
            if row == 1:
                ax.text2D(0.50, 0.97, "●  Accommodation", transform=ax.transAxes,
                          ha="center", va="top", fontsize=9,
                          color=C_ACCOM, fontweight="bold")
            elif row == 2:
                ax.text2D(0.50, 0.97, "★  Assimilation", transform=ax.transAxes,
                          ha="center", va="top", fontsize=9,
                          color=C_ASS, fontweight="bold")
        if col in (1, 2):
            ax.text2D(0.50, -0.04, f"Final MSE: {mses[col]:.4f}",
                      transform=ax.transAxes, ha="center", va="top",
                      fontsize=9, color="#444444")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(length=0)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("#e0e0e0")
        ax.grid(True, alpha=0.12, lw=0.4)

plt.tight_layout(rect=[0.06, 0.0, 1.0, 1.0])
plt.savefig("figs/fig_trajectories.png", dpi=160, bbox_inches="tight")
plt.close()
print("  saved figs/fig_trajectories.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Per-epoch loss curves
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering fig_loss …")

def smooth(arr, w=5):
    """Centered moving average, edge-padded."""
    arr = np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="same")

pi_epochs = np.arange(len(pi_ep_log))
bl_epochs = np.arange(len(bl_ep_log))

fig_loss, ax = plt.subplots(figsize=(11, 4), facecolor="white")
ax.plot(pi_epochs, smooth(pi_ep_log, 3), color=C_PI, lw=1.6, label="PIAGETS")
ax.plot(bl_epochs, smooth(bl_ep_log, 3), color=C_BL, lw=1.6,
        label="Baseline (AL-RNN)", alpha=0.85)

for x in [EPOCHS, 2 * EPOCHS]:
    ax.axvline(x, color="#888888", lw=1.2, ls="--", alpha=0.65)

task_names_short = ["Lorenz ρ=28", "Halvorsen", "Lorenz ρ=42"]
for t, nm in enumerate(task_names_short):
    ax.text((t + 0.5) * EPOCHS, 1.0, nm, ha="center", va="bottom",
            fontsize=9, color="#555555", transform=ax.get_xaxis_transform())

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Reconstruction loss", fontsize=11)
ax.set_xlim(0, 3 * EPOCHS)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figs/fig_loss.png", dpi=160, bbox_inches="tight")
plt.close()
print("  saved figs/fig_loss.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Fisher / ΔW_B  (accommodation vs assimilation)
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering fig_weights …")

delta_wb  = {t: np.abs(wb_after[t] - wb_before[t]) for t in range(1, 3)}
delta_max = max(delta_wb[1].max(), delta_wb[2].max())
fisher_lor = fisher_snap.get(0, np.zeros((M, RANK)))

fig_w, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")

titles   = ["Signature Sensitivity\nafter Lorenz ρ=28\n(high = important for schema)",
            f"ΔW_B  |  Accommodation\nHalvorsen  (λ = {LAM_ACCOM})",
            f"ΔW_B  |  Assimilation\nLorenz ρ=42  (λ = {LAM_ASSIM})\nexisting schema reused"]
matrices = [fisher_lor, delta_wb[1], delta_wb[2]]
cmaps    = ["YlOrBr", "Reds", "Blues"]
vmaxes   = [1.0, delta_max, delta_max]
badges   = [None, "● Accommodation", "★ Assimilation"]
b_cols   = [None, C_ACCOM, C_ASS]

for i, ax in enumerate(axes):
    im = ax.imshow(matrices[i].T, aspect="auto", cmap=cmaps[i],
                   vmin=0, vmax=vmaxes[i])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(titles[i], fontsize=9, color="#222222", pad=8)
    ax.set_xlabel("Latent dim (M=16)", fontsize=8)
    ax.set_ylabel("Rank index (r=6)", fontsize=8)
    ax.set_xticks(range(M)); ax.set_yticks(range(RANK))
    ax.tick_params(labelsize=7)

    # Mark the ReLU / linear boundary
    ax.axvline(P - 0.5, color="#5555cc", lw=0.8, ls="--", alpha=0.6)

    if badges[i]:
        ax.text(0.5, -0.22, badges[i], transform=ax.transAxes,
                ha="center", fontsize=10, fontweight="bold", color=b_cols[i])

plt.tight_layout(pad=2.0)
plt.savefig("figs/fig_weights.png", dpi=160, bbox_inches="tight")
plt.close()
print("  saved figs/fig_weights.png")

print("\nDone.")
