"""
plot_thomas_curriculum.py
=========================
Curriculum: Lorenz(rho=28) → Thomas(b=0.19) → Lorenz(rho=42)

Produces:
  figs/fig_trajectories_thomas.png
  figs/fig_weights_thomas.png

Training: two-pass per PIAGETS task (Pass 1 = 400ep for loss curve,
Pass 2 = best-ep with proper alpha annealing for snapshot).
reg_lambda: 0.02 for Lorenz tasks, 0.005 for Thomas (avoids negative loss
from high-entropy chaotic signature while still breaking 1-region collapse).
"""
import sys, copy, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
SEQ_LEN         = 200
BATCH           = 64
LR              = 1e-3
EPOCHS          = 600
N_WARMUP        = 30
ANNEAL_FRAC     = 1 / 3
LAM_EWC         = 5.0
LAM_ASSIM       = 4.0
LAM_ACCOM       = 2.0
SIG_FISHER_KW   = dict(n_avg=3, T_warmup=100, T_track=200, beta=10.0)
TRAJ_LEN        = 400
ALPHA_VIZ       = 0.4

REG_LORENZ  = 0.02   # entropy reg for Lorenz → multiple ReLU regions
REG_THOMAS  = 0.005  # smaller for Thomas → avoids negative total loss

# ── Colors ─────────────────────────────────────────────────────────────────────
C_GT    = "#E69F00"
C_PI    = "#56B4E9"
C_BL    = "#CC79A7"
C_ASS   = "#009E73"
C_ACCOM = "#CC2222"

# ── Task stream ────────────────────────────────────────────────────────────────
print("Generating data …")
train_seqs = [
    ("Lorenz ρ=28",   systems.lorenz(n=N_DATA, dt=0.05, rho=28)[0].astype("float32"), 0),
    ("Thomas b=0.19", systems.thomas(n=N_DATA, dt=0.05)[0].astype("float32"),          1),
    ("Lorenz ρ=42",   systems.lorenz(n=N_DATA, dt=0.05, rho=42)[0].astype("float32"), 0),
]
first_cls = train_seqs[0][2]

REG_BY_CLS = {0: REG_LORENZ, 1: REG_THOMAS}

print("Generating ground-truth phase portraits …")
gt_trajs = [
    systems.lorenz(n=15_000, dt=0.05, rho=28)[0].astype("float32"),
    systems.thomas(n=15_000, dt=0.05)[0].astype("float32"),
    systems.lorenz(n=15_000, dt=0.05, rho=42)[0].astype("float32"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def fresh_model(seed=0):
    torch.manual_seed(seed)
    return ALRNN(latent_dim=M, obs_dim=d, P=P, rank=RANK)


def _train(model, data, epochs, cl=None, n_warmup=0, anneal_frac=0.0,
           alpha_end=0.10, epoch_log=None, silent=False, tag="", reg_lambda=0.0):
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


def gt_strand(gt_data, traj_len=TRAJ_LEN):
    start = np.random.randint(0, len(gt_data) - traj_len)
    return [gt_data[start:start + traj_len]]


@torch.no_grad()
def model_strand(model, gt_data, traj_len=TRAJ_LEN):
    start = np.random.randint(0, len(gt_data) - traj_len - 1)
    x = torch.as_tensor(gt_data[start:start + traj_len + 1],
                        dtype=torch.float32).unsqueeze(0)
    return [model.forced_rollout(x, alpha=ALPHA_VIZ).squeeze(0).numpy()]


@torch.no_grad()
def eval_mse(model, data, n_batches=50):
    x = torch.as_tensor(data, dtype=torch.float32)
    losses = []
    for _ in range(n_batches):
        s = np.random.randint(0, len(x) - SEQ_LEN - 1)
        xb = x[s:s + SEQ_LEN + 1].unsqueeze(0)
        pred = model.forced_rollout(xb, alpha=ALPHA_VIZ)
        losses.append(((pred - xb[:, 1:]) ** 2).mean().item())
    return float(np.mean(losses))


# ── Baseline ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Baseline — sequential, no protection, best-epoch per task")
print("=" * 60)

baseline     = fresh_model(0)
bl_ep_log    = []
bl_snapshots = []

for t, (name, data, true_cls) in enumerate(train_seqs):
    print(f"\n  ── Task {t}: {name} ──")
    pre_bl_sd = copy.deepcopy(baseline.state_dict())

    task_log = []
    _train(baseline, data, EPOCHS, epoch_log=task_log,
           reg_lambda=REG_BY_CLS[true_cls], tag=f"base_P1/{name}")
    bl_ep_log.extend(task_log)

    best_ep = max(int(np.argmin(task_log)) + 1, 5)
    print(f"  → Best ep {best_ep}/{EPOCHS}  recon={task_log[best_ep-1]:.5f}")

    baseline.load_state_dict(pre_bl_sd)
    _train(baseline, data, best_ep,
           reg_lambda=REG_BY_CLS[true_cls], silent=True, tag=f"base_P2/{name}")
    bl_snapshots.append(copy.deepcopy(baseline))

# ── PIAGETS — two-pass ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIAGETS — two-pass per task")
print("=" * 60)

pmodel = fresh_model(0)
cl = pg.PIAGETSContinual(
    lam_ewc=LAM_EWC, lam_assim=LAM_ASSIM, lam_accom=LAM_ACCOM,
    sig_fisher_kwargs=SIG_FISHER_KW,
)

pi_ep_log    = []
pi_snapshots = []
probe_modes  = []
wb_before    = {}
wb_after     = {}
fisher_snap  = {}

for t, (name, data, true_cls) in enumerate(train_seqs):
    print(f"\n  ── Task {t}: {name} (class={true_cls}) ──")
    reg = REG_BY_CLS[true_cls]

    if t > 0:
        if true_cls == first_cls:
            mode, lam = "assimilation", LAM_ASSIM
            print(f"  → ASSIMILATION  λ={lam}")
        else:
            mode, lam = "accommodation", LAM_ACCOM
            print(f"  → ACCOMMODATION λ={lam}")
    else:
        mode = "accommodation"
    probe_modes.append(mode)

    # Save pre-task state for two-pass
    pre_model_sd = copy.deepcopy(pmodel.state_dict())
    pre_cl       = copy.deepcopy(cl)

    n_wu  = N_WARMUP    if t > 0 else 0
    afrac = ANNEAL_FRAC if t > 0 else 0.0

    if t > 0:
        cl.lam_ewc = lam
        cl.qr_init(pmodel)

    wb_before[t] = pmodel.W_B.data.clone().numpy()

    # ── Pass 1: full training for loss curve ───────────────────────────────────
    task_loss = []
    _train(pmodel, data, EPOCHS, cl=cl,
           n_warmup=n_wu, anneal_frac=afrac, alpha_end=0.10,
           epoch_log=task_loss, reg_lambda=reg, tag=f"P1/{name}")
    pi_ep_log.extend(task_loss)

    # Find best epoch (skip warmup)
    search = task_loss[n_wu:]
    best_ep = (n_wu + int(np.argmin(search)) + 1) if search else EPOCHS
    best_ep = max(best_ep, n_wu + 5)
    print(f"  → Best ep {best_ep}/{EPOCHS}  "
          f"recon={task_loss[best_ep-1]:.5f}  final={task_loss[-1]:.5f}")

    # ── Pass 2: re-train to best_ep with same alpha schedule ──────────────────
    pmodel.load_state_dict(pre_model_sd)
    cl = copy.deepcopy(pre_cl)
    if t > 0:
        cl.lam_ewc = lam
        cl.qr_init(pmodel)

    _train(pmodel, data, best_ep, cl=cl,
           n_warmup=n_wu, anneal_frac=ANNEAL_FRAC, alpha_end=0.10,
           reg_lambda=reg, silent=True, tag=f"P2/{name}")

    wb_after[t] = pmodel.W_B.data.clone().numpy()
    pi_snapshots.append(copy.deepcopy(pmodel))

    cl.store_task(pmodel, data, true_class=true_cls, use_sig_fisher=True)
    if cl._F_B is not None:
        fisher_snap[t] = cl._F_B.numpy().copy()
    cl.restore_merged(pmodel)

print(f"\n  Modes: {probe_modes}")

# ── MSE ────────────────────────────────────────────────────────────────────────
print("\nComputing MSEs …")
all_data = [d for _, d, _ in train_seqs]
pi_mses  = [eval_mse(pi_snapshots[t], all_data[t]) for t in range(3)]
bl_mses  = [eval_mse(bl_snapshots[t], all_data[t]) for t in range(3)]
print("  PIAGETS:  " + "  ".join(f"{m:.4f}" for m in pi_mses))
print("  Baseline: " + "  ".join(f"{m:.4f}" for m in bl_mses))

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Trajectory grid
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering fig_trajectories_thomas …")

VIEW = [
    (18, -65),   # Lorenz ρ=28: classic butterfly
    (25,  40),   # Thomas: shows the tangled 3-fold spiral
    (25, -45),   # Lorenz ρ=42: slightly rotated
]
row_labels = ["First system:\nLorenz ρ=28",
              "Second system:\nThomas b=0.19",
              "Third system:\nLorenz ρ=42"]
col_labels = ["Ground Truth", "PIAGETS", "Baseline (AL-RNN)"]
col_colors = [C_GT, C_PI, C_BL]

fig = plt.figure(figsize=(15, 14), facecolor="white")
for row in range(3):
    mses = [None, pi_mses[row], bl_mses[row]]
    traj_len = TRAJ_LEN * 10 if row == 1 else TRAJ_LEN
    strand_sets = [
        gt_strand(gt_trajs[row], traj_len=traj_len),
        model_strand(pi_snapshots[row], gt_trajs[row], traj_len=traj_len),
        model_strand(bl_snapshots[row], gt_trajs[row], traj_len=traj_len),
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
                      fontsize=13, fontweight="bold", color="#2d2d2d",
                      va="center", ha="center", rotation=90)
        if row == 0:
            ax.set_title(col_labels[col], fontsize=15, fontweight="bold",
                         color="#2d2d2d", pad=6)
        if col == 1:
            if row == 1:
                ax.text2D(0.50, 0.97, "●  Accommodation", transform=ax.transAxes,
                          ha="center", va="top", fontsize=13,
                          color=C_ACCOM, fontweight="bold")
            elif row == 2:
                ax.text2D(0.50, 0.97, "★  Assimilation", transform=ax.transAxes,
                          ha="center", va="top", fontsize=13,
                          color=C_ASS, fontweight="bold")
        if col in (1, 2):
            ax.text2D(0.50, -0.04, f"Final MSE: {mses[col]:.4f}",
                      transform=ax.transAxes, ha="center", va="top",
                      fontsize=13, color="#444444")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(length=0)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("#e0e0e0")
        ax.grid(True, alpha=0.12, lw=0.4)

plt.tight_layout(rect=[0.06, 0.0, 1.0, 1.0])
plt.savefig("figs/fig_trajectories_thomas.png", dpi=160, bbox_inches="tight")
plt.close()
print("  saved figs/fig_trajectories_thomas.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Weights / Fisher
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering fig_weights_thomas …")

delta_wb  = {t: np.abs(wb_after[t] - wb_before[t]) for t in range(1, 3)}
delta_max = max(delta_wb[1].max(), delta_wb[2].max())
fisher_lor = fisher_snap.get(0, np.zeros((M, RANK)))

fig_w, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")

titles   = ["Signature Sensitivity\nafter Lorenz ρ=28\n(high = important for schema)",
            f"ΔW_B  |  Accommodation\nThomas b=0.19  (λ = {LAM_ACCOM})",
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
    ax.axvline(P - 0.5, color="#5555cc", lw=0.8, ls="--", alpha=0.6)
    if badges[i]:
        ax.text(0.5, -0.22, badges[i], transform=ax.transAxes,
                ha="center", fontsize=10, fontweight="bold", color=b_cols[i])

plt.tight_layout(pad=2.0)
plt.savefig("figs/fig_weights_thomas.png", dpi=160, bbox_inches="tight")
plt.close()
print("  saved figs/fig_weights_thomas.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-epoch loss curves
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering fig_loss_thomas …")

def smooth(arr, w=5):
    arr = np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="same")

fig_loss, ax = plt.subplots(figsize=(11, 4), facecolor="white")
ax.plot(np.arange(len(pi_ep_log)), smooth(pi_ep_log, 3),
        color=C_PI, lw=1.6, label="PIAGETS")
ax.plot(np.arange(len(bl_ep_log)), smooth(bl_ep_log, 3),
        color=C_BL, lw=1.6, label="Baseline (AL-RNN)", alpha=0.85)

for x in [EPOCHS, 2 * EPOCHS]:
    ax.axvline(x, color="#888888", lw=1.2, ls="--", alpha=0.65)

task_names_short = ["Lorenz ρ=28", "Thomas b=0.19", "Lorenz ρ=42"]
for t, nm in enumerate(task_names_short):
    ax.text((t + 0.5) * EPOCHS, 1.0, nm, ha="center", va="bottom",
            fontsize=9, color="#555555", transform=ax.get_xaxis_transform())

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Reconstruction loss", fontsize=11)
ax.set_xlim(0, 3 * EPOCHS)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figs/fig_loss_thomas.png", dpi=160, bbox_inches="tight")
plt.close()
print("  saved figs/fig_loss_thomas.png")

print("\nDone.")
