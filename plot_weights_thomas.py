"""
plot_weights_thomas.py
======================
Generates figs/fig_weights_thomas.png using Thomas as the accommodation task.
Curriculum: Lorenz(rho=28) -> Thomas(b=0.19) -> Lorenz(rho=42)
Only trains PIAGETS (no baseline needed); no trajectory or loss figures.
"""
import sys, warnings
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import systems
from alrnn import ALRNN
import piagets as pg

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
np.random.seed(42)
torch.manual_seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
M, P, d, RANK = 16, 6, 3, 6
N_DATA   = 8_000
SEQ_LEN  = 100
BATCH    = 64
LR       = 1e-3
EPOCHS   = 400
N_WARMUP = 30
ANNEAL_FRAC = 1 / 3
LAM_EWC  = 5.0
LAM_ASSIM  = 4.0
LAM_ACCOM  = 2.0
SIG_FISHER_KW = dict(n_avg=3, T_warmup=100, T_track=200, beta=10.0)

C_ACCOM = "#CC2222"
C_ASS   = "#009E73"

# ── Data ───────────────────────────────────────────────────────────────────────
print("Generating data …")
train_seqs = [
    ("Lorenz ρ=28",   systems.lorenz(n=N_DATA, dt=0.05, rho=28)[0].astype("float32"), 0),
    ("Thomas b=0.19", systems.thomas(n=N_DATA, dt=0.05)[0].astype("float32"),          1),
    ("Lorenz ρ=42",   systems.lorenz(n=N_DATA, dt=0.05, rho=42)[0].astype("float32"), 0),
]
first_cls = train_seqs[0][2]

# ── Model & CL ────────────────────────────────────────────────────────────────
def fresh_model(seed=0):
    torch.manual_seed(seed)
    return ALRNN(latent_dim=M, obs_dim=d, P=P, rank=RANK)

def _train(model, data, epochs, cl=None, n_warmup=0, anneal_frac=0.0,
           reg_lambda=0.0, tag=""):
    pg.train_with_ewc(
        model, data, cl=cl, epochs=epochs,
        seq_len=SEQ_LEN, batch=BATCH, lr=LR,
        alpha=0.5, alpha_end=0.10,
        reg_lambda=reg_lambda, device="cpu",
        n_warmup=n_warmup, anneal_frac=anneal_frac,
        log=lambda s: print(f"  [{tag}] {s}", flush=True),
    )

pmodel = fresh_model(0)
cl = pg.PIAGETSContinual(
    lam_ewc=LAM_EWC, lam_assim=LAM_ASSIM, lam_accom=LAM_ACCOM,
    sig_fisher_kwargs=SIG_FISHER_KW,
)

wb_before    = {}
wb_after     = {}
fisher_snap  = {}

for t, (name, data, true_cls) in enumerate(train_seqs):
    print(f"\n── Task {t}: {name} (class={true_cls}) ──")

    if t > 0:
        if true_cls == first_cls:
            lam = LAM_ASSIM; print(f"  ASSIMILATION  λ={lam}")
        else:
            lam = LAM_ACCOM; print(f"  ACCOMMODATION λ={lam}")
        cl.lam_ewc = lam
        cl.qr_init(pmodel)

    wb_before[t] = pmodel.W_B.data.clone().numpy()

    n_wu  = N_WARMUP    if t > 0 else 0
    afrac = ANNEAL_FRAC if t > 0 else 0.0
    # Thomas (cls=1) is sensitive to entropy reg → use reg_lambda=0 for it
    reg = 0.0 if true_cls == 1 else 0.02

    _train(pmodel, data, EPOCHS, cl=cl,
           n_warmup=n_wu, anneal_frac=afrac,
           reg_lambda=reg, tag=name)

    wb_after[t] = pmodel.W_B.data.clone().numpy()

    cl.store_task(pmodel, data, true_class=true_cls, use_sig_fisher=True)
    if cl._F_B is not None:
        fisher_snap[t] = cl._F_B.numpy().copy()
    cl.restore_merged(pmodel)

# ── Weights figure ─────────────────────────────────────────────────────────────
print("\nRendering fig_weights_thomas …")

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
print("Saved figs/fig_weights_thomas.png")
