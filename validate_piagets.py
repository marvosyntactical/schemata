"""PIAGETS long-sequence validation — MSE / dφ / BWT / schema recognition / convergence.

Task stream (T=10, three maximally topologically distinct schema classes)
------------------------------------------------------------------------
  Class 0 — Lorenz      (chaotic strange attractor)   × 4
  Class 1 — torus       (quasiperiodic, 2D)           × 3
  Class 2 — van_der_pol (relaxation limit cycle)      × 3

Methods
-------
  baseline         — naive sequential fine-tuning, no protection
  piagets          — sig-Fisher EWC + SLAO QR-init (global EMA, fixed λ)
  piagets_adaptive — PIAGETS + φ-probe adaptive λ
  piagets_inv      — PIAGETS with inverted Fisher weights (high-F params released)

Metrics
-------
  mse_BWT     — backward transfer in MSE space (positive = forgetting, lower is better)
  dphi_BWT    — backward transfer in φ-distance (positive = forgetting, lower is better)
  MSE_diag    — mean reconstruction MSE right after training each task
  dphi_diag   — mean φ-distance to oracle right after training each task
  rec_ora     — nearest-task-oracle schema recognition accuracy
  walltime    — total seconds per method
  conv_half   — median epochs to reach 50% of MSE drop, per task
"""
import sys, copy, time, warnings
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

EP_ORACLE     = 300
EP_TASK0      = 300
EP_FINE_LOR   = 300
EP_FINE_TORUS = 350
EP_FINE_VDP   = 200

LAM_EWC      = 5.0
LAM_ASSIM    = 10.0
LAM_ACCOM    = 2.0
SCHEMA_SIGMA = 2.0
SIG_FISHER_KW = dict(n_avg=3, T_warmup=100, T_track=200, beta=10.0)
PHI_REG_KW    = dict(T_warmup=100, T_track=100, beta=10.0)
LAM_PHI       = 0.15
R_LIN         = 2
ASSIM_MSE_RATIO = 10.0
ORTHO_K         = 2   # new dirs per task added to joint basis; budget capped at r//2
N_PROBE_EPOCHS  = 100  # probe epochs; 100 is enough for Lorenz gates to freeze (φ_act_std→0)
N_WARMUP        = 30   # warmup epochs: W_B frozen, λ=0 (let W_A recover from QR-reinit)
ANNEAL_FRAC     = 1/3  # fraction of post-warmup epochs where λ decays linearly → 0

PHI_T_WARMUP, PHI_T_TRACK, PHI_BETA, PHI_N_AVG = 100, 200, 10.0, 3
DEVICE = "cpu"

# ─── Task stream ─────────────────────────────────────────────────────────────
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
CLASSES   = [c for _, _, c, _ in STREAM]
N_SCHEMAS = len(set(CLASSES))

CLASS_NAMES = {0: "Lor", 1: "Tor", 2: "VdP"}

# ─── Methods ─────────────────────────────────────────────────────────────────
# Keys used in cfg:
#   use_cl        — whether to create a PIAGETSContinual object
#   use_sig       — use signature-Fisher (else prediction-Fisher)
#   adaptive      — use Option-B φ-probe (brief λ=0 fine-tune → φ → centroid comparison)
#   use_merge     — call restore_merged after each task (global EMA)
#   class_ewc     — per-class EWC anchors
#   class_b       — per-class (W_B, W_A) module + restore at inference
#   ortho         — orthogonal gradient masking during accommodation
#   no_slao       — EWC on all params incl. W_A; no QR-init; no restore_merged
METHODS = {
    "baseline":              dict(use_cl=False, use_sig=False, adaptive=False,
                                  use_merge=False, class_ewc=False, class_b=False,
                                  ortho=False, no_slao=False, invert_fisher=False,
                                  warmup=False),
    "piagets":               dict(use_cl=True,  use_sig=True,  adaptive=False,
                                  use_merge=True,  class_ewc=False, class_b=False,
                                  ortho=False, no_slao=False, invert_fisher=False,
                                  warmup=True),
    "piagets_adaptive":      dict(use_cl=True,  use_sig=True,  adaptive=True,
                                  use_merge=True,  class_ewc=False, class_b=False,
                                  ortho=False, no_slao=False, invert_fisher=False,
                                  warmup=True),
    "piagets_inv":           dict(use_cl=True,  use_sig=True,  adaptive=False,
                                  use_merge=True,  class_ewc=False, class_b=False,
                                  ortho=False, no_slao=False, invert_fisher=True,
                                  warmup=True),
    "piagets_ada_nolora":    dict(use_cl=True,  use_sig=True,  adaptive=True,
                                  use_merge=False, class_ewc=False, class_b=False,
                                  ortho=False, no_slao=True,  invert_fisher=False,
                                  warmup=False),
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

RANK = P   # rank=6: keeps Lorenz 1-region so φ geometry is class-discriminative

def fresh_model(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return ALRNN(latent_dim=M, obs_dim=d, P=P, rank=RANK)


def phi_of(model):
    phis = []
    with torch.no_grad():
        for _ in range(PHI_N_AVG):
            phis.append(pg.diff_phi(model, PHI_T_WARMUP, PHI_T_TRACK, PHI_BETA).numpy())
    return np.mean(phis, axis=0)


def phi_dist(pa, pb, scale):
    return float(np.linalg.norm((pa - pb) / (scale + 1e-9)))


def train(model, data, epochs, cl=None, label="", alpha=0.5, alpha_end=0.05,
          regime=None, current_class=None, accommodation=False, epoch_log=None,
          n_warmup=0, anneal_frac=0.0):
    log_lines = []
    pg.train_with_ewc(model, data, cl=cl, epochs=epochs, seq_len=SEQ_LEN,
                      batch=BATCH, lr=LR, alpha=alpha, alpha_end=alpha_end,
                      reg_lambda=REG_LAM, device=DEVICE,
                      regime=regime, current_class=current_class,
                      accommodation=accommodation, epoch_log=epoch_log,
                      n_warmup=n_warmup, anneal_frac=anneal_frac,
                      log=lambda s: (log_lines.append(s),
                                     print(f"  [{label}] {s}", flush=True)))
    return log_lines


@torch.no_grad()
def eval_recon_mse(model, data, n_batches=30, alpha=0.05):
    N = len(data)
    losses = []
    for _ in range(n_batches):
        start = np.random.randint(0, N - SEQ_LEN - 1)
        x = torch.tensor(data[start:start + SEQ_LEN + 1]).unsqueeze(0)
        preds = model.forced_rollout(x, alpha=alpha)
        losses.append(((preds - x[:, 1:]) ** 2).mean().item())
    return float(np.mean(losses))


def conv_half_epochs(epoch_mse: list) -> int:
    """Epochs until MSE crosses the midpoint between initial and final value.
    Returns len(epoch_mse) if the midpoint is never reached (no convergence)."""
    if len(epoch_mse) < 2:
        return len(epoch_mse)
    m0, mf = epoch_mse[0], epoch_mse[-1]
    mid = (m0 + mf) / 2.0
    if mf >= m0:        # MSE went up — no convergence
        return len(epoch_mse)
    for ep, m in enumerate(epoch_mse):
        if m <= mid:
            return ep + 1
    return len(epoch_mse)


# ─── Step 1: Oracle models ────────────────────────────────────────────────────
print("=" * 64)
print("Step 1: Oracle models")
print("=" * 64)

oracle_phis = []
oracle_sigs = []
task_data   = []
seed_obs    = np.zeros(d, dtype="float32")

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

phi_scale = np.std(np.stack(oracle_phis), axis=0)
phi_scale[phi_scale < 1e-9] = 1.0
print(f"\n  φ scale: min={phi_scale.min():.3f}  max={phi_scale.max():.3f}")

# Per-class oracle centroids
class_phis = {}
for i, cls in enumerate(CLASSES):
    class_phis.setdefault(cls, []).append(oracle_phis[i])
centroid = {c: np.mean(ps, axis=0) for c, ps in class_phis.items()}

# Random model baseline for FWT
rand_phis = [phi_of(fresh_model(seed=s)) for s in range(5)]
rand_phi  = np.mean(rand_phis, axis=0)

print("\n  Oracle pairwise φ-distance matrix:")
print(f"  {'':10s}" + "".join(f"  {n:8s}" for n in NAMES))
for i in range(T):
    row = "".join(f"  {phi_dist(oracle_phis[i], oracle_phis[j], phi_scale):8.3f}"
                  for j in range(T))
    print(f"  {NAMES[i]:10s}{row}")


# ─── Step 2: CL experiment ────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("Step 2: Sequential CL")
print("=" * 64)

# Tracking arrays
dphi_mat     = {m: np.full((T, T), np.nan) for m in METHODS}  # φ-distance (lower=better)
MSE_mat      = {m: np.full((T, T), np.nan) for m in METHODS}  # reconstruction MSE
mse_diag     = {m: np.zeros(T) for m in METHODS}              # MSE right after training
mse_start    = {m: np.zeros(T) for m in METHODS}              # MSE before training
rec_ora      = {m: np.zeros(T) for m in METHODS}
rec_cen      = {m: np.zeros(T) for m in METHODS}
pred_cls     = {m: np.full(T, -1, dtype=int) for m in METHODS}  # predicted class (nearest centroid)
walltime     = {m: 0.0 for m in METHODS}
conv_records = {m: [] for m in METHODS}   # [(mode_str, conv_half_ep), ...]
per_cls_conv = {m: {c: [] for c in set(CLASSES)} for m in METHODS}  # class → [conv_half]
cl_objects   = {}  # method → PIAGETSContinual (for post-run φ analysis)

for method, cfg in METHODS.items():
    print(f"\n{'─'*60}")
    print(f"  Method: {method.upper()}")
    print(f"{'─'*60}")

    t_start = time.time()
    m = fresh_model(seed=0)
    cl = (pg.PIAGETSContinual(
              lam_ewc=LAM_EWC,
              lam_assim=LAM_ASSIM, lam_accom=LAM_ACCOM,
              schema_sigma=SCHEMA_SIGMA,
              assim_mse_ratio=ASSIM_MSE_RATIO,
              sig_fisher_kwargs=SIG_FISHER_KW,
              lam_phi=0.0,
              r_lin=None,
              phi_reg_kwargs=PHI_REG_KW,
              use_class_ewc=cfg.get("class_ewc", False),
              use_class_b=cfg.get("class_b", False),
              use_ortho=cfg.get("ortho", False),
              ortho_k=ORTHO_K,
              no_slao=cfg.get("no_slao", False),
              invert_fisher=cfg.get("invert_fisher", False))
          if cfg["use_cl"] else None)
    cl_objects[method] = cl

    for t, (name, gen, true_cls, ep_fine) in enumerate(STREAM):
        data = task_data[t]
        ep   = EP_TASK0 if t == 0 else ep_fine
        mode = "accommodation"   # default — updated by probe below

        print(f"\n  t={t} {name} (class={true_cls})")

        if cl is not None and t > 0:
            if cfg["adaptive"]:
                # Option B: brief λ=0 fine-tune on a model copy → compute φ
                # → compare to stored class-centroid φs.  Works at any LoRA rank.
                lam, mode, _ = cl.probe_from_phi_finetune(
                    m, data,
                    n_probe_epochs=N_PROBE_EPOCHS,
                    seq_len=SEQ_LEN, batch=BATCH,
                    probe_lr=LR, device=DEVICE)
                cl.lam_ewc = lam
                cl.current_mode = mode

            if not cfg.get("no_slao"):
                # class_b assimilation: restore class module first so qr_init
                # orthogonalizes within the class subspace, not the global one.
                if cfg.get("class_b") and mode == "assimilation" and true_cls in cl._class_modules:
                    cl.restore_merged_class(m, true_cls)
                    cl._A_prev = m.W_A.data.clone()
                cl.qr_init(m)

        # MSE before training — measures warm-start quality for assimilation
        mse_start[method][t] = eval_recon_mse(m, task_data[t])
        print(f"    mse_start={mse_start[method][t]:.5f}")

        n_wu = N_WARMUP if (t > 0 and cfg.get("warmup")) else 0
        epoch_log = []
        train(m, data, ep,
              cl=(cl if cfg["use_cl"] else None),
              label=f"{method}/{name}",
              current_class=true_cls if cfg.get("class_ewc") else None,
              accommodation=(mode == "accommodation"),
              epoch_log=epoch_log,
              n_warmup=n_wu,
              anneal_frac=ANNEAL_FRAC if cfg["use_cl"] else 0.0)

        # Convergence speed: epochs to reach midpoint of MSE drop
        conv_h = conv_half_epochs(epoch_log)
        conv_records[method].append((mode, conv_h))
        per_cls_conv[method][true_cls].append(conv_h)
        mode_tag = 'S' if mode == 'assimilation' else 'A'
        print(f"    conv_half={conv_h}ep  mode={mode_tag}")

        # MSE on current task before restore_merged (measures fine-tuned quality)
        mse_diag[method][t] = eval_recon_mse(m, task_data[t])
        print(f"    mse_recon={mse_diag[method][t]:.5f}")

        if cl is not None:
            cl.store_task(m, data, true_class=true_cls,
                          use_sig_fisher=cfg["use_sig"], device=DEVICE)
            if cfg["use_merge"]:
                if cfg.get("class_b"):
                    # Per-class methods: restore class-specific W_B for current task
                    cl.restore_merged_class(m, true_cls)
                else:
                    cl.restore_merged(m)

        # ── Evaluate: fill R[t, :] and MSE_mat[t, :] ──────────────────────
        # For per-class methods, we need to swap in the class-specific W_B for
        # each task i so that R[t,i] reflects the schema module for class i.
        # We use a temporary model copy to avoid modifying the training model.
        if cfg.get("class_b") and cl is not None:
            m_eval = copy.deepcopy(m)
            for i in range(t + 1):
                cl.restore_merged_class(m_eval, CLASSES[i])
                phi_i = phi_of(m_eval)
                dphi_mat[method][t, i] = phi_dist(phi_i, oracle_phis[i], phi_scale)
                MSE_mat[method][t, i] = eval_recon_mse(m_eval, task_data[i])
            # Recognition uses the current-task W_B (already restored above)
            cl.restore_merged_class(m_eval, true_cls)
            phi_now = phi_of(m_eval)
            del m_eval
        else:
            phi_now = phi_of(m)
            for i in range(t + 1):
                dphi_mat[method][t, i] = phi_dist(phi_now, oracle_phis[i], phi_scale)
                MSE_mat[method][t, i] = eval_recon_mse(m, task_data[i])

        # Schema recognition
        dists_ora = {i: phi_dist(phi_now, oracle_phis[i], phi_scale) for i in range(t + 1)}
        best_ora  = min(dists_ora, key=dists_ora.get)
        rec_ora[method][t] = int(CLASSES[best_ora] == true_cls)

        dists_cen  = {c: phi_dist(phi_now, centroid[c], phi_scale) for c in centroid}
        best_cen   = min(dists_cen, key=dists_cen.get)
        rec_cen[method][t]  = int(best_cen == true_cls)
        pred_cls[method][t] = best_cen

        print(f"    rec_ora={int(CLASSES[best_ora]==true_cls)}  "
              f"rec_cen={int(best_cen==true_cls)}  "
              f"nearest={NAMES[best_ora]}")

    walltime[method] = time.time() - t_start
    print(f"  [walltime] {method}: {walltime[method]:.0f}s")


# ─── Step 3: Compute and report metrics ──────────────────────────────────────
print("\n" + "=" * 64)
print("Step 3: Metrics")
print("=" * 64)


def dphi_bwt(D):
    """Positive = forgetting (φ-distance to oracle grew). Lower is better."""
    return float(np.mean([D[T-1, i] - D[i, i] for i in range(T-1)]))


def mse_bwt(M):
    """Positive = forgetting (MSE on old tasks got worse). Lower is better."""
    return float(np.mean([M[T-1, i] - M[i, i] for i in range(T-1)]))


def adjusted_rand_index(labels_true, labels_pred):
    """Compute ARI without sklearn dependency."""
    from collections import Counter
    n = len(labels_true)
    assert n == len(labels_pred)
    # contingency table
    pairs_true = Counter(zip(labels_true, labels_pred))
    # sum of C(n_ij, 2)
    sum_comb   = sum(v * (v - 1) // 2 for v in pairs_true.values())
    # row/col sums
    row = Counter(labels_true);  col = Counter(labels_pred)
    sum_row = sum(v * (v - 1) // 2 for v in row.values())
    sum_col = sum(v * (v - 1) // 2 for v in col.values())
    n_pairs = n * (n - 1) // 2
    expected = sum_row * sum_col / n_pairs if n_pairs else 0
    max_term = (sum_row + sum_col) / 2
    denom    = max_term - expected
    return (sum_comb - expected) / denom if denom > 1e-12 else 1.0

# Convergence speed summary
def conv_summary(records):
    """Return (overall_median, assimilation_median, accommodation_median)."""
    all_h = [h for _, h in records]
    ass_h = [h for mode, h in records if mode == "assimilation"]
    acc_h = [h for mode, h in records if mode == "accommodation"]
    med  = float(np.median(all_h)) if all_h else float("nan")
    a_med = float(np.median(ass_h)) if ass_h else float("nan")
    c_med = float(np.median(acc_h)) if acc_h else float("nan")
    return med, a_med, c_med


def mse_start_summary(records, mse_s):
    """Mean mse_start for assimilation tasks vs accommodation tasks."""
    ass = [mse_s[i] for i, (mode, _) in enumerate(records) if mode == "assimilation"]
    acc = [mse_s[i] for i, (mode, _) in enumerate(records) if mode == "accommodation"]
    return (float(np.mean(ass)) if ass else float("nan"),
            float(np.mean(acc)) if acc else float("nan"))


header = (f"  {'method':22s}  {'mse_BWT':>8}  {'dphi_BWT':>9}  "
          f"{'mse':>9}  {'dphi':>8}  {'ARI':>5}  "
          f"{'cLor':>5}  {'cTor':>5}  {'cVdP':>5}  "
          f"{'c_ass':>6}  {'c_acc':>6}  {'wtime':>7}")
print(header)
print("  " + "-" * (len(header) - 2))

result_lines = []
for method in METHODS:
    D       = dphi_mat[method]
    db      = dphi_bwt(D)
    mb      = mse_bwt(MSE_mat[method])
    ro      = rec_ora[method].mean()
    rc      = rec_cen[method].mean()
    dphi_d  = float(np.mean([D[i, i] for i in range(T)]))
    mse_avg = mse_diag[method].mean()
    c_all, c_ass, c_acc = conv_summary(conv_records[method])
    ms_ass, ms_acc = mse_start_summary(conv_records[method], mse_start[method])
    wt      = walltime[method]
    ari     = adjusted_rand_index(CLASSES, pred_cls[method].tolist())
    # Per-class mean convergence
    c_lor = float(np.mean(per_cls_conv[method][0])) if per_cls_conv[method][0] else float("nan")
    c_tor = float(np.mean(per_cls_conv[method][1])) if per_cls_conv[method][1] else float("nan")
    c_vdp = float(np.mean(per_cls_conv[method][2])) if per_cls_conv[method][2] else float("nan")
    print(f"  {method:22s}  {mb:+8.4f}  {db:+9.3f}  "
          f"{mse_avg:9.5f}  {dphi_d:8.3f}  {ari:5.3f}  "
          f"{c_lor:5.1f}  {c_tor:5.1f}  {c_vdp:5.1f}  "
          f"{c_ass:6.1f}  {c_acc:6.1f}  {wt:6.0f}s")
    result_lines.append((method, mb, db, ro, rc, dphi_d, mse_avg, c_all, c_ass, c_acc,
                         ms_ass, ms_acc, wt, D, ari, c_lor, c_tor, c_vdp))

# Full dφ matrices
print()
for method, mb, db, ro, rc, dphi_d, mse_avg, c_all, c_ass, c_acc, ms_ass, ms_acc, wt, D, ari, *_ in result_lines:
    print(f"\n  dφ matrix — {method}  (mse_BWT={mb:+.4f}  dphi_BWT={db:+.3f}  ARI={ari:.3f}  MSE={mse_avg:.5f}):")
    hdr = "  {:6s}".format("t\\i") + "".join(f"  {n:8s}" for n in NAMES)
    print(hdr)
    for t in range(T):
        row = "".join(f"  {D[t,i]:8.3f}" if i <= t else "          "
                      for i in range(T))
        print(f"  t={t}  {NAMES[t]:8s}{row}")

# Convergence breakdown per method
print("\n  Convergence detail (S=assimilation A=accommodation, epochs-to-half-MSE):")
for method in METHODS:
    recs = conv_records[method]
    detail = "  ".join(f"t{i}:{'S' if mode=='assimilation' else 'A'}{h:3d}"
                       for i, (mode, h) in enumerate(recs))
    print(f"  {method:22s}  {detail}")


# ─── Step 4: Figures ─────────────────────────────────────────────────────────
print("\nGenerating figures …")

COLS = {
    "baseline":           "#999999",
    "piagets":            "#4e79a7",
    "piagets_adaptive":   "#59a14f",
    "piagets_inv":        "#f28e2b",
    "piagets_ada_nolora": "#e15759",
}

# Figure 1: dφ-matrix heatmaps (lower=better, green=close to oracle)
fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 8), squeeze=False)
axes = axes[0]
fig.suptitle("dφ[t,i] = φ_dist(model_t, oracle_i)  (lower/greener = better)", fontsize=11)
for ax, method in zip(axes, METHODS):
    D = dphi_mat[method]
    masked = np.ma.masked_where(np.isnan(D), D)
    im = ax.imshow(masked, cmap="RdYlGn_r", vmin=0.0, vmax=20.0, aspect="auto")
    ax.set_title(f"{method}\nmse_BWT={mse_bwt(MSE_mat[method]):+.4f}  rec_ora={rec_ora[method].mean():.2f}",
                 fontsize=8)
    ax.set_xticks(range(T))
    ax.set_xticklabels(NAMES, rotation=40, ha="right", fontsize=6)
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"t={t} {NAMES[t]}" for t in range(T)], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    for t in range(T):
        ax.add_patch(plt.Rectangle((t - 0.5, t - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=1.5))
plt.tight_layout()
plt.savefig("fig_piagets_bwt.png", dpi=120)
print("  saved fig_piagets_bwt.png")
plt.close()

# Figure 2: 5-panel summary
fig, axes2 = plt.subplots(1, 5, figsize=(26, 5))
xs = np.arange(len(METHODS))
labels = list(METHODS.keys())
col_list = [COLS[m] for m in METHODS]

ax_mb, ax_db, ax_m, ax_r, ax_c = axes2

# MSE BWT (primary: lower/more negative = better)
mse_bwts = [mse_bwt(MSE_mat[m]) for m in METHODS]
ax_mb.bar(xs, mse_bwts, color=col_list, alpha=0.85)
ax_mb.axhline(0, color="black", lw=0.8)
ax_mb.set_title("MSE backward transfer\n(lower = less forgetting)")
ax_mb.set_xticks(xs); ax_mb.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
for x, v in zip(xs, mse_bwts):
    ax_mb.text(x, v + 0.002 * (1 if v >= 0 else -1), f"{v:+.4f}", ha="center", fontsize=7)

# dφ BWT (secondary)
dphi_bwts = [dphi_bwt(dphi_mat[m]) for m in METHODS]
ax_db.bar(xs, dphi_bwts, color=col_list, alpha=0.85)
ax_db.axhline(0, color="black", lw=0.8)
ax_db.set_title("dφ backward transfer\n(lower = less forgetting)")
ax_db.set_xticks(xs); ax_db.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
for x, v in zip(xs, dphi_bwts):
    ax_db.text(x, v + 0.05 * abs(v) + 0.05, f"{v:+.2f}", ha="center", fontsize=7)

# MSE diagonal
ax_m.bar(xs, [mse_diag[m].mean() for m in METHODS], color=col_list, alpha=0.85)
ax_m.set_title("Mean MSE at training time\n(lower = better)")
ax_m.set_xticks(xs); ax_m.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)

# rec_ora over time
ax_r.set_title("Schema recognition (nearest-task-oracle)")
for method in METHODS:
    ax_r.plot(range(T), rec_ora[method], "o-", color=COLS[method], label=method, lw=2)
ax_r.axhline(1 / N_SCHEMAS, color="gray", ls="--", label="chance")
ax_r.set_xticks(range(T))
ax_r.set_xticklabels([f"t={t}\n{NAMES[t]}" for t in range(T)], fontsize=6.5)
ax_r.set_ylim(-0.05, 1.1); ax_r.legend(fontsize=7)

# Convergence speed (epochs to half)
for method in METHODS:
    vals = [h for _, h in conv_records[method]]
    ax_c.plot(range(T), vals, "o-", color=COLS[method], label=method, lw=2)
ax_c.set_title("Convergence speed (epochs to 50% MSE drop)\nlower = faster")
ax_c.set_xticks(range(T))
ax_c.set_xticklabels([f"t={t}\n{NAMES[t]}" for t in range(T)], fontsize=6.5)
ax_c.legend(fontsize=7)

plt.tight_layout()
plt.savefig("fig_piagets_final.png", dpi=120)
print("  saved fig_piagets_final.png")
plt.close()

# ─── Step 5: Text summary ────────────────────────────────────────────────────
with open("piagets_bwt_results.txt", "w") as f:
    f.write("PIAGETS benchmark results\n")
    f.write(f"Stream: {NAMES}\nClasses: {CLASSES}\n")
    f.write(f"λ_EWC={LAM_EWC}  λ_assim={LAM_ASSIM}  λ_accom={LAM_ACCOM}  "
            f"σ_schema={SCHEMA_SIGMA}  N_WARMUP={N_WARMUP}  ANNEAL_FRAC={ANNEAL_FRAC:.2f}\n\n")
    f.write(f"  {'method':22s}  {'mse_BWT':>8}  {'dphi_BWT':>9}  "
            f"{'mse':>9}  {'dphi':>8}  {'ARI':>5}  "
            f"{'cLor':>5}  {'cTor':>5}  {'cVdP':>5}  "
            f"{'c_ass':>6}  {'c_acc':>6}  {'wtime':>7}\n")
    for method, mb, db, ro, rc, dphi_d, mse_avg, c_all, c_ass, c_acc, ms_a, ms_c, wt, D, ari, c_lor, c_tor, c_vdp in result_lines:
        f.write(f"  {method:22s}  {mb:+8.4f}  {db:+9.3f}  "
                f"{mse_avg:9.5f}  {dphi_d:8.3f}  {ari:5.3f}  "
                f"{c_lor:5.1f}  {c_tor:5.1f}  {c_vdp:5.1f}  "
                f"{c_ass:6.1f}  {c_acc:6.1f}  {wt:6.0f}s\n")
    f.write("\n")
    for method, mb, db, ro, rc, dphi_d, mse_avg, c_all, c_ass, c_acc, ms_a, ms_c, wt, D, ari, c_lor, c_tor, c_vdp in result_lines:
        f.write(f"\n{method}:\n")
        f.write(f"  mse_BWT={mb:+.5f}  dphi_BWT={db:+.3f}  ARI={ari:.3f}  "
                f"mse={mse_avg:.5f}  dphi={dphi_d:.3f}  walltime={wt:.0f}s\n")
        f.write(f"  conv Lor/Tor/VdP: {c_lor:.1f} / {c_tor:.1f} / {c_vdp:.1f}\n")
        f.write(f"  conv all/assim/accom: {c_all:.1f} / {c_ass:.1f} / {c_acc:.1f}\n")
        f.write(f"  mse_start assim/accom: {ms_a:.4f} / {ms_c:.4f}\n")
        f.write(f"  MSE diagonal: " +
                "  ".join(f"{mse_diag[method][i]:.5f}" for i in range(T)) + "\n")
        f.write(f"  dφ diagonal:  " +
                "  ".join(f"{D[i,i]:.3f}" for i in range(T)) + "\n")
        f.write(f"  mse_start per task: " +
                "  ".join(f"{'S' if mode=='assimilation' else 'A'}{mse_start[method][i]:.4f}"
                           for i, (mode, _) in enumerate(conv_records[method])) + "\n")
        f.write(f"  conv_half per task: " +
                "  ".join(f"{'S' if mode=='assimilation' else 'A'}{h}"
                           for mode, h in conv_records[method]) + "\n")
        f.write(f"  mse_BWT per task: " +
                "  ".join(f"{MSE_mat[method][T-1,i]-MSE_mat[method][i,i]:+.5f}"
                           for i in range(T-1)) + "\n")
        f.write(f"  dphi_BWT per task: " +
                "  ".join(f"{D[T-1,i]-D[i,i]:+.3f}" for i in range(T-1)) + "\n")

print("  saved piagets_bwt_results.txt")

# ─── Step 6: φ clustering analysis ──────────────────────────────────────────
# For each method that uses signature EWC, print the stored φ embeddings
# per task and check whether they cluster into the correct 3 schema classes.
print("\n" + "=" * 64)
print("Step 6: φ-embedding clustering")
print("=" * 64)
print(f"  φ layout: phi_lin(M-P={M-P})  phi_acts(P={P})  phi_act_std(P={P})"
      f"  phi_topo(1)  phi_lyap(1)  → total {M-P + P + P + 2} dims")
print(f"  Key slice for separation: phi_act_std = indices [{M}:{M+P}]")
print(f"  Oracle phi_act_std.mean():  Lorenz≈0  Torus/VdP≈0.43\n")

for method in METHODS:
    cl = cl_objects.get(method)
    if cl is None or not cl._task_phis:
        continue
    print(f"  ── {method} ──")
    task_phis = cl._task_phis   # list of (class_id, phi_vec)

    # Group by class
    from collections import defaultdict
    by_class: dict = defaultdict(list)
    for cls_id, phi_vec in task_phis:
        by_class[cls_id].append(phi_vec)

    phis_all = np.array([p for _, p in task_phis])
    scale    = np.std(phis_all, axis=0)
    scale[scale < 1e-9] = 1.0

    # Per-class statistics on the key phi_act_std slice
    act_sl = slice(M, M + P)
    print(f"  {'cls':>4}  {'name':>5}  {'n':>2}  "
          f"{'φ_act_std.mean':>14}  {'φ_act_std.std':>13}  {'φ_lyap.mean':>11}")
    for cls_id in sorted(by_class):
        arr = np.array(by_class[cls_id])
        act_means = arr[:, act_sl].mean(axis=1)
        lyap_vals = arr[:, -1]
        print(f"  {cls_id:>4}  {CLASS_NAMES[cls_id]:>5}  {len(arr):>2}  "
              f"  {act_means.mean():>8.4f}±{act_means.std():>5.4f}  "
              f"  {arr[:, act_sl].std(axis=0).mean():>7.4f}         "
              f"  {lyap_vals.mean():>8.4f}±{lyap_vals.std():>5.4f}")

    # Inter-class distances in normalised φ-space
    centroids = {c: np.mean(np.array(v), axis=0) for c, v in by_class.items()}
    classes   = sorted(centroids)
    print(f"\n  Inter-class φ distances (normalised):")
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            d = float(np.linalg.norm((centroids[c1] - centroids[c2]) / (scale + 1e-9)))
            print(f"    {CLASS_NAMES[c1]} vs {CLASS_NAMES[c2]}: {d:.3f}")

    # Nearest-centroid accuracy on the stored task phis
    correct = 0
    for cls_id, phi_vec in task_phis:
        pred = min(centroids, key=lambda c: np.linalg.norm((phi_vec - centroids[c]) / (scale + 1e-9)))
        correct += int(pred == cls_id)
    print(f"\n  Nearest-centroid accuracy on stored task φs: "
          f"{correct}/{len(task_phis)} = {correct/len(task_phis)*100:.0f}%\n")

print("\nDone.")
