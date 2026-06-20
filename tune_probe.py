"""Quick diagnostic: probe accuracy vs. n_warmup_epochs and SCHEMA_SIGMA.

For each task t=1..9, trains a fresh r=12 model for n_warmup epochs on that task's
data and checks whether schema_probe correctly classifies the task (assimilation for
repeat class, accommodation for first occurrence).  Uses three class oracles for the
phi-library.

Usage:  python tune_probe.py
"""
import sys, warnings
import numpy as np
import torch

sys.path.insert(0, ".")
import systems
from alrnn import ALRNN
import piagets as pg

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
np.random.seed(0)
torch.manual_seed(0)

M, P, d  = 16, 6, 3
N_DATA   = 8000
SEQ_LEN  = 100
BATCH    = 64
LR       = 1e-3
RANK     = 12

# ── Task stream (same as validate_piagets.py) ─────────────────────────────────
STREAM = [
    ("Lor_r28",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=28)[0].astype("float32"),   0),
    ("Tor_0.382", lambda: systems.torus(n=N_DATA, omega2=0.382)[0].astype("float32"),        1),
    ("VdP_m1.5",  lambda: systems.van_der_pol(n=N_DATA, mu=1.5)[0].astype("float32"),       2),
    ("Lor_r35",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=35)[0].astype("float32"),   0),
    ("Tor_0.618", lambda: systems.torus(n=N_DATA, omega2=0.618)[0].astype("float32"),        1),
    ("VdP_m3.0",  lambda: systems.van_der_pol(n=N_DATA, mu=3.0)[0].astype("float32"),       2),
    ("Lor_r40",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=40)[0].astype("float32"),   0),
    ("VdP_m5.0",  lambda: systems.van_der_pol(n=N_DATA, mu=5.0)[0].astype("float32"),       2),
    ("Tor_0.271", lambda: systems.torus(n=N_DATA, omega2=0.271)[0].astype("float32"),        1),
    ("Lor_r45",   lambda: systems.lorenz(n=N_DATA, dt=0.05, rho=45)[0].astype("float32"),   0),
]
T = len(STREAM)

def fresh_model(seed=42):
    torch.manual_seed(seed)
    return ALRNN(latent_dim=M, obs_dim=d, P=P, rank=RANK)

def phi_of(model, n_avg=3, T_warmup=100, T_track=200, beta=10.0):
    phis = []
    with torch.no_grad():
        for _ in range(n_avg):
            phis.append(pg.diff_phi(model, T_warmup, T_track, beta).numpy())
    return np.mean(phis, axis=0)

def train_model(model, data, epochs, lr=LR, alpha=0.5, alpha_end=0.05):
    pg.train_with_ewc(model, data, cl=None, epochs=epochs, seq_len=SEQ_LEN,
                      batch=BATCH, lr=lr, alpha=alpha, alpha_end=alpha_end,
                      log=lambda _: None)

# ── Step 1: Train three class oracles (one per class) ────────────────────────
print("Training class oracles (r=12, 200 epochs) ...")
class_oracles = {}   # class → model
task_data = []
for name, gen, cls in STREAM:
    task_data.append(gen())

# First occurrence of each class → oracle
class_first = {}
for i, (name, gen, cls) in enumerate(STREAM):
    if cls not in class_first:
        class_first[cls] = i

for cls, t in sorted(class_first.items()):
    name, _, _ = STREAM[t]
    m = fresh_model()
    print(f"  Oracle class={cls}  task={name}")
    train_model(m, task_data[t], epochs=200)
    class_oracles[cls] = m
    print(f"    φ[-3:] = {phi_of(m)[-3:]}")

# Build phi-library from oracle phis and compute phi_scale
all_oracle_phis = [phi_of(class_oracles[c]) for c in range(3)]
phi_scale = np.std(np.stack(all_oracle_phis), axis=0)
phi_scale[phi_scale < 1e-9] = 1.0
print(f"\n  phi_scale: min={phi_scale.min():.3f} max={phi_scale.max():.3f}")

# ── Step 2: Probe accuracy vs. n_warmup and SCHEMA_SIGMA ─────────────────────
WARMUP_VALS = [10, 20, 30, 50]
SIGMA_VALS  = [1.5, 2.0, 2.5]

# Ground truth for each task t=1..9:
# "correct" mode = assimilation if the class was seen in tasks 0..t-1, else accommodation
CLASSES = [cls for _, _, cls in STREAM]
seen_classes = set()
correct_modes = []
for t, (name, gen, cls) in enumerate(STREAM):
    if t == 0:
        correct_modes.append(None)   # no probe at t=0
    else:
        correct_modes.append("assimilation" if cls in seen_classes else "accommodation")
    seen_classes.add(cls)

print("\n  Ground truth:")
for t in range(1, T):
    name, _, cls = STREAM[t]
    print(f"    t={t} {name} class={cls} → {correct_modes[t]}")

# Build a synthetic phi-library using oracle phis (class→centroid, one entry each)
# This simulates what the CL model would have in _task_phis after seeing tasks 0..t-1
# (Using oracle φ as proxy for the CL model's learned φ)
all_seen: list = []   # list of (class, phi_vec) tuples accumulated as t progresses

print(f"\n  Probe accuracy (using fresh r=12 model for each task, no QR-init warmup proxy)")
print(f"  {'task':12s}  true_mode  ", end="")
for n in WARMUP_VALS:
    print(f"n={n:2d}  ", end="")
print()

for sigma in SIGMA_VALS:
    print(f"\n  SCHEMA_SIGMA = {sigma}")
    print(f"  {'task':12s}  true_mode  ", end="")
    for n in WARMUP_VALS:
        print(f"n={n:2d}  ", end="")
    print()

    correct_total   = {n: 0 for n in WARMUP_VALS}
    total_probed    = 0
    all_seen_local  = []

    for t, (name, gen, cls) in enumerate(STREAM):
        if t == 0:
            # First task: add to seen library using oracle φ
            all_seen_local.append((cls, phi_of(class_oracles[cls])))
            continue

        true_mode = correct_modes[t]
        total_probed += 1
        print(f"  t={t} {name:12s}  {true_mode:14s} ", end="", flush=True)

        # Build a dummy PIAGETSContinual object with the accumulated library
        dummy = pg.PIAGETSContinual(
            lam_ewc=5.0, lam_assim=10.0, lam_accom=2.0,
            schema_sigma=sigma,
            sig_fisher_kwargs=dict(n_avg=3, T_warmup=100, T_track=200, beta=10.0))
        dummy._task_phis = list(all_seen_local)

        for n_warmup in WARMUP_VALS:
            m = fresh_model(seed=t)   # proxy for post-QR-init state
            train_model(m, task_data[t], epochs=n_warmup)
            lam, mode, schema = dummy.schema_probe(m, phi_scale, n_avg=2,
                                                    T_warmup=80, T_track=150, beta=10.0)
            correct = (mode == true_mode)
            correct_total[n_warmup] += int(correct)
            marker = "✓" if correct else "✗"
            print(f"  {marker}  ", end="", flush=True)

        print()

        # Update library with oracle φ of this task's class (proxy for CL model's φ)
        all_seen_local.append((cls, phi_of(class_oracles[cls])))

    print(f"  accuracy:")
    for n in WARMUP_VALS:
        acc = correct_total[n] / max(total_probed, 1)
        print(f"    n={n:2d}: {acc:.2f}  ({correct_total[n]}/{total_probed})")

print("\nDone — choose n_warmup and SCHEMA_SIGMA based on accuracy table above.")
print("Recommendation: n_warmup giving >0.8 accuracy with smallest SIGMA that separates classes.")
