"""Sleep EEG sleep-staging with AL-RNN signature embedding.

Extended run adding Fisher and silhouette weighting to neuro_sleep.py.
The previous experiment found:
  - model-side signature: ARI≈0.00 (reconstruction collapses on stochastic EEG)
  - per-channel geometry + Ward:   ARI=0.55, LOO=0.86  ← best
  - band-power reference:          ARI=0.46, LOO=0.83
  - BUT reliability weighting diluted geometry back to ARI=0.18
    (the "coarse-dilutes-sharp" problem reliability can't solve)

This run adds Fisher weights (supervised) which should fix the dilution,
and also tests silhouette weights (unsupervised, bootstrapped).

Training improvements over the previous run:
  - EPOCHS_TRAIN 110 → 300
  - P_NONLIN      3  → 6
  - latent_dim   16  → 32
  - alpha       0.15 fixed → 0.5 → 0.05 annealing

Results are cached to /tmp/eeg_sigs.pkl; re-analysis skips re-fitting.
Also runs DSA (Dynamical Similarity Analysis) on the fitted models:
  - extract time-averaged Jacobian K = mean_t J(z_t) from free-run
  - whiten by latent covariance: K_w = Σ^{-1/2} K Σ^{1/2}
  - pairwise distance = min_{C∈O(n)} ‖C K_w C^T - K_w'‖_F  (two-sided Procrustes)
"""
import gc
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import mne
mne.set_log_level("ERROR")
from mne.datasets.sleep_physionet.age import fetch_data
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

import systems
import alrnn
import plrnn
import signatures as sig

torch.set_num_threads(8)

# --------------------------------------------------------------------------- #
#  parameters
# --------------------------------------------------------------------------- #
STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
CLASSES    = ["W", "N2", "N3", "REM"]
PER_CLASS  = 8
CHANS      = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
SUBJECTS   = [0, 1, 2]
SF         = 100.0

LATENT_DIM   = 32
P_NONLIN     = 6
EPOCHS_TRAIN = 300
SAVE_PATH    = "/tmp/eeg_sigs.pkl"


# --------------------------------------------------------------------------- #
#  data loading
# --------------------------------------------------------------------------- #
def load_epochs():
    pool = {c: [] for c in CLASSES}
    paths = fetch_data(subjects=SUBJECTS, recording=[1], on_missing="warn")
    for psg, hyp in paths:
        raw = mne.io.read_raw_edf(psg, preload=False)
        raw.pick(CHANS)
        raw.load_data()
        raw.filter(0.3, 35.0)
        ann = mne.read_annotations(hyp)
        raw.set_annotations(ann, emit_warning=False)
        present = {d: i + 1 for i, d in enumerate(sorted(STAGE_MAP))
                   if d in ann.description}
        if not present:
            del raw; gc.collect(); continue
        ev, _ = mne.events_from_annotations(raw, event_id=present,
                                             chunk_duration=30.0)
        inv = {v: STAGE_MAP[k] for k, v in present.items()}
        ep = mne.Epochs(raw, ev, tmin=0.0, tmax=30.0 - 1 / SF,
                        baseline=None, preload=True, on_missing="ignore")
        X = ep.get_data(copy=True).astype(np.float32)
        y = np.array([inv[c] for c in ep.events[:, 2]])
        for c in CLASSES:
            for seg in X[y == c]:
                pool[c].append(seg)
        del raw, ep, X, y; gc.collect()

    rng = np.random.default_rng(0)
    segs, labels = [], []
    for c in CLASSES:
        arr = pool[c]
        idx = rng.permutation(len(arr))[:PER_CLASS]
        for i in idx:
            segs.append(pool[c][i])
            labels.append(c)
    return segs, np.array(labels)


# --------------------------------------------------------------------------- #
#  fit + extract
# --------------------------------------------------------------------------- #
def fit_and_extract(seg):
    """seg: (3, 3000) float32 → (AL-RNN signature, trained model)."""
    raw = seg.T.astype(np.float64)          # (3000, 3) observations
    data = systems.canonicalize(raw)        # gauge-fixed for the fit
    m = alrnn.ALRNN(latent_dim=LATENT_DIM, obs_dim=3, P=P_NONLIN)
    # anneal 0.5→0.05: start strongly forced (model sees data shape early),
    # end weakly forced (shapes autonomous dynamics, Lyapunov / regions valid)
    plrnn.train(m, data, alpha=0.5, alpha_end=0.05, epochs=EPOCHS_TRAIN,
                seq_len=100, reg_lambda=0.05, log=lambda *a: None)
    # model-side blocks from gauge-fixed fit; geometry blocks from RAW
    # (canonicalise whitens away spatial/spectral signal we want to keep)
    s = sig.extract(m, data[0], obs_data=raw, geom_per_channel=True,
                    k_max=6, lyap_n=1200, n_visit=2500, warmup=400,
                    ph_max_dim=1, ph_n_sub=100)
    return s, m


# --------------------------------------------------------------------------- #
#  baselines
# --------------------------------------------------------------------------- #
def bandpower_feat(seg):
    bands = [(0.5, 4), (4, 8), (8, 13), (12, 16), (16, 30)]
    feats = []
    for ch in seg:
        f, p = welch(ch, fs=SF, nperseg=512)
        tot = p.sum() + 1e-12
        feats += [p[(f >= lo) & (f < hi)].sum() / tot for lo, hi in bands]
    return np.array(feats)


def dmd_feat(seg, k=4, delay=3):
    Z = StandardScaler().fit_transform(seg.T)
    Z = PCA(n_components=min(k, seg.shape[0]), random_state=0).fit_transform(Z)
    H = np.concatenate([Z[d:len(Z) - delay + d] for d in range(delay)], axis=1)
    A = H[1:].T @ np.linalg.pinv(H[:-1].T)
    ev = np.linalg.eigvals(A)
    ev = ev[np.argsort(-np.abs(ev))][:k]
    ev = np.pad(ev, (0, max(0, k - len(ev))))
    return np.concatenate([np.abs(ev), np.abs(np.angle(ev))])


# --------------------------------------------------------------------------- #
#  DSA — Dynamical Similarity Analysis on fitted model Jacobians
# --------------------------------------------------------------------------- #
def _avg_jacobian(model, n=3000, warmup=500):
    """Visit-weighted average Jacobian K = mean_t J(z_t) and latent covariance."""
    M, P, d = model.M, model.P, model.d
    z = torch.zeros(M, dtype=model.A.dtype)
    z[:d] = torch.tensor(np.random.randn(d) * 0.2, dtype=model.A.dtype)
    K_acc = np.zeros((M, M))
    Z_list = []
    with torch.no_grad():
        for t in range(n + warmup):
            D = torch.zeros(M, dtype=model.A.dtype)
            D[:P] = (z[:P] > 0).to(model.A.dtype)
            # J(z) = diag(A) + W * D_omega   (D_omega broadcast over columns)
            J = torch.diag(model.A) + model.W * D
            z_next = model.step(z)
            if t >= warmup and torch.isfinite(z_next).all():
                K_acc += J.numpy()
                Z_list.append(z.numpy().copy())
            z = z_next if torch.isfinite(z_next).all() else z * 0.0
    count = max(len(Z_list), 1)
    K_avg = K_acc / count
    Sigma = np.cov(np.array(Z_list).T) if len(Z_list) > 10 else np.eye(M)
    return K_avg, Sigma


def _whiten(K, Sigma):
    """K_w = Σ^{-1/2} K Σ^{1/2}, Frobenius-normalised."""
    reg = 1e-4 * np.trace(Sigma) / len(Sigma)
    vals, vecs = np.linalg.eigh(Sigma + reg * np.eye(len(Sigma)))
    vals = np.maximum(vals, 1e-8)
    S_h  = vecs @ np.diag(np.sqrt(vals))  @ vecs.T
    S_ih = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    K_w = S_ih @ K @ S_h
    return K_w / max(np.linalg.norm(K_w, 'fro'), 1e-8)


def _procrustes2(K1, K2, n_iter=20, lr=0.05):
    """d = min_{C∈O(n)} ‖CK₁Cᵀ − K₂‖_F  (two-sided Procrustes).

    Init via symmetric-part eigenvectors; refine with Riemannian gradient
    descent on O(n) (projected gradient + SVD retraction).
    """
    # Initialise: align symmetric parts' eigenbases
    Q1 = np.linalg.eigh(0.5 * (K1 + K1.T))[1]
    Q2 = np.linalg.eigh(0.5 * (K2 + K2.T))[1]
    U0, _, Vt0 = np.linalg.svd(Q2 @ Q1.T)
    C = U0 @ Vt0

    best = np.linalg.norm(C @ K1 @ C.T - K2, 'fro')
    for _ in range(n_iter):
        R = C @ K1 @ C.T - K2
        g = 2.0 * (R @ C @ K1.T + R.T @ C @ K1)
        # Project gradient to tangent space of O(n): subtract symmetric component
        gC = C.T @ g
        g_riem = g - C @ (0.5 * (gC + gC.T))
        # SVD retraction onto O(n)
        U, _, Vt = np.linalg.svd(C - lr * g_riem)
        C = U @ Vt
        val = np.linalg.norm(C @ K1 @ C.T - K2, 'fro')
        if val < best:
            best = val
    return best


def dsa_distance_matrix(models, n_jac=3000, warmup=500, seed=0):
    """Pairwise DSA distances between AL-RNN models."""
    np.random.seed(seed)
    print(f"[DSA] extracting avg Jacobians ({n_jac} steps) ...", flush=True)
    ops = []
    for i, m in enumerate(models):
        K, Sigma = _avg_jacobian(m, n=n_jac, warmup=warmup)
        ops.append(_whiten(K, Sigma))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(models)}", flush=True)

    n = len(ops)
    D = np.zeros((n, n))
    print(f"[DSA] computing {n*(n-1)//2} pairwise Procrustes distances ...",
          flush=True)
    for i in range(n):
        for j in range(i + 1, n):
            d = _procrustes2(ops[i], ops[j])
            D[i, j] = D[j, i] = d
    return D


def euclid_D(F):
    F = StandardScaler().fit_transform(F)
    return np.linalg.norm(F[:, None] - F[None], axis=2)


# --------------------------------------------------------------------------- #
#  evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate(name, D, labels, n_clusters):
    aris = {}
    for meth in ("average", "ward"):
        pred = sig.cluster(D, n_clusters=n_clusters, method=meth)
        aris[meth] = adjusted_rand_score(labels, pred)
    knn = KNeighborsClassifier(n_neighbors=3, metric="precomputed")
    correct = 0
    for tr, te in LeaveOneOut().split(D):
        knn.fit(D[np.ix_(tr, tr)], labels[tr])
        correct += int(knn.predict(D[np.ix_(te, tr)])[0] == labels[te][0])
    loo = correct / len(labels)
    print(f"  {name:38s}  ARI(avg)={aris['average']:+.3f}  ARI(ward)={aris['ward']:+.3f}"
          f"  LOO-3NN={loo:.2f}")
    return aris, loo


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    # --- load or fit ---
    if __import__("os").path.exists(SAVE_PATH):
        print(f"[cache] loading {SAVE_PATH} ...", flush=True)
        with open(SAVE_PATH, "rb") as f:
            cache = pickle.load(f)
        sigs   = cache["sigs"]
        labels = cache["labels"]
        segs   = cache["segs"]
        models = cache.get("models", None)
        print(f"  loaded {len(sigs)} signatures, classes "
              f"{dict(zip(*np.unique(labels, return_counts=True)))}",
              flush=True)
        if models is None:
            print("  (no models in cache — DSA will be skipped)", flush=True)
        print(flush=True)
    else:
        print("[load] Sleep-EDF epochs ...", flush=True)
        segs, labels = load_epochs()
        print(f"  {len(segs)} epochs; "
              f"classes {dict(zip(*np.unique(labels, return_counts=True)))}", flush=True)
        chance = max(np.bincount([CLASSES.index(l) for l in labels])) / len(labels)
        print(f"  majority-class baseline = {chance:.2f}\n", flush=True)

        print(f"[fit] {LATENT_DIM}-dim ALRNN, P={P_NONLIN}, {EPOCHS_TRAIN} epochs, "
              f"alpha 0.5→0.05 ...", flush=True)
        sigs, models = [], []
        t0 = time.time()
        for i, seg in enumerate(segs):
            s, m = fit_and_extract(seg)
            sigs.append(s)
            models.append(m)
            gc.collect()
            if (i + 1) % 4 == 0 or i == 0:
                print(f"  {i+1:2d}/{len(segs)}  ({time.time()-t0:.0f}s)  "
                      f"n_regions={s.meta['n_regions']}  "
                      f"clip={s.meta['clip_rate']:.2f}  "
                      f"lyap={s.blocks['lyap_signs']['data']}", flush=True)

        print(f"\n[fit] wall time {time.time()-t0:.0f}s", flush=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({"sigs": sigs, "labels": labels, "segs": segs,
                         "models": models}, f)
        print(f"[cache] saved to {SAVE_PATH}\n", flush=True)

    int_labels = np.array([CLASSES.index(l) for l in labels])
    n_cl = len(CLASSES)

    # --- weighting ---
    groups = [[sigs[i] for i in range(len(sigs)) if labels[i] == c]
              for c in CLASSES]
    rel  = sig.estimate_reliability(groups)
    fw   = sig.fisher_weights(sigs, int_labels)

    # bootstrap silhouette from reliability clustering
    D_boot, _ = sig.combine_distance(sigs, gamma=0.2, geom=1.0, reliability=rel)
    pred_boot  = sig.cluster(D_boot, n_clusters=n_cl, method="ward")
    sil_w      = sig.silhouette_weights(sigs, pred_boot)

    # --- per-block diagnostics ---
    iu   = np.triu_indices(len(sigs), 1)
    same = int_labels[iu[0]] == int_labels[iu[1]]
    print("[per-block] within / across mean distance (★ = discriminative):")
    print(f"  {'block':38s} {'class':11s}  {'w':>6} {'a':>6} {'ratio':>5}"
          f"  {'rel':>5} {'fish':>5} {'sil':>5}")
    for b in sorted({n for s in sigs for n in s.blocks}):
        D, cls = sig.block_distance_matrix(sigs, b)
        if D is None:
            continue
        w, a = D[iu][same].mean(), D[iu][~same].mean()
        flag = " ★" if a > 1.3 * (w + 1e-9) else ""
        print(f"  {b:38s} [{cls:11s}]  {w:6.3f} {a:6.3f} {w/(a+1e-9):5.2f}"
              f"  {rel.get(b,0):5.2f} {fw.get(b,0):5.2f} {sil_w.get(b,0):5.2f}{flag}")

    # --- combined results ---
    print(f"\n[results] cluster / LOO agreement with sleep-stage labels "
          f"({n_cl} classes, Ward unless noted):", flush=True)

    # model-side only (gamma=0.2, geom=0 → ignore geometry channel entirely)
    D, _ = sig.combine_distance(sigs, gamma=0.2, geom=0.0, reliability=rel)
    evaluate("model-side only (rel)", D, labels, n_cl)

    # geometry channel only (rotation-invariant pooled)
    geom_inv_blocks = {b: 1.0 for s in sigs for b, v in s.blocks.items()
                       if v["class"] == "geometry" and "perchannel" not in b}
    D, _ = sig.combine_distance(sigs, weights=geom_inv_blocks)
    evaluate("geometry only, pooled (rotation-inv)", D, labels, n_cl)

    # per-channel geometry only (gauge-bearing)
    geom_pc_blocks = {b: 1.0 for s in sigs for b, v in s.blocks.items()
                      if "perchannel" in b}
    D, _ = sig.combine_distance(sigs, weights=geom_pc_blocks)
    evaluate("geometry only, per-channel (gauge)", D, labels, n_cl)

    # full embedding, three weighting modes
    for wname, warg in [("reliability", rel), ("fisher", fw), ("silhouette", sil_w)]:
        for gamma, geom in [(0.2, 1.0), (0.0, 1.0), (0.2, 0.0)]:
            tag = f"full [{wname}] γ={gamma:.1f} geom={geom:.0f}"
            D, _ = sig.combine_distance(sigs, gamma=gamma, geom=geom, reliability=warg)
            evaluate(tag, D, labels, n_cl)

    # baselines on identical epochs
    Dbp  = euclid_D(np.array([bandpower_feat(s) for s in segs]))
    Ddmd = euclid_D(np.array([dmd_feat(s) for s in segs]))
    print()
    evaluate("band-power (reference)", Dbp, labels, n_cl)
    evaluate("DMD/Koopman (cheap)", Ddmd, labels, n_cl)

    # DSA on fitted AL-RNN models
    if models is not None:
        print(flush=True)
        D_dsa = dsa_distance_matrix(models)
        evaluate("DSA (model Jacobians, Procrustes)", D_dsa, labels, n_cl)
    else:
        print("\n[DSA] skipped (no models in cache)", flush=True)

    # --- top reliability blocks ---
    print("\n[reliability] top blocks:")
    for k in sorted(rel, key=lambda b: -rel[b])[:8]:
        print(f"   {k:38s}  rel={rel[k]:.2f}  fish={fw.get(k,0):.2f}"
              f"  sil={sil_w.get(k,0):.2f}")

    print("\n[fisher] top blocks:")
    for k in sorted(fw, key=lambda b: -fw[b])[:8]:
        print(f"   {k:38s}  rel={rel.get(k,0):.2f}  fish={fw[k]:.2f}"
              f"  sil={sil_w.get(k,0):.2f}")

    # --- n_regions / clip summary by class ---
    print("\n[reconstruction quality] per class:")
    for c in CLASSES:
        idxs = [i for i, l in enumerate(labels) if l == c]
        nr = [sigs[i].meta["n_regions"] for i in idxs]
        cl = [sigs[i].meta["clip_rate"] for i in idxs]
        print(f"  {c:5s}  n_regions={np.mean(nr):.1f}±{np.std(nr):.1f}"
              f"  clip={np.mean(cl):.2f}±{np.std(cl):.2f}")


if __name__ == "__main__":
    main()
