"""Run the AL-RNN schema signature (signatures.py) on REAL labelled neural data:
Sleep-EDF (PhysioNet) polysomnography, where the label is the sleep stage and the
stages are genuinely distinct DYNAMICAL regimes -- wake (fast, low-amplitude),
N2 (spindles), N3 (slow-wave delta), REM (mixed). This is the fair testbed for a
*temporal-dynamical* signature, unlike motor-imagery EEG whose discriminative
information is spatial/lateralised (summary.md sect. 3.5).

Per 30 s epoch we fit a small AL-RNN to the 3-channel (2 EEG + EOG) trajectory,
read the full gauge-free signature, build the channel-weighted distance, cluster,
and ask whether the clusters agree with the stage labels (adjusted Rand index).
Two baselines on the SAME epochs put the number in context:
  - band-power  : relative delta/theta/alpha/sigma/beta power (a strong, standard
                  sleep-staging feature -- the geometry/spectral reference).
  - DMD/Koopman : the repo's cheap data-side temporal signature.

  python neuro_sleep.py
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import torch
import mne
mne.set_log_level("ERROR")
from mne.datasets.sleep_physionet.age import fetch_data
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

import systems, alrnn, plrnn, signatures as sig

torch.set_num_threads(2)

STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
CLASSES = ["W", "N2", "N3", "REM"]      # dynamically-distinct, well-populated stages
PER_CLASS = 9                            # epochs per class (balanced)
CHANS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
SUBJECTS = [0, 1, 2, 3]
EPOCHS_TRAIN = 110
SF = 100.0
P_NONLIN = 3            # # nonlinear (ReLU) units; P=M turns the AL-RNN into a full PLRNN


def load_epochs():
    """Balanced 30 s epochs across a few subjects, labelled by sleep stage.
    Memory-frugal: pick the 3 channels BEFORE loading data, work in float32, and
    free each night before the next (the workstation has ~2 GB free)."""
    import gc
    pool = {c: [] for c in CLASSES}
    paths = fetch_data(subjects=SUBJECTS, recording=[1], on_missing="warn")
    for psg, hyp in paths:
        raw = mne.io.read_raw_edf(psg, preload=False)
        raw.pick(CHANS)                            # 3 of 7 channels, then load
        raw.load_data()
        raw.filter(0.3, 35.0)                      # keep slow waves; drop drift/EMG
        ann = mne.read_annotations(hyp)
        raw.set_annotations(ann, emit_warning=False)
        present = {d: i + 1 for i, d in enumerate(sorted(STAGE_MAP)) if d in ann.description}
        if not present:
            del raw; gc.collect(); continue
        ev, _ = mne.events_from_annotations(raw, event_id=present, chunk_duration=30.)
        inv = {v: STAGE_MAP[k] for k, v in present.items()}
        ep = mne.Epochs(raw, ev, tmin=0.0, tmax=30.0 - 1 / SF, baseline=None,
                        preload=True, on_missing="ignore")
        X = ep.get_data(copy=True).astype(np.float32)   # (n, 3, 3000)
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
            segs.append(pool[c][i]); labels.append(c)
    return segs, np.array(labels)


# ----------------------------- signatures ---------------------------------- #
def fit_and_extract(seg):
    """seg: (3, 3000) -> trained AL-RNN signature."""
    raw = seg.T.astype(np.float64)                            # (3000, 3) observations
    data = systems.canonicalize(raw)                          # gauge-fixed, for the fit
    m = alrnn.ALRNN(latent_dim=16, obs_dim=3, P=P_NONLIN)
    plrnn.train(m, data, alpha=0.15, epochs=EPOCHS_TRAIN, seq_len=100,
                reg_lambda=0.05, log=lambda *a: None)
    # model-side channels from the gauge-fixed fit; data-side geometry channel from
    # the RAW observations (canonicalisation whitens away the spatial/spectral signal)
    s = sig.extract(m, data[0], obs_data=raw, geom_per_channel=True, k_max=6,
                    lyap_n=1200, n_visit=2500, warmup=400)
    return s, data


def bandpower_feat(seg):
    bands = [(0.5, 4), (4, 8), (8, 13), (12, 16), (16, 30)]    # delta theta alpha sigma beta
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


# ----------------------------- evaluation ---------------------------------- #
def evaluate(name, D, labels, n_clusters):
    ari = {}
    for meth in ("average", "ward"):
        pred = sig.cluster(D, n_clusters=n_clusters, method=meth)
        ari[meth] = adjusted_rand_score(labels, pred)
    # leave-one-out kNN on the distance matrix = supervised discriminability ceiling
    knn = KNeighborsClassifier(n_neighbors=3, metric="precomputed")
    correct = 0
    for tr, te in LeaveOneOut().split(D):
        knn.fit(D[np.ix_(tr, tr)], labels[tr])
        correct += int(knn.predict(D[np.ix_(te, tr)])[0] == labels[te][0])
    loo = correct / len(labels)
    print(f"  {name:22s}  ARI(avg)={ari['average']:+.3f} ARI(ward)={ari['ward']:+.3f}"
          f"  LOO-3NN acc={loo:.2f}")
    return ari, loo


def euclid_D(F):
    F = StandardScaler().fit_transform(F)
    return np.linalg.norm(F[:, None] - F[None], axis=2)


def main():
    print("[load] Sleep-EDF epochs ...", flush=True)
    segs, labels = load_epochs()
    print(f"  {len(segs)} epochs; classes {dict(zip(*np.unique(labels, return_counts=True)))}",
          flush=True)
    chance = max(np.bincount([CLASSES.index(l) for l in labels])) / len(labels)
    print(f"  majority-class baseline acc = {chance:.2f}\n", flush=True)

    # --- our AL-RNN signature ---
    print("[fit] training one AL-RNN per epoch + extracting signature ...", flush=True)
    sigs, recon = [], []
    t0 = time.time()
    import gc
    for i, seg in enumerate(segs):
        s, _ = fit_and_extract(seg)
        sigs.append(s)
        recon.append(s.meta["n_regions"])
        gc.collect()
        if (i + 1) % 6 == 0:
            print(f"  {i+1}/{len(segs)}  ({time.time()-t0:.0f}s)  "
                  f"last n_regions={s.meta['n_regions']} clip={s.meta['clip_rate']:.2f}",
                  flush=True)
    groups = [[sigs[i] for i in range(len(sigs)) if labels[i] == c] for c in CLASSES]
    rel = sig.estimate_reliability(groups)

    import pickle
    with open("/tmp/sleep_sigs.pkl", "wb") as f:
        pickle.dump({"sigs": sigs, "labels": labels, "segs": segs, "rel": rel}, f)
    print("[cache] wrote /tmp/sleep_sigs.pkl (sigs+labels+segs for fast iteration)",
          flush=True)

    print("\n[results] cluster agreement with sleep-stage labels "
          f"({len(CLASSES)} classes):", flush=True)
    # model-side channels only (geom=0): the gauge-free attractor signature
    Dmodel, _ = sig.combine_distance(sigs, gamma=0.2, geom=0.0, reliability=rel)
    evaluate("model-side only", Dmodel, labels, len(CLASSES))
    # + the data-side geometry/spectral channel
    Dfull, _ = sig.combine_distance(sigs, gamma=0.2, geom=1.0, reliability=rel)
    evaluate("+ geometry channel", Dfull, labels, len(CLASSES))
    # geometry channel alone (data-side spectral + spatial only)
    geom_blocks = {b for s in sigs for b, v in s.blocks.items()
                   if v["class"] == "geometry"}
    Dgeom, _ = sig.combine_distance(sigs, weights={b: 1.0 for b in geom_blocks})
    evaluate("geometry channel only", Dgeom, labels, len(CLASSES))

    # --- baselines on the same epochs ---
    Dbp = euclid_D(np.array([bandpower_feat(s) for s in segs]))
    evaluate("band-power (ref)", Dbp, labels, len(CLASSES))
    Ddmd = euclid_D(np.array([dmd_feat(s) for s in segs]))
    evaluate("DMD/Koopman (cheap)", Ddmd, labels, len(CLASSES))

    # most reliable signature blocks
    print("\n[reliability] top signature blocks (Layer-2):", flush=True)
    for k in sorted(rel, key=lambda b: -rel[b])[:6]:
        print(f"   {k:24s} {rel[k]:.2f}")


if __name__ == "__main__":
    main()
