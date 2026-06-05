"""Test the schema-recognition idea on REAL neural data with context switching.

Data: PhysioNet EEG Motor Movement/Imagery (via MNE), subject 1, runs 4/8/12 =
motor IMAGERY of left fist (T1) vs right fist (T2), interleaved with rest (T0).
Each cued epoch is a short trajectory of the cortical dynamics in a given CONTEXT.

We compute, per epoch, a data-side dynamical signature -- the Koopman/DMD
eigenvalue spectrum of a delay-embedded, PCA-reduced multivariate segment (the
robust "cheap-probe" version of our SC-RNN embedding) -- and ask: do different
contexts elicit *distinguishable* dynamical regimes (discrimination), and does
the same context recur consistently (the basis for assimilation)?

  python neuro_eegbci.py
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")
from mne.datasets import eegbci
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load(subject=1, runs=(4, 8, 12)):
    fns = eegbci.load_data(subject, list(runs), update_path=True)
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in fns])
    raw.filter(8., 30.)                       # mu/beta motor rhythms
    ev, eid = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, eeg=True)
    # 4 s imagery window after each cue
    ep = mne.Epochs(raw, ev, event_id=eid, tmin=0.5, tmax=3.5, picks=picks,
                    baseline=None, preload=True)
    X = ep.get_data()                          # (n_epochs, n_ch, n_time)
    y = ep.events[:, 2]                        # context code
    label = {v: k for k, v in eid.items()}
    return X, np.array([label[c] for c in y])


def koopman_signature(seg, k=6, delay=3):
    """Data-side Koopman/DMD spectrum of one epoch. seg: (n_ch, n_time).
    PCA-reduce channels -> delay-embed -> fit linear operator A (X' = A X) ->
    sorted eigenvalues as (|lambda|, |angle|). This is the gauge-free temporal
    signature, computed without fitting an RNN."""
    Z = StandardScaler().fit_transform(seg.T)            # (T, n_ch)
    Z = PCA(n_components=k, random_state=0).fit_transform(Z)   # (T, k)
    H = np.concatenate([Z[d:len(Z) - delay + d] for d in range(delay)], axis=1)
    X0, X1 = H[:-1].T, H[1:].T                            # (kd, T-1)
    A = X1 @ np.linalg.pinv(X0)                           # DMD operator
    ev = np.linalg.eigvals(A)
    ev = ev[np.argsort(-np.abs(ev))][:k]
    ev = np.pad(ev, (0, max(0, k - len(ev))))
    return np.concatenate([np.abs(ev), np.abs(np.angle(ev))])


def main():
    print("[load] PhysioNet EEGBCI subject 1, motor imagery (left vs right) ...")
    X, ctx = load()
    print(f"  {len(X)} epochs, {X.shape[1]} ch, {X.shape[2]} samples each; "
          f"contexts: {dict(zip(*np.unique(ctx, return_counts=True)))}")

    sig = np.array([koopman_signature(e) for e in X])
    print(f"[signature] {sig.shape[1]}-dim Koopman/DMD signature per epoch")

    # --- discrimination: classify context from the dynamical signature (CV) ---
    names = {"T0": "rest", "T1": "imagine-L", "T2": "imagine-R"}
    for pair in [("T1", "T2"), ("T0", "T1"), ("T0", "T2")]:
        m = np.isin(ctx, pair)
        Xs, ys = sig[m], ctx[m]
        clf = LogisticRegression(max_iter=2000, C=0.5)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        acc = cross_val_score(clf, StandardScaler().fit_transform(Xs), ys, cv=cv)
        chance = max(np.mean(ys == pair[0]), np.mean(ys == pair[1]))
        print(f"  {names[pair[0]]:>10} vs {names[pair[1]]:<10}: "
              f"CV acc {acc.mean():.2f} ± {acc.std():.2f}  (chance {chance:.2f})")

    # --- visualise: 2D PCA of signatures, coloured by context ---
    P = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(sig))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c, col in zip(["T0", "T1", "T2"], ["gray", "tab:blue", "tab:red"]):
        m = ctx == c
        ax.scatter(P[m, 0], P[m, 1], c=col, label=names[c], s=45,
                   edgecolor="k", linewidth=0.4, alpha=0.8)
    ax.set_xlabel("signature PC1"); ax.set_ylabel("signature PC2")
    ax.set_title("EEG motor-imagery: per-epoch Koopman signature by context\n"
                 "(real data; do contexts form distinguishable dynamical regimes?)")
    ax.legend()
    fig.tight_layout(); fig.savefig("fig_eeg_contexts.png", dpi=120)
    print("[fig] wrote fig_eeg_contexts.png")


if __name__ == "__main__":
    main()
