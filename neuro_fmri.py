"""Schema-recognition on the lab's fMRI BOLD data (Koppe et al. 2019, PLOS CB).

26 subjects, 20 prefrontal/parietal ROIs x 360 TRs of BOLD, + 5 stimulus-input
regressors. We test the recognition premise in a *validatable* way: dynamical
fingerprinting. Split each subject's recording into two halves; compute a
gauge-free dynamical signature per half; ask whether half-1 of a subject best
matches that SAME subject's half-2 (within-subject < across-subject). Subjects
play the role of schemas, the two halves are within-class "variants", and the
subject ID is ground truth.

Signatures compared:
  - dynamical (ours): Koopman/DMD eigenvalue spectrum of the PCA-reduced BOLD
    (the data-side version of SC-RNN's Koopman channel).
  - static FC (reference): upper-triangle of the ROI correlation matrix
    (the classic connectome fingerprint, Finn et al. 2015) -- a baseline.

  python neuro_fmri.py
"""
import glob, numpy as np, scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all():
    out = []
    for f in sorted(glob.glob("data_fmri/datafile_*.mat")):
        S = sio.loadmat(f)["PLRNN"][0, 0]
        out.append((np.asarray(S["data"], float), np.asarray(S["Inp"], float)))
    return out                                   # list of (20x360 BOLD, 5x360 Inp)


def dmd_signature(bold, k=8):
    """Koopman/DMD eigenvalue spectrum of the PCA-reduced multivariate BOLD.
    bold: (n_roi, T)."""
    Z = StandardScaler().fit_transform(bold.T)            # (T, n_roi)
    Z = PCA(n_components=k, random_state=0).fit_transform(Z)
    X0, X1 = Z[:-1].T, Z[1:].T
    A = X1 @ np.linalg.pinv(X0)                            # (k,k) DMD operator
    ev = np.linalg.eigvals(A)
    ev = ev[np.argsort(-np.abs(ev))]
    return np.concatenate([np.abs(ev), np.angle(ev)])     # 2k


def fc_signature(bold):
    C = np.corrcoef(bold)                                  # (n_roi,n_roi)
    return C[np.triu_indices_from(C, k=1)]                 # upper triangle


def fingerprint(sigs_h1, sigs_h2):
    """Cross-half identification accuracy (both directions). sigs_*: (n_subj, D).
    Match by correlation similarity (Finn protocol). Chance = 1/n_subj."""
    def corr_sim(A, B):
        A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return A @ B.T
    S = corr_sim(sigs_h1, sigs_h2)
    n = len(S)
    acc1 = np.mean(S.argmax(1) == np.arange(n))            # h1 -> h2
    acc2 = np.mean(S.argmax(0) == np.arange(n))            # h2 -> h1
    return 0.5 * (acc1 + acc2), S


def main():
    data = load_all()
    n = len(data)
    print(f"[load] {n} subjects, BOLD {data[0][0].shape} (ROI x TR)")
    half = data[0][0].shape[1] // 2

    mats = {}
    for name, sigfun in [("dynamical (DMD/Koopman, ours)", dmd_signature),
                         ("static FC (reference)", fc_signature)]:
        h1 = np.array([sigfun(b[:, :half]) for b, _ in data])
        h2 = np.array([sigfun(b[:, half:]) for b, _ in data])
        acc, S = fingerprint(h1, h2)
        print(f"[fingerprint] {name:<32}: ID acc {acc:.2f}  (chance {1/n:.3f})")
        mats[name] = (S, acc)

    # figure: contrast the two cross-half similarity matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, (S, acc)) in zip(axes, mats.items()):
        im = ax.imshow(S, cmap="magma")
        ax.set_xlabel("subject (half 2)"); ax.set_ylabel("subject (half 1)")
        ax.set_title(f"{name}\nID acc {acc:.2f} (bright diagonal = recognised)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("fMRI fingerprinting: temporal dynamics fail, spatial connectivity "
                 "succeeds", fontsize=12)
    fig.tight_layout(); fig.savefig("fig_fmri_fingerprint.png", dpi=120)
    print("[fig] wrote fig_fmri_fingerprint.png")


if __name__ == "__main__":
    main()
