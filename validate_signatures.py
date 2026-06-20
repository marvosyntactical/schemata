"""Validate the signature embedding (signatures.py): diverse attractors should
land far apart, affine variants of one attractor should land close.

Extended run on A6000: 5 classes × 6 variants, 500 epochs, GPU training,
beta_2 (voids) in persistent homology (was disabled for memory), k_max=10.

Topological ladder:
  limit_cycle  -- periodic     (beta_1=1, one zero Lyapunov, entropy=0)
  torus        -- quasiperiodic (beta_1=2, two zeros, entropy=0)
  rossler      -- chaotic, 1-lobe (one positive Lyapunov)
  lorenz       -- chaotic, 2-lobe (different branched-manifold template)
  van_der_pol  -- relaxation limit cycle: topologically = limit_cycle
                  (beta_1=1, one zero Lyapunov) but waveform/rate very
                  different. Tests the gamma knob: gamma=0 should merge
                  vdp with limit_cycle; gamma>0 should split them.

Also validates the three weighting modes from signatures.py:
  reliability   -- Layer-2 unsupervised (estimate_reliability)
  fisher        -- supervised Fisher ratio (fisher_weights)
  silhouette    -- unsupervised post-hoc (silhouette_weights, bootstrapped
                   from the reliability clustering)
"""
import pickle
import time
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score

import systems
import alrnn
import plrnn
import metrics
import signatures as sig

# For this 16-dim model CPU is ~2x faster than CUDA (GPU launch overhead dominates
# at small batch size). 8 threads hits the sweet spot on this 128-core machine.
DEVICE = "cpu"
torch.set_num_threads(8)

CLASSES = ["limit_cycle", "torus", "rossler", "lorenz", "van_der_pol"]
N_VAR   = 6      # affine variants per class
EPOCHS  = 500    # training epochs per model  (was 180)
N       = 8000   # trajectory length
FAITHFUL_DSTSP = 16.0
SAVE_PATH = "/tmp/synth_sigs.pkl"


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #
def fit(data):
    data = systems.canonicalize(data)
    m = alrnn.ALRNN(latent_dim=16, obs_dim=3, P=3)
    # anneal alpha 0.3 → 0.05: start strongly teacher-forced (covers both attractor
    # lobes early), end weakly forced (shapes autonomous free-run dynamics).
    plrnn.train(m, data, alpha=0.3, alpha_end=0.05, epochs=EPOCHS, seq_len=100,
                reg_lambda=0.05, device=DEVICE, log=lambda *a: None)
    m.cpu()   # move back; free_run / enumerate_visited_regions use numpy
    return m, data


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _cluster_eval(sigs, labels, rel=None, gamma=0.2, method="ward"):
    """Combined distance → cluster → ARI/silhouette/sep. Returns (ari, sil, w, a, D)."""
    D, _ = sig.combine_distance(sigs, gamma=gamma, reliability=rel)
    n_cl = len(np.unique(labels))
    pred = sig.cluster(D, n_clusters=n_cl, method=method)
    ari  = adjusted_rand_score(labels, pred)
    sil  = silhouette_score(D, labels, metric="precomputed") if n_cl > 1 else 0.0
    iu   = np.triu_indices(len(sigs), 1)
    same = labels[iu[0]] == labels[iu[1]]
    w_m  = D[iu][same].mean()
    a_m  = D[iu][~same].mean()
    return ari, sil, w_m, a_m, D


def _print_table_row(tag, gamma, ari, sil, sep, w_m, a_m):
    print(f"  {tag:22s}  γ={gamma:.1f}  ARI={ari:+.2f}  sil={sil:+.3f}  "
          f"sep={sep:.2f}  (w={w_m:.3f}  a={a_m:.3f})")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    sigs, labels, groups, recon = [], [], [], []
    print(f"[fit] {len(CLASSES)} classes × {N_VAR} variants, "
          f"{EPOCHS} epochs, device={DEVICE}\n")
    t0 = time.time()

    for ci, c in enumerate(CLASSES):
        grp = []
        for v in range(N_VAR):
            data, _ = systems.SYSTEMS[c](n=N)
            data = systems.affine_variant(data, seed=100 * seed + 10 * ci + v)
            t = time.time()
            m, cdata = fit(data)
            s = sig.extract(m, cdata[0], k_max=10, lyap_n=3000,
                            n_visit=5000, warmup=500, ph_max_dim=2, ph_n_sub=150)
            gen  = m.free_run(cdata[0], n=5000)
            ok   = np.isfinite(gen).all(1)
            dstsp = metrics.d_stsp(gen[ok], cdata) if ok.mean() > 0.9 else float("inf")
            sigs.append(s); labels.append(ci); grp.append(s); recon.append(dstsp)
            print(f"  {c:12s} v{v}  {time.time()-t:5.1f}s  "
                  f"regions={s.meta['n_regions']:2d}  clip={s.meta['clip_rate']:.2f}  "
                  f"D_stsp={dstsp:6.2f}  lyap={s.blocks['lyap_signs']['data']}")
        groups.append(grp)

    labels = np.array(labels)
    recon  = np.array(recon)
    print(f"\n[fit] wall time {time.time()-t0:.0f}s\n")

    with open(SAVE_PATH, "wb") as f:
        pickle.dump({"sigs": sigs, "labels": labels, "recon": recon,
                     "groups": groups, "classes": CLASSES}, f)
    print(f"[cache] saved to {SAVE_PATH}\n")

    # --- block-level diagnostics ---
    iu   = np.triu_indices(len(sigs), 1)
    same = labels[iu[0]] == labels[iu[1]]

    # Layer-2 reliability (unsupervised, from variant groups as replicates)
    rel  = sig.estimate_reliability(groups)
    # Fisher weights (supervised: uses ground-truth class labels)
    fw   = sig.fisher_weights(sigs, labels)
    # Silhouette weights: bootstrapped from a first reliability-based clustering
    D_boot, _ = sig.combine_distance(sigs, gamma=0.2, reliability=rel)
    pred_boot  = sig.cluster(D_boot, n_clusters=len(CLASSES), method="ward")
    sil_w = sig.silhouette_weights(sigs, pred_boot)

    print("[per-block] within / across mean distance (★ = discriminative):")
    print(f"  {'block':34s} {'class':11s}  {'w':>6} {'a':>6} {'ratio':>5}  "
          f"{'rel':>5} {'fish':>5} {'sil':>5}")
    for b in sorted({n for s in sigs for n in s.blocks}):
        D, cls = sig.block_distance_matrix(sigs, b)
        if D is None:
            continue
        w, a = D[iu][same].mean(), D[iu][~same].mean()
        flag = " ★" if a > 1.3 * (w + 1e-9) else ""
        print(f"  {b:34s} [{cls:11s}]  {w:6.3f} {a:6.3f} {w/(a+1e-9):5.2f}  "
              f"{rel.get(b,0):5.2f} {fw.get(b,0):5.2f} {sil_w.get(b,0):5.2f}{flag}")

    # --- combined embedding: 3 weighting modes × 3 gamma values ---
    print("\n[combined] ARI / silhouette / separation (ward linkage):")
    print(f"  {'weighting':22s}  {'γ':>3}  {'ARI':>5}  {'sil':>6}  "
          f"{'sep':>5}  {'within':>7}  {'across':>7}")
    for mode_name, rel_arg in [
            ("reliability",  rel),
            ("fisher",       fw),
            ("silhouette",   sil_w),
    ]:
        for gamma in [0.0, 0.2, 1.0]:
            ari, sil, w_m, a_m, _ = _cluster_eval(sigs, labels, rel=rel_arg, gamma=gamma)
            sep = a_m / (w_m + 1e-9)
            print(f"  {mode_name:22s}  {gamma:.1f}  {ari:+.2f}  {sil:+.3f}  "
                  f"{sep:5.2f}  {w_m:7.3f}  {a_m:7.3f}")

    # --- gamma-knob test: vdp should merge with limit_cycle at gamma=0 ---
    lc_idx  = CLASSES.index("limit_cycle")
    vdp_idx = CLASSES.index("van_der_pol")
    labels_topo = np.where(labels == vdp_idx, lc_idx, labels)
    print("\n[gamma-knob] van_der_pol grouped with limit_cycle (4-class) vs apart (5-class):")
    print(f"  {'γ':>4}  {'ARI-5class':>10}  {'ARI-4class':>10}  "
          f"(4-class should beat 5-class at γ=0)")
    for gamma in [0.0, 0.2, 1.0]:
        D, _ = sig.combine_distance(sigs, gamma=gamma, reliability=rel)
        ari5 = adjusted_rand_score(labels,      sig.cluster(D, n_clusters=5))
        ari4 = adjusted_rand_score(labels_topo, sig.cluster(D, n_clusters=4))
        print(f"  {gamma:4.1f}  {ari5:10.2f}  {ari4:10.2f}")

    # --- faithful subset: embedding quality divorced from reconstruction failures ---
    keep = np.array([(recon[i] < FAITHFUL_DSTSP and sigs[i].meta["n_regions"] > 1)
                     for i in range(len(sigs))])
    print(f"\n[faithful] {keep.sum()}/{len(sigs)} models pass "
          f"(D_stsp < {FAITHFUL_DSTSP} and n_regions > 1):")
    if keep.sum() >= len(CLASSES) and len(np.unique(labels[keep])) >= 2:
        fs  = [s for i, s in enumerate(sigs) if keep[i]]
        fl  = labels[keep]
        f_groups = [[fs[j] for j in range(len(fs)) if fl[j] == ci]
                    for ci in range(len(CLASSES)) if (fl == ci).any()]
        f_rel = sig.estimate_reliability(f_groups)
        f_fw  = sig.fisher_weights(fs, fl)
        for mode_name, rel_arg in [("reliability", f_rel), ("fisher", f_fw)]:
            for gamma in [0.0, 0.2]:
                ari, sil, w_m, a_m, _ = _cluster_eval(fs, fl, rel=rel_arg, gamma=gamma)
                print(f"  {mode_name:12s}  γ={gamma:.1f}  ARI={ari:+.2f}  "
                      f"sil={sil:+.3f}  sep={a_m/(w_m+1e-9):.2f}")
    else:
        print("  too few faithful models (raise EPOCHS or lower FAITHFUL_DSTSP).")

    # --- distance matrix for inspection ---
    D, _ = sig.combine_distance(sigs, gamma=0.2, reliability=rel)
    names = [f"{CLASSES[l][:4]}{i % N_VAR}" for i, l in enumerate(labels)]
    print(f"\n[matrix] combined distance γ=0.2, reliability ({len(names)}×{len(names)}):")
    print("         " + " ".join(f"{n:>7}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:>6}  " + " ".join(f"{D[i,j]:7.3f}" for j in range(len(names))))


if __name__ == "__main__":
    main()
