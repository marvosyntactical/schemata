"""Validate the signature embedding (signatures.py): diverse attractors should
land far apart, affine variants of one attractor should land close.

Topological ladder of reconstructable classes:
  limit_cycle  -- periodic        (beta_1=1, one zero Lyapunov exponent)
  torus        -- quasiperiodic   (beta_1=2, two zero exponents, zero entropy)
  rossler      -- chaotic         (one positive exponent, branched manifold)

Each appears as N_VAR affine variants (a smooth conjugacy + noise = SAME schema,
different observation frame). A good embedding: within-class << across-class.
"""
import time
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score

import systems
import alrnn
import plrnn
import metrics
import signatures as sig

torch.set_num_threads(2)
CLASSES = ["limit_cycle", "torus", "rossler"]
N_VAR = 3
EPOCHS = 180
N = 8000
FAITHFUL_DSTSP = 16.0     # a model worse than this failed to reconstruct


def fit(data):
    data = systems.canonicalize(data)
    m = alrnn.ALRNN(latent_dim=16, obs_dim=3, P=3)
    plrnn.train(m, data, alpha=0.15, epochs=EPOCHS, seq_len=100,
                reg_lambda=0.05, log=lambda *a: None)
    return m, data


def main(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sigs, labels, groups, recon = [], [], [], []
    print(f"[fit] {len(CLASSES)} classes x {N_VAR} variants, {EPOCHS} epochs\n")
    for ci, c in enumerate(CLASSES):
        grp = []
        for v in range(N_VAR):
            data, _ = systems.SYSTEMS[c](n=N)
            data = systems.affine_variant(data, seed=100 * seed + 10 * ci + v)
            t = time.time()
            m, cdata = fit(data)
            s = sig.extract(m, cdata[0], k_max=6, lyap_n=1500,
                            n_visit=3000, warmup=400)
            # reconstruction sanity: a model that failed to fit contaminates its
            # signature -- surface it rather than trust it blindly.
            gen = m.free_run(cdata[0], n=4000)
            ok = np.isfinite(gen).all(1)
            dstsp = metrics.d_stsp(gen[ok], cdata) if ok.mean() > 0.9 else float("inf")
            sigs.append(s); labels.append(ci); grp.append(s)
            recon.append(dstsp)
            print(f"  {c:12s} v{v}  {time.time()-t:5.1f}s  "
                  f"regions={s.meta['n_regions']:2d} clip={s.meta['clip_rate']:.2f}  "
                  f"D_stsp={dstsp:6.2f}  lyap_signs={s.blocks['lyap_signs']['data']}")
        groups.append(grp)
    labels = np.array(labels)

    # Layer-2 reliability from the variant groups (replicates of each schema)
    rel = sig.estimate_reliability(groups)
    print("\n[reliability] Layer-2 weights (variant groups as replicates):")
    for k in sorted(rel, key=lambda b: -rel[b]):
        print(f"   {k:26s} {rel[k]:.3f}")

    # --- per-block separation diagnostic (within vs across, unweighted) ---
    iu = np.triu_indices(len(sigs), 1)
    same = labels[iu[0]] == labels[iu[1]]
    print("\n[per-block] within / across mean distance (ratio<1 = discriminative):")
    for b in sorted({n for s in sigs for n in s.blocks}):
        D, cls = sig.block_distance_matrix(sigs, b)
        if D is None:
            continue
        w, a = D[iu][same].mean(), D[iu][~same].mean()
        flag = "  <-- signal" if a > 1.3 * (w + 1e-9) else ""
        print(f"   {b:26s} [{cls:11s}] within={w:6.3f} across={a:6.3f} "
              f"ratio={w/(a+1e-9):.2f}{flag}")

    # --- combined embedding at three gamma settings ---
    print("\n[combined] within vs across + clustering, by rate-weight gamma:")
    print(f"   {'gamma':>5} {'within':>7} {'across':>7} {'sep':>5} "
          f"{'ARI':>5} {'silhouette':>10}")
    for gamma in [0.0, 0.2, 1.0]:
        D, _ = sig.combine_distance(sigs, gamma=gamma, reliability=rel)
        within, across = D[iu][same].mean(), D[iu][~same].mean()
        pred = sig.cluster(D, n_clusters=len(CLASSES))
        ari = adjusted_rand_score(labels, pred)
        sil = silhouette_score(D, labels, metric="precomputed")
        tag = {0.0: "topological-only", 0.2: "default", 1.0: "DSA-like"}[gamma]
        print(f"   {gamma:5.1f} {within:7.3f} {across:7.3f} "
              f"{across/(within+1e-9):5.2f} {ari:5.2f} {sil:10.3f}   {tag}")

    # --- faithful subset: isolates embedding quality from the training lottery ---
    recon = np.array(recon)
    keep = np.array([(recon[i] < FAITHFUL_DSTSP and sigs[i].meta["n_regions"] > 1)
                     for i in range(len(sigs))])
    print(f"\n[faithful subset] {keep.sum()}/{len(sigs)} models reconstructed "
          f"(D_stsp<{FAITHFUL_DSTSP}, >1 region):")
    if keep.sum() >= len(CLASSES) and len(np.unique(labels[keep])) >= 2:
        fs = [s for i, s in enumerate(sigs) if keep[i]]
        fl = labels[keep]
        fiu = np.triu_indices(len(fs), 1)
        fsame = fl[fiu[0]] == fl[fiu[1]]
        for gamma in [0.0, 0.2, 1.0]:
            D, _ = sig.combine_distance(fs, gamma=gamma, reliability=rel)
            w, a = D[fiu][fsame].mean(), D[fiu][~fsame].mean()
            sep = a / (w + 1e-9)
            if len(np.unique(fl)) == len(CLASSES):
                pred = sig.cluster(D, n_clusters=len(np.unique(fl)))
                ari = adjusted_rand_score(fl, pred)
            else:
                ari = float("nan")
            print(f"   gamma={gamma:.1f}  within={w:.3f} across={a:.3f} "
                  f"sep={sep:.2f} ARI={ari:.2f}")
    else:
        print("   too few faithful models for a clean test (raise EPOCHS).")

    # show the default-gamma matrix
    D, _ = sig.combine_distance(sigs, gamma=0.2, reliability=rel)
    names = [f"{CLASSES[l][:4]}{i%N_VAR}" for i, l in enumerate(labels)]
    print("\n[matrix] combined distance (gamma=0.2):")
    print("        " + " ".join(f"{n:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:>6} " + " ".join(f"{D[i,j]:6.2f}" for j in range(len(names))))


if __name__ == "__main__":
    main()
