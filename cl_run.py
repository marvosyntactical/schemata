"""Continual-learning DSR with a shared-reader schema memory (dsa_native_ssm.md §8).

For each system in a labelled stream: probe-fit an AL-RNN module, read its
gauge-free embedding, and let the schema memory decide assimilate (reuse a known
module) vs accommodate (allocate a new one) -- the decision happening at training
time. Evaluates decision accuracy, reuse rate, and forgetting (vs a naive single-
model baseline).

  python cl_run.py --mode probe   # separability of class embeddings (set tau)
  python cl_run.py --mode cl      # the full CL stream + evaluation
"""
import argparse
import numpy as np
import torch

import systems
import plrnn          # reuse the model-agnostic GTF trainer
import alrnn
import embeddings
import metrics
from schema_memory import SchemaMemory

# Channel weights from the probe diagnostics: the coordinate-invariant dynamical
# channel (Lyapunov spectrum + dimension) is the only one with a clean within-vs-
# across gap. The Koopman channel has large within-class scatter (fitted-operator
# variability) and the symbolic channel is degenerate on these near-linear systems
# (ReLU units barely flip). v0 decides on the invariant channel; the others are
# extracted and reported but not yet weighted -- the gap they need to close is the
# faithfulness (5.A) / canonical-partition work in dsa_native_ssm.md.
WEIGHTS = {"koopman": 0.0, "symbolic": 0.0, "dynamical": 1.0}


def fit_module(data, P=3, latent=16, alpha=0.15, epochs=170, reg_lambda=0.05, quiet=True):
    data = systems.canonicalize(data)          # fix the rotation/reflection gauge
    m = alrnn.ALRNN(latent_dim=latent, obs_dim=data.shape[1], P=P)
    log = (lambda *a: None) if quiet else print
    plrnn.train(m, data, alpha=alpha, epochs=epochs, seq_len=100,
                reg_lambda=reg_lambda, log=log)
    return m, data


# ---------------------------------------------------------------- probe mode
def probe(seed=0):
    """Train one module per base class, print the z-scored embedding distance
    matrix. Within-class (variant) distances should be << across-class."""
    classes = ["limit_cycle", "torus", "rossler"]
    embs, names, slc = [], [], None
    mem = SchemaMemory()
    print("[probe] fitting two variants per class ...")
    for c in classes:
        for v in range(2):
            data, _ = systems.SYSTEMS[c](n=12000)
            data = systems.affine_variant(data, seed=100 * seed + v)
            m, cdata = fit_module(data)
            e, slc = embeddings.embed(m, cdata[0])
            embs.append(e); names.append(f"{c}#{v}")
            mem.observe(e)
            print(f"  embedded {c}#{v}")
    mu, sd = mem._stats()
    Z = np.array([(e - mu) / sd for e in embs])

    def show(name, lo, hi):
        D = np.linalg.norm(Z[:, lo:hi][:, None] - Z[:, lo:hi][None], axis=2)
        print(f"\n[probe] z-scored distance matrix -- {name} channel:")
        print("        " + "  ".join(f"{n:>12}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:>12} " + "  ".join(f"{D[i,j]:12.2f}" for j in range(len(names))))

    for ch, (lo, hi) in slc.items():
        show(ch, lo, hi)

    # channel-weighted combined distance (what the memory actually uses)
    print("\n[probe] z-scored distance matrix -- WEIGHTED combined "
          f"({WEIGHTS}):")
    Dw = np.zeros((len(names), len(names)))
    for ch, (lo, hi) in slc.items():
        w = WEIGHTS.get(ch, 1.0)
        Dc = np.sum((Z[:, lo:hi][:, None] - Z[:, lo:hi][None]) ** 2, axis=2)
        Dw += w * Dc
    Dw = np.sqrt(Dw)
    print("        " + "  ".join(f"{n:>12}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>12} " + "  ".join(f"{Dw[i,j]:12.2f}" for j in range(len(names))))


# ------------------------------------------------------------------- cl mode
def run_cl(seed=0, tau=3.0, beta=1.0):
    stream = systems.make_stream(seed=seed)
    mem = SchemaMemory(tau=tau, beta=beta, slices=embeddings.SLICES, weights=WEIGHTS)
    modules = []                 # committed (frozen) module per accommodated schema
    proto_label = []             # ground-truth label each prototype was created with
    log = []
    seen_labels = set()

    print(f"[cl] stream of {len(stream)} systems, tau={tau} beta={beta}\n")
    print(f"{'#':>2} {'true':>12} {'decision':>11} {'->schema':>9} "
          f"{'nearest d':>9} {'correct':>7}")
    first_module = {}            # first module per schema, for forgetting eval
    first_data = {}
    for i, (data, true) in enumerate(stream):
        m, cdata = fit_module(data)
        emb, _ = embeddings.embed(m, cdata[0])
        decision, k, dists, attn = mem.query(emb)

        genuinely_new = true not in seen_labels
        if decision == "accommodate":
            kk = mem.add_prototype(emb, true)
            modules.append(m); proto_label.append(true)
            assigned = kk
            correct = genuinely_new
            if true not in first_module:
                first_module[true] = m; first_data[true] = cdata
        else:
            mem.update_prototype(k, emb)
            assigned = k
            correct = (not genuinely_new) and (proto_label[k] == true)
        seen_labels.add(true)

        dmin = dists.min() if len(dists) else float("nan")
        log.append((true, decision, proto_label[assigned], correct))
        print(f"{i:>2} {true:>12} {decision:>11} {proto_label[assigned]:>9} "
              f"{dmin:9.2f} {str(correct):>7}")

    # --- evaluation ---
    acc = np.mean([c for *_, c in log])
    n_assim = sum(1 for _, d, *_ in log if d == "assimilate")
    print(f"\n[eval] decision accuracy : {acc:.0%}")
    print(f"[eval] reuse (assimilate): {n_assim}/{len(log)} systems reused a module")
    print(f"[eval] schemas allocated : {len(modules)} (ground-truth classes: "
          f"{len(set(t for t,*_ in log))})")

    # --- forgetting: committed modules are frozen -> 0 by construction.
    #     Contrast with a naive single model fine-tuned through the stream.
    print("\n[eval] forgetting test (reconstruction D_stsp on FIRST system per schema):")
    print("  modular (frozen modules):")
    for lbl, m in first_module.items():
        gen = m.free_run(first_data[lbl][0], n=4000)
        d = metrics.d_stsp(gen[np.isfinite(gen).all(1)], first_data[lbl]) \
            if np.isfinite(gen).all() else float("inf")
        print(f"    {lbl:>12}: D_stsp {d:6.2f}  (unchanged -- module never overwritten)")

    print("  naive single-model baseline (one AL-RNN fine-tuned through stream):")
    base = alrnn.ALRNN(latent_dim=16, obs_dim=3, P=3)
    base_first = {}
    for i, (raw, true) in enumerate(stream):
        data = systems.canonicalize(raw)
        plrnn.train(base, data, alpha=0.2, epochs=40, seq_len=100, log=lambda *a: None)
        if true not in base_first:
            gen = base.free_run(data[0], n=4000)
            d = metrics.d_stsp(gen[np.isfinite(gen).all(1)], data) \
                if np.isfinite(gen).all() else float("inf")
            base_first[true] = (data, d)
    for lbl, (data, d0) in base_first.items():
        gen = base.free_run(data[0], n=4000)
        d1 = metrics.d_stsp(gen[np.isfinite(gen).all(1)], data) \
            if np.isfinite(gen).all() else float("inf")
        print(f"    {lbl:>12}: D_stsp {d0:6.2f} -> {d1:6.2f}  (drift = forgetting)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "cl"], default="probe")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.mode == "probe":
        probe(seed=args.seed)
    else:
        run_cl(seed=args.seed, tau=args.tau, beta=args.beta)
