"""E0 — smallest closed DSR loop on Lorenz (primer §3, experiments_00 §2).

Train a PLRNN, free-run generate, overlay vs truth, report invariant metrics.
Success criterion is invariant agreement, NOT forecast accuracy.

Usage:  python run_e0.py [--tau 5] [--latent 20] [--epochs 40]
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import systems
import plrnn
import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="lorenz")
    ap.add_argument("--alpha", type=float, default=0.3, help="GTF mixing (0=BPTT,1=hard)")
    ap.add_argument("--alpha_end", type=float, default=None, help="anneal alpha -> this")
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="e0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[data] generating {args.system} ...")
    data, _ = systems.SYSTEMS[args.system]()
    train_data, test_data = data[:15000], data[15000:]

    print(f"[train] PLRNN latent={args.latent} alpha={args.alpha} epochs={args.epochs}")
    model = plrnn.PLRNN(latent_dim=args.latent, obs_dim=data.shape[1])
    hist = plrnn.train(model, train_data, alpha=args.alpha,
                       alpha_end=args.alpha_end, epochs=args.epochs)

    print("[gen] free-running model ...")
    gen = model.free_run(test_data[0], n=len(test_data))

    finite = np.isfinite(gen).all(axis=1)
    frac_ok = finite.mean()
    gen_ok = gen[finite]
    # Divergence: free-run leaving the data's bounding box by a wide margin is
    # the unbounded/over-expansive failure mode, even if values stay "finite".
    data_scale = np.abs(test_data).max()
    gen_scale = np.abs(gen_ok).max() if len(gen_ok) else np.inf
    diverged = (frac_ok < 0.9) or (gen_scale > 10 * data_scale)
    if diverged:
        print(f"[warn] free-run DIVERGED: {1-frac_ok:.0%} non-finite, "
              f"max|gen|={gen_scale:.1e} vs max|data|={data_scale:.1f} "
              f"(unbounded / over-expansive failure mode).")

    # ---- invariant metrics ----
    print("\n[metrics] (judge by invariants, not forecasts)")
    if len(gen_ok) > 1000:
        print(f"  D_stsp           : {metrics.d_stsp(gen_ok, test_data):.4f}")
        print(f"  power-spec dist  : {metrics.ps_distance(gen_ok, test_data):.4f}")
    spec = metrics.lyapunov_spectrum(model, test_data[0])
    print(f"  max Lyapunov     : {spec[0]:+.4f} nats/step  "
          f"(Lorenz true λ1≈+0.009/step at dt=0.01)")
    print(f"  Lyap spectrum    : {np.array2string(spec[:3], precision=4)}")
    print(f"  Kaplan–Yorke dim : {metrics.kaplan_yorke(spec):.3f}  "
          f"(Lorenz true ≈ 2.06)")

    # ---- figure: true vs generated attractor + loss ----
    fig = plt.figure(figsize=(13, 4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(*test_data[:5000].T, lw=0.3, color="k")
    ax1.set_title(f"true {args.system}")
    ax2 = fig.add_subplot(132, projection="3d")
    g = gen_ok[:5000]
    ax2.plot(*g.T, lw=0.3, color="C3")
    ax2.set_title("PLRNN free-run")
    ax3 = fig.add_subplot(133)
    ax3.plot(hist)
    ax3.set_xlabel("epoch"); ax3.set_ylabel("train MSE"); ax3.set_yscale("log")
    ax3.set_title("teacher-forced loss")
    fig.tight_layout()
    path = f"{args.out}_{args.system}_alpha{args.alpha}.png"
    fig.savefig(path, dpi=120)
    print(f"\n[fig] wrote {path}")


if __name__ == "__main__":
    main()
