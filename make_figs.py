"""Render figures for SC_RNN.md from the actual logged run results (no training;
numbers are copied verbatim from the runs recorded in dsa_native_ssm.md)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["LC#0", "LC#1", "T#0", "T#1", "R#0", "R#1"]

# --- per-channel z-scored distance matrices (probe, canonicalised, no reg) ---
dyn = np.array([
    [0.00, 1.92, 3.24, 2.90, 3.73, 2.79],
    [1.92, 0.00, 4.39, 3.81, 4.52, 3.99],
    [3.24, 4.39, 0.00, 1.11, 2.72, 2.18],
    [2.90, 3.81, 1.11, 0.00, 2.95, 2.45],
    [3.73, 4.52, 2.72, 2.95, 0.00, 1.18],
    [2.79, 3.99, 2.18, 2.45, 1.18, 0.00]])
koop = np.array([
    [0.00, 6.43, 6.28, 3.76, 3.77, 4.27],
    [6.43, 0.00, 4.48, 4.36, 3.81, 4.04],
    [6.28, 4.48, 0.00, 4.78, 5.23, 3.25],
    [3.76, 4.36, 4.78, 0.00, 2.97, 3.31],
    [3.77, 3.81, 5.23, 2.97, 0.00, 3.17],
    [4.27, 4.04, 3.25, 3.31, 3.17, 0.00]])
sym = np.array([
    [0.00, 4.70, 4.92, 3.52, 0.00, 0.00],
    [4.70, 0.00, 0.37, 1.30, 4.70, 4.70],
    [4.92, 0.37, 0.00, 1.44, 4.92, 4.92],
    [3.52, 1.30, 1.44, 0.00, 3.52, 3.52],
    [0.00, 4.70, 4.92, 3.52, 0.00, 0.00],
    [0.00, 4.70, 4.92, 3.52, 0.00, 0.00]])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, M, t in zip(axes, [dyn, koop, sym],
                    ["dynamical (invariant) — clean blocks",
                     "Koopman (fitted) — noisy",
                     "symbolic (degenerate) — collapses"]):
    im = ax.imshow(M, cmap="viridis_r", vmin=0, vmax=6)
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=45)
    ax.set_yticks(range(6)); ax.set_yticklabels(names)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                    color="white" if M[i, j] > 3 else "black", fontsize=8)
    ax.set_title(t, fontsize=11)
    # mark the within-class 2x2 blocks
    for s in (0, 2, 4):
        ax.add_patch(plt.Rectangle((s - .5, s - .5), 2, 2, fill=False,
                                   edgecolor="red", lw=2))
fig.suptitle("Per-channel embedding distances (red = within-class pairs; "
             "want within < across)", fontsize=12)
fig.tight_layout()
fig.savefig("fig_channels.png", dpi=120)
print("wrote fig_channels.png")

# --- CL decision timeline (no-reg run) ---
steps = list(range(8))
true = ["LC", "R", "T", "LC", "R", "T", "R", "LC"]
dec = ["acc", "acc", "acc", "assim", "assim", "assim", "assim", "assim"]
nd = [np.nan, 3.46, 3.16, 1.96, 0.08, 1.69, 1.44, 1.35]
correct = [True, True, True, True, False, True, False, True]
tau = 2.05
fig, ax = plt.subplots(figsize=(9, 4))
for i in steps:
    c = "tab:green" if correct[i] else "tab:red"
    y = nd[i] if not np.isnan(nd[i]) else 0
    mk = "o" if dec[i] == "assim" else "s"
    ax.scatter(i, y, c=c, marker=mk, s=140, zorder=3,
               edgecolor="k", linewidth=0.6)
    ax.annotate(f"{true[i]}", (i, y), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)
ax.axhline(tau, ls="--", c="gray", label=f"τ = {tau} (assimilate below)")
ax.set_xlabel("stream step"); ax.set_ylabel("distance to nearest prototype")
ax.set_title("CL decisions  (○ assimilate, □ accommodate;  green=correct, "
             "red=wrong)  — 75% accuracy")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig("fig_decisions.png", dpi=120)
print("wrote fig_decisions.png")

# --- forgetting: modular (frozen) vs naive baseline before/after ---
classes = ["limit_cycle", "rossler", "torus"]
modular = [20.87, 9.27, 13.99]
naive_before = [15.97, 10.18, 12.67]
naive_after = [9.32, 16.71, 17.99]
x = np.arange(3); w = 0.25
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w, modular, w, label="modular SC-RNN (frozen module)", color="tab:blue")
ax.bar(x, naive_before, w, label="naive baseline — right after fit", color="tab:orange")
ax.bar(x + w, naive_after, w, label="naive baseline — end of stream", color="tab:red")
for i in range(3):
    ax.annotate("", xy=(i + w, naive_after[i]), xytext=(i, naive_before[i]),
                arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylabel("D_stsp on FIRST system of each class  (lower = better)")
ax.set_title("Forgetting: naive baseline drifts up on rossler & torus; "
             "modular modules are frozen (0 drift)")
ax.legend()
fig.tight_layout(); fig.savefig("fig_forgetting.png", dpi=120)
print("wrote fig_forgetting.png")
