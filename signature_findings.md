# Schema Signatures for Dynamical Systems: System, Method, and Findings

A self-contained account of the embedding we built to cluster trained dynamical-systems
models, how the comparison and weighting work, and what we learned running it on
artificial attractors and on real sleep-EEG. Written to be readable without prior
dynamical-systems background; terms are defined as they appear.

Code: [signatures.py](signatures.py) (the embedding), [alrnn.py](alrnn.py) (the model +
region machinery), [metrics.py](metrics.py) (Lyapunov), [validate_signatures.py](validate_signatures.py)
(artificial-attractor experiment), [neuro_sleep.py](neuro_sleep.py) (EEG experiment).
Conceptual background: [embedding.md](embedding.md).

---

## 0. The problem in one paragraph

We train a model to reproduce the *dynamics* of some time series — the rules by which the
system evolves, not just the values it took. We then want a fixed-length **coordinate vector
(a "signature")** for each trained model, such that two models implementing *structurally
similar* dynamics land near each other. With such a map, a stream of fitted models can be
**clustered**, and each cluster centre is a *schema*: a canonical dynamical motif (a fixed
point, an oscillation, a chaotic attractor). The hard part is choosing what "structurally
similar" means and computing it cheaply.

---

## 1. The substrate: what a dynamical system is, and the model we read from

A **dynamical system** here is a rule `z → F(z)` that advances a state `z` one step in time.
Iterating it from a starting point traces a **trajectory**. Key objects:

- **Fixed point**: a state that maps to itself, `F(z*) = z*`. The system sits still there.
  A fixed point is *stable* (an **attractor**: nearby states fall in), *unstable* (a
  **repeller**: they fly out), or a *saddle* (attracts along some directions, repels along
  others). The number of repelling directions is the **stability index**.
- **Limit cycle**: a closed loop the trajectory settles onto — a periodic oscillation.
- **Torus / quasiperiodic**: two incommensurate oscillations superimposed; the trajectory
  fills the surface of a doughnut and never exactly repeats.
- **Chaos**: bounded but never-repeating motion with *sensitive dependence* — nearby
  trajectories separate exponentially. The set the trajectory explores is a **strange
  attractor** (e.g. Lorenz, Rössler).
- **Attractor**: the set a trajectory settles onto after transients die out.

We model these with an **Almost-Linear RNN (AL-RNN)** (Brenner et al., NeurIPS 2024):

```
z_t = A z_{t-1} + W g(z_{t-1}) + h ,     g = ReLU on the first P units, identity on the rest
```

Only `P` of the `M` latent units pass through a ReLU nonlinearity. This makes the system
**piecewise-linear**: the `P` ReLU units each have an on/off boundary at `z = 0`, splitting
state space into **regions**. Inside any one region the map is a *single linear (affine) map*
`z → W_Ω z + h`, where `W_Ω = diag(A) + W·D_Ω` and `D_Ω` is the on/off pattern of the ReLU
units in that region. Two facts make this model ideal as a substrate:

1. **The local linearisation is exact and free.** In each region the Jacobian (the matrix
   that governs how perturbations grow) is literally `W_Ω` — no numerical estimation.
2. **It induces a symbolic code.** Label each region by its on/off pattern; a trajectory
   becomes a sequence of symbols, and the region-to-region transitions form a directed graph.

Trained AL-RNNs visit only a *handful* of regions (the paper reports 2–3 for Lorenz/Rössler),
which is what makes the exact enumerations below cheap. The new methods
`region_matrix(pattern)` and `enumerate_visited_regions(...)` in [alrnn.py](alrnn.py) expose
this machinery: one free-run collects every visited region with its exact `W_Ω`, the symbol
sequence, the transition counts, and the **clip-activation rate** (the model has a saturating
clamp for stability; when it engages, the clean piecewise-linear algebra below no longer holds,
so we measure it and flag it rather than trust it silently).

---

## 2. The signatures — one block per axis of the dynamics

Each block in [signatures.py](signatures.py) is a labelled descriptor, tagged by the kind of
**equivalence** it respects (this tag drives the weighting in §4):

- **`topological`** — invariants preserved under *topological conjugacy*: a continuous,
  invertible re-coordinatisation that may bend and stretch but not tear. Counts of fixed
  points, their stability indices, which periodic orbits exist, entropy, loop counts, the
  sign of chaos. These are the "shape of the dynamics" and define what we cluster by.
- **`rate`** — quantities that change under a smooth re-coordinatisation or a change of time
  step: eigenvalue magnitudes, oscillation frequencies, Lyapunov *magnitudes*, fractal
  dimension. Two systems doing "the same thing at a different speed" differ here but agree
  topologically. Carried on a separate, down-weighted axis.
- **`geometry`** — *data-side*, gauge-bearing descriptors read from the raw observations
  rather than the model (added later; see §6.2). Spectral and spatial structure.

Each block is either a **scalar vector** (fixed length) or a **cloud** (an unordered set of
eigenvalues whose size varies per model).

### 2.1 Equilibrium portrait — *where does the system stand still, and is it stable?*
In each visited region solve `z* = (I − W_Ω)⁻¹ h`. Keep it only if `z*` actually lies in that
region (sign pattern matches) — otherwise it is a *virtual* fixed point and discarded. For each
real fixed point, the eigenvalues of `W_Ω` give its stability index (how many directions repel).
**Signature:** number of fixed points + histogram of stability indices (topological) and the
pooled eigenvalues (rate). *(`equilibrium_portrait`)*

### 2.2 Periodic-orbit spectrum — *what cycles exist?*
A length-`k` symbol word corresponds to the composed map `M_w = W_{Ω_k}···W_{Ω_1}`. Its cycle
point `(I − M_w)⁻¹ b_w` is a real period-`k` orbit only if its `k` iterates each stay in the
prescribed regions. The eigenvalues of `M_w` are the **Floquet multipliers** (do orbits nearby
spiral in or out). We search only admissible walks in the transition graph, cap the length,
and skip non-primitive repeats (a fixed point is not also a "period-2 orbit"). **Signature:**
which periods exist + count per period (topological), pooled multipliers (rate).
*(`periodic_orbits`)*

### 2.3 Symbolic-graph spectrum — *how are the regions arranged?*
Build the directed transition graph on visited regions. From its adjacency matrix `B`:
- **Topological entropy** `log ρ(B)` (`ρ` = largest eigenvalue): the growth rate of the number
  of allowed symbol words — one scalar measuring dynamical complexity.
- **Closed-walk counts** `tr(Bᵏ)`: how many length-`k` symbolic loops exist.
- **Normalised Laplacian spectrum**: a size-comparable fingerprint of the graph's connectivity
  (eigenvalues in `[0,2]`, so a 3-region and a 7-region model are still comparable).
- **Number of strongly connected components.**
This is the principled replacement for "how many regions" (a single integer that says nothing
about arrangement). All topological. *(`symbolic_graph`)*

### 2.4 Generator geometry — *how are the linear pieces glued together?*
Every region matrix is the shared backbone `A` plus a rank-one update per active ReLU unit. So
the whole arrangement is encoded by `A` and the `P` columns of `W` for the nonlinear units.
**Signature:** eigenvalues of the backbone (rate), the change in spectral radius when each
switch turns on (rate), and the principal angle between each switch direction and the backbone's
dominant subspace (topological — how the switch aligns with the flow). Distinguishes two models
with the same transition graph that bend the vector field differently. *(`generator_geometry`)*

### 2.5 Lyapunov signature — *chaotic, oscillatory, or decaying?*
The **Lyapunov spectrum** measures the average exponential rate at which nearby trajectories
separate along each direction, computed by a QR iteration along a free-run using the exact
region Jacobians. The **signs** are near-topological: one positive exponent = chaos; a zero
exponent = a sustained oscillation direction; all negative = collapse to a point. The
**magnitudes** are rate. The **Kaplan–Yorke dimension** estimates the attractor's fractal
dimension from the spectrum. **Signature:** sign counts (topological); spectrum + dimension
(rate). Reuses [metrics.py](metrics.py). *(`lyapunov_signature`)*

### 2.6 Attractor topology — *what shape is the attractor?*
**Persistent homology** of the free-run point cloud yields **Betti numbers**: `β₀` = connected
components, `β₁` = loops, `β₂` = enclosed voids. A point attractor shows `β₀` only; a limit
cycle adds one loop; a torus has two loops; a chaotic branched manifold has its own signature.
This is the geometric shape the spectral methods are blind to. Optional dependency (`ripser`);
we cap at loops (`β₀,β₁`) because voids make the computation memory-explosive. *(`attractor_topology`)*

### 2.7 Spectral block — *the DSA-style coordinate*
The pooled eigenvalues of all visited region Jacobians: a single global "linear operator"
descriptor, like Dynamical Similarity Analysis but kept as *one block among many* rather than
the whole metric. Rate. *(`region_spectrum`)*

---

## 3. How two models are compared, exactly

A model's signature is a dict of blocks. Comparing models is done **per block, then combined** —
not by gluing everything into one flat vector, because the blocks live in incompatible spaces.

1. **Per-block distance** (`block_distance_matrix`):
   - *scalar* blocks: stack across all models, **z-score each coordinate** (so a count and an
     entropy are on the same scale), then Euclidean distance.
   - *cloud* blocks (eigenvalue sets of varying size): **2-Wasserstein distance** between the
     two clouds. Each eigenvalue is represented by `(|λ|, |angle(λ)|)` — magnitude and
     reflection-folded phase — and we sum the 1-D optimal-transport distance over each. This is
     exactly the spectral notion DSA reduces to for normal operators, and it handles clouds of
     different sizes (models of different dimension).
2. **Commensuration** (Layer 3): divide each block's distance matrix by its own median, so a
   block measured in "Wasserstein units" and one in "z-scored Euclidean units" contribute
   comparably.
3. **Weighted combine** (`combine_distance`): `D(i,j) = sqrt( Σ_b w_b · (d_b(i,j)/scale_b)² )`.
4. **Cluster** the resulting `K×K` distance matrix with any distance-matrix method —
   hierarchical (`average`, `ward`, `complete` linkage), HDBSCAN, spectral. `cluster()` wraps
   SciPy hierarchical clustering.

Cost: `O(K)` to featurise plus `O(K²)` closed-form distances — **no per-pair optimisation**,
unlike DSA which runs a manifold optimisation for every pair.

---

## 4. The weighting — three layers, and why each exists

> The central decision: **the choice of weights is the choice of equivalence relation.** There
> is no neutral weighting to discover. If you weight rate features heavily you cluster by
> "same speed"; if you weight topological features you cluster by "same shape". This is a
> modelling choice, made explicit, not a hyperparameter to tune away.

**Layer 1 — class weight (the decision).** Each block's `class` tag sets its base weight:
`topological → 1`, `rate → gamma`, `geometry → geom`. So `gamma=0` clusters purely by
topology; `gamma≈1` recovers a DSA-like rate-sensitive metric; `geom=0` ignores the data-side
channel. This is the knob that says what "similar" means.

**Layer 2 — reliability (unsupervised, learned).** `estimate_reliability(groups)` takes
*replicate groups* — several signatures of the *same* target (retrainings, seeds, or within-
class variants) — and weights each block by how *stable* it is:
`w_b ≈ median_all(d_b) / (median_within(d_b) + reg·median_all(d_b))`.
A block with small within-group scatter relative to the spread across targets is reliable and
up-weighted; a block that is **constant everywhere** (carries no information) gets weight 0
automatically. This replaces hand-tuned weights: in the artificial-attractor run it up-weighted
the Lyapunov-sign block 6× and zeroed the degenerate orbit-count blocks with no manual input.
Its limit (found on EEG): it measures *local* discriminability, so it cannot tell a
"coarse-but-discriminative" block from a "sharp-and-discriminative" one — see §6.3.

**Layer 3 — commensuration (mechanical).** The per-block median normalisation above, so Layers
1–2 weights mean what they say.

### Knobs to turn

| knob | where | effect |
|---|---|---|
| `gamma` | `combine_distance` | weight on **rate** blocks. 0 = topological clustering, ~1 = DSA-like. |
| `geom` | `combine_distance` | weight on the **data-side geometry** channel. 0 = model-side only. |
| `reliability` | `combine_distance` / `estimate_reliability` | per-block auto-weights from replicate groups; down-weights noisy/degenerate blocks. |
| `weights` | `combine_distance` | explicit `{block: w}` override of Layers 1–2 (e.g. one block only). |
| `P` | `ALRNN` | number of nonlinear units. `P=M` makes it a full PLRNN (max capacity). |
| `geom_per_channel` | `extract` | include the gauge-bearing per-channel spectral block (for real sensor data). |
| linkage / `n_clusters` | `cluster` | clustering algorithm and granularity (`ward` was markedly better on EEG). |
| `k_max` | `extract` | maximum period searched in the orbit block. |

---

## 5. Findings I — artificial attractors

**Setup** ([validate_signatures.py](validate_signatures.py)): three reconstructable classes on
a topological ladder — **limit cycle** (one loop, one zero Lyapunov exponent), **torus**
(quasiperiodic, two zero exponents, zero entropy), **Rössler** (chaotic, one positive exponent).
Each appears as 3 **affine variants** (a random rotation+scale+translation+noise = the *same*
dynamics in a different coordinate frame). A good embedding puts variants close and classes far.

**Result.** Where reconstruction succeeded the embedding works as designed:
- The three **torus** variants clustered tightly (pairwise distance **0.27–0.35**), well
  separated from the limit cycle (~1.0). On the faithfully-reconstructed subset the
  across-class / within-class separation ratio was **1.67** at `gamma=0`.
- **Layer-2 reliability behaved correctly**: it up-weighted `lyap_signs` (within/across distance
  ratio 0.51 — the cleanest discriminator) to 6× and auto-zeroed the degenerate orbit blocks.
- The `gamma` knob behaved as predicted: on clean models topological-only separated best;
  on the noisier full set the rate channel rescued discrimination (ARI 0.23 → 0.48 as `gamma`
  rose to 1).

**The ceiling — and an independent confirmation.** 4 of 9 fits failed: two limit-cycle variants
railed into the clip, one Rössler collapsed to a fixed point, and — the telling one — the
"successfully" trained Rösslers showed Lyapunov signs `[0 positive, 1 zero, 15 negative]`: they
reconstructed Rössler as **quasiperiodic, not chaotic**, and so the embedding correctly placed
them *next to the torus cluster*. This reproduces the project's known result that chaos is
misread as quasiperiodicity, and pins the cause: **the embedding is faithful — it clusters by
the dynamics actually present in the trained model, and the errors are upstream reconstruction
failures, not embedding blind spots.** Reading `clip_rate`, `n_regions`, and `lyap_signs` tells
you which models to trust.

---

## 6. Findings II — real EEG (sleep staging)

### 6.1 Primer on the data
**Sleep-EDF** (PhysioNet) is overnight **polysomnography**: continuous recordings of a sleeping
person's brain and body signals, with an expert **hypnogram** labelling every 30-second epoch
with a **sleep stage**. We used 2 EEG channels (Fpz-Cz, Pz-Oz) + 1 EOG (eye movement) at 100 Hz,
from 4 subjects, balanced to **9 epochs each of 4 stages**:
- **W (wake)** — fast, low-amplitude activity (alpha/beta rhythms).
- **N2** — light sleep, marked by *sleep spindles* (~12–16 Hz bursts).
- **N3** — deep slow-wave sleep, dominated by large *delta* waves (0.5–4 Hz).
- **REM** — dreaming sleep; low-amplitude mixed-frequency EEG resembling wake, plus eye
  movements.

These stages differ in genuine **dynamical regime** (slow-wave vs spindle vs fast), which makes
sleep the *fair* test for a dynamics-based signature — unlike motor-imagery EEG, whose
discriminative information is spatial/lateralised. The label is the stage; we ask whether our
clusters agree with it. We score with two numbers:
- **ARI (Adjusted Rand Index)**: agreement between our *unsupervised* clusters and the labels,
  chance-corrected. 0 ≈ chance, 1 = perfect.
- **LOO-3NN (leave-one-out 3-nearest-neighbour accuracy)**: a *supervised* ceiling — hold out
  each epoch, predict its stage from its 3 nearest neighbours in the signature distance. Tests
  whether the geometry is class-discriminative *at all*. Majority-class baseline = 0.25.

### 6.2 The arc of results

| signature | ARI | LOO-3NN | reading |
|---|---|---|---|
| **model-side only** (the attractor signature) | ≈ 0.00 | 0.19–0.25 | **chance** — carries no stage signal |
| model-side, **P=16** (full-PLRNN capacity) | 0.064 | 0.25 | still chance — capacity is not the fix |
| **+ geometry channel** (rotation-invariant) | 0.08 | 0.78 | discriminable locally, clusters poorly |
| **geometry, per-channel** (gauge-bearing) + Ward | **0.55** | **0.86** | **beats the reference** |
| band-power (reference) | 0.46 | 0.83 | standard sleep-staging feature |
| DMD/Koopman (cheap temporal) | 0.02 | 0.39 | weak |

Three things happened, in order:

1. **The model-side attractor signature is at chance.** The fits collapse: most epochs give
   `n_regions=1, clip=1.00`. A 30-second window of stochastic, high-dimensional EEG is *not* a
   low-dimensional deterministic attractor, so reconstruction has nothing to lock onto, and
   every collapsed model yields the same degenerate signature.

2. **More capacity does not fix it.** Raising `P` from 3 to 16 (a full piecewise-linear PLRNN)
   let some models populate 16–18 regions — so the parsimony bottleneck is real — but the
   signature stayed at chance (LOO 0.25). The binding constraint is *reconstruction collapse +
   modality mismatch*, not the architecture's information bottleneck. More nonlinear units
   cannot manufacture a low-dimensional attractor where the signal is spectral.

3. **A data-side geometry channel recovers the signal.** We added blocks read directly from the
   raw observations rather than the model: `spectral_signature` (the oscillatory content — how
   much delta vs fast power, rotation-invariant by using the cross-spectral trace),
   `spatial_signature` (the covariance spectrum / connectome axis), and — crucially —
   `spectral_signature_perchannel` (per-channel band power, *gauge-bearing*). The
   rotation-invariant version separates locally (LOO 0.78) but clusters poorly (ARI 0.08); the
   **per-channel** version, with **Ward** clustering, reaches **ARI 0.55, LOO 0.86 — beating the
   band-power reference**. Rotation-invariance, essential for the gauge-free synthetic case, was
   exactly what crippled clustering on real sensor data where *which* electrode carries the
   slow waves is itself informative.

### 6.3 A weighting limitation surfaced
Naively concatenating all geometry blocks with equal weight *dilutes* the sharp per-channel
block back to ARI 0.08, and Layer-2 reliability only partially recovers it (→ 0.18). Reliability
weights by *within- vs across-class distance* (local discriminability), which the coarse pooled
blocks also have; it cannot see that the per-channel block yields **cleaner global clusters**.
So Layer 2 handles *noisy/degenerate* blocks but not *coarse-dilutes-sharp*. The practical fix:
use the per-channel block as the primary geometry descriptor when channels are meaningful, or
add a cluster-quality / Fisher-based weighting when labels are available.

---

## 7. Takeaways

1. **The embedding is faithful and works when reconstruction works.** On clean attractors it
   clusters variants of one system tightly and separates topological classes, with the
   weighting layers behaving as designed. Its errors trace to reconstruction, not to the
   signature — and the diagnostics (`clip_rate`, `n_regions`, `lyap_signs`) tell you which.
2. **Reconstruction fidelity is the ceiling, and it is modality-dependent.** Clean
   low-dimensional deterministic systems reconstruct; 30 s of stochastic EEG does not. Capacity
   (`P`) is not the lever — it is whether the data *is* a low-dimensional attractor.
3. **On real neural data the signal is spatial/spectral, not temporal-dynamical.** This
   confirms and extends the project's central thesis with the full pipeline on a new labelled
   dataset whose stages genuinely differ dynamically: the attractor/topology channels are at
   chance, while a gauge-bearing geometry channel matches and beats the standard baseline.
4. **Weighting is a modelling decision and it is doing real work.** `gamma`, `geom`, and the
   reliability layer visibly change what clusters form. Reliability automates the obvious cases
   (degenerate, noisy) but not cluster-shape quality.
5. **Gauge-freedom is regime-dependent.** Rotation-invariance is right for synthetic affine
   variants and wrong for real sensors with meaningful channels — the same knob (`geom_per_channel`)
   has to be set per regime.

---

## 8. Where to go next

- **Robustness / honest ARI on EEG.** Scale beyond 36 epochs / 4 subjects with
  *subject-stratified* cross-validation, to rule out subject-identity leakage in the LOO number
  and stabilise the ARI estimate.
- **Cluster-quality-aware block weighting.** Replace or augment Layer-2 reliability with a
  weighting that rewards blocks producing clean partitions (silhouette- or Fisher-based when
  labels exist), to fix the coarse-dilutes-sharp problem in §6.3.
- **Input-driven reconstruction for real data.** The collapse on EEG argues for the lab's
  stimulus-driven PLRNN (modelling external input rather than pretending the dynamics are
  autonomous) before expecting model-side channels to carry signal.
- **Make the geometry channel first-class.** Promote spectral/spatial blocks (and connectivity
  for many-channel data) to a standard, separately-weighted part of the signature — the
  data-side channel is where the real-data signal lives.
- **Persistent homology at full strength.** `β₂` (voids) was disabled for memory; on hardware
  that allows it, the full Betti signature sharpens the attractor-shape block on synthetic data.

---

### References
- Brenner, Hemmer, Monfared, Durstewitz. *Almost-Linear RNNs Yield Highly Interpretable Symbolic
  Codes in Dynamical Systems Reconstruction.* NeurIPS 2024. arXiv:2410.14240
- Ostrow, Eisen, Kozachkov, Fiete. *Beyond Geometry: Dynamical Similarity Analysis.* NeurIPS 2023.
  arXiv:2306.10168
- Finn et al. *Functional connectome fingerprinting.* Nat. Neurosci. 2015.
- Sleep-EDF Database, PhysioNet (sleep-cassette study).
- See also [embedding.md](embedding.md) (signature menu), [summary.md](summary.md) (project
  context), [dsa_native_ssm.md](dsa_native_ssm.md) (gauge / comparison theory).
