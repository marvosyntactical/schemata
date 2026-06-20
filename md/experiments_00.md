# Experiments 00 — Familiarising with DSR on SSMs, and telling parametrisations apart

*Living planning doc. Scope: artificial data only; minimal from-scratch PLRNN as the SSM substrate.
Falsification discipline (primer §4.5): before running an experiment, write the single plot that would
falsify its claim. If you cannot, it is the wrong experiment.*

Grounding: this realises the "First experiments" of `primer.md` §4.5 with the §4.3 catalogue of
invariants as the comparison kernel. The headline question driving the design is:

> Given two parametrisations (two θ) of an SSM, are their realised dynamics **structurally the same
> kind of attractor, or completely different ones** — and how do I measure that?

---

## 0. The artificial-data testbed (fix once, reuse everywhere)

Three families, chosen so ground-truth topology is known and within/across-class is decisive.

**(a) Topologically distinct chaotic attractors — the "different schema" set.**
- **Lorenz-63** (σ=10, ρ=28, β=8/3): two-wing template, Z₂ symmetry, 3 fixed points = origin saddle
  + two unstable foci. Positive λ₁; Kaplan–Yorke dim ≈ 2.06.
- **Rössler** (a=b=0.2, c=5.7): single-fold funnel, Smale-horseshoe template, simpler symbolic
  dynamics. Positive λ₁; dim ≈ 2.01.
- Deliberately **similar on the surface** (both 3D, one positive Lyapunov exponent, fractal dim ≈ 2)
  but **inequivalent branched-manifold templates** (Birman–Williams). The ideal "lookalike but
  genuinely different" pair. Add a third (Chen / Thomas / Chua double-scroll) once the pair works.

**(b) Within-class / nuisance variants — the "same schema" set (invariance ground truth).**
Take Lorenz and apply *known* schema-preserving transforms 𝒢 (primer §4.2):
- affine coordinate change `z ↦ A z + b` (smooth conjugacy),
- nonlinear invertible observation warp `g` (homeomorphism),
- time rescaling (orbital equivalence — flows),
- partial observation (observe one coordinate, recover via Takens delay-embedding),
- additive observation noise.
By construction all are topologically conjugate to base Lorenz.

**(c) Surface-statistic-matched non-attractors — the falsifier set.**
A phase-randomised surrogate and/or a stochastic AR model tuned to match Lorenz's power spectrum and
1-step autocorrelation but with **no** chaotic attractor. Any metric that calls these "same as Lorenz"
is fooled by surface statistics.

Implementation: `solve_ivp` (RK45), discard transient, standardise, fix dt.

---

## 1. SSM substrate

Minimal from-scratch PLRNN (primer §3.2):
`z_t = A z_{t-1} + W φ(z_{t-1}) + h + ε_t`, `φ = ReLU`, A diagonal.
Observation: linear Gaussian decoder `x_t = B z_t (+ b)` for now.
Train by sparse teacher forcing / BPTT (primer §3.4). Analytic skeleton extraction (§3.2):
per-orthant fixed points `z* = (I − A − W D)⁻¹ h`, stability from `eig(A + W D)`.
(AL-RNN with symbolic itineraries adopted later, from E2, when the topology angle needs it.)

---

## 2. Experiment ladder

### E0 — Smallest closed loop (DSR mechanics)  ← build & run first
Train PLRNN on Lorenz; free-run generate; overlay generated vs true attractor; compute invariant
metrics (D_stsp, power-spectrum distance, max Lyapunov, Kaplan–Yorke dim). Vary teacher-forcing
interval τ to provoke the four failure modes (gradient explosion / TF-crutch / fixed-point collapse /
single-trajectory overfit). **Forecast accuracy is not the criterion** (primer §3.1).
- *Falsifier:* generated invariant density visibly misses a wing despite low 1-step error.

### E1 — Read the dynamical skeleton
On the trained PLRNN, compute analytic fixed points per orthant and their stability; count/type vs the
true system (Lorenz: saddle + 2 unstable foci). Connects a trained SSM to its interpretable skeleton.
- *Falsifier:* a reconstruction with good D_stsp but the wrong number/type of fixed points.

### E2 — Topologically distinct attractors side by side
Identical-architecture PLRNNs on Lorenz vs Rössler (vs a third). Compare reconstructed geometry,
fixed-point structure, Lyapunov spectrum, symbolic itinerary (AL-RNN), persistent homology (H₀/H₁) of
the attractor cloud. *Goal:* what does a topological difference look like in each invariant?
- *Falsifier:* no invariant separates Lorenz from Rössler.

### E2b — Nonlinearity dial
Sweep P (number of ReLU units, AL-RNN) → minimum P to reconstruct Rössler vs Lorenz. A cheap
per-attractor signature ("how much nonlinearity does each attractor cost").

### E3 — Within-class invariance (the 𝒢 dial)
Train PLRNNs on the Lorenz nuisance-variants. Which invariants stay put within class? Produce the
table **transform × invariant → preserved?** Expected texture: Lyapunov exponents not preserved under
time-rescaling (but entropy ordering is); persistent-homology bars survive homeomorphism but distort in
length; a Koopman/DSA distance is blind to the linear alignment it quotients but not to arbitrary
nonlinear warps unless the EDMD dictionary is rich enough.
- *Falsifier:* an invariant we expected to be 𝒢-invariant moves across within-class variants.

### E4 — The comparison kernel (discrimination)  ← the operational answer
Distance matrix over **all** trained models {Lorenz variants, Rössler variants, AR/surrogate fakes}
under each candidate metric (DSA, DFORM, D_stsp, power-spectrum dist, persistent-homology bottleneck,
Lyapunov-spectrum dist); cluster. Does clustering recover ground-truth classes? Which metric separates
within-class (small) from across-class (large) and which is fooled by surface-matched fakes?
- *Falsifier:* no metric's clustering matches the ground-truth equivalence classes.

### E5 — Boundary preview (accommodation)
Sweep Rössler c through period-doubling (1→2→4→chaos); train an SSM at each; watch invariants move;
locate the topological-type change (new H₁ loops, changing symbolic alphabet). A bifurcation as a
discontinuity in schema space — the accommodation trigger (primer §4.4, §4.5(3)).
- *Falsifier:* invariants vary smoothly across a known bifurcation (no detectable discontinuity).

---

## 3. Telling two parametrisations apart — methodology

**Negative result first: do NOT compare in parameter space θ.**
1. *Gauge redundancy.* The PLRNN/RNN has exact symmetries (permute latent units, sign flips, similarity
   transform of the linear part) that leave the realised dynamics conjugate — many θ, one dynamics, so
   ‖θ₁−θ₂‖ is meaningless.
2. *θ→dynamics is violently nonlinear.* Nearby θ can sit on opposite sides of a bifurcation
   (Eisenmann et al., loss jumps). Closeness in weights ≠ closeness in behaviour.

**Correct move: compare realised dynamics through Φ that factors through the quotient 𝒮/∼.**
"Same kind vs different" ⇔ same stratum? Cheap→expensive hierarchy (stop once separated):

| Level | Invariant | Catches | Blind to |
|---|---|---|---|
| Surface | D_stsp, power spectrum | wrong long-term statistics | **lookalikes pass** — necessary, not sufficient |
| Topological | fixed-point/cycle count & type (analytic for PLRNN), persistent homology H₀/H₁, topological entropy, symbolic dynamics | different attractor topology | smooth/timescale differences |
| Smooth | Lyapunov spectrum, Kaplan–Yorke dim | C^k-conjugacy differences (finer) | may over-split topologically-equal systems |
| Learned | DSA (Koopman-spectral metric, quotients a linear alignment); DFORM (learned diffeomorphism aligning vector fields; residual = distance from smoothly-conjugate) | spectral/orbital structure | their own implicit 𝒢 |

Decision rule:
- topological invariants match + DSA small → **same kind** (same topological type, plausibly conjugate).
- surface stats match but topological invariants differ → **lookalike** (different attractor, similar
  statistics) — the dangerous case; trust topology over D_stsp.
- topology matches but Lyapunov spectra differ → same type, different smooth/timescale structure
  (e.g. time-rescaled); whether 𝒢 calls this same is a *dial choice*, not an objective fact.

Caveats that bound what "tell apart" can mean (primer §7):
- **Identifiability ceiling:** from data we recover only up to conjugacy + invariant measure, never the
  unique vector field. Can't certify two θ are identical — only same/different class up to chosen 𝒢.
- **Bifurcations are ill-posed:** at a stratum boundary, membership is structurally unstable; the
  ambiguity is the feature (it is where accommodation lives), not a bug.

**First-day intuition pump (cheap, on top of E0):** linearly interpolate θ(s)=(1−s)θ₁+sθ₂ between two
trained models, free-run and plot the attractor at each s. You see bifurcations as the attractor
restructures and "gauge" stretches where nothing changes — the most convincing demonstration that
θ-distance ≠ dynamics-distance.

---

## 4. Tooling
- Data: `scipy.integrate.solve_ivp`.
- SSM: minimal PLRNN (this repo). AL-RNN / symbolic itineraries from `DurstewitzLab/ALRNN-DSR` adopted
  at E2.
- Metrics: D_stsp (binned/GMM KL), Welch power-spectrum distance, Lyapunov spectrum (QR method),
  Kaplan–Yorke. DSA via `mitchellostrow/DSA`; DFORM repo (needs trained vector fields, not raw data);
  TDA via `ripser`/`giotto-tda`.

## 4b. E0 results (run log)

Substrate: minimal clipped PLRNN, generalized teacher forcing (GTF), identity decoder
(observe first 3 latent dims), 15k train / 5k test, standardised.

**The four failure modes were all reproduced** (primer §3.4), which is the main pedagogical
payoff — and one metric bug was found and fixed in the process:

| Setting | Free-run outcome | Mode |
|---|---|---|
| `A=0.9`, `W~1/√M`, hard TF | NaN at init | forward/gradient explosion |
| contractive init, hard TF | → fixed point (λ≈0) | collapse / TF-crutch |
| GTF α=0.3, no clip, long train | diverged \|gen\|~1e29 | unbounded / over-expansive |
| GTF α anneal 0.5→0.1 | rides the clip walls | over-expansion (contained) |

*Metric bug:* `d_stsp` originally took its histogram range from both clouds, so a single
divergent outlier collapsed all mass into one bin and reported a fake-excellent 0.27. Fixed to
anchor the range on the **true** attractor and clip outliers in (divergence now scores worse).
A divergence flag (free-run leaving the data box by >10×) was added to `run_e0`.

**Clean success — Rössler (single-scroll), GTF α=0.15, clip=8, latent 30, 250 ep:**
- max Lyapunov **+0.0018**, spectrum `[+0.0018, ~0, −0.038]` — correct (+,0,−) chaotic-flow signature
- **Kaplan–Yorke dim 2.007** vs true ≈ 2.01 — essentially exact
- D_stsp 3.70, power-spectrum distance 0.27; figure `e0_rossler_alpha0.15.png` shows the chaotic band

**Key finding (feeds E2): topological complexity, not "chaos", is the axis of difficulty.**
A shallow PLRNN reconstructs **Rössler** cleanly (single fold, no lobe-switching) but persistently
fails on **Lorenz**: every Lorenz config gave either a thin one-lobe band (under-expansion,
D_stsp~10) or divergence — never the two-wing butterfly. Lorenz's bistable two-lobe topology is
what the shallow model cannot self-sustain. This is exactly why the lab uses dendritic/clipped
PLRNNs, and it is a concrete, ground-truthed instance of the project's thesis that *the
topological type is the hard, meaningful structure* — visible here as reconstruction difficulty
before we even reach the comparison metrics.

*Recipe that works:* contractive init (`A=0.8`, `W` std 0.05) + GTF α∈[0.15,0.3] + state clip ≫
attractor extent + grad clip. Anneal-to-weak-forcing overshoots into expansion; hard forcing
collapses. The usable α-window is narrow and system-dependent — itself worth noting.

## 5. Status
- [x] E0 — smallest closed loop: Rössler reconstructed (dim 2.01 ✓); Lorenz two-wing is the open
      difficulty (shallow-PLRNN limitation, motivates dendritic/clipped variants)
- [ ] E1 — dynamical skeleton
- [ ] E2 / E2b — distinct attractors; nonlinearity dial
- [ ] E3 — within-class invariance table
- [ ] E4 — comparison kernel / distance matrix
- [ ] E5 — bifurcation sweep
