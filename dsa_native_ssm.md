# Building the DSA mechanism into an SSM: requirements, gaps, and engineering

*Working note. Goal: figure out whether and how an SSM's own latent dynamics can BE a
Dynamical-Similarity-Analysis operator, rather than DSA being a post-hoc tool applied to a trained
model. Two questions throughout: (1) how to make the latent operator a faithful Koopman operator;
(2) how to handle gauge. Approximations and open problems flagged inline.*

---

## 0. The thing to get right first: there are two gauges, at two different places

DSA quotients a **data/observation gauge** — the arbitrary basis in which a *fixed* set of
trajectories is coordinatized (which channels, which PCA axes, which delay stack). DSA cannot choose
this basis; it is handed the data, so it must remove the basis freedom *after the fact*.

An SSM that represents dynamics in a learned latent has a **latent/parametrization gauge**: for any
invertible `T`, the maps `(E, K, D) -> (TE, T K T⁻¹, D T⁻¹)` realize identical input–output behaviour.
This is intrinsic to representing a flow in *learned coordinates* — not borrowed from anywhere.

The bridge — and the reason "build DSA in" is not just "run DSA on the weights":

> The encoder *chooses* the SSM's latent basis, i.e. it chooses an internal measurement basis for the
> dynamics. So the SSM's latent gauge is the **controllable** analogue of DSA's **uncontrollable**
> data gauge. Building DSA into the model converts a *quotient-after-the-fact* problem (DSA, fixed
> data basis) into a *gauge-fix-by-construction* problem (SSM, chosen latent basis).

Both gauges still exist and must be handled at the right stage:

- **outer (data) gauge** — different recordings / rotated sensors feeding the SSM → handle by training
  the *encoder* to be invariant (augmentation / contrastive).
- **inner (latent) gauge** — `T K T⁻¹` in the model's own coordinates → handle by *canonicalizing the
  operator* (whitening + Procrustes, or a canonical form).

DSA collapses both into one post-hoc orthogonal Procrustes. An SSM can separate them and handle each
where it is cheapest. Keep these two levels distinct for the rest of the note.

---

## 1. What the DSA *mechanism* requires (abstractly)

Strip DSA to its function: assign to a system a **comparable dynamical signature**. That needs three
things, in order.

1. **A linear representation of the dynamics on a fixed observable space.** A finite observable map
   `φ: state → ℝⁿ` and a linear operator `K` with `φ(x_{t+1}) ≈ K φ(x_t)`. The content of "≈" is
   *Koopman-invariance*: `K` must map `span(φ)` into itself over a *multi-step* horizon, not just
   one step. (One-step least squares is DMD; multi-step faithfulness is what makes `K` a Koopman
   approximation rather than a regression.)

2. **A canonical or alignable form**, so two operators can be compared at all. Either align them
   (Procrustes) or express them in a normal form.

3. **The right gauge group** — the one that quotients exactly the representational freedom (basis in
   `φ`-space) and nothing dynamical. The *choice of group is the definition of "same dynamics."*

That is the whole mechanism: **(lift to linear) → (canonicalize/align) → compare**, with the group
choice fixing the equivalence relation.

---

## 2. How DSA achieves each requirement (and the cost of each choice)

| Requirement | DSA's concrete choice | What it buys / its limit |
|---|---|---|
| 1. linear representation | **delay embedding** (Takens lift) + **reduced-rank DMD** (PCA to `r` dims, then one-step least-squares `K`) | computable from raw data; but one-step LS only — no explicit multi-step invariance objective, so `K` faithful only to the extent the delay+PCA subspace happens to be Koopman-invariant. Chaos has *continuous* Koopman spectrum → no finite invariant subspace → irreducible residual. |
| 2. canonical/alignable | **orthogonal Procrustes over vector fields**: `d(A,B)=min_{C∈O(n)} ‖CAC^T−B‖_F` | gives a genuine *metric* (O(n) is compact, norm-preserving). Sensitive to eigenvector geometry / non-normality, not just spectrum — finer than "compare eigenvalues." |
| 3. gauge group | **orthogonal group O(n)** (after normalization/whitening of the observables) | O(n) similarity preserves eigenvalues *and* is rigid, so the residual is a metric. Quotients only rigid rotation of the observable basis. |

Precise statement of DSA's equivalence relation: two systems are "the same" iff their finite Koopman
approximations are **orthogonally similar**. This is:
- *finer* than "same Koopman spectrum" (it also pins eigenvector arrangement up to a global rotation);
- *coarser* than equality of operators;
- and only a **surrogate** for conjugacy. If `F,G` are conjugate via `h` (`h∘F=G∘h`), their Koopman
  operators are related by the composition operator `C_h: g ↦ g∘h`. For *linear* `h`, `C_h` is exactly
  an (orthogonal, after whitening) similarity → DSA sees it as distance 0. For *nonlinear* `h`, `C_h`
  is an infinite-dim non-orthogonal operator → DSA only approximately quotients it. **So DSA exactly
  captures conjugacies that act linearly on its chosen observables, and approximates the rest.** This
  is the precise content of "DSA is a linear surrogate" (primer §4.3 row 4).

---

## 3. What an SSM already provides

Two SSM senses (primer §3.1); both already contain most of the mechanism.

**Structured linear SSM (S4 / Mamba / linear RNN):** `h_t = Ā h_{t-1} + B̄ u_t`, `y_t = C h_t`. The
latent recurrence **is** a linear operator `Ā` on a learned latent — i.e. requirement 1's `K`,
*already there by construction*. Encoder = input map; decoder = `C`. This is structurally a Koopman
model; what it is not is *autonomous* or *trained to be faithful* to a data-generating flow.

**PLRNN / AL-RNN (statistical SSM):** latent is piecewise-linear; the linear block (`A`, plus `W` on
the linear units of an AL-RNN) is a per-region linear operator. In the near-linear regime (small `P`)
it approximates a single Koopman operator. The ReLU units are exactly a *nonlinear residual* on top of
a linear core.

Mapping SSM components to the DSA requirements:

| DSA needs | SSM already has |
|---|---|
| observable lift `φ` | the **encoder / recurrent latent** (recurrence ⇒ implicit delay embedding ⇒ Takens; handles partial observation) |
| linear operator `K` | the **latent recurrence operator** — literally linear for S4/Mamba; near-linear core for AL-RNN |
| back to observables | the **decoder** — gives the autoencoder loop a Koopman-AE needs |
| multi-step fitting | **BPTT / teacher forcing** infrastructure — reusable for a multi-step linearity loss |

So: an SSM is *architecturally already a Koopman autoencoder.* It has lift + latent operator +
decoder + multi-step training. It is roughly two-thirds of the DSA mechanism, sitting unused.

---

## 4. What an SSM lacks (the gap to DSA)

1. **No enforced Koopman-faithfulness.** Nothing in standard training says "the latent operator, run
   *autonomously*, reproduces the system's dynamics linearly over long horizons." S4/Mamba's latent is
   linear but *input-driven* and optimized for sequence prediction; a PLRNN's latent is PWL, not one
   operator. Missing: a multi-step *latent-linearity consistency* objective.

2. **No defined gauge / no canonical form.** Training lands `K` in a random latent basis (`T K T⁻¹` for
   arbitrary `T`). Two SSMs fit to the *same* system have incomparable weight matrices. Missing: either
   whitening-to-fix-the-group-to-O(n), or a canonical normal form.

3. **No comparison in the objective.** DSA is a metric; an SSM emits no signature and is not trained so
   that its operator is *discriminative* across systems / *invariant* within a class. Missing: a
   comparison-aware loss if we want the operator to be a usable schema code (not just faithful).

4. **No Koopman-spectrum target / no continuous-spectrum handling.** Standard training does not target
   `eig(K)`; for chaos there is no exact finite `K`. (This same gap is what produced the *unbounded
   free-run divergence* in E0 — an unconstrained latent operator can have `|λ|>1`.) Missing: spectral
   constraint + an explicit nonlinear/forcing residual.

---

## 5. Engineering it in

### 5.A Make the latent operator a faithful Koopman operator

Architecture (a Koopman autoencoder, in the lineage of Lusch et al. 2018; Azencot et al. consistent
Koopman AE; Brunton et al. Koopman review): `x_t —E→ φ_t ∈ ℝⁿ —K→ φ_{t+1} —D→ x̂_t`.

Objectives (beyond reconstruction):

1. **Multi-step latent-linearity consistency** — the load-bearing term:
   `L_lin = Σ_{t,m} ‖ E(x_{t+m}) − Kᵐ E(x_t) ‖²` for horizons `m = 1…H`.
   `m=1` is DMD; `H>1` is what forces the encoder to find an (approximately) **Koopman-invariant**
   subspace. This is the single thing that turns a learned latent into a faithful operator.

2. **Decode + prediction consistency:** `D(φ_t)≈x_t` and `D(Kᵐ φ_t)≈x_{t+m}`.

3. **Backward consistency (Azencot):** also learn `K_back` with `K K_back ≈ I`. Improves conditioning
   and stability of `K` — directly relevant to the boundedness failure.

4. **Spectral constraint for the right regime — and it fixes E0's divergence.** Constrain `eig(K)` to
   live on/inside the unit circle. A Koopman operator with `|λ|≤1` *cannot* blow up, so this is a
   *principled* cure for the unbounded free-run we kept hitting, not a clamp. Implementations:
   parametrize `K` as near-unitary (`K = exp(S)` with `S` skew-symmetric → `|λ|=1`, marginal
   stability appropriate for a sustained attractor), or a stable Schur parametrization with bounded
   diagonal. (This couples requirement 1 and the canonical-form choice in 5.B.)

5. **Handle chaos / continuous spectrum — the almost-linear move.** No finite `K` represents chaos
   exactly. Two options, both *measurable*:
   - **Nonlinear residual:** `φ_{t+1} = K φ_t + r(φ_t)` with `r` small and sparse — *this is exactly the
     AL-RNN with small P.* The `P` ReLU units carry the part of the dynamics outside every finite
     Koopman-invariant subspace.
   - **HAVOK forcing:** augment with an intermittent forcing channel `v_t` (a trailing delay coordinate
     acting as input on the linear core), modelling the continuous-spectrum part as forcing.
   Either way, **the size of the required residual / forcing is a quantity, not a nuisance:** it
   measures how far the system is from DSA's linear assumption. The AL-RNN `P`-dial *is* this measure
   — "how much of the dynamics refuses to linearize" = "how much DSA structurally cannot see."

Note on autonomy: for attractor reconstruction we run the SSM **autonomously** (free-running), so we
want the *autonomous* operator. S4/Mamba's input-driven form is fine for *forced* systems but for DSR
we use the generative mode DSR already uses.

### 5.B Handle gauge

Recall the two levels (§0). Inner (latent) gauge first, since it is the one that blocks comparison.

**Inner gauge — three handles, choose by what you want out:**

- **(B1) Whiten + Procrustes (gives a metric).** Constrain the observables to be whitened
  (`Cov(φ)=I`, via a loss or a normalization layer). *This is the engineering step that makes O(n)
  the correct group:* with an un-whitened latent the residual gauge is full `GL(n)` and
  orthogonal Procrustes is too tight; after whitening the only remaining freedom is `O(n)`, exactly
  DSA's group, and `d(K₁,K₂)=min_{C∈O(n)}‖CK₁C^T−K₂‖` is the right distance. This is "run DSA on the
  model's own operators" — but the operators are now clean, low-dim, and faithful (because of 5.A),
  so the comparison is far better-conditioned than DMD-on-noisy-data.

- **(B2) Canonical form (gives a closed-form code, no alignment).** Parametrize `K` directly in a
  normal form so there is *no continuous gauge left*:
  - **Modal / block-diagonal form:** `K = blkdiag` of 2×2 rotation–scaling blocks (one per complex
    eigen-pair: a frequency + a decay) plus 1×1 real blocks. The schema becomes a *list of modes* +
    couplings — maximally interpretable. Canonical up to block ordering and rotation within degenerate
    blocks.
  - **Ordered real Schur form:** quasi-upper-triangular, eigenvalue blocks ordered by frequency /
    magnitude. Residual gauge is discrete (signs / ties). Comparison is then *direct* — no Procrustes.
  - **Diagonal (eigenbasis):** coarsest; equals "compare eigenvalues," discards non-normal coupling.
  Cost: ordered-spectral parametrizations are numerically delicate at **eigenvalue crossings /
  degeneracies** (continuity of the form breaks where `λ_j → λ_k`). A generic-position assumption is
  usually fine; aggressive schedules or symmetric systems can violate it.

- **(B3) Train for invariance (learned, for unknown/nonlinear gauges).** Augment by applying known
  coordinate changes (the §4.2 𝒢) to the *input* and require the resulting operators to be equal under
  the chosen quotient (or a derived permutation-invariant readout — e.g. sorted eigenvalues — to be
  identical). Use this for the **outer (data) gauge**, where the nuisance is in how the data was
  measured and may be nonlinear, so neither whitening nor a latent canonical form reaches it.

**Recommended split:** outer/data gauge → encoder invariance (B3); inner/latent gauge → whitening +
Procrustes (B1) *or* canonical modal form (B2). B1 if you want a metric (clustering, the §4.4
assimilate/accommodate threshold); B2 if you want an explicit, human-readable schema code.

### 5.C The resulting object, in one line

> **Encoder (→ whitened observables) + near-unitary linear core `K` in modal form + small AL-RNN
> nonlinear residual + decoder, trained with multi-step latent-linearity consistency, backward
> consistency, and a spectral constraint.**

That is a trainable, faithful, gauge-controlled DSA operator: the model's *own* latent dynamics are
the comparison object, the data-gauge is absorbed by the encoder, the latent-gauge is fixed by
whitening/modal form, the chaotic part is quarantined into a measured residual `P`, and the spectral
constraint simultaneously buys boundedness.

---

## 6. Honest caveats and open problems

- **Inherited blind spot.** A DSA-native SSM is still a *linear* surrogate: it can confuse
  topologically distinct systems that share Koopman-spectral structure (our E0 Lorenz/Rössler vs the
  surface-matched AR falsifier). Building DSA in does not fix this — it *localizes* it. Topological /
  symbolic / TDA invariants must run alongside to catch what the linear core cannot. This is exactly
  what E4 is meant to measure.
- **Does multi-step consistency actually find an invariant subspace for chaos, or just overfit a
  horizon `H`?** Open. The honest read: it finds the best `H`-step linear subspace; everything beyond
  it is forced into the residual `P`. Whether the split (linear core vs `P`) is *identifiable* and
  *stable across seeds* is itself an experiment (and the seed-stability falsifier in EK0).
- **Whitening vs faithfulness can fight.** Forcing `Cov(φ)=I` constrains the encoder and may trade off
  against linearity consistency. Needs an ablation; the right weighting is not obvious.
- **Modal-form degeneracies.** Eigenvalue crossings during training (§5.B) can destabilize the
  canonical parametrization — a real numerical risk for systems with symmetry-induced repeated
  eigenvalues (e.g. Lorenz's Z₂).
- **Identifiability ceiling stands (primer §7).** Even done perfectly we recover the dynamics only up
  to the chosen quotient + invariant measure — never the unique vector field. The win is that we get to
  *choose and engineer* the quotient (the gauge group), rather than inherit whatever DSA's whitening
  happened to impose.

---

## 7. Minimal experiment hook (EK0)

On the working E0 Rössler model: read off the latent operator (or train the small Koopman-AE above),
then test the three things this note claims an SSM lacks → can be engineered in:
1. **Faithfulness:** does `spec(K)` match an EDMD/HAVOK Koopman spectrum from raw Rössler data?
2. **Gauge-stability:** two seeds → is `d_O(n)(K₁,K₂) ≈ 0` after whitening + Procrustes (B1)? If not,
   the operator is not a stable signature and we need the consistency loss (5.A) / canonical form
   (B2) before any comparison is meaningful. *(Falsifier for the whole readout idea.)*
3. **Discrimination:** `d(Rössler, Lorenz) ≫ d(Rössler, Rössler′)`, and where does the AR-surrogate
   lookalike land — inside or outside the Rössler cluster? (The linear-surrogate blind-spot test.)

---

## 8. Shared-reader schema memory for *continual* DSR (the build)

The objective is **continual learning, Setting A** (primer §4.1): a stream of systems arrives; we fit
**one** model sequentially and, *at training time*, decide whether each new system **reuses** the
structure of a known schema (assimilate) or needs **new** structure (accommodate). The schema memory
is not a post-hoc analysis — it is the controller that routes training.

### 8.1 Architecture (modular shared-reader)

- **Shared across all systems:** the **embedding extractor** Φ (gauge-free, multi-channel; §1) and the
  **attention controller / prototype memory** `{c_k}`. These are the "shared reader": one common code
  space every system is mapped into.
- **Reused per known schema:** a **dynamics module** `Mₖ` (a small AL-RNN: ReLU on `P` units → linear
  core + symbolic vocabulary). Assimilation = route to an existing `Mₖ`.
- **Per system (within-class chart):** a small **affine chart** `(B_s, b_s)` mapping the system's
  observation frame ↔ the module's latent. This absorbs the coordinate-change nuisance (the §4.2 𝒢),
  so two affine variants of one attractor share a module and differ only in their chart.

Why modular and not one big shared backbone: a single `(A,W)` cannot *be* both Rössler-dynamics and
limit-cycle-dynamics, and module isolation gives **no-forgetting by construction** — the cleanest way
to make the assimilate/accommodate decision *structural* (route vs spawn) rather than a soft prior. The
single-shared-vocabulary backbone (DynaMix-like) is the harder Setting-C horizon; not first.

### 8.2 The CL decision, at training time (probe → commit)

For each incoming system `Dₛ`:
1. **Probe:** spawn a provisional module, short-fit it (few epochs, GTF) → read its embedding
   `Φ(Dₛ)` (Koopman spectrum ⊕ symbolic-graph invariants ⊕ Lyapunov/dim features; all gauge-free).
2. **Retrieve:** attention over prototypes, `a_k = softmax(β · sim(Φ(Dₛ), c_k))`.
3. **Decide:**
   - `max_k a_k ≥ τ` → **assimilate** to `k* = argmax`. Discard the probe; **reuse module `Mₖ*`**,
     fine-tune only the within-class chart `(B_s,b_s)` (cheap, few params). Update `c_{k*}` (running mean).
   - else → **accommodate**: finish-train the probe into a new module `M_{K+1}`; add prototype `c_{K+1}`.
4. **Record** the decision against the ground-truth schema label.

`β` (attention temperature) and `τ` are the boundary-sharpness knobs (the §β↔τ correspondence): `β→∞`
recovers the hard `argmin` of primer §4.4. Sweeping them traces the assimilate/accommodate boundary —
which should align with bifurcation sets (primer §4.4), the formal "boundary is the hard part."

### 8.3 Evaluation (CL-specific, ground-truthed)

Synthetic stream with **known schema labels** (topologically distinct, reconstructable classes +
within-class affine/noise variants). Metrics:
- **Decision accuracy:** assimilate-to-correct-schema and accommodate-when-genuinely-new, vs ground
  truth. *The primary number.*
- **Forgetting:** after the full stream, re-evaluate reconstruction (D_stsp) on the *first* system of
  each schema. Module isolation should give ≈0 forgetting; contrast against a naive single-model
  baseline that overwrites. *Falsifier: assimilation degrades an earlier schema's reconstruction.*
- **Reuse rate / parameter savings:** fraction of systems that correctly reused a module (the CL win:
  new system handled by a chart, not a new module).
- **Boundary behaviour:** sweep a parameter toward a bifurcation (E5) and watch the decision flip from
  assimilate to accommodate; check it flips *at* the topological-type change.

### 8.4 What this is not (honest scope)
v0 demonstrates the *loop* on cleanly-separable, reconstructable classes (limit cycle / torus /
single-scroll chaos). It does **not** yet solve hard reconstructions (Lorenz two-wing, E0's open
problem) nor the shared-vocabulary backbone. The decisive claim is narrow and falsifiable: *a
gauge-free embedding + attention controller makes correct reuse/allocate decisions against ground-truth
conjugacy, with no-forgetting via module reuse.*

### 8.5 v0 results (run log)

Stream of 8 systems over 3 topologically-distinct classes (limit cycle / torus /
single-scroll chaos) as affine within-class variants; AL-RNN modules (latent 16, P=3),
PCA-canonicalised frames, decision on the channel-weighted distance.

**Probe (per-channel separability) — the load-bearing diagnostic:**
- **Dynamical** (Lyapunov spectrum + Kaplan–Yorke dim): clean gap, max-within 1.92 <
  min-across 2.18. The coordinate-invariant channel works.
- **Koopman** (fitted linear-core eigenvalues): large within-class scatter (LC↔LC 6.43) —
  fitted-operator variability swamps class structure. Unusable as-is; needs the
  faithfulness/whitening work (§5.A/B).
- **Symbolic** (activation-pattern transition graph): **degenerate** — several modules sit at
  distance 0 because on near-linear systems the P ReLU units barely flip → 1-symbol itinerary →
  trivial graph. The generating-partition problem, concrete.

**CL loop (τ=2.05, dynamical channel only):**
- Decision accuracy **75%** (6/8); **exactly 3 schemas allocated** for 3 classes (no over/under-
  allocation); **5/8 systems reused** a module.
- **No-forgetting by construction** (frozen modules) vs a naive single-model baseline that
  **forgot 2/3 classes** (Rössler D_stsp 10.2→15.5, torus 12.4→18.3).
- **Both errors are Rössler→torus**, and they *persist under better fitting* — so it is not
  underfitting but a **structural blind spot**: Rössler and torus share dim ≈2 and the dynamical
  channel cannot separate chaos from quasiperiodicity. The disambiguator is the **symbolic
  channel** (positive entropy vs zero) — which is exactly the degenerate one.

**Conclusion.** The CL mechanism is sound (correct allocation count, reuse, zero forgetting), and
the discrimination ceiling is set precisely by the missing topological channel — a direct,
ground-truthed confirmation of the multi-channel argument (§1): no single channel suffices, and
here the absent one *causes* the errors.

**Next step (clear).** Make the symbolic channel work: force the modules to exercise their ReLU
partition (raise P, or add a region-usage / entropy regularizer so the itinerary is non-trivial),
then the chaos-vs-quasiperiodic distinction enters the embedding and should close the Rössler/torus
confusion. This is the smallest decisive next experiment.

### 8.6 Region-usage regularizer: negative result, and the real bottleneck

Added a per-unit activation-entropy regularizer (maximise binary entropy of each ReLU unit's
active-rate) so the P units flip and the itinerary stops collapsing to one symbol. The units did
flip, but:

- **Symbolic channel still not class-discriminative.** Probe: `limit_cycle#0` and `rossler#0` remain
  *identical* (distance 0), within-class Rössler scatter large. Making units flip did not create
  class-consistent symbolic structure.
- **CL accuracy unchanged at 75%**, with a *worse* failure mode: only **2 schemas allocated** (torus
  conflated into Rössler) — the perturbed fits pushed torus and Rössler embeddings together.

**Root cause (ties back to E0).** The schema signature is **downstream of reconstruction fidelity**.
When a small module reconstructs Rössler as a limit cycle (E0's open problem — chaos is hard), its
symbolic dynamics is periodic (zero entropy), *identical* to a real limit cycle, and its Lyapunov
signature lacks the positive exponent — so **no channel, symbolic or dynamical, can place it.** A
regularizer cannot manufacture chaotic structure the reconstruction lacks; it only perturbs the fit.

**Conclusion for the thread.** The schema-memory / CL logic is sound (correct on the reliably-
reconstructed limit-cycle class every time; demonstrates reuse and zero forgetting). The
discrimination ceiling among dimension-≈2 attractors (chaos vs quasiperiodicity) is set by **whether
the AL-RNN actually reconstructs the chaos** — i.e. E0's core difficulty, not the embedding or memory
design. **The next real lever is reconstruction fidelity for chaotic attractors** (larger/better
modules per the E0 Rössler recipe; a boundedness-and-expansion-aware objective; or a chaos-promoting
spectral target), *before* adding more comparison channels. More channels cannot fix a signature
computed from a wrong attractor.
