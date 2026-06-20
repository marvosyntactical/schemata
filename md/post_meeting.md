# Post-meeting writeup: PIAGETS/SLAO — critical analysis and next steps

---

## 1. What the signature-Fisher actually computes

`signature_fisher` returns, for each parameter θ_j:

    F_j^Φ = (1/n_avg) Σ_r Σ_k  (∂φ_k^(r) / ∂θ_j)²

where r indexes independent attractor realisations and k indexes φ-dimensions.
This is the squared **column norm** of the Jacobian J of φ w.r.t. θ, averaged over
attractor trajectories.  Geometrically: F_j^Φ measures how much the φ-embedding
changes when θ_j is perturbed along its own axis.

### Connection to the Fisher Information Matrix

Standard EWC (Kirkpatrick et al.) uses the diagonal of the data Fisher:

    F_j^data = E_x [ (∂ log p(x|θ) / ∂θ_j)² ]

The signature Fisher is the Fisher of a *different* model: a Gaussian likelihood over
task identity in φ-space,

    p(task | θ) = N(φ(θ), σ² I)

Expanding the score:

    ∂ log p / ∂θ_j = − (1/σ²) Σ_k (φ_k − φ_task,k) · ∂φ_k/∂θ_j

At the anchor θ* (where φ(θ*) ≈ φ_task), the squared expected score reduces to:

    F_j^Φ ∝ (1/σ²) Σ_k (∂φ_k/∂θ_j)²

Exactly what we compute (up to the σ² constant absorbed into λ).  So the signature
Fisher is **the EWC Fisher of the claim "this model has task-identity φ_task"**,
rather than the claim "this model fits data x."

### Why it makes sense for AL-RNNs specifically

The key insight: we do not care whether parameter θ_j reproduces any particular
trajectory segment; we care whether θ_j encodes the *dynamical character* of the
attractor — its geometry, topology, chaotic exponents.  The φ-embedding is explicitly
designed to capture those properties (eigenvalue magnitudes, activation rates,
Lyapunov exponent, topological entropy).  Protecting parameters to which φ is
sensitive therefore protects the *identity of the learned dynamical regime*, not just
the fit to a finite data batch.

A further advantage: the signature Fisher is data-free (no batches needed), computed
from an autonomous rollout.  It therefore does not depend on which particular
trajectory segments were in the training batch, making it more stable and reproducible
than the data Fisher.

### Arguments against

1. **φ ≠ trajectory quality.** A model can have the correct φ-signature (attractor
   geometry matches) but reconstruct individual trajectories poorly — wrong transient
   dynamics, wrong phase, or wrong scaling.  EWC on φ protects the attractor shape but
   not the path to it.  The relevant practical quantity, reconstruction MSE on task-i
   data, is not guaranteed to be protected.

2. **Teacher-forced vs. autonomous mismatch.** Training uses teacher-forced rollouts
   (α-scheduled mixing); φ is computed from a fully autonomous rollout.  The parameters
   that matter most for reconstruction performance may differ from those that control the
   free-running attractor.  Concretely: W_B controls how much the nonlinear path feeds
   back into the linear state; under forcing (α < 1) the model is partially driven by
   data and never fully enters its own autonomous regime.  Protecting W_B to preserve
   the autonomous φ may not be the same as protecting reconstruction accuracy under
   forcing.

3. **Fixed initial condition.**  φ is always computed starting from z = 0.  If the
   attractor has a complex basin structure, the model's true φ on the actual training
   distribution may differ.  This makes the Fisher an estimate for one specific
   trajectory, not an expectation over the attractor.

4. **EWC on A and h may be harmful.** A (diagonal linear recurrence) sets the
   memory timescale of each latent unit; h (bias) shifts the operating point.
   Different tasks may genuinely require different A and h — Lorenz needs fast chaotic
   mixing, VdP needs slower oscillatory dynamics.  Strongly protecting A and h with
   EWC may prevent the model from adjusting memory timescales between tasks, which
   is a fundamental mode of adaptation.  The current results (R_diag ≈ −11 for
   PIAGETS methods) are consistent with the model being over-constrained.  Worth
   ablating: EWC on W_B only vs. W_B + A + h.

---

## 2. The φ-component weighting problem

The current Fisher weights all 24 φ-components equally:

    F_j^Φ = Σ_k (∂φ_k/∂θ_j)²

The 24 components are:
- (M-P)=10 linear-core eigenvalue magnitudes
- P=6 soft activation rates (mean)
- P=6 soft activation std (variability)
- 1 soft topological entropy
- 1 max Lyapunov exponent

These have vastly different inter-task discriminative power.  From the oracle run:

    Lor_r28:  φ[-3:] = [0.006, 1.339, -0.027]   ← topo-entropy = 1.34
    Tor_0.382: φ[-3:] = [0.456, 0.468,  0.004]   ← topo-entropy = 0.47
    VdP_m1.5:  φ[-3:] = [0.419, 0.428, -0.001]   ← topo-entropy = 0.43

The Lyapunov exponent (last component) is near-zero for ALL three classes here, so it
contributes almost zero inter-class signal.  The topological entropy is the primary
discriminator between Lorenz and the other two.  The Lorenz eigenvalue magnitudes
(first 10 components) will likely differ substantially from Torus/VdP, but the 6
activation rates may be similar.

This means the current unweighted Fisher over-weights the activation rate components
(12/24 of the total) which may be less discriminative, and under-weights the 1-2 most
discriminative dimensions.

### Principled weighting options

**Option A — inter-task variance weighting:**

    w_k = Var_tasks(φ_k^oracle) / mean_j[Var_tasks(φ_j^oracle)]
    F_j^Φ = Σ_k w_k · (∂φ_k/∂θ_j)²

Focus protection on φ-dimensions that actually distinguish tasks.  Computed once from
oracle models; adds no new hyperparameters (it's derived from the oracle run that
already happens at the start of validation).

**Option B — Fisher discriminant ratio weighting:**

    w_k = between-class variance of φ_k / within-class variance of φ_k

Penalises components that vary a lot *within* the same class (noisy); rewards components
with high between-class contrast.  This is the LDA criterion applied to φ-space.

**Option C — Mahalanobis distance in φ-space:**

Replace the Euclidean column norm with a Mahalanobis norm:

    F_j^Φ = (∂φ/∂θ_j)^T Σ^{-1}_φ (∂φ/∂θ_j)

where Σ_φ is the within-class covariance of φ across oracle models.  This is the full
Fisher of an LDA model on φ and is theoretically principled — it up-weights dimensions
that vary little within class but much between classes.  In practice Σ_φ is low-rank
(three class means, high-dimensional φ), so a pseudoinverse or regularised inverse is
needed.

**Recommendation:** Start with Option A (inter-task variance weighting) — zero extra
hyperparameters, one call to compute w_k from existing oracle models, expected to
substantially improve Fisher quality.

---

## 3. Circular metrics: does it make sense to report BWT, rec_ora, rec_cen?

### The circularity problem

BWT, rec_ora, and rec_cen all measure distances in φ-space.  PIAGETS/SLAO explicitly
optimises in φ-space (B_merge EMA creates a centroid in φ-space; signature-Fisher EWC
protects parameters that move φ).

Consequence:
- **BWT** measures mean_i(R[T-1,i] − R[i,i]) where R[t,i] = −||φ(model_t) − φ(oracle_i)||.
  PIAGETS-EWC minimises ||φ(θ_t) − φ(θ*)|| as part of its loss, so BWT is partially
  measuring whether the EWC term successfully fired.  It is *not* an independent
  evaluation of forgetting.
- **rec_ora** measures whether φ(model_t) is nearest to φ(oracle_t).  PIAGETS's B_merge
  step deliberately moves the model to a *blend* of all task φ's (not any one oracle),
  so rec_ora is penalising something PIAGETS is by construction doing.
- **rec_cen** uses class centroids of oracle φ's.  Slightly less circular because
  centroids aggregate, but still φ-based.

Methods that do *not* use φ in their loss (vanilla, baseline, pred_ewc) are being
compared to PIAGETS on a metric that is tied to φ — which is an asymmetric playing
field.  The current numbers show piagets having the best BWT (+1.09) but worst rec_ora
(0.40).  The second result is not surprising: piagets explicitly blends φ away from any
single oracle.  The first result is also not surprising: piagets EWC penalises φ drift.

This does **not** mean the φ-based metrics are wrong — they are informative about the
internal structure of the learned representations.  But they are not measuring whether
the model can actually reconstruct the dynamics of past tasks.

### The metric we should be using as primary

**Reconstruction-BWT:**

Compute an MSE matrix:

    M[t, i] = MSE(model_t, held_out_data_i)

where held_out_data_i is a test set for task i (different random seed from training).
This gives:

    BWT_mse = mean_{i<T} (M[T-1, i] − M[i, i])

M[i,i] is the MSE right after training on task i (best achievable without forgetting).
M[T-1,i] is how well the final model reconstructs task i's trajectories.
BWT_mse < 0 means the model has forgotten; BWT_mse ≈ 0 means no forgetting in
reconstruction space.

This is independent of φ and directly measures whether the CL method preserves
practical task knowledge.

**Additional metrics:**

- **Trajectory quality per task**: Phase portrait coverage (Hausdorff distance between
  model attractor cloud and true attractor cloud), or Wasserstein distance between
  empirical trajectory distributions.  This penalises "right φ but wrong orbit."
- **Plasticity**: epochs needed to fine-tune model_T to oracle-level MSE on task i,
  starting from the continual model.  Lower is better — a good CL method should
  preserve the attractor structure such that re-adaptation is cheap.
- **Class-conditioned MSE**: Given a test trajectory labelled with class c, evaluate
  the model's MSE on it after all 10 tasks.  Averaged within class.  This tests whether
  the merged model is a usable generalist or a confused blend.

---

## 4. Does adaptive λ actually make sense?

### The Piagetian story

The hypothesis: when a new task is "assimilation" (within existing schemas), strong EWC
(high λ) is safe — the model barely needs to move, so protection is costless.  When it
is "accommodation" (novel), relaxed EWC (low λ) is needed — the model must restructure.

### Where it holds

The logic is sound under specific conditions:
1. The probe correctly identifies the mode (currently 8/9 accuracy, which is good).
2. "Assimilation" genuinely requires small parameter changes.
3. "Accommodation" genuinely requires large parameter changes that EWC would block.

Under these conditions, adaptive λ is a form of *elastic constraint* that tightens when
unnecessary and loosens when needed.

### Where it fails

**The convergence speed argument does not hold in the CL sense.**  A low λ on
accommodation episodes makes the model converge quickly on the *new task* reconstruction
loss, but this is done at the expense of forgetting old tasks.  If the goal is
"converge to a good model of all past tasks while learning the new one," relaxing λ
provides no benefit — it just shifts the bias toward the current task.

**The probe sets λ based on the merged model's MSE, not the fine-tuned model's.**  The
merged model is not specialized for any task, so its MSE on the new data is generally
high (even if the new task is similar to a past one in the fine-tuned model's sense).
The ref_max calibration introduced in the probe_from_task_start fixes this partially,
but the reference window is inherently noisy — large ratios from high-MSE merged states
at early tasks can dominate the scale.

**Hard thresholds amplify variance.**  Changing λ from 2 to 10 across a single
threshold creates a cliff.  A task with ratio = 9.9 (ASSIMILATION, λ=10) and a task
with ratio = 10.1 (ACCOMMODATION, λ=2) are treated completely differently, but may be
nearly identical in difficulty.  The current results show this in the pred_ewc_adaptive
numbers: BWT=-2.56 with adaptive λ vs BWT=-1.28 with fixed λ=5.0.  The adaptive
mechanism, despite fixing the catastrophic failure from the first run, does not
outperform a well-chosen fixed λ on BWT.

**Asymmetry between λ values:**  λ_assim = 10 is 5× the default (5), while λ_accom = 2
is 2.5× smaller.  If most tasks are "assimilation" (as they are after task 1 in this
stream), the average effective λ is much higher than 5.  This over-constrains the model
and explains why BWT is actually worse with adaptive λ (−2.56) than fixed λ (−1.28) for
pred_ewc variants.  The adaptive mechanism is not symmetric around the fixed-λ
operating point.

**What accommodation actually means for the EWC anchor:**  On an accommodation task,
we run with λ_accom but then call store_task and update the EWC anchor to include the
new task's Fisher.  This is correct — the anchor expands to include the new task.  But
the EWC penalty during accommodation training still anchors to the *previous* tasks.
If λ_accom = 2 is too low, those anchors are too weak and forgetting happens.  If λ_accom
is too high, we can't learn the new task.  This is the same tension as fixed-λ EWC; the
probe adds complexity without resolving the fundamental trade-off.

### A better adaptive mechanism

Rather than setting λ globally at task start, consider:

**Online adaptive λ during training:**
Track the rate of φ-drift during training:

    dφ/deps = ||φ(model_{ep+1}) − φ(model_{ep})|| / ||φ(model_{ep})||

If dφ/deps exceeds a threshold (model is "forgetting fast"), increase λ for the next
epoch.  This is a closed-loop controller rather than an open-loop probe.

**Smooth λ(ratio) schedule:**
Replace the binary threshold with a smooth schedule:

    λ = λ_min + (λ_max − λ_min) · exp(−ratio / τ)

When ratio is small (assimilation), λ ≈ λ_max.  When ratio is large (accommodation),
λ ≈ λ_min.  The τ hyperparameter controls the transition width.  This eliminates the
cliff and has only one additional parameter vs. the binary scheme.

**Per-parameter λ from Fisher magnitude:**
Instead of a scalar λ, multiply the EWC penalty elementwise by a "task novelty" factor:

    L_EWC = Σ_j F_j^Φ · λ_j · (θ_j − θ_j*)²

where λ_j = λ_max if ∂φ/∂θ_j is large (this param controls the familiar φ region) and
λ_j = λ_min if ∂φ/∂θ_j is small.  This is already implicitly done by the Fisher
weights, but could be made more explicit.

---

## 5. Metrics we should be reporting but aren't

### Primary: reconstruction-space forgetting

| Metric | Definition | Why |
|--------|------------|-----|
| **BWT_mse** | mean_i(MSE[T-1,i] − MSE[i,i]) | Task performance, not φ distance |
| **MSE_diag** | mean_i MSE[i,i] | Per-task fit quality (currently reported as mse_avg, but only at train time) |
| **MSE_final** | mean_i MSE[T-1,i] | Final generalist model quality |
| **Worst_case BWT_mse** | min_i(MSE[T-1,i] − MSE[i,i]) | Catastrophic forgetting flag |

### Secondary: trajectory quality

- **Hausdorff distance** between sampled model trajectory and true attractor cloud (per
  task, for the final model): "does the model explore the right region of phase space?"
- **Lyapunov exponent error**: |LE_model − LE_true| per task.  Does the final model
  have the right chaotic character?
- **Power spectrum MSE**: Does the frequency content of model-generated trajectories
  match the true system?  Particularly relevant for torus (sharp frequency peaks) vs.
  Lorenz (broadband).

### Diagnostic: plasticity and transfer

- **Fine-tuning efficiency**: Start from model_T; fine-tune on task-i data for k epochs.
  Record MSE at k = 10, 50, 100 epochs.  Compares how much "useful prior" each CL
  method has retained.
- **Forward transfer (properly measured)**: MSE of model_t on task_{t+1} data *before*
  training on task_{t+1}.  Current FWT uses R-matrix (φ-distance), not MSE.

### Currently reported — re-assessment

- **R_diag_mean** is fine as a diagnostic (how specialized is the model right after each
  task?), but its absolute value reflects both φ-space scale and training duration, not
  just forgetting.
- **rec_ora and rec_cen** are useful diagnostics for whether the CL method maintains
  separable φ-representations, but should not be primary metrics.  They tell us about
  the geometry of the learned latent space, not about task performance.
- **BWT (φ-based)** and **FWT (φ-based)** should be supplemented by MSE-based versions.
  The φ-based BWT is informative for PIAGETS methods specifically but is not directly
  comparable across methods that use vs. don't use φ.

---

## 6. Visualisations to make

### Attractor phase portraits (the pretty pictures)

For each method × final model (after all 10 tasks):
- Plot autonomous trajectory in 3D phase space for each task's data.
- The model receives the ground-truth initial condition and runs freely for N steps.
- Compare side-by-side with the true attractor for each system.

This is the most direct visual demonstration of what CL preserves.  Expected results:
- vanilla: only reproduces the last task (Lor_r45) correctly.
- baseline: messy attractors on most tasks by the end.
- piagets: should show a "blended" attractor that partially covers multiple regimes.
- piagets_adaptive: similar to piagets but with sharper per-class structure if rec_ora is higher.

For each method × each task at the time of training (i.e., model_t on task_t data):
- Shows how well each method learns each individual task in isolation.

### Per-task MSE matrix heatmap

    M[t, i] = MSE(model_t, data_i_test)

Colorscale: low MSE = good (green), high MSE = bad (red).  Diagonal = freshly trained
quality.  Show this for each method.  This is the reconstruction-space analogue of the
current R-matrix and should replace it as the primary evaluation figure.

### φ-trajectory in PCA space

Project all φ vectors (oracle phis, and CL model phis at each step) into 2D PCA.
Show:
- Oracle phis as large dots coloured by class (Lorenz=blue, Torus=orange, VdP=green).
- CL model's φ trajectory over tasks as a connected path, with arrows.
Expected insight: piagets paths stay in the "interior" of the oracle constellation (good
blending but poor specialisation); vanilla paths jump to each oracle and then abandon it.

### EWC Fisher importance map

Heatmap of F_B[i,j] (importance of W_B[i,j] for each task).  Show how the consolidation
region in weight space evolves across tasks.  Should reveal whether the Fisher concentrates
on interpretable subspaces of W_B.

### Probe decision timeline

For the adaptive methods: bar chart with one bar per task, coloured by
ASSIMILATION/ACCOMMODATION and marked correct/incorrect vs. ground truth (first
occurrence = accommodation, repeat class = assimilation).  Overlay the ratio values.
This communicates the probe's behaviour more clearly than printed logs.

### Parameter drift over training

For each method, track per-epoch:
- ||W_B − W_B^0||_F (drift from initial W_B after task 0)
- ||A − A^0||_F
- EWC loss magnitude

This shows whether methods are actually constrained by EWC or whether λ is ineffectively
small.  If EWC loss ≈ recon loss, the EWC is active; if EWC loss << recon loss, λ is
too small.

---

## 7. Variants and experiments to run next

### Priority 1: Fix the metric

1. **Add MSE matrix evaluation**: After each task t, evaluate the current model on
   held-out test data for all tasks 0..t.  Report BWT_mse and MSE final per task.
   This is independent of φ and gives ground truth on forgetting.

2. **Trajectory quality plots** for the current results (no new training needed):
   Run the final model on each task's data and plot phase portraits.  Already have
   the models; just need to add a plotting loop.

### Priority 2: Ablate the EWC target

3. **EWC on W_B only (r=12)**: Drop the EWC on A and h that was added in this session.
   Hypothesis: protecting A and h is over-constraining the model and responsible for
   the worse R_diag mean (−11) compared to what the slides showed.  The original setup
   may have been W_B only.

4. **EWC on W_B + W_A** (the encoder factor): W_A (r×M) controls the projection of the
   latent state into the low-rank update.  Protecting W_A may make more sense than
   protecting A directly.

### Priority 3: Weighted Fisher

5. **Inter-task variance weighting** (Option A from §2):
   Compute w_k = Var_{oracles}(φ_k) and use weighted Fisher.
   Expected effect: up-weight topological entropy and eigenvalue magnitudes (high
   inter-class variance), down-weight Lyapunov exponent (near-zero for all classes here).

### Priority 4: Better adaptive λ

6. **Smooth λ(ratio) probe**: λ = λ_min + (λ_max − λ_min) · exp(−ratio / τ).
   Grid search τ ∈ {1, 5, 20}.  Compare BWT_mse (not BWT_φ) to fixed-λ baseline.

7. **Symmetric λ values**: Currently λ_assim=10 >> λ_default=5 >> λ_accom=2.
   The asymmetry biases the average effective λ upward.  Try λ_assim=7, λ_accom=3
   (symmetric around λ=5) and compare.

### Priority 5: System-level

8. **Rank ablation**: Compare r=6 (original) vs r=12 (current) with a fixed, well-tuned
   λ=5 on W_B only.  Check whether r=12 genuinely helps or hurts in the CL setting
   (more parameters = more to protect = harder EWC problem).

9. **Multi-seed runs**: All current results are for a single random seed (seed=0 for the
   CL model, seed=42 for the vanilla re-initialisation).  BWT_mse values will have
   substantial variance across seeds.  Run 3 seeds minimum before drawing conclusions.

---

## 8. Summary of findings from this session

| Question | Finding |
|----------|---------|
| φ-Fisher connection to data Fisher | Valid via Gaussian model on φ-space; protects attractor identity not trajectory fit |
| EWC on A and h | Likely harmful — different tasks need different memory timescales; ablate |
| φ-component weighting | All components weighted equally is wrong; inter-task variance weighting is principled and adds no hyperparameters |
| Metric circularity | BWT/rec_ora partially measure PIAGETS's own objective; primary metric should be BWT_mse |
| Adaptive λ | Logic is sound but binary threshold amplifies variance; smooth λ(ratio) preferred; no evidence adaptive λ speeds convergence in the useful sense |
| Current best method (φ-space metrics) | piagets: BWT=+1.09 (but rec_ora=0.40); piagets_adaptive: BWT=−0.61, rec_ora=0.70 |
| pred_ewc_adaptive fix | Threshold=10.0 eliminates catastrophic failure (BWT −4.32 → −2.56) |
| Most urgent next step | Add MSE-matrix evaluation; plot phase portraits |
