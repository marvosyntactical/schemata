# Summary — schemata in dynamical-systems reconstruction

*Capstone reference for the `schemata/` project. Ties together the theory ([primer.md](../primer.md)),
the model ([SC_RNN.md](SC_RNN.md)), the design notes ([dsa_native_ssm.md](dsa_native_ssm.md)), the
reconstruction experiments ([experiments_00.md](experiments_00.md)), and the real-neural-data tests.
Written to be read standalone.*

---

## 1. The professor's framing, as we now interpret it

The brief: *build "schemata" and "analogies" into dynamical-systems models, or identify principles
that foster it*, for neuroscience time series. The apparent paradox we spent real effort on — *DSR
wants one model = one dynamics, but CL/assimilation wants one model to handle many* — **dissolves once
you separate two levels:**

- **Object level:** a single reconstructed system `F_θ` (one attractor). DSR success is defined here.
- **Meta / repertoire level:** a model of the *space* of systems — a parameterised family
  `F_{θ_shared, w}` + a map `data → w`. Schemata, assimilation, accommodation live here.

CL operates at the **meta** level; every individual reconstruction is still an honest single-system
DSR. **CL is hierarchical/amortised DSR + a sequential decision layer** — not its opposite. A *schema*
is the structure in `w`-space; **assimilate** = the new system lands in a known region of `w`,
**accommodate** = it needs a new region.

Two readings of "schema", both grounded in the lab's own work (Koppe et al. 2019 fMRI; Durstewitz/Huys/
Koppe, *psychiatric illness as disordered network dynamics*; PLRNN on rat PFC working memory):

- **Across-subject (computational psychiatry):** each subject/condition is a member of a dynamical
  schema; accommodation = a subject whose dynamics crossed a **bifurcation** into a new regime — the
  disease-as-altered-dynamics thesis. The repertoire is a *typology of neural-dynamical regimes*.
- **Within-subject context/task switching:** the brain (esp. PFC) runs different dynamical regimes for
  different contexts. The PLRNN's **input term `C·s_t`** is the native mechanism: stimulus/context
  `s_t` reconfigures the dynamics. Schema = context-specific regime; accommodation = a novel context
  forces a new attractor (learning). **This resolves the tension** — one system whose *input-conditioned*
  dynamics realise the repertoire (no forced multistability cramming).

The three CL settings ([primer.md](../primer.md) §4.1): **A** (across-system sequential — our synthetic
build), **B** (within-stream nonstationary — where bifurcation = accommodation, the tightest DSR–CL
unification), **C** (in-context/DynaMix — horizon).

---

## 2. What we built

A full continual-learning schema pipeline on synthetic data, plus real-neural-data probes.

| File | Role |
|---|---|
| [systems.py](systems.py) | Lorenz, Rössler, limit-cycle, torus; `affine_variant` (nuisance 𝒢), `canonicalize` (PCA gauge-fix), `make_stream` (labelled CL stream) |
| [plrnn.py](plrnn.py) | PLRNN + generalized-teacher-forcing trainer (model-agnostic) |
| [alrnn.py](alrnn.py) | Almost-Linear RNN (ReLU on `P` units), `region_reg`, `itinerary` (symbolic), spectrum/Jacobian extraction |
| [metrics.py](metrics.py) | `d_stsp`, power-spectrum distance, Lyapunov spectrum (QR), Kaplan–Yorke dim |
| [embeddings.py](embeddings.py) | gauge-free signature: Koopman ⊕ symbolic ⊕ dynamical channels |
| [schema_memory.py](schema_memory.py) | prototype memory + attention + assimilate/accommodate (leader clustering) |
| [cl_run.py](cl_run.py) | the CL stream + evaluation (decision accuracy, reuse, forgetting) |
| [run_e0.py](run_e0.py) | single-system reconstruction (E0) |
| [neuro_eegbci.py](neuro_eegbci.py) | EEG motor-imagery context test (PhysioNet via MNE) |
| [neuro_fmri.py](neuro_fmri.py) | fMRI dynamical fingerprinting on the lab's BOLD data |
| [make_figs.py](make_figs.py) | figures from logged results |

---

## 3. What we learned

### 3.1 Reconstruction (DSR) is the foundation, and it's hard
- All four named failure modes ([primer.md](../primer.md) §3.4) reproduce and are cured by **generalized
  teacher forcing + contractive init + a state clip**. Rössler reconstructs cleanly (Kaplan–Yorke dim
  **2.007** vs true 2.01); the two-wing **Lorenz never** does with a shallow PLRNN.
- **Topological complexity, not "chaos", is the axis of difficulty:** Rössler (single scroll) is easy,
  Lorenz (bistable two-lobe) is not. ([experiments_00.md](experiments_00.md) §4b.)
- **A metric can lie:** our first `d_stsp` drew its histogram range from both clouds, so one divergent
  outlier collapsed everything into one bin and reported a fake-good score. Fixed by anchoring the range
  to the *true* attractor + a divergence flag. *Lesson: a judge-by-invariants pipeline is only as honest
  as the invariant's behaviour under blow-up.*

### 3.2 Schema signatures — which channel actually works
Per-channel separability on within/across-class variants ([dsa_native_ssm.md](dsa_native_ssm.md) §8.5,
`fig_channels.png`):
- **Dynamical (Lyapunov spectrum + dimension): clean.** Coordinate-invariant, real within<across gap.
- **Koopman (fitted-operator eigenvalues): noisy.** Large within-class scatter from fit variability.
- **Symbolic (activation-pattern graph): degenerate.** On near-linear systems the ReLU units barely
  flip → one-symbol itinerary → trivial graph (the generating-partition problem). A region-usage
  **regularizer did not fix it** (negative result): the symbolic signature is **downstream of
  reconstruction fidelity** — a module that renders Rössler as a limit cycle has periodic symbolic
  dynamics *identical* to a real limit cycle.

### 3.3 The continual-learning loop works; its ceiling is reconstruction
On an 8-system stream (`cl_run.py`, `fig_decisions.png`, `fig_forgetting.png`):
- **75%** decision accuracy; **exactly 3 schemas** allocated for 3 classes; **5/8** systems reused a
  module; **zero forgetting** (frozen modules) vs a naive single model that **forgot 2/3 classes**.
- Both errors are **Rössler→torus** — chaos misread as quasiperiodicity — and they **persist under
  better fitting**, so the ceiling is set by **reconstruction fidelity** (E0's open problem), not the
  schema machinery. The disambiguating channel (symbolic: positive vs zero entropy) is the degenerate
  one.

### 3.4 Conceptual clarifications (these reshaped the design)
- **The schema memory is an associative memory** = online **leader clustering** (nearest-prototype +
  novelty threshold), *not* a hashmap (keys are continuous, looked up by similarity) and *not* an LRU
  (no eviction — though capacity-bounded eviction is the natural extension, and the policy becomes a
  forgetting policy).
- **Retrieval/reuse only pays when matching is cheaper than fitting, OR data is too scarce to fit.**
  Bayesian view: schema = prior, data = likelihood. Abundant clean data washes out the prior → reuse is
  redundant for *modeling*, leaving only **DSA-esque clustering** (the recognition question). Our toy
  regime (full-fit probe + long clean data) is precisely the dead zone where reuse can't win — which is
  why our reuse showed no compute gain.
- **Real neuroscience data is the scarce/noisy regime where reuse *would* matter** — which is why the
  prof frames it for neuroscience, not clean simulations.

### 3.5 Real neural data — the spatial-vs-temporal lesson (the biggest finding)
Three independent real-data probes converge:
- **EEG motor imagery** (`neuro_eegbci.py`, `fig_eeg_contexts.png`): our global Koopman signature does
  **not** separate left/right imagery (CV acc ≤ chance). The discriminative info is **lateralised
  (spatial)**; a global temporal spectrum averages it away.
- **fMRI** (`neuro_fmri.py`, `fig_fmri_fingerprint.png`): subject-identity fingerprinting — our
  **dynamical signature 0.06** (≈ chance 0.038) vs **static functional connectivity 0.85**. Identity
  lives in **spatial connectivity**, not the temporal spectrum.

**Convergent lesson:** in short / slow / noisy real neural data, the schema-relevant information is
largely **spatial/geometric**, and a purely temporal-dynamical signature misses it. This is in real
tension with DSA's "*Beyond Geometry*" thesis: the right signature is **regime-dependent** — temporal
dynamics need clean, long, fast data; fMRI/single-subject-EEG are the opposite, and geometry wins.
*Caveat:* our dynamical signature is the cheap data-side DMD; the lab's full **input-driven PLRNN** (using
the stimulus regressors) extracts more and is the fair next test.

---

## 4. Caveats of our approach

1. **Reconstruction fidelity is the bottleneck.** Every downstream signature (symbolic especially)
   degrades when the attractor isn't reconstructed; shallow PLRNNs fail on topologically rich attractors
   (Lorenz). Dendritic/clipped PLRNNs or larger capacity are needed.
2. **Gauge handling is partial.** PCA canonicalisation fixes rotation/reflection but leaves a residual
   in-plane ambiguity for **rotationally-symmetric attractors** (limit cycle, torus) — exactly where the
   symbolic channel also wobbles.
3. **Signature blind spots are real and bit us on real data.** The temporal-dynamical signature is blind
   to spatial structure (EEG lateralisation, fMRI connectivity). A complete schema signature needs a
   **geometry channel** too.
4. **Identifiability ceiling** ([primer.md](../primer.md) §1.3, §7): from data we recover dynamics only up
   to conjugacy + invariant measure; there is no finite complete invariant for chaotic conjugacy. The
   multi-channel signature is *discriminative-on-the-families*, not complete.
5. **Probe-then-commit pays a full fit per system**, so reuse currently saves *storage and forgetting*,
   not *training FLOPs* — the efficiency case is unproven until a cheap data-side probe replaces it
   ([SC_RNN.md](SC_RNN.md) §4.2).
6. **Modular = K copies.** Rejected as wasteful; a single-backbone (input-conditioned / multistable,
   backprop-CL) design is the intended replacement — at the cost of real interference (forgetting is no
   longer free) and bounded attractor capacity.
7. **fMRI is the hardest modality for dynamics** (slow TR + HRF smoothing). Ephys would be kinder.

---

## 5. How to estimate within-schema variance

Currently each prototype is a **point** (running-mean centroid) and the decision is a fixed-radius
spherical test — it throws away within-class spread. The principled upgrade turns leader-clustering into
an **online Gaussian mixture** (Dirichlet-process mixture in the limit):

- Store per-schema **sufficient statistics** `(count, μ_k, Σ_k)`, updated online (Welford).
- Replace the global-z-scored Euclidean test with **Mahalanobis** `d_M(x,k)=√{(x−μ_k)ᵀ Σ_k⁻¹ (x−μ_k)}`.
- The threshold becomes a **statistical test**: under a Gaussian schema model `d_M² ∼ χ²(D)`, so
  `τ` → a χ² confidence level (retires the "choose τ" problem, [primer.md](../primer.md) §4.4).
- **Estimation with few samples is ill-posed** (`D≈18`, ~2–3 members/schema): use a **diagonal**
  covariance, **shrink toward the global** covariance (Ledoit–Wolf), or an **inverse-Wishart** prior;
  floor the diagonal to avoid singular `Σ_k`.

Two payoffs beyond a better boundary:
- **Auto per-schema channel weighting.** A channel noisy *for this schema* gets large variance → small
  Mahalanobis contribution → down-weighted automatically — replacing the hand-tuned `WEIGHTS` hack, per
  schema.
- **The chart for free.** The **principal axes of `Σ_k` are the within-class parameter directions** —
  the steerable axes of the equivalence class (the generativity desideratum). Estimating variance and
  learning the chart are the same operation.
- *Honest note:* `Σ_k` conflates genuine within-class spread with **embedding estimation noise**; only
  the former is the schema's true width. Better/more-stable embeddings shrink the latter.

Implementation: ~30 lines in [schema_memory.py](schema_memory.py) — `(count, mean, M2)` per prototype,
diagonal+shrinkage `Σ_k`, min-Mahalanobis, χ²-threshold accommodate test.

---

## 6. How to work with the lab's fMRI data

**Source.** [DurstewitzLab/PLRNN_SSM](https://github.com/DurstewitzLab/PLRNN_SSM) (Koppe et al. 2019,
PLOS Comp Biol). Download/extract:

```bash
git clone --depth 1 https://github.com/DurstewitzLab/PLRNN_SSM.git
cd PLRNN_SSM
python3 -c "import zipfile; zipfile.ZipFile('code_PLRNNs.zip').extractall('extracted')"
# BOLD data: extracted/code_PLRNNs/code_PLRNNreg_BOLD_SSM/data/datafile_001..026.mat
```
We copied the 26 `.mat` files to [data_fmri/](data_fmri/); [neuro_fmri.py](neuro_fmri.py) reads them.

**Format** (`scipy.io.loadmat(...)['PLRNN'][0,0]`, a MATLAB struct):
- `data`: `(20, 360)` — 20 ROIs × 360 TRs of standardised BOLD.
- `Inp`: `(5, 360)` — **5 stimulus-input regressors** (the experimental context drive `s_t`).
- `ROI`: `labs` (region names — BA 9/10/32/40/45/46/47/6/7 bilateral + cerebellum: PFC/executive),
  `coords`, `method`.
- `window`, `sigma`: preprocessing scalars; `rp`: nuisance/motion regressors.
- 26 files = 26 subjects.

**Loading pattern** (see `neuro_fmri.load_all`): iterate `datafile_*.mat`, pull `data` and `Inp`. The
`Inp` regressors are what make this an **input-driven** dataset — the natural place to test the
context-switching schema reading. Other modalities: [neuro_eegbci.py](neuro_eegbci.py) shows the MNE/
PhysioNet EEG path (`mne.datasets.eegbci.load_data`, bandpass, epoch by annotation).

---

## 7. Future directions (prioritised)

1. **Geometry ⊕ dynamics ⊕ input-response signature.** The real-data finding says add a **connectivity/
   covariance channel**. Fit the lab's **input-driven PLRNN** (BOLD + `Inp`); use its **effective-
   connectivity `W`** (spatial), its **operator spectrum** (dynamics), and its **stimulus-response** (how
   `Inp` reconfigures the phase portrait) as the schema triple. Cluster the 26 subjects by context-
   response. *This is the experiment with real teeth.*
2. **Cheap data-side probe** ⇒ realise the compute win: match from data-side EDMD/Lyapunov *before*
   committing to a fit; only fit on accommodate. Converts correct reuse decisions into actual FLOP
   savings.
3. **Fix chaotic reconstruction** to lift the 75% ceiling (dendritic/clipped PLRNN, chaos-promoting
   spectral target) — once Rössler is genuinely chaotic, both dynamical and symbolic channels separate
   it from the torus.
4. **Single-backbone backprop-CL** (the no-copies design): one input-conditioned RNN, schema repertoire
   as basin seeds / context codes, updated by backprop with **generative replay** (the model rehearses
   itself by free-running stored schemas). Report **interference** (real forgetting) and **attractor
   capacity**.
5. **Within-schema variance** (§5): online Gaussian/DP-mixture memory with χ²-thresholded accommodation.
6. **Setting B / bifurcation = accommodation:** sweep a parameter through a bifurcation; verify the
   decision flips assimilate→accommodate *at* the topological-type change ([primer.md](../primer.md) §4.4).
7. **Ephys over fMRI** for any dynamics-first claim (PFC single-unit working-memory data).

---

## 8. Artifact index

**Docs:** [primer.md](../primer.md), [SC_RNN.md](SC_RNN.md), [dsa_native_ssm.md](dsa_native_ssm.md),
[experiments_00.md](experiments_00.md), this file.
**Figures:** `e0_rossler_alpha0.15.png` (reconstruction), `fig_channels.png` (per-channel separability),
`fig_decisions.png` (CL decisions), `fig_forgetting.png` (forgetting vs naive), `fig_eeg_contexts.png`
(EEG), `fig_fmri_fingerprint.png` (fMRI temporal-vs-spatial).
**Data:** [data_fmri/](data_fmri/) (26 subjects, lab BOLD).

**One-line state of play:** the schema-CL machinery is sound (correct allocation, reuse, zero
forgetting), but every claim is gated by (a) **reconstruction fidelity** on synthetic data and (b) the
fact that on real neural data the schema-relevant information is substantially **spatial/geometric** —
so the next decisive work is a *geometry-aware, input-driven* signature on the lab's own fMRI, not more
temporal-dynamical channels.
