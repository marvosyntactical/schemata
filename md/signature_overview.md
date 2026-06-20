# Dynamical Schema Signatures: Mathematical Reference

A self-contained technical account of the signature embedding in `signatures.py`.
Covers the dynamical systems background, the mathematical definition and cost of each
signature block, the weighting framework, and an honest assessment of what the
signatures can and cannot distinguish.

**Code**: `signatures.py` (blocks + distances + weighting), `alrnn.py` (substrate),
`metrics.py` (Lyapunov), `systems.py` (test attractors).
**Experiment log**: `signature_findings.md`.

---

## 1. Philosophy

We train dynamical systems models (AL-RNNs) to reproduce the *temporal structure* of
observed time series and then ask: **which trained models implement structurally
equivalent dynamics?** The goal is a fixed-length *signature vector* for each trained
model such that models implementing the *same kind* of dynamics land near each other
in signature space, regardless of which coordinate system the data happened to be
measured in.

This clustering defines *schemata*: canonical dynamical motifs — a fixed point, an
oscillation, a chaotic attractor — that recur across subjects, sessions, or conditions.
The hard part is specifying what "structurally equivalent" means precisely and computing
it cheaply from a trained model.

The central design decision is to **separate the equivalence relation from the
computation**:

- *Topological blocks* measure properties invariant under homeomorphic
  reparametrisation of state space. These define what counts as the *same schema*.
- *Rate blocks* measure quantities that change under smooth reparametrisation or a
  change of time step. These describe *how fast* a system implements a schema.
- *Geometry blocks* are data-side descriptors that depend on the measurement frame
  (electrode placement, scanner resolution). These capture what the gauge-free model
  channels necessarily discard.

Choosing how to weight these three classes is an explicit modelling decision, not a
hyperparameter to tune away (§6.1).

---

## 2. Dynamical systems background

### 2.1 Discrete-time maps

A **dynamical system** here is a map $F : \mathbb{R}^M \to \mathbb{R}^M$ iterated as

$$z_{t+1} = F(z_t).$$

Iterating from a seed $z_0$ traces a *trajectory* $\{z_t\}_{t \ge 0}$. The key objects:

**Fixed point.** A state $z^* \in \mathbb{R}^M$ with $F(z^*) = z^*$.
Stability is determined by the eigenvalues of the Jacobian $DF(z^*)$:
- $|\lambda_i| < 1$ for all $i$: **stable** (nearby trajectories converge) — an
  *attractor*.
- $|\lambda_i| > 1$ for all $i$: **unstable** (repeller) — a *repeller*.
- Mixed signs: **saddle** — attracts along some directions, repels along others.

The **stability index** (or *unstable dimension*) is
$$\kappa(z^*) = \#\{i : |\lambda_i| > 1\}.$$
A stable fixed point has $\kappa = 0$; a fully unstable one has $\kappa = M$.
The stability index is a topological invariant: it cannot change under a
homeomorphic reparametrisation of $\mathbb{R}^M$.

**Periodic orbit.** A sequence $z^{(1)}, \ldots, z^{(k)}$ with
$F(z^{(j)}) = z^{(j+1 \bmod k)}$, so $F^k(z^{(1)}) = z^{(1)}$.
The **Floquet multipliers** are the eigenvalues of the *monodromy matrix*
$M_k = DF^k(z^{(1)}) = DF(z^{(k)}) \cdots DF(z^{(1)})$.
For a stable limit cycle, all Floquet multipliers satisfy $|\mu_i| < 1$ except
for one *marginal* multiplier $\mu = 1$ corresponding to the direction along the orbit.

**Torus / quasiperiodic motion.** Two incommensurate oscillations of frequencies
$\omega_1, \omega_2$ superimposed; the trajectory fills the surface of a torus in
state space and never exactly repeats. Characterised by *two* marginal Lyapunov
exponents and zero topological entropy.

**Chaos.** Bounded motion with *sensitive dependence on initial conditions*: nearby
trajectories separate exponentially. The set the trajectory explores is a **strange
attractor**. Characterised by at least one *positive* Lyapunov exponent and positive
topological entropy.

### 2.2 Lyapunov exponents

The **Lyapunov spectrum** $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_M$ measures
the average exponential rate of divergence along each orthogonal direction:

$$\lambda_i = \lim_{T \to \infty} \frac{1}{T} \log \| \delta z_i(T) \|,$$

where $\delta z_i(t)$ is the $i$-th orthogonalised perturbation evolved under the
Jacobian. In practice we compute this via QR iteration:

$$DF(z_t) Q_t = Q_{t+1} R_t, \qquad \lambda_i = \frac{1}{T} \sum_{t=1}^T \log |[R_t]_{ii}|.$$

The *signs* of the Lyapunov spectrum are nearly topological:

| Signs pattern | Dynamics |
|---|---|
| All negative | Stable fixed point |
| One zero, rest negative | Limit cycle |
| Two zeros, rest negative | Torus (quasiperiodic) |
| One positive | Chaos |
| Two positive | Hyperchaos |

The *magnitudes* are rate quantities: they change under smooth reparametrisation or
time rescaling.

The **Kaplan–Yorke dimension** estimates the fractal dimension of the attractor from
the Lyapunov spectrum:

$$d_{\text{KY}} = j + \frac{\sum_{i=1}^{j} \lambda_i}{|\lambda_{j+1}|},$$

where $j = \max\!\left\{k : \sum_{i=1}^{k} \lambda_i \ge 0\right\}$.

### 2.3 Topological conjugacy and gauge freedom

Two systems $F$ and $G$ are **topologically conjugate** if there exists a
homeomorphism $h$ (a continuous bijection with continuous inverse) such that

$$h \circ F = G \circ h.$$

Topological conjugacy preserves: fixed-point counts and stability indices, which
periods exist, topological entropy, Betti numbers of the attractor.

It does **not** preserve: eigenvalue magnitudes, Lyapunov magnitudes, Floquet
multiplier magnitudes, fractal dimension — these are *rate* quantities.

**Smooth** (or *metric*) **conjugacy** additionally requires $h$ to be a $C^1$
diffeomorphism. The set of quantities invariant under smooth conjugacy is smaller
(e.g., the *ratio* of Floquet multipliers is preserved but not their absolute
values). In practice the distinction between topological and smooth conjugacy is
implemented via the `gamma` parameter (§6.1).

**Gauge freedom** arises from the observation model. If observed data
$y = B z + \varepsilon$ for some unknown rotation/scaling matrix $B$, then fitting a
model on $y$ recovers $F$ only up to $B$-conjugacy: two experimenters measuring the
same brain with different electrode placements will fit models $F$ and $\tilde{F} =
B^{-1} \circ F \circ B$. A **gauge-free** signature is invariant to this rotation:
it is computed from objects that are invariant under similarity transforms of the
state space (e.g., eigenvalues of the Jacobian, rather than the Jacobian matrix
itself). The `canonicalize` function in `systems.py` removes the principal-axis
ambiguity before fitting, giving a canonical frame that makes the gauge-bearing
parameters comparable across fits.

---

## 3. The AL-RNN substrate

### 3.1 Model definition

The **Almost-Linear RNN** (AL-RNN, Brenner et al. NeurIPS 2024) is

$$z_t = A\, z_{t-1} + W\, g(z_{t-1}) + h, \qquad g(z)_i = \begin{cases} \mathrm{ReLU}(z_i) & i < P \\ z_i & i \ge P, \end{cases}$$

where $A = \mathrm{diag}(a_1,\ldots,a_M)$ is diagonal, $W \in \mathbb{R}^{M \times M}$,
$h \in \mathbb{R}^M$. Only $P$ of the $M$ units pass through a ReLU; the remaining
$M - P$ units form a *linear core*. A stability-preserving clamp
$z \mapsto \mathrm{clip}(z, -c, c)$ is applied after each step.

### 3.2 Piecewise-linear structure and regions

The ReLU boundaries $\{z : z_i = 0\}$ for $i < P$ partition state space into at
most $2^P$ *regions*, each labelled by an **activation pattern**
$\Omega \in \{0,1\}^P$ where $\Omega_i = \mathbf{1}[z_i > 0]$.

Inside region $\Omega$ the map is a *single affine map* with exact Jacobian

$$W_\Omega = \mathrm{diag}(A) + W D_\Omega, \qquad D_\Omega = \mathrm{diag}(\Omega_1,\ldots,\Omega_P,1,\ldots,1) \in \mathbb{R}^{M \times M}.$$

Two facts make this substrate ideal for signature extraction:

1. **The Jacobian is exact and free.** No numerical differentiation is needed; $W_\Omega$
   is a closed-form matrix computable from the parameters alone.
2. **Trajectories induce a symbolic code.** Label each step by its region pattern; the
   trajectory becomes a sequence of symbols over the alphabet $\{0,1\}^P$, and the
   set of visited regions with their transition counts forms a directed graph.

In practice, trained AL-RNNs visit only a *handful* of regions (typically 2–5) even
though $2^P$ are available. This is what makes the symbolic enumerations tractable.

### 3.3 Clip rate

The clamp $\mathrm{clip}(z, -c, c)$ breaks the affine algebra: when the clamp
engages, $z^* = (I - W_\Omega)^{-1} h$ is no longer a valid fixed-point formula
for that region. The **clip rate** — the fraction of steps at which the clamp
activates — is measured and surfaced in the metadata. A model with high clip rate
(> 10–20%) should be treated with caution; its symbolic analysis may not be
reliable.

---

## 4. Signature blocks

Each block is a labelled descriptor tagged by its equivalence class:
`topological`, `rate`, or `geometry`. Blocks are either a **scalar** vector
(fixed or padded length, compared by $z$-scored Euclidean distance) or a **cloud**
(an unordered set of complex numbers of model-dependent size, compared by
1-Wasserstein on modulus and argument separately).

### 4.1 Equilibrium portrait

**Intuition.** Where does the system stand still, and how stable is it there?

**Definition.** For each visited region $\Omega$, solve

$$z^* = (I - W_\Omega)^{-1} h$$

and test whether $z^*$ lies in region $\Omega$ (i.e. $\Omega_i = \mathbf{1}[z^*_i > 0]$
for $i < P$). If not, the equilibrium is *virtual* — a mathematical artefact of
extending the region's affine map outside its domain — and is discarded. Real
fixed points are counted and their stability indices $\kappa = \#\{i : |\lambda_i(W_\Omega)| > 1\}$
are histogrammed.

**Blocks produced.**
- `eq_count` (topological, scalar): $[n_{\text{fp}},\, n_{\text{flagged}}]$ where
  $n_{\text{flagged}}$ counts near-singular cases with $\mathrm{cond}(I - W_\Omega) > 10^8$.
- `eq_index_hist` (topological, scalar): histogram of stability indices $\in \{0,\ldots,8\}$.
- `eq_spectrum` (rate, cloud): pooled eigenvalues $\lambda(W_\Omega)$ over all real
  fixed points.

**Cost.** One linear solve per visited region — $O(r M^3)$ where $r$ is the number
of visited regions (typically 2–5, at most $2^P$).

**What it captures.** Number of equilibria and their stability type. Distinguishes:
fixed-point attractors ($n_{\text{fp}} \ge 1, \kappa = 0$) from oscillatory systems
($n_{\text{fp}} = 0$ or all unstable). Cannot distinguish two systems with the same
fixed-point count and stability-index histogram but different Floquet structure
(e.g., a limit cycle coexisting with an unstable focus vs a limit cycle coexisting with a
saddle of the same index).

### 4.2 Periodic-orbit spectrum

**Intuition.** What cycles exist, and how stable are they?

**Definition.** A *closed walk of length $k$* in the symbolic transition graph is a
sequence $(\Omega_1,\ldots,\Omega_k)$ with $\Omega_{j+1}$ reachable from $\Omega_j$
in the transition graph and $\Omega_1$ reachable from $\Omega_k$. For such a walk,
the *composite map* is

$$M_w = W_{\Omega_k} \cdots W_{\Omega_1}, \qquad b_w = \sum_{j=1}^k W_{\Omega_k} \cdots W_{\Omega_{j+1}} h.$$

A period-$k$ orbit exists if the *candidate point*

$$z^{(1)} = (I - M_w)^{-1} b_w$$

lies in region $\Omega_1$, and each iterate $z^{(j+1)} = W_{\Omega_j} z^{(j)} + h$
lies in the prescribed region $\Omega_{j+1}$. The **Floquet multipliers** of this
orbit are the eigenvalues of $M_w$.

Only *primitive* closed walks are considered (non-repetitions), up to cyclic
rotation. The search is a depth-first enumeration of admissible walks.

**Blocks produced.**
- `po_count_per_period` (topological, scalar): number of real period-$k$ orbits for
  $k = 1, \ldots, k_{\max}$.
- `po_period_set` (topological, scalar): binary vector indicating which periods exist.
- `po_floquet` (rate, cloud): pooled Floquet multipliers over all found orbits.

**Cost.** $O(r^{k_{\max}} / k_{\max})$ closed-walk candidates (pruned by graph
admissibility), each requiring $O(k_{\max} M^3)$ work. For $r = 3$, $k_{\max} = 6$:
roughly $3^6 / 6 \approx 120$ candidates. For $r = 6$ with a sparse graph: still
manageable. Becomes expensive if $r \ge 5$ and the transition graph is dense;
cap $k_{\max}$ appropriately (6–8 is safe for typical AL-RNN fits).

**What it captures.** Existence and multiplicity of periodic orbits up to period
$k_{\max}$. Distinguishes: a system with no admissible closed walks (fixed point or
degenerate) from one with a period-2 orbit; a limit cycle (one closed walk) from a
system with many periods (chaotic pre-cursor or period-doubling cascade).
Cannot distinguish two systems with the same set of orbit periods but different
orbit *amplitudes* or *shapes* (rate information).

### 4.3 Symbolic-graph spectrum

**Intuition.** How are the visited regions arranged, and how complex is the symbolic
dynamics?

**Definition.** Let $B \in \mathbb{R}^{r \times r}$ be the raw transition-count matrix
($B_{ij}$ = number of times region $j$ followed region $i$) and $A = (B > 0)$ the
Boolean admissibility matrix. Then:

- **Topological entropy**: $h_{\mathrm{top}} = \log \rho(A)$ where
  $\rho(A) = \max_i |\lambda_i(A)|$ is the spectral radius. This is the
  growth rate of the number of admissible symbol words of length $k$
  (which grows as $e^{k h_{\mathrm{top}}}$). A fixed point or limit cycle has
  $h_{\mathrm{top}} = 0$; a chaotic system has $h_{\mathrm{top}} > 0$.

- **Closed-walk counts**: $\mathrm{tr}(A^k)$ for $k = 1,\ldots, k_{\max}$ counts
  the number of length-$k$ symbolic loops, encoding the graph's periodic structure.

- **Normalised Laplacian spectrum**: define the symmetrised adjacency
  $A_s = \max(A, A^\top)$, degree matrix $D = \mathrm{diag}(A_s \mathbf{1})$, and

  $$L = I - D^{-1/2} A_s D^{-1/2}.$$

  The eigenvalues $0 = \mu_1 \le \mu_2 \le \cdots \le \mu_r \le 2$ are invariant
  under node relabelling and lie in $[0, 2]$, making them comparable across models
  with different numbers of visited regions.

- **Number of strongly connected components** (SCCs): how many irreducible subgraphs
  exist.

**Blocks produced.**
- `graph_scalars` (topological, scalar): $[r,\, h_{\mathrm{top}},\, n_{\mathrm{SCC}}]$.
- `graph_closed_walks` (topological, scalar): $[\mathrm{tr}(A),\ldots,\mathrm{tr}(A^{k_{\max}})]$.
- `graph_laplacian` (topological, cloud): eigenvalues of $L$.

**Cost.** $O(r^3 + r k_{\max})$ — trivial for $r \le 8$.

**What it captures.** The symbolic complexity and connectivity of the dynamics.
A fixed point has $r = 1$, $h_{\mathrm{top}} = 0$; a limit cycle has $h_{\mathrm{top}} = 0$
and $\mathrm{tr}(A^k) = 1$ (one closed walk); a chaotic attractor with 2 regions has
$h_{\mathrm{top}} = \log 2$ and $\mathrm{tr}(A^k) = 2^k$. The Laplacian spectrum is a
graph-isomorphism-sensitive fingerprint (two non-isomorphic graphs can have the same
Laplacian spectrum, but this is rare in practice).

Cannot distinguish: two chaotic systems with the same topological entropy but
different branched-manifold templates (e.g., Lorenz vs Rössler — both can achieve
$h_{\mathrm{top}} = \log 2$ with a 2-region model). The attractor topology block (§4.6)
is designed for exactly this.

### 4.4 Generator geometry

**Intuition.** How are the linear pieces *glued together*? Where do the switching
directions point relative to the dominant flow?

**Definition.** The linear core (all ReLU units off) has Jacobian

$$W_0 = \mathrm{diag}(A) + W \, \mathrm{diag}(0,\ldots,0,1,\ldots,1),$$

with spectral radius $\rho_0 = \rho(W_0)$. Activating unit $m$ adds a rank-one update:

$$W_0 + \Delta_m, \qquad \Delta_m = W_{\cdot m} e_m^\top,$$

where $W_{\cdot m}$ is the $m$-th column of $W$. This changes the spectral radius to
$\rho_m = \rho(W_0 + \Delta_m)$.

Let $U \in \mathbb{R}^{M \times r}$ be the leading $r$-dimensional invariant subspace
of $W_0$ (from the $r$ largest-magnitude eigenvalues). The **switching angle** for
unit $m$ is

$$\theta_m = \min_{j} \angle(W_{\cdot m},\, U_j) \in [0, \pi/2],$$

the smallest principal angle between the switching direction and the dominant subspace.
Small $\theta_m$ means unit $m$ switches along the dominant flow; large $\theta_m$
means it switches transversally.

**Blocks produced.**
- `gen_backbone` (rate, cloud): eigenvalues of $W_0$.
- `gen_switch_shift` (rate, scalar): $[\rho_m - \rho_0]_{m < P}$ — how each switch
  changes the spectral radius.
- `gen_switch_angle` (topological, scalar): $[\theta_m]_{m < P}$ — switch alignment
  with dominant subspace.

**Cost.** $O(P M^3)$ — one eigendecomposition per switch.

**What it captures.** Two models with identical topological and rate signatures but
different geometric arrangements of their nonlinear units (one bending the vector
field along the attractor, another bending it across) can be separated here.
Particularly useful for distinguishing Lorenz-like systems (symmetric double-lobe:
the switch angle for the two-lobe bifurcation is transversal to the dominant axis)
from Rössler-like systems (spiral-folded: the switch is more aligned with the flow).

Cannot distinguish: systems where the dominant subspace is numerically ill-defined
(degenerate $W_0$), or systems where the switching geometry is the same but the
number of switches differs.

### 4.5 Lyapunov signature

**Intuition.** Is the system chaotic, oscillatory, or decaying? How fast?

**Definition.** The Lyapunov spectrum is computed by QR iteration along a model free-run,
using the **exact** region Jacobians $W_{\Omega(t)}$ (no numerical estimation needed):

$$Q_{t+1} R_t = W_{\Omega(t)} Q_t, \qquad \hat\lambda_i = \frac{1}{T} \sum_{t=1}^T \log |[R_t]_{ii}|.$$

After warmup of 500 steps, average over $T = 2500$–$3000$ steps.

**Blocks produced.**
- `lyap_signs` (topological, scalar): $[n_+,\, n_0,\, n_-]$ — counts of positive,
  near-zero ($|\lambda| < 10^{-3}$), and negative exponents.
- `lyap_ky` (rate, scalar): $[d_{\text{KY}}]$ — Kaplan–Yorke dimension.
- `lyap_spectrum` (rate, cloud): the full spectrum $\hat\lambda_1,\ldots,\hat\lambda_M$.

**Cost.** $O(T M^3)$ for the QR decompositions — typically 2–3 seconds.

**What it captures.** The coarsest dynamical classification. The sign pattern
$[n_+, n_0, n_-]$ is the single most discriminative block in the artificial attractor
experiments (Layer-2 reliability upweights it 6×). Cannot distinguish: (a) two
chaotic systems with the same $n_+$ but different Kaplan–Yorke dimensions (handled
by `lyap_ky`); (b) a correctly reconstructed Rössler ($n_+ = 1$) from a misread
Rössler that was reconstructed as quasiperiodic ($n_0 = 1$) — this is a
**reconstruction failure**, not an embedding gap.

### 4.6 Attractor topology

**Intuition.** What *shape* is the attractor? Point, loop, torus, branched manifold?

**Definition.** Run the model free for $n = 3000$ steps, subsample to $n_{\text{sub}}$
points, and compute the **persistent homology** (Vietoris–Rips filtration) up to
homological degree `max_dim`. Persistent homology extracts **Betti numbers**:

- $\beta_0$: number of connected components.
- $\beta_1$: number of independent loops (1-cycles).
- $\beta_2$: number of enclosed voids (2-cycles).

For each homological degree $d \in \{0,1,2\}$ we record the count of persistent
features, their total lifetime, and their maximum lifetime. A feature "born" at
filtration scale $\epsilon_b$ and "dying" at $\epsilon_d$ has lifetime
$\epsilon_d - \epsilon_b$; long-lived features are topologically significant,
short-lived ones are noise.

Expected signatures:
| Attractor | $\beta_0$ | $\beta_1$ | $\beta_2$ |
|---|---|---|---|
| Fixed point | 1 | 0 | 0 |
| Limit cycle | 1 | 1 | 0 |
| Torus | 1 | 2 | 1 |
| Rössler (1-lobe) | 1 | 1 | 0 |
| Lorenz (2-lobe) | 1 | 2 | 0 |

**Block produced.** `topo_persistence` (topological, scalar): $[|\text{features}_d|, \sum \text{life}_d, \max \text{life}_d]_{d=0}^{\texttt{max\_dim}}$, length $3(\texttt{max\_dim}+1)$.

**Cost.** The Vietoris–Rips complex has $O(n_{\text{sub}}^{k+1})$ simplices in degree
$k$. Practically:
- `max_dim=1`, $n_{\text{sub}} = 800$: $\approx 5$ s.
- `max_dim=2`, $n_{\text{sub}} = 150$: $\approx 2$ s.
- `max_dim=2`, $n_{\text{sub}} = 400$: $\approx 40$ s.

Use `max_dim=2, ph_n_sub=150` on hardware where this is the right tradeoff
(as in the current experiment); fall back to `max_dim=1` for heavier data pipelines.

**What it captures.** The topological type of the attractor in a coordinate-free way.
This is the only block that can distinguish the Lorenz 2-lobe manifold from Rössler
by $\beta_1 = 2$ vs $\beta_1 = 1$, and a torus from a limit cycle by $\beta_2 = 1$
vs $\beta_2 = 0$. It is also blind to: the *geometric* shape of the loops (a round
cycle vs a flattened ellipse), the *knotting* or *linking numbers* of the loops, and
any topological feature finer than the persistence threshold.

**Limitation: knotting.** Two limit cycles can be topologically equivalent as
abstract manifolds ($\beta_1 = 1$) but *knotted differently* in $\mathbb{R}^3$ —
distinguishable only by knot invariants (Alexander polynomial, knot group). Rössler
can form different knots depending on parameters. Persistent homology (a homotopy
invariant) cannot detect knotting; dedicated knot-invariant computation would be
needed. This is a known gap for distinguishing Rössler-type attractors with different
winding numbers.

### 4.7 Region spectrum (DSA-style coordinate)

**Intuition.** A single global "linear operator" descriptor aggregating all visited
region Jacobians.

**Definition.** Pool the eigenvalues of all visited $W_\Omega$:
$\bigcup_{\Omega \in \text{visited}} \lambda(W_\Omega)$.

**Block produced.** `spectral_region` (rate, cloud).

**Cost.** $O(r M^3)$.

**Relation to DSA.** Dynamical Similarity Analysis (Ostrow et al. NeurIPS 2023)
compares models by optimising a rotation to align their trajectories and then
measuring the resulting operator distance. For normal operators this is equivalent to
comparing eigenvalue spectra via 2-Wasserstein, which is exactly what `spectral_region`
computes — without the expensive per-pair optimisation. DSA is $O(K^2)$ manifold
optimisations; our approach is $O(K)$ extractions plus $O(K^2)$ Wasserstein distances.

**What it captures.** The aggregate spectral fingerprint of the dynamics. Useful as a
rate-level summary when topological blocks are degenerate (single-region collapse).
Cannot distinguish: two systems with identical eigenvalue multisets but different
transition structure (e.g., two regions with the same spectrum but arranged as a
cycle vs as two disconnected SCCs).

### 4.8 Data-side geometry blocks

These blocks are read from the *raw observations* $y \in \mathbb{R}^{T \times d}$
rather than from the trained model. They are present only when `obs_data` is passed
to `extract()`.

**Rotation-invariant spectral signature.** The Welch power spectral density (PSD) of
each channel is summed across channels:

$$S(\omega) = \frac{1}{d} \sum_{c=1}^d P_c(\omega),$$

which equals the trace of the cross-spectral density matrix — invariant under
orthogonal mixing of channels ($y \mapsto Q y$ for orthogonal $Q$). Normalised to
unit area and binned into 6 log-spaced fractional-Nyquist bands, this gives an
oscillatory fingerprint. A centroid and spectral entropy are also computed.

- `geom_spectral_bands` (geometry, scalar): log band powers (length 6).
- `geom_spectral_summary` (geometry, scalar): [centroid, entropy].

**Gauge-bearing per-channel spectral signature.** Per-channel relative band powers:

$$p_{c,b} = \frac{\int_{\omega \in \text{band}_b} P_c(\omega)\, d\omega}{\int_0^{\omega_{\text{Nyq}}} P_c(\omega)\, d\omega}.$$

Not rotation-invariant: which *channel* carries the slow-wave power is itself
informative for real sensors. Crucial for sleep EEG (deep slow-wave sleep has large
delta power on specific channels); harmful for gauge-free synthetic experiments.

- `geom_spectral_perchannel` (geometry, scalar): length $6d$.

**Spatial covariance spectrum.**

$$C = \mathrm{Cov}(y),\quad \nu = \lambda(C)/\mathrm{tr}(C) \in \mathbb{R}^d_{\ge 0}.$$

The normalised eigenvalue spectrum and participation ratio
$\Pi = (\mathrm{tr}\, C)^2 / (\|C\|_F^2)$ describe spatial anisotropy.
This is the *connectome fingerprint* axis: how spatially heterogeneous the signal is.

- `geom_cov_spectrum` (geometry, scalar): normalised eigenvalues (length $d$).
- `geom_effective_dim` (geometry, scalar): $[\Pi]$.

---

## 5. Comparison: distances and commensuration

### 5.1 Per-block distance matrix

Given $K$ models with signatures, we compute a $K \times K$ distance matrix for each
block separately, then combine.

**Scalar blocks.** Stack all models' data into a $K \times L$ matrix $X$. Standardise
each coordinate:

$$\tilde X_{ik} = \frac{X_{ik} - \bar X_k}{\sigma_k + 10^{-9}},$$

then $d_b(i,j) = \|\tilde X_i - \tilde X_j\|_2$. The $z$-scoring puts a count and an
entropy on the same scale before comparing.

**Cloud blocks.** An eigenvalue cloud $\mathcal{C} \subset \mathbb{C}$ is mapped to
two real 1-D distributions of *modulus* $|\lambda|$ and *reflection-folded argument*
$|\arg(\lambda)| \in [0, \pi]$. The latter folds $\lambda$ and $\bar\lambda$ onto the
same value, making the comparison invariant to complex-conjugate pairs (which always
appear together for real matrices). The distance is then

$$d_b(i,j) = W_1(\{|\lambda_k^{(i)}|\}, \{|\lambda_k^{(j)}|\}) + W_1(\{|\arg \lambda_k^{(i)}|\}, \{|\arg \lambda_k^{(j)}|\}),$$

where the **1-Wasserstein distance** (earth-mover distance) between two 1-D empirical
distributions is

$$W_1(\mu,\nu) = \int_0^1 |F_\mu^{-1}(t) - F_\nu^{-1}(t)|\, dt,$$

computed in $O(n \log n)$ by sorting. This handles clouds of *different sizes*
(models of different dimension or number of visited regions), which Euclidean distance
cannot.

**Why Wasserstein?** For normal operators, DSA's canonical metric reduces to exactly
this Wasserstein distance on eigenvalue multisets. The Wasserstein metric is
stable under perturbation (small parameter changes give small signature changes) and
satisfies the triangle inequality, making it a proper metric.

### 5.2 Commensuration (Layer 3)

Different blocks live on incompatible scales (Wasserstein on eigenvalue moduli vs
$z$-scored Euclidean on symbol counts). Before combining, each block distance matrix
is divided by its own *median of non-zero off-diagonal entries*:

$$\hat d_b(i,j) = \frac{d_b(i,j)}{\mathrm{median}_{i' < j',\, d_b > 0}\, d_b(i',j')},$$

so every block contributes on a scale where 1 = "a typical model pair". After this,
the Layer 1–2 weights $w_b$ mean what they say (doubling $w_b$ doubles that block's
contribution).

---

## 6. Weighting framework

The combined distance is

$$D(i,j) = \sqrt{\sum_b w_b\, \hat d_b(i,j)^2},$$

where the weights $w_b$ are the product of three layers.

### 6.1 Layer 1: class weight — the modelling decision

$$w_b^{(1)} = \begin{cases} 1 & \text{class}(b) = \texttt{topological} \\ \gamma & \text{class}(b) = \texttt{rate} \\ g & \text{class}(b) = \texttt{geometry} \end{cases}$$

**This is the most important user-facing knob.** The choice of $\gamma$ defines the
equivalence relation:

- $\gamma = 0$: *topological* clustering — two systems doing the same thing at
  different speeds land in the same cluster. This is the right choice when you want
  "limit cycle" vs "chaos" regardless of frequency.
- $\gamma \approx 1$: *rate-sensitive* clustering — similar to DSA. Distinguishes
  fast and slow oscillators.
- $g = 0$: ignore data-side geometry (model-side only; correct for synthetic data
  where the observation frame is arbitrary).
- $g > 0$: include data-side geometry (correct for real sensors where the observation
  frame is physically meaningful).

The parameter `geom_per_channel` in `extract()` controls whether the gauge-bearing
per-channel block is included; it should be `True` for real EEG/fMRI and `False` for
synthetic data.

### 6.2 Layer 2a: reliability (unsupervised)

Given *replicate groups* — several signatures of the *same* target (retrainings, seeds,
or affine variants) — weight each block by how *stable* it is:

$$w_b^{\text{rel}} = \frac{\mathrm{med}_{\text{all}}(d_b)}{\mathrm{med}_{\text{within}}(d_b) + \epsilon\, \mathrm{med}_{\text{all}}(d_b)},$$

where $\mathrm{med}_{\text{all}}$ averages over all $\binom{K}{2}$ pairs and
$\mathrm{med}_{\text{within}}$ averages only over within-group pairs.

A block is *reliable* if its within-group scatter is small relative to its overall
range. A block that is **constant everywhere** has
$\mathrm{med}_{\text{all}} \approx 0 \Rightarrow w^{\text{rel}} \approx 0$
(automatically zeroed — carries no information). A block that scatters as much within
groups as between them has $w^{\text{rel}} \approx 1/(1+\epsilon) \approx 1$.

**Does not require labels.** The replicate groups can come from multiple seeds or
affine variants of the same system — a natural structure in the reconstruction
pipeline.

**Limitation: coarse dilutes sharp.** If a coarse block and a fine block both have
small within-group scatter (both are locally discriminative), reliability cannot
distinguish them. Mixing them equally dilutes the sharper block. Fisher or silhouette
weighting (§6.3–6.4) fixes this.

Weights are normalised: $\mathbb{E}_b[w_b^{\text{rel}}] = 1$ over informative blocks.

### 6.3 Layer 2b: Fisher weights (supervised)

$$w_b^{\text{Fisher}} = \frac{\bar{a}_b}{\bar{w}_b + \epsilon\, \bar{a}_b},$$

where

$$\bar{a}_b = \frac{1}{|\mathcal{P}_{\neq}|} \sum_{(i,j): y_i \neq y_j} d_b(i,j), \qquad \bar{w}_b = \frac{1}{|\mathcal{P}_{=}|} \sum_{(i,j): y_i = y_j} d_b(i,j)$$

are the mean *across-class* and *within-class* distances under block $b$.

A block that perfectly separates classes ($\bar{w}_b \to 0$, $\bar{a}_b$ large) gets
weight $\to 1/\epsilon$; a block carrying no class signal ($\bar{a}_b \approx \bar{w}_b$)
gets weight $\approx 1/(1+\epsilon)$; a constant block ($\bar{a}_b \approx 0$) gets
weight $\approx 0$.

**Requires labels.** This is the key difference from reliability. Fisher weights can
only be computed when ground-truth class memberships are available (e.g., the true
attractor type in synthetic experiments, or the sleep stage in the EEG experiment).
In production (truly unsupervised), use reliability or silhouette instead.

Compared to reliability, Fisher is *global*: it measures whether a block produces
clean global partitions, not just whether it is locally stable. It correctly identifies
a block that produces large between-class distances and small within-class distances
even if both are large in absolute terms (reliability would up-weight such a block
only if the within-group scatter is small, which might not be the case if classes are
intrinsically noisy).

### 6.4 Layer 2c: silhouette weights (unsupervised, post-hoc)

Given cluster assignments $\hat y$ (from a first-pass clustering), the silhouette of
sample $i$ under block $b$ is

$$s_b(i) = \frac{b_b(i) - a_b(i)}{\max(a_b(i),\, b_b(i))}, \quad s_b(i) \in [-1, 1],$$

where $a_b(i)$ is the mean distance from $i$ to all other samples in its cluster
(compactness) and $b_b(i)$ is the mean distance to the nearest out-cluster
(separation). A positive silhouette means the sample is well-placed; negative means
it would fit better in a different cluster.

The block weight is

$$w_b^{\text{sil}} = \max\!\left(0,\, \frac{1}{K}\sum_{i=1}^K s_b(i)\right),$$

zeroing blocks whose geometry *contradicts* the cluster assignment. Normalised to unit
mean over positive-weight blocks.

**Does not require ground-truth labels**, but requires an initial cluster assignment
(which can itself come from a reliability-based run). This makes it a *two-pass*
unsupervised method: run once with reliability weights, cluster, then reweight by
silhouette and cluster again.

**Relation to Fisher.** When the initial clustering is perfect, silhouette weights
approximate Fisher weights. When the initial clustering is noisy, silhouette weights
can propagate the noise (the block that best supports a *wrong* cluster gets
up-weighted). Fisher, being supervised, is immune to this.

### 6.5 Summary of weight combinations

| Weight | Labels? | Corrects for | Best when |
|---|---|---|---|
| Uniform | No | — | Baseline / exploration |
| Reliability | No (replicates) | Noisy/degenerate blocks | Replicate groups available |
| Fisher | Yes | Coarse-dilutes-sharp, noise | Labels known |
| Silhouette | No (clusters) | Coarse-dilutes-sharp | Labels unknown; run after reliability pass |

**User-facing knobs summary:**

| Knob | Location | Effect |
|---|---|---|
| `gamma` | `combine_distance` | Rate weight: 0 = topological, ~1 = DSA-like |
| `geom` | `combine_distance` | Data-side geometry weight |
| `reliability` | `combine_distance` | Layer-2 unsupervised weights from replicate groups |
| `weights` | `combine_distance` | Override to Fisher/silhouette/manual |
| `geom_per_channel` | `extract` | Include gauge-bearing per-channel block |
| `ph_max_dim` | `extract` | PH degree cap: 2 includes voids, 1 for speed |
| `ph_n_sub` | `extract` | PH subsample size (tradeoff: 150 ≈ 2s, 400 ≈ 40s) |
| `k_max` | `extract` | Periodic-orbit search depth (6–8 safe; cap for dense graphs) |
| `lyap_n` | `extract` | Lyapunov averaging length (3000 for clean synthetic) |
| `n_clusters`, linkage | `cluster` | Clustering granularity and algorithm |

---

## 7. Necessity and sufficiency

### 7.1 What the signatures can separate

The collective signature, used at $\gamma = 0$ (topological blocks only), can in
principle distinguish the following topological classes:

| Class pair | Key discriminating blocks |
|---|---|
| Fixed point vs limit cycle | `lyap_signs` ($n_0 = 0$ vs $1$), `eq_count` |
| Limit cycle vs torus | `lyap_signs` ($n_0 = 1$ vs $2$), `topo_persistence` ($\beta_1 = 1$ vs $2$, $\beta_2 = 0$ vs $1$) |
| Torus vs chaos | `lyap_signs` ($n_+ = 0$ vs $\ge 1$), `graph_scalars` ($h_{\mathrm{top}} = 0$ vs $> 0$) |
| Rossler vs Lorenz (both chaotic) | `topo_persistence` ($\beta_1 = 1$ vs $2$), `generator_geometry` (switch angles), `graph_closed_walks` |
| Limit cycle vs chaos | `lyap_signs`, `graph_scalars`, `po_count_per_period` |

*Necessary condition for any pair to be separated*: the training must have
faithfully reconstructed the attractor. A model that collapsed (all Lyapunov negative,
$n_{\text{regions}} = 1$, high clip rate) carries no dynamical information;
two collapsed models have identical degenerate signatures regardless of what their
training data was. The diagnostics `clip_rate`, `n_regions`, and `D_stsp`
(state-space divergence) identify failed reconstructions.

### 7.2 What the signatures cannot distinguish

**Knotted attractors.** Two limit cycles that are topologically identical as abstract
1-manifolds but knotted differently in the ambient 3-space (e.g., a trivial loop vs a
trefoil knot) have the same Betti numbers, same Lyapunov signs, and same symbolic
structure. Distinguishing them requires knot invariants (Alexander polynomial,
writhe). This matters for Rössler attractors with different winding numbers.

**Subtler template differences.** The branched-manifold *template* theory (Gilmore,
Lefranc) classifies chaotic attractors more finely than homology: e.g., two attractors
on the Smale horseshoe template vs the Lorenz template have different *linking numbers*
between periodic orbits. These require computing linking numbers of the found periodic
orbits — computationally feasible but not yet implemented.

**Sensitive dependence on quantitative parameters.** At $\gamma = 0$, two limit cycles
with very different periods are in the same topological class. Only at $\gamma > 0$
does the rate channel separate them. For neuroscience (where 4 Hz delta vs 12 Hz
spindles are distinct clinically), setting $\gamma > 0$ is appropriate; for
schema-level clustering ($\gamma = 0$) they merge, which may or may not be desired.

**Stochastic vs deterministic.** A stochastic process driven by noise is not a
deterministic dynamical system; its trajectory has no attractor in the standard sense.
The AL-RNN is forced to fit a deterministic approximation, which typically collapses
(high clip rate, $n_{\text{regions}} = 1$). This is the fundamental ceiling for
real EEG/fMRI data (§6.2 of `signature_findings.md`).

### 7.3 Possible extensions

- **Linking numbers of periodic orbits**: compute from the found orbit set in §4.2
  using the signed crossing number. Distinguishes Rössler winding variants.
- **Topological template**: fit a branched-manifold template to the orbit structure
  using the Birman–Williams projection.
- **Input-driven reconstruction**: replace the autonomous AL-RNN with a stimulus-driven
  PLRNN for real neural data, where the signal is partially driven by inputs rather
  than purely autonomous.
- **Full β₂ with larger n_sub**: `ph_max_dim=2, ph_n_sub=400` gives meaningful void
  signatures at ~40s per model; feasible for batch experiments on A6000-class hardware.
- **Spectral sequence / persistent homology with coefficients**: distinguishes
  non-orientable manifolds (Möbius band vs cylinder) if they appear.

---

## Appendix: Implementation sketch

```
extract(model, seed_obs, ...)
  ├─ enumerate_visited_regions(n=5000)     → rd   [§3.2]
  ├─ equilibrium_portrait(rd)              [§4.1]
  ├─ periodic_orbits(rd, k_max=8)          [§4.2]
  ├─ symbolic_graph(rd, k_max=8)           [§4.3]
  ├─ generator_geometry(model)             [§4.4]
  ├─ lyapunov_signature(model, n=3000)     [§4.5]
  ├─ attractor_topology(model, max_dim=2)  [§4.6]
  ├─ region_spectrum(rd)                   [§4.7]
  └─ [if obs_data:] spectral_signature(obs_data)     [§4.8]
                     spatial_signature(obs_data)
                     spectral_signature_perchannel(obs_data)

combine_distance(sigs, gamma, reliability)
  for each block b:
    D_b = block_distance_matrix(sigs, b)   [§5.1]
    scale_b = median(D_b)                  [§5.2]
    w_b = class_weight(b) × reliability(b) [§6.1, 6.2]
  return sqrt(Σ_b w_b (D_b/scale_b)²)

cluster(D, n_clusters, method="ward")
  → scipy hierarchical clustering on precomputed D
```
