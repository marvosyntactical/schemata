# Embedding DSR Models: Signatures for Clustering Reconstructed Dynamics

The goal is a map from a trained dynamical-systems-reconstruction (DSR) model to a fixed coordinate vector, such that models implementing structurally similar dynamics land close together. This document sets up the two reference points we already have (the AL-RNN as the model class, DSA as the comparison method), states what we actually want, shows where the pair falls short, and lists the topological, symbolic, and spectral quantities we can compute instead, with implementation notes.

## 1. The AL-RNN

The Almost-Linear RNN (Brenner, Hemmer, Monfared, Durstewitz, NeurIPS 2024) is a piecewise-linear RNN in which only a few units carry a nonlinearity. The latent recursion is

$$
z_t = A\,z_{t-1} + W\,\Phi^*(z_{t-1}) + h,
$$

with $z_t \in \mathbb{R}^M$, $A$ diagonal (linear self-connections), $W$ the full coupling, $h$ a bias, and

$$
\Phi^*(z) = \big(z_1,\dots,z_{M-P},\ \max(0,z_{M-P+1}),\dots,\max(0,z_M)\big)^\top .
$$

Only $P \ll M$ units pass through a ReLU; the remaining $M-P$ are linear. Writing the ReLU slope as a state-dependent diagonal gate $D_\Omega$ (entry $1$ where a unit is active, $0$ otherwise) turns the map into an explicitly piecewise-affine system,

$$
z_{t+1} = W_\Omega\, z_t + h, \qquad W_\Omega = A + W D_\Omega .
$$

The gate has $2^P$ configurations, so state space splits into at most $2^P$ subregions $U_e$ separated by the switching manifolds $\{z_m = 0\}$ of the nonlinear units. Inside each subregion the dynamics are a single affine map with constant Jacobian $W_\Omega$. Two facts make this class convenient as a substrate for comparison:

- The Jacobians are available in closed form. There is no need to estimate a linearization from data; $W_\Omega$ is read off from $A$, $W$, and the active set.
- The subregions induce a symbolic coding. Label each region with a symbol, and a trajectory becomes a symbol sequence; the realized region-to-region transitions form a directed graph that summarizes the coarse dynamics.

The reported empirical point is that trained AL-RNNs occupy very few subregions (three for Lorenz-63, two for Rössler), recovering the handcrafted minimal piecewise-linear forms of those attractors. The parsimony is what makes the exact enumerations below tractable: searching over realized regions is cheap when only a handful are visited.

## 2. DSA, and the pieces it is built from

Dynamical Similarity Analysis (Ostrow, Eisen, Kozachkov, Fiete, NeurIPS 2023) compares two systems by the linear operators that govern their delay-embedded trajectories. Three ingredients.

### 2.1 Time-delay embedding

Given an observable time series $\{x_t\}$ (possibly scalar), the delay map sends each time point to a window of its own past:

$$
x_t \ \longmapsto\ \big(x_t,\ x_{t-\tau},\ x_{t-2\tau},\dots,x_{t-(d-1)\tau}\big) \in \mathbb{R}^d .
$$

Takens' embedding theorem says that for a generic observable and delay, and $d$ larger than twice the box dimension of the attractor, this map is an embedding: the reconstructed point cloud is diffeomorphic to the true attractor. Partial observation is recovered by looking at the past. This is the same mechanism the AL-RNN's linear units implement internally, as a learned linear filter bank over past nonlinear activations.

### 2.2 The Hankel matrix

Stacking delay windows as columns gives a Hankel matrix, constant along anti-diagonals:

$$
\mathcal{H} =
\begin{pmatrix}
x_1 & x_2 & \cdots & x_{m}\\
x_2 & x_3 & \cdots & x_{m+1}\\
\vdots & & & \vdots\\
x_d & x_{d+1} & \cdots & x_{m+d-1}
\end{pmatrix}.
$$

Rows index delay, columns index time. Its singular value decomposition $\mathcal{H} = U \Sigma V^\top$ produces eigen-time-delay coordinates (the columns of $V$) in which the dynamics are close to linear. This is the Hankel Alternative View of Koopman (HAVOK).

### 2.3 The Koopman operator and the DMD fit

Koopman theory replaces a nonlinear map on states with a linear operator acting on observables, at the cost of that operator being infinite-dimensional. Dynamic Mode Decomposition (DMD) takes a finite truncation: fit a fixed matrix $A$ by reduced-rank least squares so that the dominant delay coordinates advance linearly,

$$
v_{t+1} \approx A\, v_t , \qquad A = \arg\min_{A}\ \textstyle\sum_t \| v_{t+1} - A v_t \|^2 .
$$

With enough delays the linear $A$ captures global nonlinear behavior, since the nonlinearity has been absorbed into the embedding. This $A$ is the per-model featurization. It depends on two hyperparameters: the number of delays and the truncation rank.

### 2.4 Procrustes over vector fields

A dynamics matrix is not a point cloud, so the comparison cannot use ordinary orthogonal Procrustes. Under a change of basis $z \mapsto Qz$, a vector field transforms by conjugation, $A \mapsto Q A Q^\top$, not by left multiplication. DSA therefore defines

$$
d_{\text{DSA}}(A_x, A_y) = \min_{Q \in O(n)} \big\| A_x - Q A_y Q^\top \big\|_F ,
$$

the Procrustes Analysis over Vector Fields. Ordinary Procrustes has a closed form via an SVD; this one does not, because the term $Q A_y Q^\top$ is quadratic in $Q$. It is solved as an optimization on the orthogonal group: parameterize $Q = \exp(S)$ with $S$ skew-symmetric, or use a Cayley transform, and run gradient descent on the manifold. The result is a proper metric. For normal operators it reduces to the $2$-Wasserstein distance between the eigenvalue spectra, so in that regime DSA is exactly spectral; for non-normal operators it retains the Schur structure (transient amplification) as well.

## 3. What we want

For embedding DSR models into a space where structurally similar systems cluster, and for treating cluster centers as schemata (canonical dynamical motifs), the requirements are:

1. **Cheap and per-model.** Each model should map to a coordinate vector in $O(N)$ work, with a closed-form distance afterward. No per-pair optimization. DSA fails this: every comparison runs a Procrustes solve on $O(n)$, so a distance matrix over $K$ models costs $K^2$ manifold optimizations.

2. **Clusters by structural similarity.** Closeness should track topological or orbital equivalence of the dynamics: same number and type of invariant sets, same recurrence structure, same complexity. It should be insensitive to coordinate geometry and, to the extent we want topological classes, insensitive to exact rates.

3. **Interpretable coordinates.** For the schema reading to mean anything, axes should correspond to identifiable dynamical features (count of unstable spirals, presence of a saddle, a loop in the attractor, a level of symbolic complexity), not opaque latent dimensions.

## 4. Why AL-RNN and DSA are not yet enough

The AL-RNN is a good model class and a poor comparison method on its own. It hands us closed-form Jacobians and a symbolic graph, but the headline summary it offers, the count $2^P$, is a single integer that says nothing about how the regions are arranged or what dynamics live in them. Two models with $2^P = 8$ can have completely different transition graphs, different numbers of fixed points, and different attractor topology.

DSA is a comparison method and a coarse one for this purpose. It compares a single global linear operator up to rotation, which produces two cross-cutting failures relative to topological equivalence:

- **Over-discrimination on rate.** Eigenvalue moduli and arguments are rates and frequencies. Topological conjugacy does not preserve them; it preserves only the count of eigenvalues inside versus outside the unit circle (the stable and unstable dimensions). Two conjugate systems running at different speeds get a nonzero DSA distance, and the same model sampled at a different step size moves in DSA space unless normalized. DSA also fails the triangle inequality across operators of different dimension, so comparing AL-RNNs of different $M$ is already unsound.

- **Under-discrimination on nonlinear structure.** One operator cannot encode multistability and basin structure (it is fit on data from a single attractor), the number and stability index of fixed points, the symbolic arrangement of regions, topological entropy, or the periodic-orbit skeleton of a chaotic attractor. Two limit cycles of equal period but different shape have the same Koopman frequencies and are identical to DSA. It is also blind to noise level.

So DSA measures rates and frequencies of one linearization, invariant to geometry, variant to timescale. That is a legitimate notion of similarity for "same computation, different geometry," and the wrong default if the target is structural or topological equivalence. The combination we want exploits the AL-RNN's closed form to compute the invariants DSA cannot, and keeps a DSA-style operator as one block among several rather than the whole metric.

## 5. The signature menu

Group the computable quantities by the axis each one covers. The AL-RNN's exact Jacobians make most of these algebraic rather than estimated.

### 5.1 Equilibrium portrait (local spectra, topological counts)

Within region $\Omega$ the affine map has a candidate fixed point

$$
z^*_\Omega = (I - W_\Omega)^{-1} h ,
$$

which is real only if $z^*_\Omega$ actually lies in $U_\Omega$, that is, if the sign pattern of its nonlinear coordinates matches $\Omega$. Otherwise it is virtual and discarded. For each real fixed point, the eigenvalues of $W_\Omega$ give the stability index $\#\{\,|\lambda| > 1\,\}$ (the unstable dimension).

Signature: the number of fixed points, the histogram of stability indices, and the pooled eigenvalue distribution. The index counts are topological-conjugacy invariants; the eigenvalues themselves are only smooth-conjugacy invariants, so keep them in a separate block and label them as rate information.

Implementation. Enumerate only the regions a free-running trajectory actually visits, collected as the set of distinct sign patterns; this replaces the $2^P$ worst case with the handful the model uses. Per region: `np.linalg.solve(I - W_Omega, h)`, check the sign pattern, then `np.linalg.eigvals(W_Omega)`. Microseconds each. Watch for near-singular $I - W_\Omega$ when an eigenvalue approaches $1$ (the non-hyperbolic case the paper sets aside); flag rather than invert blindly.

### 5.2 Periodic-orbit spectrum (cycles, Floquet multipliers)

A length-$k$ symbol word $w = (\Omega_1,\dots,\Omega_k)$ corresponds to the composed affine map $M_w = W_{\Omega_k}\cdots W_{\Omega_1}$ with accumulated offset $b_w$. Its candidate cycle point is $z^* = (I - M_w)^{-1} b_w$, valid only if the $k$ successive iterates each fall in their prescribed regions. The Floquet multipliers are the eigenvalues of $M_w$.

Signature: which periods exist up to some $k_{\max}$, the count per period, and the pooled multiplier distribution. The set of admissible periods is closely tied to topological entropy and is a strong fingerprint of a chaotic attractor's unstable-orbit skeleton.

Implementation. Restrict the word search to admissible walks in the transition graph (Section 5.3), which prunes the search drastically, and cap $k$. Matrix products and eigendecompositions of small matrices. The validity check (iterates staying in their regions) is the expensive part; cache visited regions.

### 5.3 Symbolic-graph spectrum (the arrangement of the regions)

This is the direct replacement for the scalar $2^P$. Build the directed transition graph $G = (V, E)$ on realized regions, with adjacency matrix $B$, by running the model and recording region-to-region transitions. From $B$:

- Topological entropy $h_{\text{top}} = \log \rho(B)$, where $\rho$ is the spectral radius; the growth rate of admissible words, one scalar.
- The normalized Laplacian spectrum or its spectral density, a size-comparable descriptor of connectivity that separates a ring from a tree from a fully connected graph at equal node count.
- Closed-walk counts $\operatorname{tr}(B^k)$, the symbolic period-$k$ orbit counts.
- The number of strongly connected components and recurrent classes; the spectral gap of the frequency-weighted Laplacian for metastability.

Implementation. Record transitions during a long free run; build $B$ with `scipy.sparse`. Use `scipy.sparse.linalg.eigs` for $\rho(B)$ and `networkx` for components and Laplacian spectra. Two cautions. First, the graph depends on trajectory length and sampling, so fix a generation protocol and use long runs. Second, the entropy and graph quantities are exact only if the switching-manifold partition is Markov; if it is merely a topological partition the graph over-approximates the dynamics, and $h_{\text{top}}$ is an upper bound. For cross-model comparison, normalize spectra to be independent of the number of realized regions (use the normalized Laplacian, whose eigenvalues sit in $[0,2]$, or a spectral density).

### 5.4 Generator geometry (how the linear pieces are glued)

Finer than the graph. Every region Jacobian is generated from a shared base and rank-one updates,

$$
W_\Omega = A + \sum_{m \,\text{active}} W_{:,m}\, e_m^\top ,
$$

so crossing the single manifold $\{z_m = 0\}$ changes the Jacobian by the rank-one term $W_{:,m} e_m^\top$. The whole arrangement of $2^P$ linear maps is encoded by $A$ together with the $P$ columns $\{W_{:,m}\}$ for the nonlinear units. Useful pair-invariants:

- The eigenvalues of $A$, the linear backbone the linear units carry (oscillatory and slow modes).
- For each nonlinear unit $m$, the eigenvalue shift induced by its activation, that is, the spectrum of $A + W_{:,m} e_m^\top$ relative to $A$.
- The principal angles between $\operatorname{span}(W_{:,m})$ and the invariant subspaces of $A$, measuring how each switch aligns with the backbone modes.

This separates two AL-RNNs that share a transition graph but bend the vector field differently across the manifolds.

Implementation. `np.linalg.eigvals(A)`; eigenvalues of the rank-one updates; `scipy.linalg.subspace_angles` for principal angles. The constraint that matters: $A$ and $W$ co-transform under a latent basis change $z \mapsto Tz$, so only invariants of the pair are meaningful. Use spectra and angles, never raw matrix entries. Handle eigenvalue degeneracies and complex-conjugate pairs consistently.

### 5.5 Ergodic and Lyapunov signature (global nonlinear rates)

The maximal exponent and the full spectrum come from a QR iteration along a generated trajectory. Because the per-step Jacobian is exactly the region matrix $W_{\Omega(t)}$, no automatic differentiation is needed:

$$
Q_t R_t = W_{\Omega(t)}\, Q_{t-1}, \qquad
\lambda_i = \lim_{T\to\infty} \frac{1}{T}\sum_{t=1}^{T} \log \big| R_{t,ii} \big| .
$$

Add the Kaplan-Yorke dimension $D_{KY} = j + \frac{\sum_{i\le j}\lambda_i}{|\lambda_{j+1}|}$, where $j$ is the largest index with a nonnegative partial sum. Signs and dimension are close to topological; magnitudes are rates. These are the genuine nonlinear analogues of DSA's linear eigenvalues, computed on the attractor.

Implementation. Free-run the model, grab $W_{\Omega(t)}$ at each step (the sign pattern is already computed during generation), and `np.linalg.qr` the propagated frame. Discard a transient, then average over a long run. The region boundaries are non-differentiable on a measure-zero set; ignore crossings or perturb off them.

### 5.6 Attractor topology (geometry-aware, the DSA blind spot)

Persistent homology of the generated point cloud gives the Betti numbers $\beta_0, \beta_1, \beta_2$: connected components, loops, and voids. A point attractor shows $\beta_0$ only; a limit cycle adds one $\beta_1$ loop; a torus has two $\beta_1$ and one $\beta_2$; a Lorenz-type branched manifold has its own signature. This is the attractor shape DSA is invariant to.

Implementation. `ripser` or `gudhi` on a subsample of a few thousand points from a free run, capping homology dimension at two. Use either the latent trajectory or a delay embedding of the observable. Cost grows quickly with point count and homology dimension, so subsample, and threshold by persistence (bar lifetime) to suppress noise. Reduce each diagram to a fixed-length vector (Betti curve, persistence image, or persistence statistics) so it fits the embedding.

### 5.7 Optional spectral block

Keep a DSA-style coordinate so nothing it does capture is lost: the pooled region-Jacobian eigenvalues, or a DMD fit of the generated trajectory. It is now one block among several rather than the metric itself.

## 6. Assembly and clustering

Concatenate the blocks into a per-model vector. The blocks are heterogeneous (scalars, eigenvalue clouds, persistence diagrams), so use a per-block distance and combine:

- Scalars ($h_{\text{top}}$, fixed-point count, $D_{KY}$, Betti numbers): standardize, then Euclidean.
- Eigenvalue and Floquet clouds: $2$-Wasserstein between spectra, the same choice DSA reduces to in the normal case.
- Persistence diagrams: Wasserstein between diagrams, or Euclidean on persistence images.

Combine block distances with explicit weights, then cluster with a method that takes a distance matrix (hierarchical, HDBSCAN, spectral). Use MDS only for visualization. The whole pipeline is $O(K)$ featurization plus $O(K^2)$ closed-form distances, with no per-pair optimization.

Use this as a screen, not a verdict. Reserve DFORM (graded diffeomorphic conjugacy) or DSA as a second-tier, pairwise check inside a candidate cluster, where the count of pairs is small.

## 7. Ceilings to design around

Three limits are structural, and ignoring them produces clusters that look clean and mean little.

- **No finite invariant set is complete for topological conjugacy.** The classification problem is, in general, undecidable. The embedding screens (conjugate models match on conjugacy-invariant coordinates), it does not certify (matching everything we compute is necessary, never sufficient). Two non-conjugate systems can collide.

- **The choice of invariants is the choice of equivalence relation.** Betti numbers, topological entropy, fixed-point indices, the graph up to relabeling, and Lyapunov signs are topological invariants. Eigenvalue and Lyapunov magnitudes and the spectrum of $A$ are rate quantities that vary within a conjugacy class. Mixing them lets rate differences smear topological classes, the same over-discrimination that afflicts DSA. If the target is topological clustering, weight the topological block and carry rate on a separate, labeled axis. There is no neutral weighting to discover; this is the decision.

- **Symbolic signatures inherit the Markov problem.** Entropy from $\rho(B)$ and the graph spectrum are exact only when the partition is Markov, ideally generating. Otherwise they describe an over-approximation, and entropy is a bound. Verify the partition before trusting these on chaotic models, or report them as bounds.

Two operational requirements follow. Normalize anything dimensionful to a common time unit and control for $M$ and $P$, or the embedding reproduces DSA's cross-dimension failure. And expect estimation noise in the Lyapunov spectrum, the persistence diagrams, and the real-versus-virtual fixed-point test near non-hyperbolic points; these are the coordinates most likely to jitter between retrainings of the same target.

## 8. A schema-oriented alternative

If the schema reading is the priority, an alternative to generic invariants is to embed each model by its graded conjugacy distance to a fixed bank of normal-form prototypes (saddle, Hopf, bistable switch, Lorenz template, and so on). Each coordinate is then how strongly one canonical motif is present, which is interpretable by construction and costs $O(K \times \#\text{prototypes})$ rather than $O(K^2)$. This is the Smooth Prototype Equivalences idea (Friedman et al., 2025) and the line of dynamical-archetype work. The tradeoff is expressiveness: the embedding sees only what the prototype bank contains, so the bank becomes the modeling choice that carries the weight.

## References

- Manuel Brenner, Christoph Hemmer, Zahra Monfared, Daniel Durstewitz. *Almost-Linear RNNs Yield Highly Interpretable Symbolic Codes in Dynamical Systems Reconstruction.* NeurIPS 2024. https://arxiv.org/abs/2410.14240
- Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete. *Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis.* NeurIPS 2023. https://arxiv.org/abs/2306.10168
- Ruiqi Chen, Giacomo Vedovati, Todd Braver, ShiNung Ching. *DFORM: Diffeomorphic Vector Field Alignment for Assessing Dynamics Across Learned Models.* 2024. https://arxiv.org/abs/2402.09735
- Roy Friedman, Noa Moriel, Matthew Ricci, Guy Pelc, Yair Weiss, Mor Nitzan. *Characterizing Nonlinear Dynamics via Smooth Prototype Equivalences.* 2025. https://arxiv.org/abs/2503.10336
- Steven Brunton, Bingni Brunton, Joshua Proctor, Eurika Kaiser, J. Nathan Kutz. *Chaos as an Intermittently Forced Linear System (HAVOK).* Nature Communications, 2017. https://www.nature.com/articles/s41467-017-00030-8