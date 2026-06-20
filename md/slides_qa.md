# Slides Q&A

---

## 1. The 5 signatures in detail

---

**(a) $|\lambda(A_\text{lin})|$ — linear-core eigenvalue magnitudes. Category: geometric.**

**What it captures.** The AL-RNN has $M-P$ units that are always linear ($g(z)_i = z_i$ for $i \geq P$). Their dynamics are governed by the sub-block $B_\text{lin} = \text{diag}(A_{P:}) + W_{P:,P:}$, whose eigenvalue magnitudes encode the Koopman-like linear flow: how fast the system contracts or expands along each invariant direction, and whether eigenmodes are real (monotone) or complex (oscillatory). This is a **spectral geometry** property — it characterises the shape and stability of the linear part of the attractor without reference to any specific trajectory. Under smooth invertible coordinate changes the eigenvalue magnitudes change in general, making this gauge-dependent; but for a fixed latent parameterisation it gives a stable, low-dimensional fingerprint of the linear skeleton.

**Why geometric, not topological.** Topological invariants are preserved under all homeomorphisms; eigenvalue magnitudes are not (they depend on the metric structure of the dynamics). They are closer to Riemannian/spectral geometry: properties of a specific representation of the dynamics, not the underlying topology.

**Cost and accuracy.** One $M\times M$ matrix product ($W_B W_A$), then a $10\times10$ eigendecomposition. Essentially free. Accuracy: exact up to floating point. Yields $M-P = 10$ values.

---

**(b) $\bar{s}_i$ — mean activation rate of ReLU unit $i$. Category: rate.**

$$\bar{s}_i = \frac{1}{T}\sum_t \mathbb{1}[z_{i,t}>0]$$

**What it captures.** The time-average probability that nonlinear unit $i$ is in its active (linear) state versus its inactive (zero) state. Equivalently, it measures the **occupation measure** of the attractor in the symbolic (activation-pattern) space, projected onto a single unit. For Lorenz (which collapses to a single activation pattern under GTF-BPTT): all $\bar{s}_i \approx 0.88$ — the system spends almost all time with units active. For torus or VdP (oscillators that cycle through multiple patterns): values spread across $(0,1)$. This is a **rate** property — a temporal average over the trajectory, not a topological or shape property.

**Cost and accuracy.** Binary indicator reads during the free-run — essentially free on top of the rollout. Converges by the law of large numbers at $T\sim2000$, but initial-condition dependent; rare attractors near the boundary of two activation patterns may give inconsistent estimates. Yields $P = 6$ values.

---

**(c) $n_\text{regions}$ — number of distinct activation patterns. Category: topological.**

At each step, form the binary vector $D_\Omega = (\mathbb{1}[z_{1,t}>0],\ldots,\mathbb{1}[z_{P,t}>0]) \in \{0,1\}^P$. Count the number of distinct patterns visited.

**What it captures.** The piecewise-linear structure of the AL-RNN partitions latent space into up to $2^P$ linear regions, each with a distinct Jacobian. The number of regions visited by the attractor is a coarse measure of the **topological complexity** of the attractor's embedding in the piecewise-linear partition — roughly, how many distinct "faces" of the piecewise-linear map the trajectory uses. A fixed point or very simple limit cycle might use 1 region; a complex limit cycle or chaotic attractor uses many. This is **topological** in the sense that it is invariant under continuous deformations that do not move the trajectory across region boundaries (i.e., it is a property of the symbolic dynamics, not the metric geometry).

**Limitation.** Rare regions (visited infrequently) may be missed with finite $T$; not differentiable; sensitive to trajectory initialisation. Yields 1 integer.

**Cost.** Same rollout, plus a set-cardinality operation. Cheap.

---

**(d) $h_\text{top} = \log\rho(T)$ — topological entropy. Category: topological.**

Build the empirical transition matrix $T_{ij}$ = fraction of steps where activation pattern $i$ is followed by pattern $j$ (a $n_\text{regions}\times n_\text{regions}$ stochastic matrix). Take the log spectral radius:
$$h_\text{top} = \log \rho(T).$$

**What it captures.** Topological entropy is a classical dynamical-systems invariant measuring the **exponential growth rate of the number of distinguishable symbolic trajectories** as their length increases. Formally it is the topological entropy of the symbolic dynamical system induced by the activation-pattern sequence. It is:
- Zero for a fixed point (one symbol, no transitions).
- Low for a simple limit cycle (few transitions, transition matrix has small $\rho$).
- High for chaos (many distinct trajectories, large $\rho$).

This is a bona fide **topological invariant**: it does not depend on the specific metric, only on the combinatorial structure of the symbolic transition graph. It is preserved under topological conjugacy of the symbolic dynamics.

**Relation to $n_\text{regions}$.** $n_\text{regions}$ counts vertices of the symbolic graph; $h_\text{top}$ counts the complexity of paths through it. Two systems with the same $n_\text{regions}$ can have very different $h_\text{top}$ (e.g., a cycle visits $k$ regions with $h_\text{top}=0$; a full shift on $k$ symbols has $h_\text{top}=\log k$).

**Cost.** Spectral radius of a small matrix ($n_\text{regions}\times n_\text{regions}$, typically $\leq 20\times20$). Cheap. Accuracy: approximates true topological entropy; consistent as $T\to\infty$ but sensitive to undersampled transitions for rare symbolic sequences. Yields 1 scalar.

---

**(e) $\lambda_\text{max}$ — largest Lyapunov exponent. Category: rate.**

QR iteration on exact piecewise-linear Jacobians: at each step
$$J_t = \text{diag}(A) + W\cdot\text{diag}(g'(z_t)), \qquad g'(z)_i = \mathbb{1}[z_i>0] \text{ for } i<P, \; 1 \text{ otherwise.}$$
Propagate a unit perturbation $\delta z$ forward as $\delta z \leftarrow J_t \delta z / \|J_t \delta z\|$ and accumulate $\lambda_\text{max} = \frac{1}{T}\sum_t \log\|J_t \delta z_t\|$.

**What it captures.** The exponential rate of divergence (or convergence) of nearby trajectories — the fundamental dynamical characterisation of the attractor type:
- $\lambda_\text{max} > 0$: chaotic (Lorenz). Nearby trajectories diverge exponentially.
- $\lambda_\text{max} = 0$: neutral (limit cycle, torus). Nearby trajectories neither converge nor diverge along the neutral direction.
- $\lambda_\text{max} < 0$: contracting (stable fixed point). Under GTF-BPTT, Lorenz models often collapse to linear regime giving $\lambda_\text{max} \approx -0.03$.

This is a **rate** property: it measures an exponential growth rate, which is a temporal average over the trajectory, analogous to a reaction rate or a spectral gap. Unlike topological entropy (which counts combinatorial paths), $\lambda_\text{max}$ measures metric divergence. The *sign* of $\lambda_\text{max}$ is topologically meaningful (it determines the attractor type), but the *value* depends on the metric and the specific parameterisation.

**Cost.** $O(T \cdot M^3)$ — one Jacobian-vector product and norm per step. For $M=16$, $T=2000$: the most expensive of the five, but still fast in practice (seconds on CPU). Accuracy: converges to the true MLE by Oseledets's theorem; at $T=2000\approx90$ Lyapunov times for Lorenz, highly accurate. Yields 1 scalar.

---

## 2. $\theta^*_i$ in EWC

$\theta^*_i$ is the parameter vector saved after gradient descent converged on the previous task — you simply copy the weights at the end of training on task $t{-}1$.

Yes, the EWC loss is exactly a weighted MSE: for each parameter $\theta_i$, penalise squared deviation from $\theta^*_i$, weighted by the Fisher diagonal $F_i$:
$$\mathcal{L}_\text{EWC} = \frac{\lambda}{2}\sum_i F_i (\theta_i - \theta^*_i)^2.$$
The fully general version would be $(\theta-\theta^*)^T F (\theta-\theta^*)$ with the full $n\times n$ Fisher matrix — equivalent to a Mahalanobis distance in parameter space — but that is intractable for large models, so the diagonal approximation is universal in practice.

---

## 3. Citation on EWC / Fisher variance for DSR

No clean citation exists specifically for EWC failing on DSR due to Fisher variance. The theoretical argument is: $F_i = \mathbb{E}[(\partial\mathcal{L}_\text{recon}/\partial\theta_i)^2]$ estimated from random batches of a chaotic trajectory has variance that grows exponentially with batch length, because Lyapunov divergence amplifies gradient differences between nearby trajectories. The closest citations are:

- **Mikhaeil et al. 2022** (*"On the difficulty of learning chaotic dynamics with RNNs"*, NeurIPS) — shows gradient norms explode along the leading Lyapunov direction, making training unstable. This directly implies that Fisher estimates from short batches are dominated by trajectory-specific variance rather than parameter importance.
- **Kirkpatrick et al. 2017** themselves note that Fisher quality degrades when the loss landscape is non-convex. Chaotic systems are a pathological case of this.

The claim should be presented as a principled conjecture rather than a proven fact, unless you run the ablation explicitly.

---

## 4. Orbit-structure parameters, eigenvalues of $A$, why low $F_i$

**"Orbit structure"** = what type of attractor the model produces: fixed point, limit cycle, or strange attractor. The parameters determining this are:

- Eigenvalues of $B_\text{lin} = \text{diag}(A_{P:}) + W_{P:,P:}$: spectral radius $>1$ means the linear part is expansive (needed for chaos or oscillation); $<1$ means contracting. Shifting eigenvalues across $|\lambda|=1$ changes whether the system is stable, oscillatory, or chaotic.
- The nonlinear coupling columns $W_{B,:,:P}$ (the columns of $W_B$ multiplying $\text{relu}(z_{:P})$): these determine how strongly ReLU switching reshapes the trajectory, carving out the attractor geometry.

**Why do these get low $F_i$?**

$$F_i = \mathbb{E}\!\left[\!\left(\frac{\partial\mathcal{L}_\text{recon}}{\partial\theta_i}\right)^{\!2}\right], \qquad \mathcal{L}_\text{recon} = \|z_{t+1} - \hat{z}_{t+1}\|^2.$$

The one-step prediction is $\hat{z}_{t+1} = A\odot z_t + g(z_t)W_A^TW_B^T + h$. A small change to an eigenvalue of $B_\text{lin}$ shifts the linear recurrence slightly but barely changes the one-step prediction — the trajectory has not had time to diverge yet. So $\partial\mathcal{L}_\text{recon}/\partial A_i$ is small $\Rightarrow$ small $F_i$.

In contrast, $h$ (bias) shifts every prediction directly and uniformly $\Rightarrow$ large gradient $\Rightarrow$ high $F_i$. Same for weights that tune the immediate residual fit.

**"Protects the wrong things":** EWC therefore anchors strongly to $h$ and short-horizon fitting parameters (high $F_i$) while allowing the eigenvalues of $A$ to drift (low $F_i$). But for DSR, eigenvalue structure determines whether the model has a strange attractor vs. a fixed point — exactly what needs to be preserved across tasks.

---

## 5. Why $F_i^\Phi = \|\partial\Phi/\partial\theta_i\|^2$ works as a sensitivity measure

Because $\Phi(\theta)\in\mathbb{R}^D$ is vector-valued, $\partial\Phi/\partial\theta_i \in\mathbb{R}^D$ is the $i$-th column of the Jacobian $J = \partial\Phi/\partial\theta \in\mathbb{R}^{D\times n}$. The quantity
$$F_i^\Phi = \|J_{:,i}\|^2 = \sum_{k=1}^D \left(\frac{\partial\Phi_k}{\partial\theta_i}\right)^2$$
is the squared $\ell_2$ norm of that column — it measures how much $\Phi$ changes in total (across all $D$ components) when you perturb $\theta_i$.

**On the sign concern:** $F_i^\Phi$ does not need to know whether we want $\Phi_k$ higher or lower. $(\partial\Phi_k/\partial\theta_i)^2$ is large whether the effect is positive or negative. We only need to know that $\theta_i$ has large influence on $\Phi_k$ — that is captured by the squared derivative regardless of sign. Intuitively: if $\partial\Phi_k/\partial\theta_i = +5$, moving $\theta_i$ by $\varepsilon$ disrupts the signature by $5\varepsilon$. If it equals $-5$, same disruption in the other direction. Both warrant protection.

**Formal analogy:** in information geometry, the standard Fisher metric is $g_{ii}(\theta) = \mathbb{E}[(\partial_i\log p)^2]$ — curvature of the scalar log-likelihood in direction $\theta_i$. Our $F_i^\Phi = \|J_{:,i}\|^2$ is the same concept applied to a vector-valued function: total curvature of $\Phi$ in direction $\theta_i$. It is also the $i$-th diagonal entry of the Gram matrix $J^TJ$.

---

## 6. Gauge dependence and gauge invariance

**Gauge equivalence:** Two parameterisations $\theta$ and $\tilde\theta$ are gauge-equivalent if there exists an invertible $G\in\mathbb{R}^{M\times M}$ such that $\tilde{z}_t = Gz_t$ for all $t$ — i.e., one is a latent-basis rotation of the other, producing the same observable output. For the factored model, $W_B \to GW_B$ and $W_A \to W_AG^{-1}$ preserves the product $W_BW_A$ and hence all predictions.

**Gauge-dependent:** A quantity that changes under $G$. Individual entries of $W_B$ are gauge-dependent — rotating the latent basis changes each entry, so EWC anchoring to specific $\hat{W}_B$ penalises latent-basis rotations that leave the dynamics unchanged. This is physically wrong: two models with identical attractors but different latent bases would be pushed toward different parameter values.

**Gauge-invariant:** A quantity that does not change under $G$. Lyapunov exponents, topological entropy, eigenvalue magnitudes of the dynamics matrix — all defined from trajectory statistics under the observation map — are invariant under invertible latent-basis changes, because the attractor geometry is preserved. So $\Phi(\theta) = \Phi(G\theta G^{-1})$, and therefore $F_i^\Phi$ correctly assigns zero importance to parameter changes that are "merely" gauge transformations.

---

## 7. Low-variance, stable, gauge-invariance are unverified

Correct. These are theoretical properties:
- **Low-variance / stable:** not empirically verified; we haven't compared $\text{Var}(F_i^\Phi)$ vs $\text{Var}(F_i^\text{pred})$ across random seeds or batch samples.
- **Gauge-invariance:** not numerically verified; we haven't checked that $F_i^\Phi$ is consistent across gauge-equivalent parameterisations.

These should be presented as principled conjectures or design goals, not established results.

---

## 8. Slide: differentiable surrogates

```latex
\begin{frame}{Differentiable Surrogates for $\Phi$}
\textit{Key trick: replace $\mathbb{1}[z_i>0]$ by $\sigma(\beta z_i)$ with $\beta=10$:}
\medskip
\small
\renewcommand{\arraystretch}{1.45}
\begin{tabular}{ll}
\hline
\textbf{Discrete} & \textbf{Soft surrogate} \\
\hline
$\bar s_i$: mean activation rate
  & $\phi^{(b)}_i = \tfrac{1}{T}\sum_t \sigma(\beta z_{i,t})$ \\
$n_\mathrm{regions}$: \# activation patterns
  & $\phi^{(c)}_i = \mathrm{std}_t\,\sigma(\beta z_{i,t})$\quad(0 if 1-region; $>0$ if multi) \\
$|\lambda(A_\mathrm{lin})|$: linear eigvals
  & $\phi^{(a)} = |\lambda(B_\mathrm{lin}+\varepsilon I)|$,\;
    $B_\mathrm{lin}=\mathrm{diag}(A_{P:})+W_{P:,P:}$ \\
$h_\mathrm{top}=\log\rho(T)$: topo.\ entropy
  & $\phi^{(d)}=\log(\rho(\widetilde T)+\varepsilon)$,\;
    $\widetilde T_{ij}=\tfrac{1}{T}\sum_t\sigma(\beta z_{i,t})\sigma(\beta z_{j,t+1})$ \\
$\lambda_\mathrm{max}$: max Lyapunov
  & $\phi^{(e)}$: power iter.\ on
    $\widetilde J_t=\mathrm{diag}(A)+\mathrm{diag}(\sigma(\beta z_{t,<P})\oplus\mathbf{1}_{M-P})\cdot W$ \\
\hline
\end{tabular}
\medskip
\begin{itemize}[<+->]
  \item All 24 components autodiff through $(W_B,\,W_A,\,A,\,h)$
  \item $\varepsilon I$ perturbation prevents degenerate backward through \texttt{eigvals}
\end{itemize}
\end{frame}
```

---

## 9. Why $A$ is a basis and $B$ holds task-dependent coefficients

In LoRA $\Delta W = BA$ with $B\in\mathbb{R}^{M\times r}$ and $A\in\mathbb{R}^{r\times M}$: $A$ is the down-projection (projects $M$-dim activations into $r$-dim subspace) and $B$ is the up-projection (maps back). The rows of $A$ span the subspace where the low-rank correction operates — think of them as **basis vectors**. The columns of $B$ encode **how much each basis direction contributes** to the output.

**The argument from SLAO (Qiao & Mahdavi, arXiv:2512.23017):** tasks share the same latent space, so the relevant low-rank subspace (WHAT directions matter) is relatively stable across tasks and benefits from being refreshed to an orthonormal basis. The task-specific information (HOW MUCH each direction contributes) lives in $B$ and should be accumulated via EMA. QR-reinitialising $A^{(0)}_t$ from $A_\text{ft}^{(t-1)}$ gives orthonormal rows spanning the same subspace without redundancy — a canonical "basis hygiene" operation.

No other citation for this specific argument — it is SLAO's core contribution.

---

## 10. The adaptive probe: notation and criterion

**Lowercase $\phi$ vs. uppercase $\Phi$:** they are the same object — the 24-dim differentiable signature vector. The notation drifted through the writeup. Unify to one symbol; $\Phi$ is the natural choice since it is introduced on the Signature Embedding slide.

**Normalisation by $\sigma_\phi$:** this is a per-component standard deviation computed across all stored oracle signatures. Without it, components with large absolute scale dominate the distance. It is standard $z$-score normalisation before computing $\ell_2$ distances.

**$\sigma_\text{schema}$:** a hyperparameter (set to $2.0$ in experiments) — a scalar multiplier controlling how permissive the assimilation threshold is.

**$\sigma_\text{intra}$:** computed from the stored library — the mean distance of stored task signatures to their own class centroid. If Lorenz variants are all similar, $\sigma_\text{intra}$ is small (tight threshold). If VdP spans a wide parameter range, $\sigma_\text{intra}$ is larger (wider assimilation zone).

**The criterion $d^* < \sigma_\text{schema}\cdot\sigma_\text{intra}$:** assimilate if the distance to the nearest centroid is less than $\sigma_\text{schema}$ intra-class standard deviations. Adaptive to within-class heterogeneity.

---

## 11. Why high $\lambda$ for assimilation, low for accommodation?

The $\lambda$ cached after task $t$ is used **during training on task $t{+}1$**, not during task $t$. Walk through both cases:

**Assimilation case** (task $t$ recognised as a known class, e.g. Lorenz):
- We just confirmed the model encodes a known, valuable schema
- Cache $\lambda_\text{EWC} \leftarrow \lambda_\text{assim}$ (HIGH) for task $t{+}1$
- During task $t{+}1$, the high EWC penalty strongly anchors $W_B$ near the consolidated $\hat{W}_B$ — it **protects the Lorenz knowledge accumulated so far** from being overwritten by whatever task $t{+}1$ is
- The protection is of **past accumulated knowledge**, not of task $t$'s learning

**Accommodation case** (task $t$ not recognised — new territory):
- The accumulated $\hat{W}_B$ reflects something uncertain/unvalidated
- Cache $\lambda_\text{EWC} \leftarrow \lambda_\text{accom}$ (LOW) for task $t{+}1$
- Low EWC during task $t{+}1$ $\Rightarrow$ more plasticity, allowing $t{+}1$ to reshape the representation more freely

**Your intuition is partially right:** we would also want high plasticity when the next task is a new class. But at probe time we don't know what task $t{+}1$ will be — we only know what task $t$ was. The probe answers: "Is the knowledge we just accumulated worth strongly protecting?" If yes (assimilation), protect it. If no (accommodation, uncharted territory), be flexible.

---

## 12. Probe-before-append: the self-matching artefact

If we first append $(c_t, \phi_t)$ to the library and then run the probe, the probe computes distances from $\phi_t$ to all stored signatures. But $\phi_t$ is now in the library — its distance to itself is exactly $0$. Since $0 < $ any positive threshold, the probe would always declare ASSIMILATION regardless of which class task $t$ belongs to.

Probing before appending asks the correct question: "Does the model I just trained resemble any **previously seen** task?" — using only the history that existed before task $t$. Appending afterwards adds task $t$ to the history for future probes.

---

## 13. 72 backward passes — not one

It is not one pass. To compute $F_i^\Phi = \|J_{:,i}\|^2$ for all parameters $i$ simultaneously, we need the full Jacobian $J = \partial\Phi/\partial\theta \in\mathbb{R}^{D\times n}$.

In reverse-mode autodiff, one backward pass gives $v^T J$ for a chosen vector $v$ — one row of $J$, i.e., the gradient of one scalar output $\Phi_k$ with respect to all parameters. To get all $D=24$ rows, we need $D$ backward passes.

We then accumulate $F_i^\Phi = \sum_k (\partial\Phi_k/\partial\theta_i)^2$ across rows.

The $72 = 24$ dimensions $\times$ $3$ averaging rollouts (`n_avg=3` for variance reduction). Forward-mode autodiff would give one column $Ju$ per pass, but since $D \ll n$ (24 $\ll$ thousands of parameters), $D$ reverse passes is the efficient direction. An alternative using Hutchinson's trace estimator ($\text{tr}(J^TJ) \approx \frac{1}{m}\sum_j \|Jv_j\|^2$ for random $v_j$) could reduce the count, but the current implementation does it exactly.

---

## 14. Where $t^{-1/2}$ comes from

Directly from SLAO (Qiao & Mahdavi 2024). The argument:

$1/\sqrt{t}$ is the natural online learning rate schedule — it appears in Pegasos, AdaGrad, and regret-optimal online convex optimisation. It has two useful properties:

- $\sum_{t=1}^\infty 1/\sqrt{t}$ diverges $\Rightarrow$ the EMA never fully freezes; late tasks still receive non-trivial updates.
- $\sum_{t=1}^T 1/\sqrt{t} = O(\sqrt{T})$ grows only sublinearly $\Rightarrow$ early tasks are given progressively more protection as the stream grows.

An alternative justification: $1/\sqrt{t}$ is the rate at which the empirical mean of an IID sequence converges to the true mean, so it is the right discount for averaging $W_B^{(t)}$ across tasks with diminishing uncertainty.

---

## 15. Where $\lambda$ is used in the pseudocode

You are right — line 8 sets $\lambda_\text{EWC} \leftarrow \lambda_\text{next}$ but the pseudocode never shows it being consumed. It enters the **training loop** for task $t{+}1$, not at the task boundary:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda_\text{EWC}\sum_{ij}\hat{F}_{ij}(W_{B,ij}-\hat{W}_{B,ij})^2.$$

The pseudocode only covers task-boundary bookkeeping. To make the slide self-contained, you could add a line:

```
during task t+1:  L_total = L_recon + lambda_EWC * L_EWC(W_B, F_hat, W_B_hat)
```

---

## 16. The negative sign in $R_{t,i}$; why not ARI?

**The negative sign** is a pure sign convention so that "higher = better" holds everywhere. Define $\phi\text{-dist}(a,b) = \|(a-b)/\sigma_\phi\|_2 \geq 0$. Then:
$$R_{t,i} = -\phi\text{-dist}(\phi(\theta_t), \phi_i^\text{ora}).$$
$R_{t,i}=0$ when the model is exactly at oracle $i$; it becomes more negative as the model drifts. This makes the BWT formula $\text{BWT} = \frac{1}{T-1}\sum_{i<T}(R_{T-1,i}-R_{i,i})$ read naturally: positive $=$ models ended up closer to past oracles than they were right after training, i.e., the stream helped rather than hurt.

**Why $\phi$-distance rather than ARI:** ARI measures clustering accuracy — binary correct/incorrect class label. $\phi$-distance measures the *degree* of alignment in the continuous signature space: two methods that both correctly label a task can have very different $\phi$-distances to the oracle. $\phi$-distance gives a continuous signal and captures partial retention that binary metrics miss. The rec\_ora metric is the ARI-equivalent we already report for classification accuracy.

---

## 17. What was achieved intuitively

A baseline model trained sequentially on $T$ systems converges entirely to the last system's attractor (catastrophic forgetting: $R_{T-1,i} \ll 0$ for $i < T-1$). PIAGETS keeps the model's dynamical fingerprint $\Phi$ simultaneously close to all past oracle fingerprints while still adapting to each new task.

The practical payoff is twofold:

1. **Meta-initialisation:** $B_\text{merge}$ accumulates the relevant low-rank structure across all schema classes, so fine-tuning from it on any previously-seen system class converges faster than fine-tuning from random init.
2. **Schema classification:** the $\Phi$-library maps a new system's signature to its nearest schema class. For neuroscience: given a new patient's fMRI, extract $\Phi$, find the nearest schema, retrieve the corresponding warm start and class label.

The system does *not* produce a single model that simultaneously runs multiple dynamics — the DSR model at inference time still represents one system at a time. The value is fast adaptation and interpretable schema labels, accumulated without catastrophic forgetting.

---

## 18. Should we report reconstruction loss?

Yes, and its absence is a real gap. $\phi$-distance is a proxy — we never directly measure whether the model produces good trajectories. Two failure modes are invisible to $\phi$-distance:

1. The model gets close to oracle $\Phi$ but degenerates in reconstruction (a degenerate model with a collapsed trajectory can still produce distinctive $\Phi$ values).
2. The positive BWT artefact from $B_\text{merge}$ blending: $B_\text{merge}$ scores well in $\phi$-space even though it cannot reconstruct any specific task.

The correct primary metric for the meta-initialisation use-case would be: fine-tune from $B_\text{merge}$ for $K$ gradient steps on task $i$; measure reconstruction MSE vs. fine-tuning from random init for the same $K$ steps. The ratio — "adaptation efficiency" — is what the schema library actually promises. Without it, we cannot verify the core claim.
