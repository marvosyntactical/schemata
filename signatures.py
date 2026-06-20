"""Schema signatures for clustering dynamical systems from trained AL-RNNs.

Implements the signature menu of embedding.md (sect. 5) on top of the closed-form
region machinery in alrnn.py. Each block is a labelled descriptor of one axis of
the dynamics, tagged by the *equivalence relation* it respects:

  class='topological' -- conjugacy invariants (fixed-point counts and stability
      indices, admissible periods, topological entropy, graph spectrum up to
      relabeling, Lyapunov signs, Betti numbers). These DEFINE the clustering.
  class='rate'        -- smooth-conjugacy / metric quantities that vary *within*
      a conjugacy class (eigenvalue and Floquet moduli, Lyapunov magnitudes,
      Kaplan-Yorke dim, the backbone spectrum). Carried on a separate, labelled,
      down-weighted axis (embedding.md sect. 7: "this is the decision").
  class='geometry'    -- DATA-side, gauge-bearing descriptors read from the raw
      observations rather than the trained model: the spectral shape (oscillatory
      content) and the spatial covariance spectrum. These capture exactly what the
      gauge-free model channels discard, and are what actually separates real
      neural data (summary.md sect. 3.5: spectral/spatial info dominates on EEG/
      fMRI). Summaries are rotation-invariant (trace-PSD, covariance eigenvalues)
      so a smooth-conjugacy variant still matches, but they DO see amplitude and
      frequency. Present only when extract() is given the observation series.

Block kinds:
  'scalar' -- fixed-or-padded vector, compared by z-scored Euclidean (population).
  'cloud'  -- an unordered set of (complex) numbers of model-dependent size,
      compared by 2-Wasserstein, the metric DSA reduces to in the normal case.

Assembly (combine_distance) applies three weighting layers:
  Layer 1  topological vs rate  -- class weight 1 vs gamma (the chosen relation).
  Layer 2  reliability          -- estimate_reliability() down-weights blocks that
      are estimation-noisy (large within-target scatter relative to the spread
      across targets), replacing the hand-tuned WEIGHTS hack in cl_run.py.
  Layer 3  commensuration       -- each heterogeneous block distance is divided by
      its median before combining, so the weights mean what they say.
"""
import numpy as np
from numpy.linalg import eigvals
import scipy.linalg
from scipy.stats import wasserstein_distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import metrics

try:
    from ripser import ripser            # optional: persistent homology (sect. 5.6)
    _HAS_PH = True
except Exception:
    _HAS_PH = False


# --------------------------------------------------------------------------- #
#  block container
# --------------------------------------------------------------------------- #
def _block(name, cls, kind, data):
    arr = np.asarray(data)
    return {"name": name, "class": cls, "kind": kind, "data": arr}


class Signature:
    """A trained model's full signature: a dict of named blocks + metadata."""

    def __init__(self, blocks, meta=None):
        self.blocks = blocks
        self.meta = meta or {}

    def names(self):
        return sorted(self.blocks)


def _pattern_matches(zstar, pat):
    """A candidate equilibrium is *real* only if its nonlinear coordinates carry
    the sign pattern of the region that generated it (embedding.md sect. 5.1);
    otherwise it is virtual and discarded."""
    P = len(pat)
    active = (zstar[:P] > 0).astype(np.int8)
    return bool(np.all(active == pat))


# --------------------------------------------------------------------------- #
#  5.1  equilibrium portrait
# --------------------------------------------------------------------------- #
def equilibrium_portrait(rd, max_index=8, cond_tol=1e8):
    h, M = rd["h"], len(rd["h"])
    I = np.eye(M)
    n_fp, flagged = 0, 0
    idx_hist = np.zeros(max_index + 1)
    clouds = []
    for s, W in rd["regions"].items():
        ImW = I - W
        if np.linalg.cond(ImW) > cond_tol:        # near non-hyperbolic: flag, skip
            flagged += 1
            continue
        zstar = np.linalg.solve(ImW, h)
        if not _pattern_matches(zstar, rd["patterns"][s]):
            continue                              # virtual fixed point
        n_fp += 1
        ev = eigvals(W)
        k_unstable = int((np.abs(ev) > 1.0).sum())
        idx_hist[min(k_unstable, max_index)] += 1
        clouds.append(ev)
    cloud = np.concatenate(clouds) if clouds else np.zeros(0, complex)
    return {
        "eq_count": _block("equilibria/count", "topological", "scalar",
                           [n_fp, flagged]),
        "eq_index_hist": _block("equilibria/stability_index", "topological",
                                "scalar", idx_hist),
        "eq_spectrum": _block("equilibria/eigenvalues", "rate", "cloud", cloud),
    }


# --------------------------------------------------------------------------- #
#  5.2  periodic-orbit spectrum
# --------------------------------------------------------------------------- #
def _closed_walks(syms, edges, k_max):
    """All admissible closed walks of length <= k_max in the transition graph,
    deduplicated by cyclic rotation. The graph has a handful of nodes (the AL-RNN
    visits few regions), so brute-force DFS is cheap and the admissibility prune
    is what keeps the period search tractable (embedding.md sect. 5.2)."""
    succ = {a: [b for b in syms if (a, b) in edges] for a in syms}
    walks = []
    for start in syms:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt in succ[node]:
                if nxt == start:
                    walks.append(tuple(path))     # closes a walk of length len(path)
                if len(path) < k_max:
                    stack.append((nxt, path + [nxt]))
    seen, uniq = set(), []
    for w in walks:
        if _is_repetition(w):                     # skip non-primitive cycles:
            continue                              # (0,0)=(0)x2, (0,1,0,1)=(0,1)x2
        key = min(tuple(w[i:] + w[:i]) for i in range(len(w)))
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq


def _is_repetition(w):
    """True if w is a shorter word repeated -- a lower-period orbit in disguise,
    which must not be counted again at the higher period."""
    k = len(w)
    for p in range(1, k):
        if k % p == 0 and w == w[:p] * (k // p):
            return True
    return False


def periodic_orbits(rd, M, k_max=6, cond_tol=1e10):
    regions, patterns, h = rd["regions"], rd["patterns"], rd["h"]
    syms = [int(s) for s in rd["uniq"]]
    idx, B, I = rd["index"], rd["trans"], np.eye(M)
    edges = {(a, b) for a in syms for b in syms if B[idx[a], idx[b]] > 0}

    period_count = np.zeros(k_max)
    floq = []
    for w in _closed_walks(syms, edges, k_max):
        k = len(w)
        Mw, bw = np.eye(M), np.zeros(M)
        for s in w:                               # compose in visiting order
            W = regions[s]
            Mw, bw = W @ Mw, W @ bw + h
        if np.linalg.cond(I - Mw) > cond_tol:
            continue
        zstar = np.linalg.solve(I - Mw, bw)
        state, ok = zstar.copy(), True
        for s in w:                               # validate region membership
            if not _pattern_matches(state, patterns[s]):
                ok = False
                break
            state = regions[s] @ state + h
        if not ok:
            continue
        period_count[k - 1] += 1
        floq.append(eigvals(Mw))
    cloud = np.concatenate(floq) if floq else np.zeros(0, complex)
    return {
        "po_count_per_period": _block("orbits/count_per_period", "topological",
                                      "scalar", period_count),
        "po_period_set": _block("orbits/period_set", "topological", "scalar",
                                (period_count > 0).astype(float)),
        "po_floquet": _block("orbits/floquet", "rate", "cloud", cloud),
    }


# --------------------------------------------------------------------------- #
#  5.3  symbolic-graph spectrum
# --------------------------------------------------------------------------- #
def symbolic_graph(rd, k_max=6):
    B = rd["trans"]
    k = B.shape[0]
    if k == 0:
        return {}
    A = (B > 0).astype(float)                     # admissibility matrix
    rho = float(np.max(np.abs(eigvals(A)))) if k else 0.0
    h_top = np.log(max(rho, 1e-12))               # topological entropy
    traces, Ap = [], np.eye(k)
    for _ in range(k_max):
        Ap = Ap @ A
        traces.append(float(np.trace(Ap)))        # closed-walk counts tr(A^n)
    n_scc, _ = connected_components(csr_matrix(A), directed=True,
                                    connection="strong")
    # normalized Laplacian of the symmetrized graph: eigenvalues in [0,2], so the
    # descriptor is comparable across models with different region counts.
    As = np.maximum(A, A.T)
    deg = As.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    Ln = np.eye(k) - dinv[:, None] * As * dinv[None, :]
    lap = np.sort(np.real(eigvals(Ln))).astype(complex)
    return {
        "graph_scalars": _block("graph/scalars", "topological", "scalar",
                                [k, h_top, n_scc]),
        "graph_closed_walks": _block("graph/closed_walks", "topological",
                                     "scalar", traces),
        "graph_laplacian": _block("graph/laplacian_spectrum", "topological",
                                  "cloud", lap),
    }


# --------------------------------------------------------------------------- #
#  5.4  generator geometry
# --------------------------------------------------------------------------- #
def generator_geometry(model, r=2):
    A = model.A.detach().numpy().astype(np.float64)
    W = model.W.detach().numpy().astype(np.float64)
    P, M = model.P, model.M
    dg = np.ones(M)
    dg[:P] = 0.0
    base = np.diag(A) + W * dg                     # all nonlinear units off
    base_ev = eigvals(base)
    rho0 = float(np.max(np.abs(base_ev)))
    w_ev, V = np.linalg.eig(base)
    U = np.real(V[:, np.argsort(-np.abs(w_ev))[:r]])   # leading invariant subspace
    shifts, angles = [], []
    for m in range(P):
        upd = base.copy()
        upd[:, m] += W[:, m]                       # activating unit m: rank-one add
        shifts.append(float(np.max(np.abs(eigvals(upd)))) - rho0)
        col = W[:, m:m + 1]
        if np.linalg.norm(col) > 1e-12:
            angles.append(float(scipy.linalg.subspace_angles(col, U).min()))
        else:
            angles.append(0.0)
    return {
        "gen_backbone": _block("generator/backbone_spectrum", "rate", "cloud",
                               base_ev),
        "gen_switch_shift": _block("generator/switch_radius_shift", "rate",
                                   "scalar", shifts),
        "gen_switch_angle": _block("generator/switch_angle", "topological",
                                   "scalar", angles),
    }


# --------------------------------------------------------------------------- #
#  5.5  ergodic / Lyapunov signature  (reuses metrics.py)
# --------------------------------------------------------------------------- #
def lyapunov_signature(model, seed_obs, n=2500, warmup=400, tol=1e-3):
    spec = metrics.lyapunov_spectrum(model, seed_obs, n=n, warmup=warmup)
    ky = metrics.kaplan_yorke(spec)
    n_pos = int((spec > tol).sum())
    n_zero = int((np.abs(spec) <= tol).sum())
    n_neg = int((spec < -tol).sum())
    return {
        "lyap_signs": _block("lyapunov/signs", "topological", "scalar",
                             [n_pos, n_zero, n_neg]),
        "lyap_ky": _block("lyapunov/kaplan_yorke", "rate", "scalar", [ky]),
        "lyap_spectrum": _block("lyapunov/spectrum", "rate", "cloud",
                                spec.astype(complex)),
    }


# --------------------------------------------------------------------------- #
#  5.6  attractor topology  (persistent homology; optional dependency)
# --------------------------------------------------------------------------- #
def attractor_topology(model, seed_obs, n=3000, max_dim=2, n_sub=800, thresh=2.0):
    # max_dim=2 includes H_2 (voids): sharpens Lorenz (2-lobe) vs Rossler (1-lobe)
    # separation. Was capped at 1 on workstations; 51 GB VRAM makes it tractable.
    # n_sub=800 keeps the Vietoris-Rips complex manageable.
    if not _HAS_PH:
        return {}                                  # block omitted; assembly copes
    cloud = model.free_run(seed_obs, n)
    cloud = cloud[np.isfinite(cloud).all(1)]
    if len(cloud) < 10:
        return {}
    if len(cloud) > n_sub:
        cloud = cloud[np.linspace(0, len(cloud) - 1, n_sub).astype(int)]
    dgms = ripser(cloud, maxdim=max_dim, thresh=thresh)["dgms"]
    feats = []
    for d in range(max_dim + 1):
        dg = dgms[d]
        dg = dg[np.isfinite(dg[:, 1])] if len(dg) else dg
        life = (dg[:, 1] - dg[:, 0]) if len(dg) else np.zeros(0)
        feats += [float(len(dg)),
                  float(life.sum()) if len(life) else 0.0,
                  float(life.max()) if len(life) else 0.0]
    return {"topo_persistence": _block("topology/persistence_stats",
                                       "topological", "scalar", feats)}


# --------------------------------------------------------------------------- #
#  5.7  optional spectral block (pooled region Jacobians; the DSA-style coord)
# --------------------------------------------------------------------------- #
def region_spectrum(rd):
    evs = [eigvals(W) for W in rd["regions"].values()]
    cloud = np.concatenate(evs) if evs else np.zeros(0, complex)
    return {"spectral_region": _block("spectral/region_jacobians", "rate",
                                      "cloud", cloud)}


# --------------------------------------------------------------------------- #
#  geometry / spectral channel (DATA-side, gauge-bearing)
# --------------------------------------------------------------------------- #
# log-ish frequency bands as fractions of the Nyquist rate: finer at the low end
# where slow-wave vs fast-regime differences live. Domain-general (no Hz baked in).
_BANDS_FRAC = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10),
               (0.10, 0.20), (0.20, 0.35), (0.35, 0.50)]


def spectral_signature(obs, nperseg=256):
    """Rotation-invariant spectral shape of the multivariate observation series
    obs: (T, d). The mean Welch PSD across channels is the trace of the cross-
    spectral density, hence invariant to orthogonal channel mixing; normalized and
    pooled into fractional-Nyquist bands it is the oscillatory fingerprint (e.g.
    delta-dominated slow-wave sleep vs fast wake) the gauge-free channels miss."""
    from scipy.signal import welch
    obs = np.asarray(obs, float)
    T, d = obs.shape
    f, acc = None, None
    for c in range(d):
        fi, pi = welch(obs[:, c], nperseg=min(nperseg, T))
        f = fi
        acc = pi if acc is None else acc + pi
    acc = acc / max(d, 1)
    acc = acc / (acc.sum() + 1e-12)                # normalized spectral shape
    fn = f / (f[-1] + 1e-12)                        # fraction of Nyquist
    bands = np.array([acc[(fn >= lo) & (fn < hi)].sum() for lo, hi in _BANDS_FRAC])
    centroid = float((fn * acc).sum())              # spectral centroid (rate-of-mass)
    entropy = float(-(acc * np.log(acc + 1e-12)).sum())   # spectral flatness/entropy
    return {
        "geom_spectral_bands": _block("geometry/spectral_bands", "geometry",
                                      "scalar", np.log(bands + 1e-6)),
        "geom_spectral_summary": _block("geometry/spectral_summary", "geometry",
                                        "scalar", [centroid, entropy]),
    }


def spectral_signature_perchannel(obs):
    """Per-channel relative band-power (geometry class, GAUGE-BEARING -- NOT
    rotation-invariant). When the channels are physically meaningful (real EEG
    electrodes), *which* channel carries the slow-wave / spindle power is itself
    discriminative; pooling it away (as spectral_signature does for gauge-freedom)
    costs that spatial resolution. Include this only when channel identity is
    meaningful and rotation-invariance is not required."""
    from scipy.signal import welch
    obs = np.asarray(obs, float)
    T, d = obs.shape
    feats = []
    for c in range(d):
        f, p = welch(obs[:, c], nperseg=min(256, T))
        p = p / (p.sum() + 1e-12)
        fn = f / (f[-1] + 1e-12)
        feats += [p[(fn >= lo) & (fn < hi)].sum() for lo, hi in _BANDS_FRAC]
    return {"geom_spectral_perchannel": _block("geometry/spectral_perchannel",
                                               "geometry", "scalar",
                                               np.log(np.array(feats) + 1e-6))}


def spatial_signature(obs):
    """Spatial covariance spectrum of obs: (T, d). Eigenvalues of the channel
    covariance are rotation-invariant; normalized by trace they describe the
    spatial anisotropy / effective dimensionality (participation ratio). For many-
    channel data this is the connectome-fingerprint axis (Finn et al. 2015) the
    gauge-free channels are blind to."""
    obs = np.asarray(obs, float)
    C = np.atleast_2d(np.cov(obs.T))
    ev = np.clip(np.sort(np.real(eigvals(C)))[::-1], 0, None)
    evn = ev / (ev.sum() + 1e-12)
    pr = float(ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-12))   # participation ratio
    return {
        "geom_cov_spectrum": _block("geometry/cov_spectrum", "geometry", "scalar",
                                    evn),
        "geom_effective_dim": _block("geometry/effective_dim", "geometry",
                                     "scalar", [pr]),
    }


# --------------------------------------------------------------------------- #
#  top-level extraction
# --------------------------------------------------------------------------- #
def extract(model, seed_obs, obs_data=None, geom_per_channel=False, k_max=6,
            lyap_n=2500, n_visit=4000, warmup=500, ph_max_dim=2, ph_n_sub=150):
    """Full per-model signature. O(N) in trajectory length plus small-matrix
    linear algebra over the handful of visited regions.

    ph_max_dim: persistent homology degree cap (2 = include voids/H_2; 1 = loops only).
    ph_n_sub:   subsample size for the Vietoris-Rips complex (larger = slower)."""
    rd = model.enumerate_visited_regions(seed_obs, n=n_visit, warmup=warmup)
    blocks = {}
    blocks.update(equilibrium_portrait(rd))
    blocks.update(periodic_orbits(rd, model.M, k_max=k_max))
    blocks.update(symbolic_graph(rd, k_max=k_max))
    blocks.update(generator_geometry(model))
    blocks.update(lyapunov_signature(model, seed_obs, n=lyap_n))
    blocks.update(attractor_topology(model, seed_obs, max_dim=ph_max_dim, n_sub=ph_n_sub))
    blocks.update(region_spectrum(rd))
    if obs_data is not None:                       # data-side geometry channel
        blocks.update(spectral_signature(obs_data))
        blocks.update(spatial_signature(obs_data))
        if geom_per_channel:                       # gauge-bearing, channel-aware
            blocks.update(spectral_signature_perchannel(obs_data))
    return Signature(blocks, meta={"clip_rate": rd["clip_rate"],
                                   "n_regions": int(len(rd["uniq"]))})


# --------------------------------------------------------------------------- #
#  per-block distances
# --------------------------------------------------------------------------- #
def _cloud_coords(c):
    """Reflection-invariant 1-D coordinates of an eigenvalue cloud: modulus and
    |angle|. For real spectra (Lyapunov) the angle is 0 / pi, so sign survives."""
    if len(c) == 0:
        return np.zeros(0), np.zeros(0)
    return np.abs(c), np.abs(np.angle(c))


def _w1(x, y):
    if len(x) == 0 and len(y) == 0:
        return 0.0
    x = x if len(x) else np.zeros(1)
    y = y if len(y) else np.zeros(1)
    return float(wasserstein_distance(x, y))


def cloud_distance(a, b):
    """2-Wasserstein between two eigenvalue clouds (sum over modulus and |angle|),
    the same spectral notion DSA reduces to in the normal-operator case."""
    ma, aa = _cloud_coords(a)
    mb, ab = _cloud_coords(b)
    return _w1(ma, mb) + _w1(aa, ab)


def block_distance_matrix(sigs, bname):
    """K x K distance for one block across a population of signatures, plus its
    class tag. Scalars are z-scored across the population then compared by
    Euclidean; clouds by 2-Wasserstein. Returns (None, None) if no signature
    carries the block."""
    sample = next((s.blocks[bname] for s in sigs if bname in s.blocks), None)
    if sample is None:
        return None, None
    K = len(sigs)
    D = np.zeros((K, K))
    if sample["kind"] == "scalar":
        L = max(len(s.blocks[bname]["data"]) for s in sigs if bname in s.blocks)
        X = np.zeros((K, L))
        for i, s in enumerate(sigs):
            if bname in s.blocks:
                d = np.asarray(s.blocks[bname]["data"], float).ravel()
                X[i, :len(d)] = d
        sd = X.std(0)
        sd[sd < 1e-9] = 1.0
        Xn = (X - X.mean(0)) / sd
        for i in range(K):
            for j in range(i + 1, K):
                D[i, j] = D[j, i] = np.linalg.norm(Xn[i] - Xn[j])
    else:  # cloud
        cl = [s.blocks[bname]["data"] if bname in s.blocks else np.zeros(0, complex)
              for s in sigs]
        for i in range(K):
            for j in range(i + 1, K):
                D[i, j] = D[j, i] = cloud_distance(cl[i], cl[j])
    return D, sample["class"]


# --------------------------------------------------------------------------- #
#  Layer 2 : unsupervised reliability weights
# --------------------------------------------------------------------------- #
def estimate_reliability(groups, reg=0.1):
    """Per-block reliability from replicate groups (each inner list = several
    signatures of the *same* target -- e.g. retrainings, seeds, or affine
    variants). A block is reliable when its within-group scatter is small
    relative to the spread across all targets:

        w_b  ~  median_all(d_b) / (median_within(d_b) + reg * median_all(d_b))

    This down-weights estimation-noisy blocks (Koopman fit variability, jittery
    Lyapunov/persistence) AND uninformative ones (a degenerate block that is
    constant everywhere has median_all ~ 0 -> weight 0), without needing labels.
    Weights are normalized to mean 1 over informative blocks."""
    allsigs = [s for grp in groups for s in grp]
    bnames = sorted({b for s in allsigs for b in s.blocks})
    rel = {}
    for b in bnames:
        D, _ = block_distance_matrix(allsigs, b)
        if D is None:
            continue
        total_med = np.median(D[np.triu_indices(len(allsigs), 1)])
        if total_med < 1e-8:                       # block carries no information
            rel[b] = 0.0
            continue
        within = []
        for grp in groups:
            if len(grp) < 2:
                continue
            Dg, _ = block_distance_matrix(grp, b)
            within += list(Dg[np.triu_indices(len(grp), 1)])
        within_med = np.median(within) if within else total_med
        rel[b] = total_med / (within_med + reg * total_med)
    informative = [v for v in rel.values() if v > 0]
    mean = np.mean(informative) if informative else 1.0
    return {k: (v / mean if v > 0 else 0.0) for k, v in rel.items()}


# --------------------------------------------------------------------------- #
#  Layer 2 alternatives: supervised Fisher weights and silhouette weights
# --------------------------------------------------------------------------- #
def fisher_weights(sigs, labels, reg=0.1):
    """Per-block supervised Fisher weights from ground-truth class labels.

        w_b = between_mean / (within_mean + reg * between_mean)

    A block that perfectly separates classes (within→0) gets weight >> 1; a block
    that is constant or uninformative gets weight 0. Same return format as
    estimate_reliability() so it can be passed directly to combine_distance(reliability=).

    Fixes the 'coarse-dilutes-sharp' failure of Layer-2 reliability (signature_findings
    §6.3): reliability measures local discriminability, which coarse and fine blocks can
    both satisfy; Fisher measures global between/within separation, which rewards blocks
    that produce clean global partitions."""
    labels = np.asarray(labels)
    bnames = sorted({b for s in sigs for b in s.blocks})
    K = len(sigs)
    iu = np.triu_indices(K, 1)
    same = labels[iu[0]] == labels[iu[1]]
    fw = {}
    for b in bnames:
        D, _ = block_distance_matrix(sigs, b)
        if D is None:
            fw[b] = 0.0
            continue
        off = D[iu]
        total_med = np.median(off[off > 0]) if np.any(off > 0) else 0.0
        if total_med < 1e-8:
            fw[b] = 0.0
            continue
        w_mean = float(off[same].mean()) if same.any() else 0.0
        a_mean = float(off[~same].mean()) if (~same).any() else 0.0
        if a_mean < 1e-8:
            fw[b] = 0.0
        else:
            fw[b] = a_mean / (w_mean + reg * a_mean)
    informative = [v for v in fw.values() if v > 0]
    mean_w = np.mean(informative) if informative else 1.0
    return {k: (v / mean_w if v > 0 else 0.0) for k, v in fw.items()}


def silhouette_weights(sigs, clusters, reg=0.0):
    """Per-block silhouette-based weights: rewards blocks whose geometry supports
    the given cluster assignment. Use after an initial clustering pass to refine.

        w_b = max(0, silhouette_score(D_b, clusters))

    Negative silhouettes (block contradicts clusters) are zeroed. Unsupervised:
    takes cluster assignments rather than ground-truth labels, so it can be applied
    when labels are unavailable by bootstrapping from a first-pass clustering.
    Same return format as estimate_reliability() for combine_distance(reliability=)."""
    from sklearn.metrics import silhouette_score as _sil
    clusters = np.asarray(clusters)
    bnames = sorted({b for s in sigs for b in s.blocks})
    sw = {}
    n_unique = len(np.unique(clusters))
    if n_unique < 2:
        return {b: 0.0 for b in bnames}
    for b in bnames:
        D, _ = block_distance_matrix(sigs, b)
        if D is None:
            sw[b] = 0.0
            continue
        off = D[np.triu_indices(len(sigs), 1)]
        if np.median(off[off > 0]) < 1e-8 if np.any(off > 0) else True:
            sw[b] = 0.0
            continue
        try:
            sc = float(_sil(D, clusters, metric="precomputed"))
        except Exception:
            sw[b] = 0.0
            continue
        sw[b] = max(0.0, sc)
    informative = [v for v in sw.values() if v > 0]
    mean_w = np.mean(informative) if informative else 1.0
    return {k: (v / mean_w if v > 0 else 0.0) for k, v in sw.items()}


# --------------------------------------------------------------------------- #
#  assembly : the three weighting layers
# --------------------------------------------------------------------------- #
def combine_distance(sigs, gamma=0.2, geom=1.0, reliability=None, weights=None):
    """K x K model-to-model distance over all blocks.

      Layer 1 (class)  : topological blocks weight 1, rate blocks weight gamma,
                         geometry (data-side) blocks weight geom. gamma=0 ->
                         pure topological clustering; gamma~1 -> DSA-like; geom=0
                         -> model-side only (ignore the data-side geometry channel).
      Layer 2 (reliab.): multiply by per-block reliability weights if supplied.
      Layer 3 (scale)  : divide each block distance by its median first, so the
                         heterogeneous blocks (Wasserstein vs z-Euclidean) combine
                         on a common scale and the weights are meaningful.

    `weights` overrides Layers 1-2 entirely with an explicit {block: w} map.
    Returns (D, detail) where detail[block] = (class, weight, scale)."""
    class_w = {"topological": 1.0, "rate": gamma, "geometry": geom}
    bnames = sorted({b for s in sigs for b in s.blocks})
    K = len(sigs)
    total = np.zeros((K, K))
    wsum = 0.0
    detail = {}
    for b in bnames:
        D, cls = block_distance_matrix(sigs, b)
        if D is None:
            continue
        off = D[np.triu_indices(K, 1)]
        scale = np.median(off[off > 0]) if np.any(off > 0) else 1.0
        Dn = D / scale
        if weights is not None:
            w = weights.get(b, 0.0)
        else:
            w_class = class_w.get(cls, gamma)
            w_rel = 1.0 if reliability is None else reliability.get(b, 1.0)
            w = w_class * w_rel
        if w <= 0:
            detail[b] = (cls, 0.0, scale)
            continue
        total += w * Dn ** 2
        wsum += w
        detail[b] = (cls, w, scale)
    return np.sqrt(total / max(wsum, 1e-9)), detail


def cluster(D, n_clusters=None, threshold=None, method="average"):
    """Convenience hierarchical clustering on a precomputed distance matrix.
    Give either n_clusters or a distance threshold."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method=method)
    if n_clusters is not None:
        return fcluster(Z, t=n_clusters, criterion="maxclust")
    return fcluster(Z, t=threshold, criterion="distance")
