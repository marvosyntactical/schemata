"""Gauge-free multi-channel schema embedding read off a trained AL-RNN module.

Each channel is featurised so that dot-product similarity approximates schema
similarity (primer §4.3; dsa_native_ssm.md §1):

  Koopman channel  : sorted linear-core eigenvalues (|.| and angle) -> temporal
                     structure, invariant to latent rotation (eigenvalues are).
  symbolic channel : transition-graph invariants (n_symbols, topological entropy,
                     periodic-orbit counts tr(T^n)) -> topology, invariant to
                     latent basis (graph isomorphism is gauge-free).
  dynamical channel: model Lyapunov spectrum + Kaplan-Yorke dim -> chaos/periodic/
                     quasiperiodic regime, from the exact PWL Jacobian.

The three are z-scored per channel by the SchemaMemory (which holds the running
stats); here we just produce the raw feature vector.
"""
import numpy as np
import metrics

# fixed channel layout for the default k=4, nmax=4 (koopman 2k=8, symbolic 2+nmax=6,
# dynamical 4). Used by the memory for channel-weighted distance.
SLICES = {"koopman": (0, 8), "symbolic": (8, 14), "dynamical": (14, 18)}


def _koopman_feats(model, k=4):
    ev = model.linear_core_spectrum()
    mag = np.abs(ev)[:k]
    ang = np.abs(np.angle(ev))[:k]            # |angle|: reflection-invariant
    mag = np.pad(mag, (0, k - len(mag)))
    ang = np.pad(ang, (0, k - len(ang)))
    return np.concatenate([mag, ang])          # 2k


def _symbolic_feats(model, seed_obs, nmax=4):
    _, uniq, T = model.itinerary(seed_obs)
    n_sym = len(uniq)
    # topological entropy ~ log spectral radius of the transition matrix
    if T.shape[0] >= 1:
        rho = np.max(np.abs(np.linalg.eigvals(T)))
    else:
        rho = 1.0
    entropy = np.log(max(rho, 1e-9))
    # periodic-point growth: normalised tr(T^n) for n=1..nmax (zeta coefficients)
    traces = []
    Tp = np.eye(T.shape[0])
    for _ in range(nmax):
        Tp = Tp @ T
        traces.append(np.trace(Tp))
    return np.concatenate([[n_sym, entropy], traces])   # 2 + nmax


def _dynamical_feats(model, seed_obs):
    spec = metrics.lyapunov_spectrum(model, seed_obs, n=2500, warmup=400)
    ky = metrics.kaplan_yorke(spec)
    lam_max = spec[0]
    n_pos = int((spec > 1e-3).sum())            # # positive exponents
    n_zero = int((np.abs(spec) <= 1e-3).sum())  # # near-zero (flow/torus dirs)
    return np.array([lam_max, ky, n_pos, n_zero])   # 4


def embed(model, seed_obs):
    """Concatenated raw embedding. Channel slices are fixed so the memory can
    z-score and weight per channel."""
    k = _koopman_feats(model)
    s = _symbolic_feats(model, seed_obs)
    d = _dynamical_feats(model, seed_obs)
    vec = np.concatenate([k, s, d]).astype(np.float64)
    slices = {"koopman": (0, len(k)),
              "symbolic": (len(k), len(k) + len(s)),
              "dynamical": (len(k) + len(s), len(vec))}
    return vec, slices
