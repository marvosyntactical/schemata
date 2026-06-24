"""Artificial chaotic systems for the DSR testbed (primer §0/§4.5).

Ground-truth attractors with known topology. Lorenz and Rossler are the
topologically-distinct ("different schema") pair: similar surface statistics,
inequivalent branched-manifold templates.
"""
import numpy as np
from scipy.integrate import solve_ivp


def _lorenz_rhs(t, z, sigma, rho, beta):
    x, y, w = z
    return [sigma * (y - x), x * (rho - w) - y, x * y - beta * w]


def _rossler_rhs(t, z, a, b, c):
    x, y, w = z
    return [-y - w, x + a * y, b + w * (x - c)]


def _integrate(rhs, z0, args, dt, n, transient, seed=0):
    """Integrate, drop transient, return (n, 3) standardised trajectory."""
    t_total = (n + transient) * dt
    t_eval = np.arange(0, t_total, dt)
    sol = solve_ivp(rhs, (0, t_total), z0, args=args, t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-9)
    traj = sol.y.T[transient:transient + n]
    mu, sd = traj.mean(0), traj.std(0)
    return (traj - mu) / sd, (mu, sd)


def lorenz(n=20000, dt=0.01, transient=5000, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    return _integrate(_lorenz_rhs, [1.0, 1.0, 1.0], (sigma, rho, beta),
                      dt, n, transient)


def rossler(n=20000, dt=0.05, transient=5000, a=0.2, b=0.2, c=5.7):
    return _integrate(_rossler_rhs, [1.0, 1.0, 1.0], (a, b, c),
                      dt, n, transient)


def _limit_cycle_rhs(t, z, omega):
    # stable limit cycle of radius 1 in (x,y); z decays -> dim-1 attractor
    x, y, w = z
    r2 = x * x + y * y
    return [-omega * y + x * (1 - r2), omega * x + y * (1 - r2), -0.6 * w]


def limit_cycle(n=20000, dt=0.05, transient=4000, omega=1.0):
    return _integrate(_limit_cycle_rhs, [0.1, 0.0, 1.0], (omega,),
                      dt, n, transient)


def torus(n=20000, dt=0.05, transient=2000, omega1=1.0, omega2=0.5 * (5 ** 0.5 - 1)):
    """Quasiperiodic 2-frequency signal (incommensurate omega1:omega2). The
    invariant set is a 2-torus (dim 2, two zero Lyapunov exponents, zero
    topological entropy) -- topologically distinct from both a limit cycle and
    a chaotic attractor. Generated analytically (no ODE needed)."""
    import numpy as np
    t = np.arange(n + transient) * dt
    x = np.cos(omega1 * t) + 0.5 * np.cos(omega2 * t)
    y = np.sin(omega1 * t) + 0.5 * np.sin(omega2 * t)
    w = 0.5 * np.sin(omega1 * t) * np.cos(omega2 * t)
    traj = np.stack([x, y, w], axis=1)[transient:]
    mu, sd = traj.mean(0), traj.std(0)
    return (traj - mu) / sd, (mu, sd)


def van_der_pol(n=20000, dt=0.05, transient=4000, mu=2.5):
    """Van der Pol relaxation oscillator in 3D (third coordinate decays to 0,
    giving a 2D attractor embedded in 3D, identical to limit_cycle in that
    respect). mu=2.5 gives strong relaxation: sharp spikes in x followed by
    slow recovery -- a limit cycle (beta_1=1, one zero Lyapunov) that is
    topologically identical to limit_cycle but whose waveform, time-scale, and
    rate descriptors differ markedly. Tests whether the gamma knob correctly
    groups them (gamma=0) or separates them (gamma=1)."""
    def rhs(t, z, mu):
        x, y, w = z
        return [mu * (x - x ** 3 / 3.0 - y), x / mu, -0.5 * w]
    return _integrate(rhs, [2.0, 0.0, 1.0], (mu,), dt, n, transient)


def halvorsen(n=20000, dt=0.05, transient=5000, a=1.4):
    """Halvorsen attractor (chaotic for a≈1.4, three-fold rotationally symmetric).
        dx/dt = -a*x - 4*y - 4*z - y²
        dy/dt = -a*y - 4*z - 4*x - z²
        dz/dt = -a*z - 4*x - 4*y - x²
    Visually distinct 3-lobed spiral; no branched-manifold, different topo class
    from Lorenz."""
    def rhs(t, z, a):
        x, y, w = z
        return [-a*x - 4*y - 4*w - y**2,
                -a*y - 4*w - 4*x - w**2,
                -a*w - 4*x - 4*y - x**2]
    return _integrate(rhs, [1.0, 0.0, 0.0], (a,), dt, n, transient)


def thomas(n=20000, dt=0.05, transient=5000, b=0.19):
    """Thomas' cyclically symmetric attractor (chaotic for b≈0.19).
        dx/dt = sin(y) - b*x
        dy/dt = sin(z) - b*y
        dz/dt = sin(x) - b*z
    Topologically distinct from Lorenz (no branched manifold) and Rössler
    (3-fold symmetry, complex multi-scroll structure)."""
    def rhs(t, z, b):
        x, y, w = z
        return [np.sin(y) - b*x, np.sin(w) - b*y, np.sin(x) - b*w]
    return _integrate(rhs, [0.1, 0.0, 0.0], (b,), dt, n, transient)


SYSTEMS = {"lorenz": lorenz, "rossler": rossler,
           "limit_cycle": limit_cycle, "torus": torus,
           "van_der_pol": van_der_pol, "thomas": thomas,
           "halvorsen": halvorsen}


# --- within-class variants and the labelled continual-learning stream ---

def affine_variant(data, seed, noise=0.02):
    """Apply a random rotation+scale+translation (a smooth conjugacy / coordinate
    change) plus observation noise. Same schema, different observation frame -- the
    nuisance group G the within-class chart must absorb."""
    import numpy as np
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)                      # random rotation/reflection
    scale = rng.uniform(0.7, 1.4, size=3)
    b = rng.uniform(-0.5, 0.5, size=3)
    out = (data @ Q.T) * scale + b
    out = out + noise * rng.standard_normal(out.shape)
    return (out - out.mean(0)) / out.std(0)


def canonicalize(data):
    """Rotate to a canonical frame (PCA principal axes, descending variance) and
    sign-fix each axis by its skew, then re-standardise. Removes the rotation/
    reflection gauge so the symbolic partition and fitted operator are comparable
    across variants (dsa_native_ssm.md §5.B-B1, the whitening move). NB: leaves a
    residual in-plane ambiguity for rotationally-symmetric attractors (degenerate
    PCA variances) -- a known limit, surfaced in the per-channel diagnostics."""
    import numpy as np
    X = data - data.mean(0)
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(-w)]                    # principal axes, descending variance
    Y = X @ V
    skew = (Y ** 3).mean(0)
    Y = Y * np.sign(skew + 1e-12)               # sign-fix reflections
    return (Y / Y.std(0)).astype("float32")


def make_stream(seed=0, n=12000):
    """A labelled CL stream over topologically-distinct, reconstructable schema
    classes, each appearing several times as within-class variants in a shuffled
    order. Returns list of (data, schema_label). Ground truth for evaluation."""
    import numpy as np
    base = {"limit_cycle": limit_cycle, "torus": torus, "rossler": rossler}
    # interleave so reuse decisions are non-trivial (same class recurs after others)
    order = ["limit_cycle", "rossler", "torus", "limit_cycle", "rossler",
             "torus", "rossler", "limit_cycle"]
    stream = []
    counters = {k: 0 for k in base}
    for i, name in enumerate(order):
        data, _ = base[name](n=n)
        v = affine_variant(data, seed=1000 * seed + i, noise=0.02)
        counters[name] += 1
        stream.append((v.astype("float32"), name))
    return stream

