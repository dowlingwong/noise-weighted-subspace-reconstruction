"""Physical bases for the HeST/QP arm, and the factorisation check.

What NFPA is asked to satisfy is that, to first order, one event is

    x  ~  A * h(x_v, y_v)  (x)  s(t - t0 ; tau)

    h(x_v, y_v) ~ h0 + x_v dh/dx + y_v dh/dy        channel factor, C-dim
    s(t - t0; tau) ~ s + t0 ds/dt + dtau ds/dtau    time factor, d-dim

so the admissible set is a Kronecker product of two three-dimensional tangent
spaces. These helpers build both bases from the simulator -- never from the
fit -- extend each by its next order, and measure how well the full
HeST -> QPSimulator chain actually factorises.
"""
from __future__ import annotations

import numpy as np

from QPSimulator import QPSimulator

# channel: monopole, two position dipoles, three quadrupoles
MULTIPOLE_LABELS = [r"$h_0$", r"$\partial_x h$", r"$\partial_y h$",
                    r"$q_{xy}$", r"$q_{x^2-y^2}$", r"$q_{r^2}$"]
# time: pulse, arrival-time and decay-shape derivatives, then second order
TIME_LABELS = [r"$s$", r"$\partial_t s$", r"$\partial_\tau s$",
               r"$\partial^2_{tt} s$", r"$\partial^2_{t\tau} s$",
               r"$\partial^2_{\tau\tau} s$"]


# --------------------------------------------------------------------------
# time factor
# --------------------------------------------------------------------------
def time_tangent(fs: float, n_time: int, tau_rise: float, tau_nom: float,
                 trigger_time: float, dt0: float = 4e3, dtau_frac: float = 0.05,
                 order: int = 1) -> np.ndarray:
    """[s, ds/dt0, ds/dtau] (order 1) plus the three second-order terms.

    Finite differences on the QPSimulator response itself, so the basis is
    whatever the simulator actually produces, including its sampling.
    """
    def tmpl(t0_shift, tau):
        sim = QPSimulator(sampling_frequency=fs, trace_samples=n_time,
                          tau_rise=tau_rise, tau_decay=tau,
                          trigger_time=trigger_time + t0_shift)
        return sim.template[:n_time].copy()

    h = dtau_frac * tau_nom
    s = tmpl(0.0, tau_nom)
    s_tp, s_tm = tmpl(+dt0, tau_nom), tmpl(-dt0, tau_nom)
    s_ap, s_am = tmpl(0.0, tau_nom + h), tmpl(0.0, tau_nom - h)
    cols = [s, (s_tp - s_tm) / (2 * dt0), (s_ap - s_am) / (2 * h)]
    if order >= 2:
        cols.append((s_tp - 2 * s + s_tm) / dt0 ** 2)
        cols.append((tmpl(+dt0, tau_nom + h) - tmpl(+dt0, tau_nom - h)
                     - tmpl(-dt0, tau_nom + h) + tmpl(-dt0, tau_nom - h))
                    / (4 * dt0 * h))
        cols.append((s_ap - 2 * s + s_am) / h ** 2)
    return np.column_stack(cols)


def whiten_columns(B: np.ndarray, time_weights: np.ndarray, n_time: int) -> np.ndarray:
    """Apply the stationary time whitener to each column of a (d, p) basis."""
    Bf = np.fft.rfft(B, axis=0) * time_weights[:, None]
    return np.fft.irfft(Bf, n=n_time, axis=0)


# --------------------------------------------------------------------------
# channel factor
# --------------------------------------------------------------------------
def multipole_basis(share: np.ndarray, x: np.ndarray, y: np.ndarray,
                    n_terms: int = 6):
    """Multipole expansion of the CPD collection pattern h(x, y).

    Columns respond to 1, x, y, xy, x^2-y^2, x^2+y^2: the monopole, the two
    position dipoles, then the three quadrupoles. The first three are the
    tangent basis; the rest measure how curved the position response is over
    the sampled volume. Fitted on a high-statistics calibration grid, so it is
    a measured detector property, not an artefact of the events being modelled.

    Returns (B, r2_cumulative) with B of shape (C, n_terms).
    """
    cols = [np.ones_like(x), x, y, x * y, x ** 2 - y ** 2, x ** 2 + y ** 2]
    D = np.column_stack(cols[:n_terms])
    # Orthogonalise the *polynomials* over the grid, not the response vectors.
    # On a symmetric grid this leaves 1, x and y alone and only centres the
    # radial quadrupole (r^2 - <r^2>), which is what decouples it from the
    # monopole. The event coefficients stay exactly (1, x_v, y_v, ...), so the
    # position read-out c_1/c_0 remains the vertex by construction -- whereas
    # orthogonalising the response vectors would mix c_0 and break it.
    Q, R = np.linalg.qr(D)
    Q = Q * np.sign(np.diag(R))
    coef, *_ = np.linalg.lstsq(Q, share, rcond=None)
    tot = ((share - share.mean(0)) ** 2).sum()
    r2 = [float(1 - ((share - Q[:, :p] @ np.linalg.lstsq(Q[:, :p], share,
                                                         rcond=None)[0]) ** 2).sum() / tot)
          for p in range(1, n_terms + 1)]
    return coef.T, np.array(r2)


# --------------------------------------------------------------------------
# does the chain factorise?
# --------------------------------------------------------------------------
def rank1_surrogate(share: np.ndarray, n_det: np.ndarray, t0: np.ndarray,
                    tau: np.ndarray, qp_amplitude: float, qp_fraction: float,
                    fs: float, n_time: int, tau_rise: float,
                    trigger_time: float) -> np.ndarray:
    """A * h (x) s(t - t0; tau): every QP treated as arriving at t = 0.

    The difference from the rendered traces is exactly the part of the event
    that does *not* factorise -- the spread of quasiparticle arrival times.
    """
    n, C = share.shape
    out = np.empty((n, C, n_time), dtype=np.float32)
    for i in range(n):
        sim = QPSimulator(sampling_frequency=fs, trace_samples=n_time,
                          tau_rise=tau_rise, tau_decay=float(tau[i]),
                          trigger_time=trigger_time + float(t0[i]))
        s = sim.template[:n_time]
        amp = n_det[i] * qp_amplitude / qp_fraction
        out[i] = (amp * share[i][:, None] * s[None, :]).astype(np.float32)
    return out


def factorisation_error(clean: np.ndarray, surrogate: np.ndarray) -> dict:
    num = np.linalg.norm((clean - surrogate).reshape(len(clean), -1), axis=1)
    den = np.linalg.norm(clean.reshape(len(clean), -1), axis=1)
    rel = num / np.maximum(den, 1e-30)
    return dict(median=float(np.median(rel)), p90=float(np.percentile(rel, 90)),
                mean=float(rel.mean()), per_event=rel)


# --------------------------------------------------------------------------
# read-out helper
# --------------------------------------------------------------------------
def linear_r2(truth: np.ndarray, estimate: np.ndarray):
    """R^2 of the best affine fit estimate ~ a*truth + b, plus (a, b)."""
    truth = np.asarray(truth, float)
    estimate = np.asarray(estimate, float)
    a, b = np.polyfit(truth, estimate, 1)
    ss = np.sum((estimate - (a * truth + b)) ** 2)
    return float(1 - ss / np.sum((estimate - estimate.mean()) ** 2)), float(a), float(b)

def orthonormalise(B: np.ndarray) -> np.ndarray:
    """Modified Gram-Schmidt, preserving column order and orientation.

    The monopole and the radial quadrupole of a collection pattern overlap at
    -0.90 (both are radially symmetric), which makes the gauge matrix
    ill-conditioned and leaks quadrupole content into the monopole coefficient
    the position read-out divides by. Orthogonalising in physical order keeps
    the names -- column j is "the part of multipole j not already in the lower
    ones" -- and decouples them.
    """
    B = np.asarray(B, float).copy()
    for j in range(B.shape[1]):
        for i in range(j):
            B[:, j] -= (B[:, i] @ B[:, j]) * B[:, i]
        n = np.linalg.norm(B[:, j])
        if n > 0:
            B[:, j] /= n
    return B
