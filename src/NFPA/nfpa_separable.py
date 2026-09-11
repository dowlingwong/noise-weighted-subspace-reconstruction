"""NFPA — noise-factored projection autoencoder, separable (channel x time) form.

The estimator of §5b of the manuscript, in the shape the HeST/QP arm needs:

    x  in R^{C x d}                     one event, C channels x d time samples
    Z  = Sigma_c^{-1/2} x Sigma_t^{-1/2}      separable whitening   [W]
    Gamma = U_c^T Z U_t                       encoder (Sigma^{-1}-adjoint) [E]
    Zhat  = U_c Gamma U_t^T                   decoder (tied)              [D]
    L     = mean ||Z - Zhat||_F^2             = (x-xhat)^dag Sigma^{-1} (x-xhat)

with U_c in St(C, k_c), U_t in St(d, k_t).  Parameter count
k_c(C-k_c) + k_t(d-k_t), against k(Cd-k) for the unrestricted rank-k class.

Two fitting routes, both provided so one certifies the other:

  * ``fit_nfpa_als``   — alternating Rayleigh-Ritz (covariance-weighted
    Tucker-2 / HOOI). Closed form per half-step, monotone, multi-restart.
    This is the reference optimum.
  * ``train_nfpa``     — Grassmann Adam with warm-up + cosine annealing on
    mini-batches. This is the "trained" model; it is *verified* by showing it
    reaches the ALS optimum (principal angles -> 0, loss gap -> 0), which is
    the parameterisation-consistency check the tied linear AE gets in §5.

Setting the whitener to the identity gives Iso-MPCA, the metric ablation.

Pure numpy + scipy. No autodiff: the objective is a Rayleigh quotient in each
factor and its Euclidean gradient is exact in two lines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SeparableWhitener",
    "nfpa_loss",
    "encode",
    "decode",
    "fit_nfpa_als",
    "train_nfpa",
    "TrainConfig",
    "TrainHistory",
    "principal_angles_deg",
    "kron_basis",
    "gauge_fix",
    "GaugeResult",
    "unrestricted_cwpca",
]


# ===========================================================================
# [W] separable whitening
# ===========================================================================
class SeparableWhitener:
    """Sigma_hat = Sigma_c (x) Sigma_t, estimated from noise-only records.

    Sigma_t is taken stationary, so Sigma_t^{-1/2} is a multiply by
    1/sqrt(J_k) in the rfft domain; Sigma_c is a C x C matrix whitened by its
    symmetric inverse square root. Both are NAMED parameters: swapping either
    one is the "re-estimate Sigma" repair, not a gradient step.
    """

    def __init__(self, sigma_c: np.ndarray, psd: np.ndarray, n_time: int,
                 fs: float, ridge: float = 1e-12, dc_guard: int = 1,
                 psd_floor: float = 1e-6):
        self.sigma_c = np.asarray(sigma_c, float)
        self.psd = np.asarray(psd, float)
        self.n_time = int(n_time)
        self.fs = float(fs)
        w, V = np.linalg.eigh(self.sigma_c)
        w = np.maximum(w, ridge * w.max())
        self._Wc = V @ np.diag(w ** -0.5) @ V.T
        self._Cc = V @ np.diag(w ** 0.5) @ V.T
        # DC-bin guard: the generator (and any real DAQ with an unknown
        # baseline) carries no power at f = 0, so 1/sqrt(J_0) would multiply the
        # pulse's large DC component by ~1e9. The optimal filter drops that bin
        # for the same reason; `dc_guard` is how many low bins to drop.
        j = np.maximum(self.psd, psd_floor * self.psd[1:].max())
        self._wt = j ** -0.5
        self._wt[:max(int(dc_guard), 0)] = 0.0
        self.dc_guard = int(dc_guard)
        with np.errstate(divide="ignore"):
            self._ct = np.where(self._wt > 0, 1.0 / np.where(self._wt > 0, self._wt, 1.0), 0.0)
        # scalar gain, calibrated so whitened noise has unit sample variance;
        # fixed to 1 until `calibrate` is called.
        self._gain = 1.0

    def calibrate(self, noise: np.ndarray) -> "SeparableWhitener":
        """Set the scalar gain so that whiten(noise) has RMS 1 per sample."""
        self._gain = 1.0
        self._gain = 1.0 / float(np.std(self.whiten(np.asarray(noise, float))))
        return self

    @classmethod
    def from_noise(cls, noise: np.ndarray, fs: float, **kw) -> "SeparableWhitener":
        """noise: (n, C, d) noise-only records."""
        noise = np.asarray(noise, float)
        n, C, d = noise.shape
        sigma_c = np.einsum("nct,ndt->cd", noise, noise, optimize=True) / (n * d)
        spec = np.abs(np.fft.rfft(noise, axis=-1)) ** 2
        psd = spec.mean(axis=(0, 1)) * 2.0 / (d * fs)
        psd[0] /= 2.0
        if d % 2 == 0:
            psd[-1] /= 2.0
        return cls(sigma_c, psd, d, fs, **kw).calibrate(noise)

    # -- transforms ---------------------------------------------------------
    def whiten(self, x: np.ndarray) -> np.ndarray:
        """(..., C, d) -> whitened (..., C, d)."""
        x = np.asarray(x, float)
        y = np.fft.irfft(np.fft.rfft(x, axis=-1) * (self._wt * self._gain),
                         n=self.n_time, axis=-1)
        return np.einsum("ij,...jt->...it", self._Wc, y)

    def unwhiten(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        y = np.einsum("ij,...jt->...it", self._Cc, z)
        return np.fft.irfft(np.fft.rfft(y, axis=-1) * (self._ct / self._gain),
                            n=self.n_time, axis=-1)

    @property
    def channel_whitener(self) -> np.ndarray:
        return self._Wc

    @property
    def time_weights(self) -> np.ndarray:
        """Frequency-domain multiplier of the time whitener, gain included."""
        return self._wt * self._gain


class IdentityWhitener(SeparableWhitener):
    """Sigma = I. The metric ablation (Iso-MPCA)."""

    def __init__(self, n_channels: int, n_time: int, fs: float = 1.0):
        super().__init__(np.eye(n_channels), np.ones(n_time // 2 + 1), n_time, fs)
        self._wt = np.ones_like(self._wt)
        self._ct = np.ones_like(self._ct)
        self._Wc = np.eye(n_channels)
        self._Cc = np.eye(n_channels)


# ===========================================================================
# [E]/[D] encoder, decoder, loss
# ===========================================================================
def encode(Z: np.ndarray, U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    """(n,C,d) -> (n,k_c,k_t) code."""
    return np.einsum("nct,ck,tl->nkl", Z, U_c, U_t, optimize=True)


def decode(G: np.ndarray, U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    return np.einsum("nkl,ck,tl->nct", G, U_c, U_t, optimize=True)


def nfpa_loss(Z: np.ndarray, U_c: np.ndarray, U_t: np.ndarray,
              per_sample: bool = False, exact: bool = False) -> float | np.ndarray:
    """mean_n ||Z_n - U_c U_c^T Z_n U_t U_t^T||_F^2 / (C d)  (chi^2 per d.o.f.).

    Because both factors are orthonormal the residual is Pythagorean,
    ||Z - P_c Z P_t||^2 = ||Z||^2 - ||U_c^T Z U_t||^2, so the code alone gives
    the loss and the (n, C, d) reconstruction is never formed. `exact=True`
    forms it anyway, as a check.
    """
    n, C, d = Z.shape
    if exact:
        R = Z - decode(encode(Z, U_c, U_t), U_c, U_t)
        return np.mean(R ** 2, axis=(1, 2)) if per_sample else float(np.mean(R ** 2))
    G = encode(Z, U_c, U_t)
    e = np.einsum("nct,nct->n", Z, Z, optimize=True) - np.einsum("nkl,nkl->n", G, G, optimize=True)
    e = e / (C * d)
    return e if per_sample else float(e.mean())


def _gram_c(Z: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    """M_c = mean_n Z_n P_t Z_n^T   (C x C). C is small, so form it."""
    A = np.einsum("nct,tl->ncl", Z, U_t, optimize=True)
    return np.einsum("ncl,ndl->cd", A, A, optimize=True) / len(Z)


def _grad_t(Z: np.ndarray, U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    """M_t U_t with M_t = mean_n Z_n^T P_c Z_n, never forming the d x d matrix."""
    B = np.einsum("nct,ck->nkt", Z, U_c, optimize=True)     # (n, k_c, d)
    Y = np.einsum("nkt,tl->nkl", B, U_t, optimize=True)     # (n, k_c, k_t)
    return np.einsum("nkt,nkl->tl", B, Y, optimize=True) / len(Z)


def _top_eig_t(Z: np.ndarray, U_c: np.ndarray, k: int,
               v0: np.ndarray | None = None) -> np.ndarray:
    """Top-k eigenvectors of M_t = mean_n Z_n^T P_c Z_n via a matrix-free solve."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    n, C, d = Z.shape
    B = np.einsum("nct,ck->nkt", Z, U_c, optimize=True)     # (n, k_c, d)
    Bf = B.reshape(-1, d)                                   # (n k_c, d)
    scale = 1.0 / n

    def mv(v):
        v = np.asarray(v)
        if v.ndim == 1:
            return scale * (Bf.T @ (Bf @ v))
        return scale * (Bf.T @ (Bf @ v))

    op = LinearOperator((d, d), matvec=mv, matmat=mv, dtype=float)
    ncv = min(d - 1, max(4 * k + 8, 24))
    w, V = eigsh(op, k=k, which="LA", ncv=ncv, tol=1e-10,
                 v0=None if v0 is None else np.asarray(v0).ravel()[:d])
    return V[:, np.argsort(w)[::-1]]


def _top_eig_c(Z: np.ndarray, U_t: np.ndarray, k: int) -> np.ndarray:
    w, V = np.linalg.eigh(_gram_c(Z, U_t))
    return V[:, np.argsort(w)[::-1][:k]]


# ===========================================================================
# closed form: alternating Rayleigh-Ritz (covariance-weighted Tucker-2)
# ===========================================================================
def fit_nfpa_als(Z: np.ndarray, k_c: int, k_t: int, n_iter: int = 60,
                 n_restarts: int = 10, tol: float = 1e-12, seed: int = 0,
                 return_history: bool = False):
    """Reference optimum. Returns (U_c, U_t) or (U_c, U_t, history)."""
    rng = np.random.default_rng(seed)
    n, C, d = Z.shape
    best = None
    for r in range(n_restarts):
        U_c = np.linalg.qr(rng.standard_normal((C, k_c)))[0]
        hist, prev = [], np.inf
        U_t = None
        for _ in range(n_iter):
            # each half-step is an exact Rayleigh-Ritz solve for that factor
            U_t = _top_eig_t(Z, U_c, k_t, v0=None if U_t is None else U_t[:, 0])
            U_c = _top_eig_c(Z, U_t, k_c)
            L = nfpa_loss(Z, U_c, U_t)
            hist.append(L)
            if abs(prev - L) < tol * max(prev, 1e-30):
                break
            prev = L
        if best is None or hist[-1] < best[2][-1]:
            best = (U_c.copy(), U_t.copy(), hist)
    if return_history:
        return best
    return best[0], best[1]


def unrestricted_cwpca(Z: np.ndarray, k: int):
    """Rank-k CW-PCA/EMPCA in the same metric: the unrestricted comparator."""
    n, C, d = Z.shape
    F = Z.reshape(n, C * d)
    # d*C is large; use the Gram trick on the sample side
    Gm = F @ F.T / n
    w, V = np.linalg.eigh(Gm)
    idx = np.argsort(w)[::-1][:k]
    U = F.T @ V[:, idx]
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    U = np.linalg.qr(U)[0]
    return U


# ===========================================================================
# trained NFPA: Grassmann Adam + warm-up/cosine schedule
# ===========================================================================
@dataclass
class TrainConfig:
    k_c: int = 3
    k_t: int = 3
    epochs: int = 60
    batch_size: int = 128
    lr: float = 0.35
    warmup_frac: float = 0.08
    min_lr_frac: float = 1e-3
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-12
    optimizer: str = "momentum"   # "momentum" (Riemannian heavy-ball) or "adam"
    n_restarts: int = 10
    seed: int = 0
    eval_every: int = 1


@dataclass
class TrainHistory:
    step_lr: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    angle_to_als: list = field(default_factory=list)
    epoch: list = field(default_factory=list)
    restart_final: list = field(default_factory=list)


def _cosine_lr(step: int, total: int, peak: float, warmup: int, min_frac: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + np.cos(np.pi * p)))


def _grassmann_project(U: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Project an ambient gradient onto the horizontal space at U."""
    return G - U @ (U.T @ G)


def _unit_step(S: np.ndarray) -> np.ndarray:
    """Scale a search direction to Frobenius norm sqrt(k): lr is then an angle."""
    nrm = np.linalg.norm(S)
    if nrm < 1e-30:
        return S
    return S * (np.sqrt(S.shape[1]) / nrm)


def _retract(U: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(U)
    return Q * np.sign(np.where(np.diag(R) == 0, 1.0, np.diag(R)))


def train_nfpa(Z_train: np.ndarray, Z_val: np.ndarray, cfg: TrainConfig,
               reference: tuple | None = None, verbose: bool = True):
    """Mini-batch Grassmann-Adam training of the tied separable AE.

    `reference` = (U_c*, U_t*) from ALS; when given, the mean principal angle
    to it is logged every epoch, which is the convergence certificate.
    """
    n, C, d = Z_train.shape
    steps_per_epoch = max(n // cfg.batch_size, 1)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(cfg.warmup_frac * total_steps)
    hist = TrainHistory()
    best = None

    for restart in range(cfg.n_restarts):
        rng = np.random.default_rng(cfg.seed + 1000 * restart)
        U_c = np.linalg.qr(rng.standard_normal((C, cfg.k_c)))[0]
        U_t = np.linalg.qr(rng.standard_normal((d, cfg.k_t)))[0]
        m_c = np.zeros_like(U_c); v_c = np.zeros_like(U_c)
        m_t = np.zeros_like(U_t); v_t = np.zeros_like(U_t)
        step = 0
        first = restart == 0
        for ep in range(cfg.epochs):
            order = rng.permutation(n)
            for b in range(steps_per_epoch):
                idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                Zb = Z_train[idx]
                lr = _cosine_lr(step, total_steps, cfg.lr, warmup, cfg.min_lr_frac)
                # dL/dU = -2 M U  (M = the factor's second-moment matrix)
                Gc = -2.0 * (_gram_c(Zb, U_t) @ U_c)
                Gt = -2.0 * _grad_t(Zb, U_c, U_t)
                Gc = _grassmann_project(U_c, Gc)
                Gt = _grassmann_project(U_t, Gt)
                b1, b2 = cfg.betas
                step += 1
                m_c = b1 * m_c + (1 - b1) * Gc
                v_c = b2 * v_c + (1 - b2) * Gc ** 2
                m_t = b1 * m_t + (1 - b1) * Gt
                v_t = b2 * v_t + (1 - b2) * Gt ** 2
                mhc = m_c / (1 - b1 ** step); mht = m_t / (1 - b1 ** step)
                if cfg.optimizer == "adam":
                    vhc = v_c / (1 - b2 ** step); vht = v_t / (1 - b2 ** step)
                    dc = mhc / (np.sqrt(vhc) + cfg.eps)
                    dt_ = mht / (np.sqrt(vht) + cfg.eps)
                else:                      # Riemannian heavy-ball
                    dc, dt_ = mhc, mht
                # trust-region step: the optimiser sets the direction, lr sets
                # the rotation angle. Without the rescale a per-entry step of
                # size lr rotates a d=2048 column by ~lr*sqrt(d) radians.
                sc = _unit_step(_grassmann_project(U_c, dc))
                st = _unit_step(_grassmann_project(U_t, dt_))
                U_c = _retract(U_c - lr * sc)
                U_t = _retract(U_t - lr * st)
                # transport the moments back to the new horizontal space
                m_c = _grassmann_project(U_c, m_c)
                m_t = _grassmann_project(U_t, m_t)
                if first:
                    hist.step_lr.append(lr)
            if first and (ep % cfg.eval_every == 0 or ep == cfg.epochs - 1):
                tr = nfpa_loss(Z_train, U_c, U_t)
                va = nfpa_loss(Z_val, U_c, U_t)
                hist.epoch.append(ep)
                hist.train_loss.append(tr)
                hist.val_loss.append(va)
                if reference is not None:
                    a = principal_angles_deg(kron_basis(U_c, U_t),
                                             kron_basis(*reference))
                    hist.angle_to_als.append(float(np.mean(a)))
                if verbose and ep % max(cfg.epochs // 6, 1) == 0:
                    msg = f"  epoch {ep:3d}  train {tr:.6f}  val {va:.6f}"
                    if reference is not None:
                        msg += f"  angle_to_ALS {hist.angle_to_als[-1]:7.3f} deg"
                    print(msg, flush=True)
        fin = nfpa_loss(Z_val, U_c, U_t)
        hist.restart_final.append(float(fin))
        if best is None or fin < best[2]:
            best = (U_c.copy(), U_t.copy(), fin)
    return best[0], best[1], hist


# ===========================================================================
# subspace geometry + gauge fixing (the identification step)
# ===========================================================================
def kron_basis(U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    """(C k_c, d k_t) -> (C d, k_c k_t) basis of the Kronecker subspace."""
    return np.column_stack([np.kron(U_c[:, j], U_t[:, i])
                            for j in range(U_c.shape[1])
                            for i in range(U_t.shape[1])])


def principal_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(np.abs(s), 0.0, 1.0)))


def captured_fraction(U: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each column b_j of B, the fraction of its norm inside span(U).

    ||U U^T b|| / ||b||. Invariant to any rotation of U within its span, so it
    answers "has the fit picked this physical direction up yet" without
    pretending the fitted axes come out in any particular order.
    """
    Q = np.linalg.qr(U)[0]
    proj = Q @ (Q.T @ B)
    return np.linalg.norm(proj, axis=0) / np.linalg.norm(B, axis=0)


@dataclass
class GaugeResult:
    A_c: np.ndarray          # k_c x p_c   fitted-basis -> physical tangent
    A_t: np.ndarray
    cond_c: float
    cond_t: float
    resid_c: float           # fraction of physical tangent OUTSIDE the fit
    resid_t: float
    angles_c: np.ndarray
    angles_t: np.ndarray

    def physical_code(self, G: np.ndarray) -> np.ndarray:
        """Learned code Gamma -> coefficients in the physical tangent basis."""
        Ac_inv = np.linalg.pinv(self.A_c)
        At_inv = np.linalg.pinv(self.A_t)
        return np.einsum("pk,nkl,ql->npq", Ac_inv, G, At_inv, optimize=True)


def gauge_fix(U_c: np.ndarray, U_t: np.ndarray,
              B_c: np.ndarray, B_t: np.ndarray) -> GaugeResult:
    """Express the learned factors in a physical tangent basis.

    B_c : (C, p_c) whitened channel tangent, e.g. [h, d_x h, d_y h]
    B_t : (d, p_t) whitened time tangent,    e.g. [s, d_t s, d_tau s]

    Columns are normalised first: the physical derivatives carry wildly
    different units (d_tau s is per-nanosecond), and without normalisation
    kappa would report the unit system rather than the geometry. The physical
    code is then in units of the normalised basis vectors, which a free slope
    in the read-out regression absorbs.
    """
    B_c = B_c / np.linalg.norm(B_c, axis=0, keepdims=True)
    B_t = B_t / np.linalg.norm(B_t, axis=0, keepdims=True)
    A_c = U_c.T @ B_c
    A_t = U_t.T @ B_t
    rc = np.linalg.norm(B_c - U_c @ A_c) / np.linalg.norm(B_c)
    rt = np.linalg.norm(B_t - U_t @ A_t) / np.linalg.norm(B_t)
    return GaugeResult(
        A_c=A_c, A_t=A_t,
        cond_c=float(np.linalg.cond(A_c)), cond_t=float(np.linalg.cond(A_t)),
        resid_c=float(rc), resid_t=float(rt),
        angles_c=principal_angles_deg(U_c, B_c),
        angles_t=principal_angles_deg(U_t, B_t),
    )
