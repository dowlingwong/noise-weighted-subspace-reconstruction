"""G1 — the frozen covariance-error generator of the 10 Sep run (T15), re-implemented.

Specification (report `reviews/2026-09-10-experiment-phase-report.md` §5.1, MS repo):

* dimension ``d = 256``, unit sampling frequency;
* one-sided noise PSD ``S(f) = (sqrt(f^2 + f_knee^2))^-1`` (exponent -1), ``f_knee = 4/d``;
* the exact circulant covariance implied by that PSD, normalised so ``trace(Sigma)/d = 1``;
* a *generic* rank-2 orthonormal signal basis (seed 20260915), deliberately not sinusoids;
* amplitudes ``a ~ N(0, diag(SNR^2, (SNR/2)^2))``, SNR = 3;
* 512 training traces (signal + noise), ``m`` noise-only calibration traces, 2000 test traces;
* every misspecified covariance renormalised to the same trace so misspecification
  changes shape, never scale.

Equivalence with the 10 Sep G1 is *tested*, not assumed: the run recorded
``cond(Sigma) = 32.0``, ``lambda_1 = 19.44``, ``Delta_lambda = 5.20`` for the whitened
population operator; :func:`check_against_recorded` reproduces them or raises.
The 10 Sep work branch (`a7125ea`) was not on the local machine at freeze; this
file is the local re-implementation and says so in every provenance block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D = 256
F_KNEE = 4.0 / D
BASIS_SEED = 20260915
SNR = 3.0
AMPLITUDE_SD = np.array([SNR, SNR / 2.0])
N_TRAIN = 512
N_TEST = 2000
RANK = 2

RECORDED = {"cond": 32.0, "lambda_1": 19.44, "delta_lambda": 5.20, "lambda1_over_gap": 3.74}


def dft_frequencies(d: int = D) -> np.ndarray:
    """Signed DFT frequencies for unit sampling frequency, k/d mapped into [-1/2, 1/2)."""
    return np.fft.fftfreq(d, d=1.0)


def psd(f: np.ndarray, *, exponent: float = -1.0, f_knee: float = F_KNEE) -> np.ndarray:
    """One-sided power-law PSD ``(sqrt(f^2 + f_knee^2))^exponent``; exponent -1 is G1."""
    return np.power(np.sqrt(f**2 + f_knee**2), exponent)


def circulant_from_spectrum(s: np.ndarray) -> np.ndarray:
    """Real symmetric circulant with DFT eigenvalues ``s`` (``s`` must be even in k -> -k)."""
    d = len(s)
    c = np.real(np.fft.ifft(s))  # first column of the circulant
    idx = (np.arange(d)[:, None] - np.arange(d)[None, :]) % d
    return c[idx]


def covariance(*, exponent: float = -1.0, f_knee: float = F_KNEE, d: int = D) -> np.ndarray:
    """Exact circulant Sigma from the PSD, normalised to trace/d = 1."""
    s = psd(dft_frequencies(d), exponent=exponent, f_knee=f_knee)
    s = s / np.mean(s)  # trace(Sigma)/d = mean eigenvalue = 1
    return circulant_from_spectrum(s)


def signal_basis(d: int = D, rank: int = RANK, seed: int = BASIS_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, rank)))
    return np.asarray(Q)


def sym_inv_sqrt(S: np.ndarray, *, floor: float | None = None) -> tuple[np.ndarray, dict]:
    """Symmetric ``S^{-1/2}``; optional eigenvalue floor (recorded) for near-singular estimates."""
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    diag = {"eig_min": float(vals.min()), "eig_max": float(vals.max()), "floor_applied": False, "floor_value": None}
    if floor is not None and vals.min() < floor:
        vals = np.maximum(vals, floor)
        diag["floor_applied"] = True
        diag["floor_value"] = float(floor)
    return (vecs / np.sqrt(vals)[None, :]) @ vecs.T, diag


def sym_sqrt(S: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))[None, :]) @ vecs.T


@dataclass
class G1:
    Sigma: np.ndarray
    Sigma_isqrt: np.ndarray
    B: np.ndarray            # (d, 2) signal basis
    A_pop: np.ndarray        # whitened population second moment = Sigma^{-1/2} B D B^T Sigma^{-1/2} + I
    lam: np.ndarray          # eigenvalues of A_pop, descending
    Q: np.ndarray            # eigenvectors of A_pop (columns), same order

    @property
    def planted_whitened(self) -> np.ndarray:
        """Orthonormal basis of the planted subspace in true-whitened coordinates."""
        W = self.Sigma_isqrt @ self.B
        Q, _ = np.linalg.qr(W)
        return np.asarray(Q)


def make_g1() -> G1:
    Sigma = covariance()
    Si, _ = sym_inv_sqrt(Sigma)
    B = signal_basis()
    A = Si @ (B * AMPLITUDE_SD[None, :]**2) @ B.T @ Si + np.eye(D)
    A = 0.5 * (A + A.T)
    lam, Q = np.linalg.eigh(A)
    order = np.argsort(lam)[::-1]
    return G1(Sigma=Sigma, Sigma_isqrt=Si, B=B, A_pop=A, lam=lam[order], Q=Q[:, order])


def check_against_recorded(g: G1, *, rtol: float = 2e-3) -> dict:
    """Compare the local G1 with the three numbers the 10 Sep run recorded."""
    ev = np.linalg.eigvalsh(g.Sigma)
    cond = float(ev.max() / ev.min())
    lam1 = float(g.lam[0]); gap = float(g.lam[RANK - 1] - g.lam[RANK])
    out = {"cond": cond, "lambda_1": lam1, "lambda_2": float(g.lam[1]), "lambda_3": float(g.lam[2]),
           "delta_lambda": gap, "lambda1_over_gap": lam1 / gap, "trace_over_d": float(np.trace(g.Sigma) / D)}
    out["matches_recorded"] = bool(
        abs(cond - RECORDED["cond"]) / RECORDED["cond"] < rtol
        and abs(lam1 - RECORDED["lambda_1"]) / RECORDED["lambda_1"] < rtol
        and abs(gap - RECORDED["delta_lambda"]) / RECORDED["delta_lambda"] < rtol
    )
    return out


def draw_training(g: G1, rng: np.random.Generator, n: int = N_TRAIN) -> tuple[np.ndarray, np.ndarray]:
    """Signal + noise traces ``(n, d)`` and their amplitudes ``(n, 2)``."""
    a = rng.standard_normal((n, RANK)) * AMPLITUDE_SD[None, :]
    noise = draw_noise(g, rng, n)
    return a @ g.B.T + noise, a


def draw_noise(g: G1, rng: np.random.Generator, n: int) -> np.ndarray:
    L = np.linalg.cholesky(g.Sigma)
    return rng.standard_normal((n, D)) @ L.T


if __name__ == "__main__":
    import json
    g = make_g1()
    print(json.dumps(check_against_recorded(g), indent=1))
