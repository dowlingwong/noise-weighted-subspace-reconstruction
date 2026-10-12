"""D3 signal model for the semi-synthetic injection (runbook §6.4, design §D.2).

    s = sum_{j=1,2} sum_{l=1,2,3} alpha_jl (b_jl(theta) (x) a_j)          (record index c + C t)

* spatial templates a_1, a_2 in R^C: the subject's face-evoked field (reference runs 1-2 only, selected
  channels) at the global-field-power peak inside template_windows_s[0] (~100 ms) and [1] (~170 ms),
  unit norm, sign fixed (largest-magnitude entry positive); flagged degenerate if |cos(a_1, a_2)| exceeds
  max_template_cosine or both peaks coincide (fallback: spherical-head dipoles, not implemented here);
* temporal basis of source 1: B_1 = orth[g, g', g''] for a Gaussian bump g at bump_centers_s[0] with
  standard deviation bump_width_s (amplitude, latency and width variation); the same family at
  bump_centers_s[1], projected off span(B_1) and orthonormalised, gives C_b;
  source 2 uses B_2(theta) = cos(theta) B_1 + sin(theta) C_b (orthonormal for every theta).
  At theta = 0 the signal span is exactly span(B_1) (x) span{a_1, a_2}: k = 6, (k_c, k_t) = (2, 3);
* Gaussian latents alpha_jl ~ N(0, v_jl), v_jl = scale * source_weight_j * latent_variance_l, with ONE scale
  per subject, fixed so that tr K = tr(Sigma_ref^{-1} R_s) = trace_K_target at theta = 0; the same latent
  variances are used at every theta (no per-theta rescaling: B_cert scales with tr K, so rescaling would mix
  scale and geometry in PZ1; review (c)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import symmetric


def fix_signs(V: np.ndarray) -> np.ndarray:
    V = np.atleast_2d(V)
    idx = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[idx, np.arange(V.shape[1])])
    s[s == 0] = 1.0
    return V * s


def orth(B: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(B)
    if np.min(np.abs(np.diag(r))) < 1e-12 * max(1.0, np.max(np.abs(np.diag(r)))):
        raise ValueError('rank loss in orthonormalisation')
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


def spatial_templates(evoked: np.ndarray, sfreq: float, windows_s, max_cosine: float = 0.95) -> dict:
    evoked = np.asarray(evoked, float)
    C, T = evoked.shape
    times = np.arange(T) / sfreq
    gfp = np.sqrt(np.mean(evoked ** 2, axis=0))
    cols, peaks = [], []
    for lo, hi in windows_s:
        idx = np.nonzero((times >= lo - 1e-12) & (times <= hi + 1e-12))[0]
        if len(idx) == 0:
            raise ValueError(f'template window [{lo}, {hi}] s contains no sample at {sfreq} Hz')
        i = int(idx[np.argmax(gfp[idx])])
        a = evoked[:, i]
        nrm = np.linalg.norm(a)
        if nrm <= 0:
            raise ValueError('zero evoked field at the template peak')
        cols.append(fix_signs(a[:, None] / nrm)[:, 0])
        peaks.append(i)
    A = np.stack(cols, 1)
    cos = float(abs(A[:, 0] @ A[:, 1])) if A.shape[1] > 1 else 0.0
    return {'A': A, 'peak_samples': peaks, 'peak_times_s': [p / sfreq for p in peaks], 'abs_cosine': cos,
            'degenerate': bool(cos > max_cosine or len(set(peaks)) < len(peaks)), 'gfp': gfp}


def bump_family(T: int, sfreq: float, center_s: float, width_s: float) -> np.ndarray:
    t = np.arange(T) / sfreq
    u = (t - center_s) / width_s
    g = np.exp(-0.5 * u ** 2)
    g1 = -u / width_s * g
    g2 = (u ** 2 - 1.0) / width_s ** 2 * g
    return np.stack([g, g1, g2], 1)


def temporal_bases(T: int, sfreq: float, centers_s, width_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(B_1, C_b): orthonormal T x 3 bases with B_1^T C_b = 0."""
    B1 = orth(bump_family(T, sfreq, centers_s[0], width_s))
    raw2 = bump_family(T, sfreq, centers_s[1], width_s)
    Cb = orth(raw2 - B1 @ (B1.T @ raw2))
    return B1, Cb


def rotated_basis(theta: float, B1: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """B_2(theta) = cos(theta) B_1 + sin(theta) C_b; exactly B_1 at theta = 0."""
    theta = float(theta)
    if theta == 0.0:
        return B1.copy()
    return np.cos(theta) * B1 + np.sin(theta) * Cb


@dataclass
class SignalModel:
    A: np.ndarray            # C x 2 spatial templates
    B1: np.ndarray           # T x 3
    Cb: np.ndarray           # T x 3
    theta: float
    variances: np.ndarray    # 6 latent variances (source-major: j = 0 l = 0..2, j = 1 l = 0..2)

    @property
    def C(self) -> int:
        return self.A.shape[0]

    @property
    def T(self) -> int:
        return self.B1.shape[0]

    @property
    def d(self) -> int:
        return self.C * self.T

    def temporal(self, j: int) -> np.ndarray:
        return self.B1 if j == 0 else rotated_basis(self.theta, self.B1, self.Cb)

    def templates(self) -> np.ndarray:
        """Physical template matrix (d x 6): column 3 j + l is vec(a_j b_jl^T) = kron(b_jl, a_j)."""
        cols = [np.kron(self.temporal(j)[:, l], self.A[:, j]) for j in range(2) for l in range(3)]
        return np.stack(cols, 1)

    def second_moment(self) -> np.ndarray:
        B = self.templates()
        return symmetric((B * self.variances) @ B.T)

    def draw(self, n: int, gen: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """n Gaussian records: (S (n x d) clean signals, alpha (n x 6) latents)."""
        alpha = gen.normal(size=(n, 6)) * np.sqrt(self.variances)
        return alpha @ self.templates().T, alpha


def base_variances(cfg: dict) -> np.ndarray:
    w = np.asarray(cfg['source_weights_j'], float)
    v = np.asarray(cfg['latent_variances_l'], float)
    if len(w) != 2 or len(v) != 3 or np.any(w <= 0) or np.any(v <= 0):
        raise ValueError('need 2 positive source weights and 3 positive latent variances')
    return np.kron(w, v)


def build(evoked: np.ndarray, sfreq: float, ref_cov: np.ndarray, theta: float, cfg: dict) -> tuple[SignalModel, dict]:
    """Signal model of one subject at one theta.  The latent-variance scale is computed at theta = 0
    (tr(Sigma_ref^{-1} R_s(0)) = trace_K_target) and used unchanged at every theta."""
    C, T = evoked.shape
    tp = spatial_templates(evoked, sfreq, cfg['template_windows_s'], cfg.get('max_template_cosine', 0.95))
    B1, Cb = temporal_bases(T, sfreq, cfg['bump_centers_s'], cfg['bump_width_s'])
    W = np.linalg.inv(symmetric(ref_cov))
    m0 = SignalModel(tp['A'], B1, Cb, 0.0, base_variances(cfg))
    scale = float(cfg['trace_K_target']) / float(np.trace(W @ m0.second_moment()))
    m = SignalModel(tp['A'], B1, Cb, float(theta), base_variances(cfg) * scale)
    info = {'peak_times_s': tp['peak_times_s'], 'template_abs_cosine': tp['abs_cosine'], 'degenerate': tp['degenerate'],
            'variance_scale': scale, 'variances': m.variances.tolist(), 'trace_K': float(np.trace(W @ m.second_moment())),
            'trace_K_theta0': float(cfg['trace_K_target'])}
    return m, info


def partial_trace_spectra(P: np.ndarray, C: int, T: int):
    """Eigenvalues of tr_t P (C x C) and tr_c P (T x T) for a projector P (record index c + C t)."""
    P4 = P.reshape(C, T, C, T, order='F')
    return np.linalg.eigvalsh(np.einsum('ctdt->cd', P4))[::-1], np.linalg.eigvalsh(np.einsum('ctcs->ts', P4))[::-1]
