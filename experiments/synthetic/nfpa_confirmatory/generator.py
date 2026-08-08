"""Nondegenerate multichannel signal generator (protocol section 4).

Clean events are a low-dimensional scientific pulse family

    S_i = sum_{a,b} z_{i,a,b} v_a t_b^T + alpha R_i

with orthonormal channel vectors ``v_a`` (C = 8) and orthonormal temporal
vectors ``t_b`` (T = 256). The latent coefficients ``z`` encode an event
amplitude, a rise-time parameter, and a decay-time parameter (Block H ground
truth). ``R_i`` is constructed outside the planted product subspace and
renormalized at every ``alpha`` so the ``alpha`` knob never becomes an SNR knob.

Assertions: ``C > k_c`` and ``T > k_t`` fail hard (the exact defect the
consolidation fixes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_geometry.noise import generate_colored_noise, make_powerlaw_psd

from .config import PrimaryConfig
from .covariance import full_covariance, whiten_separable


def make_channel_basis(cfg: PrimaryConfig) -> np.ndarray:
    """Two interpretable channel-loading patterns, orthonormalized (C, k_c)."""
    C, k_c = cfg.C, cfg.k_c
    v1 = np.array([1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.3, 0.2], dtype=float)[:C]
    v2 = np.array([0.5, -0.5, 0.0, 0.8, -0.8, 0.0, 0.4, -0.4], dtype=float)[:C]
    V = np.stack([v1, v2], axis=1)
    V, _ = np.linalg.qr(V)
    V = np.asarray(V)[:, :k_c]
    return V


def _asymmetric_pulse(t: np.ndarray, t0: float, sigma_r: float, sigma_d: float) -> np.ndarray:
    p = np.where(t < t0, np.exp(-(t - t0) ** 2 / (2.0 * sigma_r**2)), np.exp(-(t - t0) ** 2 / (2.0 * sigma_d**2)))
    return p / np.max(np.abs(p))


def make_temporal_basis(cfg: PrimaryConfig) -> np.ndarray:
    """Nominal asymmetric pulse plus rise/decay perturbations, orthonormalized."""
    T, k_t = cfg.T, cfg.k_t
    t = np.arange(T, dtype=float)
    t0 = 0.38 * T
    sigma_r, sigma_d = 3.0, 14.0
    eps_r, eps_d = sigma_r * 1e-2, sigma_d * 1e-2
    base = _asymmetric_pulse(t, t0, sigma_r, sigma_d)
    rise = (_asymmetric_pulse(t, t0, sigma_r + eps_r, sigma_d) - _asymmetric_pulse(t, t0, sigma_r - eps_r, sigma_d)) / (2 * eps_r)
    decay = (_asymmetric_pulse(t, t0, sigma_r, sigma_d + eps_d) - _asymmetric_pulse(t, t0, sigma_r, sigma_d - eps_d)) / (2 * eps_d)
    Tb, _ = np.linalg.qr(np.stack([base, rise, decay], axis=1))
    return np.asarray(Tb)[:, :k_t]


def kron_basis(U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    """Orthonormal basis for ``range(U_t) (x) range(U_c)`` under column-major vec.

    Columns are ``kron(u_t[:, b], u_c[:, a])`` for all ``(a, b)``; consistent
    with the paper's Eq. (8) vectorization convention (verified by test 4).
    """
    C, k_c = U_c.shape
    T, k_t = U_t.shape
    cols = [np.kron(U_t[:, b], U_c[:, a]) for b in range(k_t) for a in range(k_c)]
    B = np.column_stack(cols)
    Q, _ = np.linalg.qr(B)
    return Q


def project_out_product_subspace(R: np.ndarray, V: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """Orthogonalize each (C, T) matrix against ``span(V) (x) span(T)``."""
    B = kron_basis(V, Tb)  # (CT, k_c*k_t)
    n = R.shape[0]
    flat = R.reshape(n, -1, order="F")
    residual = flat - (flat @ B) @ B.T
    return residual.reshape(R.shape, order="F")


def sample_latents(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Sample physical latents: amplitude, rise-time, decay-time, secondary mode."""
    A = np.abs(rng.normal(1.0, 0.2, size=n))
    rise = rng.normal(0.0, 0.15, size=n)
    decay = rng.normal(0.0, 0.15, size=n)
    secondary = rng.normal(0.0, 0.5, size=n)
    return {"amplitude": A, "rise_time": rise, "decay_time": decay, "secondary": secondary}


def build_coefficients(latents: dict[str, np.ndarray], cfg: PrimaryConfig) -> np.ndarray:
    """Map physical latents to the ``(k_c, k_t)`` coefficient tensor per event."""
    n = len(latents["amplitude"])
    A = latents["amplitude"]
    r = latents["rise_time"]
    d = latents["decay_time"]
    m = latents["secondary"]
    z = np.zeros((n, cfg.k_c, cfg.k_t), dtype=float)
    z[:, 0, 0] = A
    z[:, 0, 1] = A * r
    z[:, 0, 2] = A * d
    z[:, 1, 0] = A * m
    z[:, 1, 1] = A * 0.25 * r
    z[:, 1, 2] = A * 0.25 * d
    return z


def _nonseparable_perturbations(rng: np.random.Generator, n: int, cfg: PrimaryConfig, V: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """Smooth random matrices outside the planted product subspace."""
    C, T = cfg.C, cfg.T
    raw = rng.standard_normal((n, C, T))
    kernel = np.exp(-np.arange(9) ** 2 / (2 * 3.0**2))
    kernel = kernel / kernel.sum()
    smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=2, arr=raw)
    R = project_out_product_subspace(smooth, V, Tb)
    norms = np.linalg.norm(R.reshape(n, -1), axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return R / norms[:, None]


def generate_clean_events(
    rng: np.random.Generator, n: int, cfg: PrimaryConfig, V: np.ndarray, Tb: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate ``n`` clean events and their physical latents (protocol 4)."""
    latents = sample_latents(rng, n)
    z = build_coefficients(latents, cfg)
    S_sep = np.einsum("nab,ia,jb->nij", z, V, Tb)
    R = _nonseparable_perturbations(rng, n, cfg, V, Tb)
    S = S_sep + cfg.nonseparability_alpha * R
    sep_norm = np.linalg.norm(S_sep.reshape(n, -1), axis=1, keepdims=True)
    evt_norm = np.linalg.norm(S.reshape(n, -1), axis=1, keepdims=True)
    S = S * (sep_norm / np.maximum(evt_norm, 1e-12))[:, None]
    return S, latents


def generate_noise(
    rng: np.random.Generator, n: int, cfg: PrimaryConfig, covariances: dict[str, np.ndarray]
) -> np.ndarray:
    """Draw ``(n, C, T)`` Gaussian noise with covariance ``kron(Sigma_t, Sigma_c)``."""
    if cfg.C != covariances["Sigma_c"].shape[0] or cfg.T != covariances["Sigma_t"].shape[0]:
        raise ValueError("covariance dims must match C and T")
    if covariances["Sigma_c"].shape[0] == cfg.C and np.allclose(covariances["Sigma_c"], np.eye(cfg.C)) and np.allclose(
        covariances["Sigma_t"], np.eye(cfg.T)
    ):
        return rng.standard_normal((n, cfg.C, cfg.T))
    _, psd = make_powerlaw_psd(cfg.T, cfg.sampling_frequency, kind=cfg.noise_kind, level=1.0)
    M = generate_colored_noise(rng, psd, cfg.T, cfg.sampling_frequency, n * cfg.C).reshape(n, cfg.C, cfg.T)
    Lc = np.linalg.cholesky(covariances["Sigma_c"])
    return np.einsum("ab,nbt->nat", Lc, M)


def snr_normalize(clean: np.ndarray, covariances: dict[str, np.ndarray], cfg: PrimaryConfig) -> tuple[np.ndarray, dict[str, float]]:
    """Rescale clean events so realized ``SNR^2 = E[s^T Sigma^{-1} s]`` hits target."""
    Z = whiten_separable(clean, covariances["Sigma_c"], covariances["Sigma_t"])
    snr2 = np.sum(Z.reshape(clean.shape[0], -1) ** 2, axis=1)
    realized = float(np.sqrt(np.mean(snr2)))
    scale = cfg.target_snr / max(realized, 1e-12)
    return clean * scale, {
        "target_snr": cfg.target_snr,
        "realized_snr_mean": float(np.sqrt(np.mean(snr2 * scale**2))),
        "realized_snr_std": float(np.std(np.sqrt(snr2) * scale)),
    }


@dataclass
class Dataset:
    seed: int
    regime: str
    cfg: PrimaryConfig
    V: np.ndarray
    Tb: np.ndarray
    covariances: dict[str, np.ndarray]
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    clean_train: np.ndarray
    clean_validation: np.ndarray
    clean_test: np.ndarray
    latents_train: dict[str, np.ndarray]
    latents_validation: dict[str, np.ndarray]
    latents_test: dict[str, np.ndarray]
    train_mean: np.ndarray
    snr_stats: dict[str, float]
    covariance_report: dict[str, float] = field(default_factory=dict)


def make_dataset(seed: int, regime: str, cfg: PrimaryConfig | None = None) -> Dataset:
    """Build one seed's train/validation/test events for a noise regime.

    Clean signals are generated once per seed (regime-independent) and reused
    across regimes, matching the protocol's "same generated sets within a seed".
    The training mean is estimated on training events only.
    """
    cfg = cfg or PrimaryConfig()
    if not (cfg.C > cfg.k_c and cfg.T > cfg.k_t):
        raise AssertionError(f"require C > k_c and T > k_t (got C={cfg.C}, k_c={cfg.k_c}, T={cfg.T}, k_t={cfg.k_t})")
    rng = np.random.default_rng(seed)
    V = make_channel_basis(cfg)
    Tb = make_temporal_basis(cfg)
    covariances = full_covariance(cfg, regime)

    clean_train, lat_train = generate_clean_events(rng, cfg.N_train, cfg, V, Tb)
    clean_val, lat_val = generate_clean_events(rng, cfg.N_validation, cfg, V, Tb)
    clean_test, lat_test = generate_clean_events(rng, cfg.N_test, cfg, V, Tb)

    clean_train, snr = snr_normalize(clean_train, covariances, cfg)
    clean_val, _ = snr_normalize(clean_val, covariances, cfg)
    clean_test, _ = snr_normalize(clean_test, covariances, cfg)

    X_train = clean_train + generate_noise(rng, cfg.N_train, cfg, covariances)
    X_validation = clean_val + generate_noise(rng, cfg.N_validation, cfg, covariances)
    X_test = clean_test + generate_noise(rng, cfg.N_test, cfg, covariances)

    train_mean = X_train.mean(axis=0)

    return Dataset(
        seed=seed,
        regime=regime,
        cfg=cfg,
        V=V,
        Tb=Tb,
        covariances=covariances,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        clean_train=clean_train,
        clean_validation=clean_val,
        clean_test=clean_test,
        latents_train=lat_train,
        latents_validation=lat_val,
        latents_test=lat_test,
        train_mean=train_mean,
        snr_stats=snr,
    )
