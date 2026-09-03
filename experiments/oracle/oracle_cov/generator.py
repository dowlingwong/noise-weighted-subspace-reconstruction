"""Deterministic forward model and deployment-window synthesis for ORACLE-Cov.

Wraps ``experiments.synthetic.nfpa_confirmatory.generator`` so that

* the clean forward map ``g(latents)`` is an explicit deterministic function
  (needed for Jacobians, tangent spans and Fisher matrices);
* the *training* dataset is the frozen NFPA dataset at ``alpha = 0`` (the
  clean support the subject is trained on);
* deployment windows can be synthesised with modified latent support (S
  families), modified noise covariance (N families), signal deformations
  (jitter, gain drift, channel loss) and undeclared corruptions (U families).

The SNR scale is the one the training dataset applied, so deployment events
are on the same footing as training events.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from noise_geometry.noise import generate_colored_noise, make_powerlaw_psd
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, PrimaryConfig
from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv, ar1_covariance, whiten_separable
from experiments.synthetic.nfpa_confirmatory.generator import (
    Dataset, _asymmetric_pulse, build_coefficients, make_channel_basis, make_dataset, sample_latents,
)
from noise_geometry.noise.covariance import covariance_from_psd

LATENT_NAMES = ("amplitude", "rise_time", "decay_time", "secondary", "t0_shift")


def temporal_basis_shifted(cfg: PrimaryConfig, t0_shift: float) -> np.ndarray:
    """The planted temporal basis with the pulse onset moved by ``t0_shift`` samples."""
    T = cfg.T
    t = np.arange(T, dtype=float)
    t0 = 0.38 * T + t0_shift
    sigma_r, sigma_d = 3.0, 14.0
    eps_r, eps_d = sigma_r * 1e-2, sigma_d * 1e-2
    base = _asymmetric_pulse(t, t0, sigma_r, sigma_d)
    rise = (_asymmetric_pulse(t, t0, sigma_r + eps_r, sigma_d) - _asymmetric_pulse(t, t0, sigma_r - eps_r, sigma_d)) / (2 * eps_r)
    decay = (_asymmetric_pulse(t, t0, sigma_r, sigma_d + eps_d) - _asymmetric_pulse(t, t0, sigma_r, sigma_d - eps_d)) / (2 * eps_d)
    Tb, _ = np.linalg.qr(np.stack([base, rise, decay], axis=1))
    return np.asarray(Tb)[:, : cfg.k_t]


@dataclass
class ForwardModel:
    """``g(latents) -> (n, C, T)`` clean events, deterministic, alpha = 0."""
    cfg: PrimaryConfig
    V: np.ndarray
    scale: float               # SNR scale applied by the training dataset

    def __call__(self, latents: dict[str, np.ndarray]) -> np.ndarray:
        n = len(latents["amplitude"])
        shift = latents.get("t0_shift", np.zeros(n))
        z = build_coefficients(latents, self.cfg)
        out = np.empty((n, self.cfg.C, self.cfg.T))
        # group events by identical shift so the basis is rebuilt once per distinct value
        for s in np.unique(shift):
            idx = np.where(shift == s)[0]
            Tb = temporal_basis_shifted(self.cfg, float(s))
            out[idx] = np.einsum("nab,ia,jb->nij", z[idx], self.V, Tb)
        return out * self.scale

    def jacobian(self, latents: dict[str, np.ndarray], names=LATENT_NAMES, eps: float = 1e-4) -> np.ndarray:
        """Central-difference Jacobian ``(n, C*T, m)`` (column-major vec) w.r.t. the named latents."""
        n = len(latents["amplitude"])
        cols = []
        for name in names:
            lp = {k: np.array(v, dtype=float) for k, v in latents.items()}
            lm = {k: np.array(v, dtype=float) for k, v in latents.items()}
            lp.setdefault("t0_shift", np.zeros(n)); lm.setdefault("t0_shift", np.zeros(n))
            h = eps * (10.0 if name == "t0_shift" else 1.0)   # samples for t0, latent units otherwise
            lp[name] = lp[name] + h
            lm[name] = lm[name] - h
            d = (self(lp) - self(lm)) / (2 * h)
            cols.append(d.reshape(n, -1, order="F"))
        return np.stack(cols, axis=2)


@dataclass
class TrainingSet:
    ds: Dataset
    forward: ForwardModel
    Sigma_c: np.ndarray
    Sigma_t: np.ndarray

    @property
    def cfg(self) -> PrimaryConfig:
        return self.ds.cfg

    def whiten(self, X: np.ndarray, Sigma_c=None, Sigma_t=None) -> np.ndarray:
        if Sigma_c is None and Sigma_t is None:
            if not hasattr(self, "_Wc"):
                self._Wc, self._Wt = _sqrt_inv(self.Sigma_c), _sqrt_inv(self.Sigma_t)   # cached: identical to whiten_separable
            return np.einsum("ab,...bd,de->...ae", self._Wc, X - self.ds.train_mean, self._Wt)
        Sc = self.Sigma_c if Sigma_c is None else Sigma_c
        St = self.Sigma_t if Sigma_t is None else Sigma_t
        return whiten_separable(X - self.ds.train_mean, Sc, St)


def make_training_set(seed: int, base: PrimaryConfig = PRIMARY, *, alpha: float = 0.0) -> TrainingSet:
    cfg = replace(base, nonseparability_alpha=float(alpha))
    ds = make_dataset(seed, "colored", cfg)
    # recover the SNR scale the dataset applied: clean_train = scale * g_unscaled(latents)
    V = make_channel_basis(cfg)
    fwd0 = ForwardModel(cfg, V, 1.0)
    g0 = fwd0(ds.latents_train)
    scale = float(np.sum(g0 * ds.clean_train) / max(np.sum(g0 * g0), 1e-12))
    return TrainingSet(ds=ds, forward=ForwardModel(cfg, V, scale), Sigma_c=ds.covariances["Sigma_c"], Sigma_t=ds.covariances["Sigma_t"])


# ---------------------------------------------------------------- noise models

def temporal_covariance_slope(cfg: PrimaryConfig, exponent: float) -> np.ndarray:
    """Stationary temporal covariance for a 1/f^exponent PSD (exponent 2 = training)."""
    freqs, psd = make_powerlaw_psd(cfg.T, cfg.sampling_frequency, kind=cfg.noise_kind, level=1.0)
    # rescale the declared 1/f^2 density to 1/f^exponent at equal total power
    f = np.asarray(freqs, dtype=float)
    with np.errstate(divide="ignore"):
        shape = np.where(f > 0, f ** (-(exponent - 2.0)), 1.0)
    psd2 = psd * shape
    psd2 *= psd.sum() / max(psd2.sum(), 1e-300)
    return covariance_from_psd(psd2, cfg.T, sampling_frequency=cfg.sampling_frequency)


def sample_noise(rng: np.random.Generator, n: int, cfg: PrimaryConfig, Sigma_c: np.ndarray, Sigma_t: np.ndarray,
                 *, tails_df: float | None = None) -> np.ndarray:
    """``(n, C, T)`` Gaussian (or Student-t scaled) noise with covariance kron(Sigma_t, Sigma_c)."""
    Lt = np.linalg.cholesky(Sigma_t + 1e-10 * np.eye(cfg.T))
    Lc = np.linalg.cholesky(Sigma_c)
    W = rng.standard_normal((n, cfg.C, cfg.T))
    if tails_df is not None:
        s = np.sqrt(tails_df / rng.chisquare(tails_df, size=(n, 1, 1)))
        W = W * s / np.sqrt(tails_df / (tails_df - 2.0))   # unit-variance heavy tails
    return np.einsum("ab,nbt,ts->nas", Lc, W, Lt.T)


def sample_latents_restricted(rng: np.random.Generator, n: int, *, amplitude_range=None, decay_range=None,
                              rise_range=None, t0_shift: float = 0.0, t0_jitter: float = 0.0) -> dict[str, np.ndarray]:
    """Latents from the training law, optionally truncated to a stratum or shifted in t0."""
    pools: list[dict[str, np.ndarray]] = []
    got = 0
    for _ in range(200):
        lat = sample_latents(rng, 4 * n + 64)
        keep = np.ones(len(lat["amplitude"]), dtype=bool)
        for key, rng_ in (("amplitude", amplitude_range), ("decay_time", decay_range), ("rise_time", rise_range)):
            if rng_ is not None:
                keep &= (lat[key] >= rng_[0]) & (lat[key] < rng_[1])
        pools.append({k: v[keep] for k, v in lat.items()})
        got += int(keep.sum())
        if got >= n:
            break
    else:
        raise ValueError(f"stratum too thin: {got} < {n} after 200 draws")
    lat = {k: np.concatenate([p[k] for p in pools]) for k in pools[0]}
    idx = np.arange(n)
    idx = idx[:n]
    out = {k: v[idx] for k, v in lat.items()}
    out["t0_shift"] = np.full(n, float(t0_shift)) + (rng.normal(0.0, t0_jitter, size=n) if t0_jitter > 0 else 0.0)
    if t0_jitter > 0:
        out["t0_shift"] = np.round(out["t0_shift"] * 4) / 4   # quarter-sample grid keeps basis rebuilds bounded
    return out
