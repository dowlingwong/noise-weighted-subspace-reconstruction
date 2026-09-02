"""Perturbation families for ORACLE-Cov (docs/EXPERIMENT_PLAN_PAPER3.md §3).

A family produces deployment windows ``(X, latents, meta)`` in raw measurement
coordinates.  N families change the *noise* (or deform the signal at
acquisition) and leave the latent support alone; S families keep the training
noise and move the latent support; U families are undeclared corruptions used
only as abstention targets; ``null`` / ``aligned`` are representation-level
perturbations built from the subject's own output Jacobian and are applied by
``designed_perturbation`` on a representation batch rather than on events.

Every family records its exact parameters in ``meta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from experiments.synthetic.nfpa_confirmatory.covariance import ar1_covariance

from .generator import TrainingSet, sample_latents_restricted, sample_noise, temporal_covariance_slope
from .protocol import Family

TRAIN_AMPLITUDE = (0.0, np.inf)   # training law: |N(1, 0.2)|
TRAIN_DECAY = (-np.inf, np.inf)   # training law: N(0, 0.15)


@dataclass
class WindowData:
    X: np.ndarray                     # (n, C, T) raw events
    latents: dict[str, np.ndarray]
    clean: np.ndarray                 # (n, C, T) clean signal actually present
    family: Family
    meta: dict = field(default_factory=dict)
    Sigma_c: np.ndarray | None = None  # realized noise covariance factors (None = training)
    Sigma_t: np.ndarray | None = None


def _clean_window(ts: TrainingSet, rng, n, **lat_kw):
    lat = sample_latents_restricted(rng, n, **lat_kw)
    return lat, ts.forward(lat)


def clean(ts: TrainingSet, rng: np.random.Generator, n: int, family: Family) -> WindowData:
    lat, S = _clean_window(ts, rng, n)
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"note": "training law, training noise"})


# ------------------------------------------------------------------ N families

def n1_cov_slope(ts, rng, n, family: Family) -> WindowData:
    exponent = 2.0 + family.severity
    St = temporal_covariance_slope(ts.cfg, exponent)
    lat, S = _clean_window(ts, rng, n)
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, St) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"psd_exponent": exponent}, ts.Sigma_c, St)


def n2_cov_channel(ts, rng, n, family: Family) -> WindowData:
    rho = float(np.clip(ts.cfg.channel_ar_rho + family.severity, -0.95, 0.95))
    Sc = ar1_covariance(ts.cfg.C, rho)
    lat, S = _clean_window(ts, rng, n)
    X = S + sample_noise(rng, n, ts.cfg, Sc, ts.Sigma_t) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"channel_rho": rho}, Sc, ts.Sigma_t)


def n3_jitter(ts, rng, n, family: Family) -> WindowData:
    lat, S = _clean_window(ts, rng, n, t0_jitter=family.severity)
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"t0_jitter_samples": family.severity, "note": "signal deformation at acquisition; zero-mean shift"})


def n4_gain_drift(ts, rng, n, family: Family) -> WindowData:
    lat, S = _clean_window(ts, rng, n)
    gains = 1.0 + rng.normal(0.0, family.severity, size=ts.cfg.C)
    S2 = S * gains[None, :, None]
    noise = sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) * gains[None, :, None]
    Sc = ts.Sigma_c * np.outer(gains, gains)
    return WindowData(S2 + noise + ts.ds.train_mean, lat, S2, family, {"gains": gains.tolist()}, Sc, ts.Sigma_t)


def n5_glitch(ts, rng, n, family: Family) -> WindowData:
    """Additive short glitches at rate ``severity`` per event (Tier-1b style artifact)."""
    lat, S = _clean_window(ts, rng, n)
    noise = sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t)
    t = np.arange(ts.cfg.T)
    amp_scale = float(np.abs(S).max())
    k = rng.poisson(family.severity, size=n)
    for i in range(n):
        for _ in range(k[i]):
            c = rng.integers(ts.cfg.C); t0 = rng.uniform(0, ts.cfg.T); w = rng.uniform(1.0, 4.0)
            noise[i, c] += rng.choice([-1, 1]) * 2.0 * amp_scale * np.exp(-0.5 * ((t - t0) / w) ** 2)
    return WindowData(S + noise + ts.ds.train_mean, lat, S, family, {"glitch_rate": family.severity, "n_glitches": int(k.sum())})


def n6_channel_loss(ts, rng, n, family: Family) -> WindowData:
    lat, S = _clean_window(ts, rng, n)
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + ts.ds.train_mean
    lost = rng.choice(ts.cfg.C, size=int(family.severity), replace=False)
    X[:, lost, :] = ts.ds.train_mean[lost][None]
    S2 = S.copy(); S2[:, lost, :] = 0.0
    return WindowData(X, lat, S2, family, {"lost_channels": lost.tolist()})


# ------------------------------------------------------------------ S families

def s1_amplitude(ts, rng, n, family: Family) -> WindowData:
    """Clean events with amplitude above the ``severity`` quantile of the training law."""
    q = float(family.severity)
    lo = float(np.quantile(np.abs(np.random.default_rng(0).normal(1.0, 0.2, 200000)), q))
    lat, S = _clean_window(ts, rng, n, amplitude_range=(lo, np.inf))
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"amplitude_lower_bound": lo, "quantile": q})


def s2_decay(ts, rng, n, family: Family) -> WindowData:
    q = float(family.severity)
    lo = float(np.quantile(np.random.default_rng(0).normal(0.0, 0.15, 200000), q))
    lat, S = _clean_window(ts, rng, n, decay_range=(lo, np.inf))
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"decay_lower_bound": lo, "quantile": q})


def s3_t0(ts, rng, n, family: Family) -> WindowData:
    """Training fixed t0; deployment moves it by ``severity`` samples (an unexcited direction)."""
    lat, S = _clean_window(ts, rng, n, t0_shift=family.severity)
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"t0_shift_samples": family.severity})


def s4_alpha(ts, rng, n, family: Family) -> WindowData:
    """Nonseparable deformation never excited in training (alpha = severity)."""
    from experiments.synthetic.nfpa_confirmatory.generator import _nonseparable_perturbations, make_temporal_basis
    lat, S = _clean_window(ts, rng, n)
    R = _nonseparable_perturbations(rng, n, ts.cfg, ts.forward.V, make_temporal_basis(ts.cfg))
    sep_norm = np.linalg.norm(S.reshape(n, -1), axis=1, keepdims=True)
    S2 = S + family.severity * R * sep_norm[:, :, None]
    S2 *= (sep_norm / np.maximum(np.linalg.norm(S2.reshape(n, -1), axis=1, keepdims=True), 1e-12))[:, :, None]
    X = S2 + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + ts.ds.train_mean
    return WindowData(X, lat, S2, family, {"alpha": family.severity})


# ------------------------------------------------------------------ U families

def u1_tails(ts, rng, n, family: Family) -> WindowData:
    lat, S = _clean_window(ts, rng, n)
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t, tails_df=family.severity) + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"student_t_df": family.severity})


def u2_line(ts, rng, n, family: Family) -> WindowData:
    lat, S = _clean_window(ts, rng, n)
    t = np.arange(ts.cfg.T)
    f = 0.137 * ts.cfg.sampling_frequency
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, 1))
    line = family.severity * float(np.abs(S).max()) * np.sin(2 * np.pi * f * t[None, None, :] + phase)
    X = S + sample_noise(rng, n, ts.cfg, ts.Sigma_c, ts.Sigma_t) + line + ts.ds.train_mean
    return WindowData(X, lat, S, family, {"line_frequency": f, "line_amplitude_rel": family.severity})


GENERATORS: dict[str, Callable] = {
    "clean": clean, "N1_cov_slope": n1_cov_slope, "N2_cov_channel": n2_cov_channel, "N3_jitter": n3_jitter,
    "N4_gain_drift": n4_gain_drift, "N5_glitch": n5_glitch, "N6_channel_loss": n6_channel_loss,
    "S1_amplitude": s1_amplitude, "S2_decay": s2_decay, "S3_t0": s3_t0, "S4_alpha": s4_alpha,
    "U1_tails": u1_tails, "U2_line": u2_line,
}

CONTRACT = {"clean": "clean", "N": "N", "S": "S", "U": "U"}


def family(name: str, severity: float | None = None) -> Family:
    contract = "clean" if name == "clean" else name[0]
    return Family(name=name, contract=contract, declared=contract != "U", severity=severity)


def make_window(ts: TrainingSet, fam: Family, n: int, seed: int) -> WindowData:
    rng = np.random.default_rng([seed, hash(fam.name) & 0xFFFF, int(1e6 * (fam.severity or 0.0)) & 0xFFFFFF])
    return GENERATORS[fam.name](ts, rng, n, fam)


# Declared severity grids (frozen in docs/plans/oracle/PREREGISTRATION.md)
SEVERITIES = {
    "N1_cov_slope": [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75],
    "N2_cov_channel": [-0.4, -0.2, 0.2, 0.4],
    "N3_jitter": [0.5, 1.0, 2.0, 4.0],
    "N4_gain_drift": [0.02, 0.05, 0.1, 0.2],
    "N5_glitch": [0.25, 0.5, 1.0, 2.0],
    "N6_channel_loss": [1, 2, 3],
    "S1_amplitude": [0.7, 0.85, 0.95],
    "S2_decay": [0.7, 0.85, 0.95],
    "S3_t0": [0.5, 1.0, 2.0, 4.0],
    "S4_alpha": [0.1, 0.2, 0.4],
    "U1_tails": [3.0, 5.0],
    "U2_line": [0.25, 0.5],
}


def designed_perturbation(subject, stage: str, Z: np.ndarray, kind: str, norm: float, rng: np.random.Generator) -> np.ndarray:
    """Displace every stage vector of the window by the *same* vector of length ``norm``: along a
    random direction of the output Jacobian's null space (``kind='null'``) or along its leading
    right-singular direction (``kind='aligned'``).  The Jacobian is evaluated at the window mean,
    which is also where the monitors linearise, so the null arm is a first-order null by
    construction.  Using one coherent direction per window (rather than a fresh random one per
    event) is what makes the two arms norm-matched at the window level: an unweighted Euclidean
    displacement of the window mean then cannot tell them apart."""
    mu = Z.mean(axis=0)
    if kind in ("null_amp", "aligned_amp"):
        g = subject.jac_output(stage, mu)                      # (d,) gradient of the amplitude readout
        g = g / np.linalg.norm(g)
        if kind == "aligned_amp":
            v = g
        else:
            v = rng.standard_normal(g.size); v -= (v @ g) * g; v /= np.linalg.norm(v)
        return Z + norm * v[None, :]
    J = subject.jac_recon(stage, mu)                          # (CT, d)
    _, s, Vt = np.linalg.svd(J, full_matrices=True)
    r = int(np.sum(s > 1e-10 * s[0]))
    if kind == "aligned":
        v = Vt[0]
    elif kind == "null":
        if r >= Vt.shape[0]:
            raise ValueError("output Jacobian has no null space at this stage")
        coeff = rng.standard_normal(Vt.shape[0] - r)
        v = coeff @ Vt[r:]
        v /= np.linalg.norm(v)
    else:
        raise KeyError(kind)
    return Z + norm * v[None, :]
