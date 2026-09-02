"""Monitors (alarm statistics) for ORACLE-Cov.

All monitors are fitted on clean reference windows only and return one alarm
magnitude per window.  Magnitudes are standardised by the clean-window
dispersion of the same statistic, so a value of 1 means "one clean-window
standard deviation".  ``needs_covariance`` is True only for the
pullback-whitened monitor; ``is_analytic_for`` marks the two designed
families whose result follows by construction from the same Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .protocol import Family, RepresentationBatch


def _standardise(values: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(values)); sd = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
    return mu, max(sd, 1e-12)


@dataclass
class _Base:
    name: str = "base"
    mu_stage: np.ndarray = None
    clean_stat: tuple[float, float] = (0.0, 1.0)
    analytic_families: tuple[str, ...] = ()
    two_sided: bool = False      # alarm on |standardised score| (statistics that can move either way)

    def needs_covariance(self) -> bool:
        return False

    def is_analytic_for(self, family: Family) -> bool:
        return family.name in self.analytic_families

    def raw(self, batch: RepresentationBatch) -> float:
        raise NotImplementedError

    def fit(self, clean):
        self.mu_stage = np.mean(np.concatenate([b.Z for b in clean]), axis=0)
        self._fit_extra(clean)
        self.clean_stat = _standardise(np.array([self.raw(b) for b in clean]))
        return self

    def _fit_extra(self, clean):
        pass

    def score(self, batch: RepresentationBatch) -> float:
        mu, sd = self.clean_stat
        s = float((self.raw(batch) - mu) / sd)
        return abs(s) if self.two_sided else s


class EuclideanDisplacement(_Base):
    """|mean(Z) - mu_clean|_2 — the wrong-metric statistic."""

    def __init__(self):
        super().__init__(name="euclidean")

    def raw(self, batch):
        return float(np.linalg.norm(batch.Z.mean(axis=0) - self.mu_stage))


class PullbackWhitened(_Base):
    """sqrt(delta^T J^T Sigma_w^{-1} J delta): displacement pulled through the reconstruction
    Jacobian and measured in the *realised* noise metric.  ``Sigma_w_inv`` is the inverse
    realised covariance expressed in the subject's assumed-whitened coordinates (identity when
    the assumed covariance is the true one).  Needs the covariance."""

    def __init__(self, subject, stage: str, Sigma_w_inv_factor: np.ndarray | None = None):
        super().__init__(name="pullback_whitened", analytic_families=("null", "aligned"))
        self.subject, self.stage = subject, stage
        self.Lw = Sigma_w_inv_factor      # (CT, CT) with Sigma_w^{-1} = Lw^T Lw; None = identity
        self.J = None

    def needs_covariance(self):
        return True

    def _fit_extra(self, clean):
        self.J = self.subject.jac_recon(self.stage, self.mu_stage)          # (CT, d)
        self.M = self.J if self.Lw is None else self.Lw @ self.J

    def raw(self, batch):
        delta = batch.Z.mean(axis=0) - self.mu_stage
        return float(np.linalg.norm(self.M @ delta))


class JacobianProjected(_Base):
    """|V_r^T delta|: displacement projected on the leading right-singular directions of the
    reconstruction Jacobian — the covariance-free surrogate deployable on Tier 2."""

    def __init__(self, subject, stage: str, rank: int | None = None):
        super().__init__(name="jacobian_projected", analytic_families=("null", "aligned"))
        self.subject, self.stage, self.rank = subject, stage, rank

    def _fit_extra(self, clean):
        J = self.subject.jac_recon(self.stage, self.mu_stage)
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
        r = self.rank or int(np.sum(s > 1e-8 * s[0]))
        self.Vr = Vt[:r].T; self.singular_values = s[:r]

    def raw(self, batch):
        delta = batch.Z.mean(axis=0) - self.mu_stage
        return float(np.linalg.norm(self.Vr.T @ delta))


class PullbackOutput(_Base):
    """|J_out delta|: displacement pulled through the gradient of the physics output (amplitude
    readout) at the clean mean — the one-dimensional pullback for the consequence variable."""

    def __init__(self, subject, stage: str):
        super().__init__(name="pullback_output", analytic_families=("null_amp", "aligned_amp"))
        self.subject, self.stage = subject, stage

    def _fit_extra(self, clean):
        self.g = self.subject.jac_output(self.stage, self.mu_stage) if self.stage != "recon" else None

    def raw(self, batch):
        if self.g is None:
            return 0.0
        return float(abs(self.g @ (batch.Z.mean(axis=0) - self.mu_stage)))


class OutputResidual(_Base):
    """Mean per-coordinate whitened residual of the subject's reconstruction under its assumed
    covariance (chi-bar^2); departs from its clean value under an acquisition change."""

    def __init__(self):
        super().__init__(name="output_residual", two_sided=True)

    def raw(self, batch):
        return float(batch.meta["chi2_bar"])


class LayerAlignment(_Base):
    """1 - a_{l,k}: rotation of the stage Jacobian's dominant right-singular subspace between the
    clean mean and the window mean (gauge-safe)."""

    def __init__(self, subject, stage: str, k: int = 4):
        super().__init__(name=f"layer_alignment_k{k}")
        self.subject, self.stage, self.k = subject, stage, k

    def _dominant(self, z):
        J = self.subject.jac_recon(self.stage, z)
        _, _, Vt = np.linalg.svd(J, full_matrices=False)
        return Vt[: self.k].T

    def _fit_extra(self, clean):
        self.Q_clean = self._dominant(self.mu_stage)

    def raw(self, batch):
        Q = self._dominant(batch.Z.mean(axis=0))
        a = float(np.linalg.norm(self.Q_clean.T @ Q, "fro") ** 2 / self.k)
        return 1.0 - a


def contrastive_direction_alignment(subject, stage: str, mu_clean: np.ndarray, Z: np.ndarray, k: int = 4) -> dict[str, float]:
    """Fraction of the contrastive direction v = mean(Z) - mu_clean that lies inside the clean
    dominant subspace of the stage Jacobian (attribution feature, not an alarm)."""
    v = Z.mean(axis=0) - mu_clean
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return {"contrastive_norm": 0.0, "contrastive_in_dominant": float("nan")}
    J = subject.jac_recon(stage, mu_clean)
    _, _, Vt = np.linalg.svd(J, full_matrices=False)
    Q = Vt[:k].T
    return {"contrastive_norm": n, "contrastive_in_dominant": float(np.sum((Q.T @ v) ** 2) / n ** 2)}


def make_monitor_set(subject, stage: str, *, k: int = 4, Sigma_w_inv_factor=None) -> list[_Base]:
    mons = [EuclideanDisplacement(), PullbackWhitened(subject, stage, Sigma_w_inv_factor),
            JacobianProjected(subject, stage), OutputResidual(), LayerAlignment(subject, stage, k)]
    if stage != "recon" and getattr(subject, "STAGES", {}).get(stage, 0) < 3 or stage == "code":
        mons.insert(3, PullbackOutput(subject, stage))
    return mons
