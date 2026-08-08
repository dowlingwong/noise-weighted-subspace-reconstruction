"""Frozen primary configuration for the NFPA confirmatory study (Blocks A/B/H).

Source of truth: ``notes/nfpa_submission_experiment_plan.md`` (authoritative
protocol) and ``plan/EXPERIMENT_PLAN_2026-08-08.md`` (scoping + audit). This
file is the single frozen configuration: every runner and test reads from here,
and ``config_hash`` is embedded in every output row. Do not edit after the
freeze commit without a new hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PrimaryConfig:
    C: int = 8
    T: int = 256
    k_c: int = 2
    k_t: int = 3
    N_train: int = 800
    N_validation: int = 400
    N_test: int = 2000
    N_calibration: int = 2048
    n_seeds: int = 30
    n_restarts: int = 10
    n_als_iter: int = 150
    target_snr: float = 10.0
    nonseparability_alpha: float = 0.1
    channel_ar_rho: float = 0.4
    noise_kind: str = "brownian"
    sampling_frequency: float = 1.0
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20260808
    # Block B predeclared equivalence margin (normalized chi2, absolute).
    white_noise_chi2_margin: float = 0.005
    # Block B predeclared subspace equivalence margin (mean principal angle, deg).
    white_noise_angle_margin_deg: float = 1e-2
    als_convergence_tol: float = 1e-10


def seed_list(n: int) -> list[int]:
    return list(range(int(n)))


@dataclass(frozen=True)
class RunPaths:
    root: str = "results/synthetic/nfpa_confirmatory"
    raw: str = "results/synthetic/nfpa_confirmatory/raw"
    aggregates: str = "results/synthetic/nfpa_confirmatory/aggregates"
    figures: str = "results/synthetic/nfpa_confirmatory/figures"
    logs: str = "results/synthetic/nfpa_confirmatory/logs"


def config_hash(cfg: PrimaryConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


PRIMARY = PrimaryConfig()
PRIMARY_HASH = config_hash(PRIMARY)
SEEDS_BLOCK_A = seed_list(PRIMARY.n_seeds)
SEEDS_BLOCK_B = seed_list(20)
