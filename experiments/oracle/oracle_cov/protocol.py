"""Shared types and provenance for ORACLE-Cov (ORACLE IMPLEMENTATION_PLAN §2)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, PrimaryConfig, config_hash

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "oracle"


@dataclass(frozen=True)
class Family:
    name: str            # "N1_cov_slope", "S3_t0", "U1_tails", "null", "aligned", "clean"
    contract: str        # "N" | "S" | "U" | "designed" | "clean"
    declared: bool       # False => abstention target only
    severity: float | None = None

    def key(self) -> str:
        return f"{self.name}@{self.severity}" if self.severity is not None else self.name


@dataclass
class Window:
    ids: np.ndarray       # event indices inside the batch
    event_group: int      # resampling unit: events sharing latents share a group
    family: Family
    seed: int


@dataclass
class RepresentationBatch:
    stage: str
    Z: np.ndarray               # (n, d_stage)
    outputs: np.ndarray         # (n,) amplitude estimate of the frozen subject
    recon: np.ndarray | None    # (n, C*T) whitened-space reconstruction, if the stage exposes it
    meta: dict = field(default_factory=dict)


class Metric(Protocol):
    name: str
    def fit(self, clean: Sequence[RepresentationBatch]) -> "Metric": ...
    def score(self, batch: RepresentationBatch) -> float: ...
    def needs_covariance(self) -> bool: ...
    def is_analytic_for(self, family: Family) -> bool: ...


class Consequence(Protocol):
    name: str
    def score(self, clean_outputs: np.ndarray, perturbed_outputs: np.ndarray, truth: np.ndarray) -> float: ...


@dataclass(frozen=True)
class OracleCovConfig:
    """Tier-1 configuration.  ``mode`` separates dev from confirmatory output."""
    mode: str = "dev"                       # "dev" | "confirmatory"
    base_alpha: float = 0.0                 # training support is exactly separable
    window_size: int = 100
    n_clean_windows: int = 60               # for false-alert calibration
    false_alert_rate: float = 0.01
    kappa_m: float = 0.10                   # consequence tolerance: 10 % amplitude-RMSE degradation
    equivalence_margin_f1: float = 0.05
    subject_rank: int = 6
    mlp_hidden: tuple[int, int] = (64, 32)
    mlp_epochs: int = 60
    mlp_lr: float = 2e-3
    stage_top_k: int = 4                    # k for the dominant-subspace alignment statistic
    dev_seeds: tuple[int, ...] = (0, 1, 2)
    confirmatory_seeds: tuple[int, ...] = tuple(range(100, 120))

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


DEFAULT = OracleCovConfig()


def out_dir_for(experiment_id: str, cfg: OracleCovConfig) -> Path:
    d = RESULTS_ROOT / experiment_id / cfg.mode
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(out_dir: Path, *, experiment_id: str, cfg: OracleCovConfig, base: PrimaryConfig, seeds: Sequence[int],
                   families: Sequence[Family], extra: dict[str, Any] | None = None) -> Path:
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json")
    payload = {
        "experiment_id": experiment_id, "mode": cfg.mode,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": rc.provenance(sys.argv, base),
        "oracle_cov_config": asdict(cfg), "oracle_cov_config_hash": cfg.hash(),
        "generator_config_hash": config_hash(base),
        "seeds": [int(s) for s in seeds],
        "families": [asdict(f) for f in families],
        "outputs": {str(p.relative_to(out_dir)): rc.sha256_of(p) for p in files},
        "citable": cfg.mode == "confirmatory",
    }
    if extra:
        payload.update(extra)
    path = out_dir / "manifest.json"
    rc.write_json(path, jsonable(payload))
    return path


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
