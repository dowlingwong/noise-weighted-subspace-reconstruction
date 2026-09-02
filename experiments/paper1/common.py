"""Shared helpers for the Paper 1 experiments: manifests, output dirs, fits.

Every runner writes ``results/paper1/<ID>/manifest.json`` carrying the
configuration hash, git commit, seed list, the command line and the SHA-256 of
every file it produced.  Fitting reuses the frozen NFPA confirmatory harness so
that numbers are comparable with the archived R/PS evidence.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, PrimaryConfig, config_hash
from experiments.synthetic.nfpa_confirmatory.generator import Dataset, make_dataset
from experiments.synthetic.nfpa_confirmatory.methods import FIT_FACTORY, MethodFit

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "paper1"


def out_dir_for(experiment_id: str, *, smoke: bool = False) -> Path:
    d = RESULTS_ROOT / (experiment_id + ("_smoke" if smoke else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def smoke_config(base: PrimaryConfig = PRIMARY) -> PrimaryConfig:
    """A small configuration for wiring checks; never citable."""
    return replace(base, N_train=120, N_validation=40, N_test=200, N_calibration=128, n_restarts=2, n_als_iter=40)


def fit_methods(ds: Dataset, names: Sequence[str]) -> dict[str, MethodFit]:
    fits: dict[str, MethodFit] = {}
    for name in names:
        t0 = time.time()
        fit = FIT_FACTORY[name](ds)
        fit.diag["runtime_seconds"] = time.time() - t0
        fits[name] = fit
    return fits


def write_manifest(out_dir: Path, *, experiment_id: str, cfgs: Sequence[PrimaryConfig], seeds: Sequence[int],
                   extra: dict[str, Any] | None = None) -> Path:
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json")
    payload = {
        "experiment_id": experiment_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": rc.provenance(sys.argv, cfgs[0]),
        "config_hashes": sorted({config_hash(c) for c in cfgs}),
        "configs": [asdict(c) for c in cfgs],
        "seeds": [int(s) for s in seeds],
        "outputs": {str(p.relative_to(out_dir)): rc.sha256_of(p) for p in files},
    }
    if extra:
        payload.update(extra)
    path = out_dir / "manifest.json"
    rc.write_json(path, payload)
    return path


def as_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return obj
