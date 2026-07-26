"""Config-driven entry point for the TIDMAD stage.

Keeps ``scripts/run_experiment.py`` uniform across datasets: a config in,
a metrics dict out, with the run record written by the caller. The Phase 1
implementation lives in ``scripts/tidmad_phase1_estimator_identity.py`` so that
it can also be run directly with command-line flags; this wrapper adapts a YAML
config onto the same code path rather than duplicating it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

__all__ = ["run_tidmad_phase1"]

_DEFAULTS: dict[str, Any] = {
    "mode": "synthetic",
    "seed": 20260727,
    "window_samples": 4096,
    "n_windows": 512,
    "sampling_frequency": 10_000_000.0,
    "frequency": 1_000_000.0,
    "snr": 100.0,
    "psd_kind": "pink",
    "psd_knee_hz": None,
    "learned_subspace_snr_boost": 1.0,
    "phase_errors": [0.0, 0.3, 0.6, 1.0],
    "scaling_snr_grid": [30.0, 100.0, 300.0, 1000.0],
    "science_glob": "tidmad/raw/abra_science_*.h5",
    "contract": "docs/tidmad_data_contract.json",
}


def _load_phase1_module():
    """Import the Phase 1 script as a module without duplicating its logic."""
    repo_root = Path(__file__).resolve().parents[3]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import tidmad_phase1_estimator_identity as phase1  # noqa: PLC0415

    return phase1, repo_root


def run_tidmad_phase1(config: dict[str, Any], data_root: str | Path) -> dict[str, Any]:
    """Run the Phase 1 estimator-identity gate from a config mapping."""
    phase1, repo_root = _load_phase1_module()

    merged = dict(_DEFAULTS)
    merged.update({k: v for k, v in config.items() if k in _DEFAULTS})

    contract = Path(merged["contract"])
    if not contract.is_absolute():
        contract = repo_root / contract

    args = argparse.Namespace(
        mode=str(merged["mode"]),
        seed=int(merged["seed"]),
        window_samples=int(merged["window_samples"]),
        n_windows=int(merged["n_windows"]),
        sampling_frequency=float(merged["sampling_frequency"]),
        frequency=float(merged["frequency"]),
        snr=float(merged["snr"]),
        psd_kind=str(merged["psd_kind"]),
        psd_knee_hz=merged["psd_knee_hz"],
        learned_subspace_snr_boost=float(merged["learned_subspace_snr_boost"]),
        phase_errors=[float(p) for p in merged["phase_errors"]],
        scaling_snr_grid=[float(s) for s in merged["scaling_snr_grid"]],
        data_root=Path(data_root),
        contract=contract,
        science_glob=str(merged["science_glob"]),
        output=None,
    )
    return phase1.run(args)
