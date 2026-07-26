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

__all__ = ["run_tidmad_phase1", "run_tidmad_phase"]

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

#: Extra keys used by phases 2-5 only.
_PHASE_DEFAULTS: dict[str, Any] = {
    "n_calibration_windows": 256,
    "rank": 2,
    "ranks": [1, 2, 3, 4],
    "fit_fraction": 0.7,
    "guard_windows": 4,
    "bootstrap_reps": 300,
    "block_size": 16,
    "local_radii": [8, 16, 32],
    "training_glob": "tidmad/raw/abra_training_*.h5",
    "validation_glob": "tidmad/raw/abra_validation_*.h5",
    "upstream": None,
    "model_tag": "wiener",
    "max_files": 1,
    "denoise_samples": 1 << 22,
    "coarse": True,
    "wrong_frequency_offset_bins": 37,
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


def run_tidmad_phase(config: dict[str, Any], data_root: str | Path) -> dict[str, Any]:
    """Run TIDMAD phase 1-5 from a config mapping.

    Phase 1 keeps its own entry point because its gates and arguments differ;
    phases 2-5 share the window-preparation pipeline in
    ``scripts/tidmad_phases.py`` and are dispatched here.
    """
    phase = int(config.get("phase", 1))
    if phase == 1:
        return run_tidmad_phase1(config, data_root)

    repo_root = Path(__file__).resolve().parents[3]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import tidmad_phases  # noqa: PLC0415

    merged = dict(_DEFAULTS)
    merged.update(_PHASE_DEFAULTS)
    merged.update({k: v for k, v in config.items() if k in merged})

    contract = Path(merged["contract"])
    if not contract.is_absolute():
        contract = repo_root / contract
    upstream = merged["upstream"]

    args = argparse.Namespace(
        phase=phase,
        mode=str(merged["mode"]),
        seed=int(merged["seed"]),
        window_samples=int(merged["window_samples"]),
        n_windows=int(merged["n_windows"]),
        n_calibration_windows=int(merged["n_calibration_windows"]),
        sampling_frequency=float(merged["sampling_frequency"]),
        frequency=float(merged["frequency"]),
        snr=float(merged["snr"]),
        rank=int(merged["rank"]),
        ranks=[int(r) for r in merged["ranks"]],
        fit_fraction=float(merged["fit_fraction"]),
        guard_windows=int(merged["guard_windows"]),
        bootstrap_reps=int(merged["bootstrap_reps"]),
        block_size=int(merged["block_size"]),
        local_radii=[int(r) for r in merged["local_radii"]],
        psd_kind=str(merged["psd_kind"]),
        psd_knee_hz=merged["psd_knee_hz"],
        data_root=Path(data_root),
        contract=contract,
        training_glob=str(merged["training_glob"]),
        validation_glob=str(merged["validation_glob"]),
        science_glob=str(merged["science_glob"]),
        upstream=Path(upstream) if upstream else None,
        model_tag=str(merged["model_tag"]),
        max_files=int(merged["max_files"]),
        denoise_samples=int(merged["denoise_samples"]),
        coarse=bool(merged["coarse"]),
        wrong_frequency_offset_bins=int(merged["wrong_frequency_offset_bins"]),
        output=None,
    )
    data = tidmad_phases.prepare(args)
    record = tidmad_phases.PHASES[phase](args, data)
    record["all_gates_passed"] = bool(
        all(gate.get("passed", False) for gate in record.get("gates", {}).values())
    )
    return record
