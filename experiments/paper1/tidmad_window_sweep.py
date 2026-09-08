"""P1-E13 — TIDMAD operating-point sweep, zero-predictor baseline, controls.

Review Major 2: the archived headline compares two estimators that are *both*
below the zero predictor on file 0000 (0.0747 < 0.0880 < 0.1209 amplitude
RMSE). The manuscript therefore needs either an operating point above the noise
floor or an honest floor statement, and a random-basis control that is
informative rather than saturated.

This runner loops the analysis window length and the file, calling
``run_metric_reversal`` with every other parameter taken from
``configs/tidmad/phase2_metric_reversal_real_noise.yaml``, and records the
zero-predictor RMSE alongside the two estimators. The norm-matched random-basis
control is run once at the frozen operating point (file 0000, 4096 samples).

Data: the TIDMAD HDF5 release under ``PAPER1_DATA_ROOT`` (default
``/ceph/dwong/paper1_dataset``). Without it this script exits with a clear
"data unavailable" record rather than a traceback, so an incomplete run is
visible in the results directory instead of only in a log.

Usage::

    python -m experiments.paper1.tidmad_window_sweep \
        --files abra_validation_0000.h5 abra_validation_0005.h5 abra_validation_0010.h5 \
        --windows 4096 16384 65536
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.synthetic.nfpa_confirmatory import r9_common as rc

from . import common

EXPERIMENT_ID = "P1-E13"

#: n_windows per window length, chosen so the same sample range is covered.
DEFAULT_N_WINDOWS = {4096: 512, 16384: 128, 65536: 32}
#: block size per window length, scaled the same way.
DEFAULT_BLOCK_SIZE = {4096: 16, 16384: 4, 65536: 1}
#: below this many evaluation windows a row is reported as underpowered.
MIN_EVAL_WINDOWS = 10

COLUMNS = [
    "file", "window_samples", "n_fit", "n_eval",
    "rmse_pca", "rmse_empca", "rmse_zero",
    "paired_diff_pca_minus_empca", "ci95_low_window", "ci95_high_window",
    "ci95_low_block", "ci95_high_block", "direction_ok",
    "empca_beats_zero", "pca_beats_zero", "zero_minus_empca_mean",
    "zero_minus_empca_ci95_low", "zero_minus_empca_ci95_high",
    "underpowered", "n_windows_requested", "block_size", "rank", "seed",
]


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _data_root(cfg: dict) -> Path:
    return Path(os.environ.get("PAPER1_DATA_ROOT", cfg.get("data_root", "/ceph/dwong/paper1_dataset")))


def _load_windows(cfg: dict, data_root: Path, filename: str, window_samples: int,
                  n_windows: int, n_calibration_windows: int):
    """Load one file's paired windows and the science-file noise estimate."""
    from noise_geometry.noise import inverse_psd_weights
    from noise_geometry.tidmad.loader import TidmadFile, load_contract
    from noise_geometry.tidmad.psd import welch_psd_from_windows
    from noise_geometry.tidmad.templates import nearest_bin_frequency

    contract = load_contract(cfg["contract"])
    contract.require_verified()
    validation = data_root / "tidmad" / "raw" / filename
    if not validation.exists():
        raise FileNotFoundError(str(validation))
    science = sorted(data_root.glob(cfg["science_glob"]))
    if not science:
        raise FileNotFoundError(cfg["science_glob"])

    observed, clean = [], []
    with TidmadFile(validation, contract) as handle:
        for squid, reference in handle.iter_paired_windows(
                window_samples, stride=window_samples, limit=int(n_windows)):
            observed.append(squid)
            clean.append(reference)
    noise_only = []
    with TidmadFile(science[0], contract) as handle:
        for window in handle.iter_squid_windows(
                window_samples, stride=window_samples, limit=int(n_calibration_windows)):
            noise_only.append(window)

    fs = float(contract.sampling_frequency_hz)
    frequency = contract.frequency_for_file(validation.name, None)
    estimate = welch_psd_from_windows(list(noise_only), fs, window="boxcar",
                                      detrend=True, average="median", source="science")
    return {
        "observed": np.vstack(observed), "clean": np.vstack(clean),
        "noise_only": np.vstack(noise_only), "psd": estimate.psd,
        "weights": inverse_psd_weights(estimate.psd, window_samples),
        "sampling_frequency": fs,
        "frequency": nearest_bin_frequency(frequency, window_samples, fs),
    }


def _row(cfg: dict, data, filename: str, window_samples: int, n_windows: int) -> dict:
    from noise_geometry.tidmad.analysis import build_window_set, run_metric_reversal

    windows = build_window_set(
        data["observed"], data["clean"], data["psd"], data["weights"],
        data["sampling_frequency"], data["frequency"],
        fit_fraction=float(cfg["fit_fraction"]), guard_windows=int(cfg["guard_windows"]),
    )
    block_size = DEFAULT_BLOCK_SIZE.get(window_samples, int(cfg["block_size"]))
    result = run_metric_reversal(
        windows, rank=int(cfg["rank"]), bootstrap_reps=int(cfg["bootstrap_reps"]),
        block_size=block_size, seed=int(cfg["seed"]),
    )
    wl = result["paired_contrasts_window_level"]["amplitude_abs_error_pca_minus_empca"]
    bl = result["paired_contrasts_block_level"]["amplitude_abs_error_pca_minus_empca"]
    zp = result["zero_predictor"]
    zc = zp["paired_contrasts"]["zero_predictor_paired_difference_zero_minus_empca"]
    n_eval = int(result["n_evaluation_windows"])
    direction = result["direction_check"]
    return {
        "file": filename, "window_samples": int(window_samples),
        "n_fit": int(result["n_fit_windows"]), "n_eval": n_eval,
        "rmse_pca": zp["pca_amplitude_rmse"],
        "rmse_empca": zp["empca_amplitude_rmse"],
        "rmse_zero": zp["zero_predictor_amplitude_rmse"],
        "paired_diff_pca_minus_empca": wl["mean_difference"],
        "ci95_low_window": wl["ci95_difference"][0], "ci95_high_window": wl["ci95_difference"][1],
        "ci95_low_block": bl["ci95_difference"][0], "ci95_high_block": bl["ci95_difference"][1],
        "direction_ok": bool(direction["raw_mse_favours_pca"]
                             and direction["weighted_residual_favours_empca"]
                             and direction["truth_favours_empca"]),
        "empca_beats_zero": bool(zp["empca_beats_zero_predictor"]),
        "pca_beats_zero": bool(zp["pca_beats_zero_predictor"]),
        "zero_minus_empca_mean": zc["mean_difference"],
        "zero_minus_empca_ci95_low": zc["ci95_difference"][0],
        "zero_minus_empca_ci95_high": zc["ci95_difference"][1],
        "underpowered": bool(n_eval < MIN_EVAL_WINDOWS),
        "n_windows_requested": int(n_windows), "block_size": int(block_size),
        "rank": int(cfg["rank"]), "seed": int(cfg["seed"]),
        "_full": result,
    }


def run(files: list[str], windows_list: list[int], config_path: str) -> Path:
    out = common.out_dir_for(EXPERIMENT_ID)
    cfg = _config(common.REPO_ROOT / config_path)
    data_root = _data_root(cfg)
    t_start = time.time()

    if not data_root.exists():
        record = {
            "status": "not_run",
            "reason": f"PAPER1_DATA_ROOT does not exist on this host: {data_root}",
            "files_requested": files, "windows_requested": windows_list,
            "config": config_path, "host": platform.node(),
            "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        rc.write_json(out / "summary.json", record)
        print(json.dumps(record, indent=1))
        return out

    rows, failures = [], []
    for filename in files:
        for window_samples in windows_list:
            n_windows = DEFAULT_N_WINDOWS.get(window_samples, int(cfg["n_windows"]))
            try:
                data = _load_windows(cfg, data_root, filename, window_samples, n_windows,
                                     int(cfg["n_calibration_windows"]))
                rows.append(_row(cfg, data, filename, window_samples, n_windows))
                r = rows[-1]
                print(f"[{EXPERIMENT_ID}] {filename} w={window_samples} "
                      f"rmse pca/empca/zero = {r['rmse_pca']:.4f}/{r['rmse_empca']:.4f}/"
                      f"{r['rmse_zero']:.4f} n_eval={r['n_eval']}", flush=True)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                failures.append({"file": filename, "window_samples": window_samples,
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{EXPERIMENT_ID}] FAILED {filename} w={window_samples}: {exc}", flush=True)

    if rows:
        rc.write_rows_csv(out / "window_sweep.csv",
                          [{k: v for k, v in r.items() if k != "_full"} for r in rows], COLUMNS)
        (out / "metric_reversal_full.json").write_text(json.dumps(
            {f"{r['file']}|{r['window_samples']}": r["_full"] for r in rows},
            indent=1, default=str))

    smallest_beating = [r for r in rows if r["file"] == files[0] and r["empca_beats_zero"]]
    summary = {
        "status": "complete" if rows and not failures else ("partial" if rows else "not_run"),
        "files": files, "windows": windows_list, "config": config_path,
        "failures": failures,
        "predictions": {
            "T_a_direction_ordering_holds_everywhere": {
                "n_rows": len(rows),
                "n_ok": sum(1 for r in rows if r["direction_ok"]),
                "ok": bool(rows and all(r["direction_ok"] for r in rows)),
            },
            "T_b_empca_beats_zero_predictor_somewhere_on_first_file": {
                "smallest_window_beating_zero": (min(r["window_samples"] for r in smallest_beating)
                                                 if smallest_beating else None),
                "ok": bool(smallest_beating),
                "statement_if_failed": (
                    f"the SNR floor is not reached at {max(windows_list)} samples on {files[0]}"
                    if not smallest_beating else ""),
            },
        },
        "walltime_s": time.time() - t_start, "host": platform.node(),
    }
    rc.write_json(out / "summary.json", common.as_jsonable(summary))
    print(json.dumps(summary["predictions"], indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=["abra_validation_0000.h5", "abra_validation_0005.h5",
                             "abra_validation_0010.h5"])
    ap.add_argument("--windows", type=int, nargs="+", default=[4096, 16384, 65536])
    ap.add_argument("--config", type=str,
                    default="configs/tidmad/phase2_metric_reversal_real_noise.yaml")
    a = ap.parse_args(argv)
    print(f"wrote {run(a.files, a.windows, a.config)}")


if __name__ == "__main__":
    main()
