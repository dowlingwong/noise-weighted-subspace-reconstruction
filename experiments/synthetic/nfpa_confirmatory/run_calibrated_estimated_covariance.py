"""E10b: NFPA under CALIBRATED estimated covariance (Round 10).

One fixed covariance-estimation sensitivity condition
(``analysis_label = single_condition_covariance_estimation_sensitivity``):
identical to Round 9's E10 in every scientific choice — seeds 0–9,
``N_cal = 2048``, the same calibration RNG stream, the same
positive-definiteness regularization — with exactly one change, frozen before
any full rerun: the symmetric product-scale calibration of
``covariance_estimation_v2``. Evaluation is unchanged: the ORACLE covariance
on the unchanged held-out test set.

Because NFPA's fit and reconstruction are invariant under a global rescaling
of both whitening factors, the *contrasts* here should agree with E10's; what
E10b corrects is the diagnostics, which were previously dominated by an
uncalibrated global scale and are now interpretable as estimation accuracy.
No scientific success threshold is imposed on degradation or reversal; an
unfavorable outcome is recorded and propagated to the limitations.

Usage
-----
    PYTHONPATH=src:experiments/synthetic python -m nfpa_confirmatory.run_calibrated_estimated_covariance \
        --out results/rps2026/E10b_calibrated_estimated_covariance [--resume]
    # smoke: ... --smoke --out results/rps2026/Round10_smoke/E10b
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from . import r10_common as rc
from .config import PRIMARY, config_hash
from .covariance import assert_checks_pass, run_numerical_checks
from .covariance_estimation_v2 import estimate_calibrated_axis_covariances
from .generator import make_dataset
from .metrics import normalized_chi2, raw_mse
from .methods import fit_iso_mpca, fit_nfpa, reconstruct
from .run_estimated_covariance import (
    calibration_rng,
    covariance_diagnostics,
    generate_noise,
)

EXPERIMENT = "E10b_calibrated_estimated_covariance"
ANALYSIS_LABEL = "single_condition_covariance_estimation_sensitivity"
SEEDS = tuple(range(10))

COLUMNS = (
    "config_hash",
    "seed",
    "regime",
    "method",
    "analysis_label",
    "whitening",
    "n_calibration_events",
    "test_raw_mse",
    "test_normalized_chi2",
    "archived_oracle_test_raw_mse",
    "archived_oracle_test_normalized_chi2",
    "delta_raw_mse_estimated_minus_oracle",
    "delta_normalized_chi2_estimated_minus_oracle",
    "control_reproduction_rel_error_raw_mse",
    "control_reproduction_rel_error_normalized_chi2",
    "v_c",
    "v_t",
    "v_hat",
    "product_scale_ratio_vs_oracle",
    "channel_shape_error",
    "temporal_shape_error",
    "kronecker_relative_frobenius_error",
    "oracle_whitening_residual",
    "channel_min_eigenvalue",
    "channel_condition_number",
    "temporal_min_eigenvalue",
    "temporal_condition_number",
    "runtime_seconds",
)


def load_archived() -> dict[tuple[int, str], dict[str, float]]:
    table: dict[tuple[int, str], dict[str, float]] = {}
    for row in rc.read_rows_csv(rc.ARCHIVED_E1_CSV):
        table[(int(row["seed"]), row["method"])] = {
            "test_raw_mse": float(row["test_raw_mse"]),
            "test_normalized_chi2": float(row["test_normalized_chi2"]),
        }
    return table


def run_seed(seed: int, cfg, archived: dict | None) -> tuple[list[dict], bool]:
    t0 = time.time()
    ds = make_dataset(seed, "colored", cfg)
    report = run_numerical_checks(cfg, ds.covariances["Sigma"])
    assert_checks_pass(report)
    ds.covariance_report = report
    cfg_hash = config_hash(cfg)

    Sc, St = ds.covariances["Sigma_c"], ds.covariances["Sigma_t"]
    noise = generate_noise(calibration_rng(seed), cfg.N_calibration, cfg, ds.covariances)
    if noise.shape != (cfg.N_calibration, cfg.C, cfg.T):
        raise rc.IntegrityError(f"calibration noise has shape {noise.shape}")

    est = estimate_calibrated_axis_covariances(noise)
    diag = covariance_diagnostics(est.Sigma_c_hat, est.Sigma_t_hat, Sc, St)
    # After calibration, the product-scale ratio is a real accuracy statement.
    product_scale_ratio = diag.pop("kronecker_scale_ratio")

    ds_est = replace(ds, covariances={"Sigma_c": est.Sigma_c_hat, "Sigma_t": est.Sigma_t_hat})
    fit_est = fit_nfpa(ds_est)
    Xhat_est = reconstruct(fit_est, ds_est, ds.X_test)
    nfpa_raw = raw_mse(ds.X_test, Xhat_est)
    nfpa_chi2 = normalized_chi2(ds, ds.X_test, Xhat_est)

    fit_iso = fit_iso_mpca(ds)
    Xhat_iso = reconstruct(fit_iso, ds, ds.X_test)
    iso_raw = raw_mse(ds.X_test, Xhat_iso)
    iso_chi2 = normalized_chi2(ds, ds.X_test, Xhat_iso)

    runtime = time.time() - t0
    base = {
        "config_hash": cfg_hash,
        "seed": seed,
        "regime": "colored",
        "analysis_label": ANALYSIS_LABEL,
        "n_calibration_events": cfg.N_calibration,
        "v_c": est.v_c,
        "v_t": est.v_t,
        "v_hat": est.v_hat,
        "product_scale_ratio_vs_oracle": product_scale_ratio,
        **diag,
        "runtime_seconds": runtime,
    }

    def arch(method: str, key: str):
        if archived is None:
            return None
        rec = archived.get((seed, method))
        return None if rec is None else rec[key]

    rows: list[dict] = []
    a_raw = arch("NFPA", "test_raw_mse")
    a_chi2 = arch("NFPA", "test_normalized_chi2")
    rows.append(
        {
            **base,
            "method": "NFPA",
            "whitening": "estimated_calibrated_v2",
            "test_raw_mse": nfpa_raw,
            "test_normalized_chi2": nfpa_chi2,
            "archived_oracle_test_raw_mse": a_raw,
            "archived_oracle_test_normalized_chi2": a_chi2,
            "delta_raw_mse_estimated_minus_oracle": (nfpa_raw - a_raw) if a_raw is not None else None,
            "delta_normalized_chi2_estimated_minus_oracle": (
                (nfpa_chi2 - a_chi2) if a_chi2 is not None else None
            ),
            "control_reproduction_rel_error_raw_mse": None,
            "control_reproduction_rel_error_normalized_chi2": None,
        }
    )

    i_raw = arch("Iso-MPCA", "test_raw_mse")
    i_chi2 = arch("Iso-MPCA", "test_normalized_chi2")
    rel_raw = abs(iso_raw - i_raw) / max(abs(i_raw), 1e-15) if i_raw is not None else None
    rel_chi2 = abs(iso_chi2 - i_chi2) / max(abs(i_chi2), 1e-15) if i_chi2 is not None else None
    control_ok = True
    if rel_raw is not None:
        control_ok = rel_raw <= rc.REPRODUCTION_RTOL and rel_chi2 <= rc.REPRODUCTION_RTOL
    rows.append(
        {
            **base,
            "method": "Iso-MPCA",
            "whitening": "none_covariance_independent_control",
            "test_raw_mse": iso_raw,
            "test_normalized_chi2": iso_chi2,
            "archived_oracle_test_raw_mse": i_raw,
            "archived_oracle_test_normalized_chi2": i_chi2,
            "delta_raw_mse_estimated_minus_oracle": None,
            "delta_normalized_chi2_estimated_minus_oracle": None,
            "control_reproduction_rel_error_raw_mse": rel_raw,
            "control_reproduction_rel_error_normalized_chi2": rel_chi2,
        }
    )
    return rows, control_ok


def summarize(rows, seeds) -> dict:
    out = {
        "experiment": EXPERIMENT,
        "analysis_label": ANALYSIS_LABEL,
        "n_seeds": len(seeds),
        "scope_note": (
            "One fixed calibration size and covariance condition. This measures "
            "sensitivity to estimation at N_cal=2048 under the frozen colored model; "
            "it does not establish a general covariance-robustness boundary."
        ),
        "scale_note": (
            "Product-scale calibration (corrected here, v_hat) and reciprocal "
            "per-factor scale ambiguity (inherent, handled by scale-free shape "
            "errors) are different issues and are reported separately."
        ),
    }
    nfpa = [r for r in rows if r["method"] == "NFPA"]
    for col in (
        "delta_normalized_chi2_estimated_minus_oracle",
        "delta_raw_mse_estimated_minus_oracle",
    ):
        out[f"degradation.{col}"] = rc.bootstrap_paired(
            [float(r[col]) for r in nfpa]
        )
    for metric in ("test_normalized_chi2", "test_raw_mse"):
        out[f"contrast_estimated_nfpa_minus_iso.{metric}"] = rc.paired_contrast(
            rows, metric=metric, method_a="NFPA", method_b="Iso-MPCA"
        )
    for col in (
        "v_c", "v_t", "v_hat", "product_scale_ratio_vs_oracle",
        "channel_shape_error", "temporal_shape_error",
        "kronecker_relative_frobenius_error", "oracle_whitening_residual",
    ):
        vals = np.asarray([float(r[col]) for r in nfpa])
        out[f"diagnostics.{col}"] = {
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(rc.RESULTS_ROOT / EXPERIMENT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)

    smoke = bool(args.smoke)
    cfg = rc.smoke_config() if smoke else PRIMARY
    seeds = args.seeds if args.seeds is not None else (list(rc.SMOKE_SEEDS) if smoke else list(SEEDS))
    seeds = rc.guard_seed_set(seeds, SEEDS, smoke=smoke)
    out_dir = rc.resolve_out_dir(args.out, smoke=smoke)
    archived = None if smoke else load_archived()

    expected_manifest = {
        "experiment": EXPERIMENT,
        "analysis_label": ANALYSIS_LABEL,
        "config_hash": config_hash(cfg),
        "regime": "colored",
        "methods": ["Iso-MPCA", "NFPA"],
        "columns": list(COLUMNS),
        "seeds": sorted(seeds),
        "smoke": smoke,
        "calibration": {
            "n_calibration_events": cfg.N_calibration,
            "rng": "unchanged from E10 (same SeedSequence stream)",
            "regularization": "unchanged from E10 (shrinkage=0.0, floor=None)",
            "scale_correction": "v_hat = sqrt(v_c*v_t); factors / sqrt(v_hat); frozen before any full rerun",
        },
    }
    existing = rc.prepare_output_dir(out_dir, resume=args.resume, expected=expected_manifest)
    rc.write_json(out_dir / "manifest.json", {**expected_manifest, "status": "in_progress"})

    t_start = time.time()
    controls_ok = True
    for seed in seeds:
        if seed in existing:
            print(f"[E10b] seed {seed}: reusing validated shard {existing[seed]}", flush=True)
            continue
        rows, control_ok = run_seed(seed, cfg, archived)
        if not smoke and not control_ok:
            raise rc.IntegrityError(
                f"Iso-MPCA control failed archived reproduction at seed {seed}; "
                "provenance failure — stopping before writing the shard"
            )
        controls_ok = controls_ok and control_ok
        rc.write_rows_csv(rc.shard_path(out_dir, seed), rows, COLUMNS)
        print(f"[E10b] seed {seed}: wrote shard", flush=True)

    shards = sorted(out_dir.glob("seed_*.csv"))
    if len(shards) != len(seeds):
        print(f"[E10b] expected {len(seeds)} shards, found {len(shards)}", file=sys.stderr)
        return 1
    all_rows = rc.concat_shards(shards)
    rc.write_rows_csv(out_dir / "raw.csv", all_rows, COLUMNS)
    if not smoke:
        rc.write_json(out_dir / "summary.json", summarize(all_rows, seeds))

    manifest = dict(expected_manifest)
    manifest.update(
        {
            "status": "complete",
            "control_gate_passed": controls_ok,
            "wall_seconds": time.time() - t_start,
            "shard_digests": {p.name: rc.sha256_of(p) for p in shards},
            "raw_digest": rc.sha256_of(out_dir / "raw.csv"),
            "provenance": rc.provenance(
                [sys.executable, "-m", "nfpa_confirmatory.run_calibrated_estimated_covariance", *sys.argv[1:]], cfg
            ),
        }
    )
    rc.write_json(out_dir / "manifest.json", manifest)
    print(f"[E10b] complete: {len(shards)} shards in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
