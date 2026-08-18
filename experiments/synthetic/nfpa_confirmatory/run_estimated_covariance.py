"""E10 - one fixed estimated-covariance condition for NFPA.

Every synthetic result in the study whitens with the oracle ``Sigma``. This
runner measures what estimating it costs, under frozen choices that the
executing agent has no discretion to change:

* seeds 0-9, ``N_cal = 2048`` independent noise-only calibration events;
* the calibration RNG is seeded from
  ``numpy.random.SeedSequence([20260819, seed, 10])``, separate from the dataset
  RNG, and draws from the same oracle colored model with no signal and no
  train/validation/test observations;
* ``Sigma_c`` is estimated from rows shaped ``(N_cal*T, C)`` and ``Sigma_t``
  from rows shaped ``(N_cal*C, T)`` with
  ``noise_geometry.noise.estimate_covariance(center=True)``;
* each axis is passed through ``regularize_covariance(shrinkage=0.0,
  floor=None)`` - the existing data-scaled eigenvalue floor is the only
  regularization applied;
* data are generated with the oracle covariance; NFPA is fitted and
  reconstructs through the *estimated* axis covariances; the reconstruction is
  then evaluated only with the *oracle* covariance on the unchanged held-out
  test set;
* ``Iso-MPCA`` is refitted on identical data as the covariance-independent
  control and must reproduce its archived primary metrics to 1e-9 relative.

Two identifiability cautions are built into the diagnostics. The Kronecker
factors are identified only up to a reciprocal scale, so no naive per-axis raw
relative error is reported; trace-normalized *shape* errors, one global scale
ratio, and the full-covariance error are reported instead. And the replication
unit is the calibration *event*, not the flattened row: rows within an event are
correlated, so intervals are never computed as though the rows were independent.

This is one fixed condition, not the originally proposed covariance-robustness
sweep. It measures a sensitivity, not a robustness boundary.

Usage::

    python -m experiments.synthetic.nfpa_confirmatory.run_estimated_covariance \
        --seeds 0 ... 9 --out-dir results/rps2026/E10_estimated_covariance [--resume]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from noise_geometry.noise import estimate_covariance, regularize_covariance

from . import r9_common as rc
from .config import PRIMARY, config_hash, seed_list
from .covariance import _sqrt_inv, assert_checks_pass, run_numerical_checks
from .generator import generate_noise, make_dataset
from .methods import fit_iso_mpca, fit_nfpa, reconstruct
from .metrics import normalized_chi2, raw_mse

#: Frozen calibration seed-sequence entropy, distinct from the dataset RNG.
CALIBRATION_SEED_ENTROPY = 20260819
CALIBRATION_STREAM_ID = 10

COLUMNS = [
    "config_hash",
    "seed",
    "regime",
    "method",
    "whitening",
    "n_calibration_events",
    "test_raw_mse",
    "test_normalized_chi2",
    "archived_test_raw_mse",
    "archived_test_normalized_chi2",
    "delta_raw_mse_estimated_minus_oracle",
    "delta_normalized_chi2_estimated_minus_oracle",
    "control_reproduction_rel_error_raw_mse",
    "control_reproduction_rel_error_normalized_chi2",
    "channel_shape_error",
    "temporal_shape_error",
    "kronecker_scale_ratio",
    "kronecker_relative_frobenius_error",
    "channel_min_eigenvalue",
    "channel_condition_number",
    "temporal_min_eigenvalue",
    "temporal_condition_number",
    "oracle_whitening_residual",
    "runtime_seconds",
]


# --------------------------------------------------------------------------
# Calibration and estimation
# --------------------------------------------------------------------------


def calibration_rng(seed: int) -> np.random.Generator:
    """The frozen calibration RNG, independent of the dataset RNG."""
    return np.random.default_rng(
        np.random.SeedSequence([CALIBRATION_SEED_ENTROPY, int(seed), CALIBRATION_STREAM_ID])
    )


def channel_samples(noise: np.ndarray) -> np.ndarray:
    """``(N_cal, C, T) -> (N_cal*T, C)``; each row is one time slice's channels."""
    return noise.transpose(0, 2, 1).reshape(-1, noise.shape[1])


def temporal_samples(noise: np.ndarray) -> np.ndarray:
    """``(N_cal, C, T) -> (N_cal*C, T)``; each row is one channel's time series."""
    return noise.reshape(-1, noise.shape[2])


def estimate_axis_covariances(noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and floor the two axis covariances from calibration noise."""
    Sc = regularize_covariance(
        estimate_covariance(channel_samples(noise), center=True), shrinkage=0.0, floor=None
    )
    St = regularize_covariance(
        estimate_covariance(temporal_samples(noise), center=True), shrinkage=0.0, floor=None
    )
    return Sc, St


# --------------------------------------------------------------------------
# Kronecker-aware diagnostics
# --------------------------------------------------------------------------


def _fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def shape_error(estimated: np.ndarray, oracle: np.ndarray) -> float:
    """Relative Frobenius error after trace-normalizing both factors.

    The two Kronecker factors are identified only up to a reciprocal scale, so a
    raw per-axis relative error is not a meaningful accuracy statement. Removing
    the trace makes the comparison scale-free and therefore identified.
    """
    a = np.asarray(estimated, dtype=float) / float(np.trace(estimated))
    b = np.asarray(oracle, dtype=float) / float(np.trace(oracle))
    return float(_fro(a - b) / max(_fro(b), 1e-300))


def kron_relative_frobenius_error(
    Sc_hat: np.ndarray, St_hat: np.ndarray, Sc: np.ndarray, St: np.ndarray
) -> float:
    """``||St_hat (x) Sc_hat - St (x) Sc||_F / ||St (x) Sc||_F`` without dense products.

    Uses ``||A (x) B||_F = ||A||_F ||B||_F`` and
    ``<A1 (x) B1, A2 (x) B2> = <A1, A2> <B1, B2>``, so the ``(C*T)^2`` matrix is
    never materialized.
    """
    num_sq = (
        _fro(St_hat) ** 2 * _fro(Sc_hat) ** 2
        - 2.0 * float(np.sum(St_hat * St)) * float(np.sum(Sc_hat * Sc))
        + _fro(St) ** 2 * _fro(Sc) ** 2
    )
    denom = _fro(St) * _fro(Sc)
    return float(np.sqrt(max(num_sq, 0.0)) / max(denom, 1e-300))


def oracle_whitening_residual(
    Sc_hat: np.ndarray, St_hat: np.ndarray, Sc: np.ndarray, St: np.ndarray
) -> float:
    """``||W_hat Sigma_oracle W_hat^T - I||_F / (C*T)`` via Kronecker identities.

    With ``A_c = W_c Sigma_c W_c^T`` and ``A_t = W_t Sigma_t W_t^T``,
    ``||A_t (x) A_c - I||_F^2 = ||A_t||_F^2 ||A_c||_F^2 - 2 tr(A_t) tr(A_c) + C*T``.
    """
    Wc = _sqrt_inv(Sc_hat)
    Wt = _sqrt_inv(St_hat)
    A_c = Wc @ Sc @ Wc.T
    A_t = Wt @ St @ Wt.T
    C, T = Sc.shape[0], St.shape[0]
    val_sq = _fro(A_t) ** 2 * _fro(A_c) ** 2 - 2.0 * float(np.trace(A_t)) * float(np.trace(A_c)) + C * T
    return float(np.sqrt(max(val_sq, 0.0)) / (C * T))


def _cond(A: np.ndarray) -> tuple[float, float]:
    vals = np.linalg.eigvalsh(0.5 * (A + A.T))
    lo, hi = float(vals.min()), float(vals.max())
    return lo, float(hi / lo) if lo > 0 else float("inf")


def covariance_diagnostics(Sc_hat, St_hat, Sc, St) -> dict[str, float]:
    c_min, c_cond = _cond(Sc_hat)
    t_min, t_cond = _cond(St_hat)
    return {
        "channel_shape_error": shape_error(Sc_hat, Sc),
        "temporal_shape_error": shape_error(St_hat, St),
        "kronecker_scale_ratio": float(
            (np.trace(St_hat) * np.trace(Sc_hat)) / (np.trace(St) * np.trace(Sc))
        ),
        "kronecker_relative_frobenius_error": kron_relative_frobenius_error(Sc_hat, St_hat, Sc, St),
        "channel_min_eigenvalue": c_min,
        "channel_condition_number": c_cond,
        "temporal_min_eigenvalue": t_min,
        "temporal_condition_number": t_cond,
        "oracle_whitening_residual": oracle_whitening_residual(Sc_hat, St_hat, Sc, St),
    }


# --------------------------------------------------------------------------
# Per-seed execution
# --------------------------------------------------------------------------


def run_seed(seed: int, cfg, archived: dict | None) -> tuple[list[dict], bool]:
    """Fit NFPA under estimated whitening and Iso-MPCA as the control."""
    t0 = time.time()
    ds = make_dataset(seed, "colored", cfg)
    report = run_numerical_checks(cfg, ds.covariances["Sigma"])
    assert_checks_pass(report)
    ds.covariance_report = report

    Sc, St = ds.covariances["Sigma_c"], ds.covariances["Sigma_t"]
    noise = generate_noise(calibration_rng(seed), cfg.N_calibration, cfg, ds.covariances)
    if noise.shape != (cfg.N_calibration, cfg.C, cfg.T):
        raise rc.IntegrityError(f"calibration noise has shape {noise.shape}")
    Sc_hat, St_hat = estimate_axis_covariances(noise)
    diag = covariance_diagnostics(Sc_hat, St_hat, Sc, St)

    # NFPA fitted and reconstructing through the ESTIMATED axis covariances.
    ds_est = replace(ds, covariances={"Sigma_c": Sc_hat, "Sigma_t": St_hat})
    fit_est = fit_nfpa(ds_est)
    Xhat_est = reconstruct(fit_est, ds_est, ds.X_test)
    # ...but scored only under the ORACLE covariance, on the unchanged test set.
    nfpa_raw = raw_mse(ds.X_test, Xhat_est)
    nfpa_chi2 = normalized_chi2(ds, ds.X_test, Xhat_est)

    # Iso-MPCA: covariance-independent control on identical data.
    fit_iso = fit_iso_mpca(ds)
    Xhat_iso = reconstruct(fit_iso, ds, ds.X_test)
    iso_raw = raw_mse(ds.X_test, Xhat_iso)
    iso_chi2 = normalized_chi2(ds, ds.X_test, Xhat_iso)

    runtime = time.time() - t0
    cfg_hash = config_hash(cfg)
    base = {
        "config_hash": cfg_hash,
        "seed": seed,
        "regime": "colored",
        "n_calibration_events": cfg.N_calibration,
        "runtime_seconds": runtime,
        **diag,
    }

    def archived_for(method: str, key: str):
        if archived is None:
            return None
        return archived.get((seed, method), {}).get(key)

    rows = []
    a_raw = archived_for("NFPA", "test_raw_mse")
    a_chi2 = archived_for("NFPA", "test_normalized_chi2")
    rows.append(
        {
            **base,
            "method": "NFPA",
            "whitening": "estimated",
            "test_raw_mse": nfpa_raw,
            "test_normalized_chi2": nfpa_chi2,
            "archived_test_raw_mse": a_raw,
            "archived_test_normalized_chi2": a_chi2,
            "delta_raw_mse_estimated_minus_oracle": (nfpa_raw - a_raw) if a_raw is not None else None,
            "delta_normalized_chi2_estimated_minus_oracle": (
                (nfpa_chi2 - a_chi2) if a_chi2 is not None else None
            ),
            "control_reproduction_rel_error_raw_mse": None,
            "control_reproduction_rel_error_normalized_chi2": None,
        }
    )

    i_raw = archived_for("Iso-MPCA", "test_raw_mse")
    i_chi2 = archived_for("Iso-MPCA", "test_normalized_chi2")
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
            "archived_test_raw_mse": i_raw,
            "archived_test_normalized_chi2": i_chi2,
            "delta_raw_mse_estimated_minus_oracle": None,
            "delta_normalized_chi2_estimated_minus_oracle": None,
            "control_reproduction_rel_error_raw_mse": rel_raw,
            "control_reproduction_rel_error_normalized_chi2": rel_chi2,
        }
    )
    return rows, control_ok


def load_archived() -> dict[tuple[int, str], dict[str, float]]:
    table: dict[tuple[int, str], dict[str, float]] = {}
    for row in rc.read_rows_csv(rc.ARCHIVED_E1_CSV):
        table[(int(row["seed"]), row["method"])] = {
            "test_raw_mse": float(row["test_raw_mse"]),
            "test_normalized_chi2": float(row["test_normalized_chi2"]),
        }
    return table


def summarize(rows, cfg, seeds) -> dict:
    nfpa = [r for r in rows if r["method"] == "NFPA"]
    degradation = {}
    for key, col in (
        ("raw_mse", "delta_raw_mse_estimated_minus_oracle"),
        ("normalized_chi2", "delta_normalized_chi2_estimated_minus_oracle"),
    ):
        values = [rc._as_float(r[col]) for r in nfpa]
        if any(v is None for v in values):
            degradation[key] = {"identified": False, "reason": "no archived oracle row available"}
        else:
            block = rc.bootstrap_paired(values)
            block["sign_convention"] = (
                "estimated minus archived oracle NFPA; positive means estimation made it worse"
            )
            degradation[key] = block

    contrasts = {
        metric: rc.paired_contrast(rows, metric=metric, method_a="NFPA", method_b="Iso-MPCA")
        for metric in ("test_raw_mse", "test_normalized_chi2")
    }

    diagnostics = {}
    for key in (
        "channel_shape_error",
        "temporal_shape_error",
        "kronecker_scale_ratio",
        "kronecker_relative_frobenius_error",
        "channel_min_eigenvalue",
        "channel_condition_number",
        "temporal_min_eigenvalue",
        "temporal_condition_number",
        "oracle_whitening_residual",
    ):
        values = [float(rc._as_float(r[key])) for r in nfpa]
        diagnostics[key] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "per_seed": {str(int(r["seed"])): float(rc._as_float(r[key])) for r in nfpa},
        }

    return {
        "experiment": "E10_estimated_covariance",
        "config_hash": config_hash(cfg),
        "seeds": [int(s) for s in seeds],
        "n_calibration_events": cfg.N_calibration,
        "replication_unit": (
            "calibration EVENT. Rows within an event are correlated, so neither N_cal*T nor "
            "N_cal*C is an independent sample count. All intervals here bootstrap the ten "
            "seed-level paired differences only."
        ),
        "identifiability_note": (
            "The two Kronecker factors are identified only up to a reciprocal scale. No naive "
            "per-axis raw relative error is reported; the shape errors are trace-normalized and "
            "the scale is reported once, globally."
        ),
        "scale_dominance_note": (
            "The per-axis sample estimators are unbiased only up to that same reciprocal scale: "
            "E[Sigma_c_hat] = Sigma_c * tr(Sigma_t)/T and E[Sigma_t_hat] = Sigma_t * tr(Sigma_c)/C. "
            "So kronecker_scale_ratio has an expected value of tr(Sigma_t)*tr(Sigma_c)/(C*T) rather "
            "than 1, and both kronecker_relative_frobenius_error and oracle_whitening_residual are "
            "dominated by that global factor rather than by estimation error. They are reported "
            "here exactly as the runbook defines them. The scale-free accuracy statements are the "
            "two shape errors. NFPA's fitted subspace and its reconstruction are invariant to a "
            "global rescaling of the whitening transform -- ALS minimises a scale-equivariant "
            "objective and the reconstruction unwhitens through the same scaled transform -- so "
            "this factor does not propagate into the reported held-out metrics."
        ),
        "scope_note": (
            "One fixed estimated-covariance condition: a measured sensitivity check, not a "
            "general covariance-robustness boundary."
        ),
        "evaluation_note": (
            "NFPA fits and reconstructs through the estimated axis covariances; every reported "
            "metric is computed with the oracle covariance on the unchanged held-out test set."
        ),
        "nfpa_degradation_estimated_minus_oracle": degradation,
        "paired_contrasts_nfpa_minus_iso_mpca_under_estimated_whitening": contrasts,
        "covariance_diagnostics": diagnostics,
        "bootstrap": {"resamples": rc.BOOTSTRAP_RESAMPLES, "seed": rc.BOOTSTRAP_SEED, "unit": "seed"},
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="E10 estimated covariance condition")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    cfg = rc.smoke_config() if args.smoke else PRIMARY
    expected_seeds = list(rc.SMOKE_SEEDS) if args.smoke else seed_list(PRIMARY.n_seeds)
    try:
        seeds = rc.guard_seed_set(args.seeds or expected_seeds, expected_seeds, smoke=args.smoke)
        out_dir = rc.resolve_out_dir(args.out_dir, smoke=args.smoke)
    except rc.IntegrityError as exc:
        print(f"[E10] INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    prov = rc.provenance(["run_estimated_covariance", *argv], cfg)

    print(f"[E10] resolved output directory: {out_dir}", flush=True)
    print(f"[E10] config_hash={config_hash(cfg)} N_cal={cfg.N_calibration} seeds={seeds}", flush=True)

    archived = None if args.smoke else load_archived()

    expected_manifest = {
        "experiment": "E10_estimated_covariance",
        "config_hash": config_hash(cfg),
        "regime": "colored",
        "methods": ["Iso-MPCA", "NFPA"],
        "seeds": seeds,
        "columns": COLUMNS,
    }
    try:
        existing = rc.prepare_output_dir(out_dir, resume=args.resume, expected=expected_manifest)
    except rc.IntegrityError as exc:
        print(f"[E10] INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 2

    for seed in seeds:
        if seed in existing:
            print(f"[E10] seed {seed}: reusing validated shard {existing[seed]}", flush=True)
            continue
        rows, control_ok = run_seed(seed, cfg, archived)
        if not control_ok:
            print(
                f"[E10] CONTROL GATE FAILED at seed {seed}: the Iso-MPCA refit does not reproduce "
                f"its archived primary metrics to {rc.REPRODUCTION_RTOL:g} relative; "
                f"no shard written",
                file=sys.stderr,
            )
            return 4
        rc.write_rows_csv(rc.shard_path(out_dir, seed), rows, COLUMNS)
        print(f"[E10] seed {seed}: wrote shard ({rows[0]['runtime_seconds']:.1f}s)", flush=True)

    shards = sorted(out_dir.glob("seed_*.csv"))
    if len(shards) != len(seeds):
        print(f"[E10] expected {len(seeds)} shards, found {len(shards)}", file=sys.stderr)
        return 5
    raw = rc.concat_shards(shards)
    rc.write_rows_csv(out_dir / "raw.csv", raw, COLUMNS)
    rc.write_json(out_dir / "summary.json", summarize(raw, cfg, seeds))

    ended = datetime.now(timezone.utc)
    manifest = dict(expected_manifest)
    manifest.update(
        {
            "calibration_seed_sequence": [CALIBRATION_SEED_ENTROPY, "<seed>", CALIBRATION_STREAM_ID],
            "regularization": "regularize_covariance(shrinkage=0.0, floor=None)",
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "wall_time_seconds": (ended - started).total_seconds(),
            "shard_digests": {p.name: rc.sha256_of(p) for p in shards},
            "raw_digest": rc.sha256_of(out_dir / "raw.csv"),
            "summary_digest": rc.sha256_of(out_dir / "summary.json"),
            "archived_e1_digest": rc.sha256_of(rc.ARCHIVED_E1_CSV) if not args.smoke else None,
            "provenance": prov,
        }
    )
    rc.write_json(out_dir / "manifest.json", manifest)
    print(f"[E10] complete: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
