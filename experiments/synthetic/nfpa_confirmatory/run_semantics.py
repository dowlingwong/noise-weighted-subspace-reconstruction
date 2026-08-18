"""E8 - semantic completion: planted-factor recovery and parameter decoding.

Two conditions are run and reported separately, because they have different
evidential status and must never be pooled:

``primary`` (``alpha = 0.1``, seeds 0-9)
    The archived Block A configuration. Every fit must first reproduce the
    archived ``test_normalized_chi2`` to a relative error of 1e-9, which proves
    the refit *is* the archived fit. The decoding analysis carried out here is a
    post-hoc completion of the superseded Block H and is labelled
    ``post_hoc_exploratory`` in every row and summary.

``separable`` (``alpha = 0``, seeds 0-9)
    The separable condition named by the predeclared hypothesis H3, built with
    ``dataclasses.replace(PRIMARY, nonseparability_alpha=0.0)`` and carrying its
    own configuration hash. There is no archived fit to reproduce. This is the
    delayed execution of H3 and is labelled
    ``delayed_predeclared_no_numeric_threshold``. The original protocol supplied
    no quantitative recovery threshold, so continuous errors and seed-level
    paired contrasts are reported and no binary H3 verdict is manufactured.

Usage::

    python -m experiments.synthetic.nfpa_confirmatory.run_semantics \
        --condition primary --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out-dir results/rps2026/E8_semantics [--resume]

``--gate-only`` runs the archived-fit reproduction gate for the requested seeds
and writes only the audit table, which is how the Phase 3 one-seed pre-check is
performed without creating a partial result directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import r9_common as rc
from .config import PRIMARY, config_hash, seed_list
from .covariance import assert_checks_pass, run_numerical_checks
from .generator import make_dataset
from .methods import FIT_FACTORY, fit_cw_pca
from .metrics import evaluate_fit
from .r9_semantics import SEMANTIC_METRIC_COLUMNS, evaluate_semantics

CONDITIONS = {
    "primary": {
        "label": "primary_alpha_0p1",
        "alpha": 0.1,
        "analysis_status": "post_hoc_exploratory",
        "reproduction_gate": True,
    },
    "separable": {
        "label": "separable_alpha_0",
        "alpha": 0.0,
        "analysis_status": "delayed_predeclared_no_numeric_threshold",
        "reproduction_gate": False,
    },
}

COLUMNS = [
    "config_hash",
    "seed",
    "regime",
    "condition",
    "analysis_status",
    "method",
    "metric",
    "rep_class",
    "coord",
    "nonseparability_alpha",
    "C",
    "T",
    "k_c",
    "k_t",
    "total_rank",
    "N_train",
    "N_test",
    "code_dimension",
    "decoder_numerical_rank",
    "decoder_zero_variance_columns",
    *SEMANTIC_METRIC_COLUMNS,
    "test_raw_mse",
    "test_normalized_chi2",
    "archived_reproduction_rel_error",
    "runtime_seconds",
]

AUDIT_COLUMNS = [
    "seed",
    "method",
    "expected_test_normalized_chi2",
    "observed_test_normalized_chi2",
    "absolute_error",
    "relative_error",
    "tolerance",
    "passed",
]

#: Contrasts reported for E8. The sign convention is NFPA minus Iso-MPCA.
E8_CONTRAST_METRICS = (
    "amplitude_rmse",
    "rise_time_rmse",
    "decay_time_rmse",
    "channel_projector_error",
    "temporal_projector_error",
    "joint_projector_error",
    "joint_projector_error_whitened",
)


def load_archived_chi2() -> dict[tuple[int, str], float]:
    """Archived ``test_normalized_chi2`` per (seed, method), as unrounded floats."""
    rows = rc.read_rows_csv(rc.ARCHIVED_E1_CSV)
    table: dict[tuple[int, str], float] = {}
    for row in rows:
        table[(int(row["seed"]), row["method"])] = float(row["test_normalized_chi2"])
    return table


def relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), 1e-15)


def fit_seed(seed: int, cfg, *, regime: str = "colored"):
    """Build one seed's dataset and fit all five registered methods."""
    ds = make_dataset(seed, regime, cfg)
    report = run_numerical_checks(cfg, ds.covariances["Sigma"])
    assert_checks_pass(report)
    ds.covariance_report = report
    cw_pca = fit_cw_pca(ds)
    fits = {}
    runtimes = {}
    for name in rc.REGISTERED_METHODS:
        t0 = time.time()
        fits[name] = cw_pca if name == "CW-PCA" else FIT_FACTORY[name](ds)
        runtimes[name] = time.time() - t0
    return ds, cw_pca, fits, runtimes


def run_reproduction_gate(seed: int, ds, cw_pca, fits, archived) -> tuple[list[dict], bool]:
    """Compare each refit's held-out chi-square with the archived value."""
    audit_rows: list[dict] = []
    ok = True
    for name in rc.REGISTERED_METHODS:
        expected = archived.get((seed, name))
        if expected is None:
            raise rc.IntegrityError(f"archived Block A has no row for seed {seed} method {name}")
        observed = evaluate_fit(ds, fits[name], cw_pca, ds.X_test)["normalized_chi2"]
        rel = relative_error(observed, expected)
        passed = rel <= rc.REPRODUCTION_RTOL
        ok = ok and passed
        audit_rows.append(
            {
                "seed": seed,
                "method": name,
                "expected_test_normalized_chi2": expected,
                "observed_test_normalized_chi2": observed,
                "absolute_error": abs(observed - expected),
                "relative_error": rel,
                "tolerance": rc.REPRODUCTION_RTOL,
                "passed": passed,
            }
        )
    return audit_rows, ok


def build_rows(seed: int, ds, cw_pca, fits, runtimes, spec, cfg, repro_by_method) -> tuple[list[dict], dict]:
    cfg_hash = config_hash(cfg)
    rows = []
    decoders = {}
    for name in rc.REGISTERED_METHODS:
        fit = fits[name]
        sem = evaluate_semantics(fit, ds)
        decoders[name] = sem.pop("decoder")
        test_metrics = evaluate_fit(ds, fit, cw_pca, ds.X_test)
        row = {
            "config_hash": cfg_hash,
            "seed": seed,
            "regime": ds.regime,
            "condition": spec["label"],
            "analysis_status": spec["analysis_status"],
            "method": name,
            "metric": fit.metric,
            "rep_class": fit.rep_class,
            "coord": fit.coord,
            "nonseparability_alpha": cfg.nonseparability_alpha,
            "C": cfg.C,
            "T": cfg.T,
            "k_c": cfg.k_c,
            "k_t": cfg.k_t,
            "total_rank": cfg.k_c * cfg.k_t,
            "N_train": cfg.N_train,
            "N_test": cfg.N_test,
            "code_dimension": sem["code_dimension"],
            "decoder_numerical_rank": decoders[name]["numerical_rank"],
            "decoder_zero_variance_columns": len(decoders[name]["zero_variance_columns"]),
            "test_raw_mse": test_metrics["raw_mse"],
            "test_normalized_chi2": test_metrics["normalized_chi2"],
            "archived_reproduction_rel_error": repro_by_method.get(name),
            "runtime_seconds": runtimes[name],
        }
        for key in SEMANTIC_METRIC_COLUMNS:
            row[key] = sem[key]
        rows.append(row)
    return rows, decoders


def summarize(rows: list[dict], spec, cfg, seeds) -> dict:
    """Deterministic summary: per-method means plus NFPA-minus-Iso-MPCA contrasts."""
    per_method: dict[str, dict] = {}
    for name in rc.REGISTERED_METHODS:
        subset = [r for r in rows if r["method"] == name]
        stats = {}
        for key in SEMANTIC_METRIC_COLUMNS:
            values = [rc._as_float(r[key]) for r in subset]
            if any(v is None for v in values):
                stats[key] = {"identified": False, "mean": None, "per_seed": None}
            else:
                stats[key] = {
                    "identified": True,
                    "mean": float(np.mean(values)),
                    "per_seed": {str(int(r["seed"])): float(rc._as_float(r[key])) for r in subset},
                }
        per_method[name] = stats

    contrasts = {}
    for metric in E8_CONTRAST_METRICS:
        contrasts[metric] = rc.paired_contrast(
            rows, metric=metric, method_a="NFPA", method_b="Iso-MPCA"
        )

    return {
        "condition": spec["label"],
        "analysis_status": spec["analysis_status"],
        "nonseparability_alpha": cfg.nonseparability_alpha,
        "config_hash": config_hash(cfg),
        "seeds": [int(s) for s in seeds],
        "methods": list(rc.REGISTERED_METHODS),
        "sign_convention": (
            "All contrasts are NFPA minus Iso-MPCA. A negative value favours NFPA, "
            "because every reported quantity is an error (lower is better)."
        ),
        "threshold_statement": (
            "The original protocol supplied no quantitative recovery threshold. "
            "Continuous errors and seed-level paired intervals are reported; no binary "
            "H3 verdict is derived here."
        ),
        "per_method": per_method,
        "paired_contrasts_nfpa_minus_iso_mpca": contrasts,
        "bootstrap": {
            "resamples": rc.BOOTSTRAP_RESAMPLES,
            "seed": rc.BOOTSTRAP_SEED,
            "unit": "seed",
        },
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="E8 semantic completion")
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="run the archived-fit reproduction gate only; write the audit table and exit",
    )
    args = parser.parse_args(argv)

    spec = CONDITIONS[args.condition]
    cfg_base = rc.smoke_config() if args.smoke else PRIMARY
    cfg = replace(cfg_base, nonseparability_alpha=spec["alpha"])

    expected_seeds = list(rc.SMOKE_SEEDS) if args.smoke else seed_list(PRIMARY.n_seeds)
    requested = args.seeds if args.seeds else expected_seeds
    try:
        seeds = rc.guard_seed_set(
            requested, expected_seeds, smoke=args.smoke, allow_subset=args.gate_only
        )
        out_dir = rc.resolve_out_dir(args.out_dir, smoke=args.smoke)
    except rc.IntegrityError as exc:
        print(f"[E8] INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 2
    cond_dir = out_dir / spec["label"]
    started = datetime.now(timezone.utc)
    prov = rc.provenance(["run_semantics", *argv], cfg)

    print(f"[E8] condition={args.condition} label={spec['label']}", flush=True)
    print(f"[E8] resolved output directory: {cond_dir}", flush=True)
    print(f"[E8] config_hash={config_hash(cfg)} alpha={cfg.nonseparability_alpha} seeds={seeds}", flush=True)

    archived = load_archived_chi2() if (spec["reproduction_gate"] and not args.smoke) else None
    if archived is not None:
        archived_hashes = {r["config_hash"] for r in rc.read_rows_csv(rc.ARCHIVED_E1_CSV)}
        if archived_hashes != {config_hash(cfg)}:
            print(
                f"[E8] FAIL archived config hash {archived_hashes} != this run's {config_hash(cfg)}",
                file=sys.stderr,
            )
            return 3

    # ---------------- gate-only pre-check ----------------
    if args.gate_only:
        if archived is None:
            print("[E8] --gate-only requires the primary condition without --smoke", file=sys.stderr)
            return 2
        audit: list[dict] = []
        ok = True
        for seed in seeds:
            ds, cw_pca, fits, _ = fit_seed(seed, cfg)
            rows, seed_ok = run_reproduction_gate(seed, ds, cw_pca, fits, archived)
            audit.extend(rows)
            ok = ok and seed_ok
            print(f"[E8] gate seed {seed}: {'PASS' if seed_ok else 'FAIL'}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "archived_fit_reproduction_gate_seed_precheck.csv"
        rc.write_rows_csv(path, audit, AUDIT_COLUMNS)
        print(f"[E8] wrote {path}", flush=True)
        return 0 if ok else 4

    expected_manifest = {
        "experiment": "E8_semantics",
        "condition": spec["label"],
        "config_hash": config_hash(cfg),
        "regime": "colored",
        "methods": list(rc.REGISTERED_METHODS),
        "seeds": seeds,
        "columns": COLUMNS,
    }
    try:
        existing = rc.prepare_output_dir(cond_dir, resume=args.resume, expected=expected_manifest)
    except rc.IntegrityError as exc:
        print(f"[E8] INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 2

    audit_rows: list[dict] = []
    for seed in seeds:
        if seed in existing:
            print(f"[E8] seed {seed}: reusing validated shard {existing[seed]}", flush=True)
            continue
        t0 = time.time()
        ds, cw_pca, fits, runtimes = fit_seed(seed, cfg)
        repro_by_method: dict[str, float] = {}
        if archived is not None:
            rows_audit, seed_ok = run_reproduction_gate(seed, ds, cw_pca, fits, archived)
            audit_rows.extend(rows_audit)
            repro_by_method = {r["method"]: r["relative_error"] for r in rows_audit}
            if not seed_ok:
                rc.write_rows_csv(out_dir / "archived_fit_reproduction.csv", audit_rows, AUDIT_COLUMNS)
                print(
                    f"[E8] ARCHIVED-FIT REPRODUCTION FAILED at seed {seed}; "
                    f"audit written, no semantic result recorded for this seed",
                    file=sys.stderr,
                )
                return 4
        rows, decoders = build_rows(seed, ds, cw_pca, fits, runtimes, spec, cfg, repro_by_method)
        rc.write_rows_csv(rc.shard_path(cond_dir, seed), rows, COLUMNS)
        rc.write_json(cond_dir / f"decoder_seed_{seed:02d}.json", {"seed": seed, "decoders": decoders})
        print(f"[E8] seed {seed}: wrote shard in {time.time() - t0:.1f}s", flush=True)

    # ---------------- deterministic aggregation ----------------
    shards = sorted(cond_dir.glob("seed_*.csv"))
    if len(shards) != len(seeds):
        print(f"[E8] expected {len(seeds)} shards, found {len(shards)}", file=sys.stderr)
        return 5
    raw = rc.concat_shards(shards)
    rc.write_rows_csv(cond_dir / "raw.csv", raw, COLUMNS)
    rc.write_json(cond_dir / "summary.json", summarize(raw, spec, cfg, seeds))
    if audit_rows:
        rc.write_rows_csv(out_dir / "archived_fit_reproduction.csv", audit_rows, AUDIT_COLUMNS)

    ended = datetime.now(timezone.utc)
    manifest = dict(expected_manifest)
    manifest.update(
        {
            "analysis_status": spec["analysis_status"],
            "nonseparability_alpha": cfg.nonseparability_alpha,
            "archived_reproduction_gate": bool(archived is not None),
            "reproduction_tolerance": rc.REPRODUCTION_RTOL,
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "wall_time_seconds": (ended - started).total_seconds(),
            "shard_digests": {p.name: rc.sha256_of(p) for p in shards},
            "raw_digest": rc.sha256_of(cond_dir / "raw.csv"),
            "summary_digest": rc.sha256_of(cond_dir / "summary.json"),
            "provenance": prov,
        }
    )
    rc.write_json(cond_dir / "manifest.json", manifest)
    rc.write_json(out_dir / "manifest.json", {"experiment": "E8_semantics", "last_condition": manifest})
    print(f"[E8] complete: {cond_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
