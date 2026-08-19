"""E11: corrected subspace angles (post-hoc diagnostic correction, Round 10).

Why this experiment exists
--------------------------
Round 9 pinned, and deliberately did not fix, a defect in the legacy
``methods.basis_whitened`` raw-factored branch: archived
``angle_to_cw_pca_*`` values for Iso-MPCA are not valid subspace angles
(see ``angles_v2``). E11 refits the archived configurations for seeds 0–29
and reports legacy and corrected angles side by side. All scientific
interpretation uses only the corrected ``*_v2`` columns.

Evidential status
-----------------
``analysis_label = post_hoc_diagnostic_correction``. No predicted angle
direction is introduced after observing the correction; corrected estimates
are reported whether or not they favour NFPA. The inferential unit is the
independent seed (10,000 seed-level bootstrap resamples, seed 20260819).
Cohorts are reported separately — frozen ``0–9``, replication ``10–29`` —
and a pooled cohort is shown only after both subsets, labelled
``post_hoc_pooled``.

Integrity gate
--------------
Before any angle row is accepted, the refit must reproduce the archived
``test_raw_mse`` and ``test_normalized_chi2`` for that (seed, method) to
relative error at most 1e-9, against E1 (seeds 0–9) or E9 (seeds 10–29).
A gate failure aborts the run; it is a provenance failure, not a result.

Usage
-----
    PYTHONPATH=src:experiments/synthetic python -m nfpa_confirmatory.run_corrected_angles \
        --out results/rps2026/E11_corrected_angles [--resume]
    # smoke (reduced config, reserved seeds, no archive gate):
    ... --smoke --out results/rps2026/Round10_smoke/E11
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import r10_common as rc
from . import r10_provenance as prov
from . import r10_validation as rv
from .angles_v2 import angle_metrics_v2
from .config import PRIMARY, config_hash
from .covariance import assert_checks_pass, run_numerical_checks
from .generator import make_dataset
from .metrics import subspace_angles_to
from .methods import fit_cw_pca, fit_iso_mpca, fit_nfpa

EXPERIMENT = "E11_corrected_angles"
ANALYSIS_LABEL = "post_hoc_diagnostic_correction"
FROZEN_SEEDS = tuple(range(10))
REPLICATION_SEEDS = tuple(range(10, 30))
ALL_SEEDS = FROZEN_SEEDS + REPLICATION_SEEDS
METHODS = ("CW-PCA", "NFPA", "Iso-MPCA")

COLUMNS = (
    "config_hash",
    "seed",
    "cohort",
    "regime",
    "method",
    "analysis_label",
    "archived_source",
    "reproduction_rel_error_raw_mse",
    "reproduction_rel_error_normalized_chi2",
    "test_raw_mse",
    "test_normalized_chi2",
    "angle_to_cw_pca_mean_deg_legacy",
    "angle_to_cw_pca_max_deg_legacy",
    "angle_to_cw_pca_mean_deg_v2",
    "angle_to_cw_pca_max_deg_v2",
    "runtime_seconds",
)

FITTERS = {"CW-PCA": fit_cw_pca, "NFPA": fit_nfpa, "Iso-MPCA": fit_iso_mpca}


def cohort_of(seed: int) -> str:
    return "frozen_0_9" if seed in FROZEN_SEEDS else "replication_10_29"


def load_archived() -> dict[tuple[int, str], dict[str, float]]:
    """Archived held-out metrics: E1 for seeds 0–9, E9 for seeds 10–29."""
    table: dict[tuple[int, str], dict[str, float]] = {}
    for path, source in ((rc.ARCHIVED_E1_CSV, "E1"), (rc.ARCHIVED_E9_CSV, "E9")):
        for row in rc.read_rows_csv(path):
            key = (int(row["seed"]), row["method"])
            table[key] = {
                "test_raw_mse": float(row["test_raw_mse"]),
                "test_normalized_chi2": float(row["test_normalized_chi2"]),
                "source": source,
            }
    return table


def run_seed(seed: int, cfg, archived: dict | None) -> list[dict]:
    t0 = time.time()
    ds = make_dataset(seed, "colored", cfg)
    report = run_numerical_checks(cfg, ds.covariances["Sigma"])
    assert_checks_pass(report)
    ds.covariance_report = report
    cfg_hash = config_hash(cfg)

    from .metrics import normalized_chi2, raw_mse
    from .methods import reconstruct

    fits = {name: FITTERS[name](ds) for name in METHODS}
    cw = fits["CW-PCA"]

    rows: list[dict] = []
    for name in METHODS:
        fit = fits[name]
        Xhat = reconstruct(fit, ds, ds.X_test)
        t_raw = raw_mse(ds.X_test, Xhat)
        t_chi2 = normalized_chi2(ds, ds.X_test, Xhat)

        rel_raw = rel_chi2 = None
        source = None
        if archived is not None:
            record = archived.get((seed, name))
            if record is None:
                raise rc.IntegrityError(f"no archived metrics for seed {seed} method {name}")
            source = record["source"]
            rel_raw = abs(t_raw - record["test_raw_mse"]) / max(abs(record["test_raw_mse"]), 1e-300)
            rel_chi2 = abs(t_chi2 - record["test_normalized_chi2"]) / max(
                abs(record["test_normalized_chi2"]), 1e-300
            )
            if rel_raw > rc.REPRODUCTION_RTOL or rel_chi2 > rc.REPRODUCTION_RTOL:
                raise rc.IntegrityError(
                    f"reproduction gate failed for seed {seed} method {name}: "
                    f"rel_raw={rel_raw:.3e}, rel_chi2={rel_chi2:.3e} > {rc.REPRODUCTION_RTOL}"
                )

        if name == "CW-PCA":
            legacy_mean = legacy_max = v2_mean = v2_max = None  # self-reference
        else:
            legacy = subspace_angles_to(fit, cw, ds)
            v2 = angle_metrics_v2(fit, cw, ds)
            legacy_mean = float(np.mean(legacy))
            legacy_max = float(np.max(legacy))
            v2_mean = v2["angle_to_cw_pca_mean_deg_v2"]
            v2_max = v2["angle_to_cw_pca_max_deg_v2"]

        rows.append(
            {
                "config_hash": cfg_hash,
                "seed": seed,
                "cohort": cohort_of(seed) if archived is not None else "smoke",
                "regime": "colored",
                "method": name,
                "analysis_label": ANALYSIS_LABEL,
                "archived_source": source,
                "reproduction_rel_error_raw_mse": rel_raw,
                "reproduction_rel_error_normalized_chi2": rel_chi2,
                "test_raw_mse": t_raw,
                "test_normalized_chi2": t_chi2,
                "angle_to_cw_pca_mean_deg_legacy": legacy_mean,
                "angle_to_cw_pca_max_deg_legacy": legacy_max,
                "angle_to_cw_pca_mean_deg_v2": v2_mean,
                "angle_to_cw_pca_max_deg_v2": v2_max,
                "runtime_seconds": time.time() - t0,
            }
        )
    return rows


def _method_stats(rows, method: str, column: str) -> dict:
    values = [float(r[column]) for r in rows if r["method"] == method and r[column] not in (None, "")]
    boot = rc.bootstrap_paired(values)  # seed-level percentile bootstrap of the mean
    boot.pop("fraction_negative", None)
    boot["column"] = column
    boot["method"] = method
    return boot


def summarize_cohort(rows, label: str, seeds) -> dict:
    subset = [r for r in rows if int(r["seed"]) in set(seeds)]
    out = {
        "cohort": label,
        "n_seeds": len(seeds),
        "interpretation_note": (
            "Scientific interpretation uses ONLY the *_v2 columns. Legacy columns "
            "reproduce the archived diagnostic and are invalid as subspace angles "
            "for Iso-MPCA (see angles_v2 module docstring)."
        ),
    }
    for method in ("NFPA", "Iso-MPCA"):
        for col in (
            "angle_to_cw_pca_mean_deg_v2",
            "angle_to_cw_pca_max_deg_v2",
            "angle_to_cw_pca_mean_deg_legacy",
        ):
            out[f"{method}.{col}"] = _method_stats(subset, method, col)
    out["contrast_v2_mean_angle_nfpa_minus_iso"] = rc.paired_contrast(
        subset, metric="angle_to_cw_pca_mean_deg_v2", method_a="NFPA", method_b="Iso-MPCA"
    )
    return out


def build_summary(rows) -> dict:
    """Deterministic summary from raw rows alone (regeneration-audit safe)."""
    return {
        "experiment": EXPERIMENT,
        "analysis_label": ANALYSIS_LABEL,
        "frozen": summarize_cohort(rows, "frozen_0_9", FROZEN_SEEDS),
        "replication": summarize_cohort(rows, "replication_10_29", REPLICATION_SEEDS),
        # Pooled is computed only after both subsets above exist.
        "pooled_post_hoc": {
            **summarize_cohort(rows, "post_hoc_pooled_0_29", ALL_SEEDS),
            "pooling_note": "post-hoc pooled cohort; not a 30-seed confirmatory experiment",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(rc.RESULTS_ROOT / EXPERIMENT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    prov.add_freeze_argument(ap)
    args = ap.parse_args(argv)

    smoke = bool(args.smoke)
    freeze_sha, execution_head = prov.validate_freeze_commit(
        args.implementation_freeze_commit, smoke=smoke
    )
    started_at_utc = prov._utc_now()
    source_dirty_at_start = prov.worktree_is_dirty()
    archive_audit_before = None if smoke else rv.require_authoritative_archive_unchanged()
    cfg = rc.smoke_config() if smoke else PRIMARY
    seeds = args.seeds if args.seeds is not None else (list(rc.SMOKE_SEEDS) if smoke else list(ALL_SEEDS))
    seeds = rc.guard_seed_set(seeds, ALL_SEEDS, smoke=smoke)
    out_dir = rc.resolve_out_dir(args.out, smoke=smoke)

    archived = None if smoke else load_archived()

    expected_manifest = {
        "experiment": EXPERIMENT,
        "analysis_label": ANALYSIS_LABEL,
        "config_hash": config_hash(cfg),
        "regime": "colored",
        "methods": sorted(METHODS),
        "columns": list(COLUMNS),
        "seeds": sorted(seeds),
        "smoke": smoke,
    }
    identity = rv.run_identity(freeze_sha, execution_head, config_hash(cfg))
    existing = rc.prepare_output_dir(out_dir, resume=args.resume, expected=expected_manifest)
    # Structural validation is not enough: bind the shards to the run identity
    # and their recorded digests, so a partial directory from another revision
    # cannot be reused and then attributed to this execution commit.
    existing = rv.validate_registered_shards(out_dir, existing, identity)
    rc.write_json(out_dir / "manifest.json", {
        **expected_manifest,
        "status": "in_progress",
        "run_identity": identity,
        "started_at_utc": started_at_utc,
    })

    t_start = time.time()
    for seed in seeds:
        if seed in existing:
            print(f"[E11] seed {seed}: reusing validated shard {existing[seed]}", flush=True)
            continue
        rows = run_seed(seed, cfg, archived)
        shard = rc.shard_path(out_dir, seed)
        rc.write_rows_csv(shard, rows, COLUMNS)
        rv.register_shard(out_dir, seed, shard, identity)
        print(f"[E11] seed {seed}: wrote and registered shard", flush=True)

    shards = sorted(out_dir.glob("seed_*.csv"))
    if len(shards) != len(seeds):
        print(f"[E11] expected {len(seeds)} shards, found {len(shards)}", file=sys.stderr)
        return 1
    all_rows = rc.concat_shards(shards)
    rc.write_rows_csv(out_dir / "raw.csv", all_rows, COLUMNS)

    gate_record: dict = {"validated": False}
    summary_audit: dict = {"passed": None}
    if not smoke:
        # Gates re-derived from the FINAL rows, fresh and resumed alike.
        gate_record = rv.validate_e11_final_rows(
            all_rows, columns=COLUMNS, config_hash=config_hash(cfg),
            analysis_label=ANALYSIS_LABEL, tolerance=rc.REPRODUCTION_RTOL,
        )
        summary = build_summary(all_rows)
        rc.write_json(out_dir / "summary.json", summary)
        summary_audit = rv.assert_summary_regenerates(
            out_dir / "raw.csv", summary, build_summary
        )
        rv.require_authoritative_archive_unchanged()

    # Final atomic write, after every gate, digest and audit above.
    provenance = prov.provenance_block(
        [sys.executable, "-m", "nfpa_confirmatory.run_corrected_angles", *sys.argv[1:]],
        cfg, freeze_sha=freeze_sha, execution_head=execution_head, smoke=smoke,
        started_at_utc=started_at_utc,
        source_tree_dirty_at_start=source_dirty_at_start,
        expected_output_changes_at_completion=[
            str(out_dir.relative_to(rc.RESULTS_ROOT.parent.parent))
            if str(out_dir).startswith(str(rc.RESULTS_ROOT.parent.parent))
            else str(out_dir)
        ],
    )
    provenance["completed_at_utc"] = prov._utc_now()
    manifest = dict(expected_manifest)
    manifest.update(
        {
            "status": "complete",
            "gate_record": gate_record,
            "summary_regeneration_audit": summary_audit,
            "archive_audit_before": archive_audit_before,
            "archive_audit_after": None if smoke else rv.audit_authoritative_archive(),
            "wall_seconds": time.time() - t_start,
            "shard_digests": {p.name: rc.sha256_of(p) for p in shards},
            "raw_digest": rc.sha256_of(out_dir / "raw.csv"),
            "summary_digest": (
                rc.sha256_of(out_dir / "summary.json") if (out_dir / "summary.json").exists() else None
            ),
            "provenance": provenance,
            "run_identity": identity,
            "shard_registry_digest": (
                rc.sha256_of(out_dir / "shard_registry.json")
                if (out_dir / "shard_registry.json").exists() else None
            ),
            "reproduction_gate": {
                "tolerance_rel": rc.REPRODUCTION_RTOL,
                "sources": {"seeds_0_9": "E1_nfpa_blockA/block_A_colored_raw.csv",
                            "seeds_10_29": "E9_blockA_replication/replication_raw.csv"},
                "enforced": not smoke,
            },
        }
    )
    rc.write_json(out_dir / "manifest.json", manifest)
    print(f"[E11] complete: {len(shards)} shards, raw.csv, "
          f"{'summary.json, ' if not smoke else ''}manifest.json in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
