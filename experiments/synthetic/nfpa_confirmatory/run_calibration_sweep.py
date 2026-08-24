"""E12: how the rung-3 reversal degrades as calibration data shrinks.

E10b answered "does the reversal survive an *estimated* covariance?" at one
calibration size. A referee's next question is "only one?". This runner answers
it with a curve over ``N_cal in {256, 512, 1024, 2048, 8192}``, holding every
other frozen setting fixed.

It is **post-hoc exploratory**. It was designed after the confirmatory result
existed, so it can support the estimated-covariance limitation with more
evidence; it cannot delete that limitation, and §5 must continue to say that no
confirmatory robustness boundary was established. No threshold, no predicted
direction, no success criterion: an unfavourable curve is a result to report.

The design is E10b's, once per condition:

* the dataset, the oracle evaluation covariance and the held-out test set are
  unchanged — only the covariance *used for fitting* is estimated;
* Iso-MPCA never sees the estimated covariance and is the control;
* ``N_cal = 2048`` is the **anchor** and must reproduce archived E10b to
  ``1e-9`` relative, which is what ties this sweep to archived evidence.

Each condition draws its own calibration noise from ``calibration_rng(seed)``,
exactly as E10b does, rather than slicing a prefix of a larger draw. That is
what makes the anchor bit-comparable with the archived run instead of merely
similar, and it costs nothing: conditions need not be nested to be comparable.

Usage
-----
    PYTHONPATH=src:experiments/synthetic python -m nfpa_confirmatory.run_calibration_sweep \
        --out results/rps2026/E12_calibration_size_sweep \
        --implementation-freeze-commit <SHA> [--resume]
    # one condition only:  ... --sizes 256
    # smoke:               ... --smoke --out results/rps2026/Round10_smoke/E12
"""

from __future__ import annotations

import argparse
import sys
import time

from . import r10_common as rc
from . import r10_provenance as prov
from . import r10_validation as rv
from . import r12_sweep as sw
from .config import config_hash
from .run_calibrated_estimated_covariance import COLUMNS, load_archived, run_seed

EXPERIMENT = sw.EXPERIMENT
ANALYSIS_LABEL = sw.ANALYSIS_LABEL


def run_condition(
    n_cal: int,
    *,
    out_root,
    seeds: list[int],
    smoke: bool,
    resume: bool,
    archived,
    freeze_sha,
    execution_head,
    started_at_utc: str,
    source_dirty_at_start: bool,
) -> list[dict]:
    """Run one calibration size into its own directory, with E10b's gates."""
    cfg = sw.condition_config(n_cal)
    if smoke:
        # Keep the reduced fixture's cheap dimensions but honour the swept field,
        # so a smoke run still exercises the per-condition code path.
        from dataclasses import replace

        cfg = replace(rc.smoke_config(), N_calibration=int(n_cal))
    label = sw.condition_label(n_cal)
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_manifest = {
        "experiment": EXPERIMENT,
        "condition": label,
        "analysis_label": ANALYSIS_LABEL,
        "evidence_class": sw.EVIDENCE_CLASS,
        "config_hash": config_hash(cfg),
        "regime": "colored",
        "methods": ["Iso-MPCA", "NFPA"],
        "columns": list(COLUMNS),
        "seeds": sorted(seeds),
        "smoke": smoke,
        "calibration": {
            "n_calibration_events": cfg.N_calibration,
            "rng": "calibration_rng(seed), drawn independently per condition",
            "regularization": "unchanged from E10/E10b",
            "scale_correction": "v_hat = sqrt(v_c*v_t) (covariance_estimation_v2)",
        },
        "is_anchor": int(n_cal) == sw.ANCHOR_N_CAL,
    }
    identity = rv.run_identity(freeze_sha, execution_head, config_hash(cfg))
    existing = rc.prepare_output_dir(out_dir, resume=resume, expected=expected_manifest)
    existing = rv.validate_registered_shards(out_dir, existing, identity)
    rc.write_json(
        out_dir / "manifest.json",
        {**expected_manifest, "status": "in_progress", "run_identity": identity,
         "started_at_utc": started_at_utc},
    )

    t_start = time.time()
    for seed in seeds:
        if seed in existing:
            print(f"[E12 {label}] seed {seed}: reusing validated shard", flush=True)
            continue
        rows, control_ok = run_seed(seed, cfg, archived, analysis_label=ANALYSIS_LABEL)
        if not smoke and not control_ok:
            raise rc.IntegrityError(
                f"{EXPERIMENT} {label}: Iso-MPCA control failed archived reproduction "
                f"at seed {seed}; stopping before writing the shard"
            )
        shard = rc.shard_path(out_dir, seed)
        rc.write_rows_csv(shard, rows, COLUMNS)
        rv.register_shard(out_dir, seed, shard, identity)
        print(f"[E12 {label}] seed {seed}: wrote and registered shard", flush=True)

    shards = sorted(out_dir.glob("seed_*.csv"))
    if len(shards) != len(seeds):
        raise rc.IntegrityError(
            f"{EXPERIMENT} {label}: expected {len(seeds)} shards, found {len(shards)}"
        )
    rows = rc.concat_shards(shards)
    rc.write_rows_csv(out_dir / "raw.csv", rows, COLUMNS)

    gate_record: dict = {"validated": False}
    anchor_record: dict = {"applicable": False}
    if not smoke:
        gate_record = sw.validate_condition_rows(
            rows, n_cal=n_cal, columns=COLUMNS, seeds=seeds
        )
        if int(n_cal) == sw.ANCHOR_N_CAL:
            anchor_record = sw.validate_anchor_reproduces_e10b(
                [r for r in rows if r["method"] == "NFPA"]
            )

    provenance = prov.provenance_block(
        [sys.executable, "-m", "nfpa_confirmatory.run_calibration_sweep", *sys.argv[1:]],
        cfg, freeze_sha=freeze_sha, execution_head=execution_head, smoke=smoke,
        started_at_utc=started_at_utc,
        source_tree_dirty_at_start=source_dirty_at_start,
        expected_output_changes_at_completion=[str(out_dir)],
    )
    provenance["completed_at_utc"] = prov._utc_now()
    provenance["evidence_class"] = sw.EVIDENCE_CLASS
    rc.write_json(
        out_dir / "manifest.json",
        {
            **expected_manifest,
            "status": "complete",
            "gate_record": gate_record,
            "anchor_reproduction": anchor_record,
            "wall_seconds": time.time() - t_start,
            "shard_digests": {p.name: rc.sha256_of(p) for p in shards},
            "raw_digest": rc.sha256_of(out_dir / "raw.csv"),
            "provenance": provenance,
            "run_identity": identity,
            "shard_registry_digest": (
                rc.sha256_of(out_dir / "shard_registry.json")
                if (out_dir / "shard_registry.json").exists() else None
            ),
        },
    )
    print(f"[E12 {label}] complete: {len(shards)} shards", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(rc.RESULTS_ROOT / EXPERIMENT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument(
        "--sizes", type=int, nargs="*", default=None,
        help="calibration sizes to run; defaults to the frozen sweep. A partial "
             "run is permitted, but the sweep summary is only written when every "
             "frozen size is present.",
    )
    prov.add_freeze_argument(ap)
    args = ap.parse_args(argv)

    smoke = bool(args.smoke)
    # Fail closed BEFORE any scientific computation, in this order:
    # provenance, then configuration identity, then the archive.
    freeze_sha, execution_head = prov.validate_freeze_commit(
        args.implementation_freeze_commit, smoke=smoke
    )
    started_at_utc = prov._utc_now()
    source_dirty_at_start = prov.worktree_is_dirty()
    if not smoke:
        sw.assert_anchor_hash_matches_archive()
    archive_audit_before = None if smoke else rv.require_authoritative_archive_unchanged()

    if args.sizes is None:
        sizes = sw.validate_condition_set(sw.CALIBRATION_SIZES)
    else:
        # A partial run is allowed for staging and restarts; the curve is only
        # written when every frozen size is present, so a subset can never be
        # mistaken for the sweep.
        sizes = sorted({int(n) for n in args.sizes})
    seeds = args.seeds if args.seeds is not None else (
        list(rc.SMOKE_SEEDS) if smoke else list(sw.SEEDS)
    )
    seeds = rc.guard_seed_set(seeds, sw.SEEDS, smoke=smoke)
    out_root = rc.resolve_out_dir(args.out, smoke=smoke)
    sw.refuse_archived_round10_dirs(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    archived = None if smoke else load_archived()

    print(
        f"[E12] {ANALYSIS_LABEL}: sizes={sizes} seeds={seeds} smoke={smoke}",
        flush=True,
    )
    t_start = time.time()
    by_condition: dict[int, list[dict]] = {}
    for n_cal in sizes:
        by_condition[n_cal] = run_condition(
            n_cal,
            out_root=out_root, seeds=seeds, smoke=smoke, resume=args.resume,
            archived=archived, freeze_sha=freeze_sha, execution_head=execution_head,
            started_at_utc=started_at_utc, source_dirty_at_start=source_dirty_at_start,
        )

    complete = sorted(sizes) == sorted(sw.CALIBRATION_SIZES)
    control_record: dict = {"applicable": False}
    summary_audit: dict = {"passed": None}
    if not smoke and complete:
        # A cross-condition gate: the control must not move with N_cal.
        control_record = sw.validate_control_is_covariance_independent(by_condition)
        summary = sw.write_sweep_summary(out_root, by_condition)
        summary_audit = sw.assert_sweep_summary_regenerates(out_root, sizes, summary)
        rv.require_authoritative_archive_unchanged()
    elif not complete:
        print(
            "[E12] partial sweep: sweep_summary.json NOT written. Rerun with every "
            f"frozen size {sw.CALIBRATION_SIZES} to produce the curve.",
            file=sys.stderr,
        )

    rc.write_json(
        out_root / "sweep_manifest.json",
        {
            "experiment": EXPERIMENT,
            "analysis_label": ANALYSIS_LABEL,
            "evidence_class": sw.EVIDENCE_CLASS,
            "status": "complete" if (complete and not smoke) else "partial_or_smoke",
            "calibration_sizes_run": sizes,
            "calibration_sizes_frozen": list(sw.CALIBRATION_SIZES),
            "seeds": sorted(seeds),
            "smoke": smoke,
            "anchor_n_cal": sw.ANCHOR_N_CAL,
            "anchor_config_hash": sw.ANCHOR_CONFIG_HASH,
            "control_gate": control_record,
            "summary_audit": summary_audit,
            "archive_audit_before": archive_audit_before,
            "archive_audit_after": None if smoke else rv.audit_authoritative_archive(),
            "wall_seconds": time.time() - t_start,
            "started_at_utc": started_at_utc,
            "completed_at_utc": prov._utc_now(),
            "source_tree_dirty_at_start": source_dirty_at_start,
        },
    )
    print(f"[E12] done: {len(sizes)} conditions in {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
