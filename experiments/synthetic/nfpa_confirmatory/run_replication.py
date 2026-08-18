"""E9 - isolated 20-seed post-hoc replication of colored Block A.

Runs seeds 10-29 under the unchanged frozen ``PRIMARY`` configuration and the
archived Block A row schema, into a dedicated directory. This runner can never
resolve to the archived Block A directory, and it never appends to
``block_A_colored_raw.csv``.

Labelling, which the paper must preserve:

* seeds 0-9   -> ``frozen_confirmatory`` (the frozen 10-seed amendment)
* seeds 10-29 -> ``post_hoc_replication`` (this run)
* seeds 0-29  -> ``post_hoc_pooled`` (only if the direction-agreement rule allows)

The replication summary is always produced first. The two *primary* contrasts
are held-out normalized chi-square and held-out raw MSE; the mean principal
angle to CW-PCA is a secondary geometric diagnostic. If either primary contrast
reverses sign against seeds 0-9, ``pooling_permitted`` is false, no pooled
estimate is written, and both subsets are reported side by side.

Usage::

    python -m experiments.synthetic.nfpa_confirmatory.run_replication \
        --seeds 10 ... 29 --out-dir results/rps2026/E9_blockA_replication [--resume]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from . import r9_common as rc
from .config import PRIMARY, config_hash
from .covariance import assert_checks_pass, run_numerical_checks
from .generator import make_dataset
from .methods import FIT_FACTORY, fit_cw_pca
from .output import COLUMNS, build_row

#: Seeds 10-29 inclusive: the registered replication extension.
REPLICATION_SEEDS = list(range(10, 30))

#: The two primary metric-reversal contrasts, then the secondary diagnostic.
PRIMARY_CONTRASTS = ("test_normalized_chi2", "test_raw_mse")
SECONDARY_CONTRAST = "angle_to_cw_pca_mean_deg"


def run_seed_rows(seed: int, cfg, regime: str = "colored") -> list[dict]:
    ds = make_dataset(seed, regime, cfg)
    report = run_numerical_checks(cfg, ds.covariances["Sigma"])
    assert_checks_pass(report)
    ds.covariance_report = report
    cw_pca = fit_cw_pca(ds)
    rows = []
    cfg_hash = config_hash(cfg)
    for name in rc.REGISTERED_METHODS:
        t0 = time.time()
        fit = cw_pca if name == "CW-PCA" else FIT_FACTORY[name](ds)
        row = build_row(ds, fit, cw_pca, runtime=time.time() - t0)
        # ``build_row`` stamps the module-level PRIMARY hash; under ``--smoke``
        # the reduced fixture has its own hash and must not borrow PRIMARY's.
        row["config_hash"] = cfg_hash
        rows.append(row)
    return rows


def contrast_block(rows, metric: str) -> dict:
    return rc.paired_contrast(rows, metric=metric, method_a="NFPA", method_b="Iso-MPCA")


def summarize_subset(rows, label: str, seeds) -> dict:
    out = {
        "label": label,
        "seeds": [int(s) for s in seeds],
        "n_seeds": len(seeds),
        "primary_contrasts": {m: contrast_block(rows, m) for m in PRIMARY_CONTRASTS},
        "secondary_contrast": {SECONDARY_CONTRAST: contrast_block(rows, SECONDARY_CONTRAST)},
    }
    for block in list(out["primary_contrasts"].values()) + list(out["secondary_contrast"].values()):
        block["inconclusive"] = not block.get("excludes_zero", False)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="E9 isolated 20-seed replication")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    cfg = rc.smoke_config() if args.smoke else PRIMARY
    expected_seeds = list(rc.SMOKE_SEEDS) if args.smoke else REPLICATION_SEEDS
    try:
        seeds = rc.guard_seed_set(args.seeds or expected_seeds, expected_seeds, smoke=args.smoke)
        out_dir = rc.resolve_out_dir(args.out_dir, smoke=args.smoke)
    except rc.IntegrityError as exc:
        print(f"[E9] INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    prov = rc.provenance(["run_replication", *argv], cfg)

    print(f"[E9] resolved output directory: {out_dir}", flush=True)
    print(f"[E9] config_hash={config_hash(cfg)} seeds={seeds}", flush=True)

    expected_manifest = {
        "experiment": "E9_blockA_replication",
        "config_hash": config_hash(cfg),
        "regime": "colored",
        "methods": list(rc.REGISTERED_METHODS),
        "seeds": seeds,
        "columns": list(COLUMNS),
    }
    try:
        existing = rc.prepare_output_dir(out_dir, resume=args.resume, expected=expected_manifest)
    except rc.IntegrityError as exc:
        print(f"[E9] INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 2

    # Written before the first fit so a killed or capped run stays resumable.
    rc.write_json(out_dir / "manifest.json", {**expected_manifest, "status": "in_progress"})

    for seed in seeds:
        if seed in existing:
            print(f"[E9] seed {seed}: reusing validated shard {existing[seed]}", flush=True)
            continue
        t0 = time.time()
        rows = run_seed_rows(seed, cfg)
        rc.write_rows_csv(rc.shard_path(out_dir, seed), rows, COLUMNS)
        print(f"[E9] seed {seed}: wrote shard in {time.time() - t0:.1f}s", flush=True)

    shards = sorted(out_dir.glob("seed_*.csv"))
    if len(shards) != len(seeds):
        print(f"[E9] expected {len(seeds)} shards, found {len(shards)}", file=sys.stderr)
        return 5
    replication_rows = rc.concat_shards(shards)
    rc.write_rows_csv(out_dir / "replication_raw.csv", replication_rows, COLUMNS)

    replication = summarize_subset(replication_rows, "post_hoc_replication", seeds)

    # ---- the original frozen subset, read from the archive, never rewritten ----
    original = None
    pooling_permitted = False
    direction = {}
    if not args.smoke:
        original_rows = rc.read_rows_csv(rc.ARCHIVED_E1_CSV)
        original_seeds = sorted({int(r["seed"]) for r in original_rows})
        original = summarize_subset(original_rows, "frozen_confirmatory", original_seeds)
        agree = True
        for metric in PRIMARY_CONTRASTS:
            o = original["primary_contrasts"][metric]["mean_difference"]
            r = replication["primary_contrasts"][metric]["mean_difference"]
            same = bool(np.sign(o) == np.sign(r))
            direction[metric] = {
                "original_mean_difference": o,
                "replication_mean_difference": r,
                "sign_agrees": same,
            }
            agree = agree and same
        pooling_permitted = agree

    summary = {
        "experiment": "E9_blockA_replication",
        "config_hash": config_hash(cfg),
        "sign_convention": (
            "All contrasts are NFPA minus Iso-MPCA on held-out test metrics. "
            "Negative favours NFPA for the two error metrics."
        ),
        "primary_contrasts_are": list(PRIMARY_CONTRASTS),
        "secondary_contrast_is": SECONDARY_CONTRAST,
        "secondary_contrast_caveat": (
            "angle_to_cw_pca_* is computed by methods.basis_whitened, whose raw-plus-factored "
            "branch -- reached only by Iso-MPCA -- unstacks a (C*T, rank) basis with "
            "B.reshape(rank, C, T, order='F') instead of B.T.reshape(...). That has the right "
            "shape but interleaves the basis index with the channel index, so Iso-MPCA's angle "
            "columns are computed from a scrambled basis. Round 9 discovered this while "
            "implementing E8 and deliberately did NOT change the function, because doing so "
            "would silently redefine an archived Block A column and break comparability between "
            "these replication rows and seeds 0-9. The two PRIMARY contrasts "
            "(test_normalized_chi2, test_raw_mse) do not use this function and are unaffected. "
            "The desk should decide whether to recompute the angle diagnostic for both subsets "
            "with the corrected convention; see r9_semantics.fitted_basis_whitened."
        ),
        "replication": replication,
        "original_frozen_confirmatory": original,
        "direction_agreement": direction,
        "pooling_permitted": pooling_permitted,
        "pooling_rule": (
            "Pooling is permitted only when both primary contrasts agree in sign between "
            "seeds 0-9 and seeds 10-29. A confidence interval containing zero is reported "
            "as inconclusive even when its mean sign agrees."
        ),
        "bootstrap": {"resamples": rc.BOOTSTRAP_RESAMPLES, "seed": rc.BOOTSTRAP_SEED, "unit": "seed"},
    }
    rc.write_json(out_dir / "replication_summary.json", summary)
    print(f"[E9] replication summary written; pooling_permitted={pooling_permitted}", flush=True)

    pooled_path = out_dir / "pooled_summary.json"
    if pooling_permitted:
        pooled_rows = list(rc.read_rows_csv(rc.ARCHIVED_E1_CSV)) + replication_rows
        pooled_seeds = sorted({int(r["seed"]) for r in pooled_rows})
        pooled = summarize_subset(pooled_rows, "post_hoc_pooled", pooled_seeds)
        rc.write_json(
            pooled_path,
            {
                "experiment": "E9_blockA_replication",
                "label_warning": (
                    "This is a post-hoc pooled aggregate of a frozen 10-seed confirmatory "
                    "experiment and a 20-seed post-hoc replication. It is NOT a 30-seed "
                    "confirmatory experiment."
                ),
                "frozen_confirmatory": original,
                "post_hoc_replication": replication,
                "post_hoc_pooled": pooled,
            },
        )
        print("[E9] pooled summary written (directions agree)", flush=True)
    else:
        print("[E9] pooled summary omitted: directions disagree or smoke run", flush=True)

    ended = datetime.now(timezone.utc)
    manifest = dict(expected_manifest)
    manifest.update(
        {
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "wall_time_seconds": (ended - started).total_seconds(),
            "pooling_permitted": pooling_permitted,
            "shard_digests": {p.name: rc.sha256_of(p) for p in shards},
            "replication_raw_digest": rc.sha256_of(out_dir / "replication_raw.csv"),
            "replication_summary_digest": rc.sha256_of(out_dir / "replication_summary.json"),
            "pooled_summary_digest": rc.sha256_of(pooled_path) if pooled_path.exists() else None,
            "archived_e1_digest": rc.sha256_of(rc.ARCHIVED_E1_CSV) if not args.smoke else None,
            "provenance": prov,
        }
    )
    rc.write_json(out_dir / "manifest.json", manifest)
    print(f"[E9] complete: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
