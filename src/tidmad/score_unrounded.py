"""Resolve the rung-4 contrast that the upstream benchmark quantises away.

The problem
-----------
``benchmark.py`` aggregates its per-second SNR products like this
(``calculateBenchmark``, line 139)::

    score = np.round(np.sum(np.multiply(snr_sg, snr_squid))/snr_squid.size,
                     decimals=2) + 1e-10
    return math.log(score, 5.27)

The raw score is **rounded to two decimal places** before the logarithm. Any two
methods whose raw scores differ by less than 0.005 therefore return a bit-
identical result.

On 2026-08-15 both rung-4 arms returned exactly ``-2.353761452599672`` in band
and ``-2.7708098959837675`` over all twenty files -- the latter identical to the
round-1 Savitzky-Golay reproduction. Inverting the logarithm, both arms sit at a
rounded raw score of 0.02 and 0.01 respectively, against FCNet's 53,439. Two
models trained for 4.2 hours each, whose denoised outputs differ by 60 MB,
landed in the same 0.01 bin.

That is a display-resolution artifact, not a null result about the training
metric. Reporting it as "changing the metric made no difference" would be wrong.

What this module does
---------------------
It reuses the release's **own** ``GetOneSecPSD``, ``getSNR`` and
``process_iteration`` verbatim -- the SNR computation is not reimplemented --
and only changes the final aggregation: no rounding, and the raw score reported
alongside the logarithmic one.

The resulting number is **not comparable to the published scoreboard**, which is
already true for this work under the Route C decision (the band-limited
representation was never comparable). It is comparable *between the two arms*,
which is the paired contrast the paper actually claims.

Usage
-----
    PYTHONPATH=src python -m tidmad.score_unrounded \\
        --upstream $TIDMAD_UPSTREAM \\
        --data-dir $TIDMAD_DENOISED_DIR \\
        --model-tag t_mse \\
        --in-band-only $RPS_RESULTS/P1_injection_survey.json \\
        --out $RPS_RESULTS/E5_t_mse_inband_unrounded.json
"""

from __future__ import annotations

import argparse
import importlib.util
import types
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .score import _stage_subset, in_band_files, out_of_band_files

LOG_BASE = 5.27
NOT_COMPARABLE = (
    "Unrounded aggregation of the release's own per-second SNR products. NOT "
    "comparable to the published scoreboard, which applies np.round(..., 2) "
    "before the logarithm. Comparable between arms scored the same way."
)


_SCRIPT_BODY_MARKER = "parser = argparse.ArgumentParser"


def _load_upstream(upstream: Path):
    """Load the release's SNR functions **verbatim**, without running its script.

    ``benchmark.py`` is a script, not a library: its argument parsing and the
    scoring call sit at module level, so a plain import would execute a full
    benchmark run. We therefore exec only the source above
    ``parser = argparse.ArgumentParser`` -- the imports and function definitions
    -- and nothing below it.

    The functions obtained this way are byte-for-byte the release's own. Nothing
    in the upstream checkout is modified or copied into this repository.
    """
    script = Path(upstream) / "benchmark.py"
    if not script.exists():
        raise SystemExit(f"benchmark.py not found under {upstream}")

    source = script.read_text()
    if _SCRIPT_BODY_MARKER not in source:
        raise SystemExit(
            f"{script}: expected to find {_SCRIPT_BODY_MARKER!r} separating the "
            "function definitions from the script body; the release layout has "
            "changed and this loader must be re-checked"
        )
    definitions = source.split(_SCRIPT_BODY_MARKER)[0]

    module = types.ModuleType("_tidmad_benchmark_defs")
    module.__file__ = str(script)
    exec(compile(definitions, str(script), "exec"), module.__dict__)
    for required in ("GetOneSecPSD", "getSNR", "process_iteration"):
        if not hasattr(module, required):
            raise SystemExit(f"{script}: {required} not found among the definitions")
    return module


def score_unrounded(upstream: Path, data_dir: Path, model_tag: str, *, coarse: bool = True) -> dict:
    """Recompute the coarse score without the two-decimal rounding."""
    bench = _load_upstream(upstream)

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith(f"abra_validation_denoised_{model_tag}_") and f.endswith(".h5")
    )
    if not files:
        raise SystemExit(f"no denoised files for tag {model_tag!r} under {data_dir}")

    n = 200 * len(files)
    if coarse:
        n = int(n / 10)

    snr_sg = np.zeros(n)
    snr_squid = np.zeros(n)
    for i in range(n):
        _, sg, squid = bench.process_iteration(i, str(data_dir), files, coarse)
        snr_sg[i], snr_squid[i] = sg, squid

    snr_sg_norm = snr_sg / np.amax(snr_sg)
    raw = float(np.sum(np.multiply(snr_sg_norm, snr_squid)) / snr_squid.size)

    rounded = float(np.round(raw, decimals=2) + 1e-10)

    # Per-window products. The score is their mean, so these are the paired
    # samples an interval must be built from. snr_sg is taken from the
    # reference channel, which every method copies through unchanged, so it is
    # identical across arms and the contrast is carried entirely by snr_squid.
    per_window = np.multiply(snr_sg_norm, snr_squid)

    return {
        "per_window_products": per_window.tolist(),
        "snr_sg_normalised": snr_sg_norm.tolist(),
        "snr_squid": snr_squid.tolist(),
        "model_tag": model_tag,
        "n_files": len(files),
        "n_seconds_scored": n,
        "raw_score_unrounded": raw,
        "log_score_unrounded": math.log(raw, LOG_BASE) if raw > 0 else float("-inf"),
        "raw_score_rounded": rounded,
        "log_score_upstream_equivalent": math.log(rounded, LOG_BASE),
        "comparability_note": NOT_COMPARABLE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--coarse", action="store_true", default=True)
    ap.add_argument("--in-band-only", type=Path, default=None, metavar="SURVEY_JSON")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.in_band_only is None:
        record = score_unrounded(args.upstream, args.data_dir, args.model_tag, coarse=args.coarse)
        record["file_set"] = "all_20"
    else:
        keep = in_band_files(args.in_band_only)
        dropped = out_of_band_files(args.in_band_only)
        with tempfile.TemporaryDirectory(prefix=f"tidmad_unrounded_{args.model_tag}_") as tmp:
            scratch = Path(tmp)
            staged = _stage_subset(Path(args.data_dir), args.model_tag, keep, scratch)
            print(f"scoring {staged} in-band files (dropped {len(dropped)})")
            record = score_unrounded(args.upstream, scratch, args.model_tag, coarse=args.coarse)
        record["file_set"] = "in_band_subset"
        record["scored_files"] = keep
        record["excluded_files"] = dropped

    print(json.dumps(record, indent=2, sort_keys=True))
    print(
        f"\n  raw (unrounded) = {record['raw_score_unrounded']:.8g}"
        f"   |   upstream would report {record['log_score_upstream_equivalent']:.6f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"  record written to {args.out}")


if __name__ == "__main__":
    main()
