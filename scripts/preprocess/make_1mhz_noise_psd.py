#!/usr/bin/env python3
"""Create a target-rate empirical PSD file from an existing one-sided PSD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
QP_DIR = REPO_ROOT / "QP_simulator"
if str(QP_DIR) not in sys.path:
    sys.path.insert(0, str(QP_DIR))

from noise_module.psd_resampling import (  # noqa: E402
    load_psd_density,
    make_target_psd_density,
    save_psd_density,
)


def infer_source_sampling_frequency(source_max_frequency_hz: float) -> float:
    """Infer source fs from the PSD Nyquist, smoothing tiny stored-grid drift."""
    nyquist = float(source_max_frequency_hz)
    rounded_nyquist = float(round(nyquist))
    if abs(nyquist - rounded_nyquist) <= 1.0:
        nyquist = rounded_nyquist
    return 2.0 * nyquist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an empirical one-sided PSD density to a target sampling "
            "frequency/trace length and save a two-row .npy PSD file."
        )
    )
    parser.add_argument("--input", required=True, help="Input PSD .npy file.")
    parser.add_argument("--output", required=True, help="Output PSD .npy file.")
    parser.add_argument(
        "--target-sampling-frequency",
        type=float,
        default=1_000_000.0,
        help="Target sampling frequency in Hz. Default: 1 MHz.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        required=True,
        help="Target trace length in samples.",
    )
    parser.add_argument(
        "--method",
        choices=["inband", "alias_fold", "synthetic_resample"],
        default="inband",
        help=(
            "inband assumes anti-alias removal above target Nyquist; alias_fold "
            "folds available high-frequency PSD into band; synthetic_resample "
            "generates old-rate Gaussian noise, resamples it, and re-estimates PSD."
        ),
    )
    parser.add_argument(
        "--source-sampling-frequency",
        type=float,
        default=None,
        help=(
            "Source sampling frequency in Hz. Required for synthetic_resample. "
            "If omitted there, it is inferred as 2 * max(input_frequency)."
        ),
    )
    parser.add_argument(
        "--anti-alias-cutoff-hz",
        type=float,
        default=None,
        help="Optional low-pass power-response passband edge in Hz.",
    )
    parser.add_argument(
        "--anti-alias-transition-hz",
        type=float,
        default=0.0,
        help="Optional low-pass transition width in Hz.",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=64,
        help="Synthetic traces used by synthetic_resample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used by synthetic_resample.",
    )
    parser.add_argument(
        "--welch-average",
        choices=["mean", "median"],
        default="mean",
        help="Welch average used by synthetic_resample.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional metadata JSON path. Defaults to output path with .json suffix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_f, source_p = load_psd_density(args.input)
    source_sampling_frequency = args.source_sampling_frequency
    if args.method == "synthetic_resample" and source_sampling_frequency is None:
        source_sampling_frequency = infer_source_sampling_frequency(source_f[-1])

    kwargs = {}
    if args.method in {"inband", "alias_fold"}:
        kwargs["anti_alias_cutoff_hz"] = args.anti_alias_cutoff_hz
        kwargs["anti_alias_transition_hz"] = args.anti_alias_transition_hz
    elif args.method == "synthetic_resample":
        kwargs["n_traces"] = args.n_traces
        kwargs["seed"] = args.seed
        kwargs["average"] = args.welch_average

    target_f, target_p, metadata = make_target_psd_density(
        source_f,
        source_p,
        args.target_sampling_frequency,
        args.target_samples,
        method=args.method,
        source_sampling_frequency=source_sampling_frequency,
        **kwargs,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_psd_density(output, target_f, target_p)

    metadata.update(
        {
            "input": str(Path(args.input).resolve()),
            "output": str(output.resolve()),
            "output_bins": int(target_f.size),
        }
    )
    metadata_output = (
        Path(args.metadata_output)
        if args.metadata_output is not None
        else output.with_suffix(".json")
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    print(f"wrote {metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
