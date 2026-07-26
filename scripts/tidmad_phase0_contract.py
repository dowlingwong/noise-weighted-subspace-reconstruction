"""TIDMAD Phase 0: establish and record the data contract.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`` Phase 0, gate G0.

This script does not analyse anything. It *inspects* the downloaded files and
writes down what is actually in them -- dataset names and shapes, dtype,
per-file hashes, and (for training/validation files) which channel carries the
SQUID readout and which carries the injected reference. Everything downstream
reads that record instead of assuming.

The reason for the ceremony: a wrong channel-semantics assumption would not
crash. It would silently produce a plausible-looking number in Phase 2 with the
roles of data and truth exchanged. The contract exists so that assumption is
made once, in writing, against the files.

Channel identification is by *evidence*, not by name. The reference channel is
the one whose spectrum is dominated by a few narrow lines (the injected
sinusoids) with a much lower broadband floor; the SQUID channel carries the
detector noise. The script reports the statistics it used so the decision can be
audited, and refuses to guess when the evidence is ambiguous.

    uv run python scripts/tidmad_phase0_contract.py \
        --data-root /ceph/dwong/paper1_dataset \
        --output docs/tidmad_data_contract.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Fraction of spectral power concentrated in the strongest bins above which a
#: channel is considered line-dominated (i.e. the injected reference).
LINE_DOMINANCE_THRESHOLD = 0.5
#: A channel is considered broadband (i.e. the SQUID readout) below this.
BROADBAND_THRESHOLD = 0.2


def _file_digest(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    """SHA-256 of the first ``max_bytes`` of a file.

    Full hashes of multi-gigabyte files are impractical to recompute on every
    run; a prefix hash still detects truncation, corruption, and accidental
    substitution, which is what this record needs to do.
    """
    h = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = max_bytes
        while remaining > 0:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _line_dominance(window: np.ndarray, top_bins: int = 8) -> float:
    """Fraction of spectral power carried by the strongest ``top_bins`` bins."""
    spectrum = np.abs(np.fft.rfft(window - window.mean())) ** 2
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    strongest = np.sort(spectrum)[-int(top_bins) :]
    return float(strongest.sum() / total)


def inspect_file(path: Path, window_samples: int, n_windows: int) -> dict:
    """Return the structural and statistical facts about one file."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("h5py is required for Phase 0 inspection") from exc

    record: dict = {
        "file": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256_prefix_64mib": _file_digest(path),
        "datasets": {},
    }
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            item = handle[key]
            shape = getattr(item, "shape", None)
            if shape is None or len(shape) != 1:
                record["datasets"][key] = {"shape": None if shape is None else list(shape)}
                continue
            probe = np.asarray(item[: int(window_samples) * int(n_windows)], dtype=np.float64)
            probe = probe[: (probe.size // window_samples) * window_samples]
            windows = probe.reshape(-1, int(window_samples)) if probe.size else np.empty((0, 0))
            dominance = (
                float(np.median([_line_dominance(w) for w in windows])) if windows.size else float("nan")
            )
            record["datasets"][key] = {
                "shape": [int(s) for s in shape],
                "dtype": str(item.dtype),
                "n_samples": int(shape[0]),
                "probe_std": float(np.std(probe)) if probe.size else float("nan"),
                "probe_mean": float(np.mean(probe)) if probe.size else float("nan"),
                "median_line_dominance": dominance,
            }
    return record


def classify_channels(record: dict) -> dict:
    """Decide which dataset is the SQUID readout and which is the reference."""
    candidates = {
        name: info
        for name, info in record["datasets"].items()
        if info.get("median_line_dominance") is not None
        and np.isfinite(info.get("median_line_dominance", np.nan))
    }
    if not candidates:
        return {"squid_dataset": None, "reference_dataset": None, "reason": "no 1-D datasets found"}

    if len(candidates) == 1:
        only = next(iter(candidates))
        return {
            "squid_dataset": only,
            "reference_dataset": None,
            "reason": "single channel present (science file): treated as the SQUID readout",
        }

    ranked = sorted(candidates.items(), key=lambda kv: kv[1]["median_line_dominance"])
    broadband_name, broadband = ranked[0]
    line_name, line = ranked[-1]

    if (
        line["median_line_dominance"] >= LINE_DOMINANCE_THRESHOLD
        and broadband["median_line_dominance"] <= BROADBAND_THRESHOLD
    ):
        return {
            "squid_dataset": broadband_name,
            "reference_dataset": line_name,
            "reason": (
                f"{line_name} is line-dominated "
                f"({line['median_line_dominance']:.3f} of power in the strongest bins) "
                f"while {broadband_name} is broadband "
                f"({broadband['median_line_dominance']:.3f})"
            ),
        }

    return {
        "squid_dataset": None,
        "reference_dataset": None,
        "reason": (
            "channel roles are ambiguous from the spectra "
            f"({ {k: round(v['median_line_dominance'], 3) for k, v in candidates.items()} }); "
            "resolve by hand against the release documentation and record the decision"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/ceph/dwong/paper1_dataset"))
    parser.add_argument("--validation-glob", type=str, default="tidmad/raw/abra_validation_*.h5")
    parser.add_argument("--science-glob", type=str, default="tidmad/raw/abra_science_*.h5")
    parser.add_argument("--window-samples", type=int, default=1 << 16)
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--sampling-frequency", type=float, default=10_000_000.0)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "tidmad_data_contract.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "metrics" / "TIDMAD_P0_inspection.json")
    args = parser.parse_args()

    validation = sorted(Path(args.data_root).glob(args.validation_glob))[: args.max_files]
    science = sorted(Path(args.data_root).glob(args.science_glob))[: args.max_files]
    if not validation and not science:
        raise SystemExit(
            f"no TIDMAD files found under {args.data_root}. Download a subset first:\n"
            "  python download_data.py -t 0 -v 2 -s 2 -o <data-root>/tidmad/raw"
        )

    inspection = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": str(args.data_root),
        "validation_files": [
            inspect_file(p, args.window_samples, args.n_windows) for p in validation
        ],
        "science_files": [inspect_file(p, args.window_samples, args.n_windows) for p in science],
    }
    for group in ("validation_files", "science_files"):
        for record in inspection[group]:
            record["channel_classification"] = classify_channels(record)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"inspection report -> {args.report}")

    reference_record = inspection["validation_files"][0] if validation else inspection["science_files"][0]
    classification = reference_record["channel_classification"]
    print(f"channel classification: {classification['reason']}")

    if classification["squid_dataset"] is None:
        print(
            "\nContract NOT written: channel roles could not be established from the data.\n"
            "Resolve by hand, then write docs/tidmad_data_contract.json with verified=true.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    squid_info = reference_record["datasets"][classification["squid_dataset"]]
    contract = {
        "squid_dataset": classification["squid_dataset"],
        "reference_dataset": classification["reference_dataset"],
        "sampling_frequency_hz": float(args.sampling_frequency),
        "dtype": squid_info.get("dtype", "unknown"),
        "max_samples": 2_000_000_000,
        "injection_frequencies_hz": [],
        "injection_amplitudes": [],
        "verified": False,
        "notes": (
            "Generated by scripts/tidmad_phase0_contract.py. Set verified=true only "
            "after (1) confirming the sampling frequency against the release, "
            "(2) filling in the injection frequencies and amplitudes, and "
            "(3) reproducing the published `none` and `savgol` denoising scores (gate G0). "
            f"Channel decision: {classification['reason']}"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"draft contract -> {args.output} (verified=false until G0 passes)")


if __name__ == "__main__":
    main()
