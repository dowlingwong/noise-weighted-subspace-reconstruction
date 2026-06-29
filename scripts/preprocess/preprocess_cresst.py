"""Build a compact CRESST development cache from the public release.

Loads ``X_{split}.npy`` + ``features_{split}.csv``, separates noise-only
baselines from accepted pulses using the ``noise``/``clean`` flags, optionally
subsamples, and writes a small NPZ under ``processed/`` so downstream
development iterates without re-reading the multi-GB release each time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.cresst import load_cresst_release, select_cresst_subsets  # noqa: E402
from noise_geometry.utils import load_config, resolve_data_root  # noqa: E402
from noise_geometry.utils.paths import ensure_dataset_layout  # noqa: E402


def _cap(traces: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= traces.shape[0]:
        return traces
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(traces.shape[0], size=int(limit), replace=False))
    return traces[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/cresst/pulse_shape_smoke.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default=None, help="override config split (train|test)")
    parser.add_argument("--max-pulses", type=int, default=5000)
    parser.add_argument("--max-noise", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config.exists() else {}
    data_root = resolve_data_root(args.data_root, config)
    root = ensure_dataset_layout("cresst", data_root)
    split = args.split or str(config.get("split", "test"))
    seed = int(config.get("seed", 22))

    traces, features, meta = load_cresst_release(root / "raw", split=split, max_traces=None)
    subsets = select_cresst_subsets(
        traces,
        features,
        noise_column=str(config.get("noise_column", "noise")),
        clean_column=str(config.get("clean_column", "clean")),
    )
    pulses = _cap(subsets["pulse_traces"], args.max_pulses, seed)
    noise = _cap(subsets["noise_traces"], args.max_noise, seed)

    out = args.out or (root / "processed" / f"cresst_{split}_cache.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pulse_traces=pulses, noise_traces=noise)

    summary = {
        "split": split,
        "n_total": meta["n_total"],
        "n_pulse_available": subsets["n_pulse"],
        "n_noise_available": subsets["n_noise"],
        "n_pulse_cached": int(pulses.shape[0]),
        "n_noise_cached": int(noise.shape[0]),
        "trace_length": int(traces.shape[1]),
        "cache": str(out),
    }
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nCache written: {out}")
    print("Next: uv run python scripts/run_experiment.py --config configs/cresst/pulse_shape_smoke.yaml")


if __name__ == "__main__":
    main()
