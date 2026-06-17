"""CRESST preprocessing scaffold: inspect expected cache layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.utils import load_config, resolve_data_root
from noise_geometry.utils.paths import ensure_dataset_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/cresst/pulse_shape_smoke.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_dataset_layout("cresst", resolve_data_root(args.data_root, config))
    print(f"CRESST dataset directory: {root}")
    print(f"Place raw pulse-shape files under: {root / 'raw'}")
    print("Preprocessing implementation target: trace inspection, baseline/noise selection, PSD/covariance estimates.")


if __name__ == "__main__":
    main()
