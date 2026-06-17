"""GWOSC preprocessing scaffold: report cache layout and next steps."""

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
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/gwosc/gw150914_smoke.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_dataset_layout("gwosc", resolve_data_root(args.data_root, config))
    print(f"GWOSC dataset directory: {root}")
    print("Preprocessing implementation target: PSD estimation, whitening metadata, and off-source injection windows.")
    print("Current runnable step: scripts/download/download_gwosc.py --download")


if __name__ == "__main__":
    main()
