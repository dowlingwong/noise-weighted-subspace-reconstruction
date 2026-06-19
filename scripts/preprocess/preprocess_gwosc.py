"""GWOSC preprocessing and independent GWpy normalization check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.utils import load_config, resolve_data_root
from noise_geometry.utils.paths import ensure_dataset_layout
from noise_geometry.gwosc import run_gwosc_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/gwosc/gw150914_smoke.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--reference-check",
        action="store_true",
        help="run PSD and whitening normalization checks against GWpy",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(args.data_root, config)
    root = ensure_dataset_layout("gwosc", data_root)
    print(f"GWOSC dataset directory: {root}")
    if not args.reference_check:
        print("Run scripts/download/download_gwosc.py --download if the cache is empty.")
        print("Pass --reference-check to compare PSD and whitening normalization with GWpy.")
        return

    reference_config = dict(config.get("gwpy_reference", {}))
    reference_config["enabled"] = True
    config["gwpy_reference"] = reference_config
    result = run_gwosc_experiment(config, data_root)
    references = {
        detector: metrics["gwpy_reference"]
        for detector, metrics in result["detectors"].items()
    }
    output = args.output or root / "processed" / f"{config.get('event', 'event')}_gwpy_reference.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(references, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
