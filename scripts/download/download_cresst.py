"""Prepare/download CRESST pulse-shape data outside the repository."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.utils import load_config, resolve_data_root
from noise_geometry.utils.paths import ensure_dataset_layout, is_within_repo

CRESST_DATASET_URL = "https://www.origins-cluster.de/odsl/dark-matter-data-center/available-datasets/cresst"


def _download_file(url: str, output: Path, *, force: bool = False) -> Path:
    if output.exists() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with output.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=output.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/cresst/pulse_shape_smoke.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--url", default=None, help="optional direct file URL if available from the CRESST page")
    parser.add_argument("--filename", default=None, help="output filename for --url")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-repo-data", action="store_true", help="allow writing large files inside this repo")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(args.data_root, config)
    if is_within_repo(data_root, REPO_ROOT) and not args.allow_repo_data:
        raise SystemExit(f"Refusing to write CRESST data inside repo: {data_root}")
    try:
        dataset_dir = ensure_dataset_layout("cresst", data_root)
    except OSError as exc:
        raise SystemExit(f"Cannot create CRESST data directory under {data_root}: {exc}") from exc

    if args.url:
        filename = args.filename or Path(args.url).name or "cresst_download"
        path = _download_file(args.url, dataset_dir / "raw" / filename, force=args.force)
        print(path)
        return

    print(f"CRESST dataset directory prepared: {dataset_dir}")
    print("No direct file URL was provided.")
    print("Manual download source:")
    print(CRESST_DATASET_URL)
    print(f"Place released pulse-shape archives under: {dataset_dir / 'raw'}")
    print("Then run scripts/preprocess/preprocess_cresst.py with the same --data-root.")


if __name__ == "__main__":
    main()
