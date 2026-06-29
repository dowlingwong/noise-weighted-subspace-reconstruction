"""Download and verify the public CRESST-II/III pulse-shape release.

The dataset is described in CRESST Collaboration, "Description of CRESST-II and
CRESST-III pulse shape data", arXiv:2508.03078 (2025), and hosted by the
ORIGINS Dark Matter Data Center (DMDC):

    https://www.origins-cluster.de/odsl/dark-matter-data-center/available-datasets/cresst

The DMDC serves files through an interactive repository browser, so there is no
single stable public direct-download URL baked into this script. Instead the
downloader is driven by a base URL (``--base-url`` or config ``cresst_base_url``)
that points at the directory holding the released files, or by explicit
per-file URLs (config ``cresst_file_urls``). It always downloads to the external
data root, records checksums/provenance, and verifies array shapes and CSV row
counts against the published manifest before declaring success.

Quick start on a remote server (downloads only the ~160 MB test split first):

    uv run python scripts/download/download_cresst.py \
        --base-url <DMDC_FILE_DIRECTORY_URL> --group test

Then the full release:

    uv run python scripts/download/download_cresst.py \
        --base-url <DMDC_FILE_DIRECTORY_URL> --group all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.utils import load_config, resolve_data_root  # noqa: E402
from noise_geometry.utils.paths import (  # noqa: E402
    dataset_root,
    ensure_dataset_layout,
    is_within_repo,
)

DATASET_PAGE = (
    "https://www.origins-cluster.de/odsl/dark-matter-data-center/"
    "available-datasets/cresst"
)
CITATION = (
    "Cite both: (1) CRESST Collaboration, 'Description of CRESST-II and "
    "CRESST-III pulse shape data', arXiv:2508.03078 (2025); (2) G. Angloher "
    "et al., 'Towards an automated data cleaning with deep learning in CRESST', "
    "Eur. Phys. J. Plus 138, 100 (2023), arXiv:2211.00564."
)


@dataclass(frozen=True)
class ReleaseFile:
    name: str
    group: str  # one of: train, test, meta
    kind: str  # one of: npy, csv
    shape: tuple[int, ...] | None = None  # expected .npy shape
    dtype: str | None = None  # expected .npy dtype (informational)
    rows: int | None = None  # expected CSV data rows (excluding header)
    required_columns: tuple[str, ...] = ()


# Published manifest (arXiv:2508.03078, Sec. III). Record counts are exact.
RELEASE_FILES: tuple[ReleaseFile, ...] = (
    ReleaseFile("X_train.npy", "train", "npy", shape=(979446, 512), dtype="float32"),
    ReleaseFile("y_train.npy", "train", "npy", shape=(979446,)),
    ReleaseFile(
        "features_train.csv", "train", "csv", rows=979446,
        required_columns=("run", "channel", "noise", "clean", "pulse_height"),
    ),
    ReleaseFile("X_test.npy", "test", "npy", shape=(78084, 512), dtype="float32"),
    ReleaseFile("y_test.npy", "test", "npy", shape=(78084,)),
    ReleaseFile(
        "features_test.csv", "test", "csv", rows=78084,
        required_columns=("run", "channel", "noise", "clean", "pulse_height"),
    ),
    ReleaseFile(
        "resolutions_test.csv", "test", "csv",
        required_columns=("run", "channel", "of_resolution"),
    ),
    ReleaseFile(
        "detectors.csv", "meta", "csv",
        required_columns=("run", "channel", "test", "nmbr_events", "noise"),
    ),
)

GROUPS = ("all", "train", "test", "meta")


def _selected_files(group: str, names: list[str] | None) -> list[ReleaseFile]:
    if names:
        wanted = set(names)
        chosen = [f for f in RELEASE_FILES if f.name in wanted]
        missing = wanted - {f.name for f in chosen}
        if missing:
            raise SystemExit(f"Unknown --files entries: {sorted(missing)}")
        return chosen
    if group == "all":
        return list(RELEASE_FILES)
    return [f for f in RELEASE_FILES if f.group == group]


def _resolve_url(spec: ReleaseFile, base_url: str | None, file_urls: dict) -> str | None:
    if spec.name in file_urls:
        return str(file_urls[spec.name])
    if base_url:
        return base_url.rstrip("/") + "/" + spec.name
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, output: Path, *, timeout: float, force: bool) -> None:
    if output.exists() and not force:
        return
    try:
        import requests
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(f"requests/tqdm required for download: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=output.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(output)


def _verify(spec: ReleaseFile, path: Path) -> dict:
    """Verify a downloaded file against the published manifest.

    Returns a record with observed properties. Raises ValueError on a
    hard integrity mismatch (wrong shape or row count = corrupt/partial file).
    """
    record: dict = {"name": spec.name, "bytes": path.stat().st_size}
    if spec.kind == "npy":
        arr = np.load(path, mmap_mode="r")  # memory-mapped: no full read
        record["shape"] = list(arr.shape)
        record["dtype"] = str(arr.dtype)
        if spec.shape is not None and tuple(arr.shape) != spec.shape:
            raise ValueError(
                f"{spec.name}: shape {tuple(arr.shape)} != expected {spec.shape}"
            )
        if spec.dtype is not None and str(arr.dtype) != spec.dtype:
            record["dtype_warning"] = f"expected {spec.dtype}, got {arr.dtype}"
    elif spec.kind == "csv":
        header, n_rows = _csv_header_and_rowcount(path)
        record["rows"] = n_rows
        record["columns"] = header
        if spec.rows is not None and n_rows != spec.rows:
            raise ValueError(
                f"{spec.name}: {n_rows} data rows != expected {spec.rows}"
            )
        missing = [c for c in spec.required_columns if c not in header]
        if missing:
            raise ValueError(f"{spec.name}: missing expected columns {missing}")
    record["sha256"] = _sha256(path)
    return record


def _csv_header_and_rowcount(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        first = fh.readline()
        header = [c.strip() for c in first.rstrip("\n").split(",")]
        n_rows = sum(1 for _ in fh)
    return header, n_rows


def _tool_versions() -> dict:
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import requests

        versions["requests"] = requests.__version__
    except Exception:  # pragma: no cover
        pass
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/cresst/pulse_shape_smoke.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--base-url", default=None, help="directory URL holding the released files")
    parser.add_argument("--group", choices=GROUPS, default="all", help="which file group to fetch")
    parser.add_argument("--files", nargs="*", default=None, help="explicit filenames to fetch")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--check", action="store_true", help="verify existing files; do not download")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--allow-repo-data", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config) if args.config.exists() else {}
    data_root = resolve_data_root(args.data_root, config)
    if is_within_repo(data_root, REPO_ROOT) and not args.allow_repo_data:
        raise SystemExit(f"Refusing to write CRESST data inside repo: {data_root}")

    base_url = (args.base_url or config.get("cresst_base_url") or "").strip() or None
    file_urls = dict(config.get("cresst_file_urls", {}))
    files = _selected_files(args.group, args.files)

    try:
        root = ensure_dataset_layout("cresst", data_root)
    except OSError as exc:
        raise SystemExit(f"Cannot create CRESST data dir under {data_root}: {exc}") from exc
    raw_dir = root / "raw"

    print(f"CRESST dataset directory: {root}")
    print(f"Source page: {DATASET_PAGE}")
    print(f"Selected ({args.group}): {', '.join(f.name for f in files)}")

    if args.dry_run:
        for spec in files:
            url = _resolve_url(spec, base_url, file_urls)
            print(f"  {spec.name:22s} <- {url or '(no URL: set --base-url)'}")
        return

    if not args.check and base_url is None and not file_urls:
        print("\nNo download URL configured.")
        print("Open the DMDC CRESST page, copy the directory URL that holds the")
        print("released files, and re-run with --base-url <URL> (or set")
        print("cresst_base_url in the config). Place files manually under:")
        print(f"  {raw_dir}")
        print("then re-run with --check to verify.")
        raise SystemExit(2)

    records, failures = [], []
    for spec in files:
        target = raw_dir / spec.name
        if not args.check:
            url = _resolve_url(spec, base_url, file_urls)
            if url is None:
                failures.append(f"{spec.name}: no URL")
                continue
            try:
                _download(url, target, timeout=args.timeout, force=args.force)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{spec.name}: download failed ({exc})")
                continue
        if not target.exists():
            failures.append(f"{spec.name}: missing at {target}")
            continue
        try:
            records.append(_verify(spec, target))
            print(f"  verified {spec.name}")
        except ValueError as exc:
            failures.append(str(exc))

    manifest = {
        "dataset": "cresst_pulse_shape",
        "source_page": DATASET_PAGE,
        "base_url": base_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool_versions": _tool_versions(),
        "citation": CITATION,
        "files": records,
        "failures": failures,
    }
    manifest_path = raw_dir / "cresst_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Citation: {CITATION}")

    if failures:
        print("\nFAILURES:")
        for line in failures:
            print(f"  - {line}")
        raise SystemExit(1)
    print(f"\nOK: {len(records)} file(s) verified. Next: scripts/preprocess/preprocess_cresst.py")


if __name__ == "__main__":
    main()
