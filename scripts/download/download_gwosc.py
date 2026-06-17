"""Download/cache small GWOSC event windows outside the repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.gwosc import dependency_status
from noise_geometry.utils import load_config, resolve_data_root
from noise_geometry.utils.paths import dataset_root, ensure_dataset_layout, is_within_repo


def _fetch_event_windows(args, config: dict, dataset_dir: Path) -> list[Path]:
    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries

    event = args.event or config.get("event", "GW150914")
    detectors = args.detectors or config.get("detectors", ["H1", "L1"])
    duration = float(args.duration or config.get("duration_seconds", 32.0))
    gps = float(event_gps(event))
    start = gps - duration / 2.0
    end = gps + duration / 2.0
    raw_dir = dataset_dir / "raw" / event
    raw_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for detector in detectors:
        output = raw_dir / f"{event}_{detector}_{int(duration)}s.npz"
        if output.exists() and not args.force:
            written.append(output)
            continue
        series = TimeSeries.fetch_open_data(detector, start, end, cache=True)
        import numpy as np

        np.savez_compressed(
            output,
            value=series.value,
            times=series.times.value,
            sample_rate=float(series.sample_rate.value),
            detector=detector,
            event=event,
            gps=gps,
            start=start,
            end=end,
        )
        written.append(output)

    metadata = {
        "event": event,
        "gps": gps,
        "detectors": detectors,
        "duration_seconds": duration,
        "files": [str(p) for p in written],
    }
    (raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/gwosc/gw150914_smoke.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--event", default=None)
    parser.add_argument("--detectors", nargs="+", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--download", action="store_true", help="download the configured event window")
    parser.add_argument("--force", action="store_true", help="overwrite cached .npz event files")
    parser.add_argument("--allow-repo-data", action="store_true", help="allow writing large files inside this repo")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(args.data_root, config)
    if is_within_repo(data_root, REPO_ROOT) and not args.allow_repo_data:
        raise SystemExit(f"Refusing to write GWOSC data inside repo: {data_root}")
    status = dependency_status()
    if not args.download:
        print("GWOSC dependency status:", status)
        print(f"Dataset directory: {dataset_root('gwosc', data_root)}")
        print("No data downloaded. Pass --download to fetch the configured event window.")
        return
    if not all(status.values()):
        print("GWOSC dependency status:", status)
        print("Install optional dependencies with: uv sync --extra gwosc")
        return
    try:
        dataset_dir = ensure_dataset_layout("gwosc", data_root)
    except OSError as exc:
        raise SystemExit(f"Cannot create GWOSC data directory under {data_root}: {exc}") from exc
    files = _fetch_event_windows(args, config, dataset_dir)
    print("Downloaded/cached files:")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
