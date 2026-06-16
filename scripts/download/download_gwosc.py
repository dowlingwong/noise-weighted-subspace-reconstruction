"""GWOSC download helper stub.

Install optional dependencies first:

    python -m pip install gwosc gwpy

The default mode is a dependency check only, so tests and smoke runs never
download large open-data products by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.gwosc import dependency_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="GW150914")
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument("--duration", type=float, default=32.0)
    parser.add_argument("--download", action="store_true", help="opt in to future data fetching")
    args = parser.parse_args()

    status = dependency_status()
    if not args.download or not all(status.values()):
        print("GWOSC dependency status:", status)
        print("No data downloaded. Install gwosc/gwpy and pass --download after the fetch implementation is completed.")
        return
    raise NotImplementedError("Actual GWOSC fetching is intentionally deferred until dependencies are available.")


if __name__ == "__main__":
    main()
