"""Fetch a small TIDMAD subset for Phase 0/1, without putting data in the repo.

The TIDMAD release ships its own downloader (``download_data.py`` in
``jessicafry/TIDMAD``), which generates and runs wget commands against OSDF
caches. This wrapper does not reimplement it: reimplementing a download client
is how checksums and cache-selection logic silently diverge. It resolves the
destination under the external data root, prints the exact upstream command,
and -- with ``--run`` -- executes it there.

Default subset is deliberately tiny: two validation files (which carry the
injected reference channel, i.e. the ground truth) and two science files (which
carry signal-free noise for PSD estimation). That is everything Phase 0 and
Phase 1 need; the full release is ~TB-scale and is not required until Phase 5.

    uv run python scripts/download/download_tidmad.py
    uv run python scripts/download/download_tidmad.py --run --upstream ~/src/TIDMAD
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/ceph/dwong/paper1_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--upstream", type=Path, default=None,
                        help="path to a checkout of jessicafry/TIDMAD containing download_data.py")
    parser.add_argument("--train-files", type=int, default=0)
    parser.add_argument("--validation-files", type=int, default=2)
    parser.add_argument("--science-files", type=int, default=2)
    parser.add_argument("--cache", type=str, default="Director",
                        choices=["NY", "NorCal", "SoCal", "Director"])
    parser.add_argument("--run", action="store_true", help="execute instead of printing")
    parser.add_argument("--allow-repo-data", action="store_true",
                        help="permit a destination inside this repository (discouraged)")
    args = parser.parse_args()

    destination = args.data_root / "tidmad" / "raw"
    if not args.allow_repo_data and REPO_ROOT in destination.resolve().parents:
        raise SystemExit(
            f"refusing to download into the repository ({destination}). "
            "Use --data-root to point at external storage, or pass --allow-repo-data."
        )

    command = [
        sys.executable, "download_data.py",
        "-o", str(destination),
        "-c", args.cache,
        "-t", str(args.train_files),
        "-v", str(args.validation_files),
        "-s", str(args.science_files),
        "-sk",
    ]

    print(f"destination: {destination}")
    print("upstream command:")
    print("  " + " ".join(command))

    if not args.run:
        print("\n(dry run; pass --run together with --upstream to execute)")
        return

    if args.upstream is None:
        raise SystemExit("--run requires --upstream <path to TIDMAD checkout>")
    script = Path(args.upstream) / "download_data.py"
    if not script.exists():
        raise SystemExit(f"upstream downloader not found: {script}")

    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=str(args.upstream), check=True)
    print(f"\ndownloaded into {destination}")
    print("next: uv run python scripts/tidmad_phase0_contract.py --data-root " f"{args.data_root}")


if __name__ == "__main__":
    main()
