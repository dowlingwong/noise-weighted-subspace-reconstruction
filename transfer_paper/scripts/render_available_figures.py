"""Render every Paper 1 figure supported by currently transferred evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from paper_support import (
    plot_filter_equivalence,
    plot_gwosc_null_calibration,
    plot_gwosc_reference_comparison,
    plot_gwosc_run_history,
    plot_paper_claim_support_matrix,
    plot_synthetic_validation_overview,
    plot_time_local_psd,
    save_figure,
)


TRANSFER_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bundle_manifest() -> None:
    data_manifest_path = TRANSFER_ROOT / "data" / "transfer_manifest.json"
    selected_run = ""
    if data_manifest_path.is_file():
        selected_run = json.loads(
            data_manifest_path.read_text(encoding="utf-8")
        ).get("selected_gwosc_run_id", "")
    manifest_path = TRANSFER_ROOT / "BUNDLE_MANIFEST.json"
    files = []
    for path in sorted(TRANSFER_ROOT.rglob("*")):
        if not path.is_file() or path == manifest_path or "__pycache__" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(TRANSFER_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "selected_gwosc_run_id": selected_run,
        "raw_data_in_transfer_bundle": False,
        "standalone_for_paper_writing": True,
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    available = [
        (
            "synthetic_validation_overview",
            plot_synthetic_validation_overview(),
        ),
        ("gwosc_null_calibration", plot_gwosc_null_calibration()),
        ("gwosc_reference_comparison", plot_gwosc_reference_comparison()),
        ("gwosc_run_history", plot_gwosc_run_history()),
        ("paper_claim_support_matrix", plot_paper_claim_support_matrix()),
        ("gwosc_filter_equivalence", plot_filter_equivalence()),
        ("gwosc_time_local_psd", plot_time_local_psd()),
    ]
    for stem, figure in available:
        if figure is None:
            print(f"Pending evidence; skipped {stem}")
            continue
        pdf, png = save_figure(figure, stem)
        plt.close(figure)
        print(f"Wrote {pdf}")
        print(f"Wrote {png}")
    _write_bundle_manifest()
    print(f"Wrote {TRANSFER_ROOT / 'BUNDLE_MANIFEST.json'}")


if __name__ == "__main__":
    main()
