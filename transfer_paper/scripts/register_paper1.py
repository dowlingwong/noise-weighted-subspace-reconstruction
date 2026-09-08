#!/usr/bin/env python3
"""Register Paper 1 (2026-09) experiment outputs into a `transfer_paper/` bundle.

Ground rule 7 of the 2026-09-06 agent instruction: a number is citable only
after the result file is copied to ``transfer_paper/data/paper1/<ID>/``, added
to ``BUNDLE_MANIFEST.json`` (with ``bytes``, ``path``, ``sha256``, ``source``),
and given a row in ``MANUSCRIPT_EVIDENCE_MAP.md``. Both repositories carry a
copy of ``transfer_paper/`` and both must be written.

This script does the first two mechanically and prints the evidence-map rows
for the third, which is a writing-control document and is appended, never
rewritten.

It is additive and idempotent: an existing manifest entry for the same path is
replaced in place, other entries are untouched, and no file outside
``data/paper1/`` is written.

Usage::

    python transfer_paper/scripts/register_paper1.py \
        --results-root results/paper1 \
        --transfer-root transfer_paper \
        [--ids P1-E0 P1-E2 ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

#: What is copied for each experiment. Files that do not exist are skipped and
#: reported, so a partially-run card registers what it has.
REGISTER: dict[str, list[str]] = {
    "P1-E0":  ["PREDECLARED.md", "regression_check.csv", "summary.json", "manifest.json"],
    "P1-E2":  ["PREDECLARED.md", "alpha_sweep.csv", "summary.json", "manifest.json",
               "fig_alpha_cost.pdf"],
    "P1-E6":  ["PREDECLARED.md", "rank_one_check.csv", "summary.json", "manifest.json"],
    "P1-E9":  ["PREDECLARED.md", "probes.csv", "summary.json", "manifest.json"],
    "P1-E11": ["PREDECLARED.md", "excited_support.csv", "complement_spread.csv",
               "summary.json", "manifest.json", "fig_excited_support.pdf"],
    "P1-E12": ["PREDECLARED.md", "ae_ablation.csv", "summary.json", "manifest.json",
               "fig_ae_ablation.pdf"],
    "P1-E4b": ["PREDECLARED.md", "hyperparameters.csv", "hyperparameters.tex",
               "summary.json", "manifest.json"],
    "P1-E13": ["PREDECLARED.md", "summary.json"],
    "P1-E10": ["summary.json"],
}

SOURCE = "results/paper1 (2026-09 agent run)"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/paper1")
    ap.add_argument("--transfer-root", default="transfer_paper")
    ap.add_argument("--ids", nargs="+", default=sorted(REGISTER))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    results_root = Path(a.results_root).resolve()
    transfer_root = Path(a.transfer_root).resolve()
    manifest_path = transfer_root / "BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"no bundle manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    entries = {e["path"]: e for e in manifest.get("files", [])}

    copied, skipped = [], []
    for exp_id in a.ids:
        for name in REGISTER.get(exp_id, []):
            src = results_root / exp_id / name
            if not src.exists():
                skipped.append(f"{exp_id}/{name}")
                continue
            rel = f"data/paper1/{exp_id}/{name}"
            dst = transfer_root / rel
            if not a.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            entries[rel] = {"bytes": src.stat().st_size, "path": rel,
                            "sha256": sha256_of(src), "source": SOURCE}
            copied.append(rel)

    manifest["files"] = [entries[k] for k in sorted(entries)]
    if not a.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"{'DRY RUN: ' if a.dry_run else ''}registered {len(copied)} files into {transfer_root}")
    for rel in copied:
        print(f"  + {rel}")
    if skipped:
        print(f"  ({len(skipped)} not present, skipped: {', '.join(skipped)})")


if __name__ == "__main__":
    main()
