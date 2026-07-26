"""EXP-07: stage an immutable, checksum-backed paper-evidence bundle.

Stages into a NEW output directory (never overwrites or deletes an existing
bundle) and fails loudly on any missing required evidence. The paper
repository should import only from a bundle that has separately passed
``scripts/verify_paper_bundle.py``.

    uv run python scripts/export_paper_bundle.py \\
      --source-root /absolute/path/to/noise-weighted-subspace-reconstruction \\
      --output /absolute/path/to/staging-directory \\
      --study-a-run <run_id> \\
      --study-b-run <run_id> \\
      --synthetic-run <label>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class MissingEvidenceError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit(source_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True
    )
    commit = out.stdout.strip()
    if not commit:
        raise RuntimeError("could not resolve source-root git commit")
    return commit


def require(path: Path, missing: list[str]) -> bool:
    if not path.exists():
        missing.append(str(path))
        return False
    return True


def copy_tree_into(src: Path, dst_root: Path, dst_relative: str, manifest: list[dict[str, Any]]) -> None:
    dst = dst_root / dst_relative
    if src.is_dir():
        for item in sorted(src.rglob("*")):
            if item.is_file():
                rel = item.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                manifest.append(
                    {
                        "bundle_path": str(target.relative_to(dst_root)),
                        "source_path": str(item),
                        "sha256": sha256_of(target),
                        "size_bytes": target.stat().st_size,
                    }
                )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append(
            {
                "bundle_path": str(dst.relative_to(dst_root)),
                "source_path": str(src),
                "sha256": sha256_of(dst),
                "size_bytes": dst.stat().st_size,
            }
        )


def gather_required_paths(source_root: Path, study_a_run: str, study_b_run: str) -> dict[str, list[Path]]:
    metrics_dir = source_root / "results" / "metrics"
    synthetic_dir = source_root / "transfer_paper" / "data" / "synthetic"
    derived_dir = source_root / "transfer_paper" / "data" / "derived"
    noise_module_dir = source_root / "noise_module" / "noise_module"
    audits_dir = source_root / "results" / "audits"
    study_a_dir = source_root / "results" / "noise_budget" / study_a_run / "study_a"
    study_b_dir = source_root / "results" / "noise_budget" / study_b_run / "study_b"
    public_data_dir = source_root / "results" / "public_data_package"

    synthetic_files = []
    for gate in [f"S{i}" for i in range(1, 10)]:
        synthetic_files.append(synthetic_dir / f"{gate}_sweep_10seeds.json")
        synthetic_files.append(synthetic_dir / f"{gate}_sweep_10seeds.csv")

    tables = [
        derived_dir / "claim_status.csv",
        derived_dir / "method_traceability.csv",
        derived_dir / "figure_index.csv",
        derived_dir / "paper_implications.csv",
    ]

    generator_convention = [
        noise_module_dir / "PAPER_CONVENTION.md",
        noise_module_dir / "paper_convention.json",
    ]

    provenance_audit = [
        audits_dir / "noise_generator_provenance.json",
        audits_dir / "noise_generator_provenance.md",
        audits_dir / "synthetic_config_provenance.csv",
        audits_dir / "rerun_decisions.json",
        audits_dir / "appendix_f_traceability.csv",
        audits_dir / "generator_distributional_diagnostics.json",
        audits_dir / "generator_distributional_diagnostics.png",
        audits_dir / "generator_distributional_diagnostics.pdf",
    ]

    return {
        "synthetic": synthetic_files,
        "tables": tables,
        "generator_convention": generator_convention,
        "provenance_audit": provenance_audit,
        "study_a_dir": [study_a_dir],
        "study_b_dir": [study_b_dir],
        "cresst_dir": [public_data_dir / "cresst"],
        "gwosc_dir": [public_data_dir / "gwosc"],
        "cresst_primary": [
            metrics_dir / "p2_cresst_validation_seed22.json",
            metrics_dir / "p2_cresst_validation_seed22.config.yaml",
        ],
        "gwosc_primary": [
            p
            for stem in ("gw150914", "filter_statistic_equivalence", "time_local_noise")
            for p in (
                metrics_dir / f"p1_gwosc_{stem}_seedna.json",
                metrics_dir / f"p1_gwosc_{stem}_seedna.config.yaml",
            )
        ],
    }


def build_bundle(source_root: Path, output: Path, study_a_run: str, study_b_run: str, synthetic_run: str) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory {output} already exists and is not empty; "
            "never overwrite an existing paper bundle -- pass a new path."
        )
    output.mkdir(parents=True, exist_ok=True)

    required = gather_required_paths(source_root, study_a_run, study_b_run)
    missing: list[str] = []
    for group, paths in required.items():
        for p in paths:
            require(p, missing)
    if missing:
        raise MissingEvidenceError(
            "missing required evidence, aborting before staging any bundle:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    manifest: list[dict[str, Any]] = []
    for f in required["synthetic"]:
        copy_tree_into(f, output, f"synthetic/{f.name}", manifest)
    for f in required["tables"]:
        copy_tree_into(f, output, f"tables/{f.name}", manifest)
    for f in required["generator_convention"]:
        copy_tree_into(f, output, f"generator_convention/{f.name}", manifest)
    for f in required["provenance_audit"]:
        copy_tree_into(f, output, f"audits/{f.name}", manifest)

    # EXP-04 confirmatory sweep: additive evidence, included if present but
    # not a hard requirement (a bundle built before EXP-04 ran is still
    # valid for everything else). Picks the newest results/sweeps_v2/ run.
    sweeps_v2_root = source_root / "results" / "sweeps_v2"
    confirmatory_run = None
    if sweeps_v2_root.is_dir():
        candidates = sorted(p for p in sweeps_v2_root.glob("sweeps_v2_*") if p.is_dir())
        confirmatory_run = candidates[-1] if candidates else None
    if confirmatory_run is not None:
        copy_tree_into(confirmatory_run, output, "sweeps_v2", manifest)

    copy_tree_into(required["study_a_dir"][0], output, "study_a", manifest)
    copy_tree_into(required["study_b_dir"][0], output, "study_b", manifest)
    copy_tree_into(required["cresst_dir"][0], output, "cresst", manifest)
    copy_tree_into(required["gwosc_dir"][0], output, "gwosc", manifest)
    for f in required["cresst_primary"]:
        copy_tree_into(f, output, f"cresst/primary/{f.name}", manifest)
    for f in required["gwosc_primary"]:
        copy_tree_into(f, output, f"gwosc/primary/{f.name}", manifest)

    top_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_repository_commit": git_commit(source_root),
        "run_identifiers": {
            "study_a_run": study_a_run,
            "study_b_run": study_b_run,
            "synthetic_run": synthetic_run,
            "s5_confirmatory_sweep_run": confirmatory_run.name if confirmatory_run is not None else None,
        },
        "file_count": len(manifest),
        "files": manifest,
    }
    (output / "BUNDLE_MANIFEST.json").write_text(json.dumps(top_manifest, indent=2) + "\n")

    print(f"staged {len(manifest)} files -> {output}")
    print(f"manifest: {output / 'BUNDLE_MANIFEST.json'}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-a-run", type=str, required=True)
    parser.add_argument("--study-b-run", type=str, required=True)
    parser.add_argument(
        "--synthetic-run",
        type=str,
        default="frozen_10seed_sweep",
        help=(
            "Label for the S1-S9 evidence bundled from transfer_paper/data/synthetic/. "
            "These are frozen 10-seed sweep files with no matching results/sweeps/ "
            "archive as of this revision (see results/audits/noise_generator_provenance.md "
            "'provenance_gap' note) -- this label does not assert an archived run_id exists."
        ),
    )
    args = parser.parse_args()

    build_bundle(
        args.source_root.resolve(),
        args.output.resolve(),
        args.study_a_run,
        args.study_b_run,
        args.synthetic_run,
    )


if __name__ == "__main__":
    main()
