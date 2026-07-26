"""EXP-07: verify a staged paper-evidence bundle.

Recomputes every checksum in ``BUNDLE_MANIFEST.json``, checks for orphan
files not covered by the manifest, cross-checks a handful of derived tables
against their authoritative JSON so a formatting bug cannot silently
contradict the source record, and runs the semantic checks required by
``docs/plans/REVISION_PLAN_2026-07-26.md`` (forbidden Study A phrases, the
obsolete fixed-modulus sampler description, and local Markdown link
resolution).

Exits non-zero on any failure. The paper repository should import only
after this passes.

    uv run python scripts/verify_paper_bundle.py <staging-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

FORBIDDEN_STUDY_A_PHRASES = ["1/f phonon", "Vibrational", "TES Johnson"]
FORBIDDEN_SAMPLER_PHRASES = [r"Uniform\[0,\s*2\\?pi\)", r"Uniform\[0,\s*2\s*\\?pi\)"]
TEXT_SUFFIXES = {".md", ".json", ".csv", ".yaml", ".yml", ".log", ".txt"}


class BundleVerificationError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_checksums(bundle_dir: Path, manifest: dict) -> list[str]:
    errors = []
    manifest_paths = set()
    for entry in manifest["files"]:
        rel = entry["bundle_path"]
        manifest_paths.add(rel)
        full = bundle_dir / rel
        if not full.exists():
            errors.append(f"manifest lists missing file: {rel}")
            continue
        actual = sha256_of(full)
        if actual != entry["sha256"]:
            errors.append(f"checksum mismatch for {rel}: manifest={entry['sha256']} actual={actual}")
        actual_size = full.stat().st_size
        if actual_size != entry["size_bytes"]:
            errors.append(f"size mismatch for {rel}: manifest={entry['size_bytes']} actual={actual_size}")

    # BUNDLE_MANIFEST.json cannot list its own checksum (self-referential);
    # manifest.json and SHA256SUMS are bundle-metadata side-files following
    # the evidence/ directory convention (see evidence/README.md), not
    # evidence content -- SHA256SUMS separately covers every other file,
    # including manifest.json, so this exclusion does not weaken the check.
    bundle_metadata_files = {"BUNDLE_MANIFEST.json", "manifest.json", "SHA256SUMS"}
    for item in bundle_dir.rglob("*"):
        if item.is_file() and item.name not in bundle_metadata_files:
            rel = str(item.relative_to(bundle_dir))
            if rel not in manifest_paths:
                errors.append(f"orphan file not covered by manifest (no checksum on record): {rel}")
    return errors


def check_derived_table_consistency(bundle_dir: Path) -> list[str]:
    errors = []

    study_a_summary_path = bundle_dir / "study_a" / "study_a_summary.json"
    study_a_csv_path = bundle_dir / "study_a" / "study_a_band_fractions.csv"
    if study_a_summary_path.exists() and study_a_csv_path.exists():
        summary = json.loads(study_a_summary_path.read_text())
        n_bands_expected = len(summary["band_edges_hz"]) - 1
        import csv

        with study_a_csv_path.open() as fh:
            n_rows = sum(1 for _ in csv.DictReader(fh))
        if n_rows != n_bands_expected:
            errors.append(
                f"study_a_band_fractions.csv has {n_rows} rows but "
                f"study_a_summary.json's band_edges_hz implies {n_bands_expected} bands"
            )
        tol = summary["quadrature_check"]
        if tol["max_frac_dev"] > 0.01 or tol["rms_frac_dev"] > 0.01:
            errors.append(f"study_a_summary.json quadrature_check exceeds its own recorded tolerance: {tol}")

    study_b_csv_path = bundle_dir / "study_b" / "study_b_scenarios.csv"
    if study_b_csv_path.exists():
        import csv

        with study_b_csv_path.open() as fh:
            for row in csv.DictReader(fh):
                if row["scenario"] == "baseline":
                    ratio = float(row["relative_sensitivity_sqrt_I_ratio"])
                    sigma_ratio = float(row["sigma_ratio_to_baseline"])
                    if abs(ratio - 1.0) > 1e-9 or abs(sigma_ratio - 1.0) > 1e-9:
                        errors.append(
                            f"study_b_scenarios.csv baseline row does not self-consistently "
                            f"equal 1.0: relative_sensitivity={ratio} sigma_ratio={sigma_ratio}"
                        )

    cresst_json_path = bundle_dir / "cresst" / "cresst_stable_summary.json"
    if cresst_json_path.exists():
        cresst = json.loads(cresst_json_path.read_text())
        if cresst["cluster_count"] != len(cresst["channels"]):
            errors.append("cresst_stable_summary.json cluster_count does not match len(channels)")
        if cresst["run_count"] != len(cresst["run_ids"]):
            errors.append("cresst_stable_summary.json run_count does not match len(run_ids)")

    gwosc_identity_path = bundle_dir / "gwosc" / "gwosc_implementation_identity_summary.json"
    gwosc_calibration_path = bundle_dir / "gwosc" / "gwosc_calibration_summary.json"
    if gwosc_identity_path.exists() and gwosc_calibration_path.exists():
        identity = json.loads(gwosc_identity_path.read_text())
        calibration = json.loads(gwosc_calibration_path.read_text())
        if identity["status"] == calibration["status"]:
            errors.append(
                "gwosc implementation-identity and calibration summaries report the same "
                "'status' value -- these must remain distinguishable metric classes"
            )

    return errors


def check_forbidden_phrases(bundle_dir: Path) -> list[str]:
    errors = []
    for path in bundle_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        is_study_a = "study_a" in str(path.relative_to(bundle_dir)).lower()
        if is_study_a:
            for phrase in FORBIDDEN_STUDY_A_PHRASES:
                if phrase in text:
                    errors.append(f"forbidden Study A phrase '{phrase}' found in {path.relative_to(bundle_dir)}")
        for pattern in FORBIDDEN_SAMPLER_PHRASES:
            if re.search(pattern, text) and "historical" not in text.lower()[:2000]:
                errors.append(
                    f"obsolete fixed-modulus sampler description pattern matched in "
                    f"{path.relative_to(bundle_dir)} (outside an explicitly historical note)"
                )
    return errors


_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def check_markdown_links(bundle_dir: Path) -> list[str]:
    errors = []
    for md_path in bundle_dir.rglob("*.md"):
        text = md_path.read_text(errors="ignore")
        for link in _MD_LINK_RE.findall(text):
            if link.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = (md_path.parent / link.split("#")[0]).resolve()
            try:
                target.relative_to(bundle_dir.resolve())
            except ValueError:
                continue  # link points outside the bundle; not this check's concern
            if not target.exists():
                errors.append(f"broken local Markdown link in {md_path.relative_to(bundle_dir)}: {link}")
    return errors


def check_source_document_drift() -> list[str]:
    """Fail if any generated ``source_documents/`` copy drifted from ``docs/``.

    Before 2026-07-26 the status documents existed as three independently
    hand-edited copies with three different hashes, and the manuscript
    repository was reading the oldest. They are now generated by
    ``scripts/sync_source_documents.py``; this check makes drift a hard error
    rather than something a reader has to notice.
    """
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        import sync_source_documents as sync
    except ImportError as exc:  # pragma: no cover - script always present
        return [f"cannot import sync_source_documents: {exc}"]

    try:
        entries = sync.build_entries(commit="verify", generated="verify")
    except sync.SyncError as exc:
        return [f"source-document sources unavailable: {exc}"]
    targets = [repo_root / sync.RELATIVE_TARGET]
    if not targets[0].exists():
        return []  # nothing generated yet in this checkout
    return [f"source-document drift: {e}" for e in sync.check_targets(entries, targets)]


def verify(bundle_dir: Path) -> list[str]:
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        return [f"no BUNDLE_MANIFEST.json found at {bundle_dir}"]
    manifest = json.loads(manifest_path.read_text())

    errors: list[str] = []
    # Checksums first and always: if a file was corrupted, every other check
    # is untrustworthy, but a check crashing on corrupted content (e.g.
    # invalid JSON) must not prevent the checksum mismatch itself from being
    # reported -- so each check runs independently and failures are caught,
    # not allowed to abort the whole verification run.
    checks = {
        "checksums": lambda: check_checksums(bundle_dir, manifest),
        "derived_table_consistency": lambda: check_derived_table_consistency(bundle_dir),
        "forbidden_phrases": lambda: check_forbidden_phrases(bundle_dir),
        "markdown_links": lambda: check_markdown_links(bundle_dir),
        "source_document_drift": check_source_document_drift,
    }
    for name, check in checks.items():
        try:
            errors += check()
        except Exception as exc:  # noqa: BLE001 - report, don't crash the verifier
            errors.append(f"check '{name}' raised {type(exc).__name__}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()

    errors = verify(args.bundle_dir.resolve())
    if errors:
        print(f"BUNDLE VERIFICATION FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"bundle verification passed: {args.bundle_dir}")


if __name__ == "__main__":
    main()
