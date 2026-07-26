"""Generate the ``transfer_paper/data/source_documents/`` copies from ``docs/``.

Why this exists
---------------
Before 2026-07-26 the same status documents existed as three independently
hand-edited copies -- ``docs/``, this repository's
``transfer_paper/data/source_documents/``, and the manuscript repository's
``transfer_paper/data/source_documents/`` -- with three different SHA-256
hashes. The manuscript repository was reading the oldest copy. A stale status
document is the one documentation defect that can put a wrong claim into a
paper, so the copies are now *generated* and hash-verified rather than
maintained.

Contract
--------
* ``docs/<name>.md`` in this repository is the single source of truth.
* Every generated copy carries a provenance header and is byte-identical to
  the source below that header.
* ``SOURCE_DOCUMENTS_MANIFEST.json`` records, for each document, the source
  path, the source SHA-256 (of the *unheadered* body), the generated file's
  own SHA-256, the git commit, and the generation timestamp.
* ``scripts/verify_paper_bundle.py`` fails if a copy disagrees with the
  manifest.

Usage
-----
    uv run python scripts/sync_source_documents.py                # this repo only
    uv run python scripts/sync_source_documents.py --latex-repo ../arXiv-v1-polished-extended-preprint
    uv run python scripts/sync_source_documents.py --check        # verify, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
RELATIVE_TARGET = Path("transfer_paper/data/source_documents")

#: Documents mirrored into the manuscript evidence tree. Paths are relative to
#: ``docs/``. Add a document here only if the manuscript actually cites it.
SYNCED_DOCUMENTS: tuple[str, ...] = (
    "CURRENT_STATUS.md",
    "EXPERIMENT_PROTOCOLS.md",
    "PAPER_REVISION_GUIDE.md",
    "results/CRESST_RESULT.md",
    "results/GWOSC_RESULT.md",
    "REMOTE_EXECUTION.md",
)

HEADER_TEMPLATE = (
    "<!-- GENERATED COPY - DO NOT EDIT.\n"
    "     Source: {source} @ {commit}\n"
    "     Generated: {generated} by scripts/sync_source_documents.py\n"
    "     Edit the source in the experiment repository and re-run the sync. -->\n\n"
)

HEADER_SENTINEL = "<!-- GENERATED COPY - DO NOT EDIT."


class SyncError(RuntimeError):
    pass


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git
        return "unknown"


def strip_header(text: str) -> str:
    """Return the document body with a previously written header removed."""
    if not text.startswith(HEADER_SENTINEL):
        return text
    end = text.find("-->")
    if end == -1:
        return text
    return text[end + len("-->") :].lstrip("\n")


def resolve_targets(latex_repo: Path | None) -> list[Path]:
    targets = [REPO_ROOT / RELATIVE_TARGET]
    if latex_repo is not None:
        resolved = latex_repo.resolve()
        if not resolved.is_dir():
            raise SyncError(f"latex repository not found: {resolved}")
        targets.append(resolved / RELATIVE_TARGET)
    return targets


def build_entries(commit: str, generated: str) -> list[dict]:
    entries = []
    for relative in SYNCED_DOCUMENTS:
        source = DOCS / relative
        if not source.exists():
            raise SyncError(
                f"source document missing: {source.relative_to(REPO_ROOT)}. "
                "Update SYNCED_DOCUMENTS after moving or renaming a document."
            )
        body = source.read_text(encoding="utf-8")
        header = HEADER_TEMPLATE.format(
            source=f"docs/{relative}",
            commit=commit,
            generated=generated,
        )
        rendered = header + body
        entries.append(
            {
                "name": Path(relative).name,
                "source_path": f"docs/{relative}",
                "source_sha256": sha256_of_bytes(body.encode("utf-8")),
                "generated_sha256": sha256_of_bytes(rendered.encode("utf-8")),
                "size_bytes": len(rendered.encode("utf-8")),
                "_rendered": rendered,
            }
        )
    return entries


def write_targets(entries: list[dict], targets: list[Path], manifest: dict) -> None:
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            (target / entry["name"]).write_text(entry["_rendered"], encoding="utf-8")
        (target / "SOURCE_DOCUMENTS_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def check_targets(entries: list[dict], targets: list[Path]) -> list[str]:
    """Return a list of drift errors; empty means every copy is in sync."""
    errors: list[str] = []
    by_name = {entry["name"]: entry for entry in entries}
    for target in targets:
        manifest_path = target / "SOURCE_DOCUMENTS_MANIFEST.json"
        if not manifest_path.exists():
            errors.append(f"{manifest_path}: missing manifest; run the sync")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = {item["name"]: item for item in manifest.get("documents", [])}
        for name, entry in by_name.items():
            copy = target / name
            if not copy.exists():
                errors.append(f"{copy}: missing generated copy")
                continue
            actual = sha256_of_bytes(copy.read_bytes())
            if name not in recorded:
                errors.append(f"{copy}: not listed in {manifest_path.name}")
                continue
            if actual != recorded[name]["generated_sha256"]:
                errors.append(
                    f"{copy}: hash disagrees with its manifest entry "
                    "(the copy was hand-edited; edit docs/ instead)"
                )
            body = strip_header(copy.read_text(encoding="utf-8"))
            if sha256_of_bytes(body.encode("utf-8")) != entry["source_sha256"]:
                errors.append(
                    f"{copy}: body differs from {entry['source_path']}; re-run the sync"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latex-repo",
        type=Path,
        default=None,
        help="path to the manuscript repository; its source_documents tree is synced too",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the copies against the manifest and exit non-zero on drift",
    )
    args = parser.parse_args()

    commit = git_commit(REPO_ROOT)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        targets = resolve_targets(args.latex_repo)
        entries = build_entries(commit, generated)
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.check:
        errors = check_targets(entries, targets)
        if errors:
            print("source-document drift detected:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            raise SystemExit(1)
        print(f"source documents in sync ({len(entries)} documents, {len(targets)} targets)")
        return

    manifest = {
        "schema": "source-documents-manifest/1.0.0",
        "generated_utc": generated,
        "source_repository_commit": commit,
        "generator": "scripts/sync_source_documents.py",
        "documents": [
            {k: v for k, v in entry.items() if not k.startswith("_")} for entry in entries
        ],
    }
    write_targets(entries, targets, manifest)
    for target in targets:
        print(f"synced {len(entries)} documents -> {target}")


if __name__ == "__main__":
    main()
