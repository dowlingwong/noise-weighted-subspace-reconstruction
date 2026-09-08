#!/usr/bin/env python3
"""Verify a ``transfer_paper/`` bundle against its ``BUNDLE_MANIFEST.json``.

Checks, for one bundle:
  * every manifest entry names a file that exists, with matching SHA-256 and
    byte count;
  * no path is listed twice;
  * (optionally, ``--require-complete``) every file present under ``data/``,
    ``figures/`` or ``tables/`` is listed, so that "manifest verifies" also
    means "manifest is complete";
  * paths recorded under the manifest's ``excluded_unavailable`` block are
    neither present nor listed, so a missing path is either an error or an
    explicitly recorded omission.

With ``--compare OTHER_BUNDLE`` it additionally reports every path present in
both bundles whose bytes differ (``--fail-on-scientific-diff`` makes a
differing data/figures/tables file an error, and ``--require-shared data/
figures/`` makes a file listed in only one bundle under those prefixes an
error, so that the comparison is not intersection-only), split into "scientific" (``data/``,
``figures/``, ``tables/``, ``notebooks/``) and "document" (everything else) so
that a legitimate documentary difference is not mistaken for a scientific one.

Exit status is 0 only when no error was found.  Usage::

    python transfer_paper/scripts/verify_bundle.py [--bundle transfer_paper]
        [--compare ../other_repo/transfer_paper] [--require-complete]
        [--fail-on-scientific-diff]

This script never modifies anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCIENTIFIC_PREFIXES = ("data/", "figures/", "tables/", "notebooks/")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(bundle: Path) -> dict:
    return json.loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))


def verify(bundle: Path, require_complete: bool) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (errors, notes, {path: sha256}) for one bundle."""
    errors: list[str] = []
    notes: list[str] = []
    manifest = load_manifest(bundle)
    entries = manifest["files"]
    seen: dict[str, str] = {}
    unavailable = manifest.get("excluded_unavailable")
    excluded = set(unavailable.keys()) if isinstance(unavailable, dict) else set()
    for entry in entries:
        rel = entry["path"]
        if rel in seen:
            errors.append(f"duplicate manifest path: {rel}")
            continue
        seen[rel] = entry["sha256"]
        target = bundle / rel
        if not target.is_file():
            errors.append(f"missing file: {rel}")
            continue
        size = target.stat().st_size
        if size != entry.get("bytes"):
            errors.append(f"byte-count mismatch: {rel} manifest={entry.get('bytes')} disk={size}")
        digest = sha256_of(target)
        if digest != entry["sha256"]:
            errors.append(f"sha256 mismatch: {rel}")
    for rel in sorted(excluded):
        if (bundle / rel).is_file():
            errors.append(f"path recorded as unavailable but present: {rel}")
        if rel in seen:
            errors.append(f"path both listed and recorded as unavailable: {rel}")
    if require_complete:
        listed = set(seen)
        for prefix in ("data", "figures", "tables"):
            root = bundle / prefix
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if path.name == ".DS_Store":
                    continue
                rel = path.relative_to(bundle).as_posix()
                if rel not in listed and rel not in excluded:
                    errors.append(f"unlisted file under {prefix}/: {rel}")
    notes.append(f"{bundle}: {len(entries)} manifest entries, {len(excluded)} recorded unavailable")
    return errors, notes, seen


def compare(a: Path, b: Path, sha_a: dict[str, str], sha_b: dict[str, str]) -> list[str]:
    lines: list[str] = []
    common = sorted(set(sha_a) & set(sha_b))
    sci, doc = [], []
    for rel in common:
        if sha_a[rel] != sha_b[rel]:
            (sci if rel.startswith(SCIENTIFIC_PREFIXES) else doc).append(rel)
    only_a = sorted(set(sha_a) - set(sha_b))
    only_b = sorted(set(sha_b) - set(sha_a))
    lines.append(f"paths in both bundles: {len(common)}; only in {a}: {len(only_a)}; only in {b}: {len(only_b)}")
    lines.append(f"scientific files differing across bundles: {len(sci)}")
    lines.extend(f"  DIFF(scientific) {rel}" for rel in sci)
    lines.append(f"document files differing across bundles: {len(doc)}")
    lines.extend(f"  diff(document) {rel}" for rel in doc)
    lines.extend(f"  only-in-{a.name}: {rel}" for rel in only_a[:50])
    lines.extend(f"  only-in-{b.name}: {rel}" for rel in only_b[:50])
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--compare", default=None, help="second bundle root to compare against")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail if a file under data/, figures/ or tables/ is not listed")
    parser.add_argument("--fail-on-scientific-diff", action="store_true",
                        help="with --compare: exit non-zero if any data/figures/tables file differs")
    parser.add_argument("--require-shared", nargs="*", default=None,
                        help="with --compare: path prefixes (e.g. data/ figures/) under which every file listed in "
                             "either bundle must be listed in both; a counterpart missing from one bundle is an error")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    errors, notes, sha_a = verify(bundle, args.require_complete)
    print("\n".join(notes))
    status = 0
    if errors:
        status = 1
        print(f"ERRORS in {bundle}: {len(errors)}")
        print("\n".join("  " + e for e in errors))
    else:
        print(f"OK: {bundle} verifies")
    if args.compare:
        other = Path(args.compare).resolve()
        errors_b, notes_b, sha_b = verify(other, args.require_complete)
        print("\n".join(notes_b))
        if errors_b:
            status = 1
            print(f"ERRORS in {other}: {len(errors_b)}")
            print("\n".join("  " + e for e in errors_b))
        else:
            print(f"OK: {other} verifies")
        lines = compare(bundle, other, sha_a, sha_b)
        print("\n".join(lines))
        if args.fail_on_scientific_diff and any(l.startswith("  DIFF(scientific)") for l in lines):
            status = 1
        if args.require_shared is not None:
            prefixes = tuple(args.require_shared) or SCIENTIFIC_PREFIXES
            missing_b = sorted(p for p in sha_a if p.startswith(prefixes) and p not in sha_b)
            missing_a = sorted(p for p in sha_b if p.startswith(prefixes) and p not in sha_a)
            for p in missing_b:
                print(f"  MISSING in {other}: {p}")
            for p in missing_a:
                print(f"  MISSING in {bundle}: {p}")
            if missing_a or missing_b:
                status = 1
            else:
                print(f"shared coverage OK under {', '.join(prefixes)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
