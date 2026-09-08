#!/usr/bin/env python3
"""Maintain ``BUNDLE_MANIFEST.json`` without touching any data file.

Two operations, both additive to the manifest's audit trail:

``--retire PATH [PATH ...] --reason TEXT``
    Remove the entries for paths that are not available in either repository
    and record them under ``excluded_unavailable`` (path -> {reason, retired,
    last_recorded_sha256, last_recorded_bytes}) so that the omission is
    explicit and ``verify_bundle.py`` can check that the path stays absent.
    Nothing is deleted from disk.  Refuses to retire a path whose file exists.

``--refresh PATH [PATH ...]``
    Recompute ``bytes`` and ``sha256`` for listed *document* entries (files at
    the bundle root, or any path you name) after their content was deliberately
    edited.  Refuses to refresh anything under ``data/`` so that archived
    scientific outputs cannot be silently re-hashed; register those through
    ``register_paper1.py`` / ``refresh_bundle.py`` instead.

Usage::

    python transfer_paper/scripts/update_manifest.py --bundle transfer_paper \
        --retire data/study_a/run.log --reason "log not archived at bundle time"
    python transfer_paper/scripts/update_manifest.py --bundle transfer_paper \
        --refresh MANUSCRIPT_EVIDENCE_MAP.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--retire", nargs="*", default=[])
    parser.add_argument("--reason", default=None)
    parser.add_argument("--refresh", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    by_path = {e["path"]: e for e in entries}
    today = _dt.date.today().isoformat()
    changed = False

    if args.retire:
        if not args.reason:
            parser.error("--retire requires --reason")
        excluded = manifest.setdefault("excluded_unavailable", {})
        for rel in args.retire:
            if (bundle / rel).is_file():
                print(f"refusing to retire a path that exists: {rel}")
                return 1
            entry = by_path.get(rel)
            if entry is None:
                print(f"not in manifest (nothing to retire): {rel}")
                continue
            excluded[rel] = {
                "reason": args.reason,
                "retired": today,
                "last_recorded_sha256": entry.get("sha256"),
                "last_recorded_bytes": entry.get("bytes"),
                "source": entry.get("source"),
            }
            entries[:] = [e for e in entries if e["path"] != rel]
            print(f"retired: {rel}")
            changed = True

    for rel in args.refresh:
        if rel.startswith("data/"):
            print(f"refusing to refresh an archived data path: {rel}")
            return 1
        target = bundle / rel
        if not target.is_file():
            print(f"cannot refresh, file missing: {rel}")
            return 1
        entry = by_path.get(rel)
        if entry is None:
            print(f"not in manifest, cannot refresh: {rel}")
            return 1
        new_sha, new_bytes = sha256_of(target), target.stat().st_size
        if entry["sha256"] == new_sha and entry.get("bytes") == new_bytes:
            print(f"unchanged: {rel}")
            continue
        entry["sha256"], entry["bytes"] = new_sha, new_bytes
        print(f"refreshed: {rel} -> {new_bytes} bytes")
        changed = True

    if changed and not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path} ({len(entries)} entries)")
    elif changed:
        print("dry run: manifest not written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
