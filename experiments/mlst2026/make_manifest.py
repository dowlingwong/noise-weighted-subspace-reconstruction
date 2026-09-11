"""Output manifest for the MLST 2026-09-10 programme (return transfer).

Lists every artefact in the versioned results tree with its size and SHA-256,
plus the EXP/MS revisions and the per-run provenance summary, so the author can
verify an import without re-running anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mlst2026.provenance import REPO_ROOT, sha256_file, write_json  # noqa: E402

RESULTS = REPO_ROOT / "results" / "mlst_2026-09-10"
MS_REPO = Path("/ceph/dwong/arXiv-v1-polished-extended-preprint")


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                         text=True, check=False)
    return out.stdout.strip()


def main() -> int:
    files = []
    for p in sorted(RESULTS.rglob("*")):
        if p.is_file() and p.name != "OUTPUT_MANIFEST.json":
            files.append({
                "path": str(p.relative_to(REPO_ROOT)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    runs = {}
    for p in sorted(RESULTS.rglob("*.json")):
        if p.name == "OUTPUT_MANIFEST.json":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        pr = d.get("provenance")
        if isinstance(pr, dict):
            runs[str(p.relative_to(REPO_ROOT))] = {
                "run_id": pr.get("run_id"),
                "commit": pr.get("git", {}).get("commit"),
                "tree_dirty": pr.get("git", {}).get("tree_dirty"),
                "wall_seconds": pr.get("resources", {}).get("wall_seconds"),
                "max_rss_mib": pr.get("resources", {}).get("max_rss_mib"),
                "seeds": pr.get("seeds"),
                "n_inputs": len(pr.get("inputs", [])),
            }
    payload = {
        "programme": "MLST revision experiment phase, 2026-09-10",
        "exp_repo": {
            "path": str(REPO_ROOT),
            "branch": _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git(REPO_ROOT, "rev-parse", "HEAD"),
            "baseline": "27220c0ed45b92cc76ab357c25f6fab590614ac9",
            "commits_since_baseline": _git(
                REPO_ROOT, "log", "--oneline",
                "27220c0ed45b92cc76ab357c25f6fab590614ac9..HEAD").splitlines(),
            "tree_clean": not _git(REPO_ROOT, "status", "--porcelain"),
        },
        "ms_repo": {
            "path": str(MS_REPO),
            "head": _git(MS_REPO, "rev-parse", "HEAD"),
            "baseline": "1ad2ac98e366815f9b8a7f6bcf6596c22ff4732c",
            "tree_clean": not _git(MS_REPO, "status", "--porcelain"),
        },
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "runs": runs,
    }
    digest = write_json(RESULTS / "OUTPUT_MANIFEST.json", payload)
    print(f"{len(files)} artefacts, {payload['total_bytes']/1e6:.2f} MB total")
    print(f"wrote {RESULTS}/OUTPUT_MANIFEST.json sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
