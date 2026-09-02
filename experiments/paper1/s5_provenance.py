"""P1-E5 — S5 provenance audit (closes OPEN_ISSUES.md #0 of the manuscript repo).

Three checks, one report:

1. **Number sweep.** Every value the manuscript's synthetic section cites for
   S5 (current text, and the superseded values recorded in OPEN_ISSUES #0) is
   searched, with a relative tolerance, in every ``S*.json`` / ``S*.csv`` under
   ``results/``, ``evidence/`` and ``transfer_paper/``.  The report lists for
   each number the files (and keys) that contain it.
2. **Digest check.** The checksummed copy of every S1–S9 ten-seed sweep in
   ``transfer_paper/data/synthetic`` is compared byte-for-byte with the copy
   under ``results/sweeps``; the 30-seed S5 confirmatory JSON is located and
   its SHA-256 recorded.
3. **Reproduction (optional, ``--rerun``).** The predeclared 30-seed S5
   protocol (``scripts/run_s5_confirmatory_sweep.py``: seeds 1000–1029,
   rank 2, test_frac 0.5, split_seed 0) is re-executed into
   ``results/paper1/P1-E5/rerun/`` and its summary means compared with the
   archived confirmatory file.

Usage::  python -m experiments.paper1.s5_provenance [--rerun] [--tol 5e-4]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

from . import common

EXPERIMENT_ID = "P1-E5"
ROOT = common.REPO_ROOT
ARCHIVED_30 = ROOT / "evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json"

# Values currently printed in 07_experiments.tex (§7.1, S5 paragraph) and the
# superseded values recorded in OPEN_ISSUES.md #0.
MANUSCRIPT_CURRENT = {
    "pca_weighted_residual": 48.419, "empca_weighted_residual": 47.869, "paired_difference": 0.550,
    "ci_low": 0.530, "ci_high": 0.571, "fractional_improvement_pct": 1.14,
    "nll_difference": 11.67, "nll_ci_low": 11.24, "nll_ci_high": 12.11, "pca_nll_mean": 1026.92,
    "pca_raw_mse": 2330.27, "empca_raw_mse": 7412.67, "max_angle_deg": 86.25, "angle_ci_low": 85.16, "angle_ci_high": 87.21,
}
MANUSCRIPT_SUPERSEDED = {
    "pca_weighted_residual": 48.418, "empca_weighted_residual": 47.878, "paired_difference": 0.540,
    "ci_low": 0.523, "ci_high": 0.560, "pca_raw_mse": 2367.99, "empca_raw_mse": 5388.75, "max_angle_deg": 86.07,
}


def _walk_numbers(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_numbers(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)
    elif isinstance(obj, str):
        try:
            yield prefix, float(obj)
        except ValueError:
            return


def candidate_files() -> list[Path]:
    files = []
    for base in ("results", "evidence", "transfer_paper"):
        for p in (ROOT / base).rglob("S*"):
            if p.is_file() and p.suffix in (".json", ".csv") and re.match(r"S\d", p.name):
                files.append(p)
    return sorted(set(files))


def load_numbers(path: Path):
    try:
        if path.suffix == ".json":
            yield from _walk_numbers(json.loads(path.read_text()))
        else:
            with path.open() as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    for k, v in row.items():
                        try:
                            yield f"row{i}.{k}", float(v)
                        except (TypeError, ValueError):
                            continue
    except Exception as exc:  # noqa: BLE001 — a corrupt file is a finding, not a crash
        yield f"__error__:{exc}", float("nan")


def number_sweep(targets: dict[str, float], tol: float) -> dict[str, list[dict]]:
    hits = {name: [] for name in targets}
    for path in candidate_files():
        for key, value in load_numbers(path):
            for name, target in targets.items():
                if abs(value - target) <= tol * max(abs(target), 1.0):
                    hits[name].append({"file": str(path.relative_to(ROOT)), "key": key, "value": value})
    return hits


def digest_check() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "transfer_paper/data/synthetic").glob("S*_sweep_10seeds.*")):
        twin = ROOT / "results/sweeps" / p.name
        rows.append({"bundle_file": str(p.relative_to(ROOT)), "bundle_sha256": rc.sha256_of(p),
                     "results_twin_exists": twin.exists(),
                     "results_twin_sha256": rc.sha256_of(twin) if twin.exists() else None,
                     "identical": twin.exists() and rc.sha256_of(twin) == rc.sha256_of(p)})
    return rows


def rerun_30_seeds(out_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_s5_confirmatory_sweep as s5  # noqa: E402

    t0 = time.time()
    result = s5.run_s5_confirmatory()
    result["runtime_seconds"] = time.time() - t0
    rc.write_json(out_dir / "S5_confirmatory_30seeds_rerun.json", common.as_jsonable(result))
    return result


def compare_with_archive(rerun: dict) -> dict:
    archived = json.loads(ARCHIVED_30.read_text())
    keys = ["pca_weighted_residual_to_observed", "weighted_pca_weighted_residual_to_observed",
            "pca_raw_residual_to_observed", "weighted_pca_raw_residual_to_observed", "pca_nll_mean",
            "subspace_angle_deg_max", "pca_over_empca_weighted_residual_ratio"]
    out = {}
    for k in keys:
        a = archived["summary"].get(k, {}).get("mean")
        b = rerun["summary"].get(k, {}).get("mean")
        out[k] = {"archived_mean": a, "rerun_mean": b,
                  "rel_diff": None if a in (None, 0) or b is None else abs(a - b) / abs(a)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="re-execute the 30-seed S5 protocol (slow)")
    ap.add_argument("--tol", type=float, default=5e-4, help="relative tolerance for the number sweep")
    a = ap.parse_args(argv)
    out = common.out_dir_for(EXPERIMENT_ID)
    report = {
        "archived_30seed_file": str(ARCHIVED_30.relative_to(ROOT)) if ARCHIVED_30.exists() else None,
        "archived_30seed_sha256": rc.sha256_of(ARCHIVED_30) if ARCHIVED_30.exists() else None,
        "number_sweep_current": number_sweep(MANUSCRIPT_CURRENT, a.tol),
        "number_sweep_superseded": number_sweep(MANUSCRIPT_SUPERSEDED, a.tol),
        "digest_check_S1_S9": digest_check(),
        "files_searched": len(candidate_files()),
        "tolerance_relative": a.tol,
    }
    if a.rerun:
        (out / "rerun").mkdir(exist_ok=True)
        rerun = rerun_30_seeds(out / "rerun")
        report["rerun_vs_archive"] = compare_with_archive(rerun)
        report["rerun_runtime_seconds"] = rerun["runtime_seconds"]
    rc.write_json(out / "provenance_report.json", common.as_jsonable(report))
    write_markdown(out / "PROVENANCE_REPORT.md", report)
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=[PRIMARY], seeds=list(range(1000, 1030)) if a.rerun else [])
    print(f"wrote {out}")


def write_markdown(path: Path, r: dict) -> None:
    L = [f"# S5 provenance report ({time.strftime('%Y-%m-%d')})", "",
         f"Archived 30-seed confirmatory file: `{r['archived_30seed_file']}` (sha256 `{r['archived_30seed_sha256']}`).",
         f"Files searched: {r['files_searched']}; relative tolerance {r['tolerance_relative']}.", "",
         "## 1. Current manuscript values — where each is found", "", "| quantity | value | found in |", "|---|---|---|"]
    for name, v in MANUSCRIPT_CURRENT.items():
        hits = r["number_sweep_current"][name]
        L.append(f"| {name} | {v} | " + ("; ".join(f"`{h['file']}`:{h['key']}" for h in hits[:4]) + (f" (+{len(hits)-4} more)" if len(hits) > 4 else "") if hits else "**not found**") + " |")
    L += ["", "## 2. Superseded values from OPEN_ISSUES #0 — where each is found", "", "| quantity | value | found in |", "|---|---|---|"]
    for name, v in MANUSCRIPT_SUPERSEDED.items():
        hits = r["number_sweep_superseded"][name]
        L.append(f"| {name} | {v} | " + ("; ".join(f"`{h['file']}`:{h['key']}" for h in hits[:4]) if hits else "**not found**") + " |")
    L += ["", "## 3. S1–S9 ten-seed sweeps: bundle vs results/sweeps", "", "| file | identical | bundle sha256 |", "|---|---|---|"]
    for d in r["digest_check_S1_S9"]:
        L.append(f"| `{d['bundle_file']}` | {'yes' if d['identical'] else ('no twin' if not d['results_twin_exists'] else '**DIFFERS**')} | `{d['bundle_sha256'][:16]}…` |")
    if "rerun_vs_archive" in r:
        L += ["", f"## 4. Re-execution of the 30-seed protocol ({r['rerun_runtime_seconds']:.0f} s)", "", "| summary mean | archived | rerun | rel. diff |", "|---|---|---|---|"]
        for k, d in r["rerun_vs_archive"].items():
            L.append(f"| {k} | {d['archived_mean']} | {d['rerun_mean']} | {d['rel_diff']} |")
    else:
        L += ["", "## 4. Re-execution", "", "Not run (`--rerun` not given)."]
    L += ["", "## Verdict", "",
          "Fill in after reading sections 1–4: (a) every current manuscript value traces to the archived 30-seed file → OPEN_ISSUES #0 closes as *substitution confirmed*; "
          "(b) the superseded values trace to another archived run → cite that run instead; (c) neither → the numbers are unrecoverable and must be regenerated from section 4."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
