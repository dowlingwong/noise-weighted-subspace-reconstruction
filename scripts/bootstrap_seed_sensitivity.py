#!/usr/bin/env python3
"""Bootstrap-seed sensitivity audit for the September 2026 Paper 1 contrasts.

The predeclarations of P1-E2, P1-E9 and P1-E11 name bootstrap seed 20260808;
the archived intervals were produced by ``r9_common.bootstrap_paired`` at its
module constant ``BOOTSTRAP_SEED = 20260819``.  This script recomputes every
archived seed-level paired interval from the archived ``per_seed_differences``
with the *same* routine at both seeds, checks that the 20260819 recomputation
reproduces the archived interval exactly, and reports how far the 20260808
interval ends move and whether any ``excludes_zero`` verdict or predeclared
gate changes.  Archived files are read only.

Usage::

    python scripts/bootstrap_seed_sensitivity.py --results results/paper1 \
        --out results/paper1/BOOTSTRAP_SEED_SENSITIVITY_2026-09-08
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.synthetic.nfpa_confirmatory.r9_common import bootstrap_paired  # noqa: E402

PREDECLARED_SEED = 20260808
ARCHIVED_SEED = 20260819


def walk(obj, path=""):
    """Yield (path, dict) for every archived bootstrap block."""
    if isinstance(obj, dict):
        if "per_seed_differences" in obj and "ci95_low" in obj and "bootstrap_seed" in obj:
            yield path, obj
        for k, v in obj.items():
            yield from walk(v, f"{path}/{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results" / "paper1"))
    ap.add_argument("--out", default=str(ROOT / "results" / "paper1" / "BOOTSTRAP_SEED_SENSITIVITY_2026-09-08"))
    ap.add_argument("--ids", nargs="*", default=["P1-E2", "P1-E9", "P1-E11"])
    args = ap.parse_args()
    rows = []
    for exp in args.ids:
        summary = Path(args.results) / exp / "summary.json"
        data = json.loads(summary.read_text())
        for path, block in walk(data):
            diffs = block["per_seed_differences"]
            n = int(block["bootstrap_resamples"])
            rec_arch = bootstrap_paired(diffs, resamples=n, seed=ARCHIVED_SEED)
            rec_pre = bootstrap_paired(diffs, resamples=n, seed=PREDECLARED_SEED)
            reproduces = (rec_arch["ci95_low"] == block["ci95_low"] and rec_arch["ci95_high"] == block["ci95_high"]
                          and int(block["bootstrap_seed"]) == ARCHIVED_SEED)
            rows.append({
                "experiment": exp, "block": path, "n_seeds": len(diffs), "resamples": n,
                "archived_seed": int(block["bootstrap_seed"]),
                "archived_ci": [block["ci95_low"], block["ci95_high"]],
                "recomputed_ci_20260819": [rec_arch["ci95_low"], rec_arch["ci95_high"]],
                "recomputed_ci_20260808": [rec_pre["ci95_low"], rec_pre["ci95_high"]],
                "reproduces_archived_exactly": bool(reproduces),
                "abs_shift_low": abs(rec_pre["ci95_low"] - rec_arch["ci95_low"]),
                "abs_shift_high": abs(rec_pre["ci95_high"] - rec_arch["ci95_high"]),
                "rel_shift_max_vs_halfwidth": max(abs(rec_pre["ci95_low"] - rec_arch["ci95_low"]),
                                                  abs(rec_pre["ci95_high"] - rec_arch["ci95_high"]))
                                              / max(1e-300, 0.5 * (rec_arch["ci95_high"] - rec_arch["ci95_low"])),
                "excludes_zero_archived": bool(block["excludes_zero"]),
                "excludes_zero_20260808": bool(rec_pre["excludes_zero"]),
                "verdict_changes": bool(block["excludes_zero"]) != bool(rec_pre["excludes_zero"]),
            })
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(rows, indent=1))
    n_rep = sum(r["reproduces_archived_exactly"] for r in rows)
    n_change = sum(r["verdict_changes"] for r in rows)
    max_rel = max(r["rel_shift_max_vs_halfwidth"] for r in rows)
    max_abs = max(max(r["abs_shift_low"], r["abs_shift_high"]) for r in rows)
    md = [f"# Bootstrap-seed sensitivity audit ({', '.join(args.ids)})", "",
          f"Archived blocks recomputed: {len(rows)}.  Exact reproduction of the archived interval at seed {ARCHIVED_SEED}: {n_rep}/{len(rows)}.",
          f"Verdict (`excludes_zero`) changes at the predeclared seed {PREDECLARED_SEED}: {n_change}.",
          f"Largest absolute shift of an interval end between the two seeds: {max_abs:.3e}; largest shift relative to the interval half-width: {100*max_rel:.1f} %.", "",
          "| experiment | block | archived CI (20260819) | CI at 20260808 | max rel. shift | verdict change |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['experiment']} | `{r['block']}` | [{r['archived_ci'][0]:+.5g}, {r['archived_ci'][1]:+.5g}] | [{r['recomputed_ci_20260808'][0]:+.5g}, {r['recomputed_ci_20260808'][1]:+.5g}] | {100*r['rel_shift_max_vs_halfwidth']:.1f} % | {'YES' if r['verdict_changes'] else 'no'} |")
    (out / "SENSITIVITY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:5]))
    return 0 if n_rep == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
