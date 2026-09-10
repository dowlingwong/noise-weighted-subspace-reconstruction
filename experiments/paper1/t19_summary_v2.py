"""T19 summary v2 (11 Sep audit) — re-summarise the frozen T19 rows with the v2 `summarise`
(boundary-tolerant eps filter; designed C/F and estimator families reported separately) and
merge the v2 re-computation of the floored S- rows (consistent decoder) as a separate block.
No fits.  Writes summary_v2.json next to the frozen summary.json (which is never overwritten).

Usage::

    python -m experiments.paper1.t19_summary_v2 [--sminus results/mlst_2026-09-11/t19_sminus_v2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments.paper1.t19_coupling import OUT_DIR, summarise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=OUT_DIR / "rows.jsonl")
    ap.add_argument("--sminus", type=Path, default=OUT_DIR.parent / "t19_sminus_v2")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "summary_v2.json")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in args.rows.read_text().splitlines() if l.strip()]
    summ = summarise(rows)
    v1 = json.loads((OUT_DIR / "summary.json").read_text())
    summ["v1_primary"] = v1["primary"]
    summ["v2_note"] = ("frozen rows re-summarised with the v2 filter (eps_op <= 0.2 + 1e-9): the v1 filter dropped the "
                       "nominal-0.2 C/F rows whose computed eps_op was 0.2000000000000001; S- rows below are the v1 (inconsistent-decoder) values")
    if args.sminus.exists():
        rows2 = [json.loads(l) for l in (args.sminus / "rows.jsonl").read_text().splitlines() if l.strip()]
        s2 = summarise(rows2)
        v1_arms = {a: v1["arms"][a] for a in v1["arms"] if a.startswith("S-")}
        summ["S_minus_v2"] = {"source": str(args.sminus), "n_rows": len(rows2), "arms": s2["arms"],
                              "effective": s2["primary"]["S_minus_effective"], "v1_arms_for_comparison": v1_arms,
                              "rows_sha256": hashlib.sha256((args.sminus / "rows.jsonl").read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ["primary"][k] for k in ("n_primary_fits", "designed_CF", "estimator_families", "ratio_cov_hat_in_band_fraction_eps_le_0.2")}, indent=1))
    if "S_minus_v2" in summ:
        print(json.dumps(summ["S_minus_v2"]["effective"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
