"""ORACLE-Cov Tier-1 runner: P3-E1 … P3-E6 share one pipeline; the experiment id selects the
family set and the analysis.  ``--mode dev`` writes to results/oracle/<ID>/dev (never citable);
``--mode confirmatory`` refuses to run unless docs/plans/oracle/PREREGISTRATION.md exists and its
hash matches the one recorded in the frozen config.

Usage::

    PYTHONPATH=src:. python -m experiments.oracle.run_tier1 --experiment P3-E1 --mode dev --seeds 0 1 2
    PYTHONPATH=src:. python -m experiments.oracle.run_tier1 --experiment ALL --mode dev --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

from .oracle_cov import families as F
from .oracle_cov import pipeline as P
from .oracle_cov.protocol import DEFAULT, OracleCovConfig, jsonable, out_dir_for, write_manifest

EXPERIMENTS = {
    "P3-E1": {"families": ["N1_cov_slope", "N2_cov_channel"], "windows": 4},
    "P3-E3": {"families": ["S1_amplitude", "S2_decay", "S3_t0", "S4_alpha", "N1_cov_slope", "N2_cov_channel", "N3_jitter", "N4_gain_drift"], "windows": 3},
    "P3-E4": {"families": list(F.SEVERITIES), "windows": 3},
    "P3-E5": {"families": [f for f in F.SEVERITIES if not f.startswith("U")], "windows": 3},
    "P3-E6": {"families": ["N1_cov_slope", "N3_jitter", "S2_decay", "S3_t0"], "windows": 3},
}
# P3-E2 (designed dissociation) and P3-E7 (cost) have their own runners.

PREREG = Path(__file__).resolve().parents[2] / "docs/plans/oracle/PREREGISTRATION.md"


def prereg_hash() -> str | None:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16] if PREREG.exists() else None


def run(exp_id: str, cfg: OracleCovConfig, seeds: list[int], *, base=PRIMARY, smoke: bool = False) -> Path:
    spec = EXPERIMENTS[exp_id]
    out = out_dir_for(exp_id + ("_smoke" if smoke else ""), cfg)
    fams = [F.family(name, sev) for name in spec["families"] for sev in F.SEVERITIES[name]]
    n_windows = 1 if smoke else spec["windows"]
    rows, far_rows = [], []
    for seed in seeds:
        t0 = time.time()
        ctx = P.build_seed_context(seed, cfg, base)
        far_rows += P.clean_far_rows(ctx)
        print(f"[{exp_id}] seed {seed}: context built in {time.time()-t0:.0f}s (train {ctx.training_seconds})", flush=True)
        for fam in fams:
            for w in range(n_windows):
                rows += P.score_window(ctx, cfg, fam, w)
            print(f"[{exp_id}] seed {seed} {fam.key()} done ({len(rows)} rows)", flush=True)
        rc.write_rows_csv(out / "rows.csv", rows, P.COLUMNS)   # checkpoint per seed
    rc.write_rows_csv(out / "rows.csv", rows, P.COLUMNS)
    rc.write_rows_csv(out / "clean_far.csv", far_rows, list(far_rows[0].keys()))
    write_manifest(out, experiment_id=exp_id, cfg=cfg, base=base, seeds=seeds, families=fams,
                   extra={"preregistration_hash": prereg_hash(), "smoke": smoke, "n_windows_per_family": n_windows})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS) + ["ALL"])
    ap.add_argument("--mode", default="dev", choices=["dev", "confirmatory"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = replace(DEFAULT, mode=a.mode)
    base = PRIMARY
    if a.smoke:
        from experiments.paper1.common import smoke_config
        base = smoke_config(); cfg = replace(cfg, window_size=40, n_clean_windows=9, mlp_epochs=10)
    if a.mode == "confirmatory":
        if prereg_hash() is None:
            sys.exit("confirmatory mode refused: docs/plans/oracle/PREREGISTRATION.md does not exist")
        seeds = a.seeds or list(cfg.confirmatory_seeds)
    else:
        seeds = a.seeds or list(cfg.dev_seeds)
    ids = list(EXPERIMENTS) if a.experiment == "ALL" else [a.experiment]
    for exp_id in ids:
        print(f"wrote {run(exp_id, cfg, seeds, base=base, smoke=a.smoke)}")


if __name__ == "__main__":
    main()
