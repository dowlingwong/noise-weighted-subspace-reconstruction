"""Add a reconstruction-based consequence ``K_recon`` to an existing Tier-1 row table.

K_recon = E|recon_whitened - clean_whitened|^2 on the window / the same quantity on the clean
reference windows - 1.  The amplitude readout at SNR 10 is noise-dominated (clean RMSE ~ the
latent spread), so its relative degradation is insensitive to most families; the whitened
reconstruction error against the clean signal is the sensitive physics-space consequence and
is the one the ORACLE proposal's Tier-1 "correct likelihood-weighted residual" refers to.
Windows are regenerated deterministically from (seed, family, severity, window).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

from .oracle_cov import families as F
from .oracle_cov import pipeline as P
from .oracle_cov.protocol import DEFAULT, RESULTS_ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", required=True); ap.add_argument("--mode", default="dev"); ap.add_argument("--kappa-m", type=float, default=0.10)
    a = ap.parse_args(argv)
    d = RESULTS_ROOT / a.experiment / a.mode
    with (d / "rows.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    cfg = replace(DEFAULT, mode=a.mode)
    cells = sorted({(int(r["seed"]), r["family"], None if r["severity"] in ("", "None") else float(r["severity"]), int(r["window"])) for r in rows})
    krec = {}
    for seed in sorted({c[0] for c in cells}):
        ctx = P.build_seed_context(seed, cfg, PRIMARY, stages={"lin_cw": ["code"], "mlp_ae": ["code"]})
        base = {}
        for kind, sub in ctx.subjects.items():
            errs = []
            for w in range(cfg.n_clean_windows // 3):
                wd = F.make_window(ctx.ts, F.family("clean"), cfg.window_size, seed=seed * 100000 + w)
                cw = ctx.ts.whiten(wd.clean + ctx.ts.ds.train_mean).reshape(len(wd.X), -1, order="F")
                errs.append(float(np.mean((sub.recon_whitened(wd.X) - cw) ** 2)))
            base[kind] = float(np.mean(errs))
        for (s, fam, sev, w) in cells:
            if s != seed:
                continue
            wd = F.make_window(ctx.ts, F.family(fam, sev), cfg.window_size, seed=seed * 100000 + 50000 + w)
            cw = ctx.ts.whiten(wd.clean + ctx.ts.ds.train_mean).reshape(len(wd.X), -1, order="F")
            for kind, sub in ctx.subjects.items():
                krec[(s, kind, fam, sev, w)] = float(np.mean((sub.recon_whitened(wd.X) - cw) ** 2)) / base[kind] - 1.0
        print(f"[augment] seed {seed} done", flush=True)
    for r in rows:
        key = (int(r["seed"]), r["subject"], r["family"], None if r["severity"] in ("", "None") else float(r["severity"]), int(r["window"]))
        r["K_recon"] = krec[key]; r["high_consequence_recon"] = bool(krec[key] >= a.kappa_m)
    cols = list(rows[0].keys())
    rc.write_rows_csv(d / "rows.csv", rows, cols)
    print(f"augmented {len(rows)} rows in {d / 'rows.csv'}")


if __name__ == "__main__":
    main()
