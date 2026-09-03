"""P3-E0b / claim C0 — physical-variable probes on frozen Tier-1 subjects.

For each dev seed and subject (``lin_cw``, ``mlp_ae``), fit ridge probes from every hooked
stage to each physical latent (amplitude, rise_time, decay_time, secondary; t0_shift is
unexcited and enters only through the S3 family) on clean training windows, then score the
*frozen* probes on held-out clean windows and on every declared family x severity window.
Per (seed, subject, stage, family, severity, variable) the row records

* ``r2_frozen``   -- R^2 of the clean-fitted probe on the window (does the physics still read
                     out through the axes it used to?);
* ``r2_refit``    -- 2-fold cross-validated R^2 of a probe refit on the window itself (is the
                     information still there at all, on any axes?);
* ``axis_cos``    -- cosine between the clean probe direction and the refit direction (did the
                     axis move?);
* ``participation`` -- effective number of representation axes the variable occupies;
* ``rmse_ratio``  -- RMSE of the clean-fitted probe on the window divided by its RMSE on held-out
                     clean windows (added after the 2026-09-03 dev run: R^2 is ill-defined on S
                     windows that narrow a variable's range, RMSE is not).

Pre-registered predictions (docs/plans/oracle/PREREGISTRATION.md, C0):
  (1) clean: r2 high for excited variables at the code stage; ~0 for t0_shift;
  (2) N families (covariance type): r2_frozen drops, r2_refit drops less, axis_cos stays high
      -- the physics is still there, the noise has moved the readout;
  (3) S families along excited coordinates (S1, S2): r2_frozen stays high (the axis exists);
      S3_t0: t0_shift r2_refit stays ~0 at the code stage (never excited; no axis to find);
  (4) mlp_ae early stages (h1) carry the variables more diffusely (participation larger).

Usage::

    PYTHONPATH=src:. python -m experiments.oracle.run_e0b_probes --mode dev --seeds 0 1 2 [--smoke]
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

from .oracle_cov import families as F
from .oracle_cov.generator import make_training_set
from .oracle_cov.protocol import DEFAULT, OracleCovConfig, out_dir_for, write_manifest
from .oracle_cov.subjects import make_subject
from .run_tier1 import prereg_hash

EXP_ID = "P3-E0b"
VARIABLES = ("amplitude", "rise_time", "decay_time", "secondary", "t0_shift")
STAGES = {"lin_cw": ["code"], "mlp_ae": ["h1", "h2", "code"]}
FAMILIES = ["N1_cov_slope", "N2_cov_channel", "N3_jitter", "N4_gain_drift", "N6_channel_loss",
            "S1_amplitude", "S2_decay", "S3_t0", "S4_alpha"]
COLUMNS = ["seed", "subject", "stage", "family", "contract", "severity", "window", "variable",
           "r2_frozen", "r2_refit", "axis_cos", "participation", "rmse_ratio", "n"]


class Probe:
    def __init__(self, ridge: float = 1e-2):
        self.ridge = ridge          # relative to n: G = Z'Z + ridge * n * I

    def fit(self, R: np.ndarray, y: np.ndarray) -> "Probe":
        self.mu, self.sd = R.mean(0), R.std(0) + 1e-12
        Z = (R - self.mu) / self.sd
        self.ym = float(y.mean())
        G = Z.T @ Z + self.ridge * len(y) * np.eye(Z.shape[1])
        self.w = np.linalg.solve(G, Z.T @ (y - self.ym))
        return self

    def predict(self, R: np.ndarray) -> np.ndarray:
        return ((R - self.mu) / self.sd) @ self.w + self.ym

    def r2(self, R: np.ndarray, y: np.ndarray) -> float:
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            return float("nan")
        return 1.0 - float(np.sum((y - self.predict(R)) ** 2)) / ss_tot

    def rmse(self, R: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self.predict(R)) ** 2)))

    def direction(self) -> np.ndarray:
        n = np.linalg.norm(self.w)
        return self.w / n if n > 0 else self.w


def cv_r2(R: np.ndarray, y: np.ndarray, folds: int = 2) -> float:
    """Cross-validated R^2 of a probe refit on the window (guards against n < dim overfit)."""
    n = len(y)
    idx = np.arange(n)
    pred = np.empty(n)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        pred[test] = Probe().fit(R[train], y[train]).predict(R[test])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")


def participation(R: np.ndarray, y: np.ndarray) -> float:
    if y.std() == 0:
        return float("nan")
    c = np.array([np.corrcoef(R[:, j], y)[0, 1] if R[:, j].std() > 0 else 0.0 for j in range(R.shape[1])])
    c2 = np.nan_to_num(c) ** 2
    if c2.sum() == 0:
        return float("nan")
    wj = c2 / c2.sum()
    return float(1.0 / np.sum(wj ** 2))


def run(cfg: OracleCovConfig, seeds: list[int], *, base=PRIMARY, smoke: bool) -> Path:
    out = out_dir_for(EXP_ID + ("_smoke" if smoke else ""), cfg)
    fams = [F.family(n, s) for n in FAMILIES for s in F.SEVERITIES[n]]
    n_windows = 1 if smoke else 3
    rows: list[dict] = []
    for seed in seeds:
        t0 = time.time()
        ts = make_training_set(seed, base, alpha=cfg.base_alpha)
        subjects = {k: make_subject(k, ts, rank=cfg.subject_rank, hidden=cfg.mlp_hidden,
                                    epochs=cfg.mlp_epochs, lr=cfg.mlp_lr, seed=seed) for k in STAGES}
        n_fit = max(3, cfg.n_clean_windows // 3)
        fit_w = [F.make_window(ts, F.family("clean"), cfg.window_size, seed=seed * 100000 + i) for i in range(n_fit)]
        held_w = [F.make_window(ts, F.family("clean"), cfg.window_size, seed=seed * 100000 + n_fit + i) for i in range(n_windows)]
        X_fit = np.concatenate([w.X for w in fit_w])
        lat_fit = {v: np.concatenate([w.latents.get(v, np.zeros(len(w.X))) for w in fit_w]) for v in VARIABLES}
        print(f"[{EXP_ID}] seed {seed}: subjects trained in {time.time()-t0:.0f}s", flush=True)
        for kind, sub in subjects.items():
            for stage in STAGES[kind]:
                R_fit = sub.represent(X_fit, stage)
                probes = {v: Probe().fit(R_fit, lat_fit[v]) if lat_fit[v].std() > 0 else None for v in VARIABLES}
                clean_rmse = {}
                for v, pf in probes.items():
                    if pf is None:
                        continue
                    Rh = np.concatenate([sub.represent(w.X, stage) for w in held_w])
                    yh = np.concatenate([np.asarray(w.latents.get(v, np.zeros(len(w.X))), float) for w in held_w])
                    clean_rmse[v] = pf.rmse(Rh, yh)
                targets = [(F.family("clean"), i, w) for i, w in enumerate(held_w)]
                targets += [(fam, i, F.make_window(ts, fam, cfg.window_size, seed=seed * 100000 + 500 + 97 * (zlib.crc32(fam.name.encode()) % 1000) + i))
                            for fam in fams for i in range(n_windows)]
                for fam, wi, w in targets:
                    R = sub.represent(w.X, stage)
                    for v in VARIABLES:
                        y = np.asarray(w.latents.get(v, np.zeros(len(w.X))), float)
                        if y.std() == 0:
                            continue
                        refit = Probe().fit(R, y)
                        pf = probes[v]
                        rows.append({
                            "seed": seed, "subject": kind, "stage": stage, "family": fam.name,
                            "contract": fam.contract, "severity": fam.severity, "window": wi, "variable": v,
                            "r2_frozen": pf.r2(R, y) if pf is not None else float("nan"),
                            "r2_refit": cv_r2(R, y),
                            "axis_cos": float(abs(pf.direction() @ refit.direction())) if pf is not None else float("nan"),
                            "participation": participation(R, y),
                            "rmse_ratio": (pf.rmse(R, y) / clean_rmse[v]) if (pf is not None and clean_rmse.get(v, 0) > 0) else float("nan"),
                            "n": len(y),
                        })
                print(f"[{EXP_ID}] seed {seed} {kind}/{stage} done ({len(rows)} rows)", flush=True)
        rc.write_rows_csv(out / "rows.csv", rows, COLUMNS)
    rc.write_rows_csv(out / "rows.csv", rows, COLUMNS)
    write_manifest(out, experiment_id=EXP_ID, cfg=cfg, base=base, seeds=seeds, families=fams,
                   extra={"preregistration_hash": prereg_hash(), "smoke": smoke, "variables": list(VARIABLES),
                          "stages": STAGES, "n_windows_per_family": n_windows})
    return out


def summarise(out: Path) -> None:
    import csv
    from collections import defaultdict
    rows = list(csv.DictReader((out / "rows.csv").open()))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["subject"], r["stage"], r["contract"], r["variable"])].append(
            (float(r["r2_frozen"]), float(r["r2_refit"]), float(r["axis_cos"]), float(r["participation"]),
             float(r.get("rmse_ratio", "nan"))))
    lines = ["subject stage contract variable | r2_frozen r2_refit axis_cos participation rmse_ratio (means)"]
    for k in sorted(agg):
        a = np.array(agg[k], float)
        m = np.nanmean(a, axis=0)
        lines.append(f"{k[0]:7s} {k[1]:5s} {k[2]:5s} {k[3]:10s} | {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f} {m[3]:5.2f} {m[4]:6.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "confirmatory"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = replace(DEFAULT, mode=a.mode)
    base = PRIMARY
    if a.smoke:
        from experiments.paper1.common import smoke_config
        base = smoke_config(); cfg = replace(cfg, window_size=40, n_clean_windows=9, mlp_epochs=10)
    if a.mode == "confirmatory" and prereg_hash() is None:
        sys.exit("confirmatory mode refused: PREREGISTRATION.md missing")
    seeds = a.seeds or list(cfg.dev_seeds)
    out = run(cfg, seeds, base=base, smoke=a.smoke)
    summarise(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
