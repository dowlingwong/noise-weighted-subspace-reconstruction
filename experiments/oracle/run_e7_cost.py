"""P3-E7 — monitoring cost: hook latency (p50/p95 over >= 30 repetitions), bytes per event per
stage, and the C1 power loss of 10 % event sampling versus full capture (rule 9)."""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import replace

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

from .oracle_cov import families as F
from .oracle_cov import pipeline as P
from .oracle_cov.protocol import DEFAULT, jsonable, out_dir_for, write_manifest

EXP = "P3-E7"


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="dev"); ap.add_argument("--seeds", type=int, nargs="*", default=[0]); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = replace(DEFAULT, mode=a.mode); base = PRIMARY
    if a.smoke:
        from experiments.paper1.common import smoke_config
        base = smoke_config(); cfg = replace(cfg, window_size=40, n_clean_windows=9, mlp_epochs=5)
    out = out_dir_for(EXP + ("_smoke" if a.smoke else ""), cfg)
    res = {"machine": platform.platform(), "repetitions": 30, "stages": {}, "sampling": {}}
    for seed in a.seeds:
        ctx = P.build_seed_context(seed, cfg, base)
        wd = F.make_window(ctx.ts, F.family("clean"), cfg.window_size, seed=999)
        for kind, sub in ctx.subjects.items():
            t_fwd = []
            for _ in range(30):
                t0 = time.perf_counter(); sub.outputs(wd.X); t_fwd.append(time.perf_counter() - t0)
            for stage in P.DEFAULT_STAGES[kind]:
                t_hook, t_score = [], []
                for _ in range(30):
                    t0 = time.perf_counter(); b = P.make_batch(sub, ctx.ts, wd, stage); t_hook.append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); [m.score(b) for m in ctx.monitors[(kind, stage)]]; t_score.append(time.perf_counter() - t0)
                Z = sub.represent(wd.X, stage)
                res["stages"][f"{kind}|{stage}"] = {
                    "forward_p50_ms": 1e3 * float(np.percentile(t_fwd, 50)), "forward_p95_ms": 1e3 * float(np.percentile(t_fwd, 95)),
                    "hook_p50_ms": 1e3 * float(np.percentile(t_hook, 50)), "hook_p95_ms": 1e3 * float(np.percentile(t_hook, 95)),
                    "score_all_monitors_p50_ms": 1e3 * float(np.percentile(t_score, 50)), "score_all_monitors_p95_ms": 1e3 * float(np.percentile(t_score, 95)),
                    "overhead_p95_fraction": float(np.percentile(t_hook, 95) / max(np.percentile(t_fwd, 95), 1e-12)),
                    "bytes_per_event_full": int(Z.shape[1] * 8), "bytes_per_event_sketch_moments": int((Z.shape[1] + Z.shape[1] * (Z.shape[1] + 1) // 2) * 8 / cfg.window_size)}
            # 10 % sampling power loss on one N and one S family at one severity, 6 windows.
            # Thresholds must be recalibrated for the sampled window size: a statistic standardised
            # on 100-event clean windows is not calibrated on 10-event ones.
            n_sub = max(4, cfg.window_size // 10)
            for stage in P.DEFAULT_STAGES[kind]:
                sub_th = {}
                calib = []
                for w in range(cfg.n_clean_windows // 3, 2 * (cfg.n_clean_windows // 3)):
                    wdc = F.make_window(ctx.ts, F.family("clean"), cfg.window_size, seed=seed * 100000 + w)
                    bc = P.make_batch(sub, ctx.ts, wdc, stage)
                    idx = np.random.default_rng(w).choice(len(wdc.X), size=n_sub, replace=False)
                    calib.append(P.RepresentationBatch(stage, bc.Z[idx], bc.outputs[idx], None, {**bc.meta, "X_whitened": bc.meta["X_whitened"][idx]}))
                from .oracle_cov.report import far_threshold
                for m in ctx.monitors[(kind, stage)]:
                    sub_th[m.name] = far_threshold(np.array([m.score(b) for b in calib]), cfg.false_alert_rate)
                for fam in (F.family("N1_cov_slope", 0.5), F.family("S3_t0", 2.0)):
                    full, sub10 = [], []
                    for w in range(6):
                        wdf = F.make_window(ctx.ts, fam, cfg.window_size, seed=seed * 100000 + 80000 + w)
                        b = P.make_batch(sub, ctx.ts, wdf, stage)
                        idx = np.random.default_rng(w).choice(len(wdf.X), size=n_sub, replace=False)
                        bs = P.RepresentationBatch(stage, b.Z[idx], b.outputs[idx], None, {**b.meta, "X_whitened": b.meta["X_whitened"][idx]})
                        for m in ctx.monitors[(kind, stage)]:
                            full.append(m.score(b) > ctx.thresholds[(kind, stage, m.name)]); sub10.append(m.score(bs) > sub_th[m.name])
                    res["sampling"][f"{kind}|{stage}|{fam.name}"] = {"power_full": float(np.mean(full)), "power_10pct": float(np.mean(sub10)), "power_loss_points": 100 * float(np.mean(full) - np.mean(sub10))}
    rc.write_json(out / "summary.json", jsonable(res))
    write_manifest(out, experiment_id=EXP, cfg=cfg, base=base, seeds=a.seeds, families=[F.family("clean")])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
