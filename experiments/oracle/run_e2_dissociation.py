"""P3-E2 — designed alarm/consequence dissociation (cov_B).

At a stage whose reconstruction Jacobian has a null space (mlp_ae ``h1``), displace every
representation by the same norm along (i) the null space and (ii) the leading right-singular
direction.  Prediction, by construction: pullback-whitened and Jacobian-projected alarms are
~0 for ``null`` and large for ``aligned``; Euclidean displacement cannot separate them; the
amplitude consequence follows the alarm of the metric-aware monitors.  The linear subject at
its code stage has no null space and is reported as such.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY

from .oracle_cov import families as F
from .oracle_cov import pipeline as P
from .oracle_cov.consequence import amplitude_rmse
from .oracle_cov.protocol import DEFAULT, Family, RepresentationBatch, jsonable, out_dir_for, write_manifest

EXP = "P3-E2"


def displaced_outputs(sub, stage, Z2):
    acts = [sub._forward_from(stage, z[None])[3 - sub.STAGES[stage]][0] for z in Z2]  # code from displaced stage
    code = np.array(acts)
    return code @ sub.readout_w[:-1] + sub.readout_w[-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev"); ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--norms", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0]); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = replace(DEFAULT, mode=a.mode); base = PRIMARY
    if a.smoke:
        from experiments.paper1.common import smoke_config
        base = smoke_config(); cfg = replace(cfg, window_size=40, n_clean_windows=9, mlp_epochs=10)
    seeds = a.seeds or list(cfg.dev_seeds)
    out = out_dir_for(EXP + ("_smoke" if a.smoke else ""), cfg)
    rows = []
    stage = "h1"
    for seed in seeds:
        ctx = P.build_seed_context(seed, cfg, base, subjects=("mlp_ae",), stages={"mlp_ae": [stage]})
        sub = ctx.subjects["mlp_ae"]
        mons = ctx.monitors[("mlp_ae", stage)]
        for w in range(3 if not a.smoke else 1):
            wd = F.make_window(ctx.ts, F.family("clean"), cfg.window_size, seed=seed * 100000 + 70000 + w)
            b0 = P.make_batch(sub, ctx.ts, wd, stage)
            base_rmse = amplitude_rmse(b0.outputs, wd.latents["amplitude"])
            rng = np.random.default_rng(seed * 7 + w)
            for norm in a.norms:
                for kind in ("null", "aligned", "null_amp", "aligned_amp"):
                    Z2 = F.designed_perturbation(sub, stage, b0.Z, kind, norm, rng)
                    out2 = displaced_outputs(sub, stage, Z2)
                    b = RepresentationBatch(stage=stage, Z=Z2, outputs=out2, recon=None,
                                            meta={"chi2_bar": b0.meta["chi2_bar"], "X_whitened": b0.meta["X_whitened"], "truth_amplitude": wd.latents["amplitude"]})
                    K = amplitude_rmse(out2, wd.latents["amplitude"]) / max(base_rmse, 1e-12) - 1.0
                    # reconstruction consequence: whitened error of the displaced reconstruction against the clean signal
                    rec2 = np.array([sub._forward_from(stage, z[None])[-1][0] for z in Z2])
                    clean_w = ctx.ts.whiten(wd.clean + ctx.ts.ds.train_mean).reshape(len(wd.X), -1, order="F")
                    K_rec = float(np.mean((rec2 - clean_w) ** 2) / max(np.mean((b0.recon - clean_w) ** 2), 1e-12) - 1.0)
                    fam = Family(kind, "designed", True, norm)
                    for m in mons:
                        if m.name in ("input_ks_bh", "output_residual"):
                            continue   # input-space statistics are untouched by a representation-level displacement
                        rows.append({"seed": seed, "window": w, "family": kind, "norm": norm, "monitor": m.name, "score": m.score(b),
                                     "threshold": ctx.thresholds[("mlp_ae", stage, m.name)], "alarm": bool(m.score(b) > ctx.thresholds[("mlp_ae", stage, m.name)]),
                                     "is_analytic": m.is_analytic_for(fam), "K": K, "K_recon": K_rec, "high_consequence": bool(K >= cfg.kappa_m)})
            print(f"[{EXP}] seed {seed} window {w} done", flush=True)
    cols = list(rows[0].keys())
    rc.write_rows_csv(out / "rows.csv", rows, cols)
    # summary: per monitor, mean score null vs aligned and the predicted sign
    summary = {}
    predicted = {("pullback_whitened", "recon"): "aligned >> null", ("jacobian_projected", "recon"): "aligned >> null",
                 ("pullback_output", "amp"): "aligned_amp >> null_amp", ("euclidean", "recon"): "cannot separate", ("euclidean", "amp"): "cannot separate"}
    for pair, (nk, ak) in {"recon": ("null", "aligned"), "amp": ("null_amp", "aligned_amp")}.items():
        for name in sorted({r["monitor"] for r in rows}):
            sn = [r["score"] for r in rows if r["monitor"] == name and r["family"] == nk]
            sa = [r["score"] for r in rows if r["monitor"] == name and r["family"] == ak]
            summary[f"{pair}:{name}"] = {"mean_null": float(np.mean(sn)), "mean_aligned": float(np.mean(sa)),
                                         "aligned_gt_null_fraction": float(np.mean([a_ > n_ for a_, n_ in zip(sa, sn)])),
                                         "predicted": predicted.get((name, pair), "no prediction")}
        summary[f"{pair}:consequence"] = {
            "mean_K_amp_null": float(np.mean([r["K"] for r in rows if r["family"] == nk and r["monitor"] == "euclidean"])),
            "mean_K_amp_aligned": float(np.mean([r["K"] for r in rows if r["family"] == ak and r["monitor"] == "euclidean"])),
            "mean_K_recon_null": float(np.mean([r["K_recon"] for r in rows if r["family"] == nk and r["monitor"] == "euclidean"])),
            "mean_K_recon_aligned": float(np.mean([r["K_recon"] for r in rows if r["family"] == ak and r["monitor"] == "euclidean"]))}
    summary["linear_subject_note"] = "lin_cw code stage: reconstruction Jacobian P has full column rank; no null space exists, so the dissociation is not constructible there (reported, not run)."
    rc.write_json(out / "summary.json", jsonable(summary))
    write_manifest(out, experiment_id=EXP, cfg=cfg, base=base, seeds=seeds, families=[Family(k, "designed", True) for k in ("null", "aligned", "null_amp", "aligned_amp")], extra={"norms": a.norms, "stage": stage})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
