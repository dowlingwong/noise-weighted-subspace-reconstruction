"""P1-E2 — cost of the separability prior: nonseparability sweep.

For alpha in a grid, fit NFPA, CW-PCA, Iso-MPCA and IsoPCA at matched rank
k_c*k_t on the colored regime and record held-out normalized chi^2, raw MSE,
the NFPA <-> CW-PCA mean principal angle and the planted projector errors.
The approximation term is the held-out chi^2 gap NFPA - CW-PCA; the metric
term is NFPA - Iso-MPCA and must keep its sign at every alpha.

Usage::

    python -m experiments.paper1.alpha_sweep --alphas 0 0.05 0.1 0.2 0.4 0.8 --seeds 0 1 ... [--smoke]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory import r9_semantics as sem
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset
from experiments.synthetic.nfpa_confirmatory.metrics import normalized_chi2, raw_mse
from noise_geometry.subspace.angles import principal_angles
from experiments.synthetic.nfpa_confirmatory.methods import reconstruct

from . import common

EXPERIMENT_ID = "P1-E2"
METHODS = ("CW-PCA", "NFPA", "Iso-MPCA", "IsoPCA")
COLUMNS = ["config_hash", "alpha", "seed", "method", "test_normalized_chi2", "test_raw_mse",
           "mean_angle_to_cw_pca_deg", "channel_projector_error", "temporal_projector_error",
           "joint_projector_error_whitened", "runtime_seconds"]


def run(alphas: list[float], seeds: list[int], *, smoke: bool) -> Path:
    base = common.smoke_config() if smoke else PRIMARY
    out = common.out_dir_for(EXPERIMENT_ID, smoke=smoke)
    cfgs = []
    rows: list[dict] = []
    for alpha in alphas:
        cfg = replace(base, nonseparability_alpha=float(alpha))
        cfgs.append(cfg)
        for seed in seeds:
            ds = make_dataset(seed, "colored", cfg)
            fits = common.fit_methods(ds, METHODS)
            cw = fits["CW-PCA"]
            B_cw = sem.fitted_basis_whitened(cw, ds)  # corrected unstacking (E11), not methods.basis_whitened
            for name, fit in fits.items():
                Xhat = reconstruct(fit, ds, ds.X_test)
                errs = sem.planted_errors(fit, ds)
                angles = (np.asarray(principal_angles(sem.fitted_basis_whitened(fit, ds).T, B_cw.T, degrees=True))
                          if name != "CW-PCA" else np.zeros(1))
                rows.append({
                    "config_hash": config_hash(cfg), "alpha": float(alpha), "seed": seed, "method": name,
                    "test_normalized_chi2": normalized_chi2(ds, ds.X_test, Xhat),
                    "test_raw_mse": raw_mse(ds.X_test, Xhat),
                    "mean_angle_to_cw_pca_deg": float(np.mean(angles)),
                    "channel_projector_error": errs.get("channel_projector_error"),
                    "temporal_projector_error": errs.get("temporal_projector_error"),
                    "joint_projector_error_whitened": errs.get("joint_projector_error_whitened"),
                    "runtime_seconds": fit.diag.get("runtime_seconds"),
                })
            print(f"[P1-E2] alpha={alpha} seed={seed} done", flush=True)
        rc.write_rows_csv(out / "alpha_sweep.csv", rows, COLUMNS)  # checkpoint after each alpha
    summary = {"alphas": alphas, "seeds": seeds, "per_alpha": {}}
    for alpha in alphas:
        sub = [r for r in rows if r["alpha"] == float(alpha)]
        summary["per_alpha"][str(alpha)] = {
            "approximation_term_chi2_nfpa_minus_cwpca": rc.paired_contrast(sub, metric="test_normalized_chi2", method_a="NFPA", method_b="CW-PCA"),
            "metric_term_chi2_nfpa_minus_isompca": rc.paired_contrast(sub, metric="test_normalized_chi2", method_a="NFPA", method_b="Iso-MPCA"),
            "metric_term_mse_nfpa_minus_isompca": rc.paired_contrast(sub, metric="test_raw_mse", method_a="NFPA", method_b="Iso-MPCA"),
            "mean_angle_nfpa_to_cwpca_deg": float(np.mean([r["mean_angle_to_cw_pca_deg"] for r in sub if r["method"] == "NFPA"])),
            "mean_angle_isompca_to_cwpca_deg": float(np.mean([r["mean_angle_to_cw_pca_deg"] for r in sub if r["method"] == "Iso-MPCA"])),
            "channel_projector_error_nfpa": float(np.mean([r["channel_projector_error"] for r in sub if r["method"] == "NFPA"])),
            "channel_projector_error_isompca": float(np.mean([r["channel_projector_error"] for r in sub if r["method"] == "Iso-MPCA"])),
        }
    rc.write_json(out / "summary.json", common.as_jsonable(summary))
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=cfgs, seeds=seeds, extra={"methods": list(METHODS)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    print(f"wrote {run(a.alphas, a.seeds, smoke=a.smoke)}")


if __name__ == "__main__":
    main()
