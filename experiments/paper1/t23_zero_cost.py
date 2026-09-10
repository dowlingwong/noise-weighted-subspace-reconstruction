"""T23 — zero-cost checks on the planted configuration (Part II protocol).

(a) Per-axis whitened eigengap of the frozen PRIMARY configuration (C = 8, T = 256,
    (k_c, k_t) = (2, 3), Brownian temporal noise, AR(1) channel correlation), from a
    large clean sample and the oracle covariance — no fits.  Registered prediction:
    the channel gap is much larger than the temporal gap, which would make the
    R/PS channel-good (0.035) / temporal-weak (0.768) recovery asymmetry a predicted
    consequence of the eps/Delta term.

(b) The quantities of Rem. 5.2 at the R/PS estimated-covariance point: a separable
    covariance estimated from N_cal = 2048 noise-only events with the frozen
    calibrated estimator (E10b), the relative operator error
    eps = ||Sigma^{-1/2}(Sigma_hat - Sigma)Sigma^{-1/2}||_op, lambda_1/Delta_lambda of
    the whitened population operator at rank 6, the bound 6 eps lambda_1/Delta_lambda,
    and whether eps <= 1/2 (in-domain), over ten calibration seeds.

Usage::

    python -m experiments.paper1.t23_zero_cost
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import generator as gen
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.covariance import full_covariance, whiten_separable
from experiments.synthetic.nfpa_confirmatory.covariance_estimation_v2 import estimate_calibrated_axis_covariances

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "mlst_2026-09-11" / "t23"
N_CLEAN = 20000
N_CAL = 2048
CAL_SEEDS = list(range(10))


def _sym_isqrt(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    return (vecs / np.sqrt(vals)[None, :]) @ vecs.T


def whitened_clean_sample(alpha: float, seed: int = 20260911, n: int = N_CLEAN):
    cfg = replace(PRIMARY, nonseparability_alpha=alpha)
    rng = np.random.default_rng(seed)
    V = gen.make_channel_basis(cfg); Tb = gen.make_temporal_basis(cfg)
    cov = full_covariance(cfg, "colored")
    clean, _ = gen.generate_clean_events(rng, n, cfg, V, Tb)
    clean, snr = gen.snr_normalize(clean, cov, cfg)
    Z = whiten_separable(clean, cov["Sigma_c"], cov["Sigma_t"])          # (n, C, T)
    Z = Z - Z.mean(axis=0, keepdims=True)
    return cfg, cov, Z, snr


def per_axis_gaps(Z: np.ndarray, cfg) -> dict:
    n, C, T = Z.shape
    # whitened noise contributes unit variance per entry, so the per-axis noise level is 1 on these scales
    M_c = np.einsum("nct,ndt->cd", Z, Z) / (n * T)
    M_t = np.einsum("nct,ncs->ts", Z, Z) / (n * C)
    ec = np.sort(np.linalg.eigvalsh(M_c))[::-1] + 1.0
    et = np.sort(np.linalg.eigvalsh(M_t))[::-1] + 1.0
    flat = Z.reshape(n, -1, order="F")
    # full CT spectrum via the n x n Gram (n >= CT here, so use the covariance directly)
    Cfull = flat.T @ flat / n
    ef = np.sort(np.linalg.eigvalsh(Cfull))[::-1] + 1.0
    kc, kt, k = cfg.k_c, cfg.k_t, cfg.k_c * cfg.k_t
    return {
        "channel": {"eigenvalues_top": [float(x) for x in ec[: kc + 2]], "gap": float(ec[kc - 1] - ec[kc]),
                    "lambda1_over_gap": float(ec[0] / (ec[kc - 1] - ec[kc]))},
        "temporal": {"eigenvalues_top": [float(x) for x in et[: kt + 2]], "gap": float(et[kt - 1] - et[kt]),
                     "lambda1_over_gap": float(et[0] / (et[kt - 1] - et[kt]))},
        "full": {"eigenvalues_top": [float(x) for x in ef[: k + 2]], "gap": float(ef[k - 1] - ef[k]),
                 "lambda1_over_gap": float(ef[0] / (ef[k - 1] - ef[k]))},
        "gap_ratio_channel_over_temporal": float((ec[kc - 1] - ec[kc]) / (et[kt - 1] - et[kt])),
    }


def rem52_at_ncal(cov: dict, lam1: float, gap: float) -> list[dict]:
    cfg = PRIMARY
    Sigma = cov["Sigma"]
    Wi = _sym_isqrt(Sigma)
    rows = []
    for seed in CAL_SEEDS:
        rng = np.random.default_rng(700_000 + seed)
        noise = gen.generate_noise(rng, N_CAL, cfg, cov)
        est = estimate_calibrated_axis_covariances(noise)
        Sigma_hat = np.kron(est.Sigma_t, est.Sigma_c)
        E = Wi @ (Sigma_hat - Sigma) @ Wi
        eps = float(np.linalg.norm(E, 2))
        rows.append({"cal_seed": seed, "N_cal": N_CAL, "eps_op": eps, "eps_fro": float(np.linalg.norm(E, "fro")),
                     "lambda1_over_gap": lam1 / gap, "bound_6eps_lam1_over_gap": 6 * eps * lam1 / gap,
                     "in_domain_eps_le_half": bool(eps <= 0.5),
                     "product_scale_ratio": float(np.trace(Sigma_hat) / np.trace(Sigma))})
    return rows


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"task": "T23", "n_clean": N_CLEAN, "per_axis": {}, "rem52": {}}
    for alpha in (0.0, PRIMARY.nonseparability_alpha):
        cfg, cov, Z, snr = whitened_clean_sample(alpha)
        gaps = per_axis_gaps(Z, cfg)
        gaps["realized_snr"] = snr
        out["per_axis"][f"alpha={alpha}"] = gaps
    # Rem. 5.2 at the frozen alpha = 0.1 configuration (the R/PS estimated-covariance point)
    cfg, cov, Z, _ = whitened_clean_sample(PRIMARY.nonseparability_alpha)
    full = out["per_axis"][f"alpha={PRIMARY.nonseparability_alpha}"]["full"]
    rows = rem52_at_ncal(cov, full["eigenvalues_top"][0], full["gap"])
    eps = [r["eps_op"] for r in rows]
    out["rem52"] = {"rows": rows, "eps_op_median": float(np.median(eps)), "eps_op_max": float(max(eps)),
                    "bound_median": float(np.median([r["bound_6eps_lam1_over_gap"] for r in rows])),
                    "in_domain_all": all(r["in_domain_eps_le_half"] for r in rows),
                    "lambda1": full["eigenvalues_top"][0], "delta_lambda_rank6": full["gap"]}
    out["registered_prediction_channel_gap_gg_temporal_gap"] = {
        a: bool(v["gap_ratio_channel_over_temporal"] > 3.0) for a, v in out["per_axis"].items()}

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return "unavailable"
    out["provenance"] = {"code_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                         "tree_dirty": bool(git("status", "--porcelain")),
                         "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "primary_config_hash": config_hash(PRIMARY), "python": sys.version.split()[0],
                         "numpy": np.__version__, "platform": platform.platform(), "hostname": socket.gethostname(),
                         "wall_seconds": time.time() - t0, "command": sys.argv}
    (OUT_DIR / "t23.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("per_axis", "registered_prediction_channel_gap_gg_temporal_gap")}, indent=1))
    print(json.dumps({k: v for k, v in out["rem52"].items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
