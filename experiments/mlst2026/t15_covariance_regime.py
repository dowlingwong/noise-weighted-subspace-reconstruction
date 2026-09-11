"""T15 - covariance-error regime on ONE explicitly frozen generator.

Protocol: MS ``plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md`` T15.

Generator mismatch, resolved
----------------------------
The programme names "the S5/S7 synthetic generator at its frozen
configuration", but no such single generator exists: S5 uses a PSD exponent of
-2 (1/f^2) with two *sinusoidal* signal directions, and S7 uses AR(1) noise
with a single pulse.  Neither is the "combined exponent -1, rank 2" model the
theorem section refers to.  Rather than silently adopting one of them, this
module defines one new generator, **G1**, and freezes it here.

G1 (frozen)
-----------
*Dimension* ``d = 256`` samples, unit sampling frequency.

*Noise covariance, including scale.*  Stationary Gaussian with one-sided PSD
``S(f) proportional to (sqrt(f^2 + f_knee^2))^{-1}`` -- **exponent -1** -- with
``f_knee = 4 / d`` (four bins), which regularises the 1/f divergence at DC and
is what fixes the noise correlation length.  ``Sigma`` is the exact circulant
covariance that PSD induces on the FFT grid, so its eigenvalues are known in
closed form.  The scale is pinned by ``trace(Sigma) / d = 1`` exactly: the
noise has unit average variance per sample, so every reported covariance error
is a relative error against a fixed, stated scale.

*Signal.*  A rank-2 subspace ``U = [u1, u2]``, two fixed orthonormal vectors
drawn once from seed 20260915 and frozen (a generic subspace, deliberately not
the sinusoid pair of S5, so that the result is not an artefact of the signal
being an eigenvector family of a circulant covariance).  Amplitudes
``a ~ N(0, diag(s1^2, s2^2))`` with ``s1 = SNR`` and ``s2 = SNR / 2`` in units
of the noise's per-sample standard deviation; the unequal strengths give a
well-defined eigengap.  A trace is ``x = U a + n``.

*Sizes.*  512 training traces (signal + noise, used to fit the subspace);
``m`` calibration traces (**noise only**) for ``Sigma_hat``; 2000 held-out test
traces for the task error.

*Misspecification operators.*  ``slope``: the assumed PSD exponent is -1.5 or
-0.5 while the truth stays -1.  ``corrlength``: the assumed knee is
``f_knee / 2`` or ``2 f_knee``, which doubles or halves the noise correlation
length -- this is the explicit translation of the programme's "AR(1) length
doubled and halved" onto G1, whose noise is power-law rather than AR(1).
Every misspecified covariance is renormalised to the same ``trace/d = 1``, so
misspecification changes the *shape* of the covariance and never its scale.

*Arms and the 80-fit limit.*  8 arms x 10 seeds = 80 fits exactly:
oracle; estimated-64/128/256/512/2048 (5 arms); misspecified-slope (one arm,
seeds 0-4 at -1.5 and seeds 5-9 at -0.5); misspecified-corrlength (one arm,
seeds 0-4 at length x2 and seeds 5-9 at length x1/2).  Each (arm, seed) is one
fit, so the count is 1 + 5 + 1 + 1 = 8 arms, 80 fits, matching the declared cap.

*Pairing.*  The seed fixes the training, calibration and test draws, so all
eight arms at a given seed see **the same data** and differ only in the
covariance they are handed.  Comparisons across arms are therefore paired.

Reported error
--------------
The theorem's quantity is the **relative operator covariance error**
``epsilon = ||Sigma_used - Sigma_true||_2 / ||Sigma_true||_2`` in the spectral
(operator) norm -- not the normalised Frobenius error the old S7 runner
reported.  Both are recorded so the difference is visible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.mlst2026.provenance import RunRecorder, write_json  # noqa: E402

# --- frozen constants ---------------------------------------------------------
D = 256
RANK = 2
N_TRAIN = 512
N_TEST = 2000
KNEE_BINS = 4.0
TRUE_EXPONENT = -1.0
SNR = 3.0
SUBSPACE_SEED = 20260915
BASE_SEED = 20260915
SEEDS = tuple(range(10))
ARMS = ("oracle", "estimated-64", "estimated-128", "estimated-256",
        "estimated-512", "estimated-2048", "misspecified-slope",
        "misspecified-corrlength")
SUBARM_SPLIT = 5  # seeds 0-4 -> first sub-arm, 5-9 -> second
SLOPE_SUBARMS = (-1.5, -0.5)
CORRLENGTH_SUBARMS = (0.5, 2.0)  # knee multiplier: 0.5 doubles the length
FIT_CAP = 80

#: sin(Theta) is computed as sqrt(1 - sigma_min^2); near sigma_min = 1 that form
#: has absolute resolution of order sqrt(machine epsilon) ~ 1.5e-8, so an
#: identically-zero angle reads as a few times 1e-8. The bound comparison
#: carries that tolerance rather than failing on floating-point noise.
SIN_THETA_TOL = 1e-6


def psd_vector(exponent: float, knee_bins: float) -> np.ndarray:
    """One-sided PSD on the rfft grid of D samples, normalised to trace/d = 1."""
    f = np.fft.rfftfreq(D, d=1.0)
    knee = knee_bins / D
    s = (np.sqrt(f ** 2 + knee ** 2)) ** exponent
    cov_diag = circulant_eigenvalues(s)
    return s / (cov_diag.mean())


def circulant_eigenvalues(psd_onesided: np.ndarray) -> np.ndarray:
    """Eigenvalues of the circulant covariance induced by a one-sided PSD.

    Each interior rfft bin contributes twice (positive and negative frequency),
    the DC and Nyquist bins once. The returned vector has length D and is the
    spectrum of Sigma.
    """
    s = np.asarray(psd_onesided, dtype=np.float64)
    ev = np.empty(D)
    ev[0] = s[0]
    ev[D // 2] = s[-1]
    ev[1:D // 2] = s[1:-1]
    ev[D // 2 + 1:] = s[1:-1][::-1]
    return ev


def covariance_from_psd(psd_onesided: np.ndarray) -> np.ndarray:
    """The exact circulant covariance matrix, symmetric and positive definite."""
    ev = circulant_eigenvalues(psd_onesided)
    c = np.real(np.fft.ifft(ev))          # first row of the circulant matrix
    idx = np.arange(D)
    return c[(idx[:, None] - idx[None, :]) % D]


def frozen_signal_basis() -> np.ndarray:
    """Two fixed orthonormal signal directions (generic, not sinusoids)."""
    rng = np.random.default_rng(SUBSPACE_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((D, RANK)))
    return np.ascontiguousarray(q[:, :RANK])


def sample_noise(rng, cov_sqrt: np.ndarray, n: int) -> np.ndarray:
    return rng.standard_normal((n, D)) @ cov_sqrt.T


def matrix_sqrt(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    vals = np.clip(vals, 1e-300, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def inv_sqrt(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    vals = np.clip(vals, 1e-300, None)
    return (vecs / np.sqrt(vals)) @ vecs.T


def relative_operator_error(S_used: np.ndarray, S_true: np.ndarray) -> float:
    """epsilon = ||S_used - S_true||_2 / ||S_true||_2 (spectral norm)."""
    return float(np.linalg.norm(S_used - S_true, 2) / np.linalg.norm(S_true, 2))


def relative_frobenius_error(S_used: np.ndarray, S_true: np.ndarray) -> float:
    """The old S7 quantity, reported alongside so the difference is visible."""
    return float(np.linalg.norm(S_used - S_true, "fro")
                 / np.linalg.norm(S_true, "fro"))


def sin_theta(A: np.ndarray, B: np.ndarray, metric_inv: np.ndarray) -> float:
    """Sine of the largest principal angle between two column spans, measured
    in the metric ``metric_inv`` (the true inverse noise covariance)."""
    def orth(M):
        G = M.T @ metric_inv @ M
        vals, vecs = np.linalg.eigh(0.5 * (G + G.T))
        vals = np.clip(vals, 1e-300, None)
        return M @ (vecs / np.sqrt(vals)) @ vecs.T
    Qa, Qb = orth(A), orth(B)
    s = np.linalg.svd(Qa.T @ metric_inv @ Qb, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - np.min(np.clip(s, -1, 1)) ** 2)))


def covariance_for_arm(arm: str, seed: int, rng, S_true: np.ndarray,
                       true_sqrt: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the covariance the estimator is handed, plus its label."""
    if arm == "oracle":
        return S_true.copy(), {"subarm": "oracle", "n_calibration": None}
    if arm.startswith("estimated-"):
        m = int(arm.split("-")[1])
        Z = sample_noise(rng, true_sqrt, m)
        S = (Z.T @ Z) / m
        return S, {"subarm": f"m={m}", "n_calibration": m}
    if arm == "misspecified-slope":
        e = SLOPE_SUBARMS[0 if seed < SUBARM_SPLIT else 1]
        return covariance_from_psd(psd_vector(e, KNEE_BINS)), \
            {"subarm": f"exponent={e}", "n_calibration": None}
    if arm == "misspecified-corrlength":
        mult = CORRLENGTH_SUBARMS[0 if seed < SUBARM_SPLIT else 1]
        return covariance_from_psd(psd_vector(TRUE_EXPONENT, KNEE_BINS * mult)), \
            {"subarm": f"knee_x{mult} (length x{1/mult:g})", "n_calibration": None}
    raise ValueError(arm)


def run_cell(arm: str, seed: int, S_true: np.ndarray, true_sqrt: np.ndarray,
             U: np.ndarray) -> dict:
    """One fit. The seed fixes the data; the arm fixes only the covariance."""
    ss = np.random.SeedSequence([BASE_SEED, seed])
    data_rng = np.random.default_rng(ss.spawn(2)[0])
    cov_rng = np.random.default_rng(np.random.SeedSequence(
        [BASE_SEED, seed, ARMS.index(arm)]))

    s_amp = np.array([SNR, SNR / 2.0])
    a_tr = data_rng.standard_normal((N_TRAIN, RANK)) * s_amp
    X_tr = a_tr @ U.T + sample_noise(data_rng, true_sqrt, N_TRAIN)
    a_te = data_rng.standard_normal((N_TEST, RANK)) * s_amp
    S_te = a_te @ U.T
    X_te = S_te + sample_noise(data_rng, true_sqrt, N_TEST)

    S_used, label = covariance_for_arm(arm, seed, cov_rng, S_true, true_sqrt)
    eps = relative_operator_error(S_used, S_true)
    eps_fro = relative_frobenius_error(S_used, S_true)

    def fit_subspace(S_metric):
        Xw = X_tr @ inv_sqrt(S_metric).T
        Gw = (Xw.T @ Xw) / N_TRAIN
        vals, vecs = np.linalg.eigh(0.5 * (Gw + Gw.T))
        order = np.argsort(vals)[::-1]
        return matrix_sqrt(S_metric) @ vecs[:, order][:, :RANK]

    U_hat = fit_subspace(S_used)
    # The oracle fit on the SAME training draw. The theorem bounds the subspace
    # shift caused by handing the estimator a perturbed covariance, so the
    # quantity it constrains is the distance to this paired oracle fit -- not
    # the distance to the planted subspace, which also contains the finite
    # training-sample error that is present even when epsilon = 0.
    U_oracle_same_data = fit_subspace(S_true)

    # population whitened operator in the TRUE metric: eigenvalues and gap
    Wt = inv_sqrt(S_true)
    Gpop = Wt @ (U * s_amp ** 2) @ U.T @ Wt.T + np.eye(D)
    pv = np.sort(np.linalg.eigvalsh(0.5 * (Gpop + Gpop.T)))[::-1]
    lam1, gap = float(pv[0]), float(pv[RANK - 1] - pv[RANK])

    S_true_inv = np.linalg.inv(S_true)
    st_planted = sin_theta(U_hat, U, S_true_inv)
    st_vs_oracle = sin_theta(U_hat, U_oracle_same_data, S_true_inv)
    st_oracle_planted = sin_theta(U_oracle_same_data, U, S_true_inv)
    bound = 6.0 * eps * lam1 / gap if gap > 0 else float("inf")

    # task error on held-out traces, in the metric actually used
    Sinv_used = np.linalg.inv(S_used)
    G = U_hat.T @ Sinv_used @ U_hat
    coeff = np.linalg.solve(G, U_hat.T @ Sinv_used @ X_te.T).T
    recon = coeff @ U_hat.T
    amp_rmse = float(np.sqrt(np.mean(np.sum((recon - S_te) ** 2, axis=1) / D)))
    # held-out probe R^2 for the true amplitudes
    A = np.linalg.lstsq(coeff, a_te, rcond=None)[0]
    resid = a_te - coeff @ A
    r2 = float(1.0 - np.sum(resid ** 2) / np.sum((a_te - a_te.mean(0)) ** 2))

    return {"arm": arm, "seed": int(seed), **label,
            "epsilon_relative_operator": eps,
            "epsilon_relative_frobenius": eps_fro,
            "lambda_1": lam1, "delta_lambda": gap,
            "lambda1_over_gap": float(lam1 / gap) if gap > 0 else float("inf"),
            "sin_theta_vs_planted": st_planted,
            "sin_theta_vs_paired_oracle_fit": st_vs_oracle,
            "sin_theta_oracle_fit_vs_planted": st_oracle_planted,
            "sin_theta_measured": st_planted,
            "bound_6_eps_lambda1_over_gap": float(bound),
            "bound_holds": bool(st_vs_oracle <= bound + SIN_THETA_TOL),
            "bound_informative": bool(bound < 1.0),
            "bound_slack": float(bound - st_vs_oracle),
            "amplitude_rmse": amp_rmse, "probe_r2": r2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory {out_dir}")

    rec = RunRecorder(run_id="T15_covariance_regime",
                      protocol_refs={"programme": "plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md#T15",
                                     "config": "plan/mlst_protocol_2026-09-10.json#t15"})
    rec.add_input("runner_source", Path(__file__).resolve())
    rec.add_seed("base_seed", BASE_SEED)
    rec.add_seed("subspace_seed", SUBSPACE_SEED)
    rec.add_seed("seeds", list(SEEDS))

    S_true = covariance_from_psd(psd_vector(TRUE_EXPONENT, KNEE_BINS))
    true_sqrt = matrix_sqrt(S_true)
    U = frozen_signal_basis()
    seeds = SEEDS[:2] if args.smoke else SEEDS

    rows = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(run_cell(arm, seed, S_true, true_sqrt, U))
            r = rows[-1]
            print(f"  [{arm:<24} seed {seed}] eps_op={r['epsilon_relative_operator']:.5f} "
                  f"eps_fro={r['epsilon_relative_frobenius']:.5f} "
                  f"sinT(planted)={r['sin_theta_vs_planted']:.4f} sinT(vs oracle)={r['sin_theta_vs_paired_oracle_fit']:.4f} "
                  f"bound={r['bound_6_eps_lambda1_over_gap']:.4g} holds={r['bound_holds']} "
                  f"amp_rmse={r['amplitude_rmse']:.5f} R2={r['probe_r2']:.5f}")
    n_fits = len(rows)
    if not args.smoke and n_fits != FIT_CAP:
        raise SystemExit(f"fit count {n_fits} != declared {FIT_CAP}")

    payload = {
        "smoke": bool(args.smoke),
        "generator": {
            "id": "G1", "d": D, "rank": RANK,
            "true_psd_exponent": TRUE_EXPONENT, "knee_bins": KNEE_BINS,
            "trace_normalisation": "trace(Sigma)/d = 1 exactly",
            "signal": "rank-2 generic orthonormal basis, seed 20260915; "
                      "a ~ N(0, diag(SNR^2, (SNR/2)^2))",
            "snr": SNR, "n_train": N_TRAIN, "n_test": N_TEST,
            "misspecification": {"slope": list(SLOPE_SUBARMS),
                                 "corrlength_knee_multiplier": list(CORRLENGTH_SUBARMS)},
            "note": "one frozen model replacing the S5 (exponent -2, sinusoid "
                    "pair) / S7 (AR(1), single pulse) mismatch",
        },
        "arms": list(ARMS), "seeds": list(seeds), "n_fits": n_fits,
        "fit_cap": FIT_CAP,
        "cells": rows,
    }
    # per-arm aggregation, paired across seeds against the oracle arm
    by = {}
    oracle = {r["seed"]: r for r in rows if r["arm"] == "oracle"}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        pair = [r["sin_theta_vs_planted"] - oracle[r["seed"]]["sin_theta_vs_planted"]
                for r in sub]
        pair_rmse = [r["amplitude_rmse"] - oracle[r["seed"]]["amplitude_rmse"]
                     for r in sub]
        by[arm] = {
            "n": len(sub),
            "epsilon_relative_operator_mean": float(np.mean([r["epsilon_relative_operator"] for r in sub])),
            "epsilon_relative_frobenius_mean": float(np.mean([r["epsilon_relative_frobenius"] for r in sub])),
            "sin_theta_vs_planted_mean": float(np.mean([r["sin_theta_vs_planted"] for r in sub])),
            "sin_theta_vs_planted_sd": float(np.std([r["sin_theta_vs_planted"] for r in sub], ddof=1)) if len(sub) > 1 else float("nan"),
            "sin_theta_vs_paired_oracle_mean": float(np.mean([r["sin_theta_vs_paired_oracle_fit"] for r in sub])),
            "bound_slack_min": float(np.min([r["bound_slack"] for r in sub])),
            "bound_mean": float(np.mean([r["bound_6_eps_lambda1_over_gap"] for r in sub])),
            "bound_holds_all": bool(all(r["bound_holds"] for r in sub)),
            "bound_informative_all": bool(all(r["bound_informative"] for r in sub)),
            "amplitude_rmse_mean": float(np.mean([r["amplitude_rmse"] for r in sub])),
            "probe_r2_mean": float(np.mean([r["probe_r2"] for r in sub])),
            "paired_sin_theta_vs_oracle_mean": float(np.mean(pair)),
            "paired_amplitude_rmse_vs_oracle_mean": float(np.mean(pair_rmse)),
        }
    payload["by_arm"] = by
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["provenance"] = rec.finalize()
    digest = write_json(out_dir / "t15_results.json", payload)
    (out_dir / "RESULTS_SHA256").write_text(f"{digest}  t15_results.json\n")
    print(f"[T15] {n_fits} fits; wrote {out_dir}/t15_results.json sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
