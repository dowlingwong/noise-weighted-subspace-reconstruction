"""TIDMAD Phase 1: noise model, whitening convention, and estimator identity.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`` Phase 1, gate G1.

Why this runs before any TIDMAD science
---------------------------------------
Paper 1's whole argument is that the *right* Sigma defines the geometry. So the
first step on a new dataset is not a comparison -- it is establishing that our
estimated Sigma, our whitening convention, and our estimator are correct on data
where the answer is known by construction. This is the same discipline that made
the GWOSC implementation checks meaningful even though its calibration failed:
without them, a null result is indistinguishable from a convention bug.

What it checks (each gate is predeclared)
-----------------------------------------
G1a  amplitude recovery is unbiased          (|bias| < 3 standard errors)
G1b  empirical spread matches the
     PSD-predicted bound                      (ratio within 10% of 1)
G1c  a no-signal null is consistent with zero (|mean| < 3 standard errors)
G1d  a *learned* rank-2 weighted subspace recovers the *analytic* quadrature
     subspace                                 (max principal angle < 5 deg)
G1e  a fixed-phase rank-1 filter loses SNR by cos^2(phase error), as Paper 1
     Eq. (16) predicts                        (agreement within 5%)
G1f  that subspace-recovery error follows the theoretical 1/SNR law
                                              (log-log slope -1 +/- 0.15)

On the amplitude scale. The injected amplitude is specified in units of the
estimator's own predicted noise (``--snr``), not in raw detector units. This
matters: a bias gate stated as a fixed percentage is only meaningful if the
statistical uncertainty is smaller than that percentage, and at low SNR it is
not. Fixing SNR fixes the power of the test, and G1a is stated in standard
errors so it stays honest at any SNR.

G1d is the real-data version of Paper 1's S2. Unlike CRESST, where the rank-6
subspace was an empirical choice, here the correct subspace is known in closed
form -- the unknown phase of a sinusoid spans exactly span{cos, sin} -- so a
learned subspace can be compared against ground truth rather than against
another fit.

Modes
-----
``--mode synthetic`` (default) runs on noise generated from a declared PSD. It
validates the estimator and the conventions, needs no data, and is the gate that
can be run today and re-run in CI.

``--mode file`` runs the identical checks with injections into *real*
signal-free TIDMAD science stretches. That is the version that matters
scientifically, because it tests the estimator against the real noise
distribution including its non-Gaussian tails. It requires a verified Phase 0
data contract.

    uv run python scripts/tidmad_phase1_estimator_identity.py
    uv run python scripts/tidmad_phase1_estimator_identity.py \
        --mode file --data-root /ceph/dwong/paper1_dataset \
        --contract docs/tidmad_data_contract.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.noise import (  # noqa: E402
    generate_colored_noise,
    inverse_psd_weights,
    make_powerlaw_psd,
)
from noise_geometry.subspace import fit_weighted_pca, principal_angles  # noqa: E402
from noise_geometry.tidmad import (  # noqa: E402
    fit_fixed_phase,
    fit_quadrature,
    fixed_phase_snr_loss,
    inject_sinusoid,
    nearest_bin_frequency,
    quadrature_amplitude_variance,
    quadrature_templates_rfft,
    run_injection_check,
    welch_psd_from_windows,
)

try:  # central rFFT<->real primitive, shared with S1/S2/S5
    from canonical.empca_equivalence_utils import complex_to_real_whitened
except ImportError:  # pragma: no cover - repo-root on path
    from src.canonical.empca_equivalence_utils import complex_to_real_whitened  # type: ignore

# --- predeclared gate thresholds (fixed before running; do not tune) --------
GATE_BIAS_SIGMA_MULTIPLE = 3.0     # G1a
GATE_SIGMA_RATIO_TOLERANCE = 0.10  # G1b
GATE_NULL_SIGMA_MULTIPLE = 3.0     # G1c
GATE_SUBSPACE_ANGLE_DEG = 5.0      # G1d, at the declared operating point
GATE_SNR_LOSS_TOLERANCE = 0.05     # G1e
GATE_SCALING_SLOPE_TOLERANCE = 0.15  # G1f


def _load_real_noise(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    """Load disjoint calibration and evaluation windows from science files."""
    from noise_geometry.tidmad.loader import TidmadFile, load_contract

    contract = load_contract(args.contract)
    contract.require_verified()

    science = sorted(Path(args.data_root).glob(args.science_glob))
    if not science:
        raise FileNotFoundError(
            f"no science files matched {args.science_glob!r} under {args.data_root}"
        )

    calibration: list[np.ndarray] = []
    evaluation: list[np.ndarray] = []
    with TidmadFile(science[0], contract) as handle:
        stream = handle.iter_squid_windows(
            args.window_samples, stride=args.window_samples, limit=2 * args.n_windows
        )
        for i, window in enumerate(stream):
            (calibration if i < args.n_windows else evaluation).append(window)

    if len(evaluation) < 2:
        raise RuntimeError("not enough science windows for a disjoint calibration/evaluation split")
    return np.vstack(calibration), np.vstack(evaluation), contract.sampling_frequency_hz


def _snr_loss_check(
    evaluation: np.ndarray,
    basis_f: np.ndarray,
    weights: np.ndarray,
    sampling_frequency: float,
    frequency: float,
    amplitude: float,
    phase_errors: np.ndarray,
) -> dict:
    """Measure the SNR cost of assuming a phase, against the cos^2 prediction.

    Injection is at a fixed phase of zero so the phase *error* is exactly the
    filter phase. The measured quantity is the squared ratio of estimated
    signal-to-noise, which is what Paper 1 Eq. (16) predicts equals cos^2 of the
    whitened angle -- for this manifold, the phase error itself. The amplitude
    ratio (which falls as cos, not cos^2) is reported alongside because
    conflating the two is the easiest mistake to make here.
    """
    injected, _ = inject_sinusoid(
        evaluation, sampling_frequency, frequency, amplitude=amplitude, phases=0.0
    )
    X_f = np.fft.rfft(injected, axis=1)

    reference = np.asarray(fit_fixed_phase(X_f, basis_f, weights, phase=0.0))
    reference_snr = float(np.mean(reference) / np.std(reference, ddof=1))

    amplitude_ratio, snr_squared_ratio = [], []
    for phase_error in phase_errors:
        amps = np.asarray(fit_fixed_phase(X_f, basis_f, weights, phase=float(phase_error)))
        amplitude_ratio.append(float(np.mean(amps) / np.mean(reference)))
        snr = float(np.mean(amps) / np.std(amps, ddof=1))
        snr_squared_ratio.append(float((snr / reference_snr) ** 2))

    predicted = [fixed_phase_snr_loss(float(p)) for p in phase_errors]
    return {
        "phase_errors_rad": [float(p) for p in phase_errors],
        "observed_snr_squared_ratio": snr_squared_ratio,
        "predicted_cos_squared": predicted,
        "observed_amplitude_ratio": amplitude_ratio,
        "predicted_cos": [float(np.cos(float(p))) for p in phase_errors],
        "max_abs_error": float(
            np.max(np.abs(np.asarray(snr_squared_ratio) - np.asarray(predicted)))
        ),
    }


def _subspace_recovery_angle(
    evaluation: np.ndarray,
    basis_f: np.ndarray,
    weights: np.ndarray,
    sampling_frequency: float,
    frequency: float,
    amplitude: float,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    """Fit a rank-2 weighted subspace and compare it to the analytic quadrature."""
    injected, _ = inject_sinusoid(
        evaluation, sampling_frequency, frequency, amplitude=amplitude, rng=rng
    )
    features = complex_to_real_whitened(np.fft.rfft(injected, axis=1), weights)
    learned = fit_weighted_pca(features, np.ones(features.shape[1]), 2)
    analytic = complex_to_real_whitened(basis_f, weights)
    angles = principal_angles(learned.components, analytic)
    return float(np.max(angles)), [float(a) for a in angles]


def _subspace_scaling_check(
    evaluation: np.ndarray,
    basis_f: np.ndarray,
    weights: np.ndarray,
    sampling_frequency: float,
    frequency: float,
    predicted_sigma: float,
    snr_grid: np.ndarray,
    seed: int,
) -> dict:
    """Test that subspace-recovery error follows the theoretical 1/SNR law.

    A single principal angle at one operating point says little: any value could
    be finite-sample error or could be bias. The falsifiable statement is the
    *rate*. Classical subspace-perturbation theory gives an estimation error
    that scales like the inverse of the per-window signal-to-noise ratio (at
    fixed sample count), so the log-log slope of angle against SNR should be
    -1. A slope near -1 says the residual angle is finite-sample error and
    vanishes with better data; a slope near 0 would say the learned subspace is
    converging to the wrong place, which is a bug or a broken convention.
    """
    angles = []
    for i, snr in enumerate(snr_grid):
        angle, _ = _subspace_recovery_angle(
            evaluation,
            basis_f,
            weights,
            sampling_frequency,
            frequency,
            float(snr) * predicted_sigma,
            np.random.default_rng(seed + 100 + i),
        )
        angles.append(angle)

    log_snr = np.log(np.asarray(snr_grid, dtype=np.float64))
    log_angle = np.log(np.asarray(angles, dtype=np.float64))
    slope, intercept = np.polyfit(log_snr, log_angle, 1)
    return {
        "snr_grid": [float(s) for s in snr_grid],
        "max_principal_angle_deg": angles,
        "log_log_slope": float(slope),
        "log_log_intercept": float(intercept),
        "expected_slope": -1.0,
        "abs_slope_error": float(abs(slope + 1.0)),
    }


def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    n_samples = int(args.window_samples)

    psd_truth = None
    if args.mode == "synthetic":
        fs = float(args.sampling_frequency)
        # A coloured floor: SQUID-like 1/f, optionally with a white knee. The
        # colouring is the point -- it separates the raw-variance direction from
        # the high-information direction, which is the regime Paper 1 is about.
        # Generation uses the repository's own generator so the discrete
        # convention lives in exactly one place.
        _, psd_truth = make_powerlaw_psd(n_samples, fs, kind=args.psd_kind, knee_hz=args.psd_knee_hz)
        calibration = generate_colored_noise(rng, psd_truth, n_samples, fs, args.n_windows)
        evaluation = generate_colored_noise(rng, psd_truth, n_samples, fs, args.n_windows)
        noise_source = f"synthetic {args.psd_kind} PSD (knee {args.psd_knee_hz} Hz)"
    else:
        calibration, evaluation, fs = _load_real_noise(args)
        noise_source = f"real science windows under {args.data_root}"

    # --- Sigma from calibration data only, never from evaluation data -------
    estimate = welch_psd_from_windows(
        list(calibration), fs, window="boxcar", detrend=True, average="median", source=noise_source
    )
    psd = estimate.psd
    weights = inverse_psd_weights(psd, n_samples)

    frequency = nearest_bin_frequency(args.frequency, n_samples, fs)
    basis_f = quadrature_templates_rfft(n_samples, fs, frequency)

    # Amplitude in units of the estimator's own predicted noise, so the gates
    # have a known statistical power rather than an accidental one.
    predicted_sigma = float(
        np.sqrt(
            quadrature_amplitude_variance(
                basis_f, psd, weights, trace_len=n_samples, sampling_frequency=fs
            )
        )
    )
    amplitude = float(args.snr) * predicted_sigma

    # --- G1a / G1b: injection recovery and predicted variance ---------------
    injection = run_injection_check(
        evaluation, psd, weights, fs, frequency, amplitude=amplitude, rng=rng
    )
    bias = injection.mean_recovered_amplitude - amplitude
    bias_standard_error = injection.empirical_sigma / np.sqrt(injection.n_windows)
    bias_z = float(bias / bias_standard_error) if bias_standard_error > 0 else float("nan")

    # Diagnostic only, and only available when the truth is known: the same
    # ratio computed with the *true* PSD isolates estimator error from
    # covariance-estimation error (Paper 1 S7 measures the latter directly).
    sigma_ratio_true_psd = None
    if psd_truth is not None:
        weights_true = inverse_psd_weights(psd_truth, n_samples)
        oracle = run_injection_check(
            evaluation,
            psd_truth,
            weights_true,
            fs,
            frequency,
            amplitude=amplitude,
            rng=np.random.default_rng(args.seed + 1),
        )
        sigma_ratio_true_psd = oracle.sigma_ratio

    # --- G1c: no-signal null -------------------------------------------------
    null_fit = fit_quadrature(np.fft.rfft(evaluation, axis=1), basis_f, weights)
    null_coefficients = np.asarray(null_fit.cos_coefficient)
    null_mean = float(np.mean(null_coefficients))
    null_sigma = float(np.std(null_coefficients, ddof=1))
    null_stderr = null_sigma / np.sqrt(null_coefficients.size)

    # --- G1d: does a learned rank-2 subspace find the analytic quadrature? ---
    learned_amplitude = amplitude * float(args.learned_subspace_snr_boost)
    max_angle_deg, angles = _subspace_recovery_angle(
        evaluation, basis_f, weights, fs, frequency, learned_amplitude, rng
    )

    # --- G1f: does that recovery error follow the theoretical 1/SNR law? -----
    scaling = _subspace_scaling_check(
        evaluation,
        basis_f,
        weights,
        fs,
        frequency,
        predicted_sigma,
        np.asarray(args.scaling_snr_grid, dtype=np.float64),
        int(args.seed),
    )

    # --- G1e: fixed-phase rank-1 SNR loss vs the cos^2 prediction ------------
    snr_loss = _snr_loss_check(
        evaluation,
        basis_f,
        weights,
        fs,
        frequency,
        amplitude,
        np.asarray(args.phase_errors, dtype=np.float64),
    )

    gates = {
        "G1a_amplitude_bias_z": {
            "value": bias_z,
            "threshold": GATE_BIAS_SIGMA_MULTIPLE,
            "passed": bool(abs(bias_z) < GATE_BIAS_SIGMA_MULTIPLE),
        },
        "G1b_sigma_over_predicted": {
            "value": injection.sigma_ratio,
            "threshold": GATE_SIGMA_RATIO_TOLERANCE,
            "passed": bool(abs(injection.sigma_ratio - 1.0) < GATE_SIGMA_RATIO_TOLERANCE),
        },
        "G1c_null_mean_z": {
            "value": float(null_mean / null_stderr) if null_stderr > 0 else float("nan"),
            "threshold": GATE_NULL_SIGMA_MULTIPLE,
            "passed": bool(abs(null_mean) < GATE_NULL_SIGMA_MULTIPLE * null_stderr),
        },
        "G1d_max_principal_angle_deg": {
            "value": max_angle_deg,
            "threshold": GATE_SUBSPACE_ANGLE_DEG,
            "passed": bool(max_angle_deg < GATE_SUBSPACE_ANGLE_DEG),
        },
        "G1e_snr_loss_max_abs_error": {
            "value": snr_loss["max_abs_error"],
            "threshold": GATE_SNR_LOSS_TOLERANCE,
            "passed": bool(snr_loss["max_abs_error"] < GATE_SNR_LOSS_TOLERANCE),
        },
        "G1f_subspace_scaling_slope_error": {
            "value": scaling["abs_slope_error"],
            "threshold": GATE_SCALING_SLOPE_TOLERANCE,
            "passed": bool(scaling["abs_slope_error"] < GATE_SCALING_SLOPE_TOLERANCE),
        },
    }

    return {
        "experiment": "TIDMAD-P1",
        "phase": "Phase 1 -- noise model, whitening convention, estimator identity",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": args.mode,
        "noise_source": noise_source,
        "configuration": {
            "seed": int(args.seed),
            "window_samples": n_samples,
            "n_windows": int(args.n_windows),
            "sampling_frequency_hz": float(fs),
            "requested_frequency_hz": float(args.frequency),
            "on_grid_frequency_hz": float(frequency),
            "requested_snr": float(args.snr),
            "predicted_sigma": predicted_sigma,
            "injected_amplitude": amplitude,
            "learned_subspace_snr_boost": float(args.learned_subspace_snr_boost),
            "psd_kind": args.psd_kind if args.mode == "synthetic" else None,
            "psd_knee_hz": args.psd_knee_hz if args.mode == "synthetic" else None,
        },
        "injection": {
            **injection.as_dict(),
            "bias_absolute": float(bias),
            "bias_standard_error": float(bias_standard_error),
            "sigma_ratio_with_true_psd": sigma_ratio_true_psd,
        },
        "null": {
            "mean": null_mean,
            "sigma": null_sigma,
            "standard_error": null_stderr,
            "n_windows": int(null_coefficients.size),
        },
        "learned_subspace": {
            "principal_angles_deg": [float(a) for a in angles],
            "max_principal_angle_deg": max_angle_deg,
            "injected_amplitude": learned_amplitude,
        },
        "subspace_scaling": scaling,
        "fixed_phase_snr_loss": snr_loss,
        "gates": gates,
        "all_gates_passed": bool(all(g["passed"] for g in gates.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--sampling-frequency", type=float, default=10_000_000.0)
    parser.add_argument("--frequency", type=float, default=1_000_000.0)
    parser.add_argument("--snr", type=float, default=100.0, help="injected amplitude / predicted sigma")
    parser.add_argument(
        "--psd-kind",
        type=str,
        default="pink",
        choices=["white", "pink", "red", "brownian", "blue", "violet"],
    )
    parser.add_argument("--psd-knee-hz", type=float, default=None)
    parser.add_argument("--learned-subspace-snr-boost", type=float, default=1.0)
    parser.add_argument(
        "--scaling-snr-grid", type=float, nargs="+", default=[30.0, 100.0, 300.0, 1000.0]
    )
    parser.add_argument("--phase-errors", type=float, nargs="+", default=[0.0, 0.3, 0.6, 1.0])
    parser.add_argument("--data-root", type=Path, default=Path("/ceph/dwong/paper1_dataset"))
    parser.add_argument(
        "--contract", type=Path, default=REPO_ROOT / "docs" / "tidmad_data_contract.json"
    )
    parser.add_argument("--science-glob", type=str, default="tidmad/raw/abra_science_*.h5")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    record = run(args)

    output = args.output or (
        REPO_ROOT / "results" / "metrics" / f"TIDMAD_P1_{args.mode}_seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"TIDMAD Phase 1 ({args.mode}) -> {output}")
    for name, gate in record["gates"].items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{status}] {name}: {gate['value']:.6g} (threshold {gate['threshold']:g})")
    if not record["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
