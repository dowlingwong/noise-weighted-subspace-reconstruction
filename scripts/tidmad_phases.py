"""TIDMAD Phases 2-5 runner.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md``.

    phase 2  metric reversal against injected truth          (gate G2)
    phase 3  rank sweep against the analytic quadrature       (gate G3)
    phase 4  calibration boundary, in-domain                  (control must pass)
    phase 5  denoised output scored by the upstream benchmark (gate G4)
    phase 6  five negative controls; every one must degrade

One runner rather than four because all four share the same window preparation,
the same noise model, and the same chronological split. Splitting them into
separate scripts would mean four copies of that pipeline, and four chances for
them to drift apart -- which is exactly the provenance problem the 2026-07-26
audit found elsewhere in this repository.

Modes
-----
``--mode synthetic`` builds windows from the repository's own noise generator
with an injected sinusoid of known amplitude and random phase. Every gate is
computable, so phases 2-4 can be exercised and regression-tested today.

``--mode file`` uses real TIDMAD windows: the SQUID channel as the observation
and the reference channel as ground truth. It requires a verified Phase 0 data
contract, and the loader enforces that.

Phase 5 requires real files in both modes -- there is nothing to score
otherwise -- and additionally a checkout of the upstream TIDMAD repository,
because the denoising score is defined by *its* ``benchmark.py`` and is
deliberately never reimplemented here.

    uv run python scripts/tidmad_phases.py --phase 2
    uv run python scripts/tidmad_phases.py --phase 3 --mode file \
        --data-root /ceph/dwong/paper1_dataset
    uv run python scripts/tidmad_phases.py --phase 5 --mode file \
        --data-root /ceph/dwong/paper1_dataset --upstream ~/src/TIDMAD
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
from noise_geometry.tidmad import (  # noqa: E402
    inject_sinusoid,
    nearest_bin_frequency,
    quadrature_amplitude_variance,
    quadrature_templates_rfft,
    welch_psd_from_windows,
)
from noise_geometry.tidmad.analysis import (  # noqa: E402
    build_window_set,
    run_metric_reversal,
    run_rank_sweep,
)
from noise_geometry.tidmad.negative_controls import run_negative_controls  # noqa: E402
from noise_geometry.tidmad.calibration import (  # noqa: E402
    assess_global_model,
    assess_local_model,
    cost_of_mismatch,
    score_dispersion,
)

# --- predeclared gates (fixed before running; do not tune) ------------------
#: G2 -- the reconstruction preferred by isotropic MSE must be *worse* against
#: truth, with a bootstrap interval excluding zero.
GATE_G2_REQUIRES_CI_ABOVE_ZERO = True
#: G3 -- rank 2 must beat rank 1 on amplitude error (CI above zero), and rank 3
#: must not improve on rank 2 by more than this fraction of the rank-2 RMSE.
GATE_G3_SATURATION_TOLERANCE = 0.05
#: Phase 4 -- the stationary synthetic control must land inside this band, or
#: the run halts: a failing control means a bug, not a finding.
CONTROL_CHI2_BAND = (0.9, 1.1)
CONTROL_DISPERSION_BAND = (0.9, 1.1)


# --------------------------------------------------------------------------
# window preparation
# --------------------------------------------------------------------------
def _synthetic_windows(args: argparse.Namespace) -> dict:
    """Observed = coloured noise + injected sinusoid; clean = the sinusoid."""
    rng = np.random.default_rng(args.seed)
    n = int(args.window_samples)
    fs = float(args.sampling_frequency)

    _, psd_truth = make_powerlaw_psd(n, fs, kind=args.psd_kind, knee_hz=args.psd_knee_hz)
    noise = generate_colored_noise(rng, psd_truth, n, fs, int(args.n_windows))
    calibration_noise = generate_colored_noise(rng, psd_truth, n, fs, int(args.n_calibration_windows))

    frequency = nearest_bin_frequency(args.frequency, n, fs)
    estimate = welch_psd_from_windows(
        list(calibration_noise), fs, window="boxcar", detrend=True, average="median", source="synthetic"
    )
    weights = inverse_psd_weights(estimate.psd, n)
    basis_f = quadrature_templates_rfft(n, fs, frequency)
    sigma = float(
        np.sqrt(
            quadrature_amplitude_variance(
                basis_f, estimate.psd, weights, trace_len=n, sampling_frequency=fs
            )
        )
    )
    amplitude = float(args.snr) * sigma

    observed, phases = inject_sinusoid(noise, fs, frequency, amplitude=amplitude, rng=rng)
    clean = observed - noise

    return {
        "observed": observed,
        "clean": clean,
        "noise_only": calibration_noise,
        "psd": estimate.psd,
        "psd_truth": psd_truth,
        "weights": weights,
        "sampling_frequency": fs,
        "frequency": frequency,
        "amplitude": amplitude,
        "predicted_sigma": sigma,
        "phases": phases,
        "source": f"synthetic {args.psd_kind} PSD",
    }


def _file_windows(args: argparse.Namespace) -> dict:
    """Observed = SQUID channel; clean = the injected reference channel."""
    from noise_geometry.tidmad.loader import TidmadFile, load_contract

    contract = load_contract(args.contract)
    contract.require_verified()
    if contract.reference_dataset is None:
        raise RuntimeError(
            "phases 2-3 need the reference channel as ground truth; point "
            "--validation-glob at training/validation files, not science files"
        )

    n = int(args.window_samples)
    validation = sorted(Path(args.data_root).glob(args.validation_glob))
    science = sorted(Path(args.data_root).glob(args.science_glob))
    if not validation:
        raise FileNotFoundError(f"no files matched {args.validation_glob!r}")
    if not science:
        raise FileNotFoundError(f"no files matched {args.science_glob!r}")

    observed, clean = [], []
    with TidmadFile(validation[0], contract) as handle:
        for squid, reference in handle.iter_paired_windows(n, stride=n, limit=int(args.n_windows)):
            observed.append(squid)
            clean.append(reference)

    noise_only = []
    with TidmadFile(science[0], contract) as handle:
        for window in handle.iter_squid_windows(
            n, stride=n, limit=int(args.n_calibration_windows)
        ):
            noise_only.append(window)

    observed = np.vstack(observed)
    clean = np.vstack(clean)
    noise_only = np.vstack(noise_only)

    fs = float(contract.sampling_frequency_hz)
    frequency = contract.frequency_for_file(
        validation[0].name, nearest_bin_frequency(args.frequency, n, fs)
    )
    estimate = welch_psd_from_windows(
        list(noise_only), fs, window="boxcar", detrend=True, average="median", source="science"
    )
    weights = inverse_psd_weights(estimate.psd, n)
    return {
        "observed": observed,
        "clean": clean,
        "noise_only": noise_only,
        "psd": estimate.psd,
        "psd_truth": None,
        "weights": weights,
        "sampling_frequency": fs,
        "frequency": nearest_bin_frequency(frequency, n, fs),
        "amplitude": None,
        "predicted_sigma": None,
        "phases": None,
        "source": f"TIDMAD files under {args.data_root}",
    }


def prepare(args: argparse.Namespace) -> dict:
    return _synthetic_windows(args) if args.mode == "synthetic" else _file_windows(args)


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
def phase2(args: argparse.Namespace, data: dict) -> dict:
    windows = build_window_set(
        data["observed"],
        data["clean"],
        data["psd"],
        data["weights"],
        data["sampling_frequency"],
        data["frequency"],
        fit_fraction=args.fit_fraction,
        guard_windows=args.guard_windows,
    )
    result = run_metric_reversal(
        windows,
        rank=int(args.rank),
        bootstrap_reps=int(args.bootstrap_reps),
        block_size=int(args.block_size),
        seed=int(args.seed),
    )

    truth_contrast = result["paired_contrasts_window_level"][
        "amplitude_abs_error_pca_minus_empca"
    ]
    block_contrast = result["paired_contrasts_block_level"][
        "amplitude_abs_error_pca_minus_empca"
    ]
    gates = {
        "G2_truth_favours_empca": {
            "value": truth_contrast["mean_difference"],
            "ci95": list(truth_contrast["ci95_difference"]),
            "passed": bool(truth_contrast["exceeds_zero"])
            if GATE_G2_REQUIRES_CI_ABOVE_ZERO
            else bool(truth_contrast["mean_difference"] > 0.0),
        },
        "G2_block_bootstrap_agrees": {
            "value": block_contrast["mean_difference"],
            "ci95": list(block_contrast["ci95_difference"]),
            "passed": bool(block_contrast["exceeds_zero"]),
        },
        "G2_metric_orderings_reversed": {
            "value": result["direction_check"]["orderings_reversed"],
            "passed": bool(result["direction_check"]["orderings_reversed"]),
        },
    }
    return {"phase": 2, "metric_reversal": result, "gates": gates}


def phase3(args: argparse.Namespace, data: dict) -> dict:
    windows = build_window_set(
        data["observed"],
        data["clean"],
        data["psd"],
        data["weights"],
        data["sampling_frequency"],
        data["frequency"],
        fit_fraction=args.fit_fraction,
        guard_windows=args.guard_windows,
    )
    result = run_rank_sweep(
        windows,
        ranks=tuple(int(r) for r in args.ranks),
        bootstrap_reps=int(args.bootstrap_reps),
        saturation_tolerance=GATE_G3_SATURATION_TOLERANCE,
        seed=int(args.seed),
    )

    contrasts = result["paired_contrasts"]
    rank2 = result["per_rank"].get("2", {})
    rank2_rmse = float(rank2.get("amplitude_rmse", np.nan))
    rank3_rmse = float(result["per_rank"].get("3", {}).get("amplitude_rmse", np.nan))
    saturation = (
        float((rank2_rmse - rank3_rmse) / rank2_rmse) if np.isfinite(rank3_rmse) and rank2_rmse > 0 else 0.0
    )

    gates = {
        "G3_rank2_beats_rank1": {
            "value": contrasts.get("amplitude_abs_error_rank1_minus_rank2", {}).get(
                "mean_difference", float("nan")
            ),
            "ci95": list(
                contrasts.get("amplitude_abs_error_rank1_minus_rank2", {}).get(
                    "ci95_difference", (float("nan"), float("nan"))
                )
            ),
            "passed": bool(
                contrasts.get("amplitude_abs_error_rank1_minus_rank2", {}).get("exceeds_zero", False)
            ),
        },
        "G3_rank3_does_not_improve": {
            "value": saturation,
            "threshold": GATE_G3_SATURATION_TOLERANCE,
            "passed": bool(saturation <= GATE_G3_SATURATION_TOLERANCE),
        },
        "G3_knee_is_rank2": {
            "value": result["smallest_rank_within_tolerance"],
            "passed": bool(int(result["smallest_rank_within_tolerance"]) == 2),
        },
        "G3_rank2_beats_fixed_phase": {
            "value": contrasts.get("amplitude_abs_error_fixed_phase_minus_rank2", {}).get(
                "mean_difference", float("nan")
            ),
            "passed": bool(
                contrasts.get("amplitude_abs_error_fixed_phase_minus_rank2", {}).get(
                    "exceeds_zero", False
                )
            ),
        },
    }
    return {"phase": 3, "rank_sweep": result, "gates": gates}


def phase4(args: argparse.Namespace, data: dict) -> dict:
    noise = np.asarray(data["noise_only"], dtype=np.float64)
    n_calibration = max(2, int(0.5 * noise.shape[0]))
    calibration, evaluation = noise[:n_calibration], noise[n_calibration:]
    if evaluation.shape[0] < 2:
        raise RuntimeError("need more noise-only windows to split calibration from evaluation")

    fs = data["sampling_frequency"]
    frequency = data["frequency"]

    # The stationary synthetic control is a halt gate, not a result: if a model
    # fitted on noise that genuinely is stationary and Gaussian fails to
    # calibrate, the failure is ours.
    rng = np.random.default_rng(int(args.seed) + 77)
    n = int(args.window_samples)
    _, control_psd = make_powerlaw_psd(n, fs, kind=args.psd_kind, knee_hz=args.psd_knee_hz)
    control_calibration = generate_colored_noise(rng, control_psd, n, fs, n_calibration)
    control_evaluation = generate_colored_noise(rng, control_psd, n, fs, evaluation.shape[0])
    control = assess_global_model(
        control_calibration, control_evaluation, fs, frequency, label="stationary_control"
    )

    control_ok = (
        CONTROL_CHI2_BAND[0] <= control.chi2_per_dof_mean <= CONTROL_CHI2_BAND[1]
        and CONTROL_DISPERSION_BAND[0] <= control.score_dispersion <= CONTROL_DISPERSION_BAND[1]
    )

    global_model = assess_global_model(calibration, evaluation, fs, frequency, label="global")
    local_models = []
    for radius in args.local_radii:
        try:
            local_models.append(
                assess_local_model(evaluation, fs, frequency, radius_windows=int(radius)).as_dict()
            )
        except ValueError as exc:  # not enough windows for this radius
            local_models.append({"label": f"local_radius_{int(radius)}", "error": str(exc)})

    # What the mismatch costs, in estimator terms rather than in chi^2.
    weights_global = inverse_psd_weights(
        welch_psd_from_windows(list(calibration), fs, window="boxcar").psd, n
    )
    _, matched_scores = score_dispersion(
        control_evaluation,
        welch_psd_from_windows(list(control_calibration), fs, window="boxcar").psd,
        inverse_psd_weights(
            welch_psd_from_windows(list(control_calibration), fs, window="boxcar").psd, n
        ),
        fs,
        frequency,
    )
    _, real_scores = score_dispersion(
        evaluation,
        welch_psd_from_windows(list(calibration), fs, window="boxcar").psd,
        weights_global,
        fs,
        frequency,
    )

    return {
        "phase": 4,
        "stationary_control": control.as_dict(),
        "global_model": global_model.as_dict(),
        "local_models": local_models,
        "cost_of_mismatch": {
            **cost_of_mismatch(matched_scores, real_scores),
            "is_null_in_synthetic_mode": bool(args.mode == "synthetic"),
            "note": (
                "In synthetic mode both arms are matched, so this is a null and "
                "should sit near 1.0; in file mode it is the real measurement -- "
                "what the inadequate noise model costs the amplitude estimate, "
                "which is the number GWOSC could not produce."
            ),
        },
        "gates": {
            "P4_control_passes": {
                "chi2_per_dof": control.chi2_per_dof_mean,
                "score_dispersion": control.score_dispersion,
                "chi2_band": list(CONTROL_CHI2_BAND),
                "dispersion_band": list(CONTROL_DISPERSION_BAND),
                "passed": bool(control_ok),
            }
        },
        "interpretation": (
            "The control is a halt gate. The global and local numbers are the "
            "boundary measurement: they are reported whichever way they fall, and "
            "the local radii were declared in the config before the run."
        ),
    }


def phase5(args: argparse.Namespace, data: dict) -> dict:
    from noise_geometry.tidmad.denoise import build_wiener_filter, denoise_stream
    from noise_geometry.tidmad.loader import TidmadFile, load_contract
    from noise_geometry.tidmad.score import DenoisedFileSpec, run_upstream_benchmark, write_denoised_file

    if args.mode != "file":
        raise SystemExit(
            "phase 5 requires --mode file: there is no denoising score without the "
            "release's own validation files and benchmark."
        )
    if args.upstream is None:
        raise SystemExit(
            "phase 5 requires --upstream <path to a jessicafry/TIDMAD checkout>. The "
            "denoising score is defined by its benchmark.py and is never "
            "reimplemented here."
        )

    contract = load_contract(args.contract)
    contract.require_verified()

    n = int(args.window_samples)
    fs = float(contract.sampling_frequency_hz)

    training = sorted(Path(args.data_root).glob(args.training_glob))
    validation = sorted(Path(args.data_root).glob(args.validation_glob))
    if not training:
        raise FileNotFoundError(f"no training files matched {args.training_glob!r}")
    if not validation:
        raise FileNotFoundError(f"no validation files matched {args.validation_glob!r}")

    # The filter is fitted on TRAINING files only -- the same information the
    # deep baselines are given -- and applied blind to validation.
    signal_windows = []
    with TidmadFile(training[0], contract) as handle:
        for window in handle.iter_squid_windows(n, stride=n, limit=int(args.n_windows)):
            signal_windows.append(window)
    wiener = build_wiener_filter(
        np.vstack(signal_windows), np.asarray(data["noise_only"]), fs, window="hann"
    )

    written = []
    for source in validation[: int(args.max_files)]:
        with TidmadFile(source, contract) as handle:
            stream = handle.read(contract.squid_dataset, 0, int(args.denoise_samples))
        denoised = denoise_stream(stream, wiener)
        spec = DenoisedFileSpec(
            source_file=source,
            destination_dir=Path(args.data_root) / "tidmad" / "denoised",
            model_tag=args.model_tag,
        )
        write_denoised_file(
            denoised,
            spec,
            source_dataset=contract.squid_dataset,
            attributes={"filter": wiener.as_dict()},
        )
        # benchmark.py always reads channel0002 too, for every model -- it
        # locates the injection frequency from the reference channel
        # regardless of which channel is under test. inference.py's own
        # output files carry it through unchanged; do the same here, or
        # upstream scoring fails with a missing-dataset KeyError.
        if contract.reference_dataset is not None:
            with TidmadFile(source, contract) as handle:
                reference = handle.read(contract.reference_dataset, 0, int(args.denoise_samples))
            write_denoised_file(
                reference,
                spec,
                source_dataset=contract.reference_dataset,
            )
        written.append(str(spec.destination))

    benchmark = run_upstream_benchmark(
        Path(args.upstream),
        Path(args.data_root) / "tidmad" / "denoised",
        args.model_tag,
        coarse=bool(args.coarse),
    )
    return {
        "phase": 5,
        "wiener_filter": wiener.as_dict(),
        "denoised_files": written,
        "upstream_benchmark": benchmark,
        "gates": {
            "G4_benchmark_ran": {
                "value": benchmark["returncode"],
                "passed": bool(benchmark["succeeded"]),
            }
        },
        "claim_boundary": (
            "This measures how much of the gap between no denoising and a trained "
            "deep denoiser is explained by using the correct noise geometry with a "
            "linear model. It is not a claim to win the benchmark."
        ),
    }


def phase6(args: argparse.Namespace, data: dict) -> dict:
    """Phase 6: break one ingredient at a time; every control must degrade."""
    windows = build_window_set(
        data["observed"],
        data["clean"],
        data["psd"],
        data["weights"],
        data["sampling_frequency"],
        data["frequency"],
        fit_fraction=args.fit_fraction,
        guard_windows=args.guard_windows,
    )
    result = run_negative_controls(
        windows,
        data["noise_only"],
        rank=int(args.rank),
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed),
        wrong_frequency_offset_bins=int(args.wrong_frequency_offset_bins),
    )

    gates = {
        f"P6_{name}_degrades": {
            "value": result["paired_contrasts"][name]["mean_difference"],
            "ci95": list(result["paired_contrasts"][name]["ci95_difference"]),
            "passed": bool(result["degraded"][name]),
        }
        for name in result["degraded"]
    }
    return {"phase": 6, "negative_controls": result, "gates": gates}


PHASES = {2: phase2, 3: phase3, 4: phase4, 5: phase5, 6: phase6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASES))
    parser.add_argument("--mode", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--n-calibration-windows", type=int, default=256)
    parser.add_argument("--sampling-frequency", type=float, default=10_000_000.0)
    parser.add_argument("--frequency", type=float, default=1_000_000.0)
    parser.add_argument("--snr", type=float, default=8.0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--guard-windows", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--wrong-frequency-offset-bins", type=int, default=37)
    parser.add_argument("--local-radii", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--psd-kind",
        type=str,
        default="pink",
        choices=["white", "pink", "red", "brownian", "blue", "violet"],
    )
    parser.add_argument("--psd-knee-hz", type=float, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("/ceph/dwong/paper1_dataset"))
    parser.add_argument(
        "--contract", type=Path, default=REPO_ROOT / "docs" / "tidmad_data_contract.json"
    )
    parser.add_argument("--training-glob", type=str, default="tidmad/raw/abra_training_*.h5")
    parser.add_argument("--validation-glob", type=str, default="tidmad/raw/abra_validation_*.h5")
    parser.add_argument("--science-glob", type=str, default="tidmad/raw/abra_science_*.h5")
    parser.add_argument("--upstream", type=Path, default=None)
    parser.add_argument("--model-tag", type=str, default="wiener")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--denoise-samples", type=int, default=1 << 22)
    parser.add_argument("--coarse", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = prepare(args)
    record = PHASES[args.phase](args, data)
    record.update(
        {
            "experiment": f"TIDMAD-P{args.phase}",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": args.mode,
            "noise_source": data["source"],
            "configuration": {
                "seed": int(args.seed),
                "window_samples": int(args.window_samples),
                "n_windows": int(args.n_windows),
                "n_calibration_windows": int(args.n_calibration_windows),
                "sampling_frequency_hz": float(data["sampling_frequency"]),
                "on_grid_frequency_hz": float(data["frequency"]),
                "snr": float(args.snr) if args.mode == "synthetic" else None,
                "injected_amplitude": data["amplitude"],
                "rank": int(args.rank),
                "ranks": [int(r) for r in args.ranks],
                "fit_fraction": float(args.fit_fraction),
                "guard_windows": int(args.guard_windows),
                "bootstrap_reps": int(args.bootstrap_reps),
                "block_size": int(args.block_size),
                "local_radii": [int(r) for r in args.local_radii],
            },
        }
    )
    record["all_gates_passed"] = bool(
        all(gate.get("passed", False) for gate in record.get("gates", {}).values())
    )

    output = args.output or (
        REPO_ROOT / "results" / "metrics" / f"TIDMAD_P{args.phase}_{args.mode}_seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"TIDMAD Phase {args.phase} ({args.mode}) -> {output}")
    for name, gate in record.get("gates", {}).items():
        status = "PASS" if gate.get("passed") else "FAIL"
        value = gate.get("value")
        shown = f"{value:.6g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        print(f"  [{status}] {name}: {shown}")
    if not record["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
