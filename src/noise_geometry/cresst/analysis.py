"""CRESST pulse reconstruction with empirical noise geometry.

Two input paths feed the same OF/PCA/EMPCA/linear-AE comparison:

- the generic NPZ/HDF5 path (``input_file`` + ``trace_key``/``label_key``);
- the public DMDC release path (``release_format: true``), which uses the
  explicit ``noise``/``clean`` flags to separate noise-only baselines from
  accepted pulses. See arXiv:2508.03078.

The release path additionally implements the CRESST Validation Protocol
(``docs/EXPERIMENT_PROTOCOLS.md``): a per-channel, time-disjoint calibration
split (Sigma is a per-detector-channel quantity, so calibration/evaluation
never cross a channel boundary), a multi-template OF bank, detector-facing
metrics (whitening ratio, detection score, trigger efficiency vs. injected
amplitude anchored to each channel's published OF resolution), signal
injection (direct + measured-PSD), negative controls, and a paired-bootstrap
acceptance gate pooled across channels. See
``docs/CRESST_PREDECLARED_GATES.md`` for the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..autoencoders import tied_linear_ae_closed_form
from ..metrics import mse, weighted_residual
from ..noise import covariance_from_psd, estimate_covariance, estimate_psd_ensemble, inverse_covariance, regularize_covariance
from ..subspace import fit_pca, fit_weighted_pca, principal_angles, project_onto_basis
from ..subspace.pca import SubspaceFit
from ..utils.paths import dataset_root
from .detector_metrics import evaluate_trigger_efficiency, paired_bootstrap_stats, residual_whitening_ratio
from .loader import load_cresst_release, load_cresst_traces, load_resolutions, select_cresst_subsets
from .negative_controls import CONTROL_NOMINAL, compute_negative_control_residuals
from .splits import count_duplicate_traces, select_by_groups, time_disjoint_split, unique_groups
from .templates import build_multi_template_bank, score_multi_template_of


def _select_input(config: dict[str, Any], root: Path) -> Path:
    configured = config.get("input_file")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else root / "raw" / path
    candidates = sorted((root / "raw").glob("**/*.npz")) + sorted((root / "raw").glob("**/*.h5"))
    candidates += sorted((root / "raw").glob("**/*.hdf5"))
    if not candidates:
        raise FileNotFoundError(
            f"No CRESST NPZ/HDF5 files under {root / 'raw'}. "
            "Run scripts/download/download_cresst.py and place the released files there."
        )
    return candidates[0]


def _prep_traces(traces: np.ndarray, *, baseline_fraction: float, max_features: int) -> np.ndarray:
    """Baseline-subtract (per-trace) and downsample by striding."""
    traces = np.asarray(traces, dtype=np.float64)
    baseline_samples = max(1, int(traces.shape[1] * float(baseline_fraction)))
    traces = traces - traces[:, :baseline_samples].mean(axis=1, keepdims=True)
    step = max(1, int(np.ceil(traces.shape[1] / int(max_features))))
    return traces[:, ::step][:, : int(max_features)]


@dataclass
class BaselineFit:
    metric: np.ndarray
    rank: int
    pca: SubspaceFit
    empca: SubspaceFit
    pca_recon: np.ndarray
    empca_recon: np.ndarray
    ae_empca_max_principal_angle_deg: float
    of_template: np.ndarray
    of_amplitude: np.ndarray
    of_recon: np.ndarray


def _fit_baselines(
    train: np.ndarray,
    test: np.ndarray,
    noise_train: np.ndarray,
    rank: int,
    covariance_shrinkage: float,
    *,
    covariance_model: str = "empirical",
) -> BaselineFit:
    """Fit OF/PCA/EMPCA/linear-AE under the empirical noise metric.

    The OF template is built from ``train`` (the calibration segment), so
    injection studies that reuse ``of_template`` are, explicitly, injecting
    the channel's own single-OF template fit on its own calibration data.

    ``covariance_model``:

    - ``"empirical"`` (default, used by the generic NPZ/HDF5 path): the raw
      sample covariance, ``n_features * (n_features + 1) / 2`` free
      parameters.
    - ``"psd_stationary"`` (used by the per-channel CRESST release path,
      where calibration noise counts are only ~500-540 per channel against
      512 features -- N ~ p, so the raw sample covariance is
      near-singular): a circulant covariance built from a PSD estimate
      (only ``n_features // 2 + 1`` free parameters), matching the
      stationary/frequency-domain framing the rest of this codebase's
      OF/whitening machinery already assumes. See
      ``noise.covariance_from_psd`` and ``docs/CRESST_PREDECLARED_GATES.md``.
    """
    if covariance_model == "empirical":
        raw_covariance = estimate_covariance(noise_train)
    elif covariance_model == "psd_stationary":
        _, psd = estimate_psd_ensemble(noise_train, average="median")
        raw_covariance = covariance_from_psd(psd, noise_train.shape[1])
    else:
        raise ValueError(f"unknown covariance_model: {covariance_model!r}")
    covariance = regularize_covariance(raw_covariance, shrinkage=covariance_shrinkage)
    metric = inverse_covariance(covariance)

    pca = fit_pca(train, rank)
    empca = fit_weighted_pca(train, metric, rank)
    ae = tied_linear_ae_closed_form(train, rank, weights=metric)
    pca_recon = project_onto_basis(test, pca.components, mean=pca.mean)
    empca_recon = project_onto_basis(test, empca.components, weights=metric, mean=empca.mean)
    angles = principal_angles(empca.components, ae.components, weights=metric)

    pulse_strength = np.max(np.abs(train), axis=1)
    pulse_train = train[pulse_strength >= np.quantile(pulse_strength, 0.5)]
    template = pulse_train.mean(axis=0)
    template = template / max(np.max(np.abs(template)), np.finfo(float).eps)
    denom = float(template @ metric @ template)
    amplitude = (test @ metric @ template) / denom
    of_recon = amplitude[:, None] * template[None, :]

    return BaselineFit(
        metric=metric,
        rank=rank,
        pca=pca,
        empca=empca,
        pca_recon=pca_recon,
        empca_recon=empca_recon,
        ae_empca_max_principal_angle_deg=float(np.max(angles)),
        of_template=template,
        of_amplitude=amplitude,
        of_recon=of_recon,
    )


def _model_and_metrics(
    train: np.ndarray,
    test: np.ndarray,
    noise_train: np.ndarray,
    config: dict[str, Any],
    source_meta: Any,
    *,
    n_total: int,
) -> dict[str, Any]:
    """Fit OF/PCA/EMPCA/linear-AE under the empirical noise metric and score."""
    if noise_train.shape[0] < 2:
        raise ValueError("not enough noise-like traces to estimate covariance")
    rank = int(config.get("rank", 4))
    fit = _fit_baselines(train, test, noise_train, rank, float(config.get("covariance_shrinkage", 0.1)))

    return {
        "experiment": str(config.get("experiment_id", "P2_CRESST")),
        "source": source_meta,
        "n_traces": int(n_total),
        "n_noise_traces": int(noise_train.shape[0]),
        "n_features": int(train.shape[1]),
        "rank": rank,
        "pca_raw_mse": float(mse(test, fit.pca_recon)),
        "empca_raw_mse": float(mse(test, fit.empca_recon)),
        "pca_weighted_residual": float(np.mean(weighted_residual(test, fit.pca_recon, fit.metric))),
        "empca_weighted_residual": float(np.mean(weighted_residual(test, fit.empca_recon, fit.metric))),
        "of_weighted_residual": float(np.mean(weighted_residual(test, fit.of_recon, fit.metric))),
        "ae_empca_max_principal_angle_deg": fit.ae_empca_max_principal_angle_deg,
    }


def _use_release(config: dict[str, Any], root: Path) -> bool:
    if "release_format" in config:
        return bool(config["release_format"])
    split = str(config.get("split", "test"))
    return (root / "raw" / f"X_{split}.npy").exists()


def _reversed_sign_channels(per_channel: dict[str, Any], key: str, pooled_mean_difference: float) -> list[str]:
    pooled_sign = np.sign(pooled_mean_difference)
    return [
        label
        for label, entry in per_channel.items()
        if not entry.get("skipped")
        and entry[key]["mean_difference"] != 0
        and np.sign(entry[key]["mean_difference"]) != pooled_sign
    ]


def _run_release(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run the CRESST Validation Protocol comparison on the public DMDC release.

    Sigma (the noise covariance/metric) is a per-detector-channel quantity:
    calibration and evaluation are split *within* each (run, channel) channel,
    chronologically by ``hours``, never pooled across channels and never
    split by a cross-channel group assignment (which would test domain
    transfer, not the linear-hierarchy claim). The acceptance gate pools
    paired per-event differences across channels and requires the bootstrap
    95% CI to exclude zero, not a raw point-estimate win.
    """
    split = str(config.get("split", "test"))
    seed = int(config.get("seed", 22))
    traces, features, meta = load_cresst_release(
        root / "raw",
        split=split,
        max_traces=config.get("max_traces"),
        seed=seed,
    )
    subsets = select_cresst_subsets(
        traces,
        features,
        noise_column=str(config.get("noise_column", "noise")),
        clean_column=str(config.get("clean_column", "clean")),
    )
    if subsets["n_pulse"] < 4:
        raise ValueError(f"too few clean pulses in split '{split}' ({subsets['n_pulse']})")
    if subsets["n_noise"] < 2:
        raise ValueError(f"too few noise traces in split '{split}' ({subsets['n_noise']})")

    channel_columns = tuple(config.get("channel_columns", ["run", "channel"]))
    calibration_fraction = float(config.get("channel_calibration_fraction", 0.7))
    time_column = str(config.get("channel_time_column", "hours"))

    pulse_features = features[subsets["is_pulse"]].reset_index(drop=True)
    noise_features = features[subsets["is_noise"]].reset_index(drop=True)
    baseline_fraction = float(config.get("baseline_fraction", 0.2))
    max_features = int(config.get("max_features", 256))
    pulses = _prep_traces(subsets["pulse_traces"], baseline_fraction=baseline_fraction, max_features=max_features)
    noise = _prep_traces(subsets["noise_traces"], baseline_fraction=baseline_fraction, max_features=max_features)
    n_duplicate_traces = count_duplicate_traces(traces)

    try:
        resolutions = load_resolutions(root / "raw", split=split, group_columns=channel_columns)
    except FileNotFoundError:
        resolutions = {}

    rank = int(config.get("rank", 4))
    covariance_shrinkage = float(config.get("covariance_shrinkage", 0.1))
    covariance_model = str(config.get("covariance_model", "psd_stationary"))
    tb_config = dict(config.get("template_bank", {}))
    dm_config = dict(config.get("detector_metrics", {}))
    fpr_points = list(config.get("fpr_points", [0.001, 0.01, 0.05]))
    fpr_diagnostic_points = sorted(set(config.get("fpr_diagnostic_points", [0.001])))
    bootstrap_reps = int(config.get("bootstrap_reps", 300))
    default_controls = [
        "wrong_psd", "permuted_time_bins", "random_weighted_basis",
        "noise_only_train", "mismatched_of_template",
    ]
    controls = list(config.get("negative_controls", default_controls))
    amplitude_multiples = list(dm_config.get("amplitude_grid_multiples_of_sigma_of", [0, 1, 2, 3, 4, 5, 6, 8, 10]))
    n_injections = int(dm_config.get("n_injections_per_amplitude", 200))
    tolerance = float(config.get("multi_template_of_competitive_tolerance", 1.1))
    fpr_primary_points = sorted(set(config.get("fpr_primary_points", [0.05])))
    fpr_threshold_ci_points = sorted(set(config.get("fpr_threshold_ci_points", [0.01, 0.05])))

    channels = unique_groups(features, channel_columns)
    per_channel: dict[str, Any] = {}
    pooled_of, pooled_empca, pooled_pca, pooled_multi = [], [], [], []
    pooled_controls_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"control": [], "nominal": []} for name in controls
    }

    for ci, channel_key in enumerate(channels):
        channel_seed = seed + ci
        label = "_".join(str(v) for v in channel_key)

        pulse_mask = select_by_groups(pulse_features, channel_columns, {channel_key})
        noise_mask = select_by_groups(noise_features, channel_columns, {channel_key})
        channel_pulses = pulses[pulse_mask]
        channel_noise = noise[noise_mask]
        channel_pulse_features = pulse_features[pulse_mask].reset_index(drop=True)
        channel_noise_features = noise_features[noise_mask].reset_index(drop=True)

        if channel_pulses.shape[0] < 8 or channel_noise.shape[0] < 4:
            per_channel[label] = {
                "channel": list(channel_key),
                "skipped": True,
                "reason": f"too few rows (pulses={channel_pulses.shape[0]}, noise={channel_noise.shape[0]})",
            }
            continue

        pulse_hours = channel_pulse_features[time_column].to_numpy(dtype=float)
        noise_hours = channel_noise_features[time_column].to_numpy(dtype=float)
        pulse_calib, pulse_eval = time_disjoint_split(pulse_hours, calibration_fraction)
        noise_calib, noise_eval = time_disjoint_split(noise_hours, calibration_fraction)

        train = channel_pulses[pulse_calib]
        test = channel_pulses[pulse_eval]
        noise_train = channel_noise[noise_calib]
        noise_background = channel_noise[noise_eval]
        train_features_pulse = channel_pulse_features[pulse_calib].reset_index(drop=True)

        if train.shape[0] < 4 or test.shape[0] < 4 or noise_train.shape[0] < 2 or noise_background.shape[0] < 2:
            per_channel[label] = {
                "channel": list(channel_key),
                "skipped": True,
                "reason": (
                    f"time-disjoint split left too few rows (train={train.shape[0]}, test={test.shape[0]}, "
                    f"noise_train={noise_train.shape[0]}, background={noise_background.shape[0]})"
                ),
            }
            continue

        fit = _fit_baselines(train, test, noise_train, rank, covariance_shrinkage, covariance_model=covariance_model)
        pca_residual = weighted_residual(test, fit.pca_recon, fit.metric)
        empca_residual = weighted_residual(test, fit.empca_recon, fit.metric)
        of_residual = weighted_residual(test, fit.of_recon, fit.metric)

        bank = build_multi_template_bank(
            train,
            train_features_pulse["rise_time"].to_numpy(dtype=float),
            train_features_pulse["decay_time"].to_numpy(dtype=float),
            rise_time_quantiles=tuple(tb_config.get("rise_time_quantiles", (1 / 3, 2 / 3))),
            decay_time_quantiles=tuple(tb_config.get("decay_time_quantiles", (1 / 3, 2 / 3))),
            t0_grid_samples=tuple(tb_config.get("t0_grid_samples", (-4, -2, 0, 2, 4))),
            min_cell_pulses=int(tb_config.get("min_cell_pulses", 2)),
        )
        multi_scores = score_multi_template_of(test, bank, fit.metric)
        multi_residual = multi_scores["weighted_residual"]

        control_residuals = compute_negative_control_residuals(
            train, test, noise_train, fit.metric, rank, fit.of_template,
            seed=channel_seed, controls=controls,
        )

        residual_traces = test - fit.empca_recon
        whitening = residual_whitening_ratio(residual_traces, noise_train)

        sigma_of = resolutions.get(channel_key)
        if sigma_of is None or not np.isfinite(sigma_of) or sigma_of <= 0:
            # No published resolution for this channel (a real release gap, not
            # a code bug -- e.g. the test split's channel (35, 56) has no row in
            # resolutions_test.csv). Fall back to the same analytic definition
            # the published OF resolution uses: the CRB-style amplitude-estimator
            # std, sigma_OF = 1/sqrt(template^T Sigma^-1 template), from this
            # channel's own calibration-segment template/metric -- NOT the
            # spread of physical pulse amplitudes, which is a much larger and
            # physically different quantity.
            denom = float(fit.of_template @ fit.metric @ fit.of_template)
            sigma_of = float(1.0 / np.sqrt(denom)) if denom > 0 else 1.0
            sigma_of_source = "fallback_of_crb_no_published_resolution"
        else:
            sigma_of_source = "resolutions_test.csv"
        amplitude_grid = np.array([m * sigma_of for m in amplitude_multiples], dtype=float)

        trigger = evaluate_trigger_efficiency(
            noise_background,
            fit.of_template,
            fit.empca.components,
            fit.empca.mean,
            fit.metric,
            fpr_points=fpr_points,
            amplitude_grid=amplitude_grid,
            n_injections=n_injections,
            bootstrap_reps=bootstrap_reps,
            seed=channel_seed,
        )

        of_vs_empca = paired_bootstrap_stats(of_residual, empca_residual, bootstrap_reps=bootstrap_reps, seed=channel_seed)
        competitive = paired_bootstrap_stats(
            tolerance * multi_residual, empca_residual, bootstrap_reps=bootstrap_reps, seed=channel_seed + 1
        )
        control_stats = {
            name: paired_bootstrap_stats(
                arr,
                empca_residual if CONTROL_NOMINAL[name] == "empca" else of_residual,
                bootstrap_reps=bootstrap_reps,
                seed=channel_seed + 2,
            )
            for name, arr in control_residuals.items()
        }

        per_channel[label] = {
            "channel": list(channel_key),
            "skipped": False,
            "sigma_of": float(sigma_of),
            "sigma_of_source": sigma_of_source,
            "n_train_pulses": int(train.shape[0]),
            "n_test_pulses": int(test.shape[0]),
            "n_train_noise": int(noise_train.shape[0]),
            "n_background_noise": int(noise_background.shape[0]),
            "pca_raw_mse": float(mse(test, fit.pca_recon)),
            "empca_raw_mse": float(mse(test, fit.empca_recon)),
            "pca_weighted_residual": float(np.mean(pca_residual)),
            "empca_weighted_residual": float(np.mean(empca_residual)),
            "of_weighted_residual": float(np.mean(of_residual)),
            "ae_empca_max_principal_angle_deg": fit.ae_empca_max_principal_angle_deg,
            "multi_template_of": {
                "weighted_residual_mean": float(np.mean(multi_residual)),
                "n_templates": int(bank.templates.shape[0]),
                "cell_labels": bank.cell_labels,
                "t0_grid_samples": [int(s) for s in bank.t0_grid],
            },
            "negative_controls": control_stats,
            "of_vs_empca": of_vs_empca,
            "empca_competitive_with_multi_template_of": competitive,
            "detector_metrics": {
                "residual_whitening_ratio": {
                    "frequency_hz": [float(f) for f in whitening["frequency_hz"]],
                    "ratio": [float(r) for r in whitening["whitening_ratio"]],
                },
                "trigger_efficiency": trigger,
            },
        }

        pooled_of.append(of_residual)
        pooled_empca.append(empca_residual)
        pooled_pca.append(pca_residual)
        pooled_multi.append(multi_residual)
        for name, arr in control_residuals.items():
            pooled_controls_arrays[name]["control"].append(arr)
            pooled_controls_arrays[name]["nominal"].append(
                empca_residual if CONTROL_NOMINAL[name] == "empca" else of_residual
            )

    n_channels_used = sum(1 for v in per_channel.values() if not v.get("skipped"))
    if n_channels_used == 0:
        raise ValueError(
            "every channel was skipped (too few rows after the per-channel time-disjoint split); cannot proceed"
        )

    pooled_of_arr = np.concatenate(pooled_of)
    pooled_empca_arr = np.concatenate(pooled_empca)
    pooled_pca_arr = np.concatenate(pooled_pca)
    pooled_multi_arr = np.concatenate(pooled_multi)

    pooled_of_vs_empca = paired_bootstrap_stats(pooled_of_arr, pooled_empca_arr, bootstrap_reps=bootstrap_reps, seed=seed)
    pooled_competitive = paired_bootstrap_stats(
        tolerance * pooled_multi_arr, pooled_empca_arr, bootstrap_reps=bootstrap_reps, seed=seed + 1
    )
    pooled_controls: dict[str, dict] = {}
    for name in controls:
        control_list = pooled_controls_arrays[name]["control"]
        if not control_list:
            continue
        pooled_control_arr = np.concatenate(control_list)
        pooled_nominal_arr = np.concatenate(pooled_controls_arrays[name]["nominal"])
        pooled_controls[name] = paired_bootstrap_stats(
            pooled_control_arr, pooled_nominal_arr, bootstrap_reps=bootstrap_reps, seed=seed + 2
        )

    of_vs_empca_reversed = _reversed_sign_channels(per_channel, "of_vs_empca", pooled_of_vs_empca["mean_difference"])
    competitive_reversed = _reversed_sign_channels(
        per_channel, "empca_competitive_with_multi_template_of", pooled_competitive["mean_difference"]
    )
    controls_reversed = {
        name: [
            label
            for label, entry in per_channel.items()
            if not entry.get("skipped")
            and entry["negative_controls"][name]["mean_difference"] != 0
            and np.sign(entry["negative_controls"][name]["mean_difference"]) != np.sign(stats["mean_difference"])
        ]
        for name, stats in pooled_controls.items()
    }

    all_controls_degraded = bool(pooled_controls) and all(v["exceeds_zero"] for v in pooled_controls.values())
    empca_beats_of = bool(pooled_of_vs_empca["exceeds_zero"])
    empca_competitive = bool(pooled_competitive["exceeds_zero"])

    result: dict[str, Any] = {
        "experiment": str(config.get("experiment_id", "P2_CRESST")),
        "source": {**meta, "n_pulse": subsets["n_pulse"], "n_noise": subsets["n_noise"]},
        "rank": rank,
        "n_traces": int(pulses.shape[0]),
        "n_features": int(pulses.shape[1]),
        "n_channels_total": len(channels),
        "n_channels_used": n_channels_used,
        "n_duplicate_traces": n_duplicate_traces,
        "covariance_model": covariance_model,
        "n_noise_traces": int(
            sum(e["n_train_noise"] for e in per_channel.values() if not e.get("skipped"))
        ),
        "ae_empca_max_principal_angle_deg": float(
            max(e["ae_empca_max_principal_angle_deg"] for e in per_channel.values() if not e.get("skipped"))
        ),
        "pca_weighted_residual": float(np.mean(pooled_pca_arr)),
        "empca_weighted_residual": float(np.mean(pooled_empca_arr)),
        "of_weighted_residual": float(np.mean(pooled_of_arr)),
        "multi_template_of_weighted_residual": float(np.mean(pooled_multi_arr)),
        "split": {
            "method": "per_channel_time_disjoint",
            "channel_columns": list(channel_columns),
            "time_column": time_column,
            "calibration_fraction_declared": calibration_fraction,
            "channels": [list(c) for c in channels],
        },
        "per_channel": per_channel,
        "pooled": {
            "of_vs_empca": pooled_of_vs_empca,
            "of_vs_empca_sign_reversed_channels": of_vs_empca_reversed,
            "empca_competitive_with_multi_template_of": pooled_competitive,
            "empca_competitive_sign_reversed_channels": competitive_reversed,
            "negative_controls": pooled_controls,
            "negative_controls_sign_reversed_channels": controls_reversed,
        },
        "acceptance": {
            "passed": bool(empca_beats_of and empca_competitive and all_controls_degraded),
            "empca_beats_single_template_of": empca_beats_of,
            "empca_competitive_with_multi_template_of": empca_competitive,
            "multi_template_of_competitive_tolerance": tolerance,
            "all_negative_controls_degraded": all_controls_degraded,
            "aggregation": (
                "pooled paired per-event differences across all used channels; each gate "
                "requires the 95% bootstrap CI to exclude zero in the required direction "
                "(exceeds_zero), not a raw point-estimate win. Per-channel results and any "
                "sign reversal vs. the pooled direction are reported under 'per_channel' / "
                "'pooled.*_sign_reversed_channels'."
            ),
            "fpr_points": fpr_points,
            "fpr_primary": fpr_primary_points,
            "fpr_threshold_ci_reported": fpr_threshold_ci_points,
            "fpr_diagnostic_only": fpr_diagnostic_points,
            "criterion": (
                "EMPCA at rank k must beat single-template OF and remain competitive "
                "with multi-template OF on the likelihood-aligned metrics, with all "
                "negative controls degrading as expected -- each judged by a pooled "
                "paired-bootstrap 95% CI excluding zero, not a raw point estimate."
            ),
        },
    }

    return result


def run_cresst_experiment(config: dict[str, Any], data_root_path: str | Path) -> dict[str, Any]:
    """Run a first OF/PCA/EMPCA/linear-AE comparison on released traces."""
    root = dataset_root("cresst", data_root_path)
    if _use_release(config, root):
        return _run_release(config, root)

    source = _select_input(config, root)
    traces, labels, metadata = load_cresst_traces(
        source,
        trace_key=config.get("trace_key"),
        label_key=config.get("label_key"),
    )
    rng = np.random.default_rng(int(config.get("seed", 22)))
    max_traces = min(int(config.get("max_traces", 2000)), traces.shape[0])
    chosen = rng.choice(traces.shape[0], size=max_traces, replace=False)
    traces = traces[chosen]
    labels = labels[chosen] if labels is not None else None

    traces = _prep_traces(
        traces,
        baseline_fraction=float(config.get("baseline_fraction", 0.2)),
        max_features=int(config.get("max_features", 256)),
    )

    permutation = rng.permutation(traces.shape[0])
    split = max(2, int(0.7 * traces.shape[0]))
    train = traces[permutation[:split]]
    test = traces[permutation[split:]]
    train_labels = labels[permutation[:split]] if labels is not None else None

    noise_label = config.get("noise_label")
    if train_labels is not None and noise_label is not None:
        noise_train = train[train_labels == noise_label]
    else:
        peak = np.max(np.abs(train), axis=1)
        threshold = np.quantile(peak, float(config.get("noise_quantile", 0.2)))
        noise_train = train[peak <= threshold]

    return _model_and_metrics(train, test, noise_train, config, metadata, n_total=traces.shape[0])
