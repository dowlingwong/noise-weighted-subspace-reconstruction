from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "src").exists() and (candidate / "implementation").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


REPO_ROOT = _find_repo_root()
NPML_DIR = REPO_ROOT / "NPML"
FIG_DIR = NPML_DIR / "figures"
TABLE_DIR = NPML_DIR / "tables"

for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "EMPCA",
    REPO_ROOT / "QP_simulator",
    REPO_ROOT / "implementation",
    NPML_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from implementation.notebook_support import (  # noqa: E402
    CanonicalConfig,
    _linear_map_from_coeff,
    _predict_from_linear_map,
    broaden_trace,
    build_of_one_sided_weights,
    choose_noise_power_for_target_fisher,
    compute_of_amplitudes,
    compute_of_statistics,
    compute_residual_summary,
    exact_isotropic_subspace,
    exact_weighted_subspace,
    fit_weighted_empca,
    make_clean_qp_trace,
    normalize_template_peak,
    phase_align_basis,
    prepare_bundle,
    principal_angles_weighted,
    rankk_gls_coefficients,
    residual_energy_per_trace,
    run_block04_theorem_suite,
    run_block07_ablation_suite,
    safe_sqrt_weights,
    shift_trace_fractional,
    stationary_noise_generator,
    weighted_cosine,
)


plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 160)


def ensure_npml_dirs() -> dict[str, Path]:
    dirs = {
        "root": NPML_DIR,
        "figures": FIG_DIR,
        "tables": TABLE_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_dataframe(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_figure(fig, name: str, dpi: int = 180) -> Path:
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def make_npml_cfg(seed: int = 20260609) -> CanonicalConfig:
    cfg = replace(
        CanonicalConfig(seed=seed),
        trace_len=2048,
        pretrigger=256,
        sim_events_small=180,
        sim_events_medium=240,
        sim_events_large=320,
        default_empca_iter=40,
        default_empca_patience=8,
        crb_replicates=320,
    )
    return cfg.validate()


def pdf_experiment_plan() -> pd.DataFrame:
    rows = [
        (
            "A",
            "Metric Ablation",
            "PCA vs EMPCA under white, pink, brownian, MMC PSD",
            "weighted residual; reconstruction MSE; principal angle; amplitude resolution",
            "Synthetic traces with colored Gaussian noise",
        ),
        (
            "B",
            "Coverage Ablation",
            "Full latent coverage vs timing/position/shape restricted training",
            "amplitude RMSE; timing RMSE; position RMSE; weighted residual",
            "Train restricted, test always full",
        ),
        (
            "C",
            "NFPA vs EMPCA",
            "Separable vs non-separable multichannel signals",
            "weighted residual; reconstruction error; subspace angle",
            "Tests channel-time factorization",
        ),
        (
            "D",
            "Architecture Bias",
            "Linear AE vs CNN AE vs Transformer AE",
            "weighted residual; reconstruction RMSE; unseen position/shape generalization",
            "Identical Mahalanobis objective",
        ),
        (
            "E",
            "Prewhitened Transformer",
            "raw/prewhitened x MSE/Mahalanobis",
            "weighted residual; amplitude/timing resolution; generalization",
            "Attention in detector geometry",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["experiment", "title", "design", "metrics", "goal"],
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def audit_src_models() -> pd.DataFrame:
    torch_ok = _torch_available()
    muon_present = bool(list(REPO_ROOT.rglob("muon.py")))
    norm_present = bool(list(REPO_ROOT.rglob("norm.py")))
    mmc_psd_path = REPO_ROOT / "data" / "Noise_PSD" / "noise_psd_from_MMC.npy"
    mmc_psd_present = mmc_psd_path.exists()

    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "experiment": "A",
            "model_family": "PCA baseline",
            "path": "implementation.notebook_support exact_isotropic_subspace",
            "status": "ready",
            "runnable_now": True,
            "notes": "Exact isotropic SVD baseline exists in implementation helpers.",
        }
    )
    rows.append(
        {
            "experiment": "A/B",
            "model_family": "EMPCA",
            "path": "src/EMPCA/empca_TCY_optimized.py",
            "status": "ready",
            "runnable_now": True,
            "notes": "Weighted EMPCA solver is importable and already wired into implementation.notebook_support.",
        }
    )
    rows.append(
        {
            "experiment": "C",
            "model_family": "NFPA demo",
            "path": "src/NFPA/nfpa_demo.py",
            "status": "partial",
            "runnable_now": True,
            "notes": "Pure numpy demo is runnable as a script, but it executes on import and hardcodes a plot write side effect.",
        }
    )
    rows.append(
        {
            "experiment": "D",
            "model_family": "Linear AE",
            "path": "implementation.notebook_support exact_weighted_subspace",
            "status": "partial",
            "runnable_now": True,
            "notes": "Exact weighted SVD gives the linear-AE optimum, but there is no standalone training wrapper under src/.",
        }
    )
    rows.append(
        {
            "experiment": "D",
            "model_family": "CNN backbone",
            "path": "src/CNN/resnet_1d.py",
            "status": "blocked",
            "runnable_now": False,
            "notes": (
                "Torch installed? "
                f"{torch_ok}. Local norm dependency present? {norm_present}. "
                "File defines a ResNet backbone, not a reconstruction autoencoder or trainer."
            ),
        }
    )
    for name in sorted((REPO_ROOT / "src" / "transformer").glob("*.py")):
        rows.append(
            {
                "experiment": "D/E",
                "model_family": name.stem,
                "path": str(name.relative_to(REPO_ROOT)),
                "status": "blocked",
                "runnable_now": False,
                "notes": (
                    "Requires torch and reconstruction.training.muon; "
                    f"torch installed={torch_ok}, muon module present={muon_present}. "
                    "These files also define transformer backbones only, not end-to-end reconstruction experiments."
                ),
            }
        )
    rows.append(
        {
            "experiment": "A",
            "model_family": "Measured MMC PSD",
            "path": "data/Noise_PSD/noise_psd_from_MMC.npy",
            "status": "blocked" if not mmc_psd_present else "ready",
            "runnable_now": mmc_psd_present,
            "notes": "The measured MMC PSD is available in the new data/Noise_PSD layout.",
        }
    )
    rows.append(
        {
            "experiment": "B/D/E",
            "model_family": "Position-aware simulator",
            "path": "QP_simulator/QPSimulator.py",
            "status": "partial",
            "runnable_now": True,
            "notes": "Current simulator exposes amplitude, tau_decay, and t0. There is no native detector-position latent, so coverage notebooks must synthesize a position-like distortion.",
        }
    )
    return pd.DataFrame(rows)


def experiment_need_table() -> pd.DataFrame:
    rows = [
        ("A", "Add measured MMC PSD file if you want the fourth noise condition exactly as written in the PDF."),
        ("B", "Implement a native position latent in the simulator if you want physically grounded position RMSE rather than a proxy distortion."),
        ("C", "Refactor src/NFPA/nfpa_demo.py into callable functions if you want it reusable outside a demo script."),
        ("D", "Add a reconstruction training stack: linear AE wrapper, CNN AE wrapper, shared Mahalanobis loss, and evaluation harness."),
        ("D", "Install torch and provide the missing local dependency used by src/CNN/resnet_1d.py."),
        ("E", "Install torch, provide reconstruction.training.muon, and add transformer reconstruction heads plus a whitening pipeline."),
        ("E", "Add four controlled training configs: raw+mse, raw+mahalanobis, whitened+mse, whitened+mahalanobis."),
    ]
    return pd.DataFrame(rows, columns=["experiment", "need"])


def _basis_reconstruction_time(
    traces_freq: np.ndarray,
    basis_f: np.ndarray,
    weights: np.ndarray,
    trace_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    coeff = rankk_gls_coefficients(traces_freq, basis_f, weights, return_complex=True)
    recon_f = coeff[:, None] * basis_f[None, :] if np.asarray(basis_f).ndim == 1 else coeff @ basis_f
    recon_t = np.fft.irfft(recon_f, n=trace_len, axis=-1)
    return np.asarray(coeff), np.asarray(recon_t, dtype=np.float64)


def run_metric_ablation(seed: int = 20260609, k: int = 4) -> dict[str, pd.DataFrame]:
    cfg = make_npml_cfg(seed=seed)
    rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []

    for noise_type in ("white", "pink", "brownian"):
        bundle, truth = __import__("implementation.notebook_support", fromlist=["simulate_controlled_family"]).simulate_controlled_family(
            cfg,
            n_events=cfg.sim_events_medium,
            tau_decay_range=(1.5e6, 4.5e6),
            t0_jitter_range=(-5e4, 5e4),
            n_qp_range=(3000, 7000),
            noise_type=noise_type,
            noise_power=1.0,
            arrival_mode="aligned",
        )
        train_idx = bundle.split_indices["train"]
        test_idx = bundle.split_indices["test"]
        emp_basis = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=k)["basis_native"]
        pca_basis = exact_isotropic_subspace(bundle.traces_freq[train_idx], k=k)
        cosines, angles = principal_angles_weighted(emp_basis, pca_basis, bundle.weights_one_sided)
        angle_rows.append(
            {
                "noise_type": noise_type,
                "mean_principal_angle_deg": float(np.mean(angles)),
                "max_principal_angle_deg": float(np.max(angles)),
                "min_principal_cosine": float(np.min(cosines)),
            }
        )

        for method, basis in (("EMPCA", emp_basis), ("PCA", pca_basis)):
            coeff_train = rankk_gls_coefficients(
                bundle.traces_freq[train_idx],
                basis,
                bundle.weights_one_sided,
                return_complex=False,
            )
            coeff_test, recon_t = _basis_reconstruction_time(
                bundle.traces_freq[test_idx],
                basis,
                bundle.weights_one_sided,
                cfg.trace_len,
            )
            amp_map = _linear_map_from_coeff(coeff_train, truth["amplitude_true"][train_idx])
            amp_pred = _predict_from_linear_map(np.real(coeff_test), amp_map)
            resid = residual_energy_per_trace(
                bundle.traces_freq[test_idx],
                basis,
                coeff_test,
                bundle.weights_one_sided,
            )
            rows.append(
                {
                    "noise_type": noise_type,
                    "method": method,
                    "weighted_residual_mean": float(np.mean(resid)),
                    "weighted_residual_std": float(np.std(resid)),
                    "reconstruction_mse_clean": float(np.mean((recon_t - truth["clean_traces"][test_idx]) ** 2)),
                    "amplitude_rmse": float(np.sqrt(np.mean((amp_pred - truth["amplitude_true"][test_idx]) ** 2))),
                }
            )

    metric_df = pd.DataFrame(rows)
    angle_df = pd.DataFrame(angle_rows)
    return {"metric_df": metric_df, "angle_df": angle_df}


def _coverage_components(cfg: CanonicalConfig) -> dict[str, np.ndarray]:
    base, _ = make_clean_qp_trace(cfg, n_qp=5000, tau_decay=3e6, arrival_mode="aligned")
    base = normalize_template_peak(base, cfg.pretrigger)
    timing = shift_trace_fractional(base, +2.5) - shift_trace_fractional(base, -2.5)
    shape_lo, _ = make_clean_qp_trace(cfg, n_qp=5000, tau_decay=1.5e6, arrival_mode="aligned")
    shape_hi, _ = make_clean_qp_trace(cfg, n_qp=5000, tau_decay=5.0e6, arrival_mode="aligned")
    shape = normalize_template_peak(shape_hi, cfg.pretrigger) - normalize_template_peak(shape_lo, cfg.pretrigger)
    position = broaden_trace(base, sigma=8.0) - broaden_trace(base, sigma=2.0)
    for key, vec in {"timing": timing, "shape": shape, "position": position}.items():
        norm = max(np.linalg.norm(vec), np.finfo(float).eps)
        if key == "timing":
            timing = vec / norm
        elif key == "shape":
            shape = vec / norm
        else:
            position = vec / norm
    return {"base": base, "timing": timing, "shape": shape, "position": position}


def _generate_coverage_dataset(
    cfg: CanonicalConfig,
    n_events: int,
    condition: str,
    noise_type: str = "pink",
    noise_power: float = 0.05,
    seed: int = 0,
) -> tuple[Any, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    comp = _coverage_components(cfg)
    base = comp["base"]
    timing = comp["timing"]
    shape = comp["shape"]
    position = comp["position"]

    amp = rng.uniform(0.75, 1.25, size=n_events)
    t0 = rng.uniform(-1.0, 1.0, size=n_events)
    pos = rng.uniform(-1.0, 1.0, size=n_events)
    gamma = rng.uniform(-1.0, 1.0, size=n_events)

    if condition == "timing_restricted":
        t0[:] = 0.0
    elif condition == "position_restricted":
        pos[:] = 0.0
    elif condition == "shape_restricted":
        gamma[:] = 0.0
    elif condition != "full":
        raise ValueError(f"Unsupported condition: {condition}")

    clean = np.zeros((n_events, cfg.trace_len), dtype=np.float64)
    for idx in range(n_events):
        trace = (
            amp[idx] * base
            + 2.20 * t0[idx] * timing
            + 2.40 * pos[idx] * position
            + 2.40 * gamma[idx] * shape
        )
        clean[idx] = trace

    ng = stationary_noise_generator(cfg, noise_type=noise_type, noise_power=noise_power, rng=rng)
    _, psd = ng.build_psd(cfg.trace_len)
    noise = np.stack([ng.generate_noise(cfg.trace_len) for _ in range(n_events)], axis=0)
    bundle = prepare_bundle(
        clean + noise,
        template_time=base,
        psd_one_sided=psd,
        cfg=cfg,
        metadata={
            "source": f"NPML-coverage-{condition}",
            "noise_type": noise_type,
            "noise_power": float(noise_power),
            "template_path": "NPML synthetic proxy family",
            "psd_path": f"analytic:{noise_type}",
        },
    )
    truth = {
        "clean_traces": clean,
        "noise_traces": noise,
        "amplitude_true": amp,
        "t0_samples": t0,
        "position_true": pos,
        "shape_true": gamma,
    }
    return bundle, truth


def run_coverage_ablation(seed: int = 20260609, k: int = 4) -> dict[str, pd.DataFrame]:
    cfg = make_npml_cfg(seed=seed)
    conditions = ["full", "timing_restricted", "position_restricted", "shape_restricted"]
    train_data = {
        condition: _generate_coverage_dataset(
            cfg,
            n_events=cfg.sim_events_medium,
            condition=condition,
            seed=seed + idx,
        )
        for idx, condition in enumerate(conditions)
    }
    test_bundle, test_truth = _generate_coverage_dataset(
        cfg,
        n_events=cfg.sim_events_small,
        condition="full",
        seed=seed + 100,
    )

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        train_bundle, train_truth = train_data[condition]
        train_idx = train_bundle.split_indices["train"]
        weights = train_bundle.weights_one_sided
        emp_basis = exact_weighted_subspace(train_bundle.traces_freq[train_idx], weights, k=k)["basis_native"]
        pca_basis = exact_isotropic_subspace(train_bundle.traces_freq[train_idx], k=k)

        for method, basis in (("EMPCA", emp_basis), ("PCA", pca_basis)):
            coeff_train = rankk_gls_coefficients(
                train_bundle.traces_freq[train_idx],
                basis,
                weights,
                return_complex=False,
            )
            maps = {
                "amplitude": _linear_map_from_coeff(coeff_train, train_truth["amplitude_true"][train_idx]),
                "timing": _linear_map_from_coeff(coeff_train, train_truth["t0_samples"][train_idx]),
                "position": _linear_map_from_coeff(coeff_train, train_truth["position_true"][train_idx]),
                "shape": _linear_map_from_coeff(coeff_train, train_truth["shape_true"][train_idx]),
            }
            coeff_test = rankk_gls_coefficients(
                test_bundle.traces_freq,
                basis,
                test_bundle.weights_one_sided,
                return_complex=True,
            )
            coeff_test_real = np.real(coeff_test)
            resid = residual_energy_per_trace(
                test_bundle.traces_freq,
                basis,
                coeff_test,
                test_bundle.weights_one_sided,
            )
            rows.append(
                {
                    "training_condition": condition,
                    "method": method,
                    "weighted_residual_mean": float(np.mean(resid)),
                    "amplitude_rmse": float(
                        np.sqrt(
                            np.mean(
                                (_predict_from_linear_map(coeff_test_real, maps["amplitude"]) - test_truth["amplitude_true"]) ** 2
                            )
                        )
                    ),
                    "timing_rmse": float(
                        np.sqrt(np.mean((_predict_from_linear_map(coeff_test_real, maps["timing"]) - test_truth["t0_samples"]) ** 2))
                    ),
                    "position_rmse": float(
                        np.sqrt(
                            np.mean(
                                (_predict_from_linear_map(coeff_test_real, maps["position"]) - test_truth["position_true"]) ** 2
                            )
                        )
                    ),
                    "shape_rmse": float(
                        np.sqrt(
                            np.mean(
                                (_predict_from_linear_map(coeff_test_real, maps["shape"]) - test_truth["shape_true"]) ** 2
                            )
                        )
                    ),
                }
            )

    coverage_df = pd.DataFrame(rows)
    base = coverage_df[(coverage_df["training_condition"] == "full") & (coverage_df["method"] == "EMPCA")].iloc[0]
    matrix_rows = []
    for metric_name, metric_label in (
        ("weighted_residual_mean", "weighted_residual"),
        ("amplitude_rmse", "amplitude_rmse"),
        ("timing_rmse", "timing_rmse"),
        ("position_rmse", "position_rmse"),
    ):
        matrix_rows.append(
            {
                "metric": metric_label,
                "correct_metric_full_coverage": 1.0,
                "wrong_metric_full_coverage": float(
                    coverage_df[(coverage_df["training_condition"] == "full") & (coverage_df["method"] == "PCA")][metric_name].iloc[0]
                    / max(base[metric_name], 1e-12)
                ),
                "correct_metric_restricted_coverage": float(
                    coverage_df[(coverage_df["training_condition"] != "full") & (coverage_df["method"] == "EMPCA")][metric_name].max()
                    / max(base[metric_name], 1e-12)
                ),
                "wrong_metric_restricted_coverage": float(
                    coverage_df[(coverage_df["training_condition"] != "full") & (coverage_df["method"] == "PCA")][metric_name].max()
                    / max(base[metric_name], 1e-12)
                ),
            }
        )
    matrix_df = pd.DataFrame(matrix_rows)
    return {"coverage_df": coverage_df, "matrix_df": matrix_df}


def _brownian_psd(T: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(T, d=1.0)
    f_safe = np.where(freqs == 0, 1.0, np.abs(freqs))
    psd = 1.0 / np.maximum(f_safe**2, 1e-12)
    psd[0] = psd[1]
    return psd


def _whitening_filters(psd: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    white_f = np.where(psd > 0, 1.0 / np.sqrt(psd), 0.0)
    ref = _gen_noise_from_psd(2000, psd, T, np.random.default_rng(0))
    ref_w = np.fft.irfft(np.fft.rfft(ref, axis=-1) * white_f[None, :], n=T, axis=-1)
    scale = np.std(ref_w)
    white_f = white_f / max(scale, 1e-12)
    color_f = np.where(white_f > 1e-10, 1.0 / white_f, 0.0)
    return white_f, color_f


def _gen_noise_from_psd(N: int, psd: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    nf = len(psd)
    amp = np.sqrt(np.maximum(psd, 0.0) / (2.0 * T))
    Xf = (rng.standard_normal((N, nf)) + 1j * rng.standard_normal((N, nf))) * amp[None, :]
    Xf[:, 0] = 0.0
    return np.fft.irfft(Xf, n=T, axis=-1)


def _nfpa_als(Z: np.ndarray, k_c: int, k_t: int, n_iter: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    N, C, T = Z.shape
    rng = np.random.default_rng(seed)
    U_c = np.linalg.qr(rng.standard_normal((C, k_c)))[0]
    U_t = np.linalg.qr(rng.standard_normal((T, k_t)))[0]
    hist: list[float] = []

    for _ in range(n_iter):
        D = np.einsum("nct,ck->nkt", Z, U_c)
        _, _, Vt = np.linalg.svd(D.reshape(N * k_c, T), full_matrices=False)
        U_t = Vt[:k_t].T

        E = np.einsum("nct,tl->ncl", Z, U_t)
        _, _, Vt = np.linalg.svd(E.transpose(0, 2, 1).reshape(N * k_t, C), full_matrices=False)
        U_c = Vt[:k_c].T

        coeff = np.einsum("nct,ck,tl->nkl", Z, U_c, U_t)
        recon = np.einsum("nkl,ck,tl->nct", coeff, U_c, U_t)
        hist.append(float(np.mean((Z - recon) ** 2)))

    return U_c, U_t, hist


def _kron_basis(U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.kron(U_c[:, j], U_t[:, i]) for j in range(U_c.shape[1]) for i in range(U_t.shape[1])]
    )


def _principal_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(np.abs(s), 0.0, 1.0)))


def _make_nfpa_dataset(
    seed: int,
    distortion_level: float,
    N_tr: int = 600,
    N_te: int = 240,
    C: int = 4,
    T: int = 192,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    t0 = 0.35 * T
    base_t = np.where(
        t < t0,
        np.exp(-((t - t0) ** 2) / (2.0 * 3.0**2)),
        np.exp(-((t - t0) ** 2) / (2.0 * 16.0**2)),
    )
    base_t = base_t / np.max(np.abs(base_t))
    distortion_t = shift_trace_fractional(base_t, +3.0) - shift_trace_fractional(base_t, -3.0)
    distortion_t = distortion_t / max(np.linalg.norm(distortion_t), np.finfo(float).eps)
    v_true = np.array([1.0, 0.75, 0.5, 0.3], dtype=np.float64)
    v_dist = np.array([-0.4, -0.15, 0.15, 0.35], dtype=np.float64)

    def make_signals(N: int) -> tuple[np.ndarray, np.ndarray]:
        A = rng.normal(1.0, 0.22, size=N)
        p = rng.uniform(-1.0, 1.0, size=N)
        X = np.zeros((N, C, T), dtype=np.float64)
        for idx in range(N):
            for c in range(C):
                time_profile = base_t + distortion_level * p[idx] * v_dist[c] * distortion_t
                X[idx, c] = A[idx] * v_true[c] * time_profile
        return X, p

    X_tr_clean, p_tr = make_signals(N_tr)
    X_te_clean, p_te = make_signals(N_te)
    psd = _brownian_psd(T)
    white_f, color_f = _whitening_filters(psd, T)

    def whiten(x: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(x, axis=-1) * white_f[None, None, :], n=T, axis=-1)

    def unwhiten(x: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(x, axis=-1) * color_f[None, None, :], n=T, axis=-1)

    noise_tr = np.stack([_gen_noise_from_psd(N_tr, psd, T, rng) for _ in range(C)], axis=1)
    noise_te = np.stack([_gen_noise_from_psd(N_te, psd, T, rng) for _ in range(C)], axis=1)
    X_tr = X_tr_clean + noise_tr
    X_te = X_te_clean + noise_te
    Z_tr = whiten(X_tr)
    Z_te = whiten(X_te)
    return {
        "X_tr": X_tr,
        "X_te": X_te,
        "Z_tr": Z_tr,
        "Z_te": Z_te,
        "X_te_clean": X_te_clean,
        "whiten": whiten,
        "unwhiten": unwhiten,
        "p_tr": p_tr,
        "p_te": p_te,
    }


def run_nfpa_regime_sweep(seed: int = 20260609) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    regimes = [
        ("separable", 0.00),
        ("mild_nonseparable", 0.40),
        ("strong_nonseparable", 0.80),
    ]
    for idx, (label, distortion) in enumerate(regimes):
        data = _make_nfpa_dataset(seed + idx, distortion_level=distortion)
        Z_tr = data["Z_tr"]
        Z_te = data["Z_te"]
        X_te = data["X_te"]
        X_te_clean = data["X_te_clean"]
        N_tr, C, T = Z_tr.shape
        N_te = Z_te.shape[0]
        k_c = 1
        k_t = 1
        k_tot = k_c * k_t

        U_c, U_t, hist = _nfpa_als(Z_tr, k_c=k_c, k_t=k_t, n_iter=80, seed=seed + idx)
        coeff = np.einsum("nct,ck,tl->nkl", Z_te, U_c, U_t)
        Z_nfpa = np.einsum("nkl,ck,tl->nct", coeff, U_c, U_t)
        X_nfpa = data["unwhiten"](Z_nfpa)

        Z_tr_flat = Z_tr.reshape(N_tr, -1)
        Z_te_flat = Z_te.reshape(N_te, -1)
        _, _, vh = np.linalg.svd(Z_tr_flat, full_matrices=False)
        U_empca = vh[:k_tot]
        coeff_emp = Z_te_flat @ U_empca.T
        Z_emp = coeff_emp @ U_empca
        X_emp = data["unwhiten"](Z_emp.reshape(N_te, C, T))

        B_nfpa = _kron_basis(U_c, U_t)
        B_emp = U_empca.T
        angles = _principal_angles_deg(B_nfpa, B_emp)

        rows.append(
            {
                "regime": label,
                "distortion_level": distortion,
                "nfpa_weighted_residual": float(np.mean((Z_te - Z_nfpa) ** 2)),
                "empca_weighted_residual": float(np.mean((Z_te_flat - Z_emp) ** 2)),
                "nfpa_reconstruction_mse": float(np.mean((X_nfpa - X_te_clean) ** 2)),
                "empca_reconstruction_mse": float(np.mean((X_emp - X_te_clean) ** 2)),
                "mean_principal_angle_deg": float(np.mean(angles)),
                "max_principal_angle_deg": float(np.max(angles)),
                "nfpa_final_train_chi2": float(hist[-1]),
            }
        )

    return {"nfpa_df": pd.DataFrame(rows)}


def experiment_d_status_table() -> pd.DataFrame:
    rows = [
        ("Linear AE", True, True, "Exact weighted linear optimum available via implementation.notebook_support exact_weighted_subspace."),
        ("CNN AE", True, False, "Backbone exists in src/CNN/resnet_1d.py, but torch is missing and there is no autoencoder reconstruction head/trainer."),
        ("Transformer AE", True, False, "Transformer backbones exist, but depend on torch + reconstruction.training.muon and have no reconstruction experiment harness."),
    ]
    return pd.DataFrame(rows, columns=["model", "code_present", "runnable_now", "notes"])


def experiment_e_status_table() -> pd.DataFrame:
    rows = [
        ("raw_transformer_mse", True, False, False, "missing trainer/backbone deps"),
        ("raw_transformer_mahalanobis", True, False, False, "missing trainer/backbone deps"),
        ("prewhitened_transformer_mse", True, False, True, "whitening is conceptually available but no runnable transformer training stack"),
        ("prewhitened_transformer_mahalanobis", True, False, True, "needs runnable transformer stack plus Mahalanobis training loop"),
    ]
    return pd.DataFrame(
        rows,
        columns=["configuration", "backbone_code_present", "runnable_now", "whitening_available", "notes"],
    )


def plot_readiness(df: pd.DataFrame, title: str):
    status_order = ["ready", "partial", "blocked"]
    counts = (
        df.assign(status=pd.Categorical(df["status"], categories=status_order, ordered=True))
        .groupby("status")
        .size()
        .reindex(status_order, fill_value=0)
    )
    colors = {"ready": "#2b8a3e", "partial": "#c77d00", "blocked": "#c92a2a"}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(counts.index, counts.values, color=[colors[s] for s in counts.index])
    ax.set_ylabel("count")
    ax.set_title(title)
    return fig


def plot_experiment_e_coverage(df: pd.DataFrame):
    numeric = df[["backbone_code_present", "runnable_now", "whitening_available"]].astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(numeric.to_numpy(), cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["configuration"])
    ax.set_xticks(np.arange(numeric.shape[1]))
    ax.set_xticklabels(numeric.columns, rotation=20, ha="right")
    ax.set_title("Experiment E readiness matrix")
    for i in range(numeric.shape[0]):
        for j in range(numeric.shape[1]):
            ax.text(j, i, int(numeric.iat[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def parse_transformer_imports() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((REPO_ROOT / "src" / "transformer").glob("*.py")):
        tree = ast.parse(_read_text(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                imports.append(base)
        rows.append(
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "imports_torch": any(name.startswith("torch") for name in imports),
                "imports_muon": any(name == "reconstruction.training.muon" for name in imports),
                "imports_numpy": any(name == "numpy" for name in imports),
            }
        )
    return pd.DataFrame(rows)
