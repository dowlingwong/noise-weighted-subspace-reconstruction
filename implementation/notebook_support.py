"""Shared notebook utilities for the implementation experiment suite.

The implementation notebooks are organized around two concrete data sources:

- real K-alpha traces in ``data/k_alpha/k_alpha_traces.h5`` with companion RQs;
- controlled synthetic families generated with ``QP_simulator/QPSimulator.py``.

The goal of this module is not to hide the experiment logic. It provides a
small, explicit analysis contract so the notebooks can stay short, readable,
and consistent with ``plan/experiment_checklist.md``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d


def _discover_repo_root() -> Path:
    if "__file__" in globals():
        start = Path(__file__).resolve()
        candidates = [start.parent, *start.parents]
    else:  # pragma: no cover - notebook execution fallback
        start = Path.cwd().resolve()
        candidates = [start, *start.parents]

    for candidate in candidates:
        if (
            (candidate / "plan" / "experiment_checklist.md").exists()
            and (candidate / "src").exists()
            and (candidate / "QP_simulator").exists()
            and (candidate / "implementation").exists()
        ):
            return candidate
    raise RuntimeError(f"Could not locate repo root from start path: {start}")


REPO_ROOT = _discover_repo_root()
SRC_DIR = REPO_ROOT / "src"
EMPCA_DIR = SRC_DIR / "EMPCA"
QP_DIR = REPO_ROOT / "QP_simulator"
for path in (REPO_ROOT, SRC_DIR, EMPCA_DIR, QP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from OptimumFilter import OptimumFilter  # noqa: E402
from empca_TCY_optimized import empca_solver, orthonormalize, smooth, w_orthonormalize  # noqa: E402
from empca_equivalence_utils import (  # noqa: E402
    build_of_one_sided_weights,
    phase_align_basis,
    project_gls,
    weighted_cosine,
    weighted_inner,
    weighted_residual_energy,
)
from QPSimulator import QPSimulator  # noqa: E402
from noise_module.NoiseGenerator import NoiseGenerator  # noqa: E402
from noise_module.artifact_injector import ArtifactInjector  # noqa: E402
from noise_module.multichannel_noise import MultiChannelNoiseGenerator  # noqa: E402
from noise_module.temporal_noise import TemporalNoiseWrapper  # noqa: E402


@dataclass(slots=True)
class CanonicalConfig:
    seed: int = 314159
    sampling_frequency: float = 2.5e5
    trace_len: int = 32768
    pretrigger: int = 4000
    split_train: float = 0.60
    split_val: float = 0.20
    split_test: float = 0.20
    empirical_rank_max: int = 8
    default_empca_iter: int = 60
    default_empca_patience: int = 12
    default_psd_floor_quantile: float = 0.01
    real_train_cap: int = 512
    real_eval_cap: int = 384
    sim_events_small: int = 128
    sim_events_medium: int = 160
    sim_events_large: int = 192
    crb_replicates: int = 1200
    results_tables_dir: str = "results/tables"
    results_figures_dir: str = "results/figures"
    results_manifests_dir: str = "results/manifests"
    results_notebooks_dir: str = "results/notebooks"
    real_trace_path: str = "data/k_alpha/k_alpha_traces.h5"
    real_rq_path: str = "data/k_alpha/k_alpha_rqs.h5"
    template_path: str = "data/template_K_alpha_tight.npy"
    canonical_psd_path: str = "data/weight/noise_psd_pink.npy"

    def validate(self) -> "CanonicalConfig":
        total = self.split_train + self.split_val + self.split_test
        if not np.isclose(total, 1.0):
            raise ValueError(f"split fractions must sum to 1, got {total}")
        if self.pretrigger <= 0 or self.pretrigger >= self.trace_len:
            raise ValueError("pretrigger must lie inside the trace length")
        return self


@dataclass(slots=True)
class DatasetBundle:
    traces_raw: np.ndarray
    traces_baseline: np.ndarray
    traces_freq: np.ndarray
    split_indices: dict[str, np.ndarray]
    template_time: np.ndarray
    template_freq: np.ndarray
    psd_one_sided: np.ndarray
    weights_one_sided: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    rqs: pd.DataFrame | None = None


def ensure_results_dirs(cfg: CanonicalConfig) -> dict[str, Path]:
    dirs = {
        "tables": REPO_ROOT / cfg.results_tables_dir,
        "figures": REPO_ROOT / cfg.results_figures_dir,
        "manifests": REPO_ROOT / cfg.results_manifests_dir,
        "notebooks": REPO_ROOT / cfg.results_notebooks_dir,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_figure(fig, path: Path, dpi: int = 160) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def config_as_frame(cfg: CanonicalConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [{"setting": key, "value": value} for key, value in asdict(cfg).items()]
    )


def manifest_row(
    block_id: str,
    regime: str,
    output_path: str,
    cfg: CanonicalConfig,
    bundle: DatasetBundle | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "block_id": block_id,
        "regime": regime,
        "output_path": output_path,
        "seed": int(cfg.seed),
        "trace_len": int(cfg.trace_len),
        "pretrigger": int(cfg.pretrigger),
    }
    if bundle is not None:
        row["data_source"] = bundle.metadata.get("source")
        row["n_events"] = int(bundle.metadata.get("n_events", bundle.traces_raw.shape[0]))
        row["template_path"] = bundle.metadata.get("template_path")
        row["psd_path"] = bundle.metadata.get("psd_path")
    if extra:
        row.update(extra)
    return row


def baseline_correct(
    traces: np.ndarray,
    pretrigger: int,
    return_baseline: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces, dtype=np.float64)
    baseline = np.mean(traces[..., :pretrigger], axis=-1, keepdims=True)
    corrected = traces - baseline
    if return_baseline:
        return corrected, np.squeeze(baseline, axis=-1)
    return corrected


def normalize_template_peak(template: np.ndarray, pretrigger: int) -> np.ndarray:
    template = baseline_correct(np.asarray(template)[None, :], pretrigger=pretrigger)[0]
    peak = float(np.max(np.abs(template)))
    if peak <= 0:
        return template
    return template / peak


def deterministic_split_indices(n_samples: int, cfg: CanonicalConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n_samples)
    n_train = int(round(cfg.split_train * n_samples))
    n_val = int(round(cfg.split_val * n_samples))
    n_train = min(n_train, n_samples)
    n_val = min(n_val, n_samples - n_train)
    train = np.sort(order[:n_train])
    val = np.sort(order[n_train : n_train + n_val])
    test = np.sort(order[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def subsample_indices(indices: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if cap is None or cap <= 0 or len(indices) <= cap:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=cap, replace=False)
    return np.sort(chosen)


def rfft_traces(traces: np.ndarray) -> np.ndarray:
    return np.fft.rfft(np.asarray(traces, dtype=np.float64), axis=-1)


def safe_sqrt_weights(weights: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), a_min=0.0, a_max=None))


def unwhiten_basis(U_w: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sqrt_w = safe_sqrt_weights(weights)
    out = np.zeros_like(U_w, dtype=np.complex128)
    mask = sqrt_w > 0
    out[:, mask] = U_w[:, mask] / sqrt_w[mask]
    return out


def normalize_basis_weighted_unit(U: np.ndarray, weights: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.complex128)
    out = np.array(U, copy=True)
    squeeze = False
    if out.ndim == 1:
        out = out[None, :]
        squeeze = True
    for idx in range(out.shape[0]):
        norm = np.sqrt(np.real(weighted_inner(out[idx], out[idx], weights)))
        if norm > 0:
            out[idx] /= norm
    return out[0] if squeeze else out


def principal_angles_weighted(
    U_a: np.ndarray,
    U_b: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    U_a = np.atleast_2d(np.asarray(U_a, dtype=np.complex128))
    U_b = np.atleast_2d(np.asarray(U_b, dtype=np.complex128))
    sqrt_w = safe_sqrt_weights(weights)
    A = (U_a * sqrt_w[None, :]).T
    B = (U_b * sqrt_w[None, :]).T
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    singular_values = np.linalg.svd(Qa.conj().T @ Qb, compute_uv=False)
    singular_values = np.clip(np.real(singular_values), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(singular_values))
    return singular_values, angles_deg


def exact_weighted_subspace(X_f: np.ndarray, weights: np.ndarray, k: int) -> dict[str, np.ndarray]:
    X_f = np.asarray(X_f, dtype=np.complex128)
    sqrt_w = safe_sqrt_weights(weights)
    X_w = X_f * sqrt_w[None, :]
    _, _, vh = np.linalg.svd(X_w, full_matrices=False)
    basis_whitened = vh[:k].astype(np.complex128)
    basis_native = normalize_basis_weighted_unit(unwhiten_basis(basis_whitened, weights), weights)
    return {"basis_whitened": basis_whitened, "basis_native": basis_native}


def exact_isotropic_subspace(X_f: np.ndarray, k: int) -> np.ndarray:
    _, _, vh = np.linalg.svd(np.asarray(X_f, dtype=np.complex128), full_matrices=False)
    return vh[:k].astype(np.complex128)


def fit_weighted_empca(
    X_f: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_iter: int,
    patience: int,
    init: str = "random",
    template_f: np.ndarray | None = None,
    smoothing: bool = False,
    seed: int | None = None,
    mode: str = "fast",
) -> dict[str, Any]:
    """Fit weighted EMPCA.

    ``mode='fast'`` uses the decoupled M-step (weights cancel per component);
    it is exact only at k=1. ``mode='full'`` solves the coupled normal
    equations including the off-diagonal C^dagger C terms and is required for
    correct rank-k (k>=2) subspaces — the fast-mode decoupling is the suspect
    behind the draft's Fig 15/16 contradictions (see EXPERIMENT_PLAN.md, G2).
    """
    X_f = np.asarray(X_f, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)
    if seed is not None:
        np.random.seed(seed)

    solver = empca_solver(k, X_f, weights)
    init_mode = init.lower()
    if init_mode == "svd":
        solver.eigvec = exact_weighted_subspace(X_f, weights, k)["basis_native"].copy()
        solver.coeff = solver.solve_coeff()
    elif init_mode == "template" and template_f is not None:
        base = normalize_basis_weighted_unit(np.asarray(template_f, dtype=np.complex128), weights)[None, :]
        if k > 1:
            extra = exact_weighted_subspace(X_f, weights, k)["basis_native"][1:]
            base = np.vstack([base, extra])
        solver.eigvec = orthonormalize(base.copy())
        solver.coeff = solver.solve_coeff()
    elif init_mode != "random":
        raise ValueError(f"Unsupported init mode: {init}")

    chi2_trace: list[float] = []
    best = np.inf
    stale = 0

    for _ in range(n_iter):
        eigvec = solver.solve_eigvec(mode=mode)
        if smoothing:
            eigvec = smooth(eigvec, window=15, polyord=3, deriv=0)
        solver.eigvec = orthonormalize(eigvec)
        solver.coeff = solver.solve_coeff()
        chi2 = float(solver.chi2())
        chi2_trace.append(chi2)
        if chi2 + 1e-12 < best:
            best = chi2
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return {
        "basis": solver.eigvec.copy(),
        "coeff": solver.coeff.copy(),
        "chi2_trace": np.asarray(chi2_trace, dtype=np.float64),
        "n_iter_used": int(len(chi2_trace)),
        "init": init_mode,
        "smoothing": bool(smoothing),
    }


def rankk_gls_coefficients(
    X_f: np.ndarray,
    basis_f: np.ndarray,
    weights: np.ndarray,
    return_complex: bool = False,
) -> np.ndarray:
    coeff = project_gls(np.asarray(X_f), np.asarray(basis_f), np.asarray(weights), return_complex=return_complex)
    return np.asarray(coeff)


def residual_energy_per_trace(
    X_f: np.ndarray,
    basis_f: np.ndarray,
    coeff: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return weighted_residual_energy(X_f, basis_f, coeff, weights)


def compute_residual_summary(
    X_f: np.ndarray,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    coeff_a = rankk_gls_coefficients(X_f, basis_a, weights, return_complex=True)
    coeff_b = rankk_gls_coefficients(X_f, basis_b, weights, return_complex=True)
    resid_a = residual_energy_per_trace(X_f, basis_a, coeff_a, weights)
    resid_b = residual_energy_per_trace(X_f, basis_b, coeff_b, weights)
    ks = stats.ks_2samp(resid_a, resid_b)
    return {
        "residual_mean_a": float(np.mean(resid_a)),
        "residual_mean_b": float(np.mean(resid_b)),
        "relative_mean_diff": float((np.mean(resid_a) - np.mean(resid_b)) / max(np.mean(resid_b), 1e-12)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def compute_of_amplitudes(
    X_time: np.ndarray,
    template_time: np.ndarray,
    psd_one_sided: np.ndarray,
    fs: float,
) -> np.ndarray:
    of = OptimumFilter(template_time, psd_one_sided, fs)
    return np.asarray([of.fit(trace)[0] for trace in np.asarray(X_time, dtype=np.float64)], dtype=np.float64)


def compute_of_statistics(template_f: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    fisher = float(np.real(weighted_inner(template_f, template_f, weights)))
    variance = float(1.0 / fisher) if fisher > 0 else np.inf
    return {
        "fisher_information": fisher,
        "variance_crb": variance,
        "sigma_amplitude": float(np.sqrt(variance)) if np.isfinite(variance) else np.inf,
    }


def estimate_real_psd_from_pretrigger(traces: np.ndarray, cfg: CanonicalConfig) -> tuple[np.ndarray, np.ndarray]:
    pre = np.asarray(traces[:, : cfg.pretrigger], dtype=np.float64)
    pre = pre - np.mean(pre, axis=1, keepdims=True)
    repeats = int(np.ceil(cfg.trace_len / cfg.pretrigger))
    tiled = np.tile(pre, (1, repeats))[:, : cfg.trace_len]
    freqs, psd = QPSimulator.estimate_psd(tiled, cfg.sampling_frequency)
    floor = np.quantile(psd[1:], cfg.default_psd_floor_quantile)
    psd = np.maximum(psd, floor)
    psd[0] = psd[1]
    return freqs, psd


def load_real_bundle(cfg: CanonicalConfig) -> DatasetBundle:
    trace_path = REPO_ROOT / cfg.real_trace_path
    rq_path = REPO_ROOT / cfg.real_rq_path
    template_path = REPO_ROOT / cfg.template_path
    psd_path = REPO_ROOT / cfg.canonical_psd_path

    with h5py.File(trace_path, "r") as handle:
        traces = np.asarray(handle["traces"][:], dtype=np.float64)
    with h5py.File(rq_path, "r") as handle:
        rqs = pd.DataFrame.from_records(handle["rqs"][:])
    template_time = np.load(template_path).astype(np.float64).reshape(-1)
    psd_one_sided = np.load(psd_path).astype(np.float64).reshape(-1)
    empirical_freqs, empirical_psd = estimate_real_psd_from_pretrigger(traces, cfg)

    traces_baseline, baselines = baseline_correct(traces, cfg.pretrigger, return_baseline=True)
    template_time = normalize_template_peak(template_time, cfg.pretrigger)
    traces_freq = rfft_traces(traces_baseline)
    template_freq = np.fft.rfft(template_time)
    weights = build_of_one_sided_weights(psd_one_sided, cfg.trace_len)
    split_indices = deterministic_split_indices(len(traces), cfg)

    metadata = {
        "source": "CAL-kalpha",
        "n_events": int(len(traces)),
        "trace_len": int(cfg.trace_len),
        "template_path": cfg.template_path,
        "psd_path": cfg.canonical_psd_path,
        "trace_path": cfg.real_trace_path,
        "rq_path": cfg.real_rq_path,
        "baseline_mean_mean": float(np.mean(baselines)),
        "baseline_mean_std": float(np.std(baselines)),
        "empirical_psd_mean": float(np.mean(empirical_psd)),
        "empirical_psd_std": float(np.std(empirical_psd)),
        "empirical_psd_freqs": empirical_freqs,
        "empirical_psd": empirical_psd,
    }
    return DatasetBundle(
        traces_raw=traces,
        traces_baseline=traces_baseline,
        traces_freq=traces_freq,
        split_indices=split_indices,
        template_time=template_time,
        template_freq=template_freq,
        psd_one_sided=psd_one_sided,
        weights_one_sided=weights,
        metadata=metadata,
        rqs=rqs,
    )


def prepare_bundle(
    traces: np.ndarray,
    template_time: np.ndarray,
    psd_one_sided: np.ndarray,
    cfg: CanonicalConfig,
    metadata: dict[str, Any],
) -> DatasetBundle:
    traces = np.asarray(traces, dtype=np.float64)
    traces_baseline, baselines = baseline_correct(traces, cfg.pretrigger, return_baseline=True)
    template_time = normalize_template_peak(template_time, cfg.pretrigger)
    template_freq = np.fft.rfft(template_time)
    traces_freq = rfft_traces(traces_baseline)
    split_indices = deterministic_split_indices(len(traces), cfg)
    weights = build_of_one_sided_weights(psd_one_sided, cfg.trace_len)
    out_meta = {
        "n_events": int(len(traces)),
        "trace_len": int(cfg.trace_len),
        "baseline_mean_mean": float(np.mean(baselines)),
        "baseline_mean_std": float(np.std(baselines)),
    }
    out_meta.update(metadata)
    return DatasetBundle(
        traces_raw=traces,
        traces_baseline=traces_baseline,
        traces_freq=traces_freq,
        split_indices=split_indices,
        template_time=template_time,
        template_freq=template_freq,
        psd_one_sided=np.asarray(psd_one_sided, dtype=np.float64),
        weights_one_sided=np.asarray(weights, dtype=np.float64),
        metadata=out_meta,
    )


def make_qp_simulator(
    cfg: CanonicalConfig,
    tau_decay: float = 3e6,
    trigger_time: float | None = None,
) -> QPSimulator:
    return QPSimulator(
        sampling_frequency=cfg.sampling_frequency,
        trace_samples=cfg.trace_len,
        tau_decay=tau_decay,
        trigger_time=trigger_time,
    )


def stationary_noise_generator(
    cfg: CanonicalConfig,
    noise_type: str,
    noise_power: float,
    rng: np.random.Generator | None = None,
) -> NoiseGenerator:
    return NoiseGenerator(
        {
            "noise_type": noise_type,
            "noise_power": float(noise_power),
            "sampling_frequency": cfg.sampling_frequency,
        },
        rng=rng,
    )


def make_clean_qp_trace(
    cfg: CanonicalConfig,
    n_qp: int,
    tau_decay: float = 3e6,
    t0_shift_ns: float = 0.0,
    arrival_mode: str = "aligned",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng() if rng is None else rng
    base_sim = make_qp_simulator(cfg, tau_decay=tau_decay)
    event_sim = make_qp_simulator(
        cfg,
        tau_decay=tau_decay,
        trigger_time=base_sim.trigger_time + float(t0_shift_ns),
    )
    mode = arrival_mode.lower()
    if mode == "aligned":
        arrivals = np.zeros(int(n_qp), dtype=float)
    elif mode == "stochastic":
        arrivals = rng.exponential(scale=2e6, size=int(n_qp)) + event_sim.trigger_time
    else:
        raise ValueError(f"Unsupported arrival_mode: {arrival_mode}")
    trace, amp_true = event_sim.generate(arrivals, return_amplitude=True)
    return np.asarray(trace, dtype=np.float64), float(amp_true)


def simulate_rank1_batch(
    cfg: CanonicalConfig,
    n_events: int,
    amp_range: tuple[float, float] = (0.8, 1.2),
    noise_type: str = "pink",
    noise_power: float = 1.0,
    n_qp_ref: int = 5000,
) -> tuple[DatasetBundle, dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    clean_ref, amp_ref = make_clean_qp_trace(cfg, n_qp=n_qp_ref, arrival_mode="aligned", rng=rng)
    scales = rng.uniform(amp_range[0], amp_range[1], size=n_events)
    clean = scales[:, None] * clean_ref[None, :]
    amp_true = scales * amp_ref
    ng = stationary_noise_generator(cfg, noise_type=noise_type, noise_power=noise_power, rng=rng)
    _, psd = ng.build_psd(cfg.trace_len)
    noise = np.stack([ng.generate_noise(cfg.trace_len) for _ in range(n_events)], axis=0)
    bundle = prepare_bundle(
        clean + noise,
        template_time=clean_ref,
        psd_one_sided=psd,
        cfg=cfg,
        metadata={
            "source": "SIM-rank1",
            "noise_type": noise_type,
            "noise_power": float(noise_power),
            "arrival_mode": "aligned",
            "template_path": "QPSimulator rank-1 reference",
            "psd_path": f"analytic:{noise_type}",
        },
    )
    truth = {
        "clean_traces": clean,
        "noise_traces": noise,
        "amplitude_true": amp_true,
        "scales": scales,
        "template_time": clean_ref,
    }
    return bundle, truth


def simulate_controlled_family(
    cfg: CanonicalConfig,
    n_events: int,
    tau_decay_range: tuple[float, float] = (3e6, 3e6),
    t0_jitter_range: tuple[float, float] = (0.0, 0.0),
    n_qp_range: tuple[int, int] = (5000, 5000),
    noise_type: str = "pink",
    noise_power: float = 1.0,
    arrival_mode: str = "aligned",
) -> tuple[DatasetBundle, dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    tau_vals = rng.uniform(tau_decay_range[0], tau_decay_range[1], size=n_events)
    if tau_decay_range[0] == tau_decay_range[1]:
        tau_vals[:] = tau_decay_range[0]
    t0_vals = rng.uniform(t0_jitter_range[0], t0_jitter_range[1], size=n_events)
    if t0_jitter_range[0] == t0_jitter_range[1]:
        t0_vals[:] = t0_jitter_range[0]
    n_qp_vals = rng.integers(n_qp_range[0], n_qp_range[1] + 1, size=n_events)
    if n_qp_range[0] == n_qp_range[1]:
        n_qp_vals[:] = n_qp_range[0]

    clean = np.zeros((n_events, cfg.trace_len), dtype=np.float64)
    amp_true = np.zeros(n_events, dtype=np.float64)
    for idx in range(n_events):
        clean[idx], amp_true[idx] = make_clean_qp_trace(
            cfg,
            n_qp=int(n_qp_vals[idx]),
            tau_decay=float(tau_vals[idx]),
            t0_shift_ns=float(t0_vals[idx]),
            arrival_mode=arrival_mode,
            rng=rng,
        )

    nominal_template, _ = make_clean_qp_trace(
        cfg,
        n_qp=int(np.median(n_qp_vals)),
        tau_decay=float(np.median(tau_vals)),
        t0_shift_ns=0.0,
        arrival_mode="aligned",
        rng=rng,
    )
    ng = stationary_noise_generator(cfg, noise_type=noise_type, noise_power=noise_power, rng=rng)
    _, psd = ng.build_psd(cfg.trace_len)
    noise = np.stack([ng.generate_noise(cfg.trace_len) for _ in range(n_events)], axis=0)

    bundle = prepare_bundle(
        clean + noise,
        template_time=nominal_template,
        psd_one_sided=psd,
        cfg=cfg,
        metadata={
            "source": "SIM-controlled-family",
            "noise_type": noise_type,
            "noise_power": float(noise_power),
            "arrival_mode": arrival_mode,
            "template_path": "QPSimulator controlled family nominal template",
            "psd_path": f"analytic:{noise_type}",
        },
    )
    truth = {
        "clean_traces": clean,
        "noise_traces": noise,
        "amplitude_true": amp_true,
        "tau_decay": tau_vals,
        "t0_shift": t0_vals,
        "n_qp": n_qp_vals,
    }
    return bundle, truth


def _fit_amplitude_scale(reference: np.ndarray, candidate: np.ndarray) -> float:
    num = float(np.dot(reference, candidate))
    den = float(np.dot(candidate, candidate))
    return 1.0 if den <= 0 else num / den


def _linear_map_from_coeff(coeff: np.ndarray, target: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=np.float64)
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    sol, *_ = np.linalg.lstsq(coeff, np.asarray(target, dtype=np.float64), rcond=None)
    return sol


def _predict_from_linear_map(coeff: np.ndarray, weights: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=np.float64)
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    return coeff @ np.asarray(weights, dtype=np.float64)


def _median_relative_error(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.median(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _time_shift_grid(cfg: CanonicalConfig, n_points: int = 31, frac: float = 0.10) -> np.ndarray:
    span = cfg.trace_len / cfg.sampling_frequency * 1e9 * frac
    return np.linspace(-span, span, n_points)


def dataframe_from_claim_map() -> pd.DataFrame:
    rows = [
        ("E1", "rank-1 EMPCA ≡ OF amplitude estimator", "04_empca / theorem 1", "theorem-support", "weighted cosine; amplitude correlation; median relative amplitude error; residual KS p-value", "results/tables/block_04_e1_rank1_summary.csv"),
        ("E2", "noise-aware linear AE ≡ EMPCA", "05_linear_ae / theorem 2", "theorem-support", "principal-angle cosines for k=1..3; residual mean gap; residual KS p-value", "results/tables/block_05_e2_bridge.csv"),
        ("E3", "chi^2(k) decreases monotonically with rank", "04_empca / rank-k proposition", "theorem-support", "held-out weighted residual by k for 1D and multi-D families", "results/tables/block_06_e3_monotonicity.csv"),
        ("E4", "CRB variance law", "03_optimal_filter / CRB", "theorem-support", "empirical amplitude variance vs 1 / N_Phi across noise colors", "results/tables/block_04_e4_crb.csv"),
        ("E5", "energy-resolution scaling law", "03_optimal_filter / resolution", "mixed", "sigma_E vs noise power slope; simulated residuals; real-data sigma comparison", "results/tables/block_04_e5_resolution.csv"),
        ("E6", "noise-aware EMPCA beats isotropic PCA under colored noise", "07_experiments / ablation", "theorem-support", "weighted residual gain under white, pink, brownian noise", "results/tables/block_07_e6_ablation.csv"),
        ("E7", "template mismatch bias and rank-2 recovery", "03_optimal_filter / template mismatch", "theorem-support", "mean amplitude bias; RMSE; bias-vs-cos^2 curve", "results/tables/block_07_e7_mismatch.csv"),
        ("E8", "time-shift OF recovers jittered arrivals", "03_optimal_filter / shifted OF", "theorem-support", "arrival-time RMSE; amplitude RMSE; fixed-vs-shift bias", "results/tables/block_07_e8_time_shift.csv"),
        ("E9", "EMPCA convergence diagnostics", "06_convergence / convergence theorem", "theorem-support", "chi^2 trace monotonicity; iterations to tolerance; init dependence", "results/tables/block_06_e9_convergence.csv"),
        ("E10", "non-stationary noise robustness", "06_convergence / stationarity discussion", "robustness-support", "global-vs-segment PSD RMSE and weighted residual", "results/tables/block_09_e10_nonstationary.csv"),
        ("E11", "artifact robustness and refit after flagging", "06_convergence / limitations", "robustness-support", "amplitude RMSE and weighted residual before/after flagging", "results/tables/block_09_e11_artifacts.csv"),
        ("E12", "real K-alpha verification", "07_experiments / real data verification", "real-support", "rank-1 cosine; AE bridge cosines; Gaussian amplitude histogram; sigma_A comparison", "results/tables/block_03_e12_real_summary.csv"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "experiment_id",
            "claim",
            "paper_anchor",
            "support_regime",
            "metrics",
            "artifact_path",
        ],
    )


def build_split_manifest(bundle: DatasetBundle, cfg: CanonicalConfig) -> dict[str, Any]:
    return {
        "seed": int(cfg.seed),
        "trace_len": int(cfg.trace_len),
        "pretrigger": int(cfg.pretrigger),
        "split_sizes": {k: int(v.shape[0]) for k, v in bundle.split_indices.items()},
        "source": bundle.metadata.get("source"),
        "template_path": bundle.metadata.get("template_path"),
        "psd_path": bundle.metadata.get("psd_path"),
    }


def choose_noise_power_for_target_fisher(
    template_time: np.ndarray,
    noise_type: str,
    cfg: CanonicalConfig,
    fisher_target: float = 30.0,
) -> tuple[float, float]:
    candidate_powers = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    template_f = np.fft.rfft(template_time)
    best_power = float(candidate_powers[0])
    best_fisher = np.inf
    best_dist = np.inf
    for power in candidate_powers:
        ng = stationary_noise_generator(cfg, noise_type=noise_type, noise_power=float(power))
        _, psd = ng.build_psd(cfg.trace_len)
        weights = build_of_one_sided_weights(psd, cfg.trace_len)
        fisher = float(np.real(weighted_inner(template_f, template_f, weights)))
        dist = abs(np.log(max(fisher, 1e-12)) - np.log(fisher_target))
        if dist < best_dist:
            best_dist = dist
            best_power = float(power)
            best_fisher = float(fisher)
    return best_power, best_fisher


def run_block02_audit(cfg: CanonicalConfig) -> dict[str, Any]:
    bundle = load_real_bundle(cfg)
    baseline_before = np.mean(bundle.traces_raw[:, : cfg.pretrigger], axis=1)
    baseline_after = np.mean(bundle.traces_baseline[:, : cfg.pretrigger], axis=1)
    audit_df = pd.DataFrame(
        [
            ("n_events", float(bundle.traces_raw.shape[0])),
            ("trace_len", float(bundle.traces_raw.shape[1])),
            ("baseline_before_mean", float(np.mean(baseline_before))),
            ("baseline_before_std", float(np.std(baseline_before))),
            ("baseline_after_mean", float(np.mean(baseline_after))),
            ("baseline_after_std", float(np.std(baseline_after))),
            ("canonical_psd_mean", float(np.mean(bundle.psd_one_sided))),
            ("empirical_pretrigger_psd_mean", float(np.mean(bundle.metadata["empirical_psd"]))),
            ("weight_min", float(np.min(bundle.weights_one_sided[1:]))),
            ("weight_max", float(np.max(bundle.weights_one_sided[1:]))),
        ],
        columns=["metric", "value"],
    )
    reports_df = pd.DataFrame(
        [
            {
                "trace_path": bundle.metadata["trace_path"],
                "rq_path": bundle.metadata["rq_path"],
                "template_path": bundle.metadata["template_path"],
                "psd_path": bundle.metadata["psd_path"],
            }
        ]
    )
    return {
        "bundle": bundle,
        "audit_df": audit_df,
        "reports_df": reports_df,
        "split_manifest": build_split_manifest(bundle, cfg),
    }


def run_block03_real_rank1(cfg: CanonicalConfig) -> dict[str, Any]:
    bundle = load_real_bundle(cfg)
    train_idx = subsample_indices(bundle.split_indices["train"], cfg.real_train_cap, cfg.seed)
    test_idx = subsample_indices(bundle.split_indices["test"], cfg.real_eval_cap, cfg.seed + 1)

    fit = fit_weighted_empca(
        bundle.traces_freq[train_idx],
        bundle.weights_one_sided,
        k=1,
        n_iter=cfg.default_empca_iter,
        patience=cfg.default_empca_patience,
        init="template",
        template_f=bundle.template_freq,
        seed=cfg.seed,
    )
    basis_rank1 = phase_align_basis(fit["basis"][0], bundle.template_freq, bundle.weights_one_sided)
    cosine = float(weighted_cosine(basis_rank1, bundle.template_freq, bundle.weights_one_sided))

    of_train = rankk_gls_coefficients(bundle.traces_freq[train_idx], bundle.template_freq, bundle.weights_one_sided, return_complex=False).reshape(-1)
    emp_train = rankk_gls_coefficients(bundle.traces_freq[train_idx], basis_rank1, bundle.weights_one_sided, return_complex=False).reshape(-1)
    scale = _fit_amplitude_scale(of_train, emp_train)

    of_test = rankk_gls_coefficients(bundle.traces_freq[test_idx], bundle.template_freq, bundle.weights_one_sided, return_complex=False).reshape(-1)
    emp_test = rankk_gls_coefficients(bundle.traces_freq[test_idx], basis_rank1, bundle.weights_one_sided, return_complex=False).reshape(-1) * scale
    of_td = compute_of_amplitudes(bundle.traces_baseline[test_idx], bundle.template_time, bundle.psd_one_sided, cfg.sampling_frequency)

    resid_of = residual_energy_per_trace(bundle.traces_freq[test_idx], bundle.template_freq, of_test, bundle.weights_one_sided)
    resid_emp = residual_energy_per_trace(bundle.traces_freq[test_idx], basis_rank1, emp_test / scale, bundle.weights_one_sided)
    ks = stats.ks_2samp(resid_of, resid_emp)

    amplitude_df = pd.DataFrame(
        {
            "of_gls": of_test,
            "of_time_domain": of_td,
            "empca_rank1": emp_test,
            "rq_of_ampl_0": bundle.rqs.iloc[test_idx]["OF_ampl_0"].to_numpy(),
            "resid_of": resid_of,
            "resid_empca_rank1": resid_emp,
        }
    )

    bridge_rows = []
    for k in (1, 2, 3):
        exact = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=k)
        emp = fit_weighted_empca(
            bundle.traces_freq[train_idx],
            bundle.weights_one_sided,
            k=k,
            n_iter=cfg.default_empca_iter,
            patience=cfg.default_empca_patience,
            init="svd",
            seed=cfg.seed,
        )
        cosines, angles = principal_angles_weighted(emp["basis"], exact["basis_native"], bundle.weights_one_sided)
        residual_summary = compute_residual_summary(
            bundle.traces_freq[test_idx],
            emp["basis"],
            exact["basis_native"],
            bundle.weights_one_sided,
        )
        row = {
            "k": k,
            "principal_angle_cosines": [float(x) for x in cosines],
            "principal_angles_deg": [float(x) for x in angles],
            **residual_summary,
        }
        bridge_rows.append(row)
    bridge_df = pd.DataFrame(bridge_rows)

    of_stats = compute_of_statistics(bundle.template_freq, bundle.weights_one_sided)
    sigma_obs = float(np.std(of_test, ddof=1))
    sigma_pred = float(of_stats["sigma_amplitude"])
    shapiro_subset = of_test if len(of_test) <= 500 else of_test[:500]
    shapiro = stats.shapiro(shapiro_subset)

    summary = {
        "weighted_cosine_rank1": cosine,
        "amplitude_correlation": _pearsonr(of_test, emp_test),
        "median_relative_error": _median_relative_error(emp_test, of_test),
        "rq_alignment_median_abs_diff": float(np.median(np.abs(of_test - amplitude_df["rq_of_ampl_0"]))),
        "residual_ks_statistic": float(ks.statistic),
        "residual_ks_pvalue": float(ks.pvalue),
        "sigma_obs": sigma_obs,
        "sigma_pred": sigma_pred,
        "sigma_relative_error": float(abs(sigma_pred - sigma_obs) / max(sigma_obs, 1e-12)),
        "shapiro_pvalue": float(shapiro.pvalue),
        "empca_iterations": int(fit["n_iter_used"]),
    }
    return {
        "bundle": bundle,
        "summary": summary,
        "amplitude_df": amplitude_df,
        "bridge_df": bridge_df,
        "of_stats": of_stats,
    }


def run_block04_theorem_suite(cfg: CanonicalConfig) -> dict[str, Any]:
    rank1_bundle, rank1_truth = simulate_rank1_batch(
        cfg,
        n_events=cfg.sim_events_large,
        amp_range=(0.8, 1.2),
        noise_type="pink",
        noise_power=1.0,
    )
    train_idx = rank1_bundle.split_indices["train"]
    test_idx = rank1_bundle.split_indices["test"]
    fit = fit_weighted_empca(
        rank1_bundle.traces_freq[train_idx],
        rank1_bundle.weights_one_sided,
        k=1,
        n_iter=cfg.default_empca_iter,
        patience=cfg.default_empca_patience,
        init="template",
        template_f=rank1_bundle.template_freq,
        seed=cfg.seed,
    )
    basis_rank1 = phase_align_basis(fit["basis"][0], rank1_bundle.template_freq, rank1_bundle.weights_one_sided)
    of_train = rankk_gls_coefficients(rank1_bundle.traces_freq[train_idx], rank1_bundle.template_freq, rank1_bundle.weights_one_sided, return_complex=False).reshape(-1)
    emp_train = rankk_gls_coefficients(rank1_bundle.traces_freq[train_idx], basis_rank1, rank1_bundle.weights_one_sided, return_complex=False).reshape(-1)
    scale = _fit_amplitude_scale(of_train, emp_train)
    of_test = rankk_gls_coefficients(rank1_bundle.traces_freq[test_idx], rank1_bundle.template_freq, rank1_bundle.weights_one_sided, return_complex=False).reshape(-1)
    emp_test = rankk_gls_coefficients(rank1_bundle.traces_freq[test_idx], basis_rank1, rank1_bundle.weights_one_sided, return_complex=False).reshape(-1) * scale
    rank1_summary_df = pd.DataFrame(
        [
            {
                "weighted_subspace_cosine": float(weighted_cosine(basis_rank1, rank1_bundle.template_freq, rank1_bundle.weights_one_sided)),
                "amplitude_correlation": _pearsonr(of_test, emp_test),
                "median_relative_error": _median_relative_error(emp_test, of_test),
                "residual_ks_pvalue": float(stats.ks_2samp(of_test, emp_test).pvalue),
            }
        ]
    )

    clean_ref = rank1_truth["template_time"]
    crb_rows = []
    for noise_type in ("white", "pink", "brownian"):
        chosen_power, _ = choose_noise_power_for_target_fisher(clean_ref, noise_type, cfg)
        ng = stationary_noise_generator(cfg, noise_type=noise_type, noise_power=chosen_power, rng=np.random.default_rng(cfg.seed))
        _, psd = ng.build_psd(cfg.trace_len)
        weights = build_of_one_sided_weights(psd, cfg.trace_len)
        clean_f = np.fft.rfft(clean_ref)
        predicted = compute_of_statistics(clean_f, weights)
        noisy = np.stack([clean_ref + ng.generate_noise(cfg.trace_len) for _ in range(cfg.crb_replicates)], axis=0)
        amp_hat = compute_of_amplitudes(noisy, clean_ref, psd, cfg.sampling_frequency)
        sigma2_emp = float(np.var(amp_hat, ddof=1))
        sigma2_pred = float(predicted["variance_crb"])
        crb_rows.append(
            {
                "noise_type": noise_type,
                "noise_power": chosen_power,
                "N_phi": float(predicted["fisher_information"]),
                "variance_pred": sigma2_pred,
                "variance_emp": sigma2_emp,
                "relative_error": float(abs(sigma2_emp - sigma2_pred) / max(sigma2_pred, 1e-12)),
            }
        )
    crb_df = pd.DataFrame(crb_rows)

    resolution_rows = []
    noise_powers = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    for power in noise_powers:
        bundle, truth = simulate_rank1_batch(
            cfg,
            n_events=cfg.sim_events_medium,
            amp_range=(1.0, 1.0),
            noise_type="pink",
            noise_power=float(power),
        )
        amps = compute_of_amplitudes(bundle.traces_baseline, bundle.template_time, bundle.psd_one_sided, cfg.sampling_frequency)
        stats_dict = compute_of_statistics(bundle.template_freq, bundle.weights_one_sided)
        resolution_rows.append(
            {
                "noise_power": float(power),
                "sigma_emp": float(np.std(amps, ddof=1)),
                "sigma_pred": float(stats_dict["sigma_amplitude"]),
                "relative_error": float(abs(np.std(amps, ddof=1) - stats_dict["sigma_amplitude"]) / max(stats_dict["sigma_amplitude"], 1e-12)),
            }
        )
    resolution_df = pd.DataFrame(resolution_rows)
    slope = float(np.polyfit(np.log(resolution_df["noise_power"]), np.log(resolution_df["sigma_emp"]), 1)[0])

    return {
        "rank1_summary_df": rank1_summary_df,
        "crb_df": crb_df,
        "resolution_df": resolution_df,
        "resolution_summary": {"loglog_slope_sigma_emp_vs_noise_power": slope},
    }


def run_block05_bridge_suite(cfg: CanonicalConfig) -> dict[str, Any]:
    bundle, truth = simulate_controlled_family(
        cfg,
        n_events=cfg.sim_events_medium,
        tau_decay_range=(1e6, 5e6),
        t0_jitter_range=(-1e5, 1e5),
        n_qp_range=(3000, 7000),
        noise_type="pink",
        noise_power=1.0,
        arrival_mode="aligned",
    )
    train_idx = bundle.split_indices["train"]
    test_idx = bundle.split_indices["test"]
    rows = []
    for k in (1, 2, 3):
        exact = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=k)
        emp = fit_weighted_empca(
            bundle.traces_freq[train_idx],
            bundle.weights_one_sided,
            k=k,
            n_iter=cfg.default_empca_iter,
            patience=cfg.default_empca_patience,
            init="svd",
            seed=cfg.seed,
        )
        cosines, angles = principal_angles_weighted(emp["basis"], exact["basis_native"], bundle.weights_one_sided)
        residual_summary = compute_residual_summary(
            bundle.traces_freq[test_idx],
            emp["basis"],
            exact["basis_native"],
            bundle.weights_one_sided,
        )
        rows.append(
            {
                "k": k,
                "principal_angle_cosines": [float(x) for x in cosines],
                "principal_angles_deg": [float(x) for x in angles],
                **residual_summary,
            }
        )
    bridge_df = pd.DataFrame(rows)
    return {"bundle": bundle, "truth": truth, "bridge_df": bridge_df}


def run_block06_convergence_suite(cfg: CanonicalConfig) -> dict[str, Any]:
    bundle_a, _ = simulate_rank1_batch(
        cfg,
        n_events=cfg.sim_events_medium,
        amp_range=(0.8, 1.2),
        noise_type="pink",
        noise_power=1.0,
    )
    bundle_b, _ = simulate_controlled_family(
        cfg,
        n_events=cfg.sim_events_medium,
        tau_decay_range=(1e6, 5e6),
        t0_jitter_range=(-1e5, 1e5),
        n_qp_range=(3000, 7000),
        noise_type="pink",
        noise_power=1.0,
        arrival_mode="aligned",
    )
    mono_rows = []
    for label, bundle in (("setup_A_rank1", bundle_a), ("setup_B_multiD", bundle_b)):
        train_idx = bundle.split_indices["train"]
        test_idx = bundle.split_indices["test"]
        exact = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=cfg.empirical_rank_max)
        for k in range(1, cfg.empirical_rank_max + 1):
            basis = exact["basis_native"][:k]
            coeff = rankk_gls_coefficients(bundle.traces_freq[test_idx], basis, bundle.weights_one_sided, return_complex=True)
            resid = residual_energy_per_trace(bundle.traces_freq[test_idx], basis, coeff, bundle.weights_one_sided)
            mono_rows.append(
                {
                    "setup": label,
                    "k": k,
                    "chi2_proxy_mean": float(np.mean(resid)),
                    "chi2_proxy_std": float(np.std(resid)),
                }
            )
    monotonicity_df = pd.DataFrame(mono_rows)

    train_idx = bundle_b.split_indices["train"]
    convergence_rows = []
    summary_rows = []
    for k in (1, 2, 3):
        for init in ("random", "svd"):
            fit = fit_weighted_empca(
                bundle_b.traces_freq[train_idx],
                bundle_b.weights_one_sided,
                k=k,
                n_iter=80,
                patience=80,
                init=init,
                seed=cfg.seed,
            )
            chi2 = fit["chi2_trace"]
            delta = np.diff(chi2)
            tol_hits = np.where(np.abs(delta) / max(chi2[0], 1e-12) < 1e-6)[0]
            iter_to_tol = int(tol_hits[0] + 1) if len(tol_hits) else int(len(chi2))
            summary_rows.append(
                {
                    "k": k,
                    "init": init,
                    "iterations_used": int(fit["n_iter_used"]),
                    "monotone_nonincreasing": bool(np.all(delta <= 1e-9)),
                    "iter_to_relative_tol_1e-6": iter_to_tol,
                    "chi2_final": float(chi2[-1]),
                }
            )
            for step, value in enumerate(chi2):
                convergence_rows.append({"k": k, "init": init, "iteration": step, "chi2": float(value)})
    convergence_df = pd.DataFrame(convergence_rows)
    convergence_summary_df = pd.DataFrame(summary_rows)

    rank_summary_df = (
        monotonicity_df[monotonicity_df["setup"] == "setup_B_multiD"]
        .sort_values("k")
        .assign(relative_drop_vs_k1=lambda df: (df["chi2_proxy_mean"].iloc[0] - df["chi2_proxy_mean"]) / df["chi2_proxy_mean"].iloc[0])
    )
    return {
        "monotonicity_df": monotonicity_df,
        "convergence_df": convergence_df,
        "convergence_summary_df": convergence_summary_df,
        "rank_summary_df": rank_summary_df,
    }


def run_block07_ablation_suite(cfg: CanonicalConfig) -> dict[str, Any]:
    base_clean_bundle, base_truth = simulate_controlled_family(
        cfg,
        n_events=cfg.sim_events_medium,
        tau_decay_range=(1e6, 5e6),
        t0_jitter_range=(-8e4, 8e4),
        n_qp_range=(3000, 7000),
        noise_type="white",
        noise_power=1.0,
        arrival_mode="aligned",
    )
    clean_traces = base_truth["clean_traces"]
    split = base_clean_bundle.split_indices
    ablation_rows = []
    for noise_type in ("white", "pink", "brownian"):
        ng = stationary_noise_generator(cfg, noise_type=noise_type, noise_power=1.0, rng=np.random.default_rng(cfg.seed))
        _, psd = ng.build_psd(cfg.trace_len)
        noise = np.stack([ng.generate_noise(cfg.trace_len) for _ in range(len(clean_traces))], axis=0)
        bundle = prepare_bundle(
            clean_traces + noise,
            template_time=base_clean_bundle.template_time,
            psd_one_sided=psd,
            cfg=cfg,
            metadata={"source": f"SIM-ablation-{noise_type}", "noise_type": noise_type, "template_path": "QPSimulator nominal template", "psd_path": f"analytic:{noise_type}"},
        )
        train_idx = split["train"]
        test_idx = split["test"]
        exact_w = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=3)["basis_native"]
        exact_i = exact_isotropic_subspace(bundle.traces_freq[train_idx], k=3)
        coeff_w = rankk_gls_coefficients(bundle.traces_freq[test_idx], exact_w, bundle.weights_one_sided, return_complex=True)
        coeff_i = rankk_gls_coefficients(bundle.traces_freq[test_idx], exact_i, bundle.weights_one_sided, return_complex=True)
        resid_w = residual_energy_per_trace(bundle.traces_freq[test_idx], exact_w, coeff_w, bundle.weights_one_sided)
        resid_i = residual_energy_per_trace(bundle.traces_freq[test_idx], exact_i, coeff_i, bundle.weights_one_sided)
        ablation_rows.append(
            {
                "noise_type": noise_type,
                "weighted_residual_mean": float(np.mean(resid_w)),
                "isotropic_residual_mean": float(np.mean(resid_i)),
                "relative_improvement": float((np.mean(resid_i) - np.mean(resid_w)) / max(np.mean(resid_i), 1e-12)),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)

    mismatch_bundle, mismatch_truth = simulate_controlled_family(
        cfg,
        n_events=cfg.sim_events_medium,
        tau_decay_range=(1e6, 5e6),
        t0_jitter_range=(0.0, 0.0),
        n_qp_range=(4000, 6000),
        noise_type="pink",
        noise_power=1.0,
        arrival_mode="aligned",
    )
    train_idx = mismatch_bundle.split_indices["train"]
    test_idx = mismatch_bundle.split_indices["test"]
    nominal_template = mismatch_bundle.template_freq

    fit_k1 = fit_weighted_empca(
        mismatch_bundle.traces_freq[train_idx], mismatch_bundle.weights_one_sided, k=1,
        n_iter=cfg.default_empca_iter, patience=cfg.default_empca_patience,
        init="template", template_f=nominal_template, seed=cfg.seed,
    )
    fit_k2 = fit_weighted_empca(
        mismatch_bundle.traces_freq[train_idx], mismatch_bundle.weights_one_sided, k=2,
        n_iter=cfg.default_empca_iter, patience=cfg.default_empca_patience,
        init="template", template_f=nominal_template, seed=cfg.seed,
    )
    u_k1 = phase_align_basis(fit_k1["basis"][0], nominal_template, mismatch_bundle.weights_one_sided)
    coeff_of_train = rankk_gls_coefficients(mismatch_bundle.traces_freq[train_idx], nominal_template, mismatch_bundle.weights_one_sided, return_complex=False).reshape(-1)
    coeff_k1_train = rankk_gls_coefficients(mismatch_bundle.traces_freq[train_idx], u_k1, mismatch_bundle.weights_one_sided, return_complex=False).reshape(-1)
    coeff_k2_train = rankk_gls_coefficients(mismatch_bundle.traces_freq[train_idx], fit_k2["basis"], mismatch_bundle.weights_one_sided, return_complex=False)
    amp_train = mismatch_truth["amplitude_true"][train_idx]
    amp_test = mismatch_truth["amplitude_true"][test_idx]
    map_k1 = _linear_map_from_coeff(coeff_k1_train, amp_train)
    map_k2 = _linear_map_from_coeff(coeff_k2_train, amp_train)
    coeff_of_test = rankk_gls_coefficients(mismatch_bundle.traces_freq[test_idx], nominal_template, mismatch_bundle.weights_one_sided, return_complex=False).reshape(-1)
    coeff_k1_test = rankk_gls_coefficients(mismatch_bundle.traces_freq[test_idx], u_k1, mismatch_bundle.weights_one_sided, return_complex=False).reshape(-1)
    coeff_k2_test = rankk_gls_coefficients(mismatch_bundle.traces_freq[test_idx], fit_k2["basis"], mismatch_bundle.weights_one_sided, return_complex=False)
    amp_k1 = _predict_from_linear_map(coeff_k1_test, map_k1)
    amp_k2 = _predict_from_linear_map(coeff_k2_test, map_k2)

    mismatch_df = pd.DataFrame(
        [
            {
                "method": "OF_nominal",
                "mean_relative_bias": float(np.mean(coeff_of_test - amp_test) / np.mean(amp_test)),
                "rmse": float(np.sqrt(np.mean((coeff_of_test - amp_test) ** 2))),
            },
            {
                "method": "EMPCA_k1",
                "mean_relative_bias": float(np.mean(amp_k1 - amp_test) / np.mean(amp_test)),
                "rmse": float(np.sqrt(np.mean((amp_k1 - amp_test) ** 2))),
            },
            {
                "method": "EMPCA_k2",
                "mean_relative_bias": float(np.mean(amp_k2 - amp_test) / np.mean(amp_test)),
                "rmse": float(np.sqrt(np.mean((amp_k2 - amp_test) ** 2))),
            },
        ]
    )

    curve_rows = []
    nominal_clean, _ = make_clean_qp_trace(cfg, n_qp=5000, tau_decay=3e6, arrival_mode="aligned")
    nominal_f = np.fft.rfft(normalize_template_peak(nominal_clean, cfg.pretrigger))
    for tau_decay in np.linspace(1e6, 5e6, 9):
        clean_tau, _ = make_clean_qp_trace(cfg, n_qp=5000, tau_decay=float(tau_decay), arrival_mode="aligned")
        clean_tau = normalize_template_peak(clean_tau, cfg.pretrigger)
        clean_tau_f = np.fft.rfft(clean_tau)
        cosine = weighted_cosine(clean_tau_f, nominal_f, mismatch_bundle.weights_one_sided)
        a_of_clean = rankk_gls_coefficients(clean_tau_f[None, :], nominal_f, mismatch_bundle.weights_one_sided, return_complex=False).reshape(-1)[0]
        curve_rows.append(
            {
                "tau_decay": float(tau_decay),
                "cosine_squared": float(cosine ** 2),
                "of_estimate_for_unit_shape": float(a_of_clean),
            }
        )
    mismatch_curve_df = pd.DataFrame(curve_rows)

    jitter_bundle, jitter_truth = simulate_controlled_family(
        cfg,
        n_events=cfg.sim_events_small,
        tau_decay_range=(3e6, 3e6),
        t0_jitter_range=(-0.10 * cfg.trace_len / cfg.sampling_frequency * 1e9, 0.10 * cfg.trace_len / cfg.sampling_frequency * 1e9),
        n_qp_range=(5000, 5000),
        noise_type="pink",
        noise_power=1.0,
        arrival_mode="aligned",
    )
    test_idx = jitter_bundle.split_indices["test"]
    shift_grid = _time_shift_grid(cfg, n_points=31, frac=0.10)
    filters = []
    for shift_ns in shift_grid:
        template = make_qp_simulator(cfg).get_template_at_shift(float(shift_ns))
        filters.append(OptimumFilter(template, jitter_bundle.psd_one_sided, cfg.sampling_frequency))
    best_amp = []
    best_shift_samples = []
    for trace in jitter_bundle.traces_baseline[test_idx]:
        amps = np.array([flt.fit(trace)[0] for flt in filters], dtype=np.float64)
        best_idx = int(np.argmax(amps))
        best_amp.append(float(amps[best_idx]))
        best_shift_samples.append(float(shift_grid[best_idx] / (1e9 / cfg.sampling_frequency)))
    best_amp = np.asarray(best_amp)
    best_shift_samples = np.asarray(best_shift_samples)
    fixed_amp = compute_of_amplitudes(
        jitter_bundle.traces_baseline[test_idx],
        jitter_bundle.template_time,
        jitter_bundle.psd_one_sided,
        cfg.sampling_frequency,
    )
    true_shift_samples = jitter_truth["t0_shift"][test_idx] / (1e9 / cfg.sampling_frequency)
    shift_df = pd.DataFrame(
        [
            {
                "method": "time_shift_OF",
                "arrival_time_rmse_samples": float(np.sqrt(np.mean((best_shift_samples - true_shift_samples) ** 2))),
                "amplitude_rmse": float(np.sqrt(np.mean((best_amp - jitter_truth["amplitude_true"][test_idx]) ** 2))),
                "mean_relative_bias": float(np.mean(best_amp - jitter_truth["amplitude_true"][test_idx]) / np.mean(jitter_truth["amplitude_true"][test_idx])),
            },
            {
                "method": "fixed_OF",
                "arrival_time_rmse_samples": np.nan,
                "amplitude_rmse": float(np.sqrt(np.mean((fixed_amp - jitter_truth["amplitude_true"][test_idx]) ** 2))),
                "mean_relative_bias": float(np.mean(fixed_amp - jitter_truth["amplitude_true"][test_idx]) / np.mean(jitter_truth["amplitude_true"][test_idx])),
            },
        ]
    )

    return {
        "ablation_df": ablation_df,
        "mismatch_df": mismatch_df,
        "mismatch_curve_df": mismatch_curve_df,
        "shift_df": shift_df,
    }


def shift_trace_fractional(trace: np.ndarray, shift_samples: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    grid = np.arange(trace.shape[0], dtype=np.float64)
    return np.interp(grid, grid - shift_samples, trace, left=trace[0], right=trace[-1])


def broaden_trace(trace: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(trace, dtype=np.float64)
    return gaussian_filter1d(np.asarray(trace, dtype=np.float64), sigma=sigma, mode="nearest")


def build_proxy_family(template_time: np.ndarray, cfg: CanonicalConfig) -> dict[str, np.ndarray]:
    centered = baseline_correct(template_time[None, :], cfg.pretrigger)[0]
    timing = shift_trace_fractional(centered, +2.0) - shift_trace_fractional(centered, -2.0)
    width = broaden_trace(centered, sigma=4.0) - broaden_trace(centered, sigma=1.0)
    mean_like = centered / max(np.linalg.norm(centered), np.finfo(float).eps)
    return {
        "mean_like": mean_like,
        "template_like": mean_like,
        "timing_like": timing / max(np.linalg.norm(timing), np.finfo(float).eps),
        "width_like": width / max(np.linalg.norm(width), np.finfo(float).eps),
    }


def run_block08_pc_suite(cfg: CanonicalConfig) -> dict[str, Any]:
    bundle = load_real_bundle(cfg)
    train_idx = subsample_indices(bundle.split_indices["train"], cfg.real_train_cap, cfg.seed)
    test_idx = subsample_indices(bundle.split_indices["test"], cfg.real_eval_cap, cfg.seed + 1)
    proxy_time = build_proxy_family(bundle.template_time, cfg)
    proxy_freq = {name: np.fft.rfft(value) for name, value in proxy_time.items()}

    uncentered = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=3)["basis_native"]
    mean_train = np.mean(bundle.traces_baseline[train_idx], axis=0, keepdims=True)
    centered_freq = rfft_traces(bundle.traces_baseline - mean_train)
    centered = exact_weighted_subspace(centered_freq[train_idx], bundle.weights_one_sided, k=3)["basis_native"]

    overlap_rows = []
    for label, basis in (("uncentered", uncentered), ("centered", centered)):
        for comp_idx in range(3):
            component = basis[comp_idx]
            for proxy_name, proxy in proxy_freq.items():
                overlap_rows.append(
                    {
                        "fit_variant": label,
                        "component": f"PC{comp_idx + 1}",
                        "proxy": proxy_name,
                        "weighted_cosine": float(weighted_cosine(component, proxy, bundle.weights_one_sided)),
                    }
                )
    overlap_df = pd.DataFrame(overlap_rows)

    coeff_unc = rankk_gls_coefficients(bundle.traces_freq[test_idx], uncentered, bundle.weights_one_sided, return_complex=False)
    amp_proxy = rankk_gls_coefficients(bundle.traces_freq[test_idx], proxy_freq["template_like"], bundle.weights_one_sided, return_complex=False).reshape(-1)
    timing_proxy = rankk_gls_coefficients(bundle.traces_freq[test_idx], proxy_freq["timing_like"], bundle.weights_one_sided, return_complex=False).reshape(-1)
    width_proxy = rankk_gls_coefficients(bundle.traces_freq[test_idx], proxy_freq["width_like"], bundle.weights_one_sided, return_complex=False).reshape(-1)

    corr_rows = []
    proxy_map = {
        "amplitude_proxy": amp_proxy,
        "timing_proxy": timing_proxy,
        "width_proxy": width_proxy,
        "rq_A": bundle.rqs.iloc[test_idx]["A"].to_numpy(),
        "rq_time_shift": bundle.rqs.iloc[test_idx]["time_shift"].to_numpy(),
    }
    for comp_idx in range(coeff_unc.shape[1]):
        for proxy_name, proxy_values in proxy_map.items():
            corr_rows.append(
                {
                    "component": f"PC{comp_idx + 1}",
                    "proxy": proxy_name,
                    "pearson_r": _pearsonr(coeff_unc[:, comp_idx], proxy_values),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    summary = {"note": "Interpretation is descriptive: overlap and correlation, not proof of physical identity."}
    return {"overlap_df": overlap_df, "corr_df": corr_df, "summary": summary}


def run_block09_robustness_suite(cfg: CanonicalConfig) -> dict[str, Any]:
    base_bundle, base_truth = simulate_controlled_family(
        cfg,
        n_events=cfg.sim_events_medium,
        tau_decay_range=(1e6, 5e6),
        t0_jitter_range=(-8e4, 8e4),
        n_qp_range=(3500, 6500),
        noise_type="pink",
        noise_power=1.0,
        arrival_mode="aligned",
    )
    clean = base_truth["clean_traces"]
    amp_true = base_truth["amplitude_true"]
    rng = np.random.default_rng(cfg.seed)

    base_ng = stationary_noise_generator(cfg, noise_type="pink", noise_power=1.0, rng=rng)
    stationary_noise = np.stack([base_ng.generate_noise(cfg.trace_len) for _ in range(len(clean))], axis=0)
    stationary_noisy = clean + stationary_noise
    _, global_psd_stationary = base_ng.build_psd(cfg.trace_len)

    temporal = TemporalNoiseWrapper(
        {
            "mode": "piecewise",
            "n_segments": 4,
            "crossfade_len": 64,
            "vary_noise_power": True,
            "noise_power_scale_range": [0.7, 1.3],
        },
        rng=rng,
    )
    nonstationary_noise = np.stack(
        [temporal.apply(np.zeros(cfg.trace_len), base_generator=base_ng) for _ in range(len(clean))],
        axis=0,
    )
    nonstationary_noisy = clean + nonstationary_noise
    _, global_psd_nonstationary = QPSimulator.estimate_psd(nonstationary_noise, cfg.sampling_frequency)

    segment_psds = []
    segment_size = len(clean) // 4
    for seg in range(4):
        seg_slice = slice(seg * segment_size, (seg + 1) * segment_size if seg < 3 else len(clean))
        _, seg_psd = QPSimulator.estimate_psd(nonstationary_noise[seg_slice], cfg.sampling_frequency)
        segment_psds.append(seg_psd)

    nominal_template = base_bundle.template_time
    stationary_amp = compute_of_amplitudes(stationary_noisy, nominal_template, global_psd_stationary, cfg.sampling_frequency)
    global_amp = compute_of_amplitudes(nonstationary_noisy, nominal_template, global_psd_nonstationary, cfg.sampling_frequency)
    segment_amp = np.zeros(len(clean), dtype=np.float64)
    for seg in range(4):
        seg_slice = slice(seg * segment_size, (seg + 1) * segment_size if seg < 3 else len(clean))
        segment_amp[seg_slice] = compute_of_amplitudes(nonstationary_noisy[seg_slice], nominal_template, segment_psds[seg], cfg.sampling_frequency)

    nonstationary_df = pd.DataFrame(
        [
            {
                "case": "stationary_global_psd",
                "amplitude_rmse": float(np.sqrt(np.mean((stationary_amp - amp_true) ** 2))),
                "amplitude_bias": float(np.mean(stationary_amp - amp_true)),
            },
            {
                "case": "nonstationary_global_psd",
                "amplitude_rmse": float(np.sqrt(np.mean((global_amp - amp_true) ** 2))),
                "amplitude_bias": float(np.mean(global_amp - amp_true)),
            },
            {
                "case": "nonstationary_segment_psd",
                "amplitude_rmse": float(np.sqrt(np.mean((segment_amp - amp_true) ** 2))),
                "amplitude_bias": float(np.mean(segment_amp - amp_true)),
            },
        ]
    )

    artifact_ng = stationary_noise_generator(cfg, noise_type="pink", noise_power=1.0, rng=np.random.default_rng(cfg.seed + 1))
    artifact_noise = np.stack([artifact_ng.generate_noise(cfg.trace_len) for _ in range(len(clean))], axis=0)
    injector = ArtifactInjector(
        {
            "sampling_frequency": cfg.sampling_frequency,
            "enable_glitches": True,
            "glitch_rate": 0.1,
            "glitch_amp_range": [0.1, 0.4],
            "enable_sparse_impulses": True,
            "impulse_probability": 5e-5,
            "impulse_sigma": 0.3,
        },
        rng=np.random.default_rng(cfg.seed + 2),
    )
    artifacted = np.stack([injector.apply(clean[idx] + artifact_noise[idx]) for idx in range(len(clean))], axis=0)
    artifact_bundle = prepare_bundle(
        artifacted,
        template_time=base_bundle.template_time,
        psd_one_sided=base_bundle.psd_one_sided,
        cfg=cfg,
        metadata={"source": "SIM-artifacted", "template_path": "QPSimulator nominal template", "psd_path": "analytic:pink"},
    )
    train_idx = artifact_bundle.split_indices["train"]
    test_idx = artifact_bundle.split_indices["test"]

    fit_dirty = fit_weighted_empca(
        artifact_bundle.traces_freq[train_idx], artifact_bundle.weights_one_sided, k=2,
        n_iter=cfg.default_empca_iter, patience=cfg.default_empca_patience, init="svd", seed=cfg.seed,
    )
    coeff_train_dirty = rankk_gls_coefficients(artifact_bundle.traces_freq[train_idx], fit_dirty["basis"], artifact_bundle.weights_one_sided, return_complex=False)
    map_dirty = _linear_map_from_coeff(coeff_train_dirty, amp_true[train_idx])
    coeff_test_dirty = rankk_gls_coefficients(artifact_bundle.traces_freq[test_idx], fit_dirty["basis"], artifact_bundle.weights_one_sided, return_complex=False)
    amp_pred_dirty = _predict_from_linear_map(coeff_test_dirty, map_dirty)
    resid_train_dirty = residual_energy_per_trace(artifact_bundle.traces_freq[train_idx], fit_dirty["basis"], coeff_train_dirty, artifact_bundle.weights_one_sided)
    threshold = float(np.quantile(resid_train_dirty, 0.90))
    keep_train = train_idx[resid_train_dirty <= threshold]

    fit_clean = fit_weighted_empca(
        artifact_bundle.traces_freq[keep_train], artifact_bundle.weights_one_sided, k=2,
        n_iter=cfg.default_empca_iter, patience=cfg.default_empca_patience, init="svd", seed=cfg.seed,
    )
    coeff_train_clean = rankk_gls_coefficients(artifact_bundle.traces_freq[keep_train], fit_clean["basis"], artifact_bundle.weights_one_sided, return_complex=False)
    map_clean = _linear_map_from_coeff(coeff_train_clean, amp_true[keep_train])
    coeff_test_clean = rankk_gls_coefficients(artifact_bundle.traces_freq[test_idx], fit_clean["basis"], artifact_bundle.weights_one_sided, return_complex=False)
    amp_pred_clean = _predict_from_linear_map(coeff_test_clean, map_clean)

    artifact_df = pd.DataFrame(
        [
            {
                "pass": "contaminated_fit",
                "amplitude_rmse": float(np.sqrt(np.mean((amp_pred_dirty - amp_true[test_idx]) ** 2))),
                "weighted_residual_mean": float(np.mean(residual_energy_per_trace(artifact_bundle.traces_freq[test_idx], fit_dirty["basis"], coeff_test_dirty, artifact_bundle.weights_one_sided))),
            },
            {
                "pass": "flagged_refit",
                "amplitude_rmse": float(np.sqrt(np.mean((amp_pred_clean - amp_true[test_idx]) ** 2))),
                "weighted_residual_mean": float(np.mean(residual_energy_per_trace(artifact_bundle.traces_freq[test_idx], fit_clean["basis"], coeff_test_clean, artifact_bundle.weights_one_sided))),
            },
        ]
    )

    multi = MultiChannelNoiseGenerator(
        {"noise_type": "pink", "noise_power": 1.0, "sampling_frequency": cfg.sampling_frequency},
        config={"mode": "shared_private", "n_channels": 4, "corr_strength": 0.35},
        seed=cfg.seed,
    )
    X, meta = multi.generate(cfg.trace_len, C=4, return_metadata=True)
    multichannel_df = pd.DataFrame(
        [
            {
                "n_channels": 4,
                "corr_strength_config": 0.35,
                "mean_offdiag_corr": float(meta["mean_offdiag_corr"]),
                "std_summed_trace": float(np.std(np.sum(X, axis=0))),
            }
        ]
    )
    return {
        "nonstationary_df": nonstationary_df,
        "artifact_df": artifact_df,
        "multichannel_df": multichannel_df,
    }


# ======================================================================
# PSD unit conventions (G2 repair) and blocks 11-12
# ======================================================================
#
# Two PSD conventions coexist in this repository:
#
# 1. *Physical* one-sided PSD, units A^2/Hz, normalised so that
#    sum(J_k) * fs / N equals the mean noise power. This is what
#    ``QPSimulator.estimate_psd`` returns and what ``OptimumFilter``
#    expects as input (its internal fft/fs scaling assumes it).
#
# 2. *DFT-power* spectrum, J_dft[k] = E|rfft(noise)[k]|^2, which is what
#    ``NoiseGenerator.build_psd`` returns. For raw-``np.fft.rfft`` data,
#    OF-convention weights MUST be built from THIS convention:
#    with w = build_of_one_sided_weights(J_dft, N), the GLS amplitude
#    satisfies Var(A_hat) = 1 / N_Phi with N_Phi = <s, s>_w exactly
#    (verified empirically to ~1% for white/pink/Brownian).
#
# The two differ by the exact factor  J_dft = J_phys * (N * fs / 2).
# Mixing them (physical-PSD weights with raw-rfft data) inflates a
# predicted sigma by sqrt(N*fs/2); the draft's Fig 7 "Brownian 184x"
# is consistent with such a convention mix (184^2 ~ 2N for N = 16384).
# All block-11/12 code states explicitly which convention each array is in.


def psd_physical_to_dft(J_phys: np.ndarray, trace_len: int, fs: float) -> np.ndarray:
    """Physical one-sided PSD (A^2/Hz) -> DFT-power convention E|rfft|^2."""
    return np.asarray(J_phys, dtype=np.float64) * (trace_len * fs / 2.0)


def psd_dft_to_physical(J_dft: np.ndarray, trace_len: int, fs: float) -> np.ndarray:
    """DFT-power convention E|rfft|^2 -> physical one-sided PSD (A^2/Hz)."""
    return np.asarray(J_dft, dtype=np.float64) / (trace_len * fs / 2.0)


def raw_mse_from_freq_residual(resid_f: np.ndarray, trace_len: int) -> np.ndarray:
    """Per-trace time-domain MSE from an rfft-domain residual (Parseval)."""
    resid_f = np.atleast_2d(np.asarray(resid_f))
    mag2 = np.abs(resid_f) ** 2
    total = mag2[:, 0].copy()
    if trace_len % 2 == 0:
        total += 2.0 * np.sum(mag2[:, 1:-1], axis=1) + mag2[:, -1]
    else:
        total += 2.0 * np.sum(mag2[:, 1:], axis=1)
    return total / (trace_len ** 2)


def amplitude_basis_from_subspace(
    basis: np.ndarray,
    template_f: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Rotate a rank-k basis so row 0 is the w-normalized projection of the
    template onto span(basis); remaining rows complete a w-orthonormal set.

    The GLS coefficient on row 0 is then the physically meaningful
    'amplitude at rank k' (PC2..k absorb shape variation w-orthogonally)."""
    U = np.atleast_2d(np.asarray(basis, dtype=np.complex128))
    w = np.asarray(weights, dtype=np.float64)
    coeff = project_gls(np.asarray(template_f)[None, :], U, w, return_complex=True)
    proj = np.asarray(coeff).reshape(-1) @ U
    proj = normalize_basis_weighted_unit(proj, w)
    proj = phase_align_basis(proj, np.asarray(template_f), w)
    stacked = np.vstack([proj[None, :], U])
    ortho = w_orthonormalize(stacked, w)
    norms = np.sqrt(np.abs(np.array([np.real(weighted_inner(row, row, w)) for row in ortho])))
    keep = ortho[norms > 1e-6]
    return keep[: U.shape[0]]


def template_unit_amplitudes(
    X_f: np.ndarray,
    basis_amp: np.ndarray,
    template_f: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """GLS amplitude in TEMPLATE units from a w-orthonormal amplitude basis.

    Row 0 of ``basis_amp`` is w-normalized, so its raw GLS coefficient is in
    w-norm units; dividing by gamma = Re<u1, s>_w converts to the same units
    as the OF amplitude (template-peak ADC)."""
    coeff = np.atleast_2d(
        rankk_gls_coefficients(X_f, basis_amp, weights, return_complex=False)
    )
    gamma = float(np.real(weighted_inner(basis_amp[0], np.asarray(template_f), weights)))
    if abs(gamma) < 1e-30:
        raise ValueError("amplitude basis has no overlap with template")
    return coeff[:, 0] / gamma


def simulate_jitter_family(
    cfg: CanonicalConfig,
    n_events: int,
    noise_type: str,
    noise_power: float,
    seed: int,
    jitter_ns: float = 2e5,
    n_qp: int = 5000,
    trigger_sample: int | None = None,
    n_null: int | None = None,
) -> dict[str, Any]:
    """Fixed-amplitude, timing-jittered pulse family (the paper's Exp E set).

    All QPs arrive simultaneously (fixed shape), n_qp fixed (fixed true
    amplitude), t0 ~ Uniform(-jitter, +jitter) -> genuinely 2-D signal family
    (mean pulse + timing derivative). Trigger placed after cfg.pretrigger so
    baseline windows stay signal-free. Returns clean/noisy/null traces, a
    unit-peak template, and the PSD in BOTH conventions."""
    rng = np.random.default_rng(seed)
    if trigger_sample is None:
        trigger_sample = cfg.pretrigger + 500
    if trigger_sample >= cfg.trace_len:
        raise ValueError("trigger_sample beyond trace end")
    dt_ns = 1e9 / cfg.sampling_frequency
    trigger_ns = trigger_sample * dt_ns
    base_sim = QPSimulator(
        sampling_frequency=cfg.sampling_frequency,
        trace_samples=cfg.trace_len,
        trigger_time=trigger_ns,
    )
    t0_shifts = rng.uniform(-jitter_ns, jitter_ns, size=n_events)
    clean = np.empty((n_events, cfg.trace_len), dtype=np.float64)
    amp_true = float(n_qp) * base_sim.qp_amplitude
    for i in range(n_events):
        ev_sim = QPSimulator(
            sampling_frequency=cfg.sampling_frequency,
            trace_samples=cfg.trace_len,
            trigger_time=trigger_ns + float(t0_shifts[i]),
        )
        clean[i] = ev_sim.generate(np.zeros(n_qp, dtype=float))
    template_time = base_sim.generate(np.zeros(n_qp, dtype=float)) / amp_true

    ng = NoiseGenerator(
        {"noise_type": noise_type, "noise_power": float(noise_power),
         "sampling_frequency": cfg.sampling_frequency},
        rng=rng,
    )
    _, J_dft = ng.build_psd(cfg.trace_len)
    J_phys = psd_dft_to_physical(J_dft, cfg.trace_len, cfg.sampling_frequency)
    noise = np.stack([ng.generate_noise(cfg.trace_len) for _ in range(n_events)], axis=0)
    n_null = n_events if n_null is None else int(n_null)
    null_noise = np.stack([ng.generate_noise(cfg.trace_len) for _ in range(n_null)], axis=0)
    return {
        "clean": clean,
        "noisy": clean + noise,
        "null_noise": null_noise,
        "template_time": template_time,
        "t0_shift_ns": t0_shifts,
        "amp_true": amp_true,
        "J_dft": J_dft,
        "J_phys": J_phys,
        "seed": int(seed),
        "noise_type": noise_type,
        "noise_power": float(noise_power),
    }


def _split_two_way(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_frac * n))
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def run_block11_crb_units(cfg: CanonicalConfig, n_replicates: int | None = None,
                          noise_types: tuple[str, ...] = ("white", "pink", "brownian")) -> pd.DataFrame:
    """E4 repair: CRB attainment with explicitly consistent PSD conventions.

    For each noise color, compares the empirical OF amplitude variance with
    (i) 1/N_Phi from DFT-convention weights, (ii) 1/kernel_normalization from
    OptimumFilter fed the physical PSD, and (iii) the WRONG mixed-convention
    prediction, whose error factor ~ N*fs/2 documents the draft's Fig 7 bug."""
    n_replicates = cfg.crb_replicates if n_replicates is None else int(n_replicates)
    dt_ns = 1e9 / cfg.sampling_frequency
    sim = QPSimulator(
        sampling_frequency=cfg.sampling_frequency,
        trace_samples=cfg.trace_len,
        trigger_time=(cfg.pretrigger + 500) * dt_ns,
    )
    clean = sim.generate(np.zeros(5000, dtype=float))
    amp_true = 5000.0 * sim.qp_amplitude
    template = clean / amp_true
    template_f = np.fft.rfft(template)
    rows = []
    _ALL_TYPES = ("white", "pink", "brownian")
    for noise_type in noise_types:
        type_idx = _ALL_TYPES.index(noise_type)
        ng = NoiseGenerator(
            {"noise_type": noise_type, "noise_power": 1.0,
             "sampling_frequency": cfg.sampling_frequency},
            seed=cfg.seed + 7919 * type_idx,  # decorrelate colors
        )
        _, J_dft = ng.build_psd(cfg.trace_len)
        J_phys = psd_dft_to_physical(J_dft, cfg.trace_len, cfg.sampling_frequency)
        w_dft = build_of_one_sided_weights(J_dft, cfg.trace_len)
        w_phys_wrong = build_of_one_sided_weights(J_phys, cfg.trace_len)
        of = OptimumFilter(template, J_phys, cfg.sampling_frequency)
        amps = np.empty(n_replicates, dtype=np.float64)
        for r in range(n_replicates):
            amps[r] = of.fit(clean + ng.generate_noise(cfg.trace_len))[0]
        var_emp = float(np.var(amps, ddof=1))
        nphi_dft = float(np.real(weighted_inner(template_f, template_f, w_dft)))
        nphi_mixed = float(np.real(weighted_inner(template_f, template_f, w_phys_wrong)))
        kern = float(of._kernel_normalization)
        rows.append({
            "noise_type": noise_type,
            "n_replicates": n_replicates,
            "amp_true": amp_true,
            "amp_mean": float(np.mean(amps)),
            "var_emp": var_emp,
            "var_pred_dft_weights": 1.0 / nphi_dft,
            "var_pred_of_kernel": 1.0 / kern,
            "var_pred_mixed_WRONG": 1.0 / nphi_mixed,
            "rel_err_dft": abs(var_emp - 1.0 / nphi_dft) * nphi_dft,
            "rel_err_kernel": abs(var_emp - 1.0 / kern) * kern,
            "mixed_error_factor": (1.0 / nphi_mixed) / var_emp,
            "expected_mix_factor_N_fs_over_2": cfg.trace_len * cfg.sampling_frequency / 2.0,
        })
    return pd.DataFrame(rows)


def run_block11_sigma_scaling(
    cfg: CanonicalConfig,
    noise_powers: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0),
    n_events: int = 400,
) -> pd.DataFrame:
    """E5 repair: sigma_E vs noise power, consistent conventions, GLS amplitudes."""
    dt_ns = 1e9 / cfg.sampling_frequency
    sim = QPSimulator(
        sampling_frequency=cfg.sampling_frequency,
        trace_samples=cfg.trace_len,
        trigger_time=(cfg.pretrigger + 500) * dt_ns,
    )
    clean = sim.generate(np.zeros(5000, dtype=float))
    amp_true = 5000.0 * sim.qp_amplitude
    template_f = np.fft.rfft(clean / amp_true)
    rows = []
    for power in noise_powers:
        ng = NoiseGenerator(
            {"noise_type": "pink", "noise_power": float(power),
             "sampling_frequency": cfg.sampling_frequency},
            seed=cfg.seed + int(round(float(power) * 1000)),  # decorrelate sweep points
        )
        _, J_dft = ng.build_psd(cfg.trace_len)
        w = build_of_one_sided_weights(J_dft, cfg.trace_len)
        noisy = np.stack([clean + ng.generate_noise(cfg.trace_len) for _ in range(n_events)], axis=0)
        amps = project_gls(rfft_traces(noisy), template_f, w, return_complex=False)
        nphi = float(np.real(weighted_inner(template_f, template_f, w)))
        rows.append({
            "noise_power": float(power),
            "sigma_emp": float(np.std(amps, ddof=1)),
            "sigma_pred": float(1.0 / np.sqrt(nphi)),
        })
    df = pd.DataFrame(rows)
    df["relative_error"] = np.abs(df["sigma_emp"] - df["sigma_pred"]) / df["sigma_pred"]
    return df


def run_block11_rank_study(
    cfg: CanonicalConfig,
    n_seeds: int = 8,
    n_events: int = 240,
    ranks: tuple[int, ...] = (1, 2, 3, 4, 5),
    noise_types: tuple[str, ...] = ("white", "pink", "brownian"),
    noise_power: float = 1.0,
    jitter_ns: float = 2e5,
    empca_mode: str = "full",
) -> dict[str, Any]:
    """Exp E repair (Figs 7/15/16): sigma_E(k)/sigma_OF, amplitude bias at
    k=1 vs k=2 under timing jitter, and residual-whiteness KS p versus rank
    with a Monte-Carlo noise-only null (instead of an analytic chi^2 null).

    Uses mode='full' EMPCA (exact coupled M-step) by default."""
    rank_rows = []
    amp_samples = {nt: {1: [], 2: []} for nt in noise_types}
    amp_true_by_type: dict[str, float] = {}
    for noise_type in noise_types:
        for s in range(n_seeds):
            seed = cfg.seed + 1000 * s
            fam = simulate_jitter_family(
                cfg, n_events=n_events, noise_type=noise_type,
                noise_power=noise_power, seed=seed, jitter_ns=jitter_ns,
            )
            amp_true_by_type[noise_type] = fam["amp_true"]
            w = build_of_one_sided_weights(fam["J_dft"], cfg.trace_len)
            template_f = np.fft.rfft(fam["template_time"])
            X_f = rfft_traces(baseline_correct(fam["noisy"], cfg.pretrigger))
            null_f = rfft_traces(baseline_correct(fam["null_noise"], cfg.pretrigger))
            tmpl_f = np.fft.rfft(
                baseline_correct(fam["template_time"][None, :], cfg.pretrigger)[0]
            )
            train_idx, test_idx = _split_two_way(n_events, 0.6, seed)
            # --- OF baseline (zero-shift GLS == OF amplitude) ---
            amp_of = project_gls(X_f[test_idx], tmpl_f, w, return_complex=False)
            sigma_of = float(np.std(amp_of, ddof=1))
            bias_of = float(np.mean(amp_of) - fam["amp_true"]) / fam["amp_true"]
            for k in ranks:
                fit = fit_weighted_empca(
                    X_f[train_idx], w, k=k,
                    n_iter=cfg.default_empca_iter, patience=cfg.default_empca_patience,
                    init="template", template_f=tmpl_f, seed=seed, mode=empca_mode,
                )
                basis_amp = amplitude_basis_from_subspace(fit["basis"], tmpl_f, w)
                amp_k = template_unit_amplitudes(X_f[test_idx], basis_amp, tmpl_f, w)
                resid_test = residual_energy_per_trace(
                    X_f[test_idx], basis_amp,
                    rankk_gls_coefficients(X_f[test_idx], basis_amp, w, return_complex=True), w,
                )
                resid_null = residual_energy_per_trace(
                    null_f, basis_amp,
                    rankk_gls_coefficients(null_f, basis_amp, w, return_complex=True), w,
                )
                ks = stats.ks_2samp(resid_test, resid_null)
                rank_rows.append({
                    "noise_type": noise_type,
                    "seed": seed,
                    "k": int(k),
                    "sigma_E": float(np.std(amp_k, ddof=1)),
                    "sigma_OF": sigma_of,
                    "sigma_ratio": float(np.std(amp_k, ddof=1) / sigma_of),
                    "bias_rel": float(np.mean(amp_k) - fam["amp_true"]) / fam["amp_true"],
                    "bias_rel_OF": bias_of,
                    "ks_stat": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "resid_test_mean": float(np.mean(resid_test)),
                    "resid_null_mean": float(np.mean(resid_null)),
                    "empca_iters": int(fit["n_iter_used"]),
                    "empca_mode": empca_mode,
                })
                if k in (1, 2):
                    amp_samples[noise_type][k].append(np.asarray(amp_k, dtype=np.float64))
    rank_df = pd.DataFrame(rank_rows)
    agg_df = (
        rank_df.groupby(["noise_type", "k"])
        .agg(
            sigma_ratio_mean=("sigma_ratio", "mean"),
            sigma_ratio_std=("sigma_ratio", "std"),
            bias_rel_mean=("bias_rel", "mean"),
            bias_rel_std=("bias_rel", "std"),
            bias_rel_OF_mean=("bias_rel_OF", "mean"),
            ks_pvalue_median=("ks_pvalue", "median"),
            ks_pvalue_min=("ks_pvalue", "min"),
            ks_pvalue_max=("ks_pvalue", "max"),
        )
        .reset_index()
    )
    amp_pooled = {
        nt: {k: (np.concatenate(v) if len(v) else np.array([])) for k, v in d.items()}
        for nt, d in amp_samples.items()
    }
    return {
        "rank_df": rank_df,
        "agg_df": agg_df,
        "amp_pooled": amp_pooled,
        "amp_true": amp_true_by_type,
    }


def run_block11_reversal_rank_sweep(
    cfg: CanonicalConfig,
    n_seeds: int = 8,
    n_train: int = 400,
    n_test: int = 200,
    ranks: tuple[int, ...] = (1, 2, 3),
    noise_types: tuple[str, ...] = ("white", "pink", "brownian"),
    noise_power: float = 1.0,
    empca_mode: str = "full",
) -> pd.DataFrame:
    """Table 4 / Table 7 reconciliation: isotropic PCA vs weighted EMPCA,
    rank sweep, with BOTH Delta-chi^2 definitions reported explicitly:

    - delta_chi2_vs_iso(k)  = (chi2_iso(k) - chi2_emp(k)) / chi2_iso(k)
      [Table 4's quantity at k=1]
    - delta_chi2_vs_rank1(k) = (chi2_emp(1) - chi2_emp(k)) / chi2_emp(1)
      [Table 7's quantity]
    The draft conflated the two (-0.23% vs -0.03%); they are different.

    Isotropic PCA is fitted in the time domain (uncentered SVD, the raw-MSE
    objective); weighted EMPCA in the rfft domain with DFT-convention weights."""
    rows = []
    n_events = n_train + n_test
    for noise_type in noise_types:
        for s in range(n_seeds):
            seed = cfg.seed + 1000 * s
            fam = simulate_jitter_family(
                cfg, n_events=n_events, noise_type=noise_type,
                noise_power=noise_power, seed=seed, n_null=8,
            )
            w = build_of_one_sided_weights(fam["J_dft"], cfg.trace_len)
            X_time = baseline_correct(fam["noisy"], cfg.pretrigger)
            X_f = rfft_traces(X_time)
            tmpl_f = np.fft.rfft(
                baseline_correct(fam["template_time"][None, :], cfg.pretrigger)[0]
            )
            train_idx, test_idx = _split_two_way(n_events, n_train / n_events, seed)
            # isotropic PCA, time domain, uncentered
            _, _, vh = np.linalg.svd(X_time[train_idx], full_matrices=False)
            vh = vh.copy()
            tmpl_t0 = baseline_correct(fam["template_time"][None, :], cfg.pretrigger)[0]
            if float(vh[0] @ tmpl_t0) < 0:
                vh[0] *= -1.0
            chi2_emp_k1 = None
            for k in ranks:
                iso_basis_t = vh[:k]
                coeff_iso = X_time[test_idx] @ iso_basis_t.T
                recon_iso = coeff_iso @ iso_basis_t
                mse_iso = float(np.mean((X_time[test_idx] - recon_iso) ** 2))
                iso_basis_f = rfft_traces(iso_basis_t)
                resid_iso_f = X_f[test_idx] - rfft_traces(recon_iso)
                chi2_iso = float(np.mean(np.sum((np.abs(resid_iso_f) ** 2) * w[None, :], axis=1)))

                fit = fit_weighted_empca(
                    X_f[train_idx], w, k=k,
                    n_iter=cfg.default_empca_iter, patience=cfg.default_empca_patience,
                    init="template", template_f=tmpl_f, seed=seed, mode=empca_mode,
                )
                coeff_emp = rankk_gls_coefficients(X_f[test_idx], fit["basis"], w, return_complex=True)
                resid_emp = residual_energy_per_trace(X_f[test_idx], fit["basis"], coeff_emp, w)
                chi2_emp = float(np.mean(resid_emp))
                recon_f = np.atleast_2d(coeff_emp) @ np.atleast_2d(fit["basis"])
                mse_emp = float(np.mean(raw_mse_from_freq_residual(X_f[test_idx] - recon_f, cfg.trace_len)))
                if k == min(ranks):
                    chi2_emp_k1 = chi2_emp
                cosines, angles = principal_angles_weighted(fit["basis"], iso_basis_f, w)
                # sigma_E for both methods (amplitude = template-aligned component)
                basis_amp = amplitude_basis_from_subspace(fit["basis"], tmpl_f, w)
                amp_emp = template_unit_amplitudes(X_f[test_idx], basis_amp, tmpl_f, w)
                tmpl_t = baseline_correct(fam["template_time"][None, :], cfg.pretrigger)[0]
                gamma_iso = float(iso_basis_t[0] @ tmpl_t)
                amp_iso = coeff_iso[:, 0] / gamma_iso if abs(gamma_iso) > 1e-30 else coeff_iso[:, 0]
                rows.append({
                    "noise_type": noise_type,
                    "seed": seed,
                    "k": int(k),
                    "mse_iso": mse_iso,
                    "mse_empca": mse_emp,
                    "delta_mse_iso_advantage": (mse_emp - mse_iso) / mse_iso,
                    "chi2_iso": chi2_iso,
                    "chi2_empca": chi2_emp,
                    "delta_chi2_vs_iso": (chi2_iso - chi2_emp) / chi2_iso,
                    "delta_chi2_vs_rank1": (chi2_emp_k1 - chi2_emp) / chi2_emp_k1,
                    "theta_w_first_deg": float(angles[0]),
                    "theta_w_last_deg": float(angles[-1]),
                    "sigma_E_empca": float(np.std(amp_emp, ddof=1)),
                    "sigma_E_iso_pc1": float(np.std(amp_iso, ddof=1)),
                    "bias_rel_empca": float(np.mean(amp_emp) - fam["amp_true"]) / fam["amp_true"],
                    "bias_rel_iso": float(np.mean(amp_iso) - fam["amp_true"]) / fam["amp_true"],
                    "empca_mode": empca_mode,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Block 12: G1 real-data isotropic-vs-weighted metric reversal (K-alpha)
# ----------------------------------------------------------------------

def load_real_lean(cfg: CanonicalConfig, chunk: int = 256) -> dict[str, Any]:
    """Memory-lean K-alpha loader (float32 time, complex64 rfft).

    The full-precision ``load_real_bundle`` needs ~3.4 GB; this variant stays
    under ~1.3 GB so the full 4358x32768 dataset fits in small-RAM workers.
    complex64 gives ~1e-7 relative precision: ample for MSE/chi^2/sigma_E
    (it is NOT sufficient for the 1e-8-level equivalence checks of block 03)."""
    trace_path = REPO_ROOT / cfg.real_trace_path
    rq_path = REPO_ROOT / cfg.real_rq_path
    template_path = REPO_ROOT / "data/k_alpha/template_K_alpha_tight.npy"
    if not template_path.exists():
        template_path = REPO_ROOT / cfg.template_path

    with h5py.File(rq_path, "r") as handle:
        rqs = pd.DataFrame.from_records(handle["rqs"][:])

    with h5py.File(trace_path, "r") as handle:
        dset = handle["traces"]
        n, d = dset.shape
        assert d == cfg.trace_len, (d, cfg.trace_len)
        X_time = np.empty((n, d), dtype=np.float32)
        X_freq = np.empty((n, d // 2 + 1), dtype=np.complex64)
        pre = np.empty((n, cfg.pretrigger), dtype=np.float32)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            block = np.asarray(dset[lo:hi], dtype=np.float64)
            base = np.mean(block[:, : cfg.pretrigger], axis=1, keepdims=True)
            pre[lo:hi] = (block[:, : cfg.pretrigger] - base).astype(np.float32)
            block -= base
            X_time[lo:hi] = block.astype(np.float32)
            X_freq[lo:hi] = np.fft.rfft(block, axis=1).astype(np.complex64)
    template_time = normalize_template_peak(
        np.load(template_path).astype(np.float64).reshape(-1), cfg.pretrigger
    )
    return {
        "X_time": X_time,
        "X_freq": X_freq,
        "pretrigger_traces": pre,
        "template_time": template_time,
        "template_freq": np.fft.rfft(template_time),
        "rqs": rqs,
        "n_events": int(n),
    }


def estimate_real_psd_segments(
    pretrigger_traces: np.ndarray,
    cfg: CanonicalConfig,
) -> dict[str, np.ndarray]:
    """Physical one-sided PSD from pre-trigger noise, interpolated onto the
    full-trace frequency grid (log-log), with DC handled by extension.

    Avoids the tiling artifact of ``estimate_real_psd_from_pretrigger`` (the
    tiled copy imposes spurious periodicity at fs/pretrigger harmonics)."""
    seg = np.asarray(pretrigger_traces, dtype=np.float64)
    seg = seg - np.mean(seg, axis=1, keepdims=True)
    freqs_seg, J_seg = QPSimulator.estimate_psd(seg, cfg.sampling_frequency)
    freqs_full = np.fft.rfftfreq(cfg.trace_len, d=1.0 / cfg.sampling_frequency)
    pos = freqs_seg > 0
    logJ = np.interp(
        np.log(np.maximum(freqs_full, freqs_seg[pos][0])),
        np.log(freqs_seg[pos]),
        np.log(np.maximum(J_seg[pos], 1e-300)),
    )
    J_full = np.exp(logJ)
    J_full[0] = J_full[1]
    floor = np.quantile(J_full[1:], cfg.default_psd_floor_quantile)
    J_full = np.maximum(J_full, floor)
    return {
        "freqs_seg": freqs_seg,
        "J_seg_phys": J_seg,
        "freqs_full": freqs_full,
        "J_phys": J_full,
        "J_dft": psd_physical_to_dft(J_full, cfg.trace_len, cfg.sampling_frequency),
    }


def _fit_empca_lean(
    X_f: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_iter: int,
    patience: int,
    template_f: np.ndarray,
    mode: str = "full",
    seed: int | None = None,
) -> dict[str, Any]:
    """EMPCA fit that preserves the input dtype (complex64-safe), template+SVD
    init, no smoothing. Mirrors fit_weighted_empca but without the complex128
    upcast of the full data matrix."""
    if seed is not None:
        np.random.seed(seed)
    solver = empca_solver(k, X_f, np.asarray(weights, dtype=np.float64))
    base = normalize_basis_weighted_unit(
        np.asarray(template_f, dtype=np.complex128), weights
    )[None, :]
    if k > 1:
        # SVD of the whitened data via Gram trick (lean)
        sqrt_w = safe_sqrt_weights(weights).astype(np.float32)
        Xw = X_f * sqrt_w[None, :]
        G = (Xw @ Xw.conj().T).astype(np.complex128)
        vals, vecs = np.linalg.eigh(G)
        order = np.argsort(vals)[::-1][: k]
        comps = (vecs[:, order].T.conj() @ Xw).astype(np.complex128)
        mask = sqrt_w > 0
        comps[:, mask] /= sqrt_w[mask][None, :]
        comps = normalize_basis_weighted_unit(comps, weights)
        base = np.vstack([base, comps[1:k]])
    solver.eigvec = orthonormalize(base.copy())
    solver.coeff = solver.solve_coeff()
    chi2_trace: list[float] = []
    best, stale = np.inf, 0
    for _ in range(n_iter):
        solver.eigvec = orthonormalize(solver.solve_eigvec(mode=mode))
        solver.coeff = solver.solve_coeff()
        chi2 = float(solver.chi2())
        chi2_trace.append(chi2)
        if chi2 + 1e-12 < best:
            best, stale = chi2, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return {
        "basis": np.asarray(solver.eigvec, dtype=np.complex128),
        "chi2_trace": np.asarray(chi2_trace),
        "n_iter_used": len(chi2_trace),
    }


def run_block12_real_reversal(
    cfg: CanonicalConfig,
    ranks: tuple[int, ...] = (1, 2, 3, 4, 5),
    kalpha_line_ev: float | None = 5895.0,
    empca_mode: str = "full",
    n_iter: int | None = None,
) -> dict[str, Any]:
    """G1: isotropic PCA vs Sigma-weighted EMPCA on real K-alpha pulses.

    Fits both methods on identical training pulses; evaluates on held-out
    pulses: raw time-domain MSE (isotropic's objective), weighted residual
    chi^2 (the detector likelihood), and the energy resolution sigma_E at the
    calibration line. Also returns the whitened angle theta_w between the two
    learned bases, the OF baseline, the E12 amplitude histogram inputs, and
    the real-PSD audit arrays (replacement for placeholder Fig 21).

    ``kalpha_line_ev``: line energy used to express sigma_E in eV
    (default Mn K-alpha 5895 eV — CONFIRM for this detector); relative
    resolution is always reported and is line-independent."""
    n_iter = cfg.default_empca_iter if n_iter is None else int(n_iter)
    data = load_real_lean(cfg)
    psd = estimate_real_psd_segments(data["pretrigger_traces"], cfg)
    w = build_of_one_sided_weights(psd["J_dft"], cfg.trace_len)
    split = deterministic_split_indices(data["n_events"], cfg)
    train_idx, test_idx = split["train"], split["test"]
    X_time, X_freq = data["X_time"], data["X_freq"]
    tmpl_f = data["template_freq"]

    # ---------------- OF baseline (canonical OptimumFilter, physical PSD) ---
    of = OptimumFilter(data["template_time"], psd["J_phys"], cfg.sampling_frequency)
    amp_of = np.array(
        [of.fit(np.asarray(X_time[i], dtype=np.float64))[0] for i in test_idx]
    )
    amp_of_all = np.array(
        [of.fit(np.asarray(X_time[i], dtype=np.float64))[0] for i in range(data["n_events"])]
    )
    rq_of = data["rqs"]["OF_ampl_0"].to_numpy(dtype=np.float64)
    of_cross_corr = float(np.corrcoef(amp_of_all, rq_of)[0, 1])

    # ---------------- isotropic PCA via Gram trick (time domain, uncentered)
    Xtr = X_time[train_idx]
    G = (Xtr @ Xtr.T).astype(np.float64)
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1][: max(ranks)]
    iso_basis = (vecs[:, order].T @ Xtr).astype(np.float64)
    iso_basis /= np.linalg.norm(iso_basis, axis=1, keepdims=True)
    # sign-align PC1 to template
    if float(iso_basis[0] @ data["template_time"]) < 0:
        iso_basis[0] *= -1.0

    rows = []
    fits: dict[int, dict[str, Any]] = {}
    Xtr_f = X_freq[train_idx]
    Xte_f = np.asarray(X_freq[test_idx], dtype=np.complex128)
    Xte_t = np.asarray(X_time[test_idx], dtype=np.float64)
    for k in ranks:
        fit = _fit_empca_lean(
            Xtr_f, w, k=k, n_iter=n_iter, patience=cfg.default_empca_patience,
            template_f=tmpl_f, mode=empca_mode, seed=cfg.seed,
        )
        fits[k] = fit
        # EMPCA held-out metrics
        coeff_emp = rankk_gls_coefficients(Xte_f, fit["basis"], w, return_complex=True)
        resid_emp_f = Xte_f - np.atleast_2d(coeff_emp) @ np.atleast_2d(fit["basis"])
        chi2_emp = float(np.mean(np.sum((np.abs(resid_emp_f) ** 2) * w[None, :], axis=1)))
        mse_emp = float(np.mean(raw_mse_from_freq_residual(resid_emp_f, cfg.trace_len)))
        basis_amp = amplitude_basis_from_subspace(fit["basis"], tmpl_f, w)
        amp_emp = template_unit_amplitudes(Xte_f, basis_amp, tmpl_f, w)
        # isotropic held-out metrics
        iso_k = iso_basis[:k]
        coeff_iso = Xte_t @ iso_k.T
        resid_iso_t = Xte_t - coeff_iso @ iso_k
        mse_iso = float(np.mean(resid_iso_t ** 2))
        resid_iso_f = np.fft.rfft(resid_iso_t, axis=1)
        chi2_iso = float(np.mean(np.sum((np.abs(resid_iso_f) ** 2) * w[None, :], axis=1)))
        gamma_iso = float(iso_k[0] @ data["template_time"])
        amp_iso = coeff_iso[:, 0] / gamma_iso if abs(gamma_iso) > 1e-30 else coeff_iso[:, 0]
        iso_f = np.fft.rfft(iso_k, axis=1)
        cosines, angles = principal_angles_weighted(fit["basis"], iso_f, w)

        def _res(amp):
            mu = float(np.mean(amp))
            sd = float(np.std(amp, ddof=1))
            return mu, sd, sd / abs(mu)

        mu_e, sd_e, rel_e = _res(amp_emp)
        mu_i, sd_i, rel_i = _res(amp_iso)
        mu_o, sd_o, rel_o = _res(amp_of)
        rows.append({
            "k": int(k),
            "mse_iso": mse_iso, "mse_empca": mse_emp,
            "delta_mse_iso_advantage": (mse_emp - mse_iso) / mse_iso,
            "chi2_iso": chi2_iso, "chi2_empca": chi2_emp,
            "delta_chi2_vs_iso": (chi2_iso - chi2_emp) / chi2_iso,
            "sigma_E_rel_empca": rel_e, "sigma_E_rel_iso": rel_i, "sigma_E_rel_of": rel_o,
            "sigma_E_ev_empca": rel_e * kalpha_line_ev if kalpha_line_ev else np.nan,
            "sigma_E_ev_iso": rel_i * kalpha_line_ev if kalpha_line_ev else np.nan,
            "sigma_E_ev_of": rel_o * kalpha_line_ev if kalpha_line_ev else np.nan,
            "theta_w_first_deg": float(angles[0]),
            "theta_w_last_deg": float(angles[-1]),
            "empca_iters": int(fit["n_iter_used"]),
        })
    reversal_df = pd.DataFrame(rows)

    # ---------------- E12 extras: amplitude histogram + CRB comparison -----
    nphi = float(np.real(weighted_inner(tmpl_f, tmpl_f, w)))
    sigma_crb = 1.0 / np.sqrt(nphi)
    sh_stat, sh_p = stats.shapiro(amp_of_all[: min(len(amp_of_all), 4999)])
    e12 = {
        "amp_of_all": amp_of_all,
        "rq_of_ampl": rq_of,
        "of_cross_corr": of_cross_corr,
        "sigma_A_obs": float(np.std(amp_of_all, ddof=1)),
        "sigma_A_crb": float(sigma_crb),
        "crb_ratio": float(np.std(amp_of_all, ddof=1) / sigma_crb),
        "shapiro_stat": float(sh_stat),
        "shapiro_p": float(sh_p),
    }
    return {
        "reversal_df": reversal_df,
        "psd": psd,
        "e12": e12,
        "split_sizes": {k_: int(v.shape[0]) for k_, v in split.items()},
        "kalpha_line_ev": kalpha_line_ev,
    }
