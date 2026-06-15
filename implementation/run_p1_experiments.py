"""
P1 roadmap experiments: E8, E9, G3, G4, and G5 (E10 + E11).

Atomic checkpoint pattern, matching the P0 driver.

Outputs:
    results/checkpoints/p1/*.json
    results/tables/p1_*.csv
    results/figures/p1_*.png

Usage:
    python implementation/run_p1_experiments.py
    python implementation/run_p1_experiments.py --budget 2
    python implementation/run_p1_experiments.py --figs-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))
import notebook_support as ns
from notebook_support import (
    ArtifactInjector,
    CanonicalConfig,
    MultiChannelNoiseGenerator,
    NoiseGenerator,
    QPSimulator,
    baseline_correct,
    build_of_one_sided_weights,
    exact_weighted_subspace,
    fit_weighted_empca,
    project_gls,
    rankk_gls_coefficients,
    residual_energy_per_trace,
    rfft_traces,
)

REPO = Path(__file__).parent.parent
CKPT = REPO / "results" / "checkpoints" / "p1"
FIGS = REPO / "results" / "figures"
TABLES = REPO / "results" / "tables"
for d in (CKPT, FIGS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CFG = replace(CanonicalConfig().validate(), trace_len=4096, pretrigger=600)
FS = CFG.sampling_frequency
BASE_SEED = 27182818
NOISE_POWER = 1.0


def _seed(offset: int) -> int:
    return BASE_SEED + 1000 * int(offset)


def _save(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    tmp.rename(path)


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _build_noise(noise_type: str, seed: int, power: float = NOISE_POWER):
    rng = np.random.default_rng(seed)
    ng = NoiseGenerator(
        {"noise_type": noise_type, "noise_power": float(power), "sampling_frequency": FS},
        rng=rng,
    )
    _, j_dft = ng.build_psd(CFG.trace_len)
    w = build_of_one_sided_weights(j_dft, CFG.trace_len)
    return ng, j_dft, w


def _template_and_sim(n_qp: int = 5000):
    dt_ns = 1e9 / FS
    trigger_ns = (CFG.pretrigger + 500) * dt_ns
    sim = QPSimulator(sampling_frequency=FS, trace_samples=CFG.trace_len, trigger_time=trigger_ns)
    clean = sim.generate(np.zeros(n_qp))
    amp_true = float(n_qp) * sim.qp_amplitude
    template_t = baseline_correct((clean / amp_true)[None, :], CFG.pretrigger)[0]
    template_f = np.fft.rfft(template_t)
    return sim, template_t, template_f, amp_true


def _simulate_clean_family(
    n_events: int,
    seed: int,
    n_qp_range=(5000, 5000),
    jitter_samples: float = 0.0,
    tau_range=(3e6, 3e6),
):
    rng = np.random.default_rng(seed)
    dt_ns = 1e9 / FS
    trigger_ns = (CFG.pretrigger + 500) * dt_ns
    n_lo, n_hi = int(n_qp_range[0]), int(n_qp_range[1])
    n_qp = np.full(n_events, n_lo, dtype=int) if n_lo == n_hi else rng.integers(n_lo, n_hi + 1, size=n_events)
    jitter_ns = rng.uniform(-jitter_samples, jitter_samples, size=n_events) * dt_ns
    tau = np.full(n_events, float(tau_range[0])) if tau_range[0] == tau_range[1] else rng.uniform(tau_range[0], tau_range[1], size=n_events)
    clean = np.empty((n_events, CFG.trace_len), dtype=float)
    amp_true = np.empty(n_events, dtype=float)
    base = QPSimulator(sampling_frequency=FS, trace_samples=CFG.trace_len, trigger_time=trigger_ns)
    for i in range(n_events):
        ev = QPSimulator(
            sampling_frequency=FS,
            trace_samples=CFG.trace_len,
            trigger_time=trigger_ns + float(jitter_ns[i]),
            tau_decay=float(tau[i]),
        )
        clean[i] = ev.generate(np.zeros(int(n_qp[i])))
        amp_true[i] = float(n_qp[i]) * base.qp_amplitude
    nominal = base.generate(np.zeros(5000))
    template_t = baseline_correct((nominal / (5000.0 * base.qp_amplitude))[None, :], CFG.pretrigger)[0]
    return {
        "clean": clean,
        "amp_true": amp_true,
        "t0_shift_samples": jitter_ns / dt_ns,
        "template_t": template_t,
        "template_f": np.fft.rfft(template_t),
    }


def unit_e9():
    """EM convergence: chi2 traces for k=1..3, random vs SVD init."""
    path = CKPT / "e9.json"
    if path.exists():
        return
    seed = _seed(9)
    fam = _simulate_clean_family(
        n_events=260,
        seed=seed,
        n_qp_range=(3000, 7000),
        jitter_samples=20.0,
        tau_range=(1e6, 5e6),
    )
    ng, _, w = _build_noise("pink", seed + 1)
    noise = np.stack([ng.generate_noise(CFG.trace_len) for _ in range(len(fam["clean"]))])
    x_f = rfft_traces(baseline_correct(fam["clean"] + noise, CFG.pretrigger)).astype(np.complex128)
    train = np.arange(0, 180)

    rows = []
    traces = []
    for k in (1, 2, 3):
        finals = {}
        first_chi2 = None
        for init in ("random", "svd"):
            fit = fit_weighted_empca(
                x_f[train],
                w,
                k=k,
                n_iter=60,
                patience=60,
                init=init,
                seed=seed + 31 * k + (0 if init == "random" else 1),
                mode="full",
            )
            chi2 = np.asarray(fit["chi2_trace"], dtype=float)
            if first_chi2 is None:
                first_chi2 = float(chi2[0])
            delta = np.diff(chi2)
            rel_step = np.abs(delta) / max(abs(float(chi2[0])), 1e-30)
            hits = np.where(rel_step < 1e-6)[0]
            iter_to_tol = int(hits[0] + 1) if len(hits) else int(len(chi2))
            monotone = bool(np.all(delta <= max(1e-7 * abs(float(chi2[0])), 1e-6)))
            finals[init] = float(chi2[-1])
            rows.append(
                {
                    "k": k,
                    "init": init,
                    "n_iter_used": int(fit["n_iter_used"]),
                    "iter_to_relative_tol_1e-6": iter_to_tol,
                    "monotone_nonincreasing": monotone,
                    "chi2_initial": float(chi2[0]),
                    "chi2_final": float(chi2[-1]),
                }
            )
            for it, value in enumerate(chi2):
                traces.append({"k": k, "init": init, "iteration": it, "chi2": float(value)})
        gap = abs(finals["random"] - finals["svd"]) / max(abs(first_chi2), 1e-30)
        for row in rows:
            if row["k"] == k:
                row["final_init_gap_rel_to_initial"] = gap
    _save(path, {"rows": rows, "traces": traces})


def unit_e8():
    """Time-shift OF filter bank for jittered fixed-amplitude traces."""
    path = CKPT / "e8.json"
    if path.exists():
        return
    seed = _seed(8)
    rng = np.random.default_rng(seed)
    n_events = 260
    n_qp = 5000
    sim, template_t, template_f, amp_true = _template_and_sim(n_qp)
    ng, j_dft, w = _build_noise("pink", seed + 1)
    dt_ns = 1e9 / FS
    true_shift_samples = rng.uniform(-12.0, 12.0, size=n_events)
    clean = np.empty((n_events, CFG.trace_len), dtype=float)
    for i, shift_samp in enumerate(true_shift_samples):
        ev = QPSimulator(
            sampling_frequency=FS,
            trace_samples=CFG.trace_len,
            trigger_time=sim.trigger_time + float(shift_samp) * dt_ns,
        )
        clean[i] = ev.generate(np.zeros(n_qp))
    noise = np.stack([ng.generate_noise(CFG.trace_len) for _ in range(n_events)])
    x_t = baseline_correct(clean + noise, CFG.pretrigger)
    x_f = rfft_traces(x_t).astype(np.complex128)

    grid_samples = np.arange(-14.0, 14.0001, 0.5)
    amp_grid = np.empty((n_events, len(grid_samples)), dtype=float)
    for j, shift_samp in enumerate(grid_samples):
        tmpl = baseline_correct(sim.get_template_at_shift(float(shift_samp) * dt_ns)[None, :], CFG.pretrigger)[0]
        tmpl_f = np.fft.rfft(tmpl)
        amp_grid[:, j] = project_gls(x_f, tmpl_f, w, return_complex=False)
    best = np.argmax(amp_grid, axis=1)
    best_shift = grid_samples[best]
    best_amp = amp_grid[np.arange(n_events), best]
    fixed_amp = project_gls(x_f, template_f, w, return_complex=False)

    rows = [
        {
            "method": "time_shift_OF",
            "arrival_time_rmse_samples": float(np.sqrt(np.mean((best_shift - true_shift_samples) ** 2))),
            "mean_relative_bias": float(np.mean(best_amp / amp_true - 1.0)),
            "amplitude_rmse": float(np.sqrt(np.mean((best_amp - amp_true) ** 2))),
        },
        {
            "method": "fixed_OF",
            "arrival_time_rmse_samples": np.nan,
            "mean_relative_bias": float(np.mean(fixed_amp / amp_true - 1.0)),
            "amplitude_rmse": float(np.sqrt(np.mean((fixed_amp - amp_true) ** 2))),
        },
    ]
    _save(
        path,
        {
            "rows": rows,
            "true_shift_samples": true_shift_samples.tolist(),
            "best_shift_samples": best_shift.tolist(),
            "fixed_amp": fixed_amp.tolist(),
            "shifted_amp": best_amp.tolist(),
            "amp_true": float(amp_true),
        },
    )


def _estimate_multichannel_cov(noise_f: np.ndarray, shrink: float = 1e-3) -> np.ndarray:
    """Return covariance per rfft bin, shape (F, C, C)."""
    n_noise, c, f = noise_f.shape
    cov = np.empty((f, c, c), dtype=np.complex128)
    for kk in range(f):
        y = noise_f[:, :, kk]
        ck = (y.conj().T @ y) / max(n_noise, 1)
        diag = np.mean(np.real(np.diag(ck)))
        cov[kk] = ck + (shrink * diag + 1e-9) * np.eye(c)
    return cov


def _one_sided_factor(k: int, n_freq: int) -> float:
    if k == 0:
        return 0.0
    if k == n_freq - 1:
        return 0.5
    return 2.0


def _joint_of_amplitudes(x_f: np.ndarray, s_f: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n_events, c, f = x_f.shape
    amps = np.empty(n_events, dtype=float)
    den = 0.0
    inv_cov = []
    for kk in range(f):
        alpha = _one_sided_factor(kk, f)
        inv = np.linalg.inv(cov[kk])
        inv_cov.append(inv)
        sk = s_f[:, kk]
        den += alpha * np.real(np.vdot(sk, inv @ sk))
    for i in range(n_events):
        num = 0.0 + 0.0j
        for kk, inv in enumerate(inv_cov):
            alpha = _one_sided_factor(kk, f)
            sk = s_f[:, kk]
            num += alpha * np.vdot(sk, inv @ x_f[i, :, kk])
        amps[i] = float(np.real(num / den))
    return amps


def _whiten_multichannel(x_f: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n_events, c, f = x_f.shape
    out = np.empty((n_events, c * f), dtype=np.complex128)
    for kk in range(f):
        alpha = _one_sided_factor(kk, f)
        chol = np.linalg.cholesky(cov[kk])
        y = np.linalg.solve(chol, x_f[:, :, kk].T).T * np.sqrt(alpha)
        out[:, kk * c : (kk + 1) * c] = y
    return out


def unit_g3():
    """Multichannel joint OF and rank-1 whitened subspace equivalence."""
    path = CKPT / "g3.json"
    if path.exists():
        return
    seed = _seed(3)
    rng = np.random.default_rng(seed)
    n_events = 420
    n_noise = 700
    c = 4
    gains = np.array([1.0, 0.70, 0.38, 0.12], dtype=float)
    n_qp = 5000
    _, template_t, _, amp_true = _template_and_sim(n_qp)
    signal_t = gains[:, None] * (template_t[None, :] * amp_true)
    signal_f = np.fft.rfft(gains[:, None] * template_t[None, :], axis=1)

    base_cfg = {"noise_type": "pink", "noise_power": 1.0, "sampling_frequency": FS}
    multi = MultiChannelNoiseGenerator(
        base_cfg,
        config={"mode": "shared_private", "n_channels": c, "corr_strength": 0.65, "channel_gain_jitter": 0.02},
        seed=seed,
    )
    noise = np.stack([multi.generate(CFG.trace_len, C=c) for _ in range(n_events)])
    noise_cal = np.stack([multi.generate(CFG.trace_len, C=c) for _ in range(n_noise)])
    x_t = signal_t[None, :, :] + noise
    x_t = x_t - np.mean(x_t[:, :, : CFG.pretrigger], axis=2, keepdims=True)
    noise_cal = noise_cal - np.mean(noise_cal[:, :, : CFG.pretrigger], axis=2, keepdims=True)
    x_f = np.fft.rfft(x_t, axis=2).astype(np.complex128)
    noise_f = np.fft.rfft(noise_cal, axis=2).astype(np.complex128)
    cov = _estimate_multichannel_cov(noise_f, shrink=2e-3)

    joint_amp = _joint_of_amplitudes(x_f, signal_f, cov)
    cov_diag = np.array([np.diag(np.diag(cov[k])) for k in range(cov.shape[0])])
    naive_amp = _joint_of_amplitudes(x_f, signal_f, cov_diag)

    x_white = _whiten_multichannel(x_f, cov)
    s_white = _whiten_multichannel(signal_f[None, :, :], cov)[0]
    _, _, vh = np.linalg.svd(x_white[:260], full_matrices=False)
    u = vh[0]
    if np.real(np.vdot(u, s_white)) < 0:
        u = -u
    coeff = x_white[260:] @ np.conj(u)
    gamma = np.vdot(u, s_white)
    emp_amp = np.real(coeff / gamma)
    joint_test = joint_amp[260:]
    corr = float(np.corrcoef(emp_amp, joint_test)[0, 1])

    _save(
        path,
        {
            "n_channels": c,
            "mean_offdiag_corr": float(np.mean([np.corrcoef(noise_cal[i])[np.triu_indices(c, 1)].mean() for i in range(25)])),
            "amp_true": float(amp_true),
            "sigma_joint": float(np.std(joint_amp, ddof=1)),
            "sigma_naive_diag": float(np.std(naive_amp, ddof=1)),
            "joint_improvement_frac": float(1.0 - np.std(joint_amp, ddof=1) / np.std(naive_amp, ddof=1)),
            "corr_joint_empca": corr,
            "bias_joint": float(np.mean(joint_amp / amp_true - 1.0)),
            "bias_naive_diag": float(np.mean(naive_amp / amp_true - 1.0)),
        },
    )


def _estimate_j_dft_from_noise(noise: np.ndarray) -> np.ndarray:
    nf = rfft_traces(baseline_correct(noise, CFG.pretrigger))
    j = np.mean(np.abs(nf) ** 2, axis=0)
    floor = np.quantile(j[1:], 0.01)
    j = np.maximum(j, floor)
    j[0] = j[1]
    return j


def unit_g4():
    """Finite noise-trace PSD estimation robustness."""
    path = CKPT / "g4.json"
    if path.exists():
        return
    seed = _seed(4)
    sim, template_t, template_f, amp_true = _template_and_sim(5000)
    ng_eval, j_oracle, w_oracle = _build_noise("pink", seed + 1)
    n_eval = 500
    clean = sim.generate(np.zeros(5000))
    eval_noise = np.stack([ng_eval.generate_noise(CFG.trace_len) for _ in range(n_eval)])
    x_f = rfft_traces(baseline_correct(clean[None, :] + eval_noise, CFG.pretrigger))
    amp_oracle = project_gls(x_f, template_f, w_oracle, return_complex=False)
    sigma_oracle = float(np.std(amp_oracle, ddof=1))
    rows = []
    for n_noise in (20, 50, 100, 200, 500, 1000):
        ng_cal, _, _ = _build_noise("pink", seed + 100 + n_noise)
        noise_cal = np.stack([ng_cal.generate_noise(CFG.trace_len) for _ in range(n_noise)])
        j_hat = _estimate_j_dft_from_noise(noise_cal)
        w_hat = build_of_one_sided_weights(j_hat, CFG.trace_len)
        amp = project_gls(x_f, template_f, w_hat, return_complex=False)
        sigma = float(np.std(amp, ddof=1))
        rows.append(
            {
                "n_noise": n_noise,
                "sigma": sigma,
                "sigma_oracle": sigma_oracle,
                "degradation_frac": float(sigma / sigma_oracle - 1.0),
                "bias": float(np.mean(amp / amp_true - 1.0)),
            }
        )
    _save(path, {"rows": rows, "sigma_oracle": sigma_oracle})


def _amp_rmse(x_t: np.ndarray, amp_true: np.ndarray, template_f: np.ndarray, w: np.ndarray):
    x_f = rfft_traces(baseline_correct(x_t, CFG.pretrigger))
    amp = project_gls(x_f, template_f, w, return_complex=False)
    resid = residual_energy_per_trace(x_f, template_f, amp, w)
    return amp, float(np.sqrt(np.mean((amp - amp_true) ** 2))), float(np.mean(resid))


def unit_e10():
    """Non-stationary run-period noise: global vs segment PSD."""
    path = CKPT / "e10.json"
    if path.exists():
        return
    seed = _seed(10)
    fam = _simulate_clean_family(n_events=320, seed=seed, n_qp_range=(3500, 6500), jitter_samples=0.0)
    _, _, template_f, _ = _template_and_sim(5000)
    # Change PSD shape by run segment, not just total variance.  A global PSD
    # averages incompatible noise colors; per-segment PSDs recover the correct
    # matched filter for each period.
    segment_specs = [
        ("white", 0.8),
        ("pink", 1.0),
        ("brownian", 1.2),
        ("pink", 4.0),
    ]
    seg_len = len(fam["clean"]) // len(segment_specs)
    noises = []
    rows = []
    for idx, (noise_type, power) in enumerate(segment_specs):
        sl = slice(idx * seg_len, (idx + 1) * seg_len if idx < len(segment_specs) - 1 else len(fam["clean"]))
        ng, _, _ = _build_noise(noise_type, seed + idx, power=float(power))
        noises.append(np.stack([ng.generate_noise(CFG.trace_len) for _ in range(sl.stop - sl.start)]))
    noise = np.vstack(noises)
    x_t = fam["clean"] + noise
    j_global = _estimate_j_dft_from_noise(noise)
    w_global = build_of_one_sided_weights(j_global, CFG.trace_len)
    _, rmse_global, chi2_global = _amp_rmse(x_t, fam["amp_true"], template_f, w_global)
    rows.append({"case": "nonstationary_global_psd", "amplitude_rmse": rmse_global, "chi2_mean": chi2_global})
    amp_seg = np.empty(len(fam["clean"]))
    resid_seg = np.empty(len(fam["clean"]))
    for idx, _spec in enumerate(segment_specs):
        sl = slice(idx * seg_len, (idx + 1) * seg_len if idx < len(segment_specs) - 1 else len(fam["clean"]))
        j_seg = _estimate_j_dft_from_noise(noise[sl])
        w_seg = build_of_one_sided_weights(j_seg, CFG.trace_len)
        x_f_seg = rfft_traces(baseline_correct(x_t[sl], CFG.pretrigger))
        amp_seg[sl] = project_gls(x_f_seg, template_f, w_seg, return_complex=False)
        resid_seg[sl] = residual_energy_per_trace(x_f_seg, template_f, amp_seg[sl], w_seg)
    rmse_seg = float(np.sqrt(np.mean((amp_seg - fam["amp_true"]) ** 2)))
    chi2_seg = float(np.mean(resid_seg))
    rows.append({"case": "nonstationary_segment_psd", "amplitude_rmse": rmse_seg, "chi2_mean": chi2_seg})
    ng_stat, j_stat, w_stat = _build_noise("pink", seed + 99, power=1.0)
    stat_noise = np.stack([ng_stat.generate_noise(CFG.trace_len) for _ in range(len(fam["clean"]))])
    _, rmse_stat, chi2_stat = _amp_rmse(fam["clean"] + stat_noise, fam["amp_true"], template_f, w_stat)
    rows.insert(0, {"case": "stationary_oracle_psd", "amplitude_rmse": rmse_stat, "chi2_mean": chi2_stat})
    _save(
        path,
        {
            "rows": rows,
            "rmse_improvement_segment_vs_global": float((rmse_global - rmse_seg) / rmse_global),
            "chi2_improvement_segment_vs_global": float((chi2_global - chi2_seg) / chi2_global),
        },
    )


def unit_e11():
    """Artifact contamination with residual-threshold flagging."""
    path = CKPT / "e11.json"
    if path.exists():
        return
    seed = _seed(11)
    fam = _simulate_clean_family(n_events=360, seed=seed, n_qp_range=(3500, 6500), jitter_samples=8.0)
    _, _, template_f, _ = _template_and_sim(5000)
    ng, j_dft, w = _build_noise("pink", seed + 1)
    noise = np.stack([ng.generate_noise(CFG.trace_len) for _ in range(len(fam["clean"]))])
    injector = ArtifactInjector(
        {
            "sampling_frequency": FS,
            "enable_glitches": True,
            "glitch_rate": 0.45,
            "glitch_amp_range": [0.3, 1.0],
            "glitch_duration_samples": [16, 180],
            "enable_sparse_impulses": True,
            "impulse_probability": 2e-4,
            "impulse_sigma": 0.7,
        },
        seed=seed + 2,
    )
    x_t = np.stack([injector.apply(fam["clean"][i] + noise[i]) for i in range(len(fam["clean"]))])
    x_f = rfft_traces(baseline_correct(x_t, CFG.pretrigger))
    amp = project_gls(x_f, template_f, w, return_complex=False)
    resid = residual_energy_per_trace(x_f, template_f, amp, w)
    threshold = float(np.quantile(resid, 0.80))
    keep = resid <= threshold
    rmse_all = float(np.sqrt(np.mean((amp - fam["amp_true"]) ** 2)))
    rmse_keep = float(np.sqrt(np.mean((amp[keep] - fam["amp_true"][keep]) ** 2)))
    rows = [
        {
            "pass": "contaminated_all",
            "amplitude_rmse": rmse_all,
            "weighted_residual_mean": float(np.mean(resid)),
            "kept_fraction": 1.0,
        },
        {
            "pass": "residual_flagged_keep",
            "amplitude_rmse": rmse_keep,
            "weighted_residual_mean": float(np.mean(resid[keep])),
            "kept_fraction": float(np.mean(keep)),
        },
    ]
    _save(
        path,
        {
            "rows": rows,
            "rmse_improvement_flagged": float((rmse_all - rmse_keep) / rmse_all),
            "residual_improvement_flagged": float((np.mean(resid) - np.mean(resid[keep])) / np.mean(resid)),
        },
    )


def build_queue():
    return [
        ("e9", unit_e9),
        ("e8", unit_e8),
        ("g3", unit_g3),
        ("g4", unit_g4),
        ("e10", unit_e10),
        ("e11", unit_e11),
    ]


def run(budget: int = 10000):
    q = build_queue()
    pending = [(name, fn) for name, fn in q if not (CKPT / f"{name}.json").exists()]
    print(f"[p1] {len(q) - len(pending)}/{len(q)} already done, {len(pending)} pending", flush=True)
    ran = 0
    for name, fn in pending:
        if ran >= budget:
            print(f"[p1] budget {budget} reached", flush=True)
            break
        t0 = time.time()
        print(f"  -> {name} ...", end=" ", flush=True)
        fn()
        print(f"done ({time.time() - t0:.1f}s)", flush=True)
        ran += 1
    done = sum(1 for name, _ in q if (CKPT / f"{name}.json").exists())
    print(f"[p1] {done}/{len(q)} complete", flush=True)
    if done == len(q):
        print("[p1] ALL_DONE", flush=True)


def _write_tables():
    tables = {}
    if (CKPT / "e9.json").exists():
        d = _load_json(CKPT / "e9.json")
        tables["p1_e9_convergence_summary.csv"] = pd.DataFrame(d["rows"])
        tables["p1_e9_convergence_traces.csv"] = pd.DataFrame(d["traces"])
    for key in ("e8", "g4", "e10", "e11"):
        p = CKPT / f"{key}.json"
        if p.exists():
            d = _load_json(p)
            tables[f"p1_{key}.csv"] = pd.DataFrame(d["rows"])
    if (CKPT / "g3.json").exists():
        tables["p1_g3_multichannel.csv"] = pd.DataFrame([_load_json(CKPT / "g3.json")])
    for name, df in tables.items():
        df.to_csv(TABLES / name, index=False)
    return tables


def generate_figures():
    _write_tables()
    if (CKPT / "e9.json").exists():
        d = _load_json(CKPT / "e9.json")
        df = pd.DataFrame(d["traces"])
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for (k, init), sub in df.groupby(["k", "init"]):
            y = sub["chi2"].to_numpy()
            ax.plot(sub["iteration"], y / y[0], label=f"k={k}, {init}")
        ax.set_xlabel("EM iteration")
        ax.set_ylabel("chi2 / chi2(0)")
        ax.set_title("P1 E9: EMPCA convergence")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGS / "p1_e9_convergence.png", dpi=150)
        plt.close(fig)
    if (CKPT / "e8.json").exists():
        d = _load_json(CKPT / "e8.json")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].scatter(d["true_shift_samples"], d["best_shift_samples"], s=14, alpha=0.7)
        lim = [-13, 13]
        axes[0].plot(lim, lim, "k--", lw=1)
        axes[0].set_xlabel("true shift [samples]")
        axes[0].set_ylabel("estimated shift [samples]")
        axes[0].set_title("E8 timing")
        axes[1].hist(np.asarray(d["shifted_amp"]) / d["amp_true"] - 1, bins=35, alpha=0.7, label="shifted")
        axes[1].hist(np.asarray(d["fixed_amp"]) / d["amp_true"] - 1, bins=35, alpha=0.5, label="fixed")
        axes[1].set_xlabel("relative amplitude error")
        axes[1].set_title("E8 amplitude bias")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIGS / "p1_e8_time_shift_of.png", dpi=150)
        plt.close(fig)
    if (CKPT / "g3.json").exists():
        d = _load_json(CKPT / "g3.json")
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.bar(["joint full cov", "diag/naive"], [d["sigma_joint"], d["sigma_naive_diag"]], color=["tab:green", "tab:gray"])
        ax.set_ylabel("sigma amplitude")
        ax.set_title(f"G3 multichannel (corr OF/EMPCA = {d['corr_joint_empca']:.5f})")
        fig.tight_layout()
        fig.savefig(FIGS / "p1_g3_multichannel.png", dpi=150)
        plt.close(fig)
    if (CKPT / "g4.json").exists():
        df = pd.DataFrame(_load_json(CKPT / "g4.json")["rows"])
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.semilogx(df["n_noise"], df["degradation_frac"] * 100, "o-")
        ax.axhline(5, color="k", ls="--", lw=1)
        ax.set_xlabel("noise traces for PSD estimate")
        ax.set_ylabel("sigma degradation vs oracle [%]")
        ax.set_title("G4 covariance/PSD estimation robustness")
        fig.tight_layout()
        fig.savefig(FIGS / "p1_g4_covariance_robustness.png", dpi=150)
        plt.close(fig)
    if (CKPT / "e10.json").exists():
        df = pd.DataFrame(_load_json(CKPT / "e10.json")["rows"])
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(df["case"], df["amplitude_rmse"], color=["tab:blue", "tab:red", "tab:green"])
        ax.set_ylabel("amplitude RMSE")
        ax.set_title("E10 non-stationary noise")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIGS / "p1_e10_nonstationary.png", dpi=150)
        plt.close(fig)
    if (CKPT / "e11.json").exists():
        df = pd.DataFrame(_load_json(CKPT / "e11.json")["rows"])
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        ax.bar(df["pass"], df["amplitude_rmse"], color=["tab:red", "tab:green"])
        ax.set_ylabel("amplitude RMSE")
        ax.set_title("E11 artifact flagging")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(FIGS / "p1_e11_artifacts.png", dpi=150)
        plt.close(fig)


def verify():
    print("\n" + "=" * 65)
    print("P1 VERIFICATION REPORT")
    print("=" * 65)
    ok_all = True
    if (CKPT / "e9.json").exists():
        df = pd.DataFrame(_load_json(CKPT / "e9.json")["rows"])
        ok_mono = bool(df["monotone_nonincreasing"].all())
        ok_iter = bool((df["iter_to_relative_tol_1e-6"] <= np.where(df["k"] == 1, 25, 60)).all())
        ok_gap = bool((df["final_init_gap_rel_to_initial"] < 1e-5).all())
        ok = ok_mono and ok_iter and ok_gap
        ok_all &= ok
        print(f"  {'OK' if ok else 'FAIL'} E9 convergence: monotone={ok_mono}, iter={ok_iter}, init_gap={ok_gap}")
    if (CKPT / "e8.json").exists():
        df = pd.DataFrame(_load_json(CKPT / "e8.json")["rows"])
        ts = df[df["method"] == "time_shift_OF"].iloc[0]
        fixed = df[df["method"] == "fixed_OF"].iloc[0]
        ok = ts["arrival_time_rmse_samples"] < 2.0 and abs(ts["mean_relative_bias"]) < abs(fixed["mean_relative_bias"])
        ok_all &= ok
        print(
            f"  {'OK' if ok else 'FAIL'} E8 time-shift OF: "
            f"RMSE={ts['arrival_time_rmse_samples']:.2f} samples, "
            f"bias shifted={ts['mean_relative_bias']:+.2%}, fixed={fixed['mean_relative_bias']:+.2%}"
        )
    if (CKPT / "g3.json").exists():
        d = _load_json(CKPT / "g3.json")
        ok = d["sigma_joint"] < d["sigma_naive_diag"] and d["corr_joint_empca"] > 0.999
        ok_all &= ok
        print(
            f"  {'OK' if ok else 'FAIL'} G3 multichannel: "
            f"joint improvement={d['joint_improvement_frac']:.2%}, corr={d['corr_joint_empca']:.6f}"
        )
    if (CKPT / "g4.json").exists():
        df = pd.DataFrame(_load_json(CKPT / "g4.json")["rows"])
        last = df.sort_values("n_noise").iloc[-1]
        ok = last["degradation_frac"] < 0.05
        ok_all &= ok
        print(f"  {'OK' if ok else 'FAIL'} G4 PSD robustness: N={int(last['n_noise'])}, degradation={last['degradation_frac']:.2%}")
    if (CKPT / "e10.json").exists():
        d = _load_json(CKPT / "e10.json")
        ok = d["rmse_improvement_segment_vs_global"] > 0.05
        ok_all &= ok
        print(f"  {'OK' if ok else 'FAIL'} E10 nonstationary: segment RMSE improvement={d['rmse_improvement_segment_vs_global']:.2%}")
    if (CKPT / "e11.json").exists():
        d = _load_json(CKPT / "e11.json")
        ok = d["rmse_improvement_flagged"] > 0.10
        ok_all &= ok
        print(f"  {'OK' if ok else 'FAIL'} E11 artifacts: flagged RMSE improvement={d['rmse_improvement_flagged']:.2%}")
    if ok_all:
        print("[p1] ALL_VERIFIED")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()
    if not args.figs_only:
        run(args.budget)
    generate_figures()
    verify()
