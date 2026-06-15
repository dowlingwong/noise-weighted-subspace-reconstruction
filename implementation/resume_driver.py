"""Budget-driven, checkpointed driver for blocks 11-12.

The Cowork sandbox kills shell calls after ~45 s and background processes do
not survive between calls, so the full-scale studies are decomposed into
small resumable units. Each invocation:

    python implementation/resume_driver.py --budget 34

processes pending units until the time budget is exhausted, checkpointing
everything under results/checkpoints/b1112/. Re-invoke until it prints
ALL_DONE. State lives in unit-granular artifact files (atomic writes); a
unit is "done" when its artifact exists, so a killed call merely repeats one
unit. On an unconstrained machine, `--budget 100000` runs everything in one go.

Scales (full):
  sims  : trace_len 16384 (QPSimulator native, as in plan/experiment_checklist.md)
  rank  : 3 colors x 8 seeds x ranks 1..5, 240 events/seed, EMPCA mode='full'
  rev   : 3 colors x 8 seeds x ranks 1..3, 400 train + 200 test
  real  : all 4358 K-alpha traces, weighted subspace via exact whitened SVD
          (= EMPCA fixed point, Bridge Thm), EMPCA mode='full' cross-check k=1,2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import notebook_support as ns

CKPT = ns.REPO_ROOT / "results" / "checkpoints" / "b1112"
CKPT.mkdir(parents=True, exist_ok=True)

SIM_CFG = replace(ns.CanonicalConfig(), trace_len=16384, pretrigger=2000).validate()
REAL_CFG = ns.CanonicalConfig().validate()

NOISE_TYPES = ("white", "pink", "brownian")
N_SEEDS = 8
RANKS_RANK = (1, 2, 3, 4, 5)
RANKS_REV = (1, 2, 3)
RANKS_REAL = (1, 2, 3, 4, 5)
SCALING_POWERS = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0)
N_EVENTS_RANK = 240
N_TRAIN_REV, N_TEST_REV = 400, 200
EMPCA_ITER, EMPCA_PATIENCE = 40, 8
REAL_CHUNK = 256

# Simulation regimes for the Exp D/E repair. The draft does not document its
# SNR regime (App. G has no event-level parameters), so we run three explicit
# configs and report all of them:
#   B "highsnr"  — noise_power=1, jitter ±200 us: jitter scatter >> noise;
#                  shows where the rank-2 linearization fails (boundary).
#   A "noisedom" — noise_power calibrated per color to SNR_A = A/sigma_CRB ~ 3
#                  with jitter ±200 us: the regime consistent with the draft's
#                  Table 6 (sigma_E/sigma_CRB = 1.00 at k=1).
#   C "linjit"   — noise_power=1, jitter ±20 us (< tau_rise): the linear
#                  regime where the rank-2 derivative-mode story applies.
#   P "paperreg" — trace_len 2048 (d = 1025 rfft bins: the draft's Table 6
#                  chi^2 ~ 1029 ~ d implies THIS trace length), SNR_A = 4,
#                  jitter ±200 us: quantitatively the draft's regime
#                  (noise-dominated, subspace well above the BBP detection
#                  threshold sqrt(d/n) ~ 1.6).
SIM_CFG_P = replace(ns.CanonicalConfig(), trace_len=2048, pretrigger=256).validate()
SIM_CONFIGS = {
    "B": {"tag": "", "jitter_ns": 2e5, "power": "fixed1", "cfg": None},
    "A": {"tag": "N", "jitter_ns": 2e5, "power": "snr3", "cfg": None},
    "C": {"tag": "S", "jitter_ns": 2e4, "power": "fixed1", "cfg": None},
    "P": {"tag": "P", "jitter_ns": 2e5, "power": "snr4", "cfg": SIM_CFG_P},
    # Z "zerojit" — same regime as P but jitter = 0: the rank-1 model is
    # exact, so sigma_E(k=1)/sigma_OF = 1 is the equivalence-theorem check.
    "Z": {"tag": "Z", "jitter_ns": 0.0, "power": "snr4", "cfg": SIM_CFG_P},
}


def _cfg_of(cfg_key: str):
    return SIM_CONFIGS[cfg_key]["cfg"] or SIM_CFG


def _noise_power_for_snr(noise_type: str, target_snr: float, cfg) -> float:
    """noise_power such that A_true / sigma_CRB = target_snr (template units)."""
    cache = CKPT / f"power_{noise_type}_{cfg.trace_len}_{target_snr:g}.json"
    if cache.exists():
        return float(json.load(open(cache))["power"])
    dt_ns = 1e9 / cfg.sampling_frequency
    sim = ns.QPSimulator(
        sampling_frequency=cfg.sampling_frequency,
        trace_samples=cfg.trace_len,
        trigger_time=(cfg.pretrigger + 500) * dt_ns,
    )
    clean = sim.generate(np.zeros(5000, dtype=float))
    amp_true = 5000.0 * sim.qp_amplitude
    tmpl_f = np.fft.rfft(ns.baseline_correct((clean / amp_true)[None, :], cfg.pretrigger)[0])
    ng = ns.NoiseGenerator({"noise_type": noise_type, "noise_power": 1.0,
                            "sampling_frequency": cfg.sampling_frequency}, seed=0)
    _, J1 = ng.build_psd(cfg.trace_len)
    w1 = ns.build_of_one_sided_weights(J1, cfg.trace_len)
    sigma1 = 1.0 / np.sqrt(float(np.real(ns.weighted_inner(tmpl_f, tmpl_f, w1))))
    power = float((amp_true / (target_snr * sigma1)) ** 2)
    _atomic_save_json(cache, {"power": power, "sigma_crb_at_1": sigma1, "amp_true": amp_true})
    return power


def _cfg_power(cfg_key: str, noise_type: str) -> float:
    mode = SIM_CONFIGS[cfg_key]["power"]
    if mode == "fixed1":
        return 1.0
    target = float(mode.replace("snr", ""))
    # legacy cache name for the first config A run
    if cfg_key == "A":
        legacy = CKPT / f"power_{noise_type}.json"
        if legacy.exists():
            return float(json.load(open(legacy))["power"])
    return _noise_power_for_snr(noise_type, target, _cfg_of(cfg_key))


def _atomic_save_npz(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def _atomic_save_json(path: Path, payload):
    tmp = path.with_suffix(".tmp.json")
    with tmp.open("w") as fh:
        json.dump(payload, fh, default=float)
    tmp.rename(path)


def _seed(s: int) -> int:
    return SIM_CFG.seed + 1000 * s


# ----------------------------------------------------------------- sim units

def unit_crb(noise_type: str):
    df = ns.run_block11_crb_units(SIM_CFG, n_replicates=8000, noise_types=(noise_type,))
    _atomic_save_json(CKPT / f"crb_{noise_type}.json", df.to_dict(orient="records"))


def unit_scaling(idx: int):
    power = SCALING_POWERS[idx]
    df = ns.run_block11_sigma_scaling(SIM_CFG, noise_powers=(power,), n_events=400)
    _atomic_save_json(CKPT / f"scal_{idx}.json", df.to_dict(orient="records"))


def _family_cache(kind: str, noise_type: str, s: int, n_events: int, n_null: int,
                  cfg_key: str = "B") -> Path:
    tag = SIM_CONFIGS[cfg_key]["tag"]
    cfg = _cfg_of(cfg_key)
    path = CKPT / f"fam_{kind}{tag}_{noise_type}_{s}.npz"
    if path.exists():
        return path
    fam = ns.simulate_jitter_family(
        cfg, n_events=n_events, noise_type=noise_type,
        noise_power=_cfg_power(cfg_key, noise_type),
        jitter_ns=SIM_CONFIGS[cfg_key]["jitter_ns"],
        seed=_seed(s), n_null=n_null,
    )
    X_t = ns.baseline_correct(fam["noisy"], cfg.pretrigger)
    X_f = ns.rfft_traces(X_t).astype(np.complex64)
    null_f = ns.rfft_traces(
        ns.baseline_correct(fam["null_noise"], cfg.pretrigger)
    ).astype(np.complex64)
    tmpl_t = ns.baseline_correct(fam["template_time"][None, :], cfg.pretrigger)[0]
    _atomic_save_npz(
        path,
        X_f=X_f, null_f=null_f, X_t=X_t.astype(np.float32),
        tmpl_f=np.fft.rfft(tmpl_t), tmpl_t=tmpl_t,
        w=ns.build_of_one_sided_weights(fam["J_dft"], cfg.trace_len),
        amp_true=np.float64(fam["amp_true"]),
    )
    return path


def unit_rankfam(noise_type: str, s: int, cfg_key: str = "B"):
    _family_cache("rank", noise_type, s, N_EVENTS_RANK, N_EVENTS_RANK, cfg_key)


def unit_rank(noise_type: str, s: int, k: int, cfg_key: str = "B"):
    tag = SIM_CONFIGS[cfg_key]["tag"]
    fam = np.load(_family_cache("rank", noise_type, s, N_EVENTS_RANK, N_EVENTS_RANK, cfg_key))
    X_f = fam["X_f"].astype(np.complex128)
    null_f = fam["null_f"].astype(np.complex128)
    tmpl_f, w, amp_true = fam["tmpl_f"], fam["w"], float(fam["amp_true"])
    train_idx, test_idx = ns._split_two_way(X_f.shape[0], 0.6, _seed(s))
    amp_of = ns.project_gls(X_f[test_idx], tmpl_f, w, return_complex=False)
    fit = ns.fit_weighted_empca(
        X_f[train_idx], w, k=k, n_iter=EMPCA_ITER, patience=EMPCA_PATIENCE,
        init="template", template_f=tmpl_f, seed=_seed(s), mode="full",
    )
    basis_amp = ns.amplitude_basis_from_subspace(fit["basis"], tmpl_f, w)
    amp_k = ns.template_unit_amplitudes(X_f[test_idx], basis_amp, tmpl_f, w)
    resid_test = ns.residual_energy_per_trace(
        X_f[test_idx], basis_amp,
        ns.rankk_gls_coefficients(X_f[test_idx], basis_amp, w, return_complex=True), w)
    resid_null = ns.residual_energy_per_trace(
        null_f, basis_amp,
        ns.rankk_gls_coefficients(null_f, basis_amp, w, return_complex=True), w)
    from scipy import stats
    ks = stats.ks_2samp(resid_test, resid_null)
    row = {
        "noise_type": noise_type, "seed": _seed(s), "k": int(k), "config": cfg_key,
        "jitter_ns": SIM_CONFIGS[cfg_key]["jitter_ns"],
        "sigma_E": float(np.std(amp_k, ddof=1)),
        "sigma_OF": float(np.std(amp_of, ddof=1)),
        "sigma_ratio": float(np.std(amp_k, ddof=1) / np.std(amp_of, ddof=1)),
        "bias_rel": float(np.mean(amp_k) - amp_true) / amp_true,
        "bias_rel_OF": float(np.mean(amp_of) - amp_true) / amp_true,
        "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
        "resid_test_mean": float(np.mean(resid_test)),
        "resid_null_mean": float(np.mean(resid_null)),
        "empca_iters": int(fit["n_iter_used"]), "empca_mode": "full",
    }
    _atomic_save_json(CKPT / f"rank{tag}_{noise_type}_{s}_{k}.json", row)
    if k in (1, 2):
        np.save(CKPT / f"amp{tag}_{noise_type}_{s}_{k}.npy", amp_k)


def unit_revfam(noise_type: str, s: int, cfg_key: str = "B"):
    _family_cache("rev", noise_type, s, N_TRAIN_REV + N_TEST_REV, 8, cfg_key)


def unit_rev(noise_type: str, s: int, k: int, cfg_key: str = "B"):
    tag = SIM_CONFIGS[cfg_key]["tag"]
    fam = np.load(_family_cache("rev", noise_type, s, N_TRAIN_REV + N_TEST_REV, 8, cfg_key))
    X_f = fam["X_f"].astype(np.complex128)
    X_t = fam["X_t"].astype(np.float64)
    tmpl_f, tmpl_t, w, amp_true = fam["tmpl_f"], fam["tmpl_t"], fam["w"], float(fam["amp_true"])
    n = X_f.shape[0]
    train_idx, test_idx = ns._split_two_way(n, N_TRAIN_REV / n, _seed(s))
    # isotropic PCA (time domain, uncentered) via Gram trick
    Xtr = X_t[train_idx]
    G = Xtr @ Xtr.T
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1][:k]
    iso = vecs[:, order].T @ Xtr
    iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    if float(iso[0] @ tmpl_t) < 0:
        iso[0] *= -1.0
    coeff_iso = X_t[test_idx] @ iso.T
    recon_iso = coeff_iso @ iso
    mse_iso = float(np.mean((X_t[test_idx] - recon_iso) ** 2))
    resid_iso_f = X_f[test_idx] - ns.rfft_traces(recon_iso)
    chi2_iso = float(np.mean(np.sum((np.abs(resid_iso_f) ** 2) * w[None, :], axis=1)))
    gamma_iso = float(iso[0] @ tmpl_t)
    amp_iso = coeff_iso[:, 0] / gamma_iso

    fit = ns.fit_weighted_empca(
        X_f[train_idx], w, k=k, n_iter=EMPCA_ITER, patience=EMPCA_PATIENCE,
        init="template", template_f=tmpl_f, seed=_seed(s), mode="full",
    )
    coeff_emp = ns.rankk_gls_coefficients(X_f[test_idx], fit["basis"], w, return_complex=True)
    resid_emp = ns.residual_energy_per_trace(X_f[test_idx], fit["basis"], coeff_emp, w)
    chi2_emp = float(np.mean(resid_emp))
    recon_f = np.atleast_2d(coeff_emp) @ np.atleast_2d(fit["basis"])
    mse_emp = float(np.mean(ns.raw_mse_from_freq_residual(X_f[test_idx] - recon_f, _cfg_of(cfg_key).trace_len)))
    cosines, angles = ns.principal_angles_weighted(fit["basis"], ns.rfft_traces(iso), w)
    basis_amp = ns.amplitude_basis_from_subspace(fit["basis"], tmpl_f, w)
    amp_emp = ns.template_unit_amplitudes(X_f[test_idx], basis_amp, tmpl_f, w)
    row = {
        "noise_type": noise_type, "seed": _seed(s), "k": int(k), "config": cfg_key,
        "jitter_ns": SIM_CONFIGS[cfg_key]["jitter_ns"],
        "mse_iso": mse_iso, "mse_empca": mse_emp,
        "delta_mse_iso_advantage": (mse_emp - mse_iso) / mse_iso,
        "chi2_iso": chi2_iso, "chi2_empca": chi2_emp,
        "delta_chi2_vs_iso": (chi2_iso - chi2_emp) / chi2_iso,
        "theta_w_first_deg": float(angles[0]), "theta_w_last_deg": float(angles[-1]),
        "sigma_E_empca": float(np.std(amp_emp, ddof=1)),
        "sigma_E_iso_pc1": float(np.std(amp_iso, ddof=1)),
        "bias_rel_empca": float(np.mean(amp_emp) - amp_true) / amp_true,
        "bias_rel_iso": float(np.mean(amp_iso) - amp_true) / amp_true,
        "empca_mode": "full",
    }
    _atomic_save_json(CKPT / f"rev{tag}_{noise_type}_{s}_{k}.json", row)


def unit_simfinal():
    rank_rows = [json.load(open(CKPT / f"rank{SIM_CONFIGS[c]['tag']}_{nt}_{s}_{k}.json"))
                 for c in SIM_CONFIGS for nt in NOISE_TYPES
                 for s in range(N_SEEDS) for k in RANKS_RANK]  # includes Z
    rev_rows = [json.load(open(CKPT / f"rev{SIM_CONFIGS[c]['tag']}_{nt}_{s}_{k}.json"))
                for c in ("B", "A", "P") for nt in NOISE_TYPES
                for s in range(N_SEEDS) for k in RANKS_REV]
    crb_rows = sum((json.load(open(CKPT / f"crb_{nt}.json")) for nt in NOISE_TYPES), [])
    scal_rows = sum((json.load(open(CKPT / f"scal_{i}.json")) for i in range(len(SCALING_POWERS))), [])
    dirs = ns.ensure_results_dirs(SIM_CFG)
    pd.DataFrame(crb_rows).to_csv(dirs["tables"] / "block_11_e4_crb_units.csv", index=False)
    pd.DataFrame(scal_rows).to_csv(dirs["tables"] / "block_11_e5_sigma_scaling.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(dirs["tables"] / "block_11_expE_rank_seeds.csv", index=False)
    pd.DataFrame(rev_rows).to_csv(dirs["tables"] / "block_11_table47_reconciliation_seeds.csv", index=False)
    # pooled amplitude samples for Fig 15
    pooled = {}
    for c in SIM_CONFIGS:
        tag = SIM_CONFIGS[c]["tag"]
        for nt in NOISE_TYPES:
            for k in (1, 2):
                arrs = [np.load(CKPT / f"amp{tag}_{nt}_{s}_{k}.npy") for s in range(N_SEEDS)]
                pooled[f"{c}_{nt}_{k}"] = np.concatenate(arrs)
    np.savez(CKPT / "amp_pooled.npz", **pooled)
    _atomic_save_json(CKPT / "simfinal.json", {"done": True})


# ---------------------------------------------------------------- real units

def _real_meta():
    n = 4358
    split = ns.deterministic_split_indices(n, REAL_CFG)
    return n, split


def unit_real_load(chunk_idx: int):
    """Read one h5 chunk; write rows into on-disk memmaps; accumulate PSD."""
    import h5py
    n, split = _real_meta()
    d = REAL_CFG.trace_len
    nb = d // 2 + 1
    xt = np.lib.format.open_memmap(CKPT / "real_Xtime.npy", mode="r+" if (CKPT / "real_Xtime.npy").exists() else "w+", dtype=np.float32, shape=(n, d))
    xf = np.lib.format.open_memmap(CKPT / "real_Xfreq.npy", mode="r+" if (CKPT / "real_Xfreq.npy").exists() else "w+", dtype=np.complex64, shape=(n, nb))
    lo = chunk_idx * REAL_CHUNK
    hi = min(lo + REAL_CHUNK, n)
    with h5py.File(ns.REPO_ROOT / REAL_CFG.real_trace_path, "r") as fh:
        block = np.asarray(fh["traces"][lo:hi], dtype=np.float64)
    base = np.mean(block[:, : REAL_CFG.pretrigger], axis=1, keepdims=True)
    block -= base
    xt[lo:hi] = block.astype(np.float32)
    xf[lo:hi] = np.fft.rfft(block, axis=1).astype(np.complex64)
    pre = block[:, : REAL_CFG.pretrigger]
    pre = pre - np.mean(pre, axis=1, keepdims=True)
    spec_sum = np.sum(np.abs(np.fft.rfft(pre, axis=1)) ** 2, axis=0)
    _atomic_save_npz(CKPT / f"real_psdpart_{chunk_idx}.npz", spec_sum=spec_sum, count=np.int64(hi - lo))
    xt.flush(); xf.flush()
    _atomic_save_json(CKPT / f"real_load_{chunk_idx}.json", {"lo": lo, "hi": hi})


def unit_real_psd():
    n, _ = _real_meta()
    n_chunks = (n + REAL_CHUNK - 1) // REAL_CHUNK
    total = None
    count = 0
    for i in range(n_chunks):
        z = np.load(CKPT / f"real_psdpart_{i}.npz")
        total = z["spec_sum"] if total is None else total + z["spec_sum"]
        count += int(z["count"])
    npre = REAL_CFG.pretrigger
    J_seg = total / count * 2.0 / (npre * REAL_CFG.sampling_frequency)
    # DC / Nyquist are not doubled in the one-sided convention
    J_seg[0] /= 2.0
    if npre % 2 == 0:
        J_seg[-1] /= 2.0
    freqs_seg = np.fft.rfftfreq(npre, d=1.0 / REAL_CFG.sampling_frequency)
    freqs_full = np.fft.rfftfreq(REAL_CFG.trace_len, d=1.0 / REAL_CFG.sampling_frequency)
    pos = freqs_seg > 0
    logJ = np.interp(np.log(np.maximum(freqs_full, freqs_seg[pos][0])),
                     np.log(freqs_seg[pos]), np.log(np.maximum(J_seg[pos], 1e-300)))
    J_phys = np.exp(logJ)
    J_phys[0] = J_phys[1]
    floor = np.quantile(J_phys[1:], REAL_CFG.default_psd_floor_quantile)
    J_phys = np.maximum(J_phys, floor)
    J_dft = ns.psd_physical_to_dft(J_phys, REAL_CFG.trace_len, REAL_CFG.sampling_frequency)
    _atomic_save_npz(CKPT / "real_psd.npz", freqs_seg=freqs_seg, J_seg_phys=J_seg,
                     freqs_full=freqs_full, J_phys=J_phys, J_dft=J_dft,
                     w=ns.build_of_one_sided_weights(J_dft, REAL_CFG.trace_len))


def unit_real_of(part: int, n_parts: int = 3):
    n, _ = _real_meta()
    psd = np.load(CKPT / "real_psd.npz")
    template_path = ns.REPO_ROOT / "data/k_alpha/template_K_alpha_tight.npy"
    tmpl = ns.normalize_template_peak(np.load(template_path).astype(np.float64).reshape(-1), REAL_CFG.pretrigger)
    of = ns.OptimumFilter(tmpl, psd["J_phys"], REAL_CFG.sampling_frequency)
    xt = np.load(CKPT / "real_Xtime.npy", mmap_mode="r")
    lo = part * ((n + n_parts - 1) // n_parts)
    hi = min(lo + (n + n_parts - 1) // n_parts, n)
    amps = np.array([of.fit(np.asarray(xt[i], dtype=np.float64))[0] for i in range(lo, hi)])
    np.save(CKPT / f"real_ofamp_{part}.npy", amps)
    _atomic_save_json(CKPT / f"real_of_{part}.json", {"lo": lo, "hi": hi})


def unit_real_gram(part: int, n_parts: int = 4):
    """Accumulate the train-set time-domain Gram matrix in row blocks."""
    n, split = _real_meta()
    tr = split["train"]
    xt = np.load(CKPT / "real_Xtime.npy", mmap_mode="r")
    Xtr = np.asarray(xt[tr], dtype=np.float32)
    m = len(tr)
    step = (m + n_parts - 1) // n_parts
    lo, hi = part * step, min((part + 1) * step, m)
    Gpart = Xtr[lo:hi] @ Xtr.T
    np.save(CKPT / f"real_gram_{part}.npy", Gpart.astype(np.float64))
    _atomic_save_json(CKPT / f"real_gramjson_{part}.json", {"lo": lo, "hi": hi})


def unit_real_isobasis(n_parts: int = 4):
    n, split = _real_meta()
    tr = split["train"]
    G = np.vstack([np.load(CKPT / f"real_gram_{p}.npy") for p in range(n_parts)])
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1][: max(RANKS_REAL)]
    xt = np.load(CKPT / "real_Xtime.npy", mmap_mode="r")
    Xtr = np.asarray(xt[tr], dtype=np.float32)
    iso = (vecs[:, order].T.astype(np.float32) @ Xtr).astype(np.float64)
    iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    template_path = ns.REPO_ROOT / "data/k_alpha/template_K_alpha_tight.npy"
    tmpl = ns.normalize_template_peak(np.load(template_path).astype(np.float64).reshape(-1), REAL_CFG.pretrigger)
    if float(iso[0] @ tmpl) < 0:
        iso[0] *= -1.0
    _atomic_save_npz(CKPT / "real_isobasis.npz", iso=iso, tmpl_t=tmpl, tmpl_f=np.fft.rfft(tmpl))


def unit_real_wsvd(part: int, n_parts: int = 6):
    """Whitened complex Gram (train) in row blocks for the exact weighted subspace."""
    n, split = _real_meta()
    tr = split["train"]
    psd = np.load(CKPT / "real_psd.npz")
    sw = ns.safe_sqrt_weights(psd["w"]).astype(np.float32)
    xf = np.load(CKPT / "real_Xfreq.npy", mmap_mode="r")
    Xw = np.asarray(xf[tr], dtype=np.complex64) * sw[None, :]
    m = len(tr)
    step = (m + n_parts - 1) // n_parts
    lo, hi = part * step, min((part + 1) * step, m)
    Gpart = Xw[lo:hi] @ Xw.conj().T
    np.save(CKPT / f"real_wgram_{part}.npy", Gpart.astype(np.complex128))
    _atomic_save_json(CKPT / f"real_wgramjson_{part}.json", {"lo": lo, "hi": hi})


def unit_real_wbasis(n_parts: int = 6):
    n, split = _real_meta()
    tr = split["train"]
    G = np.vstack([np.load(CKPT / f"real_wgram_{p}.npy") for p in range(n_parts)])
    G = 0.5 * (G + G.conj().T)
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1][: max(RANKS_REAL)]
    psd = np.load(CKPT / "real_psd.npz")
    w = psd["w"]
    sw = ns.safe_sqrt_weights(w).astype(np.float32)
    xf = np.load(CKPT / "real_Xfreq.npy", mmap_mode="r")
    Xw = np.asarray(xf[tr], dtype=np.complex64) * sw[None, :]
    Uw = (vecs[:, order].T.astype(np.complex64) @ Xw).astype(np.complex128)
    basis = ns.normalize_basis_weighted_unit(ns.unwhiten_basis(Uw, w), w)
    _atomic_save_npz(CKPT / "real_wbasis.npz", basis=basis, eigvals=vals[np.argsort(vals)[::-1][:20]])


def unit_real_empcheck(k: int, max_iter: int = 30):
    """EMPCA mode='full' on the real train set, iteration-block checkpointed;
    cross-check that EM converges to the exact whitened-SVD subspace."""
    state_path = CKPT / f"real_empca_state_{k}.npz"
    psd = np.load(CKPT / "real_psd.npz")
    w = psd["w"]
    n, split = _real_meta()
    tr = split["train"]
    xf = np.load(CKPT / "real_Xfreq.npy", mmap_mode="r")
    X = np.asarray(xf[tr], dtype=np.complex64)
    iso = np.load(CKPT / "real_isobasis.npz")
    tmpl_f = iso["tmpl_f"]
    solver = ns.empca_solver(k, X, np.asarray(w, dtype=np.float64))
    if state_path.exists():
        st = np.load(state_path)
        solver.eigvec = st["eigvec"]
        chi2_trace = list(st["chi2_trace"])
    else:
        wb = np.load(CKPT / "real_wbasis.npz")["basis"][:k]
        base = ns.normalize_basis_weighted_unit(np.asarray(tmpl_f, dtype=np.complex128), w)[None, :]
        if k > 1:
            base = np.vstack([base, wb[1:k]])
        solver.eigvec = ns.orthonormalize(base.copy())
        chi2_trace = []
    solver.coeff = solver.solve_coeff()
    t0 = time.time()
    while len(chi2_trace) < max_iter and time.time() - t0 < 25:
        solver.eigvec = ns.orthonormalize(solver.solve_eigvec(mode="full"))
        solver.coeff = solver.solve_coeff()
        chi2_trace.append(float(solver.chi2()))
    _atomic_save_npz(state_path, eigvec=np.asarray(solver.eigvec, dtype=np.complex128),
                     chi2_trace=np.asarray(chi2_trace))
    if len(chi2_trace) >= max_iter:
        wb = np.load(CKPT / "real_wbasis.npz")["basis"][:k]
        cosines, angles = ns.principal_angles_weighted(
            np.asarray(solver.eigvec, dtype=np.complex128), wb, w)
        _atomic_save_json(CKPT / f"real_empcheck_{k}.json", {
            "k": k, "iters": len(chi2_trace),
            "chi2_first": chi2_trace[0], "chi2_last": chi2_trace[-1],
            "principal_cosines_vs_wsvd": [float(c) for c in cosines],
            "max_angle_deg_vs_wsvd": float(angles.max()),
        })


def unit_real_eval(k: int):
    n, split = _real_meta()
    te = split["test"]
    psd = np.load(CKPT / "real_psd.npz")
    w = psd["w"]
    iso_z = np.load(CKPT / "real_isobasis.npz")
    iso, tmpl_t, tmpl_f = iso_z["iso"], iso_z["tmpl_t"], iso_z["tmpl_f"]
    basis = np.load(CKPT / "real_wbasis.npz")["basis"][:k]
    xf = np.load(CKPT / "real_Xfreq.npy", mmap_mode="r")
    xt = np.load(CKPT / "real_Xtime.npy", mmap_mode="r")
    Xte_f = np.asarray(xf[te], dtype=np.complex128)
    Xte_t = np.asarray(xt[te], dtype=np.float64)
    # weighted method
    coeff = ns.rankk_gls_coefficients(Xte_f, basis, w, return_complex=True)
    resid_f = Xte_f - np.atleast_2d(coeff) @ np.atleast_2d(basis)
    chi2_emp = float(np.mean(np.sum((np.abs(resid_f) ** 2) * w[None, :], axis=1)))
    mse_emp = float(np.mean(ns.raw_mse_from_freq_residual(resid_f, REAL_CFG.trace_len)))
    basis_amp = ns.amplitude_basis_from_subspace(basis, tmpl_f, w)
    amp_emp = ns.template_unit_amplitudes(Xte_f, basis_amp, tmpl_f, w)
    # isotropic method
    iso_k = iso[:k]
    coeff_iso = Xte_t @ iso_k.T
    resid_t = Xte_t - coeff_iso @ iso_k
    mse_iso = float(np.mean(resid_t ** 2))
    resid_iso_f = np.fft.rfft(resid_t, axis=1)
    chi2_iso = float(np.mean(np.sum((np.abs(resid_iso_f) ** 2) * w[None, :], axis=1)))
    gamma_iso = float(iso_k[0] @ tmpl_t)
    amp_iso = coeff_iso[:, 0] / gamma_iso
    cosines, angles = ns.principal_angles_weighted(basis, np.fft.rfft(iso_k, axis=1), w)

    def _r(a):
        mu, sd = float(np.mean(a)), float(np.std(a, ddof=1))
        return mu, sd, sd / abs(mu)

    mu_e, sd_e, rel_e = _r(amp_emp)
    mu_i, sd_i, rel_i = _r(amp_iso)
    _atomic_save_json(CKPT / f"real_eval_{k}.json", {
        "k": k, "mse_iso": mse_iso, "mse_empca": mse_emp,
        "delta_mse_iso_advantage": (mse_emp - mse_iso) / mse_iso,
        "chi2_iso": chi2_iso, "chi2_empca": chi2_emp,
        "delta_chi2_vs_iso": (chi2_iso - chi2_emp) / chi2_iso,
        "amp_mean_empca": mu_e, "amp_mean_iso": mu_i,
        "sigma_E_rel_empca": rel_e, "sigma_E_rel_iso": rel_i,
        "theta_w_first_deg": float(angles[0]), "theta_w_last_deg": float(angles[-1]),
    })


def unit_real_final():
    n, _ = _real_meta()
    rows = [json.load(open(CKPT / f"real_eval_{k}.json")) for k in RANKS_REAL]
    psd = np.load(CKPT / "real_psd.npz")
    amps = np.concatenate([np.load(CKPT / f"real_ofamp_{p}.npy") for p in range(3)])
    iso_z = np.load(CKPT / "real_isobasis.npz")
    tmpl_f = iso_z["tmpl_f"]
    w = psd["w"]
    from scipy import stats as st
    nphi = float(np.real(ns.weighted_inner(tmpl_f, tmpl_f, w)))
    sh_stat, sh_p = st.shapiro(amps[: min(len(amps), 4999)])
    mu, sd = float(np.mean(amps)), float(np.std(amps, ddof=1))
    df = pd.DataFrame(rows)
    df["sigma_E_rel_of"] = sd / mu
    dirs = ns.ensure_results_dirs(REAL_CFG)
    df.to_csv(dirs["tables"] / "block_12_g1_real_reversal.csv", index=False)
    checks = {}
    for k in (1, 2):
        p = CKPT / f"real_empcheck_{k}.json"
        if p.exists():
            checks[f"empca_vs_wsvd_k{k}"] = json.load(open(p))
    payload = {
        "of_amp_mean": mu, "of_amp_sigma": sd, "of_sigma_rel": sd / mu,
        "sigma_A_crb": 1.0 / np.sqrt(nphi), "crb_ratio": sd * np.sqrt(nphi),
        "shapiro_p": float(sh_p), "n_events": int(n),
        "empca_consistency": checks,
    }
    _atomic_save_json(CKPT / "real_final.json", payload)
    np.save(CKPT / "real_ofamp_all.npy", amps)


# -------------------------------------------------------------------- queue

def build_queue():
    q = []
    for nt in NOISE_TYPES:
        q.append((f"crb_{nt}.json", lambda nt=nt: unit_crb(nt)))
    for i in range(len(SCALING_POWERS)):
        q.append((f"scal_{i}.json", lambda i=i: unit_scaling(i)))
    for c in ("B", "A", "C", "P", "Z"):
        tag = SIM_CONFIGS[c]["tag"]
        for nt in NOISE_TYPES:
            for s in range(N_SEEDS):
                q.append((f"fam_rank{tag}_{nt}_{s}.npz",
                          lambda nt=nt, s=s, c=c: unit_rankfam(nt, s, c)))
                for k in RANKS_RANK:
                    q.append((f"rank{tag}_{nt}_{s}_{k}.json",
                              lambda nt=nt, s=s, k=k, c=c: unit_rank(nt, s, k, c)))
    for c in ("B", "A", "P"):
        tag = SIM_CONFIGS[c]["tag"]
        for nt in NOISE_TYPES:
            for s in range(N_SEEDS):
                q.append((f"fam_rev{tag}_{nt}_{s}.npz",
                          lambda nt=nt, s=s, c=c: unit_revfam(nt, s, c)))
                for k in RANKS_REV:
                    q.append((f"rev{tag}_{nt}_{s}_{k}.json",
                              lambda nt=nt, s=s, k=k, c=c: unit_rev(nt, s, k, c)))
    q.append(("simfinal.json", unit_simfinal))
    n, _ = _real_meta()
    n_chunks = (n + REAL_CHUNK - 1) // REAL_CHUNK
    for i in range(n_chunks):
        q.append((f"real_load_{i}.json", lambda i=i: unit_real_load(i)))
    q.append(("real_psd.npz", unit_real_psd))
    for p in range(3):
        q.append((f"real_of_{p}.json", lambda p=p: unit_real_of(p)))
    for p in range(4):
        q.append((f"real_gramjson_{p}.json", lambda p=p: unit_real_gram(p)))
    q.append(("real_isobasis.npz", unit_real_isobasis))
    for p in range(6):
        q.append((f"real_wgramjson_{p}.json", lambda p=p: unit_real_wsvd(p)))
    q.append(("real_wbasis.npz", unit_real_wbasis))
    for k in (1, 2):
        q.append((f"real_empcheck_{k}.json", lambda k=k: unit_real_empcheck(k)))
    for k in RANKS_REAL:
        q.append((f"real_eval_{k}.json", lambda k=k: unit_real_eval(k)))
    q.append(("real_final.json", unit_real_final))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=34.0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    q = build_queue()
    pending = [(name, fn) for name, fn in q if not (CKPT / name).exists()]
    if args.status or not pending:
        print(f"{len(q) - len(pending)}/{len(q)} units done; next: "
              f"{[n for n, _ in pending[:4]]}")
        if not pending:
            print("ALL_DONE")
        return
    t0 = time.time()
    done = 0
    for name, fn in pending:
        if done and time.time() - t0 > args.budget:
            break
        # skip a unit whose expected duration won't fit (heuristic: always try
        # at least one unit per call; atomic artifacts make retries safe)
        fn()
        done += 1
        print(f"unit {name} ok ({time.time() - t0:.1f}s elapsed)", flush=True)
        if time.time() - t0 > args.budget:
            break
    remaining = len([1 for n, _ in q if not (CKPT / n).exists()])
    print(f"PROGRESS {len(q) - remaining}/{len(q)} units done")
    if remaining == 0:
        print("ALL_DONE")


if __name__ == "__main__":
    main()
