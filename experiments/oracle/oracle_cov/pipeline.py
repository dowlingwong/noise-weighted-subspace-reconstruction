"""End-to-end Tier-1 pipeline: training set -> frozen subjects -> clean reference ->
family windows -> monitor scores, baselines, consequence, tangent statistics.

One call of ``run_seed`` produces one row per (subject, stage, family, severity, window,
monitor).  Runners assemble seeds and analyses on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, PrimaryConfig

from . import families as F
from .baselines import make_baseline_set
from .consequence import AmplitudeDegradation, amplitude_rmse
from .generator import TrainingSet, make_training_set
from .monitors import contrastive_direction_alignment, make_monitor_set
from .protocol import Family, OracleCovConfig, RepresentationBatch
from .subjects import Subject, make_subject
from .tangent import TangentGeometry, fisher, tangent_geometry

EXCITED = ("amplitude", "rise_time", "decay_time", "secondary")
DEFAULT_STAGES = {"lin_cw": ["code"], "mlp_ae": ["h1", "h2", "code"]}


def make_batch(subject: Subject, ts: TrainingSet, w: F.WindowData, stage: str) -> RepresentationBatch:
    Z = subject.represent(w.X, stage)
    Zw_hat = subject.whiten_hat(w.X)
    recon = subject.recon_whitened(w.X)
    out = subject.outputs(w.X)
    return RepresentationBatch(stage=stage, Z=Z, outputs=out, recon=recon, meta={
        "chi2_bar": float(np.mean((recon - Zw_hat) ** 2)),
        "X_whitened": Zw_hat,
        "truth_amplitude": w.latents["amplitude"],
        "family": w.family, "window_meta": w.meta,
    })


@dataclass
class SeedContext:
    seed: int
    ts: TrainingSet
    tg: TangentGeometry
    subjects: dict[str, Subject]
    clean_mean_whitened: np.ndarray = None
    n_clean_reference: int = 0
    clean_energy_stats: dict = field(default_factory=dict)   # empirical clean-window energies along T_S / T_U / total
    clean_rmse: dict[str, float] = field(default_factory=dict)
    monitors: dict = field(default_factory=dict)      # (subject, stage) -> list of fitted monitors
    thresholds: dict = field(default_factory=dict)    # (subject, stage, monitor) -> FAR threshold
    clean_scores: dict = field(default_factory=dict)  # (subject, stage, monitor) -> held-out clean scores
    training_seconds: dict = field(default_factory=dict)


def build_seed_context(seed: int, cfg: OracleCovConfig, base: PrimaryConfig = PRIMARY, *, subjects=("lin_cw", "mlp_ae"),
                       stages: dict[str, list[str]] | None = None) -> SeedContext:
    stages = stages or DEFAULT_STAGES
    ts = make_training_set(seed, base, alpha=cfg.base_alpha)
    tg = tangent_geometry(ts, EXCITED, seed=seed)
    subs = {}
    for kind in subjects:
        t0 = time.time()
        subs[kind] = make_subject(kind, ts, rank=cfg.subject_rank, hidden=cfg.mlp_hidden, epochs=cfg.mlp_epochs, lr=cfg.mlp_lr, seed=seed)
        subs[kind].train_seconds = time.time() - t0
    ctx = SeedContext(seed=seed, ts=ts, tg=tg, subjects=subs)
    ctx.training_seconds = {k: s.train_seconds for k, s in subs.items()}

    # clean reference windows: thirds for fitting / threshold / held-out FAR check
    n_w = cfg.n_clean_windows
    clean_windows = [F.make_window(ts, F.family("clean"), cfg.window_size, seed=seed * 100000 + w) for w in range(n_w)]
    third = n_w // 3
    ctx.clean_mean_whitened = np.mean(np.concatenate([ts.whiten(w.X).reshape(len(w.X), -1, order="F") for w in clean_windows[:third]]), axis=0)
    ctx.n_clean_reference = third * cfg.window_size
    ener = {"total": [], "T_S": [], "T_U": []}
    for w in clean_windows[third:]:
        delta = ts.whiten(w.X).reshape(len(w.X), -1, order="F").mean(axis=0) - ctx.clean_mean_whitened
        ener["total"].append(float(delta @ delta)); ener["T_S"].append(float(np.sum((delta @ tg.T_S) ** 2)))
        ener["T_U"].append(float(np.sum((delta @ tg.T_U) ** 2)) if tg.T_U.shape[1] else 0.0)
    ctx.clean_energy_stats = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 1.0) for k, v in ener.items()}
    for kind, sub in subs.items():
        all_out = np.concatenate([sub.outputs(w.X) for w in clean_windows[:third]])
        all_truth = np.concatenate([w.latents["amplitude"] for w in clean_windows[:third]])
        ctx.clean_rmse[kind] = amplitude_rmse(all_out, all_truth)
        for stage in stages[kind]:
            batches = [make_batch(sub, ts, w, stage) for w in clean_windows]
            mons = make_monitor_set(sub, stage, k=cfg.stage_top_k) + make_baseline_set()
            for m in mons:
                m.fit(batches[:third])
                calib = np.array([m.score(b) for b in batches[third: 2 * third]])
                held = np.array([m.score(b) for b in batches[2 * third:]])
                from .report import far_threshold
                ctx.thresholds[(kind, stage, m.name)] = far_threshold(calib, cfg.false_alert_rate)
                ctx.clean_scores[(kind, stage, m.name)] = held
            ctx.monitors[(kind, stage)] = mons
    return ctx


def tangent_statistics(ctx: SeedContext, w: F.WindowData) -> dict[str, float]:
    """S/N geometric signatures in whitened observation space (true covariance)."""
    Xw = ctx.ts.whiten(w.X).reshape(len(w.X), -1, order="F")
    delta = Xw.mean(axis=0) - ctx.clean_mean_whitened
    split = ctx.tg.energy_split(delta)
    # residual variance along the excited span (should be 1 under training noise)
    resid = Xw - ctx.ts.whiten(w.clean + ctx.ts.ds.train_mean).reshape(len(w.X), -1, order="F")
    var_S = float(np.mean(np.var(resid @ ctx.tg.T_S, axis=0, ddof=1)))
    var_U = float(np.mean(np.var(resid @ ctx.tg.T_U, axis=0, ddof=1))) if ctx.tg.T_U.shape[1] else float("nan")
    # per-event displacement energy split (catches zero-mean deformations such as jitter)
    D = Xw - ctx.ts.whiten(ctx.ts.forward({k: v for k, v in w.latents.items() if k != "t0_shift"} | {"t0_shift": np.zeros(len(w.X))}) + ctx.ts.ds.train_mean).reshape(len(w.X), -1, order="F")
    eS = float(np.mean(np.sum((D @ ctx.tg.T_S) ** 2, axis=1))); eU = float(np.mean(np.sum((D @ ctx.tg.T_U) ** 2, axis=1))) if ctx.tg.T_U.shape[1] else 0.0
    eT = float(np.mean(np.sum(D ** 2, axis=1)))
    # noise-floor-corrected mean-displacement energies: for pure training noise n*|P delta|^2 ~ chi^2(dim)
    # empirical standardisation against clean windows (signal sampling variability along T_S is part of the null)
    e_tot = float(delta @ delta); e_S = float(np.sum((delta @ ctx.tg.T_S) ** 2)); e_U = float(np.sum((delta @ ctx.tg.T_U) ** 2)) if ctx.tg.T_U.shape[1] else 0.0
    zf = lambda e, key: (e - ctx.clean_energy_stats[key][0]) / max(ctx.clean_energy_stats[key][1], 1e-12)
    z_tot, z_S, z_U = zf(e_tot, "total"), zf(e_S, "T_S"), zf(e_U, "T_U")
    excess = e_tot - ctx.clean_energy_stats["total"][0]
    frac_S_excess = (e_S - ctx.clean_energy_stats["T_S"][0]) / excess if z_tot > 5 else float("nan")
    return {"mean_disp_in_T_S": split["in_T_S"], "mean_disp_in_T_U": split["in_T_U"], "mean_disp_elsewhere": split["elsewhere"],
            "mean_disp_outside_T_S": 1.0 - split["in_T_S"],
            "mean_disp_norm": float(np.linalg.norm(delta)),
            "mean_disp_excess_z": z_tot, "mean_disp_excess_z_T_S": z_S, "mean_disp_excess_z_T_U": z_U,
            "mean_disp_excess_frac_in_T_S": frac_S_excess,
            "resid_var_T_S": var_S, "resid_var_T_U": var_U,
            "event_disp_in_T_U_excess": (eU - ctx.tg.T_U.shape[1]) / max(eT, 1e-12),
            "event_disp_in_T_S_excess": (eS - ctx.tg.T_S.shape[1]) / max(eT, 1e-12)}


COLUMNS = ["seed", "subject", "stage", "family", "contract", "declared", "severity", "window", "event_group", "monitor",
           "score", "threshold", "alarm", "is_analytic", "needs_covariance", "K", "high_consequence", "clean_rmse", "window_rmse",
           "mean_disp_in_T_S", "mean_disp_in_T_U", "mean_disp_elsewhere", "mean_disp_outside_T_S", "mean_disp_norm",
           "mean_disp_excess_z", "mean_disp_excess_z_T_S", "mean_disp_excess_z_T_U", "mean_disp_excess_frac_in_T_S", "resid_var_T_S", "resid_var_T_U",
           "event_disp_in_T_U_excess", "event_disp_in_T_S_excess", "contrastive_norm", "contrastive_in_dominant", "fisher_rank_median",
           "chi2_bar", "runtime_seconds"]


def score_window(ctx: SeedContext, cfg: OracleCovConfig, fam: Family, window: int, *, stages=None) -> list[dict]:
    stages = stages or DEFAULT_STAGES
    w = F.make_window(ctx.ts, fam, cfg.window_size, seed=ctx.seed * 100000 + 50000 + window)
    t0 = time.time()
    tstats = tangent_statistics(ctx, w)
    _, ranks = fisher(ctx.ts, {k: v[:8] for k, v in w.latents.items()})
    rows = []
    for kind, sub in ctx.subjects.items():
        cons = AmplitudeDegradation(ctx.clean_rmse[kind], cfg.kappa_m)
        for stage in stages[kind]:
            b = make_batch(sub, ctx.ts, w, stage)
            K = cons.score(b.outputs, b.meta["truth_amplitude"])
            contr = contrastive_direction_alignment(sub, stage, ctx.monitors[(kind, stage)][0].mu_stage, b.Z, cfg.stage_top_k)
            for m in ctx.monitors[(kind, stage)]:
                s = m.score(b); th = ctx.thresholds[(kind, stage, m.name)]
                rows.append({"seed": ctx.seed, "subject": kind, "stage": stage, "family": fam.name, "contract": fam.contract,
                             "declared": fam.declared, "severity": fam.severity, "window": window, "event_group": window,
                             "monitor": m.name, "score": s, "threshold": th, "alarm": bool(s > th),
                             "is_analytic": m.is_analytic_for(fam), "needs_covariance": m.needs_covariance(),
                             "K": K, "high_consequence": cons.high(K), "clean_rmse": ctx.clean_rmse[kind],
                             "window_rmse": amplitude_rmse(b.outputs, b.meta["truth_amplitude"]),
                             **tstats, **contr, "fisher_rank_median": float(np.median(ranks)),
                             "chi2_bar": b.meta["chi2_bar"], "runtime_seconds": time.time() - t0})
    return rows


def clean_far_rows(ctx: SeedContext) -> list[dict]:
    """Held-out clean windows: realised false-alert rate per monitor."""
    rows = []
    for (kind, stage, name), scores in ctx.clean_scores.items():
        th = ctx.thresholds[(kind, stage, name)]
        rows.append({"seed": ctx.seed, "subject": kind, "stage": stage, "monitor": name, "n_heldout_clean": len(scores),
                     "realised_far": float(np.mean(scores > th)), "threshold": th})
    return rows
