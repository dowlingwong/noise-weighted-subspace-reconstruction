"""CLI runner for Block 11 (G2: Exp D/E repair) and Block 12 (G1: real-data
metric reversal).

Usage:
    python implementation/run_block11_12.py --part crb,scaling,rank,reversal,real
    python implementation/run_block11_12.py --quick          # smoke test
    python implementation/run_block11_12.py --part real      # G1 only

Outputs land in results/tables, results/figures, results/manifests with the
block_11_* / block_12_* prefixes used by the rest of the suite.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import notebook_support as ns


def _save(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def _save_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"  wrote {path}")


def part_crb(cfg, dirs, quick):
    df = ns.run_block11_crb_units(cfg, n_replicates=120 if quick else 3000)
    _save(df, dirs["tables"] / "block_11_e4_crb_units.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["var_emp"], width, label="empirical Var($\\hat A$)")
    ax.bar(x + width / 2, df["var_pred_dft_weights"], width, label="predicted $1/N_\\Phi$ (consistent)")
    ax.set_xticks(x, df["noise_type"])
    ax.set_yscale("log")
    ax.set_ylabel("amplitude variance")
    ax.set_title("E4 repair: CRB attainment, consistent PSD conventions")
    ax.legend()
    ns.save_figure(fig, dirs["figures"] / "block_11_e4_crb_units.png")
    plt.close(fig)
    return {"crb_max_rel_err": float(df[["rel_err_dft", "rel_err_kernel"]].to_numpy().max())}


def part_scaling(cfg, dirs, quick):
    df = ns.run_block11_sigma_scaling(cfg, n_events=80 if quick else 400)
    _save(df, dirs["tables"] / "block_11_e5_sigma_scaling.csv")
    slope = float(np.polyfit(np.log(df["noise_power"]), np.log(df["sigma_emp"]), 1)[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(df["noise_power"], df["sigma_emp"], "o", label="empirical $\\sigma_E$")
    ax.loglog(df["noise_power"], df["sigma_pred"], "-", label="theory $1/\\sqrt{N_\\Phi}$")
    ax.set_xlabel("noise power")
    ax.set_ylabel("$\\sigma_E$ (template-amplitude units)")
    ax.set_title(f"E5 repair: $\\sigma_E$ scaling (slope {slope:.3f})")
    ax.legend()
    ns.save_figure(fig, dirs["figures"] / "block_11_e5_sigma_scaling.png")
    plt.close(fig)
    return {"sigma_scaling_loglog_slope": slope}


def part_rank(cfg, dirs, quick):
    out = ns.run_block11_rank_study(
        cfg,
        n_seeds=2 if quick else 8,
        n_events=60 if quick else 240,
        ranks=(1, 2, 3) if quick else (1, 2, 3, 4, 5),
    )
    _save(out["rank_df"], dirs["tables"] / "block_11_expE_rank_seeds.csv")
    _save(out["agg_df"], dirs["tables"] / "block_11_expE_rank_agg.csv")
    agg = out["agg_df"]

    # Fig 7 replacement: sigma_E(k)/sigma_OF
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nt, marker in (("white", "o"), ("pink", "s"), ("brownian", "^")):
        sub = agg[agg["noise_type"] == nt]
        if len(sub):
            ax.errorbar(sub["k"], sub["sigma_ratio_mean"], yerr=sub["sigma_ratio_std"],
                        marker=marker, capsize=3, label=nt)
    ax.axhline(1.0, ls="--", c="k", lw=1, label="OF baseline")
    ax.set_xlabel("EMPCA rank $k$")
    ax.set_ylabel("$\\sigma_E(k)/\\sigma_E^{\\mathrm{OF}}$")
    ax.set_title("Fig 7 repair: resolution vs rank (mode='full', consistent PSD units)")
    ax.legend()
    ns.save_figure(fig, dirs["figures"] / "block_11_fig7_sigmaE_vs_rank.png")
    plt.close(fig)

    # Fig 15 replacement: amplitude distributions k=1 vs k=2
    pooled = out["amp_pooled"]
    noise_types = [nt for nt in ("white", "pink", "brownian") if nt in pooled]
    fig, axes = plt.subplots(1, len(noise_types), figsize=(4.2 * len(noise_types), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, nt in zip(axes, noise_types):
        a_true = out["amp_true"][nt]
        for k, color in ((1, "tab:blue"), (2, "tab:orange")):
            a = pooled[nt][k]
            if len(a):
                ax.hist(a / a_true, bins=40, histtype="step", lw=1.6, density=True,
                        color=color, label=f"$k={k}$ (bias {np.mean(a)/a_true-1:+.2%})")
        ax.axvline(1.0, ls="--", c="k", lw=1)
        ax.set_title(nt)
        ax.set_xlabel("$\\hat A / A_{\\mathrm{true}}$")
        ax.legend(fontsize=8)
    fig.suptitle("Fig 15 repair: amplitude under timing jitter, rank 1 vs 2")
    ns.save_figure(fig, dirs["figures"] / "block_11_fig15_amp_bias.png")
    plt.close(fig)

    # Fig 16 replacement: KS p vs rank (MC noise-only null)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rank_df = out["rank_df"]
    for nt, marker in (("white", "o"), ("pink", "s"), ("brownian", "^")):
        sub = rank_df[rank_df["noise_type"] == nt]
        if len(sub):
            med = sub.groupby("k")["ks_pvalue"].median()
            lo = sub.groupby("k")["ks_pvalue"].quantile(0.25)
            hi = sub.groupby("k")["ks_pvalue"].quantile(0.75)
            ax.plot(med.index, med.values, marker=marker, label=f"{nt} (median)")
            ax.fill_between(med.index, lo.values, hi.values, alpha=0.2)
    ax.axhline(0.05, ls="--", c="k", lw=1, label="$\\alpha=0.05$")
    ax.set_yscale("log")
    ax.set_xlabel("EMPCA rank $k$")
    ax.set_ylabel("KS $p$-value vs noise-only null (MC)")
    ax.set_title("Fig 16 repair: residual whiteness vs rank")
    ax.legend(fontsize=8)
    ns.save_figure(fig, dirs["figures"] / "block_11_fig16_ks_pvalues.png")
    plt.close(fig)

    summary = {}
    for nt in noise_types:
        sub = agg[agg["noise_type"] == nt].set_index("k")
        if 1 in sub.index and 2 in sub.index:
            summary[nt] = {
                "sigma_ratio_k1": float(sub.loc[1, "sigma_ratio_mean"]),
                "sigma_ratio_k2": float(sub.loc[2, "sigma_ratio_mean"]),
                "bias_k1": float(sub.loc[1, "bias_rel_mean"]),
                "bias_k2": float(sub.loc[2, "bias_rel_mean"]),
                "ks_p_median_k2": float(sub.loc[2, "ks_pvalue_median"]),
            }
    return {"rank_study": summary}


def part_reversal(cfg, dirs, quick):
    df = ns.run_block11_reversal_rank_sweep(
        cfg,
        n_seeds=2 if quick else 8,
        n_train=60 if quick else 400,
        n_test=30 if quick else 200,
        ranks=(1, 2) if quick else (1, 2, 3),
    )
    _save(df, dirs["tables"] / "block_11_table47_reconciliation_seeds.csv")
    agg = (
        df.groupby(["noise_type", "k"])
        .agg(
            delta_mse_iso_advantage=("delta_mse_iso_advantage", "mean"),
            delta_chi2_vs_iso=("delta_chi2_vs_iso", "mean"),
            delta_chi2_vs_iso_std=("delta_chi2_vs_iso", "std"),
            delta_chi2_vs_rank1=("delta_chi2_vs_rank1", "mean"),
            theta_w_first_deg=("theta_w_first_deg", "mean"),
            theta_w_last_deg=("theta_w_last_deg", "mean"),
            chi2_iso=("chi2_iso", "mean"),
            chi2_empca=("chi2_empca", "mean"),
        )
        .reset_index()
    )
    _save(agg, dirs["tables"] / "block_11_table47_reconciliation_agg.csv")
    return {"table47": agg.to_dict(orient="records")}


def part_real(cfg, dirs, quick):
    # real traces are fixed-length: never shrink trace_len for this part
    cfg = ns.CanonicalConfig(seed=cfg.seed).validate()
    ranks = (1, 2) if quick else (1, 2, 3, 4, 5)
    out = ns.run_block12_real_reversal(cfg, ranks=ranks, n_iter=8 if quick else None)
    df = out["reversal_df"]
    _save(df, dirs["tables"] / "block_12_g1_real_reversal.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(df["k"], df["mse_iso"], "o-", label="isotropic PCA")
    axes[0].plot(df["k"], df["mse_empca"], "s-", label="weighted EMPCA")
    axes[0].set_xlabel("rank $k$"); axes[0].set_ylabel("held-out raw MSE")
    axes[0].set_title("raw MSE (isotropic objective)")
    axes[0].legend()
    axes[1].plot(df["k"], df["chi2_iso"], "o-", label="isotropic PCA")
    axes[1].plot(df["k"], df["chi2_empca"], "s-", label="weighted EMPCA")
    axes[1].set_xlabel("rank $k$"); axes[1].set_ylabel("held-out weighted $\\chi^2$")
    axes[1].set_title("weighted residual (likelihood)")
    axes[1].legend()
    axes[2].plot(df["k"], df["sigma_E_rel_iso"] * 100, "o-", label="isotropic PCA")
    axes[2].plot(df["k"], df["sigma_E_rel_empca"] * 100, "s-", label="weighted EMPCA")
    axes[2].axhline(df["sigma_E_rel_of"].iloc[0] * 100, ls="--", c="k", lw=1, label="OF")
    axes[2].set_xlabel("rank $k$"); axes[2].set_ylabel("$\\sigma_E / E$  [%]")
    axes[2].set_title("energy resolution at the K-$\\alpha$ line")
    axes[2].legend()
    fig.suptitle("G1: real-data isotropic vs weighted (K-alpha)")
    ns.save_figure(fig, dirs["figures"] / "block_12_g1_metric_reversal.png")
    plt.close(fig)

    psd = out["psd"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(psd["freqs_seg"][1:], psd["J_seg_phys"][1:], lw=0.8, alpha=0.7,
              label="pre-trigger segments (measured)")
    ax.loglog(psd["freqs_full"][1:], psd["J_phys"][1:], lw=1.4,
              label="interpolated full-grid PSD")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [ADC$^2$/Hz]")
    ax.set_title("Real K-alpha noise PSD (Fig 21 replacement)")
    ax.legend()
    ns.save_figure(fig, dirs["figures"] / "block_12_real_psd.png")
    plt.close(fig)

    e12 = out["e12"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    amps = e12["amp_of_all"]
    ax.hist(amps, bins=60, density=True, alpha=0.6, label="OF amplitudes (all events)")
    mu, sd = float(np.mean(amps)), float(np.std(amps, ddof=1))
    grid = np.linspace(amps.min(), amps.max(), 400)
    ax.plot(grid, np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            "k-", lw=1.4, label=f"Gaussian fit ($\\sigma$={sd:.1f})")
    ax.set_xlabel("OF amplitude [ADC]")
    ax.set_title(f"E12: K-alpha amplitude histogram (Shapiro p={e12['shapiro_p']:.3f})")
    ax.legend(fontsize=8)
    ns.save_figure(fig, dirs["figures"] / "block_12_e12_amp_histogram.png")
    plt.close(fig)

    return {
        "real_reversal": df.drop(columns=[c for c in df.columns if c.startswith("sigma_E_ev")]).to_dict(orient="records"),
        "e12_sigma_A_obs": e12["sigma_A_obs"],
        "e12_sigma_A_crb": e12["sigma_A_crb"],
        "e12_crb_ratio": e12["crb_ratio"],
        "e12_shapiro_p": e12["shapiro_p"],
        "of_cross_corr_with_rqs": e12["of_cross_corr"],
        "kalpha_line_ev_assumed": out["kalpha_line_ev"],
    }


PARTS = {
    "crb": part_crb,
    "scaling": part_scaling,
    "rank": part_rank,
    "reversal": part_reversal,
    "real": part_real,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="crb,scaling,rank,reversal,real")
    ap.add_argument("--quick", action="store_true", help="small smoke-test scale")
    args = ap.parse_args()

    cfg = ns.CanonicalConfig().validate()
    if args.quick:
        cfg = replace(cfg, trace_len=4096, pretrigger=600, crb_replicates=120)
    dirs = ns.ensure_results_dirs(cfg)
    manifest = {"quick": bool(args.quick), "seed": cfg.seed, "trace_len": cfg.trace_len}
    for name in [p.strip() for p in args.part.split(",") if p.strip()]:
        t0 = time.time()
        print(f"=== part {name} ===", flush=True)
        manifest[name] = PARTS[name](cfg, dirs, args.quick)
        manifest[name + "_runtime_s"] = round(time.time() - t0, 1)
        print(f"=== part {name} done in {manifest[name + '_runtime_s']}s ===", flush=True)
    _save_json(manifest, dirs["manifests"] / ("block_11_12_manifest_quick.json" if args.quick else "block_11_12_manifest.json"))


if __name__ == "__main__":
    main()
