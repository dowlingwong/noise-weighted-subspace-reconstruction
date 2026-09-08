#!/usr/bin/env python3
"""Figures for P1-E11 (excited support) and P1-E12 (tied vs untied AE).

``fig_excited_support.pdf`` — two panels: the angle of the learned whitened
subspace to the excited tangent span T_S, and the across-initialisation spread
of the part of that subspace orthogonal to T_S, both by training support.

``fig_ae_ablation.pdf`` — two panels per dataset: the angle to the EMPCA
subspace and the obliqueness of the induced projector, by arm and stopping
rule, on a log scale.

Usage::  python scripts/make_e11_e12_figures.py [--e11 DIR] [--e12 DIR]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _f(v):
    return float(v) if v not in ("", "None", None) else np.nan


def figure_e11(d: Path) -> None:
    rows = [r for r in _read(d / "excited_support.csv")]
    spread = _read(d / "complement_spread.csv")
    n_primary = min(int(r["n_train"]) for r in rows)
    rows = [r for r in rows if int(r["n_train"]) == n_primary]
    spread = [r for r in spread if int(r["n_train"]) == n_primary]

    supports = ["S_full", "S_restricted"]
    methods = ["NFPA", "CW-PCA"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    ax = axes[0]
    width, offs = 0.35, {"NFPA": -0.18, "CW-PCA": 0.18}
    for m in methods:
        vals = [[_f(r["angle_to_TS_max"]) for r in rows if r["support"] == s and r["method"] == m]
                for s in supports]
        pos = np.arange(len(supports)) + offs[m]
        ax.bar(pos, [np.mean(v) for v in vals], width,
               yerr=[np.std(v) for v in vals], capsize=3, label=m, alpha=0.85)
    ax.axhline(5.0, color="C3", ls="--", lw=1.0, label="predicted bound (5$^\\circ$)")
    ax.set_xticks(np.arange(len(supports)))
    ax.set_xticklabels(supports)
    ax.set_ylabel(r"largest principal angle to $T_S$ (deg)")
    ax.set_title("(a) is the excited span recovered?")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for m in methods:
        vals = [[_f(r["complement_spread_deg"]) for r in spread
                 if r["support"] == s and r["method"] == m] for s in supports]
        pos = np.arange(len(supports)) + offs[m]
        ax.bar(pos, [np.mean(v) if v else np.nan for v in vals], width,
               yerr=[np.std(v) if v else np.nan for v in vals], capsize=3, label=m, alpha=0.85)
    ax.axhline(30.0, color="C3", ls="--", lw=1.0, label="P-b: $>30^\\circ$ restricted")
    ax.axhline(5.0, color="C0", ls=":", lw=1.0, label="P-b: $<5^\\circ$ full")
    ax.set_xticks(np.arange(len(supports)))
    ax.set_xticklabels(supports)
    ax.set_ylabel(r"complement spread across restarts (deg)")
    ax.set_title(r"(b) is $T_S^\perp$ fixed by the data or the init?")
    ax.legend(fontsize=7, frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(d / "fig_excited_support.pdf")
    fig.savefig(d / "fig_excited_support.png", dpi=160)
    print(f"wrote {d / 'fig_excited_support.pdf'}")


def figure_e12(d: Path) -> None:
    rows = _read(d / "ae_ablation.csv")
    datasets = sorted({r["dataset"] for r in rows})
    arms = ["a_tied_weighted", "b_untied_weighted", "c_tied_mse", "d_untied_mse"]
    stops = ["converged", "early"]

    fig, axes = plt.subplots(2, len(datasets), figsize=(4.6 * len(datasets), 6.4), squeeze=False)
    for j, ds in enumerate(datasets):
        for i, (col, label, guide) in enumerate(
                [("angle_to_empca_max_deg", "angle to EMPCA (deg)", 0.01),
                 ("obliqueness", r"$\|\Pi-\Pi^*_\Sigma\|_F/\|\Pi\|_F$", 1e-6)]):
            ax = axes[i][j]
            grouped = defaultdict(list)
            for r in rows:
                if r["dataset"] == ds:
                    grouped[(r["arm"], r["stopping"])].append(_f(r[col]))
            width = 0.38
            for si, stop in enumerate(stops):
                means = [np.mean(grouped[(arm, stop)]) if grouped[(arm, stop)] else np.nan
                         for arm in arms]
                pos = np.arange(len(arms)) + (si - 0.5) * width
                ax.bar(pos, np.maximum(means, 1e-12), width, label=stop, alpha=0.85)
            ax.axhline(guide, color="C3", ls="--", lw=1.0)
            ax.set_yscale("log")
            ax.set_xticks(np.arange(len(arms)))
            ax.set_xticklabels([a.split("_", 1)[1].replace("_", "+") for a in arms],
                               rotation=20, fontsize=7)
            ax.set_ylabel(label, fontsize=8)
            ax.set_title(f"{ds}", fontsize=9)
            ax.grid(alpha=0.25, lw=0.5, axis="y")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(d / "fig_ae_ablation.pdf")
    fig.savefig(d / "fig_ae_ablation.png", dpi=160)
    print(f"wrote {d / 'fig_ae_ablation.pdf'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e11", type=str, default="results/paper1/P1-E11")
    ap.add_argument("--e12", type=str, default="results/paper1/P1-E12")
    a = ap.parse_args(argv)
    for name, rel, fn in (("E11", a.e11, figure_e11), ("E12", a.e12, figure_e12)):
        d = REPO_ROOT / rel
        try:
            fn(d)
        except FileNotFoundError as exc:
            print(f"{name}: skipped ({exc})")


if __name__ == "__main__":
    main()
