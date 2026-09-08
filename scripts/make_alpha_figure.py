#!/usr/bin/env python3
"""P1-E2* figure: the cost of the separability prior, term by term.

Reads ``results/paper1/P1-E2/alpha_sweep.csv`` (and its ``summary.json`` for
the seed-bootstrap intervals) and writes ``fig_alpha_cost.pdf``.

Panels, all against alpha on the x axis:

  (a) approximation term  NFPA - CW-PCA   held-out chi^2, with 95 % interval
  (b) metric term         NFPA - Iso-MPCA held-out chi^2, with 95 % interval
  (c) planted channel / temporal projector errors, NFPA vs Iso-MPCA
  (d) the optimisation term: NFPA restart spread of chi^2, as error bars,
      against |metric term| (the scale it has to be small compared to)

Usage::  python scripts/make_alpha_figure.py [--dir results/paper1/P1-E2]
"""

from __future__ import annotations

import argparse
import csv
import json
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="results/paper1/P1-E2")
    a = ap.parse_args(argv)
    d = REPO_ROOT / a.dir
    rows = _read(d / "alpha_sweep.csv")
    summary = json.loads((d / "summary.json").read_text())

    alphas = sorted({float(r["alpha"]) for r in rows})
    per = summary["per_alpha"]

    def band(key):
        m = np.array([per[str(x)][key]["mean_difference"] for x in alphas])
        lo = np.array([per[str(x)][key]["ci95_low"] for x in alphas])
        hi = np.array([per[str(x)][key]["ci95_high"] for x in alphas])
        return m, lo, hi

    by = defaultdict(list)
    for r in rows:
        by[(float(r["alpha"]), r["method"])].append(r)

    def mean_of(method, col):
        return np.array([np.nanmean([_f(r[col]) for r in by[(x, method)]]) for x in alphas])

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    x = np.asarray(alphas)

    ax = axes[0, 0]
    m, lo, hi = band("approximation_term_chi2_nfpa_minus_cwpca")
    ax.fill_between(x, lo, hi, alpha=0.25, color="C0")
    ax.plot(x, m, "o-", color="C0")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_title(r"(a) approximation term: NFPA $-$ CW-PCA")
    ax.set_ylabel(r"held-out $\bar\chi^2$ difference")

    ax = axes[0, 1]
    m2, lo2, hi2 = band("metric_term_chi2_nfpa_minus_isompca")
    ax.fill_between(x, lo2, hi2, alpha=0.25, color="C3")
    ax.plot(x, m2, "o-", color="C3")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_title(r"(b) metric term: NFPA $-$ Iso-MPCA")
    ax.set_ylabel(r"held-out $\bar\chi^2$ difference")

    ax = axes[1, 0]
    for method, style in (("NFPA", "o-"), ("Iso-MPCA", "s--")):
        ax.plot(x, mean_of(method, "channel_projector_error"), style, label=f"{method} channel")
        ax.plot(x, mean_of(method, "temporal_projector_error"), style, alpha=0.55,
                label=f"{method} temporal")
    ax.set_title("(c) planted projector errors")
    ax.set_xlabel(r"nonseparability $\alpha$")
    ax.set_ylabel("projector error")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    spread_mean = np.array([per[str(v)]["nfpa_chi2_restart_spread_mean"] for v in alphas], dtype=float)
    ratio = np.array([(per[str(v)]["restart_spread_over_abs_metric_term"] or np.nan) for v in alphas],
                     dtype=float)
    ax.errorbar(x, np.zeros_like(x), yerr=spread_mean / 2.0, fmt="o", color="C2", capsize=3,
                label=r"NFPA restart spread of $\bar\chi^2$")
    ax.plot(x, np.abs(m2), "^--", color="C3", label=r"$|$metric term$|$")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_title("(d) optimisation term vs metric term")
    ax.set_xlabel(r"nonseparability $\alpha$")
    ax.legend(fontsize=7, frameon=False)
    for xi, ri in zip(x, ratio):
        if np.isfinite(ri):
            ax.annotate(f"{ri:.1%}", (xi, 0.0), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=6)

    for ax in axes.ravel():
        ax.set_xlabel(r"nonseparability $\alpha$")
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(d / "fig_alpha_cost.pdf")
    fig.savefig(d / "fig_alpha_cost.png", dpi=160)
    print(f"wrote {d / 'fig_alpha_cost.pdf'}")


if __name__ == "__main__":
    main()
