"""Build the paper's evidence figure from archived results.

Three panels, one claim each:

  A  Rung 3, colored noise. NFPA minus the isotropically trained factored
     baseline, under both metrics. The two bars point opposite ways: metric
     reversal at fixed architecture.
  B  Rung 3, white noise. The same contrast with Sigma = sigma^2 I. Both at
     zero, which isolates covariance geometry rather than rank or optimisation
     as the cause of A.
  C  Rung 4, factored attention. The same contrast, and both bars point the
     SAME way -- the reversal does not survive into this instantiation.

The A-versus-C contrast is the paper's finding, so the panels share a
normalisation: each bar is a relative paired difference, so the three rungs are
directly comparable despite their different units.

Every number is read from an archived CSV; nothing is transcribed. Run from the
repository root:

    python scripts/make_rps_figure.py --out paper_rps2026/figures/evidence.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

RESULTS = Path("results/rps2026")
BOOTSTRAP_SEED = 20260808
RESAMPLES = 10000

# Colourblind-safe (Wong): blue for the likelihood metric, orange for raw MSE.
C_CHI2 = "#0072B2"
C_MSE = "#E69F00"


def _rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _paired(rows: list[dict], a: str, b: str, col: str) -> np.ndarray:
    """Per-seed paired difference a - b for one column."""
    by_seed: dict[str, dict[str, float]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["method"]] = float(r[col])
    seeds = sorted(s for s, m in by_seed.items() if a in m and b in m)
    return np.array([by_seed[s][a] - by_seed[s][b] for s in seeds])


def _rel_ci(diff: np.ndarray, scale: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """Relative paired difference and a percentile bootstrap interval, in percent."""
    if scale == 0 or not np.isfinite(scale):
        return 0.0, 0.0, 0.0
    boots = np.array([rng.choice(diff, diff.size, replace=True).mean() for _ in range(RESAMPLES)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * diff.mean() / scale, 100 * lo / scale, 100 * hi / scale


def _nfpa_panel(csv_path: Path, rng: np.random.Generator) -> dict:
    rows = _rows(csv_path)
    out = {}
    for key, col in (("chi2", "test_normalized_chi2"), ("mse", "test_raw_mse")):
        diff = _paired(rows, "NFPA", "Iso-MPCA", col)
        base = np.mean([float(r[col]) for r in rows if r["method"] == "Iso-MPCA"])
        out[key] = _rel_ci(diff, base, rng)
        out["n_seeds"] = diff.size
    return out


def _rung4_panel(rng: np.random.Generator, prefix: str = "E5") -> dict:
    """Rung 4 contrast from the two arms' held-out evaluation series.

    ``prefix`` selects the series pair: ``E5`` is the archived first seed;
    a replication seed passes e.g. ``E7`` with files named
    ``E7_t_{mse,chi2}_s<seed>_eval_series.csv`` (any single match per arm).
    """
    series = {}
    for arm in ("t_mse", "t_chi2"):
        exact = RESULTS / f"{prefix}_{arm}_eval_series.csv"
        candidates = [exact] if exact.exists() else sorted(RESULTS.glob(f"{prefix}_{arm}_s*_eval_series.csv"))
        if len(candidates) != 1:
            raise SystemExit(
                f"expected exactly one {prefix} series for {arm}, found "
                f"{[c.name for c in candidates]}"
            )
        series[arm] = _rows(candidates[0])
    steps = {arm: [r["step"] for r in rows] for arm, rows in series.items()}
    if steps["t_mse"] != steps["t_chi2"]:
        raise SystemExit(f"{prefix}: rung-4 arms have different evaluation grids; not comparable")

    out = {}
    for key, col in (("chi2", "val_chi2"), ("mse", "val_mse")):
        a = np.array([float(r[col]) for r in series["t_chi2"]])
        b = np.array([float(r[col]) for r in series["t_mse"]])
        out[key] = _rel_ci(a - b, float(b.mean()), rng)
    out["n_evals"] = len(steps["t_mse"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("paper_rps2026/figures/evidence.pdf"))
    ap.add_argument("--with-seed2", metavar="PREFIX", default=None,
                    help="add the replication seed's contrast to panel C as hatched "
                         "bars, read from <PREFIX>_t_{mse,chi2}[_s*]_eval_series.csv "
                         "(e.g. E7). Default: seed 1 only, output unchanged.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panel order A, B, C matters: the shared rng is consumed in this order, so
    # the archived figure's intervals reproduce bit-identically. The optional
    # seed-2 panel draws after C, leaving the first three panels' draws intact.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    panel_a = _nfpa_panel(RESULTS / "E1_nfpa_blockA" / "block_A_colored_raw.csv", rng)
    panel_b = _nfpa_panel(RESULTS / "E2_nfpa_blockB" / "block_B_white_raw.csv", rng)
    rung4 = _rung4_panel(rng)
    # Single-seed mode: no whiskers. The bootstrap over evaluation checkpoints
    # of ONE trajectory describes within-run stability, not across-run
    # uncertainty; drawing it as an interval invites the pseudoreplication
    # reading. The bar (the mean paired difference) is what one seed supports.
    rung4["suppress_intervals"] = True
    c_title = ("C  rung 4, factored attention\n"
               "single seed per arm (exploratory)")
    if args.with_seed2:
        rung4["seed2"] = _rung4_panel(rng, prefix=args.with_seed2)
        rung4["suppress_intervals"] = False
        c_title = ("C  rung 4, factored attention\n"
                   "$\\chi^2$-trained $-$ MSE-trained, two seeds")
    panels = [
        ("A  rung 3, colored noise\nNFPA $-$ isotropic, same architecture", panel_a),
        ("B  rung 3, white noise\ncontrol: $\\Sigma=\\sigma^2 I$", panel_b),
        (c_title, rung4),
    ]

    # 9.6 x 1.9: short enough that the figure plus caption fits the main
    # text's page budget when scaled to \linewidth (height ~1.1 in at 5.5 in).
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 1.9), sharey=False)
    for ax, (title, p) in zip(axes, panels):
        for i, (key, colour, label) in enumerate(
            ((("chi2"), C_CHI2, "likelihood residual $\\bar\\chi^2$"),
             (("mse"), C_MSE, "raw MSE"))
        ):
            def _bar(y, stats, height, hatch=None):
                mean, lo, hi = stats
                ax.barh(y, mean, height=height, color=colour, zorder=3,
                        hatch=hatch, edgecolor="white" if hatch else None, lw=0)
                if p.get("suppress_intervals"):
                    return
                ax.plot([lo, hi], [y, y], color="0.15", lw=1.6, zorder=4)
                for x in (lo, hi):
                    ax.plot([x, x], [y - .08, y + .08], color="0.15", lw=1.6, zorder=4)

            if "seed2" in p:
                _bar(i - 0.15, p[key], 0.26)
                _bar(i + 0.15, p["seed2"][key], 0.26, hatch="////")
            else:
                _bar(i, p[key], 0.5)
            if i == 0:
                ax.set_yticks([0, 1])
                ax.set_yticklabels([])
        ax.axvline(0, color="0.35", lw=1.0, zorder=2)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("relative paired difference (%)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", color="0.9", zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        # Symmetric limits covering every bar and interval bound, with headroom.
        # Symmetric so the *sign* of each bar -- which is the whole point -- is
        # read off the zero line rather than inferred from the axis.
        extent = max(abs(v) for k in ("chi2", "mse") for v in p[k])
        if "seed2" in p:
            extent = max(extent, max(abs(v) for k in ("chi2", "mse") for v in p["seed2"][k]))
        if extent < 1e-9:
            # The white-noise control is identically zero. A tiny auto-range
            # collides its tick labels and implies precision that is not the
            # point; give it a plain readable scale instead.
            ax.set_xlim(-1.0, 1.0)
            ax.set_xticks([-1, 0, 1])
            ax.text(0.0, 0.5, "both contrasts $=0$ exactly",
                    ha="center", va="center", fontsize=8, color="0.35")
        else:
            lim = max(extent * 1.35, 0.02)
            ax.set_xlim(-lim, lim)
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

    axes[0].set_yticklabels(["$\\bar\\chi^2$", "raw MSE"], fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_CHI2),
        plt.Rectangle((0, 0), 1, 1, color=C_MSE),
    ]
    fig.legend(handles, ["likelihood residual", "raw MSE"],
               loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")

    print(f"  written {args.out} and {args.out.with_suffix('.png')}")
    for title, p in panels:
        head = title.split("\n")[0]
        n = p.get("n_seeds") or p.get("n_evals")
        print(f"  {head}   (n={n})")
        for key in ("chi2", "mse"):
            m, lo, hi = p[key]
            print(f"     {key:<5} {m:+8.3f}%  CI [{lo:+.3f}, {hi:+.3f}]"
                  f"   {'excludes 0' if (lo > 0 or hi < 0) else 'includes 0'}")
        if "seed2" in p:
            for key in ("chi2", "mse"):
                m, lo, hi = p["seed2"][key]
                s1_sign, s2_sign = np.sign(p[key][0]), np.sign(m)
                print(f"     {key:<5} {m:+8.3f}%  CI [{lo:+.3f}, {hi:+.3f}]   seed 2, "
                      f"sign {'AGREES' if s1_sign == s2_sign else 'DISAGREES'} with seed 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
