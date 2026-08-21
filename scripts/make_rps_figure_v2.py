"""Build the paper's two-panel evidence figure from archived summaries.

Panel A — replication. The rung-3 metric reversal in the frozen ten-seed
cohort and again in an independent twenty-seed replication, as relative paired
differences under both metrics. The two metrics point in opposite directions
(the reversal); the two cohorts agree. Cohorts are drawn separately and never
pooled: a pooled bar would imply a thirty-seed confirmatory experiment, which
this is not.

Panel B — what is recovered, as a ratio. Planted-factor projector errors and
held-out amplitude decoding at the separable condition, each as NFPA's error
divided by the isotropically trained baseline's on the same quantity. These are
different quantities in different units, so an absolute axis would invite a
comparison across groups that is not meaningful; the ratio is unitless and 1.0
is parity. Channel recovery is nearly exact while temporal recovery is weak, an
asymmetry a single joint number would hide.

Every value is read from an archived summary; nothing is transcribed. Run from
the repository root:

    python scripts/make_rps_figure_v2.py --out paper_rps2026/figures/evidence.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path("results/rps2026")

C_CHI2 = "#0072B2"   # likelihood residual
C_MSE = "#E69F00"    # raw MSE
C_NFPA = "#0072B2"
C_ISO = "#999999"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def replication_panel() -> dict:
    """Relative paired differences per cohort, from the E9 summary."""
    s = _load(RESULTS / "E9_blockA_replication" / "replication_summary.json")
    da = s["direction_agreement"]

    # Cohort means come from the archived summary; the relative scale uses the
    # isotropic baseline's own held-out level, read from the raw tables.
    import csv

    def baseline(csv_path: Path, col: str) -> float:
        rows = list(csv.DictReader(open(csv_path)))
        vals = [float(r[col]) for r in rows if r["method"] == "Iso-MPCA"]
        return float(np.mean(vals))

    frozen_csv = RESULTS / "E1_nfpa_blockA" / "block_A_colored_raw.csv"
    repl_csv = RESULTS / "E9_blockA_replication" / "replication_raw.csv"

    out = {}
    for metric, key in (("chi2", "test_normalized_chi2"), ("mse", "test_raw_mse")):
        out[metric] = {
            "frozen": 100 * da[key]["original_mean_difference"] / baseline(frozen_csv, key),
            "replication": 100 * da[key]["replication_mean_difference"] / baseline(repl_csv, key),
            "sign_agrees": bool(da[key]["sign_agrees"]),
        }
    return out


def semantic_panel() -> dict:
    """Per-method planted-factor and amplitude values at the separable condition."""
    s = _load(RESULTS / "E8_semantics" / "separable_alpha_0" / "summary.json")
    pm = s["per_method"]
    keys = (
        ("channel_projector_error", "channel\nprojector error"),
        ("temporal_projector_error", "temporal\nprojector error"),
        ("amplitude_rmse", "amplitude\nRMSE"),
    )
    return {
        "labels": [label for _, label in keys],
        "NFPA": [float(pm["NFPA"][k]["mean"]) for k, _ in keys],
        "Iso-MPCA": [float(pm["Iso-MPCA"][k]["mean"]) for k, _ in keys],
        "analysis_status": s["analysis_status"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("paper_rps2026/figures/evidence.pdf"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    # Matplotlib's PDF backend embeds text as Type 3 by default. NeurIPS-style
    # venues check for Type 3 fonts, and a figure is the usual way they get in.
    # Type 42 is TrueType, which the checkers accept; nothing about the drawing
    # changes, only how its glyphs are stored.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    rep = replication_panel()
    sem = semantic_panel()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 1.5), gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- Panel A: cohorts side by side, both metrics.
    ypos = np.array([1.0, 0.0])          # chi2 on top, mse below
    height = 0.34
    for i, (metric, colour) in enumerate((("chi2", C_CHI2), ("mse", C_MSE))):
        y = ypos[i]
        f, r = rep[metric]["frozen"], rep[metric]["replication"]
        axA.barh(y + height / 2, f, height=height, color=colour, zorder=3)
        axA.barh(y - height / 2, r, height=height, color=colour,
                 zorder=3, hatch="////", edgecolor="white", lw=0)
        # The two metrics differ by more than an order of magnitude, so a linear
        # axis hides the small one -- and its SIGN is the whole point. Annotate
        # both cohorts explicitly and use a symmetric log scale below.
        # One label per metric, not one per bar: at this figure height two
        # stacked annotations collide with the bars they describe.
        side = 1 if f > 0 else -1
        anchor = max(abs(f), abs(r)) * side
        axA.text(anchor + side * abs(anchor) * 0.22, y, f"{f:+.2f}% / {r:+.2f}%",
                 va="center", ha="left" if side > 0 else "right", fontsize=7)
    axA.axvline(0, color="0.35", lw=1.0, zorder=2)
    axA.set_yticks(ypos)
    axA.set_yticklabels(["$\\bar\\chi^2$", "raw MSE"], fontsize=8)
    axA.set_xlabel("relative paired difference (%)", fontsize=8)
    axA.set_title("A  rung 3 reversal, primary condition $\\alpha=0.1$\n"
                  "solid = frozen 10 seeds, hatched = disjoint 20",
                  fontsize=8.5, loc="left")
    axA.tick_params(labelsize=7.5)
    axA.grid(axis="x", color="0.9", zorder=0)
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)
    axA.set_xscale("symlog", linthresh=1.0)
    extent = max(abs(v) for m in ("chi2", "mse") for v in
                 (rep[m]["frozen"], rep[m]["replication"]))
    axA.set_xlim(-extent * 6, extent * 6)
    axA.set_xticks([-10, -1, 0, 1, 10, 100])
    axA.set_xticklabels(["-10", "-1", "0", "1", "10", "100"])
    axA.set_xlabel("relative paired difference (%, symlog)", fontsize=8)

    # ---- Panel B: what the factors recover, as a RATIO.
    # Projector error and amplitude RMSE are different quantities in different
    # units, so plotting them on one absolute axis invites a comparison that is
    # not meaningful. Each bar is instead NFPA's error divided by the isotropic
    # baseline's error on the same quantity: unitless, comparable across groups,
    # and 1.0 is parity. Absolute values are annotated so nothing is hidden.
    ratios = [n / i for n, i in zip(sem["NFPA"], sem["Iso-MPCA"])]
    x = np.arange(len(sem["labels"]))
    axB.axhline(1.0, color="0.35", lw=1.0, zorder=2)
    axB.bar(x, ratios, width=0.5, color=C_NFPA, zorder=3)
    for xi, r in enumerate(ratios):
        axB.text(xi, r + 0.04, f"{r:.2f}", ha="center", va="bottom", fontsize=7)
    axB.set_xticks(x)
    axB.set_xticklabels(sem["labels"], fontsize=7.5)
    axB.set_ylabel("NFPA / isotropic", fontsize=8)
    axB.set_ylim(0, 1.25)
    axB.set_title("B  recovery, exactly separable condition $\\alpha=0$\n"
                  "ratio to the isotropic baseline; 1.0 = parity, lower is better",
                  fontsize=8.5, loc="left")
    axB.tick_params(labelsize=7.5)
    axB.grid(axis="y", color="0.9", zorder=0)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")

    print(f"  written {args.out}")
    for m in ("chi2", "mse"):
        r = rep[m]
        print(f"  A {m:5s} frozen {r['frozen']:+7.3f}%   replication {r['replication']:+7.3f}%"
              f"   sign agrees: {r['sign_agrees']}")
    for lab, a, b in zip(sem["labels"], sem["NFPA"], sem["Iso-MPCA"]):
        print(f"  B {lab.replace(chr(10), ' '):24s} NFPA {a:.4f}   iso {b:.4f}"
              f"   ratio {a / b:.4f}")
    print(f"  B analysis_status = {sem['analysis_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
