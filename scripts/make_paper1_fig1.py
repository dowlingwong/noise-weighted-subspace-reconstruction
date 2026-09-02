"""Render Paper 1 Figure 1 from the P1-E1 data product.

Panels: (a) one noisy 8x256 event and its clean truth; (b) the two
reconstructions (isotropic factored model vs NFPA); (c) learned channel
projector U_cU_c^T against the planted one, for each method (one seed);
(d) per-seed channel and temporal projector errors for both methods;
(e) decoded amplitude RMSE per seed.  The subspace angle is deliberately
absent: the picture is the physical structure, the angle goes in the caption.

Usage::  python scripts/make_paper1_fig1.py [--in results/paper1/P1-E1] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = "#2b2b2b"; ISO = "#b45f3c"; NFPA = "#3c6e9b"; PLANT = "#6f6f6f"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results/paper1/P1-E1"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    inp = Path(a.inp)
    P = np.load(inp / "projectors.npz")
    E = np.load(inp / "event_example.npz")
    with (inp / "decoding.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    seeds = sorted({int(r["seed"]) for r in rows})

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.05], hspace=0.55, wspace=0.45)

    # (a) event
    ax = fig.add_subplot(gs[0, 0:2])
    t = np.arange(E["noisy"].shape[1])
    sc = float(np.abs(E["clean"]).max()) or 1.0
    for c in range(E["noisy"].shape[0]):
        ax.plot(t, (E["noisy"][c] - E["noisy"][c].mean()) / sc + 3 * c, color=INK, lw=0.5, alpha=0.6)
        ax.plot(t, E["clean"][c] / sc + 3 * c, color=PLANT, lw=0.9)
    ax.set_title("(a) one event: 8 channels, noisy (thin) and planted clean (grey)", fontsize=9, loc="left")
    ax.set_xlabel("sample"); ax.set_yticks([])

    # (b) reconstructions
    ax = fig.add_subplot(gs[0, 2:4])
    ch = int(np.argmax(np.abs(E["clean"]).sum(axis=1)))
    ax.plot(t, E["clean"][ch], color=PLANT, lw=1.6, label="planted clean")
    ax.plot(t, E["recon_Iso_MPCA"][ch], color=ISO, lw=1.0, label="MSE geometry (Iso-MPCA)")
    ax.plot(t, E["recon_NFPA"][ch], color=NFPA, lw=1.0, label=r"$\Sigma^{-1}$ geometry (NFPA)")
    ax.set_title(f"(b) reconstruction of channel {ch}: same architecture, two metrics", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False); ax.set_xlabel("sample")

    # (c) projectors
    s = a.seed
    panels = [("planted", P[f"seed{s:02d}_NFPA_P_channel_planted"], PLANT),
              ("Iso-MPCA (MSE)", P[f"seed{s:02d}_Iso_MPCA_P_channel_learned"], ISO),
              ("NFPA ($\\Sigma^{-1}$)", P[f"seed{s:02d}_NFPA_P_channel_learned"], NFPA)]
    # note: the planted projector is compared in each method's own coordinates; for the
    # isotropic model that is the raw channel basis, for NFPA the whitened one.
    panels[0] = ("planted (raw coords)", P[f"seed{s:02d}_Iso_MPCA_P_channel_planted"], PLANT)
    for j, (name, M, col) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(M, cmap="RdGy_r", vmin=-1, vmax=1)
        ax.set_title(name, fontsize=8, color=col, pad=2); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(P[f"seed{s:02d}_NFPA_P_channel_planted"], cmap="RdGy_r", vmin=-1, vmax=1)
    ax.set_title("planted (whitened coords)", fontsize=8, color=PLANT, pad=2); ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.125, 0.655, f"(c) channel projectors $U_cU_c^T$, seed {s}: learned (centre) against planted (outer), each in its own coordinates", fontsize=9)

    # (d) projector errors per seed
    ax = fig.add_subplot(gs[2, 0:2])
    for name, col, off in (("Iso-MPCA", ISO, -0.15), ("NFPA", NFPA, 0.15)):
        ce = [float(r["channel_projector_error"]) for r in rows if r["method"] == name]
        te = [float(r["temporal_projector_error"]) for r in rows if r["method"] == name]
        ax.scatter(np.array(seeds) + off, ce, color=col, marker="o", s=18, label=f"{name} channel")
        ax.scatter(np.array(seeds) + off, te, color=col, marker="^", s=18, facecolors="none", label=f"{name} temporal")
    ax.set_title("(d) planted-projector error per seed (0 = exact recovery)", fontsize=9, loc="left")
    ax.set_xlabel("seed"); ax.set_ylabel("normalised Frobenius error"); ax.legend(fontsize=7, frameon=False, ncol=2)

    # (e) amplitude decoding
    ax = fig.add_subplot(gs[2, 2:4])
    for name, col, off in (("Iso-MPCA", ISO, -0.15), ("NFPA", NFPA, 0.15)):
        v = [float(r["amplitude_rmse"]) for r in rows if r["method"] == name]
        ax.scatter(np.array(seeds) + off, v, color=col, s=18, label=name)
    ax.set_title("(e) decoded amplitude RMSE per seed (linear readout on the code)", fontsize=9, loc="left")
    ax.set_xlabel("seed"); ax.set_ylabel("RMSE"); ax.legend(fontsize=7, frameon=False)

    out = inp / "fig1.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(inp / "fig1.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
