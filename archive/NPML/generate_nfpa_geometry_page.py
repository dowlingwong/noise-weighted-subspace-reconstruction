from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "NPML" / "results" / "all_plots_review"


INK = "#202124"
MUTED = "#667085"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
EDGE = "#D0D5DD"
BG = "#FFFFFF"
PANEL_BG = "#F8FAFC"


def panel(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE)
        spine.set_linewidth(1.2)
    title_lines = title.count("\n") + 1
    ax.text(0.035, 0.955, title, ha="left", va="top", fontsize=16, weight="bold", color=INK, linespacing=1.05)
    ax.text(0.035, 0.895 - 0.055 * (title_lines - 1), subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)


def arrow(ax, xy0, xy1, color=MUTED, lw=1.6, scale=14) -> None:
    ax.add_patch(
        FancyArrowPatch(
            xy0,
            xy1,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_projection(ax, x0, x1, color, label) -> None:
    ax.plot([x0[0], x1[0]], [x0[1], x1[1]], color=color, linewidth=2.2)
    ax.scatter([x0[0]], [x0[1]], s=34, color=color, zorder=5)
    ax.text(x1[0] + 0.015, x1[1], label, fontsize=9.5, color=color, va="center")


def draw_geometry_hierarchy(ax) -> None:
    panel(
        ax,
        "1. Geometry hierarchy",
        "Each method defines a different signal set and projection.",
    )
    ax.text(
        0.035,
        0.835,
        "Physical latent parameters generate a nonlinear family of detector responses",
        ha="left",
        va="top",
        fontsize=9.7,
        color=INK,
    )
    ax.text(
        0.035,
        0.795,
        "that cannot be represented by a single linear subspace.",
        ha="left",
        va="top",
        fontsize=9.7,
        color=INK,
    )
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(26, 2)) * np.array([0.13, 0.08]) + np.array([0.22, 0.43])
    ax.scatter(pts[:, 0], pts[:, 1], s=13, color="#98A2B3", alpha=0.45)

    # OF line
    x = np.linspace(0.08, 0.36, 2)
    y = 0.17 + 0.60 * x
    ax.plot(x, y, color=BLUE, linewidth=3)
    ax.text(0.06, 0.73, "OF", color=BLUE, fontsize=11, weight="bold")
    ax.text(0.06, 0.69, "fixed template line", color=MUTED, fontsize=9)

    # PCA plane
    plane_x = np.array([0.47, 0.75, 0.84, 0.56, 0.47])
    plane_y = np.array([0.30, 0.22, 0.42, 0.51, 0.30])
    ax.fill(plane_x, plane_y, color=ORANGE, alpha=0.16, edgecolor=ORANGE, linewidth=2)
    ax.text(0.50, 0.73, "PCA", color=ORANGE, fontsize=11, weight="bold")
    ax.text(0.50, 0.69, "linear Euclidean subspace", color=MUTED, fontsize=9)

    # EMPCA geometry overlay
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = 0.66, 0.39
    ell_x = cx + 0.18 * np.cos(theta) * np.cos(0.45) - 0.065 * np.sin(theta) * np.sin(0.45)
    ell_y = cy + 0.18 * np.cos(theta) * np.sin(0.45) + 0.065 * np.sin(theta) * np.cos(0.45)
    ax.plot(ell_x, ell_y, color=GREEN, linewidth=2.0, alpha=0.95)
    ax.text(0.50, 0.62, "EMPCA", color=GREEN, fontsize=11, weight="bold")
    ax.text(0.50, 0.58, "linear Mahalanobis subspace", color=MUTED, fontsize=9)

    # NFPA manifold
    t = np.linspace(0, 1, 160)
    mx = 0.15 + 0.70 * t
    my = 0.12 + 0.10 * np.sin(2.2 * np.pi * t) + 0.16 * t
    ax.plot(mx, my, color=PURPLE, linewidth=3.2)
    ax.fill_between(mx, my - 0.025, my + 0.025, color=PURPLE, alpha=0.10)
    ax.text(0.06, 0.23, "NFPA / AE", color=PURPLE, fontsize=11, weight="bold")
    ax.text(0.06, 0.19, "nonlinear signal manifold", color=MUTED, fontsize=9)


def rounded_box(ax, xy, w, h, text, face="#FFFFFF", edge=EDGE, fontsize=11) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)


def draw_nfpa_map(ax) -> None:
    panel(
        ax,
        "2. NFPA as Maximum-Likelihood\nManifold Projection",
        "Projection is chosen by the Mahalanobis detector-noise metric.",
    )
    y = 0.58
    rounded_box(ax, (0.06, y), 0.16, 0.12, "trace\nx", face="#EFF8FF", edge=BLUE)
    rounded_box(ax, (0.31, y), 0.20, 0.12, "encoder\nq_phi(z|x)", face="#F0FDF4", edge=GREEN)
    rounded_box(ax, (0.60, y), 0.12, 0.12, "latent\nz", face="#FFFBEB", edge=ORANGE)
    rounded_box(ax, (0.80, y), 0.16, 0.12, "decoder\nx_hat", face="#F5F3FF", edge=PURPLE)
    arrow(ax, (0.23, y + 0.06), (0.30, y + 0.06))
    arrow(ax, (0.52, y + 0.06), (0.59, y + 0.06))
    arrow(ax, (0.73, y + 0.06), (0.79, y + 0.06))

    ax.text(
        0.50,
        0.39,
        r"$\hat{x}=f_\theta(z)$",
        ha="center",
        va="center",
        fontsize=22,
        color=PURPLE,
        weight="bold",
    )
    ax.text(
        0.50,
        0.275,
        r"$z=\arg\min_z\,(x-f_\theta(z))^T\Sigma^{-1}(x-f_\theta(z))$",
        ha="center",
        va="center",
        fontsize=14.2,
        color=INK,
    )
    ax.text(
        0.50,
        0.15,
        "The reconstruction criterion is detector-noise likelihood geometry.",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
    )


def draw_manifold_advantage(ax) -> None:
    panel(
        ax,
        "3. Why stronger than EMPCA?",
        "Linear plane -> curved noise-aware signal manifold.",
    )
    # Linear EMPCA plane, placed in the lower half so the explanatory bullets
    # remain readable.
    px = np.array([0.10, 0.42, 0.48, 0.16, 0.10])
    py = np.array([0.13, 0.09, 0.30, 0.36, 0.13])
    ax.fill(px, py, color=GREEN, alpha=0.15, edgecolor=GREEN, linewidth=2)
    ax.text(0.15, 0.40, "EMPCA", color=GREEN, fontsize=12, weight="bold")
    ax.text(0.15, 0.36, "linear weighted plane", color=MUTED, fontsize=9.5)

    # Nonlinear manifold
    t = np.linspace(0, 1, 180)
    x = 0.52 + 0.34 * t
    y = 0.10 + 0.19 * t + 0.055 * np.sin(2 * np.pi * t)
    ax.plot(x, y, color=PURPLE, linewidth=3.0)
    ax.fill_between(x, y - 0.035, y + 0.035, color=PURPLE, alpha=0.10)
    ax.scatter(x[::25], y[::25], color=PURPLE, s=18, zorder=4)
    ax.text(0.58, 0.40, "NFPA", color=PURPLE, fontsize=12, weight="bold")
    ax.text(0.58, 0.36, "nonlinear noise-aware manifold", color=MUTED, fontsize=9.5)

    arrow(ax, (0.46, 0.23), (0.54, 0.23), color=MUTED, lw=1.5)
    ax.text(0.50, 0.27, "generalizes", ha="center", fontsize=9.5, color=MUTED)

    items = [
        "nonlinear pulse-shape variation",
        "position-dependent channel sharing",
        "timing shifts",
        "amplitude-shape coupling",
        "detector response curvature",
    ]
    y0 = 0.82
    ax.text(0.08, y0, "NFPA can represent:", fontsize=12, weight="bold", color=INK, va="top")
    for i, item in enumerate(items):
        y_item = y0 - 0.055 * (i + 1)
        ax.scatter([0.10], [y_item], s=24, color=PURPLE)
        ax.text(0.14, y_item, item, fontsize=10.0, color=INK, va="center")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG, "savefig.facecolor": BG})
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.86, bottom=0.08, wspace=0.04)

    draw_geometry_hierarchy(axes[0])
    draw_nfpa_map(axes[1])
    draw_manifold_advantage(axes[2])

    fig.suptitle(
        "NFPA Learns a Nonlinear Noise-Aware Signal Manifold",
        fontsize=24,
        weight="bold",
        color=INK,
        y=0.96,
    )
    out_png = OUT_DIR / "08_nfpa_nonlinear_noise_aware_manifold.png"
    out_pdf = OUT_DIR / "08_nfpa_nonlinear_noise_aware_manifold.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_png.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
