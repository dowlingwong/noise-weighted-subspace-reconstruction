"""Generate a slide-ready NFPA factorization schematic.

The figure is intentionally schematic rather than data-driven. It illustrates
the inductive bias: NFPA constrains each basis element to be a separable
channel mode times a time mode, unlike an unconstrained EMPCA basis vector.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
TALK_DIR = REPO_ROOT / "paper2" / "results" / "_analysis" / "latest" / "figures" / "talk_deck_selected"


NAVY = "#12355B"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#D95D39"
INK = "#202124"
MUTED = "#6B7280"
PANEL_EDGE = "#D7DBE2"


def _add_panel_frame(ax, label: str, title: str, subtitle: str | None = None) -> None:
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=PANEL_EDGE,
            linewidth=1.5,
            zorder=-10,
        )
    )
    ax.text(0.035, 0.94, label, transform=ax.transAxes, fontsize=16, weight="bold", color=NAVY, va="top")
    ax.text(0.13, 0.94, title, transform=ax.transAxes, fontsize=16, weight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.13, 0.875, subtitle, transform=ax.transAxes, fontsize=11, color=MUTED, va="top")


def _inset(parent, rect: tuple[float, float, float, float]):
    fig = parent.figure
    bbox = parent.get_position()
    x, y, w, h = rect
    return fig.add_axes(
        [
            bbox.x0 + x * bbox.width,
            bbox.y0 + y * bbox.height,
            w * bbox.width,
            h * bbox.height,
        ]
    )


def _mixed_empca_basis(n_channels: int = 8, n_time: int = 96) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 1.0, n_time)
    data = np.zeros((n_channels, n_time))
    for ch in range(n_channels):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = rng.choice([1.5, 2.0, 3.0, 4.5])
        blob = np.exp(-0.5 * ((t - rng.uniform(0.25, 0.78)) / rng.uniform(0.045, 0.11)) ** 2)
        ripple = np.sin(2.0 * np.pi * freq * t + phase)
        data[ch] = rng.normal(0.0, 0.08, n_time) + rng.uniform(-1.0, 1.0) * blob + 0.25 * ripple
    data -= data.mean()
    data /= np.max(np.abs(data))
    return data


def _pulse_mode(n_time: int = 96) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_time)
    rise = 1.0 - np.exp(-t / 0.045)
    decay = np.exp(-t / 0.32)
    tail = 0.13 * np.exp(-0.5 * ((t - 0.72) / 0.13) ** 2)
    pulse = rise * decay + tail
    pulse -= pulse.min()
    pulse /= pulse.max()
    return pulse


def _style_heatmap(ax, n_channels: int, n_time: int, xlabel: bool = True) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("time", fontsize=10 if xlabel else 0, labelpad=2, color=MUTED)
    ax.set_ylabel("channels", fontsize=10, labelpad=3, color=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#C5CBD3")
        spine.set_linewidth(0.9)


def _panel_a(ax) -> None:
    _add_panel_frame(ax, "A", "EMPCA basis vector $p_i$", "arbitrary channel-time pattern")
    hax = _inset(ax, (0.12, 0.21, 0.76, 0.50))
    data = _mixed_empca_basis()
    hax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-1.0, vmax=1.0, interpolation="nearest")
    _style_heatmap(hax, data.shape[0], data.shape[1])
    ax.text(
        0.50,
        0.11,
        "one free pattern across all channels and all time samples",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color=MUTED,
    )


def _panel_b(ax) -> None:
    _add_panel_frame(ax, "B", "NFPA factorization", "basis constrained to be separable")
    a = np.array([0.18, 0.42, 0.92, 0.74, 0.55, 0.31, 0.12])
    b = _pulse_mode()
    outer = np.outer(a, b)

    avec_ax = _inset(ax, (0.06, 0.29, 0.14, 0.40))
    avec_ax.imshow(a[:, None], cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    avec_ax.set_xticks([])
    avec_ax.set_yticks([])
    avec_ax.set_ylabel("$a_i$", fontsize=13, color=INK, rotation=0, labelpad=13, va="center")
    for spine in avec_ax.spines.values():
        spine.set_color("#C5CBD3")

    bax = _inset(ax, (0.32, 0.38, 0.24, 0.22))
    bax.plot(b, color=AMBER, linewidth=3)
    bax.fill_between(np.arange(b.size), b, 0, color=AMBER, alpha=0.18)
    bax.set_xticks([])
    bax.set_yticks([])
    bax.set_title("$b_i^T$", fontsize=13, color=INK, pad=0)
    for spine in bax.spines.values():
        spine.set_visible(False)

    hax = _inset(ax, (0.70, 0.25, 0.24, 0.46))
    hax.imshow(outer, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    _style_heatmap(hax, outer.shape[0], outer.shape[1], xlabel=False)

    ax.text(0.255, 0.49, "$\\times$", transform=ax.transAxes, ha="center", va="center", fontsize=28, color=INK)
    ax.add_patch(
        FancyArrowPatch(
            (0.59, 0.49),
            (0.68, 0.49),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color=MUTED,
        )
    )
    ax.text(
        0.50,
        0.12,
        "NFPA basis = channel mode $\\times$ time mode",
        transform=ax.transAxes,
        ha="center",
        fontsize=13,
        color=INK,
        weight="bold",
    )


def _draw_detector(ax, amplitudes: np.ndarray) -> None:
    positions = np.array(
        [
            [0.20, 0.50],
            [0.36, 0.65],
            [0.36, 0.35],
            [0.52, 0.50],
            [0.68, 0.65],
            [0.68, 0.35],
            [0.84, 0.50],
        ]
    )
    for (x, y), amp in zip(positions, amplitudes):
        radius = 0.055 + 0.045 * amp
        ax.add_patch(Circle((x, y), radius, facecolor=TEAL, alpha=0.25 + 0.55 * amp, edgecolor=NAVY, linewidth=1.2))
        ax.add_patch(Circle((x, y), 0.014, facecolor=NAVY, edgecolor="none", alpha=0.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel_c(ax) -> None:
    _add_panel_frame(ax, "C", "Detector interpretation", "spatial sharing and pulse evolution")
    a = np.array([0.18, 0.42, 0.92, 0.74, 0.55, 0.31, 0.12])
    b = _pulse_mode()
    outer = np.outer(a, b)

    det_ax = _inset(ax, (0.06, 0.37, 0.27, 0.28))
    _draw_detector(det_ax, a)
    det_ax.set_title("channel sharing", fontsize=11, color=INK, pad=1)

    pulse_ax = _inset(ax, (0.38, 0.40, 0.24, 0.22))
    pulse_ax.plot(b, color=CORAL, linewidth=3)
    pulse_ax.fill_between(np.arange(b.size), b, 0, color=CORAL, alpha=0.16)
    pulse_ax.set_title("pulse shape", fontsize=11, color=INK, pad=1)
    pulse_ax.set_xticks([])
    pulse_ax.set_yticks([])
    for spine in pulse_ax.spines.values():
        spine.set_visible(False)

    hax = _inset(ax, (0.72, 0.27, 0.22, 0.42))
    hax.imshow(outer, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    _style_heatmap(hax, outer.shape[0], outer.shape[1], xlabel=False)

    ax.add_patch(
        FancyArrowPatch(
            (0.64, 0.50),
            (0.70, 0.50),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color=MUTED,
        )
    )
    ax.text(
        0.50,
        0.12,
        "NFPA learns spatial sharing $\\times$ pulse evolution",
        transform=ax.transAxes,
        ha="center",
        fontsize=13,
        color=INK,
        weight="bold",
    )


def main() -> int:
    TALK_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3), constrained_layout=False)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.92, bottom=0.08, wspace=0.035)
    _panel_a(axes[0])
    _panel_b(axes[1])
    _panel_c(axes[2])

    fig.suptitle(
        "NFPA inductive bias: separable channel-time structure",
        fontsize=22,
        weight="bold",
        color=INK,
        y=0.985,
    )

    for suffix in ("png", "pdf", "svg"):
        output_path = TALK_DIR / f"s15_nfpa_factorization_schematic.{suffix}"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {TALK_DIR / 's15_nfpa_factorization_schematic.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
