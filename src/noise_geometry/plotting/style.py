"""Small plotting utilities with lazy matplotlib import."""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, path: str | Path, *, dpi: int = 180):
    """Save a matplotlib figure, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return output
