"""Path policy for remote-server public data and local outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = Path("/ceph/dwong/paper1_dataset")


def resolve_data_root(
    cli_data_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    env_var: str = "PAPER1_DATA_ROOT",
) -> Path:
    """Resolve the public-data root.

    Precedence is CLI value, then environment variable, then config
    ``data_root``, then the remote-server default.
    """
    if cli_data_root:
        return Path(cli_data_root).expanduser().resolve()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    if config and config.get("data_root"):
        return Path(config["data_root"]).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def dataset_root(dataset: str, data_root: str | Path | None = None) -> Path:
    """Return the canonical directory for one public dataset."""
    if not dataset:
        raise ValueError("dataset name is required")
    root = Path(data_root).expanduser().resolve() if data_root else DEFAULT_DATA_ROOT
    return root / dataset.lower()


def ensure_dataset_layout(dataset: str, data_root: str | Path | None = None) -> Path:
    """Create the canonical dataset/cache/processed directories."""
    root = dataset_root(dataset, data_root)
    for subdir in ("raw", "cache", "processed"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def is_within_repo(path: str | Path, repo_root: str | Path) -> bool:
    """Return true if ``path`` is inside ``repo_root`` after resolution."""
    p = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False
