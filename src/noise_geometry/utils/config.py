"""Config-file loading for experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict for ``None``."""
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")
    data.setdefault("_config_path", str(config_path))
    return data
