"""Optional torch compatibility layer for the Paper 2 scaffold.

The current repository does not guarantee that `torch` is installed. These
helpers let the scaffold import cleanly before the actual training stack is
enabled.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on local environment
    import torch
    import torch.nn as nn
    from torch import Tensor
except Exception:  # pragma: no cover - graceful fallback before install
    torch = None
    Tensor = Any

    class _Module:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _NNNamespace:
        Module = _Module

    nn = _NNNamespace()


def require_torch() -> None:
    if torch is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "paper2 training code requires `torch`, which is not installed in "
            "the current environment."
        )
