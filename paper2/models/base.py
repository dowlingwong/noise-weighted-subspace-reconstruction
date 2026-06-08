"""Shared model interfaces for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from paper2._torch import Tensor, nn


InputMode = Literal["raw", "prewhitened"]


@dataclass(slots=True)
class ReconstructionOutput:
    x_hat: Tensor
    z: Tensor | None = None
    x_enc: Tensor | None = None
    aux: dict[str, Tensor | Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelConfig:
    trace_len: int
    n_channels: int
    latent_dim: int
    input_mode: InputMode


class BaseReconstructionModel(nn.Module):
    """Base class for native-output reconstruction models."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, x_native: Tensor) -> ReconstructionOutput:  # pragma: no cover - interface only
        raise NotImplementedError

    def encode(self, x_native: Tensor) -> Tensor | None:
        out = self.forward(x_native)
        return out.z
