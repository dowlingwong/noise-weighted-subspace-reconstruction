"""Reconstruction losses for native-space outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from paper2._torch import Tensor, nn, require_torch, torch
from paper2.data.datasets import ReconstructionBatch
from paper2.data.whitening import WhiteningOperator
from paper2.models.base import ReconstructionOutput


LossMode = Literal["mse", "mahalanobis"]


@dataclass(slots=True)
class LossOutput:
    total: Tensor
    terms: dict[str, Tensor]


def mse_raw(x: Tensor, x_hat: Tensor) -> Tensor:
    return torch.mean((x_hat - x) ** 2)


def mahalanobis_raw(
    x: Tensor,
    x_hat: Tensor,
    whitener: WhiteningOperator,
) -> Tensor:
    residual = x_hat - x
    return whitener.mahalanobis_energy(residual).mean()


class ReconstructionCriterion(nn.Module):
    """Shared criterion for AE and transformer reconstruction."""

    def __init__(self, loss_mode: LossMode, scale: float = 1.0) -> None:
        require_torch()
        super().__init__()
        self.loss_mode = loss_mode
        self.scale = float(scale)

    def forward(
        self,
        output: ReconstructionOutput,
        batch: ReconstructionBatch,
        whitener: WhiteningOperator,
    ) -> LossOutput:
        if self.loss_mode == "mse":
            total = mse_raw(batch.x, output.x_hat)
        elif self.loss_mode == "mahalanobis":
            total = mahalanobis_raw(batch.x, output.x_hat, whitener)
        else:
            raise ValueError(f"Unsupported loss mode: {self.loss_mode}")
        scaled_total = total * self.scale
        return LossOutput(
            total=scaled_total,
            terms={
                "reconstruction": total,
                "scale": torch.as_tensor(self.scale, device=total.device, dtype=total.dtype),
            },
        )
