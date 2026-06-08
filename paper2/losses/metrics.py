"""Evaluation metrics for Paper 2 reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paper2._torch import Tensor, torch
from paper2.data.whitening import WhiteningOperator


@dataclass(slots=True)
class ReconstructionMetrics:
    weighted_residual_mean: float
    reconstruction_mse: float
    amplitude_rmse: float | None = None
    timing_rmse: float | None = None
    position_rmse: float | None = None
    shape_rmse: float | None = None


def weighted_residual_mean(x: Tensor, x_hat: Tensor, whitener: WhiteningOperator) -> Tensor:
    return whitener.mahalanobis_energy(x_hat - x).mean()


def reconstruction_mse(x: Tensor, x_hat: Tensor) -> Tensor:
    return torch.mean((x_hat - x) ** 2)


def summarize_reconstruction_metrics(
    x: Tensor,
    x_hat: Tensor,
    whitener: WhiteningOperator,
    extra: dict[str, Any] | None = None,
) -> ReconstructionMetrics:
    extra = {} if extra is None else extra
    return ReconstructionMetrics(
        weighted_residual_mean=float(weighted_residual_mean(x, x_hat, whitener).detach().cpu()),
        reconstruction_mse=float(reconstruction_mse(x, x_hat).detach().cpu()),
        amplitude_rmse=extra.get("amplitude_rmse"),
        timing_rmse=extra.get("timing_rmse"),
        position_rmse=extra.get("position_rmse"),
        shape_rmse=extra.get("shape_rmse"),
    )
