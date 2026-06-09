"""Training and evaluation loops for Paper 2."""

from __future__ import annotations

from typing import Any

from paper2._torch import require_torch, torch
from paper2.data.datasets import ReconstructionBatch
from paper2.data.whitening import WhiteningOperator
from paper2.losses.reconstruction import ReconstructionCriterion
from paper2.losses.metrics import summarize_reconstruction_metrics
from paper2.models.base import BaseReconstructionModel


def _require_finite(name: str, value) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite tensor detected: {name}")


def train_one_epoch(
    model: BaseReconstructionModel,
    loader,
    criterion: ReconstructionCriterion,
    optimizer,
    whitener: WhiteningOperator,
    device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    require_torch()
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0
    for batch_idx, batch in enumerate(loader):
        meta = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.meta.items()
        }
        batch = ReconstructionBatch(x=batch.x.to(device), meta=meta)
        _require_finite(f"train batch {batch_idx} input", batch.x)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.x)
        _require_finite(f"train batch {batch_idx} output", output.x_hat)
        loss_out = criterion(output, batch, whitener)
        _require_finite(f"train batch {batch_idx} loss", loss_out.total)
        loss_out.total.backward()
        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
                error_if_nonfinite=True,
            )
            total_grad_norm += float(grad_norm.detach().cpu())
        optimizer.step()
        total_loss += float(loss_out.total.detach().cpu())
        n_batches += 1
    metrics = {"train_loss": total_loss / max(n_batches, 1)}
    if grad_clip is not None and grad_clip > 0:
        metrics["train_grad_norm"] = total_grad_norm / max(n_batches, 1)
    return metrics


@torch.no_grad() if torch is not None else (lambda fn: fn)
def evaluate(
    model: BaseReconstructionModel,
    loader,
    criterion: ReconstructionCriterion,
    whitener: WhiteningOperator,
    device,
) -> dict[str, float]:
    require_torch()
    model.eval()
    loss_total = 0.0
    weighted_total = 0.0
    mse_total = 0.0
    n_batches = 0
    for batch_idx, batch in enumerate(loader):
        meta = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.meta.items()
        }
        batch = ReconstructionBatch(x=batch.x.to(device), meta=meta)
        _require_finite(f"eval batch {batch_idx} input", batch.x)
        output = model(batch.x)
        _require_finite(f"eval batch {batch_idx} output", output.x_hat)
        loss_out = criterion(output, batch, whitener)
        _require_finite(f"eval batch {batch_idx} loss", loss_out.total)
        metrics = summarize_reconstruction_metrics(batch.x, output.x_hat, whitener)
        loss_total += float(loss_out.total.detach().cpu())
        weighted_total += metrics.weighted_residual_mean
        mse_total += metrics.reconstruction_mse
        n_batches += 1
    denom = max(n_batches, 1)
    return {
        "eval_loss": loss_total / denom,
        "weighted_residual_mean": weighted_total / denom,
        "reconstruction_mse": mse_total / denom,
    }
