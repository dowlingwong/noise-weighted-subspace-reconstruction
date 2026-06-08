"""Offline evaluation entry point for trained Paper 2 models."""

from __future__ import annotations

from pathlib import Path

from paper2._torch import require_torch, torch
from paper2.trainers.train_reconstruction import (
    build_criterion,
    build_dataloaders,
    build_model,
    build_whitener,
    load_experiment_config,
)
from paper2.trainers.loops import evaluate


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, float]:
    require_torch()
    cfg = load_experiment_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whitener = build_whitener(cfg).to(device)
    model = build_model(cfg, whitener).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    criterion = build_criterion(cfg)
    loaders = build_dataloaders(cfg)
    return evaluate(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        whitener=whitener,
        device=device,
    )
