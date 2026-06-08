"""Experiment runner entry point for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from paper2._torch import require_torch, torch
from paper2.data.datasets import (
    DatasetConfig,
    collate_reconstruction_batches,
    load_reconstruction_dataset,
)
from paper2.data.splits import make_split_indices
from paper2.data.whitening import WhiteningConfig, WhiteningOperator
from paper2.losses.reconstruction import ReconstructionCriterion
from paper2.models.linear_autoencoder import LinearAutoencoderConfig, PatchLinearAutoencoder
from paper2.models.reconstruction_ae import ReconstructionAE, ReconstructionAEConfig
from paper2.models.transformer_reconstruction import (
    TransformerReconstructionConfig,
    TransformerReconstructionModel,
)
from paper2.trainers.loops import evaluate, train_one_epoch


@dataclass(slots=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @property
    def experiment_name(self) -> str:
        return str(self.raw["experiment"]["name"])


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ExperimentConfig(raw=yaml.safe_load(handle))


def build_whitener(cfg: ExperimentConfig) -> WhiteningOperator:
    preprocess = cfg.raw["preprocessing"]
    data = cfg.raw["data"]
    return WhiteningOperator(
        WhiteningConfig(
            psd_path=preprocess["psd_path"],
            trace_len=int(data["trace_len"]),
            n_channels=int(data["n_channels"]),
        )
    )


def build_model(cfg: ExperimentConfig, whitener: WhiteningOperator):
    model_cfg = cfg.raw["model"]
    data_cfg = cfg.raw["data"]
    shared = {
        "trace_len": int(data_cfg["trace_len"]),
        "n_channels": int(data_cfg["n_channels"]),
        "latent_dim": int(model_cfg["latent_dim"]),
        "input_mode": cfg.raw["preprocessing"]["input_mode"],
    }
    family = model_cfg["family"]
    if family == "linear_ae":
        return PatchLinearAutoencoder(
            LinearAutoencoderConfig(
                **shared,
                patch_len=int(model_cfg.get("patch_len", 128)),
                patch_stride=int(model_cfg.get("patch_stride", 128)),
            ),
            whitener=whitener,
        )
    if family == "ae":
        return ReconstructionAE(
            ReconstructionAEConfig(
                **shared,
                hidden_channels=tuple(model_cfg.get("hidden_channels", [32, 64, 128])),
                kernel_size=int(model_cfg.get("kernel_size", 7)),
                stride=int(model_cfg.get("stride", 2)),
            ),
            whitener=whitener,
        )
    if family == "cnn_ae":
        return ReconstructionAE(
            ReconstructionAEConfig(
                **shared,
                hidden_channels=tuple(model_cfg.get("hidden_channels", [32, 64, 128])),
                kernel_size=int(model_cfg.get("kernel_size", 7)),
                stride=int(model_cfg.get("stride", 2)),
            ),
            whitener=whitener,
        )
    if family == "transformer":
        return TransformerReconstructionModel(
            TransformerReconstructionConfig(
                **shared,
                patch_len=int(model_cfg["patch_len"]),
                patch_stride=int(model_cfg["patch_stride"]),
                d_model=int(model_cfg["d_model"]),
                d_ff=int(model_cfg["d_ff"]),
                n_head=int(model_cfg["n_head"]),
                n_temporal_blocks=int(model_cfg["n_temporal_blocks"]),
                dropout=float(model_cfg.get("dropout", 0.1)),
            ),
            whitener=whitener,
        )
    raise ValueError(f"Unsupported model family: {family}")


def build_criterion(cfg: ExperimentConfig) -> ReconstructionCriterion:
    return ReconstructionCriterion(loss_mode=cfg.raw["loss"]["mode"])


def build_optimizer(cfg: ExperimentConfig, model) -> Any:
    require_torch()
    optim_cfg = cfg.raw["optimizer"]
    name = optim_cfg["name"].lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optim_cfg['name']}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )


def build_dataloaders(cfg: ExperimentConfig):
    require_torch()
    dataset_cfg = DatasetConfig(**cfg.raw["data"])
    dataset = load_reconstruction_dataset(dataset_cfg)
    split = make_split_indices(
        n_samples=len(dataset),
        train_fraction=dataset_cfg.train_fraction,
        val_fraction=dataset_cfg.val_fraction,
        test_fraction=dataset_cfg.test_fraction,
        seed=dataset_cfg.seed,
    )
    train_ds = torch.utils.data.Subset(dataset, split.train.tolist())
    val_ds = torch.utils.data.Subset(dataset, split.val.tolist())
    test_ds = torch.utils.data.Subset(dataset, split.test.tolist())
    loader_kwargs = {
        "batch_size": dataset_cfg.batch_size,
        "num_workers": dataset_cfg.num_workers,
        "pin_memory": dataset_cfg.pin_memory,
        "collate_fn": collate_reconstruction_batches,
    }
    return {
        "train": torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs),
        "val": torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs),
        "test": torch.utils.data.DataLoader(test_ds, shuffle=False, **loader_kwargs),
        "split": split,
        "dataset": dataset,
    }


def make_output_dir(cfg: ExperimentConfig) -> Path:
    path = Path("paper2") / "results" / cfg.experiment_name
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def save_run_artifacts(
    output_dir: Path,
    cfg: ExperimentConfig,
    history: list[dict[str, float]],
    best_metrics: dict[str, float],
) -> None:
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.raw, handle, sort_keys=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2, sort_keys=True)
    with (output_dir / "curves.csv").open("w", encoding="utf-8") as handle:
        if history:
            columns = list(history[0].keys())
            handle.write(",".join(columns) + "\n")
            for row in history:
                handle.write(",".join(str(row[col]) for col in columns) + "\n")


def run_experiment(config_path: str | Path) -> None:
    require_torch()
    cfg = load_experiment_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whitener = build_whitener(cfg).to(device)
    model = build_model(cfg, whitener).to(device)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    loaders = build_dataloaders(cfg)
    output_dir = make_output_dir(cfg)

    n_epochs = int(cfg.raw["training"]["epochs"])
    patience = int(cfg.raw["training"].get("early_stop_patience", 10))
    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] | None = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            whitener=whitener,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            whitener=whitener,
            device=device,
        )
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)

        if val_metrics["eval_loss"] < best_val:
            best_val = val_metrics["eval_loss"]
            stale = 0
            best_metrics = {
                **row,
                **evaluate(
                    model=model,
                    loader=loaders["test"],
                    criterion=criterion,
                    whitener=whitener,
                    device=device,
                ),
            }
            torch.save(model.state_dict(), output_dir / "checkpoint_best.pt")
        else:
            stale += 1
            if stale >= patience:
                break

    if best_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics.")
    save_run_artifacts(output_dir, cfg, history, best_metrics)
