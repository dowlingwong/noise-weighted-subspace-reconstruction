"""Experiment runner entry point for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
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


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @property
    def experiment_name(self) -> str:
        return str(self.raw["experiment"]["name"])

    def resolve_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path


@dataclass(slots=True)
class OptimizerBundle:
    """Small adapter for runs that use both AdamW and Muon."""

    optimizers: dict[str, Any]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            name: optimizer.state_dict()
            for name, optimizer in self.optimizers.items()
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ExperimentConfig(raw=yaml.safe_load(handle))


def build_whitener(cfg: ExperimentConfig) -> WhiteningOperator:
    preprocess = cfg.raw["preprocessing"]
    data = cfg.raw["data"]
    return WhiteningOperator(
        WhiteningConfig(
            psd_path=str(cfg.resolve_path(preprocess["psd_path"])),
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
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(optim_cfg["lr"]),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
            eps=float(optim_cfg.get("eps", 1e-8)),
        )
    if name in {"adamw_muon", "muon"}:
        return build_adamw_muon_optimizer(optim_cfg, model)
    raise ValueError(f"Unsupported optimizer: {optim_cfg['name']}")


def build_adamw_muon_optimizer(optim_cfg: dict[str, Any], model) -> OptimizerBundle:
    from paper2.trainers.muon import Muon

    adamw_params = []
    muon_params = []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_name.startswith("encoder.temporal_encoder.") and param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    if not muon_params:
        raise ValueError("Muon requested, but no transformer matrix parameters were found.")
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=float(optim_cfg.get("adamw_lr", optim_cfg.get("lr", 3e-5))),
        weight_decay=float(
            optim_cfg.get("adamw_weight_decay", optim_cfg.get("weight_decay", 0.0))
        ),
        eps=float(optim_cfg.get("eps", 1e-8)),
    )
    muon = Muon(
        muon_params,
        lr=float(optim_cfg.get("muon_lr", optim_cfg.get("lr", 3e-5))),
        weight_decay=float(optim_cfg.get("muon_weight_decay", 0.0)),
        momentum=float(optim_cfg.get("muon_momentum", 0.95)),
        nesterov=bool(optim_cfg.get("nesterov", True)),
        ns_steps=int(optim_cfg.get("ns_steps", 5)),
    )
    return OptimizerBundle({"adamw": adamw, "muon": muon})


def _as_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def init_wandb(cfg: ExperimentConfig, output_dir: Path, model) -> Any | None:
    wandb_cfg = cfg.raw.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - server dependency check
        raise RuntimeError("W&B logging requested, but `wandb` is not installed.") from exc

    init_kwargs: dict[str, Any] = {
        "project": wandb_cfg.get("project", "NPML_Paper2"),
        "name": wandb_cfg.get("name", cfg.experiment_name),
        "group": wandb_cfg.get("group"),
        "tags": _as_tags(wandb_cfg.get("tags")),
        "config": cfg.raw,
        "dir": str(output_dir),
        "mode": wandb_cfg.get("mode", "online"),
    }
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    run = wandb.init(**init_kwargs)
    if bool(wandb_cfg.get("watch_model", False)):
        wandb.watch(
            model,
            log=wandb_cfg.get("watch_log", "gradients"),
            log_freq=int(wandb_cfg.get("watch_log_freq", 100)),
        )
    return run


def log_wandb_epoch(
    wandb_run: Any | None,
    epoch: int,
    row: dict[str, float],
    best_val: float,
    stale: int,
) -> None:
    if wandb_run is None:
        return
    payload = {
        key: value
        for key, value in row.items()
        if isinstance(value, (int, float))
    }
    payload["best_val"] = best_val
    payload["stale_epochs"] = stale
    wandb_run.log(payload, step=epoch)


def finish_wandb(wandb_run: Any | None, exit_code: int) -> None:
    if wandb_run is not None:
        try:
            wandb_run.finish(exit_code=exit_code)
        except TypeError:
            wandb_run.finish()


def build_dataloaders(cfg: ExperimentConfig):
    require_torch()
    data_cfg = dict(cfg.raw["data"])
    data_cfg["trace_path"] = str(cfg.resolve_path(data_cfg["trace_path"]))
    if data_cfg.get("rq_path") is not None:
        data_cfg["rq_path"] = str(cfg.resolve_path(data_cfg["rq_path"]))
    dataset_cfg = DatasetConfig(**data_cfg)
    dataset = load_reconstruction_dataset(dataset_cfg)
    split = make_split_indices(
        n_samples=len(dataset),
        train_fraction=dataset_cfg.train_fraction,
        val_fraction=dataset_cfg.val_fraction,
        test_fraction=dataset_cfg.test_fraction,
        seed=dataset_cfg.seed,
    )
    train_indices, coverage_summary = apply_train_coverage_filter(
        dataset=dataset,
        train_indices=split.train,
        cfg=dataset_cfg,
    )
    train_ds = torch.utils.data.Subset(dataset, train_indices.tolist())
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
        "coverage_summary": coverage_summary,
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
    coverage_summary: dict[str, Any] | None = None,
) -> None:
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.raw, handle, sort_keys=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2, sort_keys=True)
    if coverage_summary is not None:
        with (output_dir / "coverage_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(coverage_summary, handle, indent=2, sort_keys=True)
    with (output_dir / "curves.csv").open("w", encoding="utf-8") as handle:
        if history:
            columns = list(history[0].keys())
            handle.write(",".join(columns) + "\n")
            for row in history:
                handle.write(",".join(str(row[col]) for col in columns) + "\n")


def apply_train_coverage_filter(
    dataset,
    train_indices,
    cfg: DatasetConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    fields = list(cfg.coverage_fields or ["amplitude", "t0"])
    summary: dict[str, Any] = {
        "coverage_mode": cfg.coverage_mode,
        "coverage_fields": fields,
        "coverage_quantile_low": float(cfg.coverage_quantile_low),
        "coverage_quantile_high": float(cfg.coverage_quantile_high),
        "train_n_before": int(train_indices.size),
    }
    if cfg.coverage_mode == "full":
        summary["train_n_after"] = int(train_indices.size)
        return train_indices, summary
    if cfg.coverage_mode != "restricted":
        raise ValueError(f"Unsupported coverage_mode: {cfg.coverage_mode}")

    metadata = getattr(dataset, "metadata", {})
    mask = np.ones(train_indices.shape[0], dtype=bool)
    per_field: dict[str, dict[str, float]] = {}
    q_low = float(cfg.coverage_quantile_low)
    q_high = float(cfg.coverage_quantile_high)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            "coverage quantiles must satisfy 0 <= low < high <= 1; "
            f"got low={q_low}, high={q_high}"
        )

    for field in fields:
        if field not in metadata:
            raise KeyError(
                f"Coverage restriction requires metadata field '{field}', "
                f"but dataset only has: {sorted(metadata)}"
            )
        values = np.asarray(metadata[field])[train_indices]
        low_value = float(np.quantile(values, q_low))
        high_value = float(np.quantile(values, q_high))
        field_mask = (values >= low_value) & (values <= high_value)
        mask &= field_mask
        per_field[field] = {
            "quantile_low_value": low_value,
            "quantile_high_value": high_value,
            "mean_before": float(np.mean(values)),
            "std_before": float(np.std(values)),
        }

    filtered = train_indices[mask]
    if filtered.size == 0:
        raise RuntimeError("Coverage restriction removed every training example.")
    summary["fields"] = per_field
    summary["train_n_after"] = int(filtered.size)
    summary["train_fraction_retained"] = float(filtered.size / max(train_indices.size, 1))
    return filtered, summary


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
    grad_clip = cfg.raw["training"].get("grad_clip")
    grad_clip = None if grad_clip is None else float(grad_clip)
    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] | None = None
    best_val = float("inf")
    stale = 0

    print(
        f"[paper2] {cfg.experiment_name}: device={device}, "
        f"epochs={n_epochs}, output={output_dir}",
        flush=True,
    )
    wandb_run = init_wandb(cfg, output_dir, model)
    completed = False
    try:
        for epoch in range(1, n_epochs + 1):
            should_stop = False
            train_metrics = train_one_epoch(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                whitener=whitener,
                device=device,
                grad_clip=grad_clip,
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
                    should_stop = True
            print(
                "[paper2] "
                f"{cfg.experiment_name} epoch={epoch:03d} "
                f"train_loss={train_metrics['train_loss']:.6g} "
                f"val_loss={val_metrics['eval_loss']:.6g} "
                f"val_weighted={val_metrics['weighted_residual_mean']:.6g} "
                f"val_mse={val_metrics['reconstruction_mse']:.6g} "
                f"best_val={best_val:.6g} stale={stale}",
                flush=True,
            )
            log_wandb_epoch(wandb_run, epoch, row, best_val, stale)
            if should_stop:
                break

        if best_metrics is None:
            raise RuntimeError("Training finished without producing validation metrics.")
        save_run_artifacts(
            output_dir,
            cfg,
            history,
            best_metrics,
            coverage_summary=loaders.get("coverage_summary"),
        )
        if wandb_run is not None:
            wandb_run.summary["best_val"] = best_val
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    wandb_run.summary[f"best/{key}"] = value
        completed = True
    finally:
        finish_wandb(wandb_run, exit_code=0 if completed else 1)
    print(f"[paper2] {cfg.experiment_name}: complete best_val={best_val:.6g}", flush=True)
