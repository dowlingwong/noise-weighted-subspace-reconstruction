"""
Training loop / logging script. Note that the configurations are messy
right now to be later fixed with the hydra package or raw JSON files.
"""
from __future__ import annotations 

import time
import math
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pathlib
from pathlib import Path
# source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh

import wandb
import torch
import torch.nn.functional as F
import numpy as np

from reconstruction.data.dataset import (
    DataConfig,
    create_dataloaders,
)
from reconstruction.models.model_transformer_mamba import TransformerConfig, Transformer
from reconstruction.utils import count_model_params, infinite_dataloader, generate_timestamp, get_next_batch
from reconstruction.training.checkpoints import save_checkpoint
from reconstruction.training.schedulers import cosine_scheduler_with_linear_warmup
from reconstruction.evaluation.model_evaluation import eval_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_BASE_CHECKPOINT_PATH = _PROJECT_ROOT / "training_checkpoints"
_BASE_CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)


@dataclass
class TrainingConfig:
    # General training config
    # num_steps: int = 10000
    num_steps: int = 1000
    num_eval_steps: int = 500
    # num_eval_steps: int = 1200
    # num_eval_steps: int = 3125
    # eval_step_period: int = 1000
    eval_step_period: int = 100
    # save_checkpoint_period: int = 2500
    save_checkpoint_period: int = 250
    total_batch_size: int = 64
    device_batch_size: int = 2
    num_workers: int = 0
    precompute_normalisers: bool = False
    scalar_loss_weights: tuple[float] = (1.0, 1.0, 1.0)
    recoil_classification: bool = False
    # grad_clip: float = 10000000.0
    grad_clip: float = 10000.0
    checkpoint_dir: str | Path = _BASE_CHECKPOINT_PATH
    # Optimiser specific config
    adamw_lr: float = 1e-3
    adamw_betas: tuple[float] = (0.9, 0.999)
    # adamw_weight_decay: float = 4e-2
    adamw_weight_decay: float = 0.1
    adamw_fused: bool = True
    use_muon: bool = False
    muon_lr: float = 1e-3
    muon_momentum: float = 0.95
    muon_weight_decay = 0.1
    nesterov: bool = True
    ns_steps: int = 5
    # Scheduler specific config
    # 10% of the training steps
    adamw_warmup_steps: int = 100
    # adamw_warmup_steps: int = 1000
    muon_warmup_steps: int = 100
    # muon_warmup_steps: int = 1000
    # Weights and biases config
    wandb_run: bool = True
    wandb_run_name: str = field(
        default_factory=lambda: f"bimamba2_full_dim_batch64_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    wandb_project_name: str = "DELight_Reconstruction"

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_with_adamw_and_muon(
    model,
    adamw_optimiser,
    adamw_scheduler,
    muon_optimiser,
    muon_scheduler,
    train_generator,
    val_generator,
    device,
    config,
    target_normaliser=None,
):
    if config.wandb_run:
        wandb_run = wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_run_name,
            config=asdict(config),
        )
    total_training_time = 0.0
    grad_accum_steps = config.total_batch_size // config.device_batch_size
    train_inputs, train_spatial_targets, train_energy_targets, train_class_targets = get_next_batch(
        train_generator,
        device,
    )
    model.train()
    for step in range(config.num_steps + 1):
        step_start_time = time.perf_counter()
        final_step = step == config.num_steps
        total_forward_duration = 0.0

        for micro_step in range(grad_accum_steps):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                forward_start = time.perf_counter()
                train_spatial_logits, train_energy_logits, train_class_logits = model(train_inputs)
                # Wait for GPU to finish processing (blocks CPU)
                torch.cuda.synchronize()
                forward_end = time.perf_counter()
                total_forward_duration += forward_end - forward_start
                spatial_loss = F.mse_loss(
                    train_spatial_logits,
                    train_spatial_targets,
                    reduction="mean",
                )
                energy_loss = F.mse_loss(
                    train_energy_logits,
                    train_energy_targets.unsqueeze(-1),
                    reduction="mean",
                )
                class_loss = F.binary_cross_entropy_with_logits(
                    train_class_logits,
                    train_class_targets.unsqueeze(-1),
                    reduction="mean",
                )
                total_loss = (
                    config.scalar_loss_weights[0] * spatial_loss
                    + config.scalar_loss_weights[1] * energy_loss
                    + config.scalar_loss_weights[2] * class_loss
                )
                total_loss /= grad_accum_steps

            total_loss.backward()
            # Need to multiply back for logging
            train_total_loss = total_loss.detach().item() * grad_accum_steps
            train_inputs, train_spatial_targets, train_energy_targets, train_class_targets = get_next_batch(
                train_generator,
                device,
            )
        
        # Forward pass duration given by time per micro-step
        average_forward_duration = total_forward_duration / grad_accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )
        grad_norm_cpu = grad_norm.item()
        adamw_optimiser.step()
        muon_optimiser.step()
        adamw_scheduler.step()
        muon_scheduler.step()
        model.zero_grad(set_to_none=True)
        step_end_time = time.perf_counter()
        step_duration = step_end_time - step_start_time
        total_training_time += step_duration

        if final_step or step % config.eval_step_period == 0:
            val_metrics = eval_model(
                model,
                val_generator,
                device,
                config,
                target_normaliser=target_normaliser,
            )
            val_total_loss = val_metrics["val_total_loss"]
            val_total_spatial_loss = val_metrics["val_total_spatial_loss"]
            val_total_energy_loss = val_metrics["val_total_energy_loss"]
            val_total_class_loss = val_metrics["val_total_class_loss"]
            val_spatial_rmse = val_metrics["val_spatial_rmse"]
            val_energy_rmse = val_metrics["val_energy_rmse"]
            val_accuracy = val_metrics["val_accuracy"]
            logger.info(
                f"Step: {step:d} |"
                f" Training loss: {train_total_loss:.4f} |"
                f" Validation loss: {val_total_loss:.4f} |"
                f" Validation spatial loss: {val_total_spatial_loss:.4f} |"
                f" Validation energy loss: {val_total_energy_loss:.4f} |"
                f" Validation class loss: {val_total_class_loss:.4f} |"
                f" Grad norm: {grad_norm_cpu:.4f} |"
                f" Forward pass duration: {average_forward_duration:.2f} s |"
                f" Step duration: {step_duration:.2f} s"
            )
            if config.wandb_run:
                wandb_run.log(
                    {
                        "step": step,
                        "train_loss": train_total_loss,
                        "val_loss": val_total_loss,
                        "val_spatial_loss": val_total_spatial_loss,
                        "val_energy_loss": val_total_energy_loss,
                        "val_class_loss": val_total_class_loss,
                        "val_spatial_rmse": val_spatial_rmse,
                        "val_energy_rmse": val_energy_rmse,
                        "val_accuracy": val_accuracy,
                        "grad_norm": grad_norm_cpu,
                        "forward_pass_duration": average_forward_duration,
                        "step_duration": step_duration,
                        "total_training_time": total_training_time,
                    }
                )
        else:
            logger.info(
                f"Step: {step:d} |"
                f" Training loss: {train_total_loss:.4f} |"
                f" Grad norm: {grad_norm_cpu:.4f} |"
                f" Forward pass duration: {average_forward_duration:.2f} s |"
                f" Step duration: {step_duration:.2f} s"
            )
        if final_step or step % config.save_checkpoint_period == 0:
            save_checkpoint(
                config.checkpoint_dir,
                step,
                model.state_dict(),
                adamw_optimiser.state_dict(),
                muon_state_dict=muon_optimiser.state_dict(),
                timestamp=generate_timestamp(),
            )
    if config.wandb_run:
        wandb_run.finish()
    logger.info("Training finished!")


def train_with_adamw_only(
    model,
    adamw_optimiser,
    adamw_scheduler,
    train_generator,
    val_generator,
    device,
    config,
    target_normaliser=None,
    energy_normaliser=None,
):
    if config.wandb_run:
        wandb_run = wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_run_name,
            config=asdict(config),
        )
    total_training_time = 0.0
    grad_accum_steps = config.total_batch_size // config.device_batch_size
    train_inputs, train_spatial_targets, train_energy_targets, train_class_targets = get_next_batch(
        train_generator,
        device,
    )
    model.train()
    for step in range(config.num_steps + 1):
        step_start_time = time.perf_counter()
        final_step = step == config.num_steps
        total_forward_duration = 0.0

        for micro_step in range(grad_accum_steps):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                forward_start = time.perf_counter()
                train_spatial_logits, train_energy_logits, train_class_logits = model(train_inputs)
                # Wait for GPU to finish processing (blocks CPU)
                torch.cuda.synchronize()
                forward_end = time.perf_counter()
                total_forward_duration += forward_end - forward_start
                spatial_loss = F.mse_loss(
                    train_spatial_logits,
                    train_spatial_targets,
                    reduction="mean",
                )
                energy_loss = F.mse_loss(
                    train_energy_logits,
                    train_energy_targets.unsqueeze(-1),
                    reduction="mean",
                )
                class_loss = F.binary_cross_entropy_with_logits(
                    train_class_logits,
                    train_class_targets.unsqueeze(-1),
                    reduction="mean",
                )
                total_loss = (
                    config.scalar_loss_weights[0] * spatial_loss
                    + config.scalar_loss_weights[1] * energy_loss
                    + config.scalar_loss_weights[2] * class_loss
                )
                total_loss /= grad_accum_steps

            total_loss.backward()
            # Need to multiply back for logging
            train_total_loss = total_loss.detach().item() * grad_accum_steps
            train_inputs, train_spatial_targets, train_energy_targets, train_class_targets = get_next_batch(
                train_generator,
                device,
            )
        
        # Forward pass duration given by time per micro-step
        average_forward_duration = total_forward_duration / grad_accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )
        grad_norm_cpu = grad_norm.item()
        adamw_optimiser.step()
        adamw_scheduler.step()
        model.zero_grad(set_to_none=True)
        step_end_time = time.perf_counter()
        step_duration = step_end_time - step_start_time
        total_training_time += step_duration

        if final_step or step % config.eval_step_period == 0:
            val_metrics = eval_model(
                model,
                val_generator,
                device,
                config,
                target_normaliser=target_normaliser,
                # energy_normaliser=energy_normaliser,
            )
            val_total_loss = val_metrics["val_total_loss"]
            val_total_spatial_loss = val_metrics["val_total_spatial_loss"]
            val_total_energy_loss = val_metrics["val_total_energy_loss"]
            val_total_class_loss = val_metrics["val_total_class_loss"]
            val_spatial_rmse = val_metrics["val_spatial_rmse"]
            val_energy_rmse = val_metrics["val_energy_rmse"]
            val_accuracy = val_metrics["val_accuracy"]
            
            logger.info(
                f"Step: {step:d} |"
                f" Training loss: {train_total_loss:.4f} |"
                f" Validation loss: {val_total_loss:.4f} |"
                f" Validation spatial loss: {val_total_spatial_loss:.4f} |"
                f" Validation energy loss: {val_total_energy_loss:.4f} |"
                f" Validation class loss: {val_total_class_loss:.4f} |"
                f" Grad norm: {grad_norm_cpu:.4f} |"
                f" Forward pass duration: {average_forward_duration:.2f} s |"
                f" Step duration: {step_duration:.2f} s"
            )
            if config.wandb_run:
                wandb_run.log(
                    {
                        "step": step,
                        "train_loss": train_total_loss,
                        "val_loss": val_total_loss,
                        "val_spatial_loss": val_total_spatial_loss,
                        "val_energy_loss": val_total_energy_loss,
                        "val_class_loss": val_total_class_loss,
                        "val_spatial_rmse": val_spatial_rmse,
                        "val_energy_rmse": val_energy_rmse,
                        "val_accuracy": val_accuracy,
                        "grad_norm": grad_norm_cpu,
                        "forward_pass_duration": average_forward_duration,
                        "step_duration": step_duration,
                        "total_training_time": total_training_time,
                    }
                )
        else:
            logger.info(
                f"Step: {step:d} |"
                f" Training loss: {train_total_loss:.4f} |"
                f" Grad norm: {grad_norm_cpu:.4f} |"
                f" Forward pass duration: {average_forward_duration:.2f} s |"
                f" Step duration: {step_duration:.2f} s"
            )
        if final_step or step % config.save_checkpoint_period == 0:
            save_checkpoint(
                config.checkpoint_dir,
                step,
                model.state_dict(),
                adamw_optimiser.state_dict(),
                muon_state_dict=None,
                timestamp=generate_timestamp(),
            )
    if config.wandb_run:
        wandb_run.finish()
    logger.info("Training finished!")


def main():
    assert torch.cuda.is_available(), "No CUDA device detected!"
    set_seed()
    device = torch.device("cuda")
    # Conduct matrix multiplication operations in TF32 instead of FP32
    torch.set_float32_matmul_precision("high")
    model_config = TransformerConfig()
    model = Transformer(model_config)
    model.to(device=device)
    # Compiled model
    num_trainable_params = count_model_params(model, trainable_only=True)
    logger.info(f"Number of trainable parameters: {num_trainable_params:,}")

    training_config = TrainingConfig()

    data_config = DataConfig(
        position_norm_cache=_PROJECT_ROOT / "cache/position_stats.pkl",
    )

    dataloaders = create_dataloaders(
        data_config,
        batch_size=training_config.device_batch_size,
        num_workers=training_config.num_workers,
        precomputed_energy=False,
        precomputed_trace=False,
    )
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    train_generator = infinite_dataloader(train_dataloader)
    val_generator = infinite_dataloader(val_dataloader)

    target_normaliser = dataloaders["train"].dataset.target_normaliser
    # energy_normaliser = dataloaders["train"].dataset.energy_normaliser

    optimisers = model.configure_optimisers(
        adamw_lr=training_config.adamw_lr,
        adamw_betas=training_config.adamw_betas,
        adamw_weight_decay=training_config.adamw_weight_decay,
        adamw_fused=training_config.adamw_fused,
        use_muon=training_config.use_muon,
        muon_lr=training_config.muon_lr,
        muon_weight_decay=training_config.muon_weight_decay,
        muon_momentum=training_config.muon_momentum,
        nesterov=training_config.nesterov,
        ns_steps=training_config.ns_steps,
    )
    adamw_optimiser, muon_optimiser = optimisers
    adamw_scheduler = cosine_scheduler_with_linear_warmup(
        adamw_optimiser,
        num_warmup_steps=training_config.adamw_warmup_steps,
        total_steps=training_config.num_steps,
    )
    
    if muon_optimiser is not None:
        logger.info("Using both AdamW and Muon optimisers")
        muon_scheduler = cosine_scheduler_with_linear_warmup(
            muon_optimiser,
            num_warmup_steps=training_config.muon_warmup_steps,
            total_steps=training_config.num_steps,
        )
        train_with_adamw_and_muon(
            model,
            adamw_optimiser,
            adamw_scheduler,
            muon_optimiser,
            muon_scheduler,
            train_generator,
            val_generator,
            device,
            training_config,
            target_normaliser=target_normaliser,
            # energy_normaliser=energy_normaliser,
        )

    else:
        logger.info("Using AdamW optimiser only")
        train_with_adamw_only(
            model,
            adamw_optimiser,
            adamw_scheduler,
            train_generator,
            val_generator,
            device,
            training_config,
            target_normaliser=target_normaliser,
            # energy_normaliser=energy_normaliser,
        )

if __name__ == "__main__":
    main()
