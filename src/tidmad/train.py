"""Minimal training loop for the two TIDMAD arms (PLAN_04 §3.4).

Only the residual metric in the objective differs between arms (``train.loss``
in the run config). Everything else — architecture, data, budget, seeds — is
identical and asserted by :func:`assert_configs_differ_only_in_loss`.

Reuses the backbone optimisers (Muon + AdamW), the cosine-with-warmup
schedulers, the checkpoint helpers, and ``set_seed`` from the DELight
vendored backbone in ``tidmad.backbone``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .backbone.checkpoints import save_checkpoint
from .backbone.schedulers import cosine_scheduler_with_linear_warmup
from .seeding import set_seed
from .backbone.model import TransformerConfig

from .config import FROZEN, TidmadRunConfig
from .loss import reconstruction_losses
from .model import TidmadTransformer

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional
    yaml = None


def load_run_config(path: Path) -> TidmadRunConfig:
    """Build a run config from a YAML arm file on top of the frozen defaults."""
    payload = {}
    if yaml is not None and Path(path).exists():
        with open(path) as fh:
            payload = yaml.safe_load(fh) or {}
    return _apply_overrides(FROZEN, payload)


def _apply_overrides(frozen: TidmadRunConfig, payload: dict) -> TidmadRunConfig:
    data = {**frozen.data.__dict__, **payload.get("data", {})}
    train = {**frozen.train.__dict__, **payload.get("train", {})}
    model = {**frozen.model.__dict__, **payload.get("model", {})}
    stft = {**frozen.stft.__dict__, **payload.get("stft", {})}
    from .config import TidmadDataConfig, TidmadModelConfig, TidmadSTFTConfig, TidmadTrainConfig

    return TidmadRunConfig(
        stft=TidmadSTFTConfig(**stft),
        model=TidmadModelConfig(**model),
        train=TidmadTrainConfig(**train),
        data=TidmadDataConfig(**data),
    )


def assert_configs_differ_only_in_loss(left: TidmadRunConfig, right: TidmadRunConfig) -> None:
    """Gate: the two arms' configs must differ in ``train.loss`` and nothing else."""
    a, b = left.as_dict(), right.as_dict()
    diffs: list[str] = []
    for key in sorted(set(a) | set(b)):
        if a[key] != b[key]:
            if key == "train" and _only_loss_differs(a["train"], b["train"]):
                continue
            diffs.append(key)
    if diffs:
        raise AssertionError(
            f"arm configs differ beyond the loss flag: {diffs}. The contrast is invalid."
        )
    if a["train"]["loss"] == b["train"]["loss"]:
        raise AssertionError("arm configs have the same loss flag; the contrast is empty.")


def _only_loss_differs(a: dict, b: dict) -> bool:
    return all(k == "loss" or a[k] == b[k] for k in set(a) | set(b))


def write_run_config(run: TidmadRunConfig, out: Path, num_params: int) -> None:
    """Write ``run_config.json`` in the ``save_resolved_config`` shape, plus loss."""
    Path(out).mkdir(parents=True, exist_ok=True)
    payload = {
        "model_variant": "tidmad_stft",
        "run_config": run.as_dict(),
        "num_trainable_params": num_params,
    }
    (out / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_model(run: TidmadRunConfig) -> TidmadTransformer:
    cfg = run.model
    torch_cfg = TransformerConfig(
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        max_seq_len=run.frames_per_window,
        patch_len=cfg.patch_len,
        patch_stride=cfg.patch_len,
        n_head=cfg.n_head,
        n_time_layers=cfg.n_time_layers,
        n_channel_layers=cfg.n_channel_layers,
        rope_base=cfg.rope_base,
        norm_eps=cfg.norm_eps,
    )
    model = TidmadTransformer(torch_cfg, n_bands=run.stft.n_bands_stacked)
    model.init_weights()
    return model


def train_step(model, optimisers, schedulers, batch, run, j_stack, device):
    input_spec, target_spec = batch  # both (B, C, M) measurement-coordinate
    input_spec = input_spec.to(device)
    target_spec = target_spec.to(device)
    out_std = model(input_spec)
    losses = reconstruction_losses(out_std, target_spec, input_spec, j_stack)
    objective = losses[run.train.loss if run.train.loss != "chi2_of" else "chi2"]
    for opt in optimisers:
        opt.zero_grad()
    objective.backward()
    for opt in optimisers:
        opt.step()
    for sched in schedulers:
        sched.step()
    return {k: float(v.item()) for k, v in losses.items()}


def _noise_only_windows(data_dir: Path, run: TidmadRunConfig):
    """Stream ``n_calibration_windows`` windows of noise-only science data."""
    from noise_geometry.tidmad.loader import TidmadFile, load_contract

    contract = load_contract(run.data.contract_path)
    contract.require_verified()
    science = sorted(data_dir.glob(run.data.psd_source_glob))
    if not science:
        raise FileNotFoundError(f"no science files matched {run.data.psd_source_glob!r} in {data_dir}")
    n = run.data.calibration_window_samples
    produced = 0
    for path in science:
        with TidmadFile(path, contract) as fh:
            for win in fh.iter_squid_windows(n, stride=n, limit=run.data.n_calibration_windows - produced):
                yield win
                produced += 1
                if produced >= run.data.n_calibration_windows:
                    return


def _train_windows(data_dir: Path, run: TidmadRunConfig):
    """Yield ``(input_spec, target_spec)`` measurement-coordinate spectrograms."""
    from noise_geometry.tidmad.loader import TidmadFile, load_contract

    from .data import build_window_stft

    contract = load_contract(run.data.contract_path)
    contract.require_verified()
    files = [Path(f) for f in run.data.train_files] or sorted(data_dir.glob("abra_training_*.h5"))
    if not files:
        raise FileNotFoundError(f"no training files found under {data_dir}")
    for path in files:
        with TidmadFile(path, contract) as fh:
            n = run.stft.window_samples
            pos = 0
            while True:
                if pos + n > fh.n_samples:
                    break
                yield build_window_stft(path, contract, pos, run.stft)
                pos += n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    run = load_run_config(args.config)
    if args.steps is not None:
        run = TidmadRunConfig(
            run.stft, run.model, run.train.__class__(**{**run.train.__dict__, "num_steps": args.steps}), run.data
        )
    set_seed(run.train.seed)
    device = torch.device(args.device)

    from .psd import estimate_band_psd, tile_for_stacked_bands

    j_band, _ = estimate_band_psd(
        _noise_only_windows(args.data_dir, run),
        run.stft,
        sampling_frequency=run.stft.sampling_frequency_hz,
        window=run.data.psd_window,
        average=run.data.psd_average,
    )
    j_stack = torch.as_tensor(tile_for_stacked_bands(j_band, run.stft), dtype=torch.float32)

    model = build_model(run).to(device)
    adamw, muon = model.configure_optimisers(
        adamw_lr=run.train.adamw_lr,
        adamw_betas=run.train.adamw_betas,
        adamw_weight_decay=run.train.adamw_weight_decay,
        adamw_fused=False,
        muon_lr=run.train.muon_lr,
        muon_momentum=run.train.muon_momentum,
        muon_nesterov=run.train.muon_nesterov,
        muon_ns_steps=run.train.muon_ns_steps,
    )
    schedulers = [
        cosine_scheduler_with_linear_warmup(adamw, run.train.warmup_steps, run.train.num_steps),
        cosine_scheduler_with_linear_warmup(muon, run.train.warmup_steps, run.train.num_steps),
    ]

    write_run_config(run, args.out, sum(p.numel() for p in model.parameters()))
    print(f"[tidmad.train] arm loss={run.train.loss} model_params={sum(p.numel() for p in model.parameters())}")

    for step, (spec_in, spec_tgt) in enumerate(_train_windows(args.data_dir, run)):
        if step >= run.train.num_steps:
            break
        metrics = train_step(
            model,
            [adamw, muon],
            schedulers,
            (torch.as_tensor(spec_in), torch.as_tensor(spec_tgt)),
            run,
            j_stack,
            device,
        )
        if step % run.train.eval_period == 0:
            print(f"step {step}: {metrics}")


if __name__ == "__main__":
    main()
