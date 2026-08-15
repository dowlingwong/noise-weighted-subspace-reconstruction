"""Inference and scoring for the TIDMAD arms (PLAN_04 §3.5).

``infer.py`` runs a trained ``tidmad_stft`` model over every validation file,
writes ``abra_validation_denoised_<arm>_00NN.h5`` in the release's layout
(denoised SQUID in ``channel0001``, the untouched reference in ``channel0002``),
reusing Paper 1's h5 writer so the file semantics match ``inference.py`` exactly.

``score.py`` then invokes the *unmodified* upstream ``benchmark.py -c`` and
captures its stdout/returncode. There is deliberately no internal reimplementation
of the score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .backbone.checkpoints import load_checkpoint

from .config import FROZEN
from .data import build_input_window_stft, spectrogram_to_series
from .train import build_model, load_run_config


def _restore_measurement_scale(out_std: torch.Tensor, spec_in: np.ndarray) -> torch.Tensor:
    """Undo input standardization on the model output's device."""
    input_tensor = torch.as_tensor(spec_in, device=out_std.device)
    mean_in = torch.nanmean(input_tensor, dim=-1, keepdim=True)
    std_in = torch.std(input_tensor, dim=-1, keepdim=True)
    return out_std * (std_in + 1e-6) + mean_in


def _window_positions(n_samples: int, window_samples: int) -> list[int]:
    """Tile a file and add one overlapping window to cover any final suffix."""
    if n_samples < window_samples:
        raise ValueError("validation file is shorter than one inference window")
    positions = list(range(0, n_samples - window_samples + 1, window_samples))
    final_start = n_samples - window_samples
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def _write_denoised_channel(denoised: np.ndarray, source: Path, destination_dir: Path, model_tag: str):
    """Port of Paper 1's writer (noise_geometry.tidmad.score.write_denoised_file)."""
    from noise_geometry.tidmad.score import DenoisedFileSpec, write_denoised_file

    spec = DenoisedFileSpec(source_file=source, destination_dir=destination_dir, model_tag=model_tag)
    write_denoised_file(
        denoised,
        spec,
        source_dataset="timeseries/channel0001/timeseries",
        attributes={"denoiser": model_tag},
    )


def _copy_reference_channel(source: Path, out_dir: Path, model_tag: str):
    """Copy the untouched reference channel into the denoised file (benchmark reads both).

    Uses the same ``DenoisedFileSpec`` as the denoised write so the reference
    lands in the same destination file via the writer's append mode.
    """
    from noise_geometry.tidmad.score import DenoisedFileSpec, write_denoised_file

    import h5py

    with h5py.File(source, "r") as f:
        ref = np.asarray(f["timeseries"]["channel0002"]["timeseries"], dtype=np.float64)
    write_denoised_file(
        ref,
        DenoisedFileSpec(source_file=source, destination_dir=out_dir, model_tag=model_tag),
        source_dataset="timeseries/channel0002/timeseries",
    )


def infer(config: Path, checkpoint: Path, data_dir: Path, out_dir: Path, model_tag: str, device: str):
    run = load_run_config(config)
    model = build_model(run).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    from noise_geometry.tidmad.loader import load_contract

    contract = load_contract(run.data.contract_path)
    contract.require_verified()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import h5py

    # Select the 20 raw release files only. The legacy server root also contains
    # pre-existing abra_validation_denoised_* products.
    for path in sorted(data_dir.glob("abra_validation_00??.h5")):
        index = path.name.split("_")[-1].split(".")[0]
        with h5py.File(path, "r") as f:
            n_samples = int(f["timeseries"]["channel0001"]["timeseries"].shape[0])
        n = run.data.window_samples
        denoised = np.empty(n_samples, dtype=np.float64)
        positions = _window_positions(n_samples, n)
        batch_size = run.train.device_batch_size
        filled_until = 0
        with torch.no_grad():
            for batch_start in range(0, len(positions), batch_size):
                batch_positions = positions[batch_start : batch_start + batch_size]
                specs = [
                    build_input_window_stft(
                        path, contract, pos, run.stft, run.data.window_samples
                    )
                    for pos in batch_positions
                ]
                spec_in = np.stack(specs)
                z = torch.as_tensor(spec_in, dtype=torch.float32, device=device)
                out_std = model(z)
                out_meas = _restore_measurement_scale(out_std, spec_in)
                for batch_index, pos in enumerate(batch_positions):
                    seg = spectrogram_to_series(
                        out_meas[batch_index].cpu().numpy(), run.stft, n
                    )
                    copy_start = max(pos, filled_until)
                    denoised[copy_start : pos + n] = seg[copy_start - pos :]
                    filled_until = pos + n
        if filled_until != n_samples:
            raise RuntimeError(
                f"inference covered {filled_until} of {n_samples} samples for {path}"
            )
        _write_denoised_channel(denoised, path, out_dir, model_tag)
        _copy_reference_channel(path, out_dir, model_tag)
        print(f"wrote {out_dir / f'abra_validation_denoised_{model_tag}_{index}.h5'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    infer(args.config, args.checkpoint, args.data_dir, args.out, args.model_tag, args.device)


if __name__ == "__main__":
    main()
