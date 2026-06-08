"""Decoder heads shared by AE and transformer reconstruction models."""

from __future__ import annotations

from dataclasses import dataclass

from paper2._torch import Tensor, nn, require_torch, torch


@dataclass(slots=True)
class OutputHeadConfig:
    in_channels: int
    out_channels: int
    kernel_size: int = 1


class NativeWaveformHead(nn.Module):
    """Simple native-space output projection."""

    def __init__(self, cfg: OutputHeadConfig) -> None:
        require_torch()
        super().__init__()
        padding = cfg.kernel_size // 2
        self.proj = nn.Conv1d(
            cfg.in_channels,
            cfg.out_channels,
            kernel_size=cfg.kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class PatchDecoder(nn.Module):
    """Token-to-waveform decoder.

    This is the minimum viable decoder for the transformer reconstruction
    wrapper: per-patch projection followed by overlap-add if stride < patch
    length.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        patch_len: int,
        patch_stride: int,
        trace_len: int,
    ) -> None:
        require_torch()
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.trace_len = trace_len
        self.n_channels = n_channels
        self.patch_proj = nn.Linear(d_model, patch_len)

    def forward(self, tokens: Tensor) -> Tensor:
        # expected tokens: (B, C, N_patch, d_model)
        bsz, n_channels, n_patch, _ = tokens.shape
        patches = self.patch_proj(tokens)
        output = torch.zeros(
            bsz,
            n_channels,
            self.trace_len,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        counts = torch.zeros_like(output)
        for idx in range(n_patch):
            start = idx * self.patch_stride
            end = min(start + self.patch_len, self.trace_len)
            width = end - start
            output[:, :, start:end] += patches[:, :, idx, :width]
            counts[:, :, start:end] += 1.0
        counts = torch.clamp(counts, min=1.0)
        return output / counts
