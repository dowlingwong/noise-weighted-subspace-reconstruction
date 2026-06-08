"""Token-preserving transformer encoder wrapper for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass

from paper2._torch import Tensor, nn, require_torch, torch


@dataclass(slots=True)
class TransformerEncoderConfig:
    trace_len: int
    n_channels: int
    patch_len: int
    patch_stride: int
    d_model: int
    d_ff: int
    n_head: int
    n_temporal_blocks: int
    dropout: float = 0.1


class TransformerEncoder(nn.Module):
    """Minimal token-preserving reconstruction encoder.

    This is intentionally smaller and cleaner than the existing classifier-like
    transformer backbones under `src/transformer/`.
    """

    def __init__(self, cfg: TransformerEncoderConfig) -> None:
        require_torch()
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_temporal_blocks,
        )

    def patchify(self, x: Tensor) -> Tensor:
        # (B, C, T) -> (B, C, N_patch, patch_len)
        return x.unfold(-1, self.cfg.patch_len, self.cfg.patch_stride)

    def forward(self, x: Tensor) -> Tensor:
        patches = self.patchify(x)
        bsz, n_channels, n_patch, patch_len = patches.shape
        tokens = self.patch_embed(patches)
        tokens = tokens.view(bsz * n_channels, n_patch, self.cfg.d_model)
        tokens = self.temporal_encoder(tokens)
        return tokens.view(bsz, n_channels, n_patch, self.cfg.d_model)
