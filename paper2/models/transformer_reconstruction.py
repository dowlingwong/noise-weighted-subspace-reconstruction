"""Transformer reconstruction wrapper for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass

from paper2._torch import Tensor, nn, require_torch
from paper2.data.whitening import WhiteningOperator
from paper2.models.base import BaseReconstructionModel, ModelConfig, ReconstructionOutput
from paper2.models.decoder_heads import PatchDecoder
from paper2.models.transformer_encoder import TransformerEncoder, TransformerEncoderConfig


@dataclass(slots=True)
class TransformerReconstructionConfig(ModelConfig):
    patch_len: int = 128
    patch_stride: int = 128
    d_model: int = 128
    d_ff: int = 512
    n_head: int = 4
    n_temporal_blocks: int = 4
    dropout: float = 0.1


class TransformerReconstructionModel(BaseReconstructionModel):
    """Reconstruction model for the Paper 2 transformer `2x2`."""

    def __init__(
        self,
        cfg: TransformerReconstructionConfig,
        whitener: WhiteningOperator | None = None,
    ) -> None:
        require_torch()
        super().__init__(cfg)
        self.whitener = whitener
        self.encoder = TransformerEncoder(
            TransformerEncoderConfig(
                trace_len=cfg.trace_len,
                n_channels=cfg.n_channels,
                patch_len=cfg.patch_len,
                patch_stride=cfg.patch_stride,
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                n_head=cfg.n_head,
                n_temporal_blocks=cfg.n_temporal_blocks,
                dropout=cfg.dropout,
            )
        )
        self.latent_proj = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.latent_backproj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.decoder = PatchDecoder(
            d_model=cfg.d_model,
            n_channels=cfg.n_channels,
            patch_len=cfg.patch_len,
            patch_stride=cfg.patch_stride,
            trace_len=cfg.trace_len,
        )

    def _prepare_encoder_input(self, x_native: Tensor) -> Tensor:
        if self.cfg.input_mode == "prewhitened":
            if self.whitener is None:
                raise RuntimeError("Whitening requested but no WhiteningOperator was provided.")
            return self.whitener.whiten_input(x_native)
        return x_native

    def forward(self, x_native: Tensor) -> ReconstructionOutput:
        x_enc = self._prepare_encoder_input(x_native)
        tokens = self.encoder(x_enc)
        z = self.latent_proj(tokens)
        decoded_tokens = self.latent_backproj(z)
        x_hat = self.decoder(decoded_tokens)
        return ReconstructionOutput(x_hat=x_hat, z=z, x_enc=x_enc)
