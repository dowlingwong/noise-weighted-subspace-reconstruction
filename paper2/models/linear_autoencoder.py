"""Patch-linear autoencoder baseline for Experiment D."""

from __future__ import annotations

from dataclasses import dataclass

from paper2._torch import Tensor, nn, require_torch
from paper2.data.whitening import WhiteningOperator
from paper2.models.base import BaseReconstructionModel, ModelConfig, ReconstructionOutput
from paper2.models.decoder_heads import PatchDecoder


@dataclass(slots=True)
class LinearAutoencoderConfig(ModelConfig):
    patch_len: int = 128
    patch_stride: int = 128


class PatchLinearAutoencoder(BaseReconstructionModel):
    """Manageable linear baseline using patchify -> linear bottleneck -> overlap-add."""

    def __init__(
        self,
        cfg: LinearAutoencoderConfig,
        whitener: WhiteningOperator | None = None,
    ) -> None:
        require_torch()
        super().__init__(cfg)
        self.whitener = whitener
        self.patch_len = cfg.patch_len
        self.patch_stride = cfg.patch_stride
        self.encoder = nn.Linear(cfg.patch_len, cfg.latent_dim, bias=False)
        self.decoder = PatchDecoder(
            d_model=cfg.latent_dim,
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
        patches = x_enc.unfold(-1, self.patch_len, self.patch_stride)
        z = self.encoder(patches)
        x_hat = self.decoder(z)
        return ReconstructionOutput(x_hat=x_hat, z=z, x_enc=x_enc)
