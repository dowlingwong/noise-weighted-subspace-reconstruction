"""Simple nonlinear reconstruction autoencoder for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass

from paper2._torch import Tensor, nn, require_torch, torch
from paper2.data.whitening import WhiteningOperator
from paper2.models.base import BaseReconstructionModel, ModelConfig, ReconstructionOutput
from paper2.models.decoder_heads import NativeWaveformHead, OutputHeadConfig


@dataclass(slots=True)
class ReconstructionAEConfig(ModelConfig):
    hidden_channels: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 7
    stride: int = 2


class ReconstructionAE(BaseReconstructionModel):
    """Small conv AE intended as the first nonlinear baseline."""

    def __init__(
        self,
        cfg: ReconstructionAEConfig,
        whitener: WhiteningOperator | None = None,
    ) -> None:
        require_torch()
        super().__init__(cfg)
        self.whitener = whitener

        encoder_layers: list[nn.Module] = []
        in_ch = cfg.n_channels
        for out_ch in cfg.hidden_channels:
            encoder_layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding=cfg.kernel_size // 2,
                    ),
                    nn.GELU(),
                ]
            )
            in_ch = out_ch
        self.encoder_stem = nn.Sequential(*encoder_layers)

        downsample_factor = cfg.stride ** len(cfg.hidden_channels)
        latent_time = max(1, cfg.trace_len // downsample_factor)
        self._latent_channels = in_ch
        self._latent_time = latent_time
        self.latent_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * latent_time, cfg.latent_dim),
            nn.GELU(),
        )
        self.latent_expand = nn.Sequential(
            nn.Linear(cfg.latent_dim, in_ch * latent_time),
            nn.GELU(),
        )

        decoder_layers: list[nn.Module] = []
        rev_channels = list(cfg.hidden_channels[::-1])
        for idx, out_ch in enumerate(rev_channels[1:] + [rev_channels[-1]]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose1d(
                        rev_channels[idx],
                        out_ch,
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding=cfg.kernel_size // 2,
                        output_padding=1,
                    ),
                    nn.GELU(),
                ]
            )
        self.decoder_stem = nn.Sequential(*decoder_layers)
        self.output_head = NativeWaveformHead(
            OutputHeadConfig(
                in_channels=rev_channels[-1],
                out_channels=cfg.n_channels,
                kernel_size=1,
            )
        )

    def _prepare_encoder_input(self, x_native: Tensor) -> Tensor:
        if self.cfg.input_mode == "prewhitened":
            if self.whitener is None:
                raise RuntimeError("Whitening requested but no WhiteningOperator was provided.")
            return self.whitener.whiten_input(x_native)
        return x_native

    def forward(self, x_native: Tensor) -> ReconstructionOutput:
        x_enc = self._prepare_encoder_input(x_native)
        features = self.encoder_stem(x_enc)
        z = self.latent_projector(features)
        expanded = self.latent_expand(z).view(
            x_native.shape[0],
            self._latent_channels,
            self._latent_time,
        )
        decoded = self.decoder_stem(expanded)
        x_hat = self.output_head(decoded)[..., : self.cfg.trace_len]
        return ReconstructionOutput(x_hat=x_hat, z=z, x_enc=x_enc)
