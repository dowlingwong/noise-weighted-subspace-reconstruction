"""TIDMAD reconstruction model: the ``current_compact`` backbone plus a
per-patch reconstruction head (PLAN_04 §3.3).

The backbone is reused unchanged through the temporal stage; the cross-band
stage replaces the backbone's patch-scaled absolute embedding with a band
embedding dimensioned to the number of bands ``C``. This is deliberate (round-2
correction T2): the backbone sizes its absolute positional embedding over the
*channel* axis from ``max_seq_len // patch_len``, which would otherwise force the
frame count ``M`` to scale with the band count. Decoupling the band embedding
lets the window be chosen on scientific grounds. Positional embeddings are
otherwise exactly as the backbone ships them: RoPE on the temporal axis, learned
absolute embeddings across bands.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .backbone.model import (
    AbsolutePositionalEmbedding,
    Transformer,
    TransformerConfig,
    normalise_input_sequence,
    patch_input_sequence,
)


class TidmadTransformer(Transformer):
    """``current_compact`` backbone with a reconstruction head.

    Input: ``(B, C, M)`` real spectrogram with ``C`` bands (real and imaginary
    parts stacked) and ``M`` frames. Output: ``(B, C, M)`` reconstruction in the
    same (standardised) coordinates; the caller maps it back to measurement
    coordinates before any loss (see :mod:`tidmad.loss`).
    """

    def __init__(self, config: TransformerConfig, n_bands: int | None = None):
        super().__init__(config)
        # Remove the task heads of the DELight model and the backbone's
        # patch-scaled absolute embedding (replaced by a band embedding below).
        del self.spatial_head
        del self.energy_head
        del self.abs_pos_embd
        self.patch_len = int(config.patch_len)
        if self.max_patches * self.patch_len != config.max_seq_len:
            raise ValueError(
                f"max_seq_len ({config.max_seq_len}) must be an exact multiple of "
                f"patch_len ({self.patch_len}); got {self.max_patches} patches"
            )
        self.n_bands = int(n_bands) if n_bands is not None else self.max_patches
        self.band_pos_embd = AbsolutePositionalEmbedding(self.n_bands, config.d_model)
        self.reconstruction_head = nn.Linear(config.d_model, self.patch_len, bias=True)

    def init_weights(self):
        # Replicate the backbone init (muP spectral init + zeroed residual
        # projections) without the base class's deleted head zeroing.
        self.apply(self._init_weights)
        for temporal_block in self.temporal_layers:
            nn.init.zeros_(temporal_block.attn.o_proj.weight)
            nn.init.zeros_(temporal_block.ffn.down_proj.weight)
        for spatial_block in self.spatial_layers:
            nn.init.zeros_(spatial_block.attn.o_proj.weight)
            nn.init.zeros_(spatial_block.ffn.down_proj.weight)
        nn.init.normal_(self.band_pos_embd.pos_embed, std=0.02)
        nn.init.zeros_(self.reconstruction_head.weight)
        nn.init.zeros_(self.reconstruction_head.bias)

    def configure_optimisers(self, **kwargs):
        """Reuse the backbone parameter split but treat the head as aux params."""
        adamw_kwargs = dict(
            lr=kwargs.get("adamw_lr", 1e-3),
            betas=kwargs.get("adamw_betas", (0.9, 0.999)),
            weight_decay=kwargs.get("adamw_weight_decay", 0.0),
            fused=kwargs.get("adamw_fused", True),
        )
        block_weight_params: list[torch.nn.Parameter] = []
        aux_params: list[torch.nn.Parameter] = []
        for block in (*self.temporal_layers, *self.spatial_layers):
            for p in block.parameters():
                (block_weight_params if p.ndim >= 2 else aux_params).append(p)
        aux_params.extend(self.band_pos_embd.parameters())
        aux_params.extend(self.final_proj_norm.parameters())
        aux_params.extend(self.reconstruction_head.parameters())
        from torch.optim import AdamW

        from .backbone.muon import Muon

        adamw = AdamW(
            [
                dict(params=self.patch_embedding.parameters()),
                dict(params=aux_params),
            ],
            **adamw_kwargs,
        )
        muon = Muon(
            block_weight_params,
            lr=kwargs.get("muon_lr", 1e-3),
            momentum=kwargs.get("muon_momentum", 0.95),
            nesterov=kwargs.get("muon_nesterov", True),
            ns_steps=kwargs.get("muon_ns_steps", 5),
        )
        return [adamw, muon]

    def _temporal_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Input ``(B, C, M)`` -> temporal-stage activations ``(B*C, N, d)``."""
        batch_size, channel_dim, seq_len = x.size()
        x = x.reshape(batch_size * channel_dim, seq_len)
        x = normalise_input_sequence(x)
        x = patch_input_sequence(x, self.config.patch_len, self.config.patch_stride)
        x = self.patch_embedding(x)
        for layer in self.temporal_layers:
            x = layer(x, self._rope_cache)
        return x

    def _band_stage(self, temporal: torch.Tensor, batch_size: int, channel_dim: int) -> torch.Tensor:
        """Temporal activations -> ``(B, N, C, d)`` after band embedding + spatial layers + norm."""
        x = temporal.view(batch_size, channel_dim, -1, self.config.d_model)
        x = x.transpose(1, 2).reshape(-1, channel_dim, self.config.d_model)
        x = self.band_pos_embd(x)
        for layer in self.spatial_layers:
            x = layer(x, None)
        x = self.final_proj_norm(x)
        return x.view(batch_size, -1, channel_dim, self.config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel_dim, _ = x.size()
        x = self._temporal_stage(x)
        x = self._band_stage(x, batch_size, channel_dim)
        x = self.reconstruction_head(x)  # (B, N, C, patch_len)
        x = x.transpose(1, 2)  # (B, C, N, patch_len)
        return x.reshape(x.size(0), x.size(1), -1)  # (B, C, M)
