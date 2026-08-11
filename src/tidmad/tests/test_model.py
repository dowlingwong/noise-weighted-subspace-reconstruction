"""STFT round-trip and reconstruction-head tests (PLAN_04 §3.6 items 2–3).

Tests use an explicit small configuration (not the heavy frozen one) so they
run quickly on CPU while exercising the same code paths.
"""

from __future__ import annotations

import numpy as np
import torch

from tidmad.config import TidmadDataConfig, TidmadRunConfig, TidmadSTFTConfig, TidmadModelConfig, TidmadTrainConfig
from tidmad.data import series_to_spectrogram, spectrogram_to_series, stft, istft
from tidmad.model import TidmadTransformer
from tidmad.backbone.model import TransformerConfig


def _small_cfg():
    """Small run config: C = 8 bands (re/im of 4 bins), M = 32 frames."""
    stft = TidmadSTFTConfig(n_fft=64, hop_length=32, win_length=64, n_bands_used=4)
    patch_len = 4
    frames = 32  # multiple of patch_len; no longer tied to C (round-2 T2)
    data = TidmadDataConfig(window_samples=(frames - 1) * stft.hop_length + stft.n_fft)
    model = TidmadModelConfig(d_model=16, d_ff=32, n_head=2, patch_len=patch_len)
    run = TidmadRunConfig(stft=stft, model=model, train=TidmadTrainConfig(num_steps=2), data=data)
    assert run.frames_per_window == frames
    return run


def test_stft_round_trip():
    cfg = TidmadSTFTConfig(n_fft=256, hop_length=128, win_length=256)
    n = 2048
    x = np.random.default_rng(3).standard_normal(n)
    X = stft(x, cfg)
    assert X.shape == (cfg.n_fft // 2 + 1, (n - cfg.n_fft) // cfg.hop_length + 1)
    rec = istft(X, cfg, length=n)
    # The Hann window is zero at the very first/last sample of each frame, so
    # the endpoints are not recoverable; the interior round-trips exactly.
    err = np.max(np.abs(rec[cfg.n_fft : -cfg.n_fft] - x[cfg.n_fft : -cfg.n_fft]))
    assert err < 1e-8, f"round-trip error {err:.2e}"


def test_band_limited_round_trip_within_kept_bands():
    run = _small_cfg()
    cfg = run.stft
    n = run.data.window_samples
    t = np.arange(n)
    # Low-frequency content well inside the kept bands (bins 0..3 of n_fft=64).
    x = np.sin(2 * np.pi * 1.0 / 64.0 * t) + 0.5 * np.sin(2 * np.pi * 2.0 / 64.0 * t)
    Z = series_to_spectrogram(x, cfg)
    assert Z.shape == (cfg.n_bands_stacked, run.frames_per_window)
    rec = spectrogram_to_series(Z, cfg, n)
    assert rec.shape == (n,)
    # The OLA reconstruction of a band-limited spectrogram is approximate, and
    # the single-window regions at each frame edge amplify window-leakage error
    # by dividing through a near-zero Hann value; compare the OLA interior only.
    m = cfg.n_fft
    err = np.max(np.abs(rec[m:-m] - x[m:-m])) / np.max(np.abs(x))
    assert err < 0.02, f"band-limited reconstruction interior relative error {err:.4f}"


def test_reconstruction_head_output_shape_matches_input():
    run = _small_cfg()
    torch_cfg = TransformerConfig(
        max_seq_len=run.frames_per_window,
        patch_len=run.model.patch_len,
        patch_stride=run.model.patch_len,
        d_model=run.model.d_model,
        d_ff=run.model.d_ff,
        n_head=run.model.n_head,
    )
    model = TidmadTransformer(torch_cfg, n_bands=run.stft.n_bands_stacked)
    batch, bands = 2, run.stft.n_bands_stacked
    x = torch.randn(batch, bands, run.frames_per_window)
    out = model(x)
    assert out.shape == (batch, bands, run.frames_per_window)


def test_band_embedding_dimensioned_to_bands():
    """Round-2 T2: the band embedding is sized to C, not to max_seq_len//patch_len."""
    run = _small_cfg()
    torch_cfg = TransformerConfig(
        max_seq_len=run.frames_per_window,
        patch_len=run.model.patch_len,
        patch_stride=run.model.patch_len,
        d_model=run.model.d_model,
        d_ff=run.model.d_ff,
        n_head=run.model.n_head,
    )
    model = TidmadTransformer(torch_cfg, n_bands=run.stft.n_bands_stacked)
    assert model.band_pos_embd.pos_embed.shape[1] == run.stft.n_bands_stacked
    assert model.n_bands == run.stft.n_bands_stacked


def test_temporal_stage_parity_with_current_compact():
    """Round-2 T2: temporal stage matches the backbone exactly; the band stage is
    deliberately divergent (band embedding sized to C instead of max_patches)."""
    from tidmad.backbone.model import Transformer as BaseTransformer

    run = _small_cfg()
    torch_cfg = TransformerConfig(
        max_seq_len=run.frames_per_window,
        patch_len=run.model.patch_len,
        patch_stride=run.model.patch_len,
        d_model=run.model.d_model,
        d_ff=run.model.d_ff,
        n_head=run.model.n_head,
        n_time_layers=run.model.n_time_layers,
        n_channel_layers=run.model.n_channel_layers,
    )
    base = BaseTransformer(torch_cfg)
    tid = TidmadTransformer(torch_cfg, n_bands=run.stft.n_bands_stacked)
    # Share the backbone weights (temporal + spatial + norm) so the comparison is
    # of the forward path, not of two differently-initialised models. strict=False
    # ignores each model's private embedding/head keys.
    tid.load_state_dict(base.state_dict(), strict=False)

    captured: dict[str, torch.Tensor] = {}

    # The temporal stage ends where the band embedding is applied: capture the
    # embedding *input* on both models.
    base.abs_pos_embd.register_forward_pre_hook(lambda _m, args: captured.__setitem__("base", args[0].detach().clone()))
    tid.band_pos_embd.register_forward_pre_hook(lambda _m, args: captured.__setitem__("tid", args[0].detach().clone()))

    x = torch.randn(2, run.stft.n_bands_stacked, run.frames_per_window)
    _ = base(x)
    _ = tid(x)
    assert "base" in captured and "tid" in captured
    assert torch.allclose(captured["base"], captured["tid"], atol=1e-6), (
        "tidmad_stft and current_compact diverge before the band embedding; "
        "the band-embedding swap changed the temporal stage"
    )
