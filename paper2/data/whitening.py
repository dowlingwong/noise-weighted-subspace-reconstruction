"""Fixed whitening operators for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper2._torch import Tensor, nn, require_torch, torch


@dataclass(slots=True)
class WhiteningConfig:
    psd_path: str
    trace_len: int
    n_channels: int
    eps: float = 1e-12


def load_one_sided_psd(psd_path: str | Path) -> np.ndarray:
    path = Path(psd_path)
    return np.load(path).astype(np.float64).reshape(-1)


def build_one_sided_weights(psd_one_sided: np.ndarray, trace_len: int) -> np.ndarray:
    psd = np.asarray(psd_one_sided, dtype=np.float64)
    n_rfft = trace_len // 2 + 1
    if psd.shape[0] != n_rfft:
        raise ValueError(f"PSD length {psd.shape[0]} does not match rfft bins {n_rfft}")

    w = np.zeros_like(psd)
    w[0] = 0.0
    if trace_len % 2 == 0:
        w[1:-1] = 2.0 / np.maximum(psd[1:-1], np.finfo(float).eps)
        w[-1] = 1.0 / max(psd[-1], np.finfo(float).eps)
    else:
        w[1:] = 2.0 / np.maximum(psd[1:], np.finfo(float).eps)
    return w


class WhiteningOperator(nn.Module):
    """Frequency-domain whitening using a fixed one-sided PSD."""

    def __init__(self, cfg: WhiteningConfig) -> None:
        require_torch()
        super().__init__()
        psd = load_one_sided_psd(cfg.psd_path)
        weights = build_one_sided_weights(psd, cfg.trace_len)
        sqrt_w = np.sqrt(np.maximum(weights, 0.0))
        inv_sqrt_w = np.zeros_like(sqrt_w)
        mask = sqrt_w > cfg.eps
        inv_sqrt_w[mask] = 1.0 / sqrt_w[mask]

        self.trace_len = int(cfg.trace_len)
        self.n_channels = int(cfg.n_channels)
        self.register_buffer("psd_one_sided", torch.as_tensor(psd, dtype=torch.float32))
        self.register_buffer("weights_one_sided", torch.as_tensor(weights, dtype=torch.float32))
        self.register_buffer("sqrt_weights_one_sided", torch.as_tensor(sqrt_w, dtype=torch.float32))
        self.register_buffer("inv_sqrt_weights_one_sided", torch.as_tensor(inv_sqrt_w, dtype=torch.float32))

    def whiten_input(self, x_native: Tensor) -> Tensor:
        x_f = torch.fft.rfft(x_native, dim=-1)
        x_fw = x_f * self.sqrt_weights_one_sided
        return torch.fft.irfft(x_fw, n=self.trace_len, dim=-1)

    def color_output(self, x_whitened: Tensor) -> Tensor:
        x_f = torch.fft.rfft(x_whitened, dim=-1)
        x_fc = x_f * self.inv_sqrt_weights_one_sided
        return torch.fft.irfft(x_fc, n=self.trace_len, dim=-1)

    def mahalanobis_energy(self, residual_native: Tensor) -> Tensor:
        residual_f = torch.fft.rfft(residual_native, dim=-1)
        weighted = torch.abs(residual_f) ** 2 * self.weights_one_sided
        return weighted.sum(dim=-1).mean(dim=-1)
