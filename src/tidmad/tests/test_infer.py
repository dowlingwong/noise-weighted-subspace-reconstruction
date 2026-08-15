"""Regression tests for the TIDMAD inference handoff."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tidmad.infer import _restore_measurement_scale, _window_positions  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_restore_measurement_scale_uses_model_output_device():
    spec_in = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out_std = torch.zeros((1, 2, 3, 4), device="cuda:0")

    restored = _restore_measurement_scale(out_std, spec_in)

    assert restored.device == out_std.device
    expected = torch.as_tensor(spec_in, device=out_std.device).mean(dim=-1)
    assert torch.allclose(restored.squeeze(0)[..., 0], expected)


@pytest.mark.parametrize("n_samples", [40, 41, 79, 80, 81, 103])
def test_window_positions_cover_every_sample(n_samples):
    positions = _window_positions(n_samples, window_samples=40)
    covered = np.zeros(n_samples, dtype=bool)
    for position in positions:
        covered[position : position + 40] = True

    assert covered.all()
    assert positions == sorted(set(positions))
    assert positions[-1] + 40 == n_samples
