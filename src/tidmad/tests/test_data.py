from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

from tidmad.data import build_input_window_stft, read_channel_pair, read_input_channel
from tidmad.config import TidmadSTFTConfig


def test_read_channel_pair_uses_full_contract_paths(tmp_path):
    path = tmp_path / "abra_training_0000.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset(
            "timeseries/channel0001/timeseries",
            data=np.asarray([-128, -127, -126, -125], dtype=np.int8),
        )
        fh.create_dataset(
            "timeseries/channel0002/timeseries",
            data=np.asarray([-120, -119, -118, -117], dtype=np.int8),
        )
    contract = SimpleNamespace(
        squid_dataset="timeseries/channel0001/timeseries",
        reference_dataset="timeseries/channel0002/timeseries",
    )
    squid, reference = read_channel_pair(path, contract, start=1, length=2)
    np.testing.assert_array_equal(squid, [1.0, 2.0])
    np.testing.assert_array_equal(reference, [9.0, 10.0])


def test_inference_reader_does_not_require_reference_channel(tmp_path):
    path = tmp_path / "abra_validation_0000.h5"
    values = np.arange(-128, -64, dtype=np.int8)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("timeseries/channel0001/timeseries", data=values)
    contract = SimpleNamespace(squid_dataset="timeseries/channel0001/timeseries")

    squid = read_input_channel(path, contract, start=0, length=64)
    spectrum = build_input_window_stft(
        path,
        contract,
        start=0,
        cfg=TidmadSTFTConfig(
            n_fft=16, hop_length=8, win_length=16, n_bands_used=4
        ),
        window_samples=64,
    )

    np.testing.assert_array_equal(squid, np.arange(64, dtype=np.float64))
    assert spectrum.shape == (8, 7)
