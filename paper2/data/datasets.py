"""Dataset contracts for Paper 2 reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

from paper2._torch import Tensor, require_torch, torch


@dataclass(slots=True)
class ReconstructionBatch:
    """One mini-batch in native detector coordinates."""

    x: Tensor
    meta: dict[str, Tensor | Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    trace_path: str
    rq_path: str | None
    trace_len: int
    n_channels: int
    batch_size: int
    pretrigger: int = 4000
    baseline_correct: bool = True
    max_events: int | None = None
    num_workers: int = 0
    pin_memory: bool = False
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 314159


class ReconstructionDataset:
    """Placeholder dataset API.

    Concrete implementation should:

    - read traces from HDF5 or numpy-backed artifacts
    - return native-space waveforms `(C, T)`
    - attach metadata such as amplitude, `t0`, position, and shape when
      available
    """

    def __init__(
        self,
        traces,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.traces = traces
        self.metadata = {} if metadata is None else metadata

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, index: int) -> ReconstructionBatch:
        require_torch()
        x = torch.as_tensor(self.traces[index], dtype=torch.float32)
        meta = {
            key: value[index] if hasattr(value, "__getitem__") else value
            for key, value in self.metadata.items()
        }
        return ReconstructionBatch(x=x, meta=meta)


def baseline_correct_traces(
    traces: np.ndarray,
    pretrigger: int,
) -> tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces, dtype=np.float64)
    baseline = np.mean(traces[..., :pretrigger], axis=-1, keepdims=True)
    return traces - baseline, np.squeeze(baseline, axis=-1)


def _load_rqs(rq_path: str | None) -> pd.DataFrame | None:
    if rq_path is None:
        return None
    with h5py.File(rq_path, "r") as handle:
        if "rqs" in handle:
            return pd.DataFrame.from_records(handle["rqs"][:])
        raise KeyError(f"No 'rqs' dataset found in {rq_path}")


def load_reconstruction_dataset(cfg: DatasetConfig) -> ReconstructionDataset:
    """Load the full dataset.

    Current implementation targets the available K-alpha single-channel files:

    - `data/k_alpha_traces.h5` with dataset `traces` of shape `(N, T)`
    - `data/k_alpha_rqs.h5` with dataset `rqs`
    """
    with h5py.File(cfg.trace_path, "r") as handle:
        if "traces" not in handle:
            raise KeyError(f"No 'traces' dataset found in {cfg.trace_path}")
        traces = np.asarray(handle["traces"][:], dtype=np.float32)

    if cfg.max_events is not None and cfg.max_events > 0:
        traces = traces[: cfg.max_events]

    if traces.ndim != 2:
        raise ValueError(f"Expected single-channel traces with shape (N, T), got {traces.shape}")
    if traces.shape[1] != cfg.trace_len:
        raise ValueError(f"Trace length mismatch: file has {traces.shape[1]}, config expects {cfg.trace_len}")

    if cfg.baseline_correct:
        traces, baselines = baseline_correct_traces(traces, cfg.pretrigger)
    else:
        baselines = np.zeros(traces.shape[0], dtype=np.float32)

    traces = traces[:, None, :]
    if cfg.n_channels != traces.shape[1]:
        raise ValueError(
            f"Current dataset loader provides {traces.shape[1]} channel(s), "
            f"but config requested n_channels={cfg.n_channels}"
        )

    rqs = _load_rqs(cfg.rq_path)
    metadata: dict[str, Any] = {"baseline_mean": baselines.astype(np.float32)}
    if rqs is not None:
        if cfg.max_events is not None and cfg.max_events > 0:
            rqs = rqs.iloc[: cfg.max_events].reset_index(drop=True)
        metadata["amplitude"] = rqs["A"].to_numpy(dtype=np.float32)
        metadata["t0"] = rqs["time_shift"].to_numpy(dtype=np.float32)
        metadata["of_amplitude"] = rqs["OF_ampl_0"].to_numpy(dtype=np.float32)
        metadata["of_time"] = rqs["OF_time_0"].to_numpy(dtype=np.float32)
        metadata["trace_index"] = rqs["trace_index"].to_numpy(dtype=np.int64)
    return ReconstructionDataset(traces=traces, metadata=metadata)


def collate_reconstruction_batches(
    batch: Iterable[ReconstructionBatch],
) -> ReconstructionBatch:
    require_torch()
    batch = list(batch)
    xs = torch.stack([item.x for item in batch], dim=0)
    meta: dict[str, list[Any]] = {}
    for item in batch:
        for key, value in item.meta.items():
            meta.setdefault(key, []).append(value)
    meta_out: dict[str, Any] = {}
    for key, values in meta.items():
        first = values[0]
        if isinstance(first, (int, float, np.integer, np.floating)):
            meta_out[key] = torch.as_tensor(values)
        else:
            meta_out[key] = values
    return ReconstructionBatch(x=xs, meta=meta_out)
