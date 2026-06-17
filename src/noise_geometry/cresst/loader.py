"""Format-tolerant CRESST pulse-shape loading helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _first_numeric_2d_dataset(group: h5py.Group) -> str | None:
    found = None

    def visitor(name: str, obj) -> None:
        nonlocal found
        if found is None and isinstance(obj, h5py.Dataset) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
            found = name

    group.visititems(visitor)
    return found


def load_cresst_traces(
    path: str | Path,
    *,
    trace_key: str | None = None,
    label_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, str]]:
    """Load row-wise traces and optional labels from NPZ or HDF5."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if trace_key is None:
                candidates = [key for key in data.files if data[key].ndim == 2 and np.issubdtype(data[key].dtype, np.number)]
                if not candidates:
                    raise ValueError(f"no numeric 2D trace array found in {source}")
                trace_key = candidates[0]
            traces = np.asarray(data[trace_key], dtype=np.float64)
            labels = np.asarray(data[label_key]) if label_key and label_key in data.files else None
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(source, "r") as handle:
            trace_key = trace_key or _first_numeric_2d_dataset(handle)
            if trace_key is None:
                raise ValueError(f"no numeric 2D trace dataset found in {source}")
            traces = np.asarray(handle[trace_key], dtype=np.float64)
            labels = np.asarray(handle[label_key]) if label_key and label_key in handle else None
    else:
        raise ValueError(f"unsupported CRESST file type: {source.suffix}")
    if traces.ndim != 2:
        raise ValueError("CRESST traces must have shape (n_traces, n_samples)")
    return traces, labels, {"source": str(source), "trace_key": str(trace_key), "label_key": str(label_key)}
