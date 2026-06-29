"""Format-tolerant CRESST pulse-shape loading helpers.

Two entry points:

- ``load_cresst_traces``: generic NPZ/HDF5 row-wise trace loader.
- ``load_cresst_release``: the public DMDC CRESST-II/III pulse-shape release
  (``X_{split}.npy`` + ``features_{split}.csv``), which carries explicit
  ``noise`` (random-triggered baseline) and ``clean`` (survived quality cuts)
  flags. See arXiv:2508.03078.
"""

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


def _coerce_bool(series) -> np.ndarray:
    """Coerce a CSV column (bool, 0/1, or 'True'/'False' text) to a bool array."""
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "1.0", "yes"}).to_numpy()


def load_cresst_release(
    raw_dir: str | Path,
    *,
    split: str = "test",
    mmap: bool = True,
    max_traces: int | None = None,
    seed: int = 22,
) -> tuple[np.ndarray, "object", dict[str, object]]:
    """Load the public DMDC CRESST pulse-shape release for one split.

    Parameters
    ----------
    raw_dir : path
        The ``.../cresst/raw`` directory containing the released files.
    split : {"train", "test"}
        Which released split to load.
    mmap : bool
        Memory-map ``X_{split}.npy`` instead of reading it fully. The full
        training array is ~2 GB, so this matters on shared servers.
    max_traces : int or None
        If set and smaller than the split, draw a reproducible random subset.

    Returns
    -------
    traces : np.ndarray, shape (n, 512)
        Raw voltage traces (float64). A subset is materialised; the full set is
        returned as a memory-map when ``mmap`` and ``max_traces`` is None.
    features : pandas.DataFrame
        Per-record features aligned row-for-row with ``traces`` (includes
        ``noise``, ``clean``, ``run``, ``channel``, ``pulse_height``).
    meta : dict
        Source paths and selection metadata.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("pandas is required for load_cresst_release") from exc

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(raw_dir)
    x_path = root / f"X_{split}.npy"
    feat_path = root / f"features_{split}.csv"
    if not x_path.exists() or not feat_path.exists():
        raise FileNotFoundError(
            f"Missing CRESST release files for split '{split}' under {root}. "
            "Run scripts/download/download_cresst.py first."
        )

    traces = np.load(x_path, mmap_mode="r" if mmap else None)
    features = pd.read_csv(feat_path)
    if traces.shape[0] != len(features):
        raise ValueError(
            f"row mismatch: {x_path.name} has {traces.shape[0]} rows, "
            f"{feat_path.name} has {len(features)}"
        )

    n = traces.shape[0]
    indices = np.arange(n)
    if max_traces is not None and max_traces < n:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(n, size=int(max_traces), replace=False))
        traces = np.asarray(traces[indices], dtype=np.float64)
        features = features.iloc[indices].reset_index(drop=True)
    elif not mmap:
        traces = np.asarray(traces, dtype=np.float64)

    meta = {
        "source_traces": str(x_path),
        "source_features": str(feat_path),
        "split": split,
        "n_total": int(n),
        "n_selected": int(traces.shape[0]),
    }
    return traces, features, meta


def select_cresst_subsets(
    traces: np.ndarray,
    features: "object",
    *,
    noise_column: str = "noise",
    clean_column: str = "clean",
) -> dict[str, np.ndarray]:
    """Split release traces into noise-only and clean-pulse subsets.

    - ``noise`` traces (random-triggered baselines) estimate the noise
      covariance / PSD.
    - ``clean`` non-noise traces are accepted particle pulses for
      reconstruction.
    """
    is_noise = _coerce_bool(features[noise_column])
    is_clean = _coerce_bool(features[clean_column])
    pulses = is_clean & ~is_noise
    return {
        "noise_traces": np.asarray(traces[is_noise], dtype=np.float64),
        "pulse_traces": np.asarray(traces[pulses], dtype=np.float64),
        "n_noise": int(is_noise.sum()),
        "n_pulse": int(pulses.sum()),
    }
