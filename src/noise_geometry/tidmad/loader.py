"""Chunked reader for the TIDMAD HDF5 release.

Design constraints, from ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md``:

* Files hold ~2e9 samples per channel. Nothing here may materialize a full
  channel; every access is windowed and streamed.
* The release documents unequal channel lengths in some files and prescribes
  truncating both channels to the first 2e9 samples. That rule is applied here,
  once, rather than at every call site.
* **Channel semantics are a Phase 0 deliverable, not an assumption.** The two
  channels in the training/validation files carry the SQUID readout and the
  injected reference, but which dataset name is which must be *verified against
  the files* and recorded in ``docs/TIDMAD_DATA_CONTRACT.md`` before any
  scientific number is produced. This module therefore takes the mapping from a
  :class:`TidmadDataContract` and refuses to guess.

``h5py`` is imported lazily so that the rest of the package (and its tests)
remain importable in environments without it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = [
    "TidmadDataContract",
    "TidmadFile",
    "iter_windows",
    "load_contract",
]

#: Release-prescribed truncation: some files have unequal channel lengths and
#: the documented remedy is to use only the first 2e9 samples of both.
DEFAULT_MAX_SAMPLES = 2_000_000_000


@dataclass(frozen=True)
class TidmadDataContract:
    """Verified facts about the release. Filled in during Phase 0.

    Every field here is something that was *checked against the files*, not
    inferred from documentation. ``verified`` must be set to ``True`` only by a
    Phase 0 run that recorded file hashes; downstream analysis code should
    refuse to run against an unverified contract.
    """

    squid_dataset: str
    reference_dataset: str | None
    sampling_frequency_hz: float
    dtype: str
    max_samples: int = DEFAULT_MAX_SAMPLES
    injection_frequencies_hz: tuple[float, ...] = ()
    injection_amplitudes: tuple[float, ...] = ()
    verified: bool = False
    notes: str = ""

    def require_verified(self) -> None:
        if not self.verified:
            raise RuntimeError(
                "TIDMAD data contract is not marked verified. Run the Phase 0 "
                "inspection (scripts/tidmad_phase0_contract.py) and record the "
                "result in docs/TIDMAD_DATA_CONTRACT.md before producing "
                "scientific numbers."
            )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def load_contract(path: str | Path) -> TidmadDataContract:
    """Load a contract written by the Phase 0 inspection."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["injection_frequencies_hz"] = tuple(payload.get("injection_frequencies_hz", ()))
    payload["injection_amplitudes"] = tuple(payload.get("injection_amplitudes", ()))
    return TidmadDataContract(**payload)


class TidmadFile:
    """Lazy, windowed access to one TIDMAD HDF5 file.

    Use as a context manager::

        with TidmadFile(path, contract) as f:
            for w in f.iter_squid_windows(1 << 20, stride=1 << 20, limit=64):
                ...
    """

    def __init__(self, path: str | Path, contract: TidmadDataContract):
        self.path = Path(path)
        self.contract = contract
        self._handle = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "TidmadFile":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "h5py is required to read TIDMAD files; install it with "
                "`uv sync --extra tidmad` or `pip install h5py`."
            ) from exc
        self._handle = h5py.File(self.path, "r")
        return self

    def __exit__(self, *exc_info) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _dataset(self, name: str):
        if self._handle is None:
            raise RuntimeError("TidmadFile must be used as a context manager")
        if name not in self._handle:
            raise KeyError(
                f"dataset {name!r} not found in {self.path.name}; available: "
                f"{list(self._handle.keys())}"
            )
        return self._handle[name]

    # -- introspection (Phase 0) ------------------------------------------
    def describe(self) -> dict:
        """Return the structural facts Phase 0 must record."""
        if self._handle is None:
            raise RuntimeError("TidmadFile must be used as a context manager")
        out = {"file": self.path.name, "datasets": {}}
        for key in self._handle.keys():
            item = self._handle[key]
            shape = getattr(item, "shape", None)
            out["datasets"][key] = {
                "shape": None if shape is None else [int(s) for s in shape],
                "dtype": str(getattr(item, "dtype", "")),
            }
        return out

    # -- data access -------------------------------------------------------
    @property
    def n_samples(self) -> int:
        ds = self._dataset(self.contract.squid_dataset)
        return int(min(ds.shape[0], self.contract.max_samples))

    def read(self, dataset: str, start: int, length: int) -> np.ndarray:
        """Read ``length`` samples starting at ``start`` as float64."""
        ds = self._dataset(dataset)
        stop = min(int(start) + int(length), int(min(ds.shape[0], self.contract.max_samples)))
        if stop <= int(start):
            return np.empty(0, dtype=np.float64)
        return np.asarray(ds[int(start) : stop], dtype=np.float64)

    def iter_squid_windows(
        self, window_samples: int, *, stride: int | None = None, start: int = 0, limit: int | None = None
    ) -> Iterator[np.ndarray]:
        """Yield successive SQUID-channel windows."""
        yield from iter_windows(
            self, self.contract.squid_dataset, window_samples, stride=stride, start=start, limit=limit
        )

    def iter_paired_windows(
        self, window_samples: int, *, stride: int | None = None, start: int = 0, limit: int | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(squid, reference)`` window pairs.

        The reference channel is the ground truth that makes Phase 2 possible:
        it is what lets the comparison be scored against the injected signal
        rather than only against a residual.
        """
        if self.contract.reference_dataset is None:
            raise ValueError(
                "this contract has no reference channel (science files carry one "
                "channel); paired windows are only available for training and "
                "validation files"
            )
        squid = iter_windows(
            self, self.contract.squid_dataset, window_samples, stride=stride, start=start, limit=limit
        )
        ref = iter_windows(
            self, self.contract.reference_dataset, window_samples, stride=stride, start=start, limit=limit
        )
        yield from zip(squid, ref)


def iter_windows(
    handle: TidmadFile,
    dataset: str,
    window_samples: int,
    *,
    stride: int | None = None,
    start: int = 0,
    limit: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield equal-length windows from a dataset without materializing it."""
    n = int(window_samples)
    if n <= 1:
        raise ValueError("window_samples must be greater than one")
    step = int(stride) if stride is not None else n
    if step <= 0:
        raise ValueError("stride must be positive")

    pos = int(start)
    produced = 0
    while limit is None or produced < limit:
        chunk = handle.read(dataset, pos, n)
        if chunk.size < n:
            return
        yield chunk
        produced += 1
        pos += step
