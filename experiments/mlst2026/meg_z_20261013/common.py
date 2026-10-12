"""Shared conventions of the meg_z kit.  Nothing runs on import.

Record layout (identical to T25/Y19): a noise window or a record is a C x T array Z (channels x time),
vectorised by column stacking, index c + C t (Fortran order), so that vec(A Z B) = (B^T (x) A) vec(Z),
a product subspace is range(U_t (x) U_c) and a separable covariance is np.kron(S_t, S_c).
Windows are stored as (n, C, T) arrays; ``vec`` turns them into (n, d) records, d = C T.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
EXP_ROOT = PACKAGE.parents[2]

#: development seed namespace (dev subjects only); the confirmation namespace is fixed at the D4 freeze
NAMESPACE_DEV = 'megz-dev-20261013'
NAMESPACE_SMOKE = 'megz-smoke-fixture-20261013'


def vec(Z: np.ndarray) -> np.ndarray:
    """(..., C, T) -> (..., C*T) with index c + C t."""
    Z = np.asarray(Z)
    C, T = Z.shape[-2:]
    return np.swapaxes(Z, -1, -2).reshape(Z.shape[:-2] + (C * T,))


def unvec(x: np.ndarray, C: int, T: int) -> np.ndarray:
    """(..., C*T) -> (..., C, T); inverse of ``vec``."""
    x = np.asarray(x)
    return np.swapaxes(x.reshape(x.shape[:-1] + (T, C)), -1, -2)


def channel_indices(keep: list[int], C: int, T: int) -> np.ndarray:
    """Record coordinates (c + C t) of the channels ``keep`` in the order (c fastest), for slicing a
    d x d moment matrix when channels are dropped (new index c' + C' t)."""
    keep = list(keep)
    return np.array([c + C * t for t in range(T) for c in keep], dtype=int)


def seed_int(*key) -> int:
    """64-bit integer from the SHA-256 of the '/'-joined key (the t25.science.rng construction)."""
    return int.from_bytes(hashlib.sha256('/'.join(map(str, key)).encode()).digest()[:8], 'little')


def rng(*key) -> np.random.Generator:
    return np.random.default_rng(seed_int(*key))


def symmetric(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2


def sha_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha_array(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype.str).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False, default=_default))
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f'not JSON serialisable: {type(o)}')


def load_json(path):
    return json.loads(Path(path).read_text())


def theta_tag(theta: float) -> str:
    return f'theta{float(theta):.2f}'
