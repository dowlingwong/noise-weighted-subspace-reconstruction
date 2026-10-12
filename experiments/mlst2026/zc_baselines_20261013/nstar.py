"""N*_pred table for the N*-rule.

Source used by the runner (registered): the T20-S rows of MS evidence/data/y15_20261012/RESULT.json,
vendored byte-for-byte as pinned/y15_RESULT_20261012.json; its SHA-256 is pinned in protocol.json and
checked by the preflight.  N*_pred = (c_U - c_F)/a - 1 (supplement S4.3, first-order crossing); theta = 0
has a = 0 and no crossing (N*_pred = None -> product always).

Cross-check (kit test, not used by the runner): recomputation through the pinned constants path
(Y19 predictions.constants = TR09 core.py, applied to the T25/T20-S generator
p1nl_dev_20261006.generator.SparseLatentModel(theta, p = 1) with the weight-table restart stream),
agreement to relative 1e-6 (tests/test_nstar.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE / 'pinned' / 'y15_RESULT_20261012.json'
SOURCE_MS_PATH = 'evidence/data/y15_20261012/RESULT.json'
SOURCE_SHA256 = 'e33551b9c077f11911768d9b78d1d1a00c58e5aded7f0b9cf3ceacdacb0beff7'
STUDY_ROWS = 'T20-S'


def source_sha() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def table(path: Path = SOURCE) -> dict:
    """{repr(theta): N*_pred or None} from the T20-S rows; refuses a source with the wrong SHA-256."""
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != SOURCE_SHA256:
        raise ValueError(f'N* source SHA-256 {got} != pinned {SOURCE_SHA256}')
    rows = [r for r in json.loads(Path(path).read_text())['locations'] if r['study'] == STUDY_ROWS]
    out = {}
    for r in rows:
        out[repr(float(r['theta']))] = None if r['N_star_pred'] is None else float(r['N_star_pred'])
    return out


def recompute(thetas=(0.0, 0.12, 0.25, 0.40)) -> dict:
    """Constants-path recomputation (test only; imports the Y19 predictions module and the p1nl generator)."""
    from experiments.mlst2026.p1nl_dev_20261006.generator import SparseLatentModel, rng
    from experiments.mlst2026.y19_20261012 import predictions as P
    out = {}
    for th in thetas:
        row = P.constants(SparseLatentModel(theta=th, p=1.0), rng('paper1-tr09-dev', 'population', th), n_random=6)
        out[repr(float(th))] = row['N_star_pred']
    return out
