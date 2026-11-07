"""Deterministic row selection and array loading for X-OPT (shared by all stages)."""
from __future__ import annotations

import json

import numpy as np

from .config import ROOT
IN_DEFAULT = ROOT + "/inputs/hcp_confirmation_20261020"


def scan(fits_path, wanted=None):
    """Stream FITS.jsonl; yield (offset, metadata) for every row (or only `wanted` scoped rows)."""
    off = 0
    with open(fits_path, 'rb') as fh:
        for line in fh:
            rec = json.loads(line)
            k = rec['kernel']
            meta = dict(offset=off, date=rec['date'], geometry=rec['geometry'], N=rec['N'],
                        arm=rec['arm'], m=rec['m'], replicate=rec['replicate'],
                        date_index=k['date_index'], theta=k['theta'],
                        lim=bool(k['optimization_limited']),
                        lim_t25=bool(k['optimization_limited_t25_rule']),
                        selected=k['selected'], status=k['selected_status'],
                        objective=k['restarts'][k['selected']]['objective'], D=rec['D'])
            off += len(line)
            if wanted is None or wanted(meta):
                yield meta


def read_row(fits_path, offset):
    with open(fits_path, 'rb') as fh:
        fh.seek(offset)
        return json.loads(fh.readline())


def pick_rows(fits_path, seed=20261106, per_date=2, max_rows=None):
    """2 rows per included date: one from each optimization_limited stratum, seeded.

    Recorded seed 20261106; the choice is fixed before any strengthened fit is computed.
    """
    by_date = {}
    for meta in scan(fits_path):
        by_date.setdefault(meta['date'], []).append(meta)
    rng = np.random.default_rng(seed)
    chosen = []
    for date in sorted(by_date):
        block = by_date[date]
        picks = []
        for pool in ([r for r in block if r['lim']], [r for r in block if not r['lim']]):
            if pool:
                picks.append(pool[int(rng.integers(len(pool)))])
        while len(picks) < per_date:
            rest = [r for r in block if r not in picks]
            picks.append(rest[int(rng.integers(len(rest)))])
        chosen.extend(picks[:per_date])
        if max_rows and len(chosen) >= max_rows:
            return chosen[:max_rows]
    return chosen


def load_inputs(date, root=IN_DEFAULT):
    roles = {}
    for role in ('ref', 'train', 'cal', 'eval'):
        with np.load(f'{root}/arrays/{date}/{role}.npz', allow_pickle=False) as a:
            roles[role] = (a['X'], {k: a[k] for k in a.files if k != 'X'})
    with open(f'{root}/arrays/{date}/Sref.json', 'rb') as fh:
        Sref = np.asarray(json.loads(fh.read()), float)
    return roles, Sref
