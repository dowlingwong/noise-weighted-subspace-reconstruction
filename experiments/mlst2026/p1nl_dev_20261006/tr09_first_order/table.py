"""Deterministic TR09 weight table on the actual T25 generator (no sampling).  Writes JSON with
provenance; see MS plan/onthefly/TR09_FIRST_ORDER_NOTE.md section 6."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from ..generator import SparseLatentModel, rng
from . import core


def provenance():
    pkg = Path(__file__).resolve().parents[1]
    root = pkg.parents[2]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(pkg.rglob('*.py')) if '__pycache__' not in p.parts}
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                       env={'GIT_OPTIONAL_LOCKS': '0', 'PATH': '/usr/bin:/bin'}).strip()
    except Exception:          # noqa: BLE001
        head = None
    return {'files': files, 'EXP_HEAD': head, 'python': platform.python_version(), 'numpy': np.__version__}


def build(thetas=(0.0, 0.12, 0.25, 0.40), ps=(1.0, 0.3, 0.1)):
    rows = []
    for th in thetas:
        base = SparseLatentModel(theta=th, p=1.0)
        pt, dist = core.population_product_optimum(base, rng('paper1-tr09-dev', 'population', th), n_random=6)
        fo = core.first_order(base, pt)
        for p in ps:
            m = SparseLatentModel(theta=th, p=p)
            s = core.summary(m, pt, fo)
            s['d_kappa_partial_second_order'] = core.unrestricted_kurtosis_second_order(m)
            if s['N_star_leading_order'] is not None:
                s['N_star_shift_first_order_product'] = -s['K'] / s['a']
                s['N_star_shift_second_order_unrestricted_partial'] = s['d_kappa_partial_second_order'] / (-s['B'])
            s['distinct_population_maxima'] = sorted({round(d['value'], 9) for d in dist})
            s['all_starts_reach_best'] = bool(max(d['gap_to_best'] for d in dist) < 1e-9)
            rows.append(s)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = {'note': 'deterministic; whitened oracle coordinates; see TR09_FIRST_ORDER_NOTE.md', 'rows': build(),
           'provenance': provenance()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in out['rows']:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != 'w_sparse'},
              [round(x, 7) for x in r['w_sparse']])


if __name__ == '__main__':
    main()
