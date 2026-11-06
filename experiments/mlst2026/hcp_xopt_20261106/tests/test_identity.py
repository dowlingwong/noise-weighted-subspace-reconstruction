"""Identity test: `kernel.assemble` must reproduce the frozen route's captured inputs bit-exactly.

The frozen route is the kit's own `learning.fit_cell`; its metric W, training moment S and
start list are captured at the call.  `assemble` (used only by timing runs, which must not
build a clean-risk object) re-derives the same quantities.  This test proves they agree
bit-for-bit, so the timing estimate is computed on the same objects as the production route.
"""
from __future__ import annotations

import sys

import numpy as np

from experiments.mlst2026.hcp_rnoise_20261019.signal import build
from experiments.mlst2026.hcp_xopt_20261106 import kernel, select


def check(n_rows=3):
    metas = select.pick_rows(select.IN_DEFAULT + '/FITS.jsonl', seed=11, per_date=1, max_rows=n_rows)
    bad = 0
    for meta in metas:
        roles, Sref = select.load_inputs(meta['date'])
        model = build(Sref, float(meta['theta']))
        _, cap_s, cap_m = kernel.frozen_route(roles, model, Sref, meta['N'], meta['m'], meta['arm'],
                                              int(meta['date_index']), meta['replicate'])
        asm = kernel.assemble(roles, model, Sref, meta['N'], meta['m'], meta['arm'],
                              int(meta['date_index']), meta['replicate'])
        checks = {
            'W': np.array_equal(asm['W'], cap_s['W']),
            'S': np.array_equal(asm['S'], cap_s['S']),
            'mu': np.array_equal(asm['mu'], cap_m['mu']),
            'n_starts': len(asm['initial']) == len(cap_s['initial']),
            'starts': all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                          for a, b in zip(asm['initial'], cap_s['initial'])),
            'indices': asm['sampled_training_indices'] == cap_s.get('sampled_indices', asm['sampled_training_indices']),
        }
        ok = all(checks.values())
        bad += 0 if ok else 1
        print(meta['date'], meta['geometry'], meta['N'], meta['arm'], meta['m'],
              meta['replicate'], 'OK' if ok else f'FAIL {checks}')
    print(f'identity test: {len(metas) - bad}/{len(metas)} rows bit-identical')
    return bad


if __name__ == '__main__':
    sys.exit(1 if check() else 0)
