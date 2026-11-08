"""Outcome-blind deterministic resume-4 scope row lists, seed 20261108.

S_full:    all 135,000 rows.
S_primary: all reference and isotropic rows + a 10 % stratified sample (date x cell x m) of
           each estimated-metric (arm,m) setting.
S_core:    all reference rows + a 20 % stratified isotropic sample + a 5 % stratified sample
           of each estimated-metric setting.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from . import select
from .config import ROOT

SEED = 20261108
COUNTS = {'S_full': 135000, 'S_primary': 54000, 'S_core': 31500}
ARM_SETTINGS = (('reference', None), ('isotropic', None), ('lw', 256), ('lw', 512),
                ('kron', 256), ('kron', 512))
FIELDS = ('offset', 'date', 'geometry', 'N', 'arm', 'm', 'replicate', 'date_index', 'theta')


def identity(r):
    return (r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m'])


def rows_for_scope(scope, fits=None):
    if scope not in COUNTS:
        raise ValueError('unknown scope')
    groups = defaultdict(list)
    for row in select.scan(fits or select.IN_DEFAULT + '/FITS.jsonl'):
        clean = {k: row[k] for k in FIELDS}
        key = (row['date'], row['geometry'], row['N'], row['arm'], row['m'])
        groups[key].append(clean)
    if len(groups) != 30 * 15 * 6:
        raise ValueError('incomplete original scope strata')
    keys = sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], str(k[4])))
    rng = np.random.default_rng(SEED)
    extras = set()
    if scope == 'S_core':
        for arm, m in ARM_SETTINGS[2:]:
            strata = [k for k in keys if (k[3], k[4]) == (arm, m)]
            if len(strata) != 450:
                raise ValueError('estimated arm strata mismatch')
            extras.update(strata[i] for i in rng.choice(len(strata), len(strata) // 2, replace=False))
    output = []
    for key in keys:
        rows = sorted(groups[key], key=lambda r: r['replicate'])
        if [r['replicate'] for r in rows] != list(range(50)):
            raise ValueError('original replicate census')
        arm = key[3]
        if scope == 'S_full':
            n = 50
        elif arm == 'reference' or (scope == 'S_primary' and arm == 'isotropic'):
            n = 50
        elif scope == 'S_primary':
            n = 5
        elif arm == 'isotropic':
            n = 10
        else:
            n = 2 + int(key in extras)
        chosen = set(range(50)) if n == 50 else set(rng.choice(50, n, replace=False).tolist())
        output.extend(r for r in rows if r['replicate'] in chosen)
    output.sort(key=lambda r: r['offset'])
    if len(output) != COUNTS[scope] or len({identity(r) for r in output}) != len(output):
        raise ValueError('scope census')
    return output


def write_rows(scope, path):
    rows = rows_for_scope(scope)
    blob = ''.join(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n' for r in rows).encode()
    Path(path).write_bytes(blob)
    return dict(scope=scope, seed=SEED, rows=len(rows), path=str(path),
                sha256=hashlib.sha256(blob).hexdigest())


if __name__ == '__main__':
    import sys
    scope = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else str(Path(ROOT) / 'out/resume4/scopes' / f'{scope}_rows.jsonl')
    print(json.dumps(write_rows(scope, path), sort_keys=True, indent=2))
