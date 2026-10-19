"""Recompute F4 using committed JSON only; never fit or access recordings.

Run from EXP with PYTHONPATH=. and meg_env Python. Writes POWER_AD8.json
exclusively, leaving the historical implementation, pilot and power files intact.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from experiments.mlst2026.hcp_rnoise_freeze_20261019.power import (
    AD8_COMMIT, AD8_SHA, compute,
)

MS = Path('/ceph/dwong/arXiv-v1-polished-extended-preprint')
BASELINE = '7a78a8f'
PUB = Path('reviews/hcp_rnoise_freeze_20261019')
AD8 = Path('reviews/hcp_rnoise_author_decisions_20261019/AD8_POWER_CELLS.md')
FIRST_F4 = '7143a875dc88e31760fe172a228dfa49b89a8923'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(ms, *args):
    return subprocess.check_output(
        ['git', '-C', str(ms), *args],
        env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'),
    )


def validate_inputs(s20, s50, predictions):
    """Check embedded pilot records, original predictions, and R20 retention."""
    dates = predictions['dates']
    if s20['dates'] != dates or s50['dates'] != dates:
        raise ValueError('STOP dates differ from committed predictions')
    original = {c['cell']: c for c in s20['cells'] if c['arm'] == 'reference'}
    extended = {c['cell']: c for c in s50['cells']}
    for summary in (s20, s50):
        cells = {c['cell']: c for c in summary['cells']}
        for power_cell in summary['power_cells']:
            key = power_cell['cell']
            g, N = power_cell['geometry'], power_cell['N']
            for date in dates:
                geometry = predictions['units'][date]['geometries'][g]
                row = geometry['row']
                if N not in geometry['grid']:
                    raise ValueError('STOP N differs from committed grid')
                expected = row['a_best'] + row['B'] / N
                if power_cell['predicted_by_date'][date] != expected:
                    raise ValueError('STOP prediction differs from committed row')
                if power_cell['certificate_by_date'][date] != row['N_star_interval']:
                    raise ValueError('STOP certificate differs from committed row')
                detail = cells[key]['per_date'][date]
                records = detail['resample_records']
                if [r['replicate'] for r in records] != list(range(summary['resamples'])):
                    raise ValueError('STOP incomplete/duplicate resample ledger')
                mean = float(np.mean([r['D'] for r in records]))
                if not np.isclose(mean, detail['D'], rtol=0, atol=1e-12):
                    raise ValueError('STOP date mean differs from resample ledger')
                if detail['D'] != power_cell['date_means'][date]:
                    raise ValueError('STOP inconsistent power date mean')
    if set(original) != set(extended):
        raise ValueError('STOP reference family changed in remedy')
    for key in original:
        for date in dates:
            before = original[key]['per_date'][date]['resample_records']
            after = extended[key]['per_date'][date]['resample_records']
            if before != after[:20]:
                raise ValueError('STOP R50 changed an original R20 record')


def result(s20, s50, predictions, previous, source_hashes):
    validate_inputs(s20, s50, predictions)
    initial = compute(s20)
    initial['pilot_sha256'] = source_hashes['PILOT_SUMMARY.json']
    if initial['pass']:
        raise ValueError('STOP historical R50 remedy ineligible after R20 pass')
    remedy = compute(s50, after_remedy=True)
    remedy['pilot_sha256'] = source_hashes['PILOT_R50_REF_SUMMARY.json']
    if initial != previous['initial'] or remedy != previous['remedy']:
        raise ValueError('STOP recomputation differs from historical power')
    if remedy['pass']:
        raise ValueError('STOP unexpected historical outcome; inspect before publication')
    controls = []
    for trial in remedy['trials']:
        controls.append({
            'n': trial['n'],
            'all_cells_pass': False,
            'cells': [{
                'cell': c['cell'], 'population_cost_prediction': 0.0,
                'half_abs_prediction': 0.0,
                'simultaneous_half_width': c['simultaneous_half_width'],
                'pass': False,
            } for c in trial['cells'] if c['cell'].startswith('control/')],
        })
    rows = [predictions['units'][d]['geometries']['control']['row']
            for d in predictions['dates']]
    if not all(r['B_dagger']['B_dagger'] == 0 and abs(r['a_best']) < 1e-9 for r in rows):
        raise ValueError('STOP theta0 certificate differs from exact-zero control')
    return {
        'status': 'STOP', 'n_min': None, 'freeze_eligible': False,
        'initial': initial, 'remedy': remedy, 'remedy_applied': True,
        'remedy_kind': previous['remedy_kind'],
        'remedy_execution': 'Reuse committed R50 outputs; no additional fits or remedy run',
        'required_cells': initial['required_cells'],
        'AD8_selection': initial['AD8_selection'],
        'AD8_theta0_wording_flag': {
            'status': 'DISCLOSED; no prediction procedure changed',
            'nominal_basis': 'Unchanged committed finite-N contrast a_best+B/N, not population cost a',
            'signed_wording': 'AD8 describes theta0 prediction as certificate-bounded nonnegative cost',
            'population_cost': 'At theta0 exact cost and Bdagger are zero; a_best residues are numerical',
            'literal_cost_interpretation': {
                'status': 'STOP', 'n_min': None,
                'reason': 'Every theta0 target is zero; strict half-width < zero is impossible',
                'R50_trials': controls,
            },
            'robustness': 'Both readings STOP; adaptive A/1 also fails under the unchanged rule',
        },
        'source_sha256': source_hashes,
        'confirmation_access': False, 'refits': 0,
        'additional_remedy_runs': 0,
        'F5': 'NOT REACHED: mandatory F4 STOP; no protocol or registration bundle',
        'F6': 'Independent F4 STOP review/audit only; no completed freeze review',
        'next_action': 'STOP with report; further scientific changes require author direction',
    }


def publish(ms=MS):
    baseline = git(ms, 'rev-parse', BASELINE).decode().strip()
    git(ms, 'merge-base', '--is-ancestor', AD8_COMMIT, FIRST_F4)
    git(ms, 'merge-base', '--is-ancestor', FIRST_F4, baseline)
    git(ms, 'merge-base', '--is-ancestor', baseline, 'origin/main')
    if sha((ms / AD8).read_bytes()) != AD8_SHA:
        raise ValueError('STOP signed AD8 identity mismatch')
    if git(ms, 'show', f'{AD8_COMMIT}:{AD8}') != (ms / AD8).read_bytes():
        raise ValueError('STOP AD8 differs from signed commit')
    preserved = {}
    paths = git(ms, 'ls-tree', '-r', '--name-only', baseline, str(PUB)).decode().splitlines()
    for path in paths:
        raw = (ms / path).read_bytes()
        if raw != git(ms, 'show', f'{baseline}:{path}'):
            raise ValueError(f'STOP historical record changed: {path}')
        preserved[path] = {'bytes': len(raw), 'sha256': sha(raw)}
    names = ['PILOT_SUMMARY.json', 'PILOT_R50_REF_SUMMARY.json',
             'DEV_PREDICTIONS.json', 'POWER.json', 'POWER_R20.json']
    data = {n: json.loads((ms / PUB / n).read_bytes()) for n in names}
    hashes = {n: sha((ms / PUB / n).read_bytes()) for n in names}
    if any(data[n]['DEV_PREDICTIONS_sha256'] != hashes['DEV_PREDICTIONS.json']
           for n in names[:2]):
        raise ValueError('STOP pilot prediction identity differs')
    output = result(data[names[0]], data[names[1]], data[names[2]], data[names[3]], hashes)
    output['provenance'] = {
        'MS_baseline_commit': baseline, 'AD8_commit': AD8_COMMIT,
        'AD8_sha256': AD8_SHA, 'first_committed_F4_output': FIRST_F4,
        'AD8_is_ancestor_of_first_F4_output': True,
        'historical_numeric_results_reproduced_exactly': True,
        'preserved_existing_files': preserved,
        'recompute_script_sha256': sha(Path(__file__).read_bytes()),
        'EXP_original_implementation_commit': 'b311a919c2cb130dd088e19fc1aec1cdfdb072f5',
    }
    target = ms / PUB / 'POWER_AD8.json'
    with target.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'file': str(target), 'sha256': sha(target.read_bytes()),
                      'status': output['status'], 'required_cells': output['required_cells'],
                      'refits': 0, 'additional_remedy_runs': 0}, indent=2))


if __name__ == '__main__':
    publish()
