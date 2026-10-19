"""AD8a/AD9 power projection from immutable nine-date JSON only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2, t
from experiments.mlst2026.hcp_rnoise_freeze_20261019.confirmation_guard import git
from experiments.mlst2026.hcp_rnoise_freeze_20261019.power import compute

BASELINE = 'e58f315'
FREEZE = Path('reviews/hcp_rnoise_freeze_20261019')
DECISIONS = Path('reviews/hcp_rnoise_author_decisions_20261019')
REQUIRED = ['A/0/reference', 'A/1/reference', 'B/0/reference', 'B/1/reference']
SOURCES = [FREEZE / n for n in ('PILOT_SUMMARY.json', 'PILOT_R50_REF_SUMMARY.json',
           'DEV_PREDICTIONS.json', 'POWER.json', 'POWER_R20.json', 'POWER_AD8.json')]
SOURCES += [DECISIONS / 'AD8_POWER_CELLS.md', DECISIONS / 'AD8a_AD9_AMENDMENT.md']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


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


def project(s20, s50, predictions):
    validate_inputs(s20, s50, predictions)
    # Also recheck every SD, predicted mean and complete ordered 15-cell family.
    historical20, historical50 = compute(s20), compute(s50, after_remedy=True)
    chosen = [r['cell'] for r in historical50['AD8_selection']
              if r['required'] and not r['cell'].startswith('control/')]
    if chosen != REQUIRED or historical20['pass'] or historical50['pass']:
        raise ValueError('STOP historical outcome or prediction-only selection changed')
    cells = {c['cell']: c for c in s50['power_cells']}
    multiplier = float(np.sqrt(8 / chi2.ppf(.2, 8)))
    evidence = []
    trials = []
    for n in range(8, 61):
        critical = float(t.ppf(1 - .05 / (2 * 34), n - 1))
        rows = []
        for key in REQUIRED:
            c = cells[key]
            target = .5 * abs(c['predicted_contrast'])
            half = critical * c['between_date_sd'] * multiplier / np.sqrt(n)
            rows.append({'cell': key, 'simultaneous_half_width': float(half),
                         'half_abs_prediction': target, 'pass': bool(half < target),
                         'margin_percent': float(100 * (target - half) / target)})
        trials.append({'n': n, 'df': n - 1, 't_critical': critical,
                       'all_cells_pass': all(r['pass'] for r in rows), 'cells': rows})
    joint = next((r['n'] for r in trials if r['all_cells_pass']), None)
    for key in REQUIRED:
        c = cells[key]
        evidence.append({**c, 'pilot_sd_upper80': c['between_date_sd'] * multiplier,
                         'n_min': next((r['n'] for r in trials
                                        if next(x for x in r['cells'] if x['cell'] == key)['pass']), None)})
    binding = [r['cell'] for r in evidence if r['n_min'] == joint]
    return {'schema': 'hcp-rnoise-power-ad9-v1', 'label': 'DEV PILOT POWER PROJECTION; not confirmation evidence',
            'status': 'PASS' if joint is not None else 'STOP', 'freeze_eligible': joint is not None,
            'n_min': joint, 'binding_cells': binding, 'n_range': [8, 60], 'confirmation_target': 60,
            'resamples': 50, 'n_dev': 9, 'pilot_dates': s50['dates'], 'family_size': 34,
            'family': '15 reference + 4 PZ4 paired interactions + 15 isotropic', 'alpha': .05,
            'SD_upper_confidence': .8, 'SD_upper_df': 8, 'SD_upper_multiplier': multiplier,
            'required_cells': REQUIRED, 'per_cell_evidence': evidence, 'trials': trials,
            'power_at_n_min': next((r for r in trials if r['n'] == joint), None),
            'power_at_60': trials[-1], 'n17_failure': next(r for r in trials if r['n'] == 17),
            'rule': 'strict t_(1-.05/(2*34),n-1)*SD*sqrt(8/chi2.ppf(.2,8))/sqrt(n) < .5*abs(mean(a_best+B/N)), every AD8a-required cell',
            'STOP_rule': 'Fewer than n_min confirmation units passing G1, G2, amended AD2 and budgets -> STOP; no confirmatory claim; exclusions reported, never replaced',
            'historical_STOP': '7a78a8f remains binding for the 12-date design; A/1 still fails n=12',
            'limitation': 'Nine-date R50 SDs are assumed to transport to 60 dates spanning 2013–2015; one pilot date is 2013. This precision projection is not a guaranteed rejection probability.',
            'confirmation_access': False, 'refits': 0, 'new_resamples': 0}


def from_committed(ms):
    data, hashes = {}, {}
    for path in SOURCES:
        raw = git(ms, 'show', f'{BASELINE}:{path}')
        if raw != (Path(ms) / path).read_bytes():
            raise ValueError('STOP committed source changed: ' + str(path))
        hashes[str(path)] = sha(raw)
        if path.suffix == '.json':
            data[path.name] = json.loads(raw)
    if any(data[n]['DEV_PREDICTIONS_sha256'] != hashes[str(FREEZE / 'DEV_PREDICTIONS.json')]
           for n in ('PILOT_SUMMARY.json', 'PILOT_R50_REF_SUMMARY.json')):
        raise ValueError('STOP pilot prediction identity differs')
    out = project(data['PILOT_SUMMARY.json'], data['PILOT_R50_REF_SUMMARY.json'], data['DEV_PREDICTIONS.json'])
    out['source_sha256'] = hashes
    out['MS_authority_commit'] = git(ms, 'rev-parse', BASELINE).decode().strip()
    return out
