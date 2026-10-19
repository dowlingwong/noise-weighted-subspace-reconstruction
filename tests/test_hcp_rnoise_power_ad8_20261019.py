"""Fabricated JSON: immutable prediction/replicate provenance and STOP safety."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'ad8_recompute', Path(__file__).parents[1] / 'scripts/hcp_rnoise_power_ad8_20261019.py')
ad8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad8)


def fabricated():
    dates = [f'd{i}' for i in range(9)]
    grids = {'control': [56, 111, 222, 444, 888],
             'A': [56, 111, 222, 444, 888], 'B': [28, 56, 111, 222, 444]}
    pred = {'dates': dates, 'units': {d: {'geometries': {
        g: {'grid': grid, 'row': {
            'a_best': 0.0 if g == 'control' else 1.0,
            'B': -111.0 if g == 'B' else -222.0,
            'N_star_interval': None if g == 'control' else [111.0 if g == 'B' else 222.0, 1000.0],
            'B_dagger': {'B_dagger': 0.0 if g == 'control' else .2},
        }} for g, grid in grids.items()}} for d in dates}}

    def summary(R):
        cells, power = [], []
        for g, grid in grids.items():
            for i, N in enumerate(grid):
                key = f'{g}/{i}/reference'
                per = {}
                pv, cert = {}, {}
                for j, d in enumerate(dates):
                    records = [{'replicate': r, 'D': 5.0 * j + r / 100} for r in range(R)]
                    per[d] = {'D': float(np.mean([r['D'] for r in records])),
                              'resample_records': records}
                    row = pred['units'][d]['geometries'][g]['row']
                    pv[d] = row['a_best'] + row['B'] / N
                    cert[d] = row['N_star_interval']
                means = {d: per[d]['D'] for d in dates}
                cells.append({'cell': key, 'arm': 'reference', 'per_date': per})
                power.append({'cell': key, 'geometry': g, 'N': N, 'date_means': means,
                              'predicted_by_date': pv, 'certificate_by_date': cert,
                              'between_date_sd': float(np.std(list(means.values()), ddof=1)),
                              'predicted_contrast': float(np.mean(list(pv.values())))})
        return {'label': 'DEV PILOT; not confirmatory', 'dates': dates,
                'n_dev': 9, 'resamples': R, 'family_size': 34, 'cells': cells, 'power_cells': power}
    return summary(20), summary(50), pred


def test_reuses_one_extension_and_discloses_both_stop_readings():
    s20, s50, pred = fabricated()
    inputs = copy.deepcopy((s20, s50, pred))
    previous = {'initial': ad8.compute(s20), 'remedy': ad8.compute(s50, True),
                'remedy_kind': 'R50'}
    hashes = {'PILOT_SUMMARY.json': '20', 'PILOT_R50_REF_SUMMARY.json': '50'}
    previous['initial']['pilot_sha256'] = '20'
    previous['remedy']['pilot_sha256'] = '50'
    result = ad8.result(s20, s50, pred, previous, hashes)
    assert (s20, s50, pred) == inputs
    assert result['status'] == 'STOP' and not result['freeze_eligible']
    assert result['refits'] == result['additional_remedy_runs'] == 0
    assert result['AD8_theta0_wording_flag']['literal_cost_interpretation']['n_min'] is None
    assert len(result['required_cells']) == 9


@pytest.mark.parametrize('corruption', ['dates', 'grid', 'prediction', 'certificate',
                                       'replicate', 'mean', 'power_mean', 'R20_prefix'])
def test_rejects_changed_predictions_or_original_replicates(corruption):
    s20, s50, pred = fabricated()
    if corruption == 'dates':
        s50['dates'] = s50['dates'][:-1]
    elif corruption == 'grid':
        pred['units']['d0']['geometries']['control']['grid'] = []
    elif corruption == 'prediction':
        s20['power_cells'][0]['predicted_by_date']['d0'] += 1
    elif corruption == 'certificate':
        s50['power_cells'][0]['certificate_by_date']['d0'] = [0, 1]
    elif corruption == 'replicate':
        s50['cells'][0]['per_date']['d0']['resample_records'].pop()
    elif corruption == 'mean':
        s20['cells'][0]['per_date']['d0']['D'] += 1
    elif corruption == 'power_mean':
        s20['power_cells'][0]['date_means']['d0'] += 1
    elif corruption == 'R20_prefix':
        s50['cells'][0]['per_date']['d0']['resample_records'][0]['diagnostic_changed'] = True
    with pytest.raises(ValueError, match='STOP'):
        ad8.validate_inputs(s20, s50, pred)
