"""Bind the unchanged prediction kernel's complete record to future array fitting."""
import hashlib
import re
import numpy as np
from .protocol import load
from experiments.mlst2026.hcp_rnoise_freeze_20261019.prediction import predict_unit, annotate
from experiments.mlst2026.hcp_rnoise_20261019.readings import pz1 as inherited_pz1

GRIDS = {'control': [56, 111, 222, 444, 888], 'A': [56, 111, 222, 444, 888],
         'B': [28, 56, 111, 222, 444]}
ROW_KEYS = {'theta', 'a_best', 'cU', 'cF', 'B', 'B_cert', 'Delta_k', 'trace_K',
    'H_min_eig', 'H_max_eig', 'gradient_norm_at_optimum', 'n_starts',
    'distinct_population_maxima', 'all_starts_reach_best', 'starts_reaching_best',
    'start_statuses', 'path', 'N_star_pred', 'N_star_interval', 'checks', 'B_dagger',
    'prestated_PZ2_miss_reasons', 'restart_agreement', 'PZ2'}


def sref_hash(Sref):
    S = np.asarray(Sref, dtype='<f8')
    if S.shape != (72, 72) or not np.isfinite(S).all():
        raise PermissionError('STOP finite d72 Sref required')
    np.linalg.cholesky(S)
    return hashlib.sha256(S.tobytes()).hexdigest()


def validate_record(record, date):
    """Schema/identity validation, not a refit or a population reoptimization."""
    try:
        dates = load()['access']['confirmation_analysis_dates']
        if (record['date'] != date or record['pre_fit'] is not True or
                type(record['date_index']) is not int or record['date_index'] != 12 + dates.index(date) or
                not re.fullmatch('[0-9a-f]{64}', record['Sref_sha256_f64le'])):
            raise ValueError('date, index, Sref or pre_fit identity')
        if record['grids']['P'] != 1111 or record['grids']['A_grid'] != GRIDS['A'] or record['grids']['B_grid'] != GRIDS['B']:
            raise ValueError('fixed prediction grids')
        fixed = record['fixed']
        if len(fixed) != 4 or [r['theta'] for r in fixed] != [0., .12, .25, .4]:
            raise ValueError('complete four fixed rows')
        if set(record['adaptive']) != {'A', 'B'} or set(record['geometries']) != set(GRIDS):
            raise ValueError('complete adaptive/geometries family')
        rows = [(r, sorted(set(GRIDS['A']+GRIDS['B']))) for r in fixed]
        for g in GRIDS:
            geo = record['geometries'][g]
            if geo['grid'] != GRIDS[g]: raise ValueError('geometry grid')
            if g == 'control':
                if geo['theta'] != 0. or geo['row'] != fixed[0]: raise ValueError('theta0 row')
                continue
            adaptive = record['adaptive'][g]
            if type(adaptive['root']) is not bool or type(adaptive['available']) is not bool:
                raise ValueError('explicit adaptive availability')
            if adaptive['root']:
                rows.append((adaptive['row'], GRIDS[g]))
                if (geo['row'] != adaptive['row'] or geo['theta'] != adaptive['theta'] or
                        adaptive['row']['theta'] != adaptive['theta'] or geo['available'] != adaptive['available'] or
                        adaptive['available'] != adaptive['row']['PZ2']['available']):
                    raise ValueError('adaptive row/geometry mismatch')
            elif adaptive['available'] or geo['available'] or not adaptive.get('reason') or not geo.get('reason') or geo.get('row') is not None:
                raise ValueError('unavailable adaptive row must retain reason')
        for row,grid in rows:
            if not ROW_KEYS.issubset(row) or row['n_starts'] != 64:
                raise ValueError('complete population diagnostics and64 starts required')
            numeric = [row[k] for k in ('theta', 'a_best', 'cU', 'cF', 'B', 'B_cert', 'Delta_k',
                       'trace_K', 'H_min_eig', 'H_max_eig', 'gradient_norm_at_optimum')]
            if not np.isfinite(numeric).all() or not isinstance(row['checks'], dict) or not row['checks']:
                raise ValueError('finite population row/checks required')
            if any(type(v) is not bool for v in row['checks'].values()):
                raise ValueError('explicit population diagnostic booleans')
            certificate=row['B_dagger']
            if (not isinstance(certificate,dict) or not np.isfinite([certificate[k] for k in
                    ('B_dagger','lambda_min','B_cert_KG','B_cert_KG_raw','zero_cut','Delta_k_KG')]).all()
                    or row['B_cert'] != certificate['B_dagger']):
                raise ValueError('complete metric-corrected Bdagger evidence')
            if type(row['restart_agreement']) is not bool or type(row['PZ2']['available']) is not bool:
                raise ValueError('explicit prediction availability/diagnostics')
            if row['N_star_pred'] is not None and not np.isfinite(row['N_star_pred']):
                raise ValueError('finite or unavailable point crossing')
            cert = row['N_star_interval']
            if cert is not None and (len(cert) != 2 or not np.isfinite(cert).all() or cert[0] > cert[1]):
                raise ValueError('finite ordered or unavailable certificate')
            if annotate(row,grid)['PZ2'] != row['PZ2']:
                raise ValueError('prediction availability/range labels inconsistent')
        if record['PZ1'] != inherited_pz1(fixed):
            raise ValueError('complete unchanged PZ1 record required')
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise PermissionError('STOP incomplete or inconsistent per-date prediction: ' + str(exc)) from exc
    return record


def prediction_record(Sref, date):
    """Array-only future wrapper; callers must qualify and preflight before access."""
    index = 12 + load()['access']['confirmation_analysis_dates'].index(date)
    out = {'date': date, 'date_index': index, 'label': 'CONFIRMATION PREDICTION; pre-fit',
           'Sref_sha256_f64le': sref_hash(Sref), **predict_unit(Sref, 1111, index)}
    out['seed_namespace'] = 'unchanged prediction namespace; confirmation date_index12+sorted60-index'
    return validate_record(out, date)


def validate_fit(record, roles, model, Sref, N, m, arm, geometry, date_index, replicate_index):
    """Reject mismatched reference, geometry, signal recipe, grids and arm settings."""
    from experiments.mlst2026.hcp_rnoise_20261019 import gates, signal
    try:
        if geometry not in GRIDS or type(N) is not int or N not in GRIDS[geometry]:
            raise ValueError('fixed geometry/N grid')
        if (arm, m) not in (('reference', None), ('lw', 256), ('lw', 512), ('kron', 256), ('kron', 512), ('isotropic', None)):
            raise ValueError('fixed six arms and calibration levels')
        if type(date_index) is not int or date_index != record['date_index']:
            raise ValueError('fixed seed index')
        if type(replicate_index) is not int or not 0 <= replicate_index < 50:
            raise ValueError('R50 indices0..49')
        if sref_hash(Sref) != record['Sref_sha256_f64le'] or not np.array_equal(gates.reference(roles['ref'][0]), Sref):
            raise ValueError('prediction/fit/reference-role Sref mismatch')
        geo = record['geometries'][geometry]
        if geometry != 'control' and not geo['available']:
            raise ValueError('unavailable adaptive geometry')
        if model.C != 12 or model.T != 6 or model.theta != geo['theta']:
            raise ValueError('fixed dimension/theta')
        expected = signal.build(Sref, geo['theta'])
        if not np.array_equal(model.templates(), expected.templates()) or not np.array_equal(model.variances, expected.variances):
            raise ValueError('fixed signal recipe/scale differs')
    except (KeyError, TypeError, ValueError, AttributeError, np.linalg.LinAlgError) as exc:
        raise PermissionError('STOP prediction-to-fit binding: ' + str(exc)) from exc
