"""X-OPT reduction: scope-aware verdict, decision changes and readouts.

Reads W2/out/resume4/XOPT_FITS.jsonl (or a given ledger), the sealed FITS.jsonl (registered
control) and the published prediction records; writes XOPT_EVIDENCE.json.

The invariant enforced here: a non-finite, missing, duplicated or failed value can only
produce a counted failure and the label INCOMPLETE -- never a decision, a verdict label or an
interval.  Only **refit cells** (cells whose rows are fully in scope) enter the verdict;
sampled cells are reported descriptively with an explicit "sample-based, not a family-34
recomputation" label and never enter the verdict.

Descriptive only: it never re-scores a registered reading.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import family
from experiments.mlst2026.hcp_rnoise_ad9_20261019.prediction import GRIDS

from .config import ROOT
W = ROOT
IN = W + '/inputs/hcp_confirmation_20261020'

PLANNED_ROWS = {'full': 135000, 'primary': 54000, 'core': 31500}
SCOPE_ALIAS = {'S_full': 'full', 'S_primary': 'primary', 'S_core': 'core',
               'full': 'full', 'primary': 'primary', 'core': 'core'}


def normalize_scope(scope):
    if scope is None:
        return 'full'
    try:
        return SCOPE_ALIAS[scope]
    except KeyError:
        raise ValueError(f'unknown scope {scope!r}')


# --------------------------------------------------------------------------- inputs
def _finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def load_deltas(path, expected_rows=None):
    """Strict ingestion: only status=='ok' rows with finite D and finite deltas.

    Rejects a duplicate identity, a non-finite D/Delta, any non-ok row, and a row census
    that is not the planned count, by raising `family.IncompleteLedger`.
    """
    rows = {}
    skipped = 0
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('status') != 'ok':
                skipped += 1
                continue
            key = (r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m'])
            if key in rows:
                raise family.IncompleteLedger(f'duplicate identity {key}')
            if not _finite(r.get('D')):
                raise family.IncompleteLedger(f'non-finite D at {key}')
            delta = r.get('delta')
            if not isinstance(delta, dict) or 'D' not in delta or 'objective' not in delta:
                raise family.IncompleteLedger(f'missing delta at {key}')
            for name in ('D', 'objective'):
                if not _finite(delta[name]):
                    raise family.IncompleteLedger(f'non-finite or missing delta.{name} at {key}')
            original = r.get('original')
            if not isinstance(original, dict) or not _finite(original.get('D')):
                raise family.IncompleteLedger(f'missing original at {key}')
            rows[key] = r
    if skipped:
        raise family.IncompleteLedger(f'{skipped} non-ok row(s) in the X-OPT ledger')
    if expected_rows is not None and len(rows) != expected_rows:
        raise family.IncompleteLedger(f'row census {len(rows)} != planned {expected_rows}')
    return rows


def n_star_targets(dates):
    """Committed per-date N_star_pred by geometry.  The value lives under ['row']."""
    out = {}
    for d in dates:
        with open(f'{IN}/attic_launcher_local/predictions/{d}.json', 'rb') as fh:
            rec = json.loads(fh.read())
        out[d] = {}
        for g in ('A', 'B'):
            geo = rec['geometries'][g]
            if geo.get('available') and geo.get('row'):
                out[d][g] = float(geo['row']['N_star_pred'])
    return out


def nearest_grid(geometry, target):
    """Nearest fixed grid N, clipped at the grid ends, ties broken to the lower N."""
    grid = GRIDS[geometry]
    if target <= grid[0]:
        return grid[0]
    if target >= grid[-1]:
        return grid[-1]
    dist = [abs(N - target) for N in grid]
    m = min(dist)
    return min(N for N, dd in zip(grid, dist) if dd == m)


# --------------------------------------------------------------------------- readouts
def sign_pattern(cells_x, dates, targets, multiples=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Reference-arm decisions at N*/4 .. 4N* for A and B, and per-date counts at N*, 2N*."""
    by_cell = {c['cell']: c for c in cells_x}
    out = {'targets': {}, 'per_date_counts': {}}
    for g in ('A', 'B'):
        vals = [t[g] for t in targets.values() if g in t]
        if not vals or not any(c.startswith(f'{g}/') for c in by_cell):
            out['targets'][g] = {'available': False}
            continue
        med = float(np.median(vals))
        rows = []
        seen = set()
        for mult in multiples:
            target = med * mult
            N = nearest_grid(g, target)
            cell = f'{g}/{GRIDS[g].index(N)}/reference'
            if cell not in by_cell:
                rows.append(dict(multiple=mult, target_N=target, grid_N=N, cell=cell,
                                 available=False))
                continue
            c = by_cell[cell]
            rows.append(dict(multiple=mult, target_N=target, grid_N=N, cell=cell,
                             duplicate_of_grid_N=None if N not in seen else True,
                             decision=c['decision'], mean=c['mean'], half_width=c['half_width']))
            seen.add(N)
        out['targets'][g] = dict(available=True, median_N_star=med, rows=rows)
        for mult in (1.0, 2.0):
            cell_counts = {'negative': 0, 'positive': 0, 'undecided': 0}
            sign_counts = {'negative': 0, 'positive': 0, 'zero': 0}
            for d in dates:
                if g not in targets[d]:
                    continue
                N = nearest_grid(g, targets[d][g] * mult)
                c = by_cell[f'{g}/{GRIDS[g].index(N)}/reference']
                cell_counts[c['decision']] += 1
                v = c['date_means'][d]
                sign_counts['negative' if v < 0 else 'positive' if v > 0 else 'zero'] += 1
            out['per_date_counts'][f'{g}_x{mult:g}'] = dict(cell_decisions=cell_counts,
                                                           per_date_mean_D_signs=sign_counts)
    return out


def per_cell_arm_delta(rows):
    groups = {}
    for (d, r, g, N, arm, m), rec in rows.items():
        groups.setdefault((g, N, arm, m), []).append(rec['delta']['D'])
    out = []
    for (g, N, arm, m), vals in sorted(groups.items(),
                                       key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][3]))):
        out.append(dict(geometry=g, N=N, arm=arm, m=m, n=len(vals),
                        mean_delta_D=float(np.mean(vals)),
                        max_abs_delta_D=float(max(abs(v) for v in vals))))
    return out


def pz4_pz5(cells_x):
    """Registered-style descriptive verdicts; available only when the PZ4 cells are refit."""
    by = {c['cell']: c for c in cells_x}
    if not all(k in by for k in list(family.PZ4_CELLS) + list(family.ISOTROPIC_CELLS)):
        return dict(available=False,
                    reason='PZ4/PZ5 cells are not refit cells in this scope; '
                           'sampled summaries only')
    pz4 = {}
    for est in ('lw', 'kron'):
        low, high = by[f'{est}_m256'], by[f'{est}_m512']
        pz4[est] = dict(low_cell=low['cell'], high_cell=high['cell'],
                        low_decision=low['decision'], high_decision=high['decision'],
                        low_mean=low['mean'], high_mean=high['mean'],
                        low_negative=low['decision'] == 'negative',
                        high_closer_to_zero=abs(high['mean']) < abs(low['mean']))
    pz4_met = all(pz4[e]['low_negative'] and pz4[e]['high_closer_to_zero'] for e in pz4)
    iso = [by[c] for c in family.ISOTROPIC_CELLS]
    pz5 = dict(decisions={c['cell']: c['decision'] for c in iso},
               decided_negative=[c['cell'] for c in iso if c['decision'] == 'negative'],
               met=not any(c['decision'] == 'negative' for c in iso))
    return dict(available=True,
                pz4=dict(estimators=pz4, met=pz4_met,
                         rule='m256 negative and |m512 mean| < |m256 mean|'),
                pz5=dict(**pz5, rule='no decided negative isotropic contrast'))


def reproducibility(rows):
    objs = [r['frozen']['replay_obj_rel'] for r in rows.values()]
    ds = [r['frozen']['replay_D_over_spread'] for r in rows.values()]
    ds = [d for d in ds if d is not None]
    return dict(n=len(rows), max_replay_obj_rel=float(max(objs)) if objs else None,
                n_replay_obj_within_1e_9=int(sum(1 for o in objs if o <= 1e-9)),
                max_replay_D_over_stored_spread=float(max(ds)) if ds else None,
                n_replay_D_within_stored_spread=int(sum(1 for d in ds if d <= 1.0)),
                max_abs_replay_D_minus_stored=float(max(abs(r['frozen']['D'] - r['original']['D'])
                                                        for r in rows.values())) if rows else None)


def route_statistics(rows):
    from collections import Counter
    routes = Counter(r['strengthened']['selected_route'] for r in rows.values())
    statuses = Counter(r['flags_after']['selected_status'] for r in rows.values())
    n = len(rows)
    cleared = int(sum(1 for r in rows.values()
                      if r['flags_after']['at_frozen_tol']['optimization_limited_t25_rule'] is False))
    cleared_s1 = int(sum(1 for r in rows.values()
                         if r['flags_after']['at_s1_tol']['optimization_limited_t25_rule'] is False))
    newly = int(sum(1 for r in rows.values()
                    if r['flags_after']['at_frozen_tol']['optimization_limited_t25_rule'] is True
                    and r['original']['optimization_limited_t25_rule'] is False))
    return dict(routes=dict(routes), winning_status=dict(statuses), n=n,
                flag_cleared_at_frozen_tol=cleared, flag_cleared_at_s1_tol=cleared_s1,
                newly_flagged_at_frozen_tol=newly,
                grad_below_frozen_tol=int(sum(1 for r in rows.values()
                                              if r['flags_after']['selected_gradient_norm'] <= 1e-8)),
                note='the S1 tolerance 1e-11 is below the Armijo floating-point floor (~2e-8); '
                     'cleared_at_s1_tol is reported only to make that explicit')


def _skipped_diagnostics(rows):
    return {}


# --------------------------------------------------------------------------- main
def main(xopt_path=None, out_path=None, scope=None):
    scope = normalize_scope(scope)
    out_path = out_path or os.path.join(W, 'out', 'resume4', 'XOPT_EVIDENCE.json')
    xopt_path = xopt_path or os.path.join(W, 'out', 'resume4', 'XOPT_FITS.jsonl')
    with open(IN + '/SUMMARY.json') as fh:
        summary = json.load(fh)
    dates = summary['cohort']['included']
    evidence = dict(study='X-OPT', status_label='post hoc sensitivity analysis; '
                               'not a registered reading', scope=scope)

    try:
        rows = load_deltas(xopt_path, expected_rows=PLANNED_ROWS[scope])
    except family.IncompleteLedger as exc:
        evidence['outcome'] = dict(family='INCOMPLETE', reason=str(exc))
        evidence['scope_detail'] = dict(expected_rows=PLANNED_ROWS[scope], loaded_rows=None)
        _write(evidence, out_path)
        return evidence

    evidence['scope_detail'] = dict(expected_rows=PLANNED_ROWS[scope], loaded_rows=len(rows),
                                    dates=len(dates), refit_cells=list(family.refit_cells(scope)),
                                    n_refit_cells=len(family.refit_cells(scope)))
    try:
        evidence['reproducibility'] = reproducibility(rows)
        evidence['route_statistics'] = route_statistics(rows)
        evidence['per_cell_arm_mean_delta_D'] = per_cell_arm_delta(rows)
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        evidence['outcome'] = dict(family='INCOMPLETE',
                                   reason=f'malformed row in the diagnostic readouts: {exc!r}')
        _write(evidence, out_path)
        return evidence

    D_x = {k: r['D'] for k, r in rows.items()}
    D_reg = family.ledger_from_fits(IN + '/FITS.jsonl')
    refit = family.refit_cells(scope)
    try:
        cells_x = family.compute_cells(D_x, dates, refit)
        cells_reg = family.compute_cells(D_reg, dates, refit)
        changes = family.classify_changes(cells_x, cells_reg)
    except family.IncompleteLedger as exc:
        evidence['outcome'] = dict(family='INCOMPLETE', reason=str(exc))
        _write(evidence, out_path)
        return evidence

    delta_D = [float(r['delta']['D']) for r in rows.values()]
    delta_obj = [float(r['delta']['objective']) for r in rows.values()]
    evidence.update(
        outcome=dict(family='ROBUST' if changes['n'] == 0 else 'SENSITIVE', **changes),
        refit_cells_note='only these cells enter the verdict; sampled cells never do',
        family34_refit=dict(xopt=cells_x, registered_control=cells_reg),
        pz4_pz5=pz4_pz5(cells_x),
        sign_pattern=sign_pattern(cells_x, dates, n_star_targets(dates)),
        min_margin=family.min_margin(cells_reg),
        sampled_cells=family.sampled_summaries(rows, set(refit), cells_reg),
        delta=dict(max_abs_delta_D=float(max(abs(v) for v in delta_D)),
                   n_positive_delta=int(sum(1 for v in delta_D if v > 0)),
                   n_negative_delta=int(sum(1 for v in delta_D if v < 0)),
                   max_abs_delta_objective=float(max(abs(v) for v in delta_obj))),
        ratio_definitions=dict(
            per_row_abs_deltaD_over_cell_half_width=family.per_row_abs_dD_over_halfwidth(
                {k: r['delta']['D'] for k, r in rows.items()}, cells_reg),
            cell_mean_shift_over_registered_half_width=family.cell_mean_shift_over_halfwidth(
                cells_x, cells_reg)[0] if cells_x else None,
            per_row_label='per-row |X-OPT D - replayed frozen D| / registered half-width '
                          '(delta is against the replayed frozen D, computed in the same '
                          'process; NOT against the stored D)'),
        decision_rule_note='a change is any label difference; sign reversals and '
                           'decided<->undecided transitions are counted separately',
    )
    _write(evidence, out_path)
    return evidence


def _write(evidence, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write('\n')
    print(json.dumps({'outcome': evidence.get('outcome'),
                      'reproducibility': evidence.get('reproducibility'),
                      'min_margin': (evidence.get('min_margin') or [None])[0]}, indent=2))
    print('wrote', out_path)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger', nargs='?', default=None)
    ap.add_argument('out', nargs='?', default=None)
    ap.add_argument('--scope', default='full')
    args = ap.parse_args()
    main(args.ledger, args.out, args.scope)
