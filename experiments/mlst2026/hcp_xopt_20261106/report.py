"""X-OPT reduction: registered family-34 recomputation, decision changes and readouts.

Reads W/out/XOPT_FITS.jsonl, the sealed FITS.jsonl (registered control) and the published
prediction records; writes XOPT_EVIDENCE.json (compact, paper-facing, sorted keys, two-space
indent, final newline).  Descriptive only: it never re-scores a registered reading, and an
incomplete or failed ledger is reported as UNAVAILABLE, never as "undecided".
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


# --------------------------------------------------------------------------- inputs
def load_deltas(path):
    """Per-row join with the stored values, flags and route.  Only status=='ok' rows."""
    rows = {}
    skipped = 0
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('status') != 'ok':
                skipped += 1
                continue
            key = (r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m'])
            rows[key] = r
    return rows, skipped


def n_star_targets(dates):
    """Committed per-date N_star_pred by geometry.  The value lives under ['row']."""
    out = {}
    for d in dates:
        rec = json.loads(open(f'{IN}/attic_launcher_local/predictions/{d}.json', 'rb').read())
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
    """Reference-arm decisions at N*/4, N*/2, N*, 2N*, 4N* for A and B.

    Targets use the date-median committed N* and are read from the nearest fixed grid cell
    (clipped at the grid ends; a target that clips onto a cell already used is reported as a
    duplicate rather than counted twice).  Per-date counts at N* and 2N* use each date's own
    committed N* and report both the mapped cell decision and the sign of that date's mean D.
    """
    by_cell = {c['cell']: c for c in cells_x}
    out = {'targets': {}, 'per_date_counts': {}}
    for g in ('A', 'B'):
        vals = [t[g] for t in targets.values() if g in t]
        if not vals:
            out['targets'][g] = {'available': False}
            continue
        med = float(np.median(vals))
        rows = []
        seen = set()
        for mult in multiples:
            target = med * mult
            N = nearest_grid(g, target)
            cell = f'{g}/{GRIDS[g].index(N)}/reference'
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
    for (g, N, arm, m), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][3]))):
        out.append(dict(geometry=g, N=N, arm=arm, m=m, n=len(vals), mean_delta_D=float(np.mean(vals)),
                        max_abs_delta_D=float(max(abs(v) for v in vals))))
    return out


def pz4_pz5(cells_x):
    """Registered-style descriptive verdicts from the recomputed intervals."""
    by = {c['cell']: c for c in cells_x}
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
    return dict(pz4=dict(estimators=pz4, met=pz4_met, rule='m256 negative and |m512 mean| < |m256 mean|'),
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


# --------------------------------------------------------------------------- main
def main(xopt_path=None, out_path=None):
    xopt_path = xopt_path or os.path.join(W, 'out', 'XOPT_FITS.jsonl')
    out_path = out_path or os.path.join(W, 'out', 'XOPT_EVIDENCE.json')
    summary = json.load(open(IN + '/SUMMARY.json'))
    dates = summary['cohort']['included']
    rows, skipped = load_deltas(xopt_path)
    evidence = dict(study='X-OPT',
                    status_label='post hoc sensitivity analysis; not a registered reading',
                    scope=dict(rows=len(rows), failed_rows_skipped=skipped, dates=len(dates),
                               geometry_N_cells=15, arm_m_settings=6, replicates_per_cell=50),
                    reproducibility=reproducibility(rows),
                    route_statistics=route_statistics(rows),
                    per_cell_arm_mean_delta_D=per_cell_arm_delta(rows),
                    delta=dict(max_abs_delta_D=float(max(abs(r['delta']['D']) for r in rows.values())),
                               max_abs_relative_delta_D=float(max(abs(r['delta']['D']) / abs(r['original']['D'])
                                                                  for r in rows.values() if r['original']['D'])),
                               n_positive_delta=int(sum(1 for r in rows.values() if r['delta']['D'] > 0)),
                               n_negative_delta=int(sum(1 for r in rows.values() if r['delta']['D'] < 0))))
    D_x = {k: r['D'] for k, r in rows.items()}
    D_reg = family.ledger_from_fits(IN + '/FITS.jsonl')
    try:
        cells_x = family.family34(D_x, dates)
        cells_reg = family.family34(D_reg, dates)
    except family.IncompleteLedger as exc:
        evidence['outcome'] = dict(family='UNAVAILABLE', reason=str(exc))
        _write(evidence, out_path)
        return evidence
    changes = family.classify_changes(cells_x, cells_reg)
    evidence.update(
        outcome=dict(family='ROBUST' if changes['n'] == 0 else 'SENSITIVE', **changes),
        family34=dict(xopt=cells_x, registered_control=cells_reg),
        pz4_pz5=pz4_pz5(cells_x),
        sign_pattern=sign_pattern(cells_x, dates, n_star_targets(dates)),
        min_margin=family.min_margin(cells_reg),
        ratio_definitions=dict(
            per_row_abs_dD_over_cell_half_width=family.per_row_abs_dD_over_halfwidth(
                {k: r['delta']['D'] for k, r in rows.items()}, cells_reg),
            cell_mean_shift_over_registered_half_width=family.cell_mean_shift_over_halfwidth(cells_x, cells_reg)[0]
            if cells_x else None),
        decision_rule_note='a change is any label difference; sign reversals and '
                           'decided<->undecided transitions are counted separately',
    )
    _write(evidence, out_path)
    return evidence


def _write(evidence, out_path):
    with open(out_path, 'w') as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps({'outcome': evidence.get('outcome'),
                      'reproducibility': evidence.get('reproducibility'),
                      'route_statistics': evidence.get('route_statistics'),
                      'min_margin': (evidence.get('min_margin') or [None])[0]}, indent=2))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
