"""Registered family-of-34 recomputation from a D ledger, using the registered rules.

Unit = recording date: per date, mean D over the 50 replicates; interval over the
included dates: two-sided Bonferroni Student-t, df = n-1, alpha = 0.05, family size
34 (15 reference cells + 4 paired PZ4 interactions + 15 isotropic cells).  Zero-SE
cells are undecided.

Ingestion is strict on purpose.  A row that was not computed, was not finite, or is
missing/duplicated must never enter a mean: an incomplete cell is reported as
UNAVAILABLE, never as "undecided", because undecided is a scientific decision and
unavailable is a computation failure (cf. hcp_rnoise_ad9_20261019/readings.py
`_valid_interval`, "Do not confuse a failed computation with a valid undecided
interval").  This is a descriptive recomputation of the registered rules; it does
not re-score the registered readings.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import t as student_t

from experiments.mlst2026.hcp_rnoise_ad9_20261019.prediction import GRIDS

FAMILY = 34
ALPHA = .05
RESAMPLES = 50
REFERENCE_CELLS = [f'{g}/{i}/reference' for g in GRIDS for i in range(5)]
ISOTROPIC_CELLS = [f'{g}/{i}/isotropic' for g in GRIDS for i in range(5)]
PAIRED_ARMS = (('lw', 256), ('lw', 512), ('kron', 256), ('kron', 512))


def interval(values, family=FAMILY, alpha=ALPHA):
    """Registered interval rule (identical formula to meg_z_20261013.runner.interval)."""
    v = np.asarray(values, float)
    n = len(v)
    mean = float(v.mean()) if n else float('nan')
    if n < 2:
        return dict(n=n, mean=mean, sd=None, ci=None, half_width=None, zero_SE=False,
                    decision='undecided', family_size=family, alpha=alpha)
    sd = float(v.std(ddof=1))
    half = float(student_t.ppf(1 - alpha / (2 * family), n - 1) * sd / np.sqrt(n))
    zero = bool(sd == 0)
    lo, hi = mean - half, mean + half
    dec = 'undecided' if zero else 'negative' if hi < 0 else 'positive' if lo > 0 else 'undecided'
    return dict(n=n, mean=mean, sd=sd, ci=[lo, hi], half_width=half, zero_SE=zero,
                decision=dec, family_size=family, alpha=alpha)


class IncompleteLedger(ValueError):
    """Raised when a cell's date x replicate ledger is incomplete, duplicated or non-finite."""


def _ingest(records):
    """(date,replicate,geometry,N,arm,m) -> finite D; raise on duplicate or non-finite."""
    D = {}
    for row in records:
        key = (row[0], row[1], row[2], row[3], row[4], row[5])
        if key in D:
            raise IncompleteLedger(f'duplicate identity {key}')
        if not np.isfinite(row[6]):
            raise IncompleteLedger(f'non-finite D at {key}')
        D[key] = float(row[6])
    return D


def ledger_from_fits(path):
    """Registered FITS.jsonl -> D ledger (strict)."""
    recs = []
    with open(path, 'rb') as fh:
        for line in fh:
            r = json.loads(line)
            recs.append((r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m'], r['D']))
    return _ingest(recs)


def ledger_from_xopt(path):
    """X-OPT_FITS.jsonl -> D ledger.  Only status=='ok' rows with finite D are admitted."""
    recs = []
    skipped = 0
    with open(path, 'rb') as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('status') != 'ok':
                skipped += 1
                continue
            recs.append((r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m'], r['D']))
    if skipped:
        raise IncompleteLedger(f'{skipped} failed/non-ok rows in the X-OPT ledger; '
                               'report the family as UNAVAILABLE, never as undecided')
    return _ingest(recs)


def _cell_values(D, dates, key):
    """The exact expected date x replicate ledger of one cell (strict completeness)."""
    g, N, arm, m = key
    out = {}
    for d in dates:
        vals = []
        for r in range(RESAMPLES):
            k = (d, r, g, N, arm, m)
            if k not in D:
                raise IncompleteLedger(f'missing {k}')
            vals.append(D[k])
        out[d] = float(np.mean(vals))
    return out


def family34(D, dates):
    """All 34 registered cells from a complete D ledger.  Raises IncompleteLedger."""
    if len(set(dates)) != len(dates) or not dates:
        raise IncompleteLedger('dates must be unique and non-empty')
    cells = []
    for cell in REFERENCE_CELLS + ISOTROPIC_CELLS:
        g, i = cell.split('/')[:2]
        arm = 'reference' if cell.endswith('reference') else 'isotropic'
        N = GRIDS[g][int(i)]
        means = _cell_values(D, dates, (g, N, arm, None))
        r = interval(list(means.values()))
        cells.append(dict(cell=cell, kind='cell', available=True, geometry=g, N=N, arm=arm, m=None,
                          date_means=means, n_dates=len(dates), **r))
    for arm, m in PAIRED_ARMS:
        means = {}
        for d in dates:
            per_cell = []
            for g in ('A', 'B'):
                for N in GRIDS[g]:
                    est = np.mean([D[(d, r, g, N, arm, m)] for r in range(RESAMPLES)])
                    ref = np.mean([D[(d, r, g, N, 'reference', None)] for r in range(RESAMPLES)])
                    per_cell.append(float(est - ref))
            means[d] = float(np.mean(per_cell))
        r = interval(list(means.values()))
        cells.append(dict(cell=f'{arm}_m{m}', kind='pz4_interaction', available=True, arm=arm, m=m,
                          date_means=means, n_dates=len(dates), **r))
    if len(cells) != FAMILY:
        raise IncompleteLedger(f'family size {len(cells)} != {FAMILY}')
    return cells


def decisions(cells):
    return {c['cell']: c['decision'] for c in cells}


def classify_changes(xopt_cells, reg_cells):
    """Every decision change, classified.  A change is any label difference, including
    a decided cell becoming undecided.  Sign reversals are counted separately because
    they are the scientifically strongest form of sensitivity."""
    reg = {c['cell']: c for c in reg_cells}
    changes = []
    n_sign_reversal = n_to_undecided = n_to_decided = 0
    for c in xopt_cells:
        before = reg[c['cell']]['decision']
        after = c['decision']
        if before == after:
            continue
        if {before, after} == {'negative', 'positive'}:
            n_sign_reversal += 1
            kind = 'sign_reversal'
        elif after == 'undecided':
            n_to_undecided += 1
            kind = 'decided_to_undecided'
        else:
            n_to_decided += 1
            kind = 'undecided_to_decided'
        changes.append(dict(cell=c['cell'], kind=kind, before=before, after=after,
                            mean_before=reg[c['cell']]['mean'], mean_after=c['mean'],
                            half_width_before=reg[c['cell']]['half_width'],
                            half_width_after=c['half_width']))
    return dict(changes=changes, n=len(changes), n_sign_reversal=n_sign_reversal,
                n_decided_to_undecided=n_to_undecided, n_undecided_to_decided=n_to_decided)


def cell_mean_shift_over_halfwidth(xopt_cells, reg_cells):
    """Cell-mean shift divided by the registered half-width (the reduction-level ratio)."""
    reg = {c['cell']: c for c in reg_cells}
    rows = []
    for c in xopt_cells:
        hw = reg[c['cell']]['half_width']
        shift = abs(c['mean'] - reg[c['cell']]['mean'])
        rows.append(dict(cell=c['cell'], abs_d_mean=shift, half_width=hw,
                         ratio=shift / hw if hw else float('inf')))
    rows.sort(key=lambda r: r['ratio'], reverse=True)
    return rows


def per_row_abs_dD_over_halfwidth(row_deltas, reg_cells):
    """Largest per-row |X-OPT D - stored D| divided by the cell's registered half-width."""
    reg = {c['cell']: c for c in reg_cells}
    worst = None
    for (d, r, g, N, arm, m), v in row_deltas.items():
        i = GRIDS[g].index(N)
        cell = f'{g}/{i}/' + (arm if m is None else f'{arm}_m{m}')
        if cell not in reg:
            continue
        hw = reg[cell]['half_width']
        ratio = abs(v) / hw if hw else float('inf')
        if worst is None or ratio > worst['ratio']:
            worst = dict(cell=cell, identity=[d, r, g, N, arm, m], abs_dD=abs(v), half_width=hw,
                         ratio=ratio)
    return worst


def min_margin(cells):
    """Smallest |mean| - half_width over decided cells: the closest flip boundary."""
    m = [(abs(c['mean']) - c['half_width'], c['cell'], c['decision']) for c in cells
         if c['decision'] != 'undecided' and np.isfinite(c['half_width'])]
    m.sort()
    return [dict(margin=float(a), cell=b, decision=c) for a, b, c in m]
