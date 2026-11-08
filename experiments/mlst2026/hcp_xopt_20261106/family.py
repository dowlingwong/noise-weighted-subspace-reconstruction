"""Registered family-of-34 recomputation from a D ledger, using the registered rules.

Unit = recording date: per date, mean D over the 50 replicates; interval over the
included dates: two-sided Bonferroni Student-t, df = n-1, alpha = 0.05, family size
34 (15 reference cells + 4 paired PZ4 interactions + 15 isotropic cells).  Zero-SE
cells are undecided.

Ingestion is strict on purpose.  A row that was not computed, was not finite, or is
missing/duplicated must never enter a mean: an incomplete cell is reported as
INCOMPLETE/UNAVAILABLE, never as "undecided", because undecided is a scientific decision
and unavailable is a computation failure (cf. hcp_rnoise_ad9_20261019/readings.py
`_valid_interval`, "Do not confuse a failed computation with a valid undecided
interval").  This is a descriptive recomputation of the registered rules; it does
not re-score the registered readings.

Scope-aware verdict: only **refit cells** (cells whose rows are fully in scope) can enter
the verdict.  Cells whose rows are only sampled are reported descriptively and never
enter the verdict.
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
PZ4_CELLS = [f'{arm}_m{m}' for arm, m in PAIRED_ARMS]

# Refit cells by scope (resume-4 Step D.c).  A cell is a refit cell only if all of its
# rows are in scope; sampled cells never enter the verdict.
REFIT_BY_SCOPE = {
    'full': tuple(REFERENCE_CELLS + ISOTROPIC_CELLS + PZ4_CELLS),
    'primary': tuple(REFERENCE_CELLS + ISOTROPIC_CELLS),
    'core': tuple(REFERENCE_CELLS),
}


def interval(values, family=FAMILY, alpha=ALPHA):
    """Registered interval rule (identical formula to meg_z_20261013.runner.interval)."""
    v = np.asarray(values, float)
    n = len(v)
    mean = float(v.mean()) if n else float('nan')
    if n < 2:
        return dict(n=n, mean=mean, sd=None, ci=None, half_width=None, zero_SE=False,
                    decision='undecided', family_size=family, alpha=alpha)
    if not np.all(np.isfinite(v)):
        raise IncompleteLedger('non-finite value reached the interval rule')
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
                               'report the family as INCOMPLETE, never as undecided')
    return _ingest(recs)


def _require(D, key):
    if key not in D:
        raise IncompleteLedger(f'missing {key}')
    v = D[key]
    if not np.isfinite(v):
        raise IncompleteLedger(f'non-finite value at {key}')
    return v


def _cell_values(D, dates, key):
    """The exact expected date x replicate ledger of one cell (strict completeness)."""
    g, N, arm, m = key
    out = {}
    for d in dates:
        vals = [_require(D, (d, r, g, N, arm, m)) for r in range(RESAMPLES)]
        out[d] = float(np.mean(vals))
    return out


def _cell_of_row(row_key):
    d, r, g, N, arm, m = row_key
    return f'{g}/{GRIDS[g].index(N)}/' + (arm if m is None else f'{arm}_m{m}')


def compute_cells(D, dates, cell_names):
    """Compute exactly the named cells from a complete-enough ledger.  Raises IncompleteLedger."""
    if len(set(dates)) != len(dates) or not dates:
        raise IncompleteLedger('dates must be unique and non-empty')
    cells = []
    for cell in cell_names:
        if cell in PZ4_CELLS:
            arm, m = cell.rsplit('_m', 1)
            m = int(m)
            means = {}
            for d in dates:
                per_cell = []
                for g in ('A', 'B'):
                    for N in GRIDS[g]:
                        est = [_require(D, (d, r, g, N, arm, m)) for r in range(RESAMPLES)]
                        ref = [_require(D, (d, r, g, N, 'reference', None)) for r in range(RESAMPLES)]
                        per_cell.append(float(np.mean(est) - np.mean(ref)))
                means[d] = float(np.mean(per_cell))
            r = interval(list(means.values()))
            cells.append(dict(cell=cell, kind='pz4_interaction', available=True, arm=arm, m=m,
                              date_means=means, n_dates=len(dates), **r))
            continue
        g, i, arm = cell.split('/')
        N = GRIDS[g][int(i)]
        means = _cell_values(D, dates, (g, N, arm, None))
        r = interval(list(means.values()))
        cells.append(dict(cell=cell, kind='cell', available=True, geometry=g, N=N, arm=arm, m=None,
                          date_means=means, n_dates=len(dates), **r))
    return cells


def family34(D, dates):
    """All 34 registered cells from a complete D ledger.  Raises IncompleteLedger."""
    cells = compute_cells(D, dates, REFIT_BY_SCOPE['full'])
    if len(cells) != FAMILY:
        raise IncompleteLedger(f'family size {len(cells)} != {FAMILY}')
    return cells


def refit_cells(scope):
    try:
        return REFIT_BY_SCOPE[scope]
    except KeyError:
        raise ValueError(f'unknown scope {scope!r}; expected one of {sorted(REFIT_BY_SCOPE)}')


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
        if c['cell'] not in reg:
            raise IncompleteLedger(f'no registered control for refit cell {c["cell"]}')
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
    """Largest per-row |X-OPT D - replayed frozen D| divided by the cell's registered
    half-width.  The delta is against the replayed frozen D in the same host process, not
    against the stored D (the label is pinned to the computation)."""
    reg = {c['cell']: c for c in reg_cells}
    worst = None
    for (d, r, g, N, arm, m), v in row_deltas.items():
        i = GRIDS[g].index(N)
        cell = f'{g}/{i}/' + (arm if m is None else f'{arm}_m{m}')
        if cell not in reg or not np.isfinite(v):
            continue
        hw = reg[cell]['half_width']
        ratio = abs(v) / hw if hw else float('inf')
        if worst is None or ratio > worst['ratio']:
            worst = dict(cell=cell, identity=[d, r, g, N, arm, m], abs_dD=abs(v), half_width=hw,
                         ratio=ratio)
    return worst


def sampled_summaries(rows, refit, cells_reg):
    """Descriptive-only summaries for cells that are NOT refit cells.

    For each sampled cell: the per-row Delta D and Delta objective distributions and a
    sample-based interval over the (sampled) date means.  These numbers are labelled
    "sample-based, not a family-34 recomputation" and never enter the verdict.
    """
    reg = {c['cell']: c for c in cells_reg}
    grouped = {}
    for key, rec in rows.items():
        cell = _cell_of_row(key)
        if cell in refit:
            continue
        grouped.setdefault(cell, []).append(rec)
    out = []
    for cell in sorted(grouped):
        recs = grouped[cell]
        dD = [float(r['delta']['D']) for r in recs]
        dobj = [float(r['delta']['objective']) for r in recs]
        per_date = {}
        for r in recs:
            per_date.setdefault(r['date'], []).append(float(r['D']))
        date_means = {d: float(np.mean(v)) for d, v in sorted(per_date.items())}
        try:
            iv = interval(list(date_means.values()))
            interval_ok = True
        except IncompleteLedger:
            iv = None
            interval_ok = False
        out.append(dict(
            cell=cell,
            label='sample-based, not a family-34 recomputation',
            n_rows=len(recs), n_dates=len(date_means),
            delta_D=dict(min=float(min(dD)), median=float(np.median(dD)), max=float(max(dD)),
                         mean=float(np.mean(dD)), max_abs=float(max(abs(v) for v in dD))),
            delta_objective=dict(min=float(min(dobj)), median=float(np.median(dobj)),
                                 max=float(max(dobj)), mean=float(np.mean(dobj))),
            sample_based_interval=iv if interval_ok else None,
            registered_half_width=reg[cell]['half_width'] if cell in reg else None,
            enters_verdict=False))
    return out


def min_margin(cells):
    """Smallest |mean| - half_width over decided cells: the closest flip boundary."""
    m = [(abs(c['mean']) - c['half_width'], c['cell'], c['decision']) for c in cells
         if c['decision'] != 'undecided' and np.isfinite(c['half_width'])]
    m.sort()
    return [dict(margin=float(a), cell=b, decision=c) for a, b, c in m]
