"""TR03: the registered X3 representation endpoints S1 and S2 (descriptive only).

Implements MS ``plan/onthefly/X3_ADDENDUM_REPRESENTATION_ENDPOINTS.md`` (registered by MS commit
8bad592, 6 Oct 2026).  The endpoints sit OUTSIDE the frozen X3 package (code identity
83360c2a...), outside its 54-contrast Bonferroni family and outside the P1/P2/SURE verdicts.
This module imports the frozen X3 package read-only and never writes into it.

Definitions (addendum section 2), for one selected model f and one seed, with the regenerated
evaluation inputs x_e (row per record; 2048 in the confirmation configuration):

* Y = (F - 1 fbar^T) W^T with W = Sigma^{-1/2} (the symmetric Li of the frozen code), so Y = F_c Li;
  C_f = Y^T Y / (n - 1); eigenvalues nu_1 >= ... >= nu_D; U_f = top-k eigenvectors; P_f = U_f U_f^T.
* P_Sigma: top-k eigenvectors of W C_s W^T, which for W = Li is K(theta) (= leading k columns of V(theta)).
* P_iso: top-k eigenvectors V_iso of the physical Cov(x) = C_s + Sigma, mapped by W and orthonormalised (QR).
* S1: d_ref = ||P_f - P_ref||_F^2 / (2k) (mean sin^2 of the k principal angles) and the largest principal
  angle in degrees, for ref in {Sigma, iso}.
* S2: leakage = 1 - (nu_1 + ... + nu_k) / tr C_f.

Reporting (addendum section 3): per (class, loss, N) the median and IQR over seeds; per (class, N) the
seed-paired difference d_Sigma(Sigma^-1 loss) - d_Sigma(isotropic loss).  No intervals, tests or decision
rules.  The expectation table (E1-E3) is a descriptive status of registered expectations, not a test.

Interpretive overlay (MS plan/onthefly/X3_ADDENDUM_OVERLAY_v1_2026-10-06.md; the registered addendum bytes are
unchanged) and implementation choices:
* U_f is undefined only where the registered top-k eigenspace is not unique, up to floating-point resolution
  tau = TIE_EPS_FACTOR * eps * D * nu_1 (a numerical-resolution tolerance, not a substantive threshold):
  'zero_output_variance' (tr C_f <= 0: every endpoint undefined); 'rank_deficient' (fewer than k eigenvalues
  above tau, so the zero eigenspace makes the top-k choice arbitrary; leakage stays defined); 'unresolved_tie'
  (nu_k - nu_{k+1} <= tau; leakage stays defined).  A small but resolved gap is NOT excluded: d and the angles
  are reported, with the relative gap (nu_k - nu_{k+1})/nu_1 and the numerical rank as diagnostics.
* Undefined values are never imputed.  A summary over fewer than all registered seeds is labelled
  'INCOMPLETE available-case' with n_defined and the undefined reasons; it is never the registered
  24-seed summary.  E1/E2 are evaluated only when both compared groups are complete; otherwise their status
  is 'not_evaluable_incomplete'.
* The addendum section-5 qualitative sentence ("approximately the covariance-optimal subspace") is withdrawn
  by the overlay: no sentence gate is computed; the numbers and the supported relative comparisons are reported.
* IQR uses numpy's default linear percentile interpolation (Q3 - Q1; Q1 and Q3 are also reported).
* "the two largest principal angles" (section 3) is read as the largest principal angle to each of the
  two references (section 2); every per-seed row also carries all k angles to both references.

Refusal (nothing but REFUSED.json is written) if any of: the export's SHA256SUMS, EXPORT.json or tables
are inconsistent; the export record, the freeze sources or the frozen source on disk does not carry the
registered X3 code identity; the run ledgers or the cell inventory are incomplete; a cell's COMPLETE.json
does not match its files; a regenerated training or evaluation data digest differs from the record; a
stored closed-form tied-linear map differs from its recomputation; a model record or its weights are
missing; the registered addendum bytes changed; a loaded in-repository module is outside the frozen
closure; or (confirmation mode) the X3 primary verdicts were not recorded first or differ from the
reduction.  All identity and data checks finish before the endpoint pass starts.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard as x3guard
from experiments.mlst2026.x3_nonlinear_metric_20261005 import science as x3s
from experiments.mlst2026.x3_nonlinear_metric_20261005 import workflow as x3w
from experiments.mlst2026.x3_nonlinear_metric_20261005.protocol import BASE, FIXTURE, enumeration

PACKAGE = Path(__file__).resolve().parent
EXP_ROOT = PACKAGE.parents[2]
VERSION = 'x3-representation-addendum/20261006-v2'
ADDENDUM_MS = 'plan/onthefly/X3_ADDENDUM_REPRESENTATION_ENDPOINTS.md'
ADDENDUM_REGISTRATION_COMMIT = '8bad592'
#: SHA-256 of the addendum bytes as committed in MS 8bad592 (checked against the MS tree at run time).
ADDENDUM_SHA256 = '37bad50385a0c672894d99051fed70c5f0c5df242bc6e01434bd7e3bc1cf5491'
REGISTERED_X3_IDENTITY = '83360c2a09cd4e06989ceeeae884429542dc682577e8bddc37a1db8f0e6fcfcc'
REGISTERED_K = 6
REGISTERED_EVAL_RECORDS = 2048
LINEAR_CHECK_TOL = 1e-8          # addendum section 2: |l| < 1e-8 is reported as a check, not an endpoint
TIE_EPS_FACTOR = 64.0            # numerical resolution: tau = 64 * eps * D * nu_1 (overlay v1, section 2)
REFERENCE_RTOL = 1e-8            # population references must be well separated (they are: gaps 4 and >0.1)
OVERLAY_MS = 'plan/onthefly/X3_ADDENDUM_OVERLAY_v1_2026-10-06.md'
WHITENING_TOL = 1e-10
CLOSED_FORM_RTOL, CLOSED_FORM_ATOL = 1e-9, 1e-11   # as in the frozen workflow.verify_cell
METRICS = ('d_sigma', 'd_iso', 'max_angle_sigma_deg', 'max_angle_iso_deg', 'leakage')
LINEAR_CLASSES = ('tied_linear', 'deep_linear')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS')


class Refusal(RuntimeError):
    """Raised before any endpoint is written; the caller writes REFUSED.json only."""


# --------------------------------------------------------------------------------------
# linear algebra (pure functions; unit-tested on small references)
# --------------------------------------------------------------------------------------
def eigh_desc(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def orthonormalise(M):
    q, r = np.linalg.qr(M)
    d = np.abs(np.diag(r))
    if d.min() <= 1e-12 * max(d.max(), 1e-300):
        raise Refusal('reference basis is rank deficient')
    return q


def projector(U):
    return U @ U.T


def subspace_distance(P1, P2, k):
    """||P1 - P2||_F^2 / (2k): the mean sin^2 of the k principal angles (in [0, 1])."""
    return float(np.linalg.norm(P1 - P2, 'fro') ** 2 / (2 * k))


def principal_angles_deg(U1, U2):
    """Principal angles (degrees, descending) between the spans of orthonormal U1 and U2."""
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return np.sort(np.degrees(np.arccos(np.clip(s, 0.0, 1.0))))[::-1]


def check_whitening(W, sigma):
    err = float(np.max(np.abs(W @ sigma @ W.T - np.eye(len(W)))))
    if err > WHITENING_TOL:
        raise Refusal(f'W Sigma W^T differs from I by {err:.3e}')
    return err


def references(c, W=None):
    """Population reference subspaces in the coordinates of whitening W (default Li, symmetric)."""
    g = x3s.geometry(c)
    sigma = x3s.t25.covariance(c)
    L, Li = x3s.powers(sigma)
    default = W is None
    W = Li if default else np.asarray(W, dtype=float)
    k = c['k']
    werr = check_whitening(W, sigma)
    K = g.K(c['theta'])
    Cs = L @ K @ L
    nu_s, V_s = eigh_desc(W @ Cs @ W.T)
    gap_s = float(nu_s[k - 1] - nu_s[k])
    if gap_s <= REFERENCE_RTOL * nu_s[0]:
        raise Refusal('population P_Sigma not identifiable')
    U_sigma = V_s[:, :k]
    nu_x, V_x = eigh_desc(Cs + sigma)
    gap_x = float(nu_x[k - 1] - nu_x[k])
    if gap_x <= REFERENCE_RTOL * nu_x[0]:
        raise Refusal('population isotropic-PCA subspace not identifiable')
    U_iso = orthonormalise(W @ V_x[:, :k])
    out = {'k': k, 'W': W, 'U_sigma': U_sigma, 'U_iso': U_iso, 'P_sigma': projector(U_sigma),
           'P_iso': projector(U_iso), 'whitening_error': werr, 'gap_sigma': gap_s, 'gap_iso_physical': gap_x,
           'L': L, 'Li': Li, 'sigma': sigma, 'p_sigma_vs_V_theta': None}
    if default:
        # addendum: P_Sigma = the leading k columns of V(theta) by mu (mu strictly decreasing)
        Vth = g.V(c['theta'])[:, np.argsort(g.mu)[::-1][:k]]
        out['p_sigma_vs_V_theta'] = float(np.max(np.abs(projector(Vth) - out['P_sigma'])))
        if out['p_sigma_vs_V_theta'] > 1e-9:
            raise Refusal('P_Sigma from K(theta) differs from the leading columns of V(theta)')
    return out


def output_covariance(F, W):
    """C_f = Y^T Y / (n - 1), Y = (F - 1 fbar^T) W^T; F is n x D with one record per ROW."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    if F.ndim != 2 or F.shape[1] != W.shape[1] or n < 2 or not np.isfinite(F).all():
        raise Refusal('model outputs malformed, non-finite or too few records')
    Y = (F - F.mean(0)) @ W.T
    return Y.T @ Y / (n - 1)


def endpoints(F, refs):
    """S1/S2 for one model's outputs F (n x D, physical, row per record)."""
    k, W = refs['k'], refs['W']
    C = output_covariance(F, W)
    nu, V = eigh_desc(C)
    tr = float(np.trace(C))
    D = len(nu)
    row = {'trace_C': tr, 'nu_k': float(nu[k - 1]), 'nu_k_plus_1': float(nu[k]), 'nu_1': float(nu[0]),
           'status': 'ok', 'd_sigma': None, 'd_iso': None, 'max_angle_sigma_deg': None, 'max_angle_iso_deg': None,
           'angles_sigma_deg': None, 'angles_iso_deg': None, 'leakage': None, 'relative_gap': None,
           'numerical_rank': None}
    if not tr > 0:
        row['status'] = 'zero_output_variance'
        return row
    tau = TIE_EPS_FACTOR * np.finfo(float).eps * D * nu[0]
    row['numerical_rank'] = int(np.sum(nu > tau))
    row['relative_gap'] = float((nu[k - 1] - nu[k]) / nu[0])
    row['leakage'] = float(1.0 - nu[:k].sum() / tr)
    if row['numerical_rank'] < k:
        row['status'] = 'rank_deficient'
        return row
    if not nu[k - 1] - nu[k] > tau:
        row['status'] = 'unresolved_tie'
        return row
    U = V[:, :k]
    P = projector(U)
    a_s, a_i = principal_angles_deg(U, refs['U_sigma']), principal_angles_deg(U, refs['U_iso'])
    row.update(d_sigma=subspace_distance(P, refs['P_sigma'], k), d_iso=subspace_distance(P, refs['P_iso'], k),
               max_angle_sigma_deg=float(a_s[0]), max_angle_iso_deg=float(a_i[0]),
               angles_sigma_deg=[float(x) for x in a_s], angles_iso_deg=[float(x) for x in a_i])
    return row


# --------------------------------------------------------------------------------------
# reduction (descriptive)
# --------------------------------------------------------------------------------------
def describe(values, n_expected, reasons=None):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    out = {'n_seeds': n_expected, 'n_defined': int(v.size), 'complete': bool(v.size == n_expected)}
    out['summary_label'] = ('complete registered-seed summary' if out['complete'] else
                            f'INCOMPLETE available-case summary ({v.size}/{n_expected} seeds defined); '
                            'not the registered complete summary; no imputation')
    rs = {}
    for r in (reasons or []):
        if r != 'ok':
            rs[r] = rs.get(r, 0) + 1
    out['undefined_reasons'] = json_dumps_sorted(rs)
    if v.size == 0:
        out.update(median=None, q1=None, q3=None, iqr=None)
    else:
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        out.update(median=float(med), q1=float(q1), q3=float(q3), iqr=float(q3 - q1))
    return out


def json_dumps_sorted(d):
    import json as _json
    return _json.dumps(d, sort_keys=True)


def reduce_rows(c, rows):
    """Medians/IQRs over the registered seeds (each seed exactly once per group), the seed-paired
    loss difference, and the descriptive status of E1-E3."""
    seeds = list(c['seeds'])
    keyset = {}
    for r in rows:
        key = (r['class'], r['loss'], r['N'])
        keyset.setdefault(key, {})
        if r['seed'] in keyset[key]:
            raise Refusal(f'duplicate seed {r["seed"]} in {key}')
        keyset[key][r['seed']] = r
    expected = {(cl, lo, n) for cl in c['classes'] for lo in c['losses'] for n in c['N']}
    if set(keyset) != expected:
        raise Refusal('endpoint rows do not cover the class x loss x N grid')
    for key, by_seed in keyset.items():
        if sorted(by_seed) != sorted(seeds):
            raise Refusal(f'seed set differs from the registered seeds in {key}')
    summary = []
    for (cl, lo, n) in sorted(expected):
        for m in METRICS:
            grp = [keyset[(cl, lo, n)][s] for s in seeds]
            reasons = [g.get('status', 'ok') if g[m] is None else 'ok' for g in grp]
            summary.append({'class': cl, 'loss': lo, 'N': n, 'metric': m,
                            **describe([g[m] for g in grp], len(seeds), reasons)})
    paired = []
    for cl in c['classes']:
        for n in c['N']:
            diffs, reasons = [], []
            for s in seeds:
                ra, rb = keyset[(cl, 'sigma_inverse', n)][s], keyset[(cl, 'isotropic', n)][s]
                a, b = ra['d_sigma'], rb['d_sigma']
                diffs.append(None if a is None or b is None else a - b)
                reasons.append('ok' if a is not None and b is not None else
                               '+'.join(sorted({r.get('status', 'undefined') for r in (ra, rb) if r['d_sigma'] is None})))
            paired.append({'class': cl, 'N': n, 'quantity': 'd_sigma(sigma_inverse) - d_sigma(isotropic)',
                           **describe(diffs, len(seeds), reasons)})
    S = {(r['class'], r['loss'], r['N'], r['metric']): r for r in summary}

    def compare(a, b):
        ra, rb = S[a], S[b]
        if not (ra['complete'] and rb['complete']):
            return {'holds': None, 'status': 'not_evaluable_incomplete',
                    'n_defined': [ra['n_defined'], rb['n_defined']], 'n_seeds': ra['n_seeds']}
        return {'holds': bool(ra['median'] < rb['median']), 'status': 'evaluated_on_complete_summaries',
                'medians': [ra['median'], rb['median']]}
    e1 = [{'class': cl, 'N': n, **compare((cl, 'sigma_inverse', n, 'd_sigma'), (cl, 'isotropic', n, 'd_sigma'))}
          for cl in c['classes'] for n in c['N']]
    e2 = [{'class': cl, 'N': n, **compare((cl, 'isotropic', n, 'd_iso'), (cl, 'isotropic', n, 'd_sigma'))}
          for cl in c['classes'] for n in c['N']]
    e3 = ([{'loss': lo, 'N': n, 'median_leakage': S[('deep_silu', lo, n, 'leakage')]['median'],
            'summary_label': S[('deep_silu', lo, n, 'leakage')]['summary_label']}
           for lo in c['losses'] for n in c['N']] if 'deep_silu' in c['classes'] else [])
    expectations = {
        'status': 'descriptive status of registered expectations (addendum section 4); not tests, no threshold',
        'E1_sigma_loss_nearer_P_sigma': e1, 'E2_iso_loss_nearer_P_iso_than_P_sigma': e2,
        'E3_silu_leakage_reported': e3,
        'section5_qualitative_sentence': ('withdrawn by overlay v1 (' + OVERLAY_MS + '): report the registered '
                                          'distances, angles and leakage and the relative comparisons E1/E2 only; '
                                          'no absolute-closeness wording'),
        'section5_never': 'no claim of superposition or of a nonlinear representation from leakage alone',
        'reading_order': 'read only after the X3 primary verdicts (P1, P2, SURE agreement) are recorded'}
    return summary, paired, expectations


# --------------------------------------------------------------------------------------
# input checks (all before any endpoint is computed)
# --------------------------------------------------------------------------------------
def check_loaded_modules():
    """Every loaded in-repository module is this package, the frozen X3 package, or a byte-identical
    accepted-T25 closure file.  The frozen guard exempts only its own package, so the same check is
    applied here with both packages exempt (the X3 files are covered by the code identity)."""
    closure = x3guard.load_closure()
    x3guard.assert_repo_origin()
    own = {str(PACKAGE.relative_to(EXP_ROOT)) + '/', str(x3guard.PACKAGE.relative_to(EXP_ROOT)) + '/'}
    reused = [r for r in x3guard.loaded_repo_modules()
              if not any(r.startswith(o) for o in own) and Path(r).name != 'conftest.py']
    try:
        return x3guard.assert_files(reused, closure)
    except x3guard.AcceptedCodeMismatch as e:
        raise Refusal(f'loaded module outside the frozen closure: {e}') from e


def check_addendum():
    p = x3w.msroot() / ADDENDUM_MS
    if not p.is_file():
        raise Refusal(f'registered addendum missing at {p}')
    got = x3w.sha(p)
    if got != ADDENDUM_SHA256:
        raise Refusal(f'registered addendum bytes changed ({got[:12]} != {ADDENDUM_SHA256[:12]})')
    return got


def check_export(root, mode, primary_verdicts=None):
    root = Path(root).resolve()
    if not (root / 'freeze.json').is_file():
        raise Refusal('not an X3 export (freeze.json absent)')
    f = x3w.read(root / 'freeze.json')
    c = f.get('config')
    want = BASE if mode == 'confirmation' else FIXTURE
    if c != want:
        raise Refusal(f'export configuration is not the registered {mode} configuration')
    if (f.get('deployment') != x3w.VERSION or f.get('config_identity') != x3w.identity(c)
            or f.get('enumeration') != enumeration(c)):
        raise Refusal('freeze record identity/config/enumeration mismatch')
    if mode == 'confirmation' and (c['k'] != REGISTERED_K or c['evaluation']['records'] != REGISTERED_EVAL_RECORDS):
        raise Refusal('confirmation configuration differs from the registered k / evaluation size')
    try:
        exported = x3w.check_export(root, f)
        x3w.check_ledgers(root, f)
    except ValueError as e:
        raise Refusal(f'export/ledger check failed: {e}') from e
    if not exported:
        raise Refusal('SHA256SUMS / EXPORT.json / tables absent: only an exported run is accepted')
    e = x3w.read(root / 'EXPORT.json')
    if e['code_identity'] != REGISTERED_X3_IDENTITY:
        raise Refusal('export record does not carry the registered X3 code identity')
    if x3w.code_identity(f['sources']) != REGISTERED_X3_IDENTITY:
        raise Refusal('freeze sources do not hash to the registered X3 code identity')
    if x3w.code_identity() != REGISTERED_X3_IDENTITY:
        raise Refusal('the frozen X3 source on disk does not hash to the registered identity')
    want_status = 'CONFIRMATION_EXPORT' if mode == 'confirmation' else 'DEVELOPMENT_EXPORT'
    if e['status'] != want_status:
        raise Refusal(f'export status {e["status"]} is not {want_status}')
    cells = f['enumeration']['cells']
    for i, cell in enumerate(cells):
        d = x3w.cell_dir(root, i)
        comp = x3w.read(d / 'COMPLETE.json')
        if comp != {'arrays_sha': x3w.sha(d / 'arrays.npz'), 'record_sha': x3w.sha(d / 'record.json')}:
            raise Refusal(f'cell {i}: COMPLETE.json does not match arrays/record')
        r = x3w.read(d / 'record.json')
        if (r.get('seed'), r.get('N')) != (cell['seed'], cell['N']):
            raise Refusal(f'cell {i}: record identity differs from the enumeration')
    return f, e, check_verdicts(root, mode, primary_verdicts)


def check_verdicts(root, mode, primary_verdicts=None):
    """Addendum section 5: S1/S2 are read only after the X3 primary verdicts are recorded.  In confirmation
    mode the receiver must pass the recorded verdicts, and they must equal the reduction's."""
    red = x3w.read(Path(root) / 'reduction.json')
    verdicts = {k: red['predictions'][k]['verdict'] for k in ('P1', 'P2', 'SURE_agreement')}
    if mode == 'confirmation' or primary_verdicts is not None:
        if primary_verdicts is None:
            raise Refusal('record the X3 primary verdicts (P1, P2, SURE agreement) first and pass --primary-verdicts')
        rec = json.loads(Path(primary_verdicts).read_text())
        if {k: rec.get(k) for k in verdicts} != verdicts:
            raise Refusal('recorded primary verdicts differ from the export reduction')
    return verdicts


def check_data(c, root, cells):
    """Regenerate every seed's data; require the recorded training and evaluation digests, the stored
    closed-form maps, every model record and the deep-model weights.  This pass precedes, and is
    separate from, the endpoint pass."""
    by_seed = {}
    for i, cell in enumerate(cells):
        by_seed.setdefault(cell['seed'], []).append(i)
    for seed, idx in by_seed.items():
        g, sigma, L, Li, train, eclean, enoisy = x3s.make_data(c, seed)
        ed = x3s.digest(enoisy)
        Wmat = {'isotropic': np.eye(g.d), 'sigma_inverse': Li @ Li}
        for i in idx:
            d = x3w.cell_dir(root, i)
            r = x3w.read(d / 'record.json')
            N = r['N']
            x, _, _ = x3s.moments(train, N)
            if r['evaluation_sha'] != ed or r['training_sha'] != x3s.digest(x):
                raise Refusal(f'BLOCKED: cell {i}: regenerated data digest differs from the record. Do not weaken '
                              'this check; run on the original X3 host and pinned environment after the frozen '
                              'audit and primary verdicts (overlay v1 section 3), else report TR03 as blocked')
            n_tr, _ = x3s.split(N, c['recipe']['validation_fraction'])
            _, mu, S = x3s.moments(train, n_tr)
            a = dict(np.load(d / 'arrays.npz', allow_pickle=False))
            for loss in c['losses']:
                A, b = x3s.closed_form(c, S, mu, Wmat[loss], c['k'])
                tag = f'tied_linear__{loss}'
                if tag + '__A' not in a or tag + '__b' not in a:
                    raise Refusal(f'cell {i}: closed-form arrays {tag} absent')
                if not (np.allclose(a[tag + '__A'], A, rtol=CLOSED_FORM_RTOL, atol=CLOSED_FORM_ATOL)
                        and np.allclose(a[tag + '__b'], b, rtol=CLOSED_FORM_RTOL, atol=CLOSED_FORM_ATOL)):
                    raise Refusal(f'cell {i}: stored closed-form map differs ({tag})')
            for cls in c['classes']:
                for loss in c['losses']:
                    tag = f'{cls}__{loss}'
                    if tag not in r['models']:
                        raise Refusal(f'cell {i}: model record {tag} absent')
                    if cls != 'tied_linear' and not x3s.state_of(a, tag):
                        raise Refusal(f'cell {i}: weights for {tag} absent')
    return {'seeds_checked': len(by_seed), 'cells_checked': len(cells)}


# --------------------------------------------------------------------------------------
# endpoint pass
# --------------------------------------------------------------------------------------
def model_outputs(c, torch, arrays, tag, X):
    cls = tag.split('__')[0]
    if cls == 'tied_linear':
        return X @ arrays[tag + '__A'].T + arrays[tag + '__b']
    model = x3s.build_model(c, torch, cls, x3s.state_of(arrays, tag))     # strict state_dict load
    with torch.no_grad():
        return model(x3s.tensor(torch, X)).numpy().copy()


def compute(c, root, cells, refs=None):
    torch = x3s.import_torch(c)          # pinned torch 2.9.1, float64, deterministic, one thread
    refs = refs or references(c)
    rows, by_seed = [], {}
    for i, cell in enumerate(cells):
        by_seed.setdefault(cell['seed'], []).append(i)
    for seed, idx in by_seed.items():
        *_, enoisy = x3s.make_data(c, seed)
        for i in idx:
            d = x3w.cell_dir(root, i)
            r = x3w.read(d / 'record.json')
            a = dict(np.load(d / 'arrays.npz', allow_pickle=False))
            for cls in c['classes']:
                for loss in c['losses']:
                    tag = f'{cls}__{loss}'
                    row = endpoints(model_outputs(c, torch, a, tag, enoisy), refs)
                    row.update({'seed': seed, 'N': r['N'], 'class': cls, 'loss': loss, 'cell': i})
                    if cls in LINEAR_CLASSES:
                        row['linear_leakage_check'] = bool(row['leakage'] is not None
                                                           and abs(row['leakage']) < LINEAR_CHECK_TOL)
                    rows.append(row)
    return rows, refs


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------
ROW_FIELDS = ['seed', 'N', 'class', 'loss', 'cell', 'status', 'd_sigma', 'd_iso', 'max_angle_sigma_deg',
              'max_angle_iso_deg', 'leakage', 'trace_C', 'nu_1', 'nu_k', 'nu_k_plus_1', 'relative_gap',
              'numerical_rank', 'linear_leakage_check',
              'angles_sigma_deg', 'angles_iso_deg']
SUMMARY_FIELDS = ['class', 'loss', 'N', 'metric', 'n_seeds', 'n_defined', 'complete', 'summary_label',
                  'undefined_reasons', 'median', 'q1', 'q3', 'iqr']
PAIRED_FIELDS = ['class', 'N', 'quantity', 'n_seeds', 'n_defined', 'complete', 'summary_label', 'undefined_reasons',
                 'median', 'q1', 'q3', 'iqr']


def _cell(v):
    if isinstance(v, list):
        return json.dumps(v)
    if v is None:
        return ''
    if isinstance(v, float):
        return repr(v)
    return v


def to_csv(rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r.get(k)) for k in fields})
    return buf.getvalue()


def write_new_text(p, text):
    with Path(p).open('x') as fh:
        fh.write(text)


def addendum_files():
    return {str(p.relative_to(EXP_ROOT)): x3w.sha(p) for p in sorted(PACKAGE.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.md') and '__pycache__' not in p.parts}


def run(mode, export, out, primary_verdicts=None):
    out = Path(out).resolve()
    export = Path(export).resolve()
    if out.exists():
        raise SystemExit('output directory must not exist')
    if out.is_relative_to(export) or export.is_relative_to(out):
        raise SystemExit('output overlaps the export')
    out.mkdir(parents=True)
    try:
        check_loaded_modules()
        addendum_sha = check_addendum()
        f, e, verdicts = check_export(export, mode, primary_verdicts)
        c = f['config']
        cells = f['enumeration']['cells']
        data = check_data(c, export, cells)
        rows, refs = compute(c, export, cells)
        summary, paired, expectations = reduce_rows(c, rows)
    except Refusal as err:
        x3w.write_new(out / 'REFUSED.json', {'refused': True, 'reason': str(err), 'version': VERSION})
        raise
    except BaseException as err:  # any other failure: no endpoint file is written either
        x3w.write_new(out / 'REFUSED.json', {'refused': True, 'reason': f'error: {err!r}',
                                             'traceback': traceback.format_exc(), 'version': VERSION})
        raise
    lin = [r for r in rows if r['class'] in LINEAR_CLASSES]
    checks = {'linear_leakage_check': {'tolerance': LINEAR_CHECK_TOL, 'rows': len(lin),
                                       'passed': sum(bool(r['linear_leakage_check']) for r in lin),
                                       'failed': [{k: r[k] for k in ('seed', 'N', 'class', 'loss', 'leakage')}
                                                  for r in lin if not r['linear_leakage_check']]},
              'undefined_rows': [{k: r[k] for k in ('seed', 'N', 'class', 'loss', 'status')}
                                 for r in rows if r['status'] != 'ok'],
              'references': {'whitening_error': refs['whitening_error'], 'gap_sigma': refs['gap_sigma'],
                             'gap_iso_physical': refs['gap_iso_physical'],
                             'p_sigma_vs_V_theta': refs['p_sigma_vs_V_theta']},
              'data': data}
    write_new_text(out / 'per_seed.csv', to_csv(rows, ROW_FIELDS))
    write_new_text(out / 'summary.csv', to_csv(summary, SUMMARY_FIELDS))
    write_new_text(out / 'paired.csv', to_csv(paired, PAIRED_FIELDS))
    x3w.write_new(out / 'expectations.json', expectations)
    x3w.write_new(out / 'checks.json', checks)
    import torch
    x3w.write_new(out / 'RUN.json', {
        'version': VERSION, 'mode': mode,
        'label': 'registered before the run, outside the reviewed package, descriptive',
        'addendum': {'path': ADDENDUM_MS, 'sha256': addendum_sha, 'registration_commit': ADDENDUM_REGISTRATION_COMMIT},
        'export': {'path': str(export), 'sha256sums': x3w.sha(export / 'SHA256SUMS'), 'export_record_status': e['status'],
                   'export_code_identity': e['code_identity'], 'reduction_sha256': e['reduction_sha256'],
                   'primary_verdicts': verdicts},
        'x3_code_identity': REGISTERED_X3_IDENTITY, 'addendum_package_files': addendum_files(),
        'overlay': OVERLAY_MS,
        'declared_choices': {'tie_eps_factor': TIE_EPS_FACTOR, 'linear_check_tol': LINEAR_CHECK_TOL,
                             'iqr': 'numpy linear percentile interpolation', 'k': c['k'],
                             'two_largest_angles_reading': 'largest principal angle to each reference'},
        'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                        'torch': torch.__version__}})
    return {'rows': len(rows), 'out': str(out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description='TR03: registered X3 representation endpoints S1/S2 (descriptive)')
    ap.add_argument('mode', choices=('confirmation', 'fixture'))
    ap.add_argument('--export', required=True, help='extracted X3 export directory (contains freeze.json, SHA256SUMS)')
    ap.add_argument('--out', required=True, help='fresh output directory')
    ap.add_argument('--primary-verdicts', help='JSON {"P1": .., "P2": .., "SURE_agreement": ..} recorded from the '
                                                'reduction BEFORE this run (required in confirmation mode)')
    a = ap.parse_args(argv)
    for k in THREAD_VARS:
        if os.environ.get(k) != '1':
            raise SystemExit(f'set {k}=1 (the frozen data regeneration is compared byte for byte)')
    print(json.dumps(run(a.mode, a.export, a.out, a.primary_verdicts)))


if __name__ == '__main__':
    main()
