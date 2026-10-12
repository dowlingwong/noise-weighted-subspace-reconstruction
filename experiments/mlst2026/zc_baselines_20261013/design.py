"""ZC registered design: grid, seeds, metric arms, methods, families and reading rules.

`build_protocol(dependencies)` is the only source of protocol.json (written by
`runner write-protocol`; the preflight rebuilds it and refuses on any difference).
`smoke_protocol()` is the same construction on the T25 FIXTURE geometry (C = 3, T = 4), a smoke-only
namespace and a fabricated N* table; it is used by tests only.
"""
from __future__ import annotations

from experiments.mlst2026.y19_20261012.generator import seed_int          # T25 byte guard runs on import
from experiments.mlst2026.t25_production_20260930 import protocol as t25p  # noqa: E402  (BASE / FIXTURE)
from . import nstar                                                         # noqa: E402

NAMESPACE = 'zc-confirm-20261013'
DEV_NAMESPACE = 'zc-dev-20261013'
SMOKE_NAMESPACE = 'zc-smoke-fixture'
N_SEEDS = 24
FOLDS = 5
LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
ALPHA = 0.05
GEN_KEYS = ('version', 'C', 'T', 'kc', 'kt', 'spectrum', 'geometry_seed', 'rho_c', 'rho_t')


def seeds(namespace: str = NAMESPACE, n: int = N_SEEDS) -> list[int]:
    """SHA-256 seed derivation (Y19 construction): first 8 bytes little-endian of sha256('<ns>/seed/<i>') mod 1e12."""
    return [seed_int(namespace, 'seed', i) % 10 ** 12 for i in range(n)]


def generator(base: dict) -> dict:
    return {k: base[k] for k in GEN_KEYS}


def _families(thetas, Ns, metrics_all, rc1_cells, alpha):
    rc2 = [[th, N, m] for th in thetas for N in Ns for m in metrics_all]
    return {
        'R-C1': {'metric': 'oracle', 'cells': [[th, N, 'oracle'] for th, N in rc1_cells],
                 'contrast': 'per seed: clean risk(N*-rule) - clean risk(CV-select)  (= regret(N*-rule) - regret(CV-select))',
                 'family_size': len(rc1_cells), 'alpha': alpha},
        'R-C2': {'cells': rc2,
                 'contrast': 'per seed: clean risk(SHR(lambda_hat)) - clean risk(better hard class), better hard class = '
                             'the class (U or F) with the smaller 24-seed mean clean risk in that (theta, N, metric) cell '
                             '(ties -> U)',
                 'family_size': len(rc2), 'alpha': alpha}}


def _build(*, base, namespace, seed_list, thetas, Ns, m, metrics_all, metrics_cv_only, nstar_table, nstar_source,
           rc1_cells, max_iter, dependencies, evidence_role, study='ZC'):
    fam = _families(thetas, Ns, metrics_all, rc1_cells, ALPHA)
    return {
        'study': study, 'version': 'zc-v1', 'evidence_role': evidence_role, 'namespace': namespace,
        'generator': generator(base),
        'generator_note': 't25_production_20260930 protocol BASE generator fields (FIXTURE for smoke); records drawn by '
                          't25.science.make_data(cfg, seed, theta) with cfg = generator | {namespace, N, m}; '
                          'calibration pool keyed (namespace, seed), shared across theta; training pool of max(N) '
                          'records keyed (namespace, seed, theta), smaller N are prefixes',
        'theta': list(thetas), 'N': list(Ns), 'm': list(m),
        'metrics': {'all_methods': list(metrics_all), 'cv_select_only': list(metrics_cv_only),
                    'definitions': {'oracle': 'W = inv(Sigma) (true noise covariance)',
                                    'kron<m>': 'W = inv(t25.estimate(calibration[:m], C, T)["kron"]) (regularized '
                                               'separable moment estimate, T25)',
                                    'full<m>': 'W = inv(t25.estimate(calibration[:m], C, T)["full"]) (regularized full '
                                               'sample covariance, T25)'}},
        'seeds': list(seed_list),
        'seed_derivation': f'int.from_bytes(sha256("{namespace}/seed/<i>")[:8], "little") mod 10^12, i = 0..{len(seed_list) - 1}',
        'task': '(seed, theta, N, metric); one JSON record per task',
        'learners': {
            'moments': 't25.science.moments (fitted mean, centred, divisor N)',
            'U': 't25.science.unrestricted(W, S, k)',
            'F': 't25.science.structured(W, S, t25.science.starts(S, C, T, kc, kt, key), max_iter, tol); '
                 'full-data key (namespace, seed, theta, N) as T25; CV fold j key (namespace, seed, theta, N, "cv-fold", j)',
            'affine_map': 'A = t25.map_for(B, W), b = (I - A) mean (fitted mean)',
            'max_iter': max_iter, 'gradient_tolerance': 1e-8, 'restarts': 4},
        'methods': {
            'CV': f'{FOLDS}-fold CV over {{U, F}}: folds = methods.folds(N, (namespace, seed, theta, N, "cv-folds"), {FOLDS}) '
                  '(same folds for every metric arm and for SHR); score = mean held-out observed weighted residual '
                  '(x - A x - b)^T W (x - A x - b) in the supplied metric W over all N held-out records; choose the '
                  'smaller (exact tie -> U); refit the chosen class on all N (= the full-data fit)',
            'RULE': 'N*-rule: product iff N < N*_pred(theta); N*_pred = None (theta = 0) -> product; the chosen '
                    'full-data fit',
            'SHR': 'top-k eigenspace of K_lambda = (1 - lambda) S_w + lambda NKP(S_w), S_w = W^{1/2} S W^{1/2} (whitened '
                   'centred sample second moment in the supplied metric), NKP = Van Loan-Pitsianis nearest Kronecker '
                   'product in the T25 vec convention (both factors projected to PSD; nkp.py); basis W^{-1/2} Q as '
                   't25.unrestricted; lambda in the grid chosen by the same folds and score (tie -> smallest lambda), '
                   'refitted on all N. lambda = 0 is bitwise U. lambda = 1 is NOT NFPA (top-k eigenvectors of '
                   'A_t (x) A_c need not form a k_c x k_t grid): SHR is a third estimator, not an interpolation '
                   'path between the hard classes. Run only on the all-methods metric arms (W^{1/2} Kronecker).',
            'metric_caveat': 'For a fixed affine map (A, b) and an independent record x = s + n, E||x - A x - b||_W^2 - '
                             'E||s - A x - b||_W^2 = tr(W Sigma) - 2 tr(W A Sigma), which depends on the fitted map '
                             'unless W = Sigma^{-1} (Prop. 2.1, where it equals d - 2k for every rank-k idempotent map). '
                             'Hence only on the oracle arm do held-out residual differences equal clean-risk differences '
                             'in expectation; on the estimated arms CV and the SHR lambda selection rank by the '
                             'W-weighted observed (noisy-target) residual, which is neither the W-clean nor the '
                             'Sigma^{-1}-clean risk, and their regret combines the data-split cost with this metric '
                             'mismatch.',
            'lambda_grid': list(LAMBDAS), 'folds': FOLDS},
        'nstar': {'table': nstar_table, 'source': nstar_source,
                  'definition': 'N*_pred = (c_U - c_F)/a - 1, supplement S4.3 first-order crossing (T20-S rows)'},
        'scoring': 't25.science.risks(A, b, L K L, Sigma, Sigma^{-1})["clean"]: exact expected clean risk in the true '
                   'Sigma^{-1} metric on an independent record (closed form), for every method and metric arm',
        'regret': 'per seed and cell: regret(X) = clean risk(X) - min(clean risk(U), clean risk(F)) (descriptive); '
                  'also reported against the per-cell better class (smaller seed-mean risk)',
        'optimization_limited': 'full-data F fit flagged if the selected NFPA restart is not stationary or the restart '
                                'clean-risk spread >= |R_F - R_U| (T25 reduce_records rule); CV fold F fits report '
                                'non-stationary selected restarts; flagged records are kept, never dropped',
        'inference': {'unit': 'seed-level contrast, mean over seeds', 'interval': 'approximate simultaneous Bonferroni '
                      'Student-t over each registered family', 'alpha': ALPHA, 'df': len(seed_list) - 1,
                      'decided_negative': 'upper limit < 0', 'decided_positive': 'lower limit > 0',
                      'zero_SE': 'a cell whose seed SD is exactly 0 is undecided'},
        'families': fam,
        'reading_notes': {
            'R-C1': 'a cell where CV-select and the N*-rule chose the same class in every seed has contrast exactly 0 '
                    '(zero SD) and is undecided, i.e. "not larger"; report it as "CV and the rule chose the same class '
                    'in all seeds" (rows carry zero_SE and cv_choice_F)',
            'R-C2': 'the better hard class is chosen on the same seeds, which biases the reference low (conservative '
                    'against SHR); with family size 72 a MET reading means "no cell showed SHR decidedly worse than '
                    'the better hard class (simultaneous Bonferroni-t over 72 cells)", not "SHR is safe"; marginal '
                    'unadjusted intervals and the largest positive seed-mean contrast are reported descriptively',
            'full_arms': 'the N*-rule is not recorded on the full-estimate arms (E1); its choice and regret there are '
                         'exactly derivable from the recorded U and F clean risks'},
        'reading_rule': {
            'R-C1': 'MET iff no R-C1 cell is decided positive (undecided and decided negative both count as "the '
                    'N*-rule regret is not larger than CV-select regret"); NOT MET otherwise, with the cells listed',
            'R-C2': 'MET iff no R-C2 cell is decided positive ("safe interpolation"); NOT MET otherwise, with the cells '
                    'listed',
            'optimization_limited': 'reported with each reading; deciding cells with optimization-limited records '
                                    'carry a flag'},
        'descriptive': ['per-cell seed-mean clean risk of U, F, CV, RULE, SHR', 'regret tables (vs per-seed min and '
                        'vs per-cell better class) for CV, RULE, SHR', 'CV choice frequencies', 'lambda_hat '
                        'distribution', 'CV-select regret on the full-estimate arms', 'SHR clean risk along the '
                        'whole lambda path (full-data fits)'],
        'exclusions': 'none; every record is kept',
        'dev_namespace': DEV_NAMESPACE,
        'dependencies': dependencies,
    }


def build_protocol(dependencies: dict) -> dict:
    base = t25p.BASE
    return _build(base=base, namespace=NAMESPACE, seed_list=seeds(), thetas=base['theta'], Ns=base['N'], m=[256, 1024],
                  metrics_all=['oracle', 'kron256', 'kron1024'], metrics_cv_only=['full256', 'full1024'],
                  nstar_table=nstar.table(),
                  nstar_source={'MS_path': nstar.SOURCE_MS_PATH, 'kit_copy': 'pinned/y15_RESULT_20261012.json',
                                'sha256': nstar.SOURCE_SHA256, 'rows': nstar.STUDY_ROWS},
                  rc1_cells=[(0.25, 400), (0.25, 800), (0.25, 1000), (0.4, 400)], max_iter=base['max_iter'],
                  dependencies=dependencies, evidence_role='CONFIRMATION')


def smoke_protocol(n_seeds: int = 3) -> dict:
    base = t25p.FIXTURE
    return _build(base=base, namespace=SMOKE_NAMESPACE, seed_list=seeds(SMOKE_NAMESPACE, n_seeds), thetas=base['theta'],
                  Ns=base['N'], m=[32, 64], metrics_all=['oracle', 'kron32', 'kron64'],
                  metrics_cv_only=['full32', 'full64'], nstar_table={'0.0': None, '0.3': 60.0},
                  nstar_source={'fabricated': 'smoke only'}, rc1_cells=[(0.3, 40)], max_iter=base['max_iter'],
                  dependencies={'files': {}, 'in_accepted_t25_closure': []},
                  evidence_role='SMOKE (T25 fixture geometry; fabricated N* table)')
