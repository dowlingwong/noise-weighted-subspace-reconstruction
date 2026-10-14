"""Interleaved block partition for MEG R1 (reference implementation; R2 must reproduce it exactly).

Roles are assigned to gaps from gap counts only (never from data values):
  * within each run r = 1..6, the gaps g = 0..G_r-1 (chronological, as defined by preprocess.noise_windows /
    read_events) are cut into consecutive blocks of B = 12 gaps; a remainder < B is appended to the last block;
  * block b of run r gets role CYCLE[(b + r - 1) mod 6], CYCLE = (ref, train, ref, cal, train, eval)
    (proportions 2:2:1:1 as in the run-blocked design; the run offset rotates every role through every
    within-run block position across the six runs);
  * guard: the first gap of every block except block 0 of a run gets role 'guard' (no windows used), so no
    two gaps of different roles are adjacent inside a run.
Deterministic, seed-free. Usage: python PARTITION_RULE.py R0_EVIDENCE.json > PARTITION_CENSUS.json
"""
import json, sys

B = 12
CYCLE = ('ref', 'train', 'ref', 'cal', 'train', 'eval')


def assign_roles(G: int, run: int, B: int = B, cycle=CYCLE) -> list:
    if G <= 0:
        return []
    nblocks = max(1, G // B)
    roles = []
    for g in range(G):
        b = min(g // B, nblocks - 1)
        start = b * B
        role = cycle[(b + run - 1) % len(cycle)]
        roles.append('guard' if (b > 0 and g == start) else role)
    return roles


def census(evidence: dict, Ts=(20, 16, 12, 10), budgets=(1600, 768, 500), margin=1.10) -> dict:
    out = {'rule': {'B': B, 'cycle': CYCLE, 'guard': 'first gap of every non-initial block'},
           'budgets_with_margin': {'train': budgets[0] * margin, 'cal': budgets[1] * margin, 'eval': budgets[2] * margin},
           'subjects': {}}
    for sub, s in evidence['subjects'].items():
        rows = {}
        for T in Ts:
            runs = s['R0_d']['events'][str(T)]['runs']
            tot = {'train': 0, 'cal': 0, 'eval': 0, 'ref_stride_T': 0, 'ref_stride_1': 0, 'guard_gaps': 0, 'gaps': 0}
            per_run = {}
            for r, v in runs.items():
                pg = sorted(v['per_gap'], key=lambda x: x['gap'])
                assert [x['gap'] for x in pg] == list(range(len(pg))), 'gap indices not contiguous'
                roles = assign_roles(len(pg), int(r))
                cnt = {k: 0 for k in ('ref', 'train', 'cal', 'eval', 'guard')}
                for x, role in zip(pg, roles):
                    cnt[role] += 1
                    tot['gaps'] += 1
                    if role == 'guard':
                        tot['guard_gaps'] += 1
                    elif role == 'ref':
                        tot['ref_stride_T'] += x['stride_T']; tot['ref_stride_1'] += x['stride_1']
                    else:
                        tot[role] += x['stride_T']
                per_run[r] = {'G': len(pg), 'gaps_by_role': cnt}
            d = 24 * T
            ok = (tot['train'] >= budgets[0] * margin and tot['cal'] >= budgets[1] * margin and tot['eval'] >= budgets[2] * margin)
            rows[str(T)] = {**tot, 'd': d, 'n_eff_over_d': tot['ref_stride_T'] / d,
                            'wishart_inflation_stride_T': tot['ref_stride_T'] / (tot['ref_stride_T'] - d - 1),
                            'meets_budgets_with_10pct_margin': ok, 'per_run': per_run}
        out['subjects'][sub] = rows
    allok = {str(T): all(out['subjects'][s][str(T)]['meets_budgets_with_10pct_margin'] for s in out['subjects']) for T in Ts}
    out['T_rule'] = {'all_subjects_meet': allok, 'chosen_T': max([T for T in Ts if allok[str(T)]], default=None),
                     'note': 'upper-bound counts (BAD annotations not applied), from R0-d events-based per-gap counts'}
    return out


if __name__ == '__main__':
    print(json.dumps(census(json.load(open(sys.argv[1]))), indent=1))
