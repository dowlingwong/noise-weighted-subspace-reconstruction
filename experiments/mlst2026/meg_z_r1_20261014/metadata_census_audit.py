"""Counts-only audit: preserve historical rounded-TSV census vs exact sample clock; fixed-grid budget audit."""
import inspect,json
from .protocol import PACKAGE
from .partition import census
from .windows import window_plan

def audit():
    geom=json.loads((PACKAGE/'fixtures/DEV_GEOMETRY.json').read_text());r0=json.loads((PACKAGE/'fixtures/R0_EVIDENCE.json').read_text());out={}
    historical=json.loads(json.dumps(census(r0)))
    for sub,v in geom['subjects'].items():
        diffs=[];pools={ro:0 for ro in ('ref','train','cal','eval')}
        for R in v['runs']:
            for g,p in zip(R['gaps'][:-1],r0['subjects'][sub]['R0_d']['events']['10']['runs'][str(R['run'])]['per_gap']):
                ds=sum(b-a for a,b in g['stretches'])-p['allowed_samples'];dt=sum((b-a)//10 for a,b in g['stretches'])-p['stride_T']
                if ds or dt:diffs.append({'run':R['run'],'gap':g['gap'],'allowed_sample_delta':ds,'stride_T_delta':dt})
            for ro,rows in window_plan(R['gaps'],R['run']).items():pools[ro]+=len(rows)
        out[sub]={'exact_clock_pools_including_final_gaps':pools,'historical_T10':historical['subjects'][sub]['10'],
            'different_gap_count':len(diffs),'sum_stride_T_difference_nonfinal':sum(r['stride_T_delta'] for r in diffs),'differences':diffs}
    return {'note':'R0 uses rounded TSV onset minus37/1100; production clock is stable raw sample/1100 plus unchanged0.0345. Exact fixture uses TSV sample minus37. Historical PARTITION_CENSUS remains exactly reproduced independently; no threshold changed.','subjects':out}
def budget_audit():
    """R2-B F1 / RD-12a: role pools under the fixed onset grid (exact dev geometry, final gaps included, BAD masks not
    applied, as the census) against the unchanged budgets x 1.1 of partition.census, the T = 10 rule and m_hi."""
    from .gates import choose_m_hi
    defaults=inspect.signature(census).parameters;budgets=defaults['budgets'].default;margin=defaults['margin'].default
    need={'train':budgets[0]*margin,'cal':budgets[1]*margin,'eval':budgets[2]*margin}
    geom=json.loads((PACKAGE/'fixtures/DEV_GEOMETRY.json').read_text());out={}
    for sub,v in geom['subjects'].items():
        pools={ro:0 for ro in ('ref','train','cal','eval')}
        for R in v['runs']:
            for ro,rows in window_plan(R['gaps'],R['run'],10).items():pools[ro]+=len(rows)
        out[sub]={'pools':pools,'meets_budgets_with_10pct_margin':all(pools[k]>=need[k] for k in need),'eval_budget_exclusion':pools['eval']<500}
    m_hi=choose_m_hi({s:r['pools']['cal'] for s,r in out.items()})
    return {'convention':'fixed onset grid (DESIGN rev 2.3, RD-12a)','T':10,'budgets_with_margin':need,'subjects':out,'m_hi':m_hi,
            'pass':all(r['meets_budgets_with_10pct_margin'] and not r['eval_budget_exclusion'] for r in out.values())}

if __name__=='__main__':print(json.dumps({**audit(),'fixed_grid_budget_audit':budget_audit()},indent=1))
