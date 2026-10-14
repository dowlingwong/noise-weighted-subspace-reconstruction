"""Counts-only audit: preserve historical rounded-TSV census vs exact sample clock."""
import json
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
if __name__=='__main__':print(json.dumps(audit(),indent=1))
