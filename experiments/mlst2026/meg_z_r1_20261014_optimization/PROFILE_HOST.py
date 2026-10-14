"""Profile identical seeded baseline/optimized calls and sealed-shape equivalence."""
import cProfile,io,json,pstats,subprocess,os
from pathlib import Path
import numpy as np
from experiments.mlst2026.meg_z_r1_20261014 import runner,controls as c
from experiments.mlst2026.meg_z_r1_20261014.tests.test_optimization import baseline

shapes,geometry,hashes=runner.context('/ceph/dwong/meg_z/derived_timingfix_20261013',
    'experiments/mlst2026/meg_z_20261013_dev_timingfix/gates.json')
profiles={};seconds={}
for label,module in [('baseline',baseline),('optimized',c)]:
    pr=cProfile.Profile();pr.enable()
    r=module.replicate(4,7,0,0,shapes,geometry)
    pr.disable();stream=io.StringIO()
    pstats.Stats(pr,stream=stream).sort_stats('cumulative').print_stats(25)
    profiles[label]=stream.getvalue();seconds[label]=r['seconds']

errors={};output_errors={}
def compare(x,y):
    if isinstance(x,dict):
        assert x.keys()==y.keys()
        return max((compare(x[k],y[k]) for k in x),default=0.)
    if isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        return max((compare(a,b) for a,b in zip(x,y)),default=0.)
    if isinstance(x,(bool,str)) or x is None:
        assert x==y;return 0.
    assert abs(x-y)<=1e-9+1e-8*abs(y),(x,y)
    return float(abs(x-y)/max(1.,abs(y)))

for si,s in enumerate(runner.DEV):
    g1,_=c.rng(4,7,si,29);g2,_=c.rng(4,7,si,29)
    actual=c.generate(shapes[s],geometry['subjects'][s]['runs'],4,g1)
    expected=baseline.generate(shapes[s],geometry['subjects'][s]['runs'],4,g2)
    assert g1.bit_generator.state==g2.bit_generator.state
    err=max(float(np.max(np.abs(actual[ro][0]-expected[ro][0]))/
                  np.sqrt(np.mean(expected[ro][0]**2))) for ro in actual)
    assert err<1e-10,(s,err)
    errors[s]=err
    # Evaluate C7 including separately estimated metrics on the two coupled draws.
    S1=c.reference(actual['ref'][0]);S2=baseline.reference(expected['ref'][0])
    actual=c.apply_cell(actual,S1,shapes[s],7,g1)
    expected=baseline.apply_cell(expected,S2,shapes[s],7,g2)
    output_errors[s]=compare(c.assess(actual,Sref=S1),baseline.assess(expected,Sref=S2))
    assert g1.bit_generator.state==g2.bit_generator.state

out=Path(__file__).parent
data={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True,
        env=os.environ|{'GIT_OPTIONAL_LOCKS':'0'}).strip(),
      'environment':runner.environment(),'identity':runner.verify_kit(),
      'baseline_source':'dffa807','profiles':profiles,'profiled_seconds':seconds,
      'sealed_shape_AR05_max_error_over_RMS':errors,
      'sealed_shape_C7_max_output_error_scaled_by_max1_abs':output_errors,
      'source_hashes':hashes,'controls_campaign_not_run':True}
(out/'PROFILE.json').write_text(json.dumps(data,indent=1)+'\n')
print(json.dumps({k:v for k,v in data.items() if k not in ('profiles','source_hashes')},indent=1))
