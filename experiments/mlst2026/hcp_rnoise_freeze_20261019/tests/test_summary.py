import copy
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_freeze_20261019 import summary as s
from experiments.mlst2026.hcp_rnoise_freeze_20261019.pilot import ARMS,arm_key

def fixture(R=20,ref=False):
    dates=[f'd{i}' for i in range(9)]
    row={'a_best':1.,'B':-222.,'N_star_interval':[222.,300.], 'PZ2':{'in_range':True},'n_starts':64,
         'H_min_eig':1.,'H_max_eig':20.,'gradient_norm_at_optimum':1e-10,'restart_agreement':True,'all_starts_reach_best':True}
    pred={'units':{d:{'PZ1':{'met':True},'fixed':[row]*4,'adaptive':{'A':{'row':row},'B':{'row':row}},
                      'geometries':{g:{'grid':[56,111,222,444,888],'row':row} for g in s.GEOMS}} for d in dates}}
    rows=[]
    for i,d in enumerate(dates):
        for g in s.GEOMS:
            for N in [56,111,222,444,888]:
                for a,m in (ARMS[:1] if ref else ARMS):
                    for r in range(R):
                        D=(1-222/N if a!='isotropic' else 3.)+.001*i+.0001*r
                        rows.append({'date':d,'geometry':g,'N':N,'arm_id':arm_key(a,m),'replicate':r,
                            'D':D,'U':{'clean_ref':2.},'F':{'clean_ref':2.+D},'D_scores':{'clean_ref':D},
                            'optimization_limited':False,'optimization_limited_t25_rule':True,
                            'selected_status':'line_search_stopped','restarts':[{'status':'line_search_stopped'}]*4,
                            'restart_spread':.001,'selected_gradient_norm':1e-8})
    return rows,dates,pred

def test_complete_pilot_aggregation():
    rows,dates,pred=fixture();o=s.aggregate(rows,dates,pred)
    assert o['record_count']==16200 and len(o['cells'])==90 and len(o['power_cells'])==15
    assert len(o['PZ_readings']['PZ4']['interactions'])==4
    assert o['PZ_readings']['PZ2']['reading']=='MET'
    assert o['PZ_readings']['PZ5']['met']
    v=o['power_cells'][0]
    assert abs(v['between_date_sd']-np.std([.001*i for i in range(9)],ddof=1))<1e-12
    with pytest.raises(ValueError,match='incomplete'):s.aggregate(rows[:-1],dates,pred)
    with pytest.raises(ValueError,match='duplicate'):s.aggregate(rows+[rows[0]],dates,pred)
    rows[0]['D']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):s.aggregate(rows,dates,pred)

def test_ref_remedy():
    rows,dates,pred=fixture(50,True);o=s.aggregate(rows,dates,pred,50,True)
    assert o['record_count']==6750 and len(o['power_cells'])==15 and o['resamples']==50
    assert 'PZ_readings' not in o
