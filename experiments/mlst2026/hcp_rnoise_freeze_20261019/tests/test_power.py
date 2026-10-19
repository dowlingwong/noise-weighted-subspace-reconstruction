import copy
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_freeze_20261019 import power as p

def pilot(sd=.001,pred=1.,R=20):
    vals=np.arange(9)*sd/np.std(np.arange(9),ddof=1)
    dates=[str(i) for i in range(9)]
    return {'label':'DEV PILOT; not confirmatory','family_size':34,'dates':dates,'n_dev':9,'resamples':R,
            'power_cells':[{'cell':cell,'date_means':dict(zip(dates,vals)),
              'geometry':cell.split('/')[0],'N':[56,111,222,444,888][int(cell.split('/')[1])],
              'certificate_by_date':dict.fromkeys(dates,[222.,1000.]),'predicted_by_date':dict.fromkeys(dates,pred),'between_date_sd':float(np.std(vals,ddof=1)),
              'predicted_contrast':float(np.mean([pred]*9))} for cell in p.EXPECTED]}

def test_power_pass_minimum8_and_strict_zero_fail():
    assert p.compute(pilot())['n_min']==8
    r=p.compute(pilot(pred=0.));assert not r['pass'] and r['remedy_required'] and not r['stop']
    r=p.compute(pilot(pred=0.,R=50),True);assert not r['pass'] and r['stop'] and r['n_min'] is None
    assert len(r['trials'])==5 and all(len(n['cells'])==9 for n in r['trials'])
    assert len(r['AD8_selection'])==15
    assert all(not n['all_cells_pass'] for n in r['trials'])

@pytest.mark.parametrize('bad',['missing','sd','pred','R','n','family','duplicate','nonfinite','dates'])
def test_invalid_power_inputs_refuse(bad):
    s=pilot()
    if bad=='missing':s['power_cells'].pop()
    if bad=='sd':s['power_cells'][0]['between_date_sd']=1.
    if bad=='pred':s['power_cells'][0]['predicted_contrast']=2.
    if bad=='R':s['resamples']=50
    if bad=='n':s['n_dev']=8
    if bad=='family':s['family_size']=1
    if bad=='duplicate':s['power_cells'][1]['cell']=s['power_cells'][0]['cell']
    if bad=='nonfinite':s['power_cells'][0]['date_means']['0']=float('nan')
    if bad=='dates':s['power_cells'][0]['date_means'].pop('0')
    with pytest.raises(ValueError):p.compute(s)


def test_AD8_selects_from_certificates_only_and_excludes_centres():
    s=pilot()
    for c in s['power_cells']:
        if c['geometry']!='control' and c['N']==222:
            c['predicted_by_date']=dict.fromkeys(s['dates'],0.)
            c['predicted_contrast']=0.
    result=p.compute(s)
    assert result['pass'] and len(result['required_cells'])==9
    assert 'A/2/reference' not in result['required_cells']
    assert 'control/2/reference' in result['required_cells']
    s['power_cells'][0]['predicted_by_date']=dict.fromkeys(s['dates'],0.)
    s['power_cells'][0]['predicted_contrast']=0.
    assert not p.compute(s)['pass']
