"""F4: mechanical accepted R1 power rule; fixed full family, one R50 remedy."""
import hashlib
import json
import numpy as np
from scipy.stats import chi2,t
from experiments.mlst2026.hcp_rnoise_20261019.gates import n_min
from experiments.mlst2026.hcp_rnoise_dev_20261019.pipeline import new_file
from .prediction import PUB
from .summary import FAMILY

EXPECTED=[f'{g}/{i}/reference' for g in ('control','A','B') for i in range(5)]


def compute(summary,after_remedy=False):
    if summary.get('label')!='DEV PILOT; not confirmatory' or summary['family_size']!=FAMILY:
        raise ValueError('STOP pilot label/frozen family mismatch')
    dates=summary['dates'];cells=summary['power_cells']
    if summary['n_dev']!=9 or len(dates)!=9 or len(set(dates))!=9:
        raise ValueError('STOP exact nine-date pilot required')
    if summary['resamples']!=(50 if after_remedy else 20):raise ValueError('STOP R20 then one R50 remedy only')
    if [c['cell'] for c in cells]!=EXPECTED:raise ValueError('STOP complete ordered fifteen reference cells required')
    sd=[];pred=[]
    for c in cells:
        if set(c['date_means'])!=set(dates) or set(c['predicted_by_date'])!=set(dates):
            raise ValueError('STOP incomplete date means/predictions')
        vals=np.asarray([c['date_means'][d] for d in dates],float)
        pv=np.asarray([c['predicted_by_date'][d] for d in dates],float)
        if not np.isfinite(vals).all() or not np.isfinite(pv).all():raise ValueError('STOP nonfinite power input')
        s=float(np.std(vals,ddof=1));p=float(np.mean(pv))
        if s!=c['between_date_sd'] or p!=c['predicted_contrast']:
            raise ValueError('STOP pilot SD/prediction aggregate differs')
        sd.append(s);pred.append(p)
    result=n_min(sd,9,pred,FAMILY,after_remedy=after_remedy,cell_ids=EXPECTED,expected_cells=EXPECTED)
    upper=np.asarray(sd)*np.sqrt(8/chi2.ppf(.2,8));target=.5*np.abs(pred)
    trials=[]
    for n in range(8,13):
        critical=float(t.ppf(1-.05/(2*FAMILY),n-1));half=critical*upper/np.sqrt(n)
        trials.append({'n':n,'df':n-1,'t_critical':critical,'all_cells_pass':bool(np.all(half<target)),
                       'cells':[{'cell':key,'pilot_sd':sd[i],'pilot_sd_upper80':float(upper[i]),
                                 'prediction':pred[i],'half_abs_prediction':float(target[i]),
                                 'simultaneous_half_width':float(half[i]),'pass':bool(half[i]<target[i])}
                                for i,key in enumerate(EXPECTED)]})
    return {'label':'DEV PILOT POWER; not confirmation evidence','rule':'strict t_(1-.05/(2*family),n-1)*SD_upper80/sqrt(n) < .5*abs(mean prediction a_best+B/N), every reference cell; n=8..12',
            'family_size':FAMILY,'alpha':.05,'n_dev':9,'SD_upper_confidence':.8,'SD_upper_df':8,
            'SD_upper_multiplier':float(np.sqrt(8/chi2.ppf(.2,8))),
            'resamples':summary['resamples'],'required_cells':EXPECTED,'trials':trials,**result}


def publish(final=False):
    p20=PUB/'PILOT_SUMMARY.json';s20=json.loads(p20.read_text());initial=compute(s20)
    initial['pilot_sha256']=hashlib.sha256(p20.read_bytes()).hexdigest()
    if not final:
        new_file(PUB/'POWER_R20.json',initial)
        if initial['pass']:
            new_file(PUB/'POWER.json',{'status':'PASS','n_min':initial['n_min'],'initial':initial,'remedy_applied':False,'confirmation_access':False})
        return initial
    if initial['pass']:raise ValueError('STOP remedy cannot be applied after an initial pass')
    p50=PUB/'PILOT_R50_REF_SUMMARY.json';s50=json.loads(p50.read_text());remedy=compute(s50,True)
    if s20['dates']!=s50['dates'] or s20['DEV_PREDICTIONS_sha256']!=s50['DEV_PREDICTIONS_sha256']:
        raise ValueError('STOP remedy changed dates/predictions')
    remedy['pilot_sha256']=hashlib.sha256(p50.read_bytes()).hexdigest()
    result={'status':'PASS' if remedy['pass'] else 'STOP','n_min':remedy['n_min'],
            'initial':initial,'remedy':remedy,'remedy_applied':True,
            'remedy_kind':'R50; reference pilot extended from R20 with new replicates20..49; unchanged family',
            'protocol_frozen':bool(remedy['pass']),'confirmation_access':False,
            'next_action':'F5 freeze bundle' if remedy['pass'] else 'STOP F4; report; no protocol freeze or registration'}
    new_file(PUB/'POWER.json',result)
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(publish(final='--final' in sys.argv),indent=2))
