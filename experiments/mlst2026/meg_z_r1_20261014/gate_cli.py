"""R3 subject gates and once-only G1 retest; no execution during R2."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.mlst2026.meg_z_20261013.preprocess import load_subject
from experiments.mlst2026.meg_z_20261013.common import atomic_json,sha_file
from .protocol import guard,load,PACKAGE,DEV
from .gates import g1_spd,noisiest_channel,role_g2,coordinates,g3,e1,budget_eval,g4
from .preprocess import process_subject

def subject(path):
    import h5py
    sd=load_subject(path,cache=False);S=sd.ref.S
    G1=g1_spd(S,sd.C,sd.T,load()['gates']['G1']['unchanged_limits'],sd.attrs['proc_history_empty'])
    result={'G1':G1,'channels':sd.channels,'C':sd.C,'T':sd.T}
    if not G1['pass']:result.update(excluded=True,derived_file_sha256=sha_file(path));return result
    result['G2']=role_g2(sd.records('training'),sd.records('evaluation'),S)
    result['G2_cal']=role_g2(sd.records('training'),sd.records('calibration'),S)
    from experiments.mlst2026.meg_z_20261013.estimators import ledoit_wolf_from_stats
    from experiments.mlst2026.meg_z_20261013.gates import g2_stationarity,g5_mardia,lag_one_dependence
    lw,lwi=ledoit_wolf_from_stats(sd.ref)
    retired=g2_stationarity(lw,sd.records('evaluation'),[.8,1.25],2.)
    result['G2_retired_LW']={'descriptive':True,'mean_whitened_eigenvalue':retired['mean_whitened_eigenvalue'],
        'MP_edge_ratios':[retired['mp_lower_over_min'],retired['max_over_mp_upper']],'shrinkage':lwi['shrinkage']}
    allX=np.concatenate([sd.records(ro) for ro in ('training','calibration','evaluation')])
    allrun=np.concatenate([sd.window_meta[ro]['run'] for ro in ('training','calibration','evaluation')]);blocked={}
    for run in sorted(set(allrun)):
        sel=allrun==run;r=role_g2(allX[~sel],allX[sel],S,details=False)
        r['G2S_pass']=all(.8<=r[k]<=1.25 for k in ('m_eval/m_train','g_bottom50','g_middle40','g_top10','physical_trace_ratio'))
        blocked[str(int(run))]=r
    result['run_blocked_G2S']=blocked
    result['G5']=g5_mardia(sd.records('training'));result['dependence']=lag_one_dependence(sd,np.linalg.inv(S))
    _,_,L=coordinates(S);roles={}
    with h5py.File(path,'r') as h:
        for role,long in [('train','training'),('cal','calibration'),('eval','evaluation')]:
            meta={k:h['windows'][long][k][()] for k in ('run','block','position','rposition')};roles[role]=(sd.records(long)@L,meta)
    result['G3']=g3(roles);result['G4']=g4(S,sd.C,sd.T);result['budget']=budget_eval(len(sd.records('evaluation')))
    result['excluded']=any(not result[g]['pass'] for g in ('G1','G2','G3')) or result['budget']['excluded']
    result['derived_file_sha256']=sha_file(path);return result

def with_retest(derived,sub,raw,expected):
    path=Path(derived)/f'{sub}.h5';first=subject(path)
    if first['G1']['pass']:first['G1_retest']=None;return first
    sd=load_subject(path,cache=False)
    if sd.attrs.get('drop_selected'):first['G1_retest']={'status':'already performed; no second retest'};return first
    if raw is None:raise RuntimeError('STOP raw path needed for once-only G1 retest')
    # R2-B F7: the retest rewrites <sub>.h5 and <sub>_summary.json; record the first files before they are replaced.
    summary=Path(derived)/f'{sub}_summary.json'
    first_files={'derived_file_sha256':sha_file(path),'summary_sha256':sha_file(summary) if summary.is_file() else None,
                 'summary':json.loads(summary.read_text()) if summary.is_file() else None}
    drop=noisiest_channel(sd.ref.S,sd.C,sd.T,sd.channels)['channel'];process_subject(raw,sub,derived,expected,[drop]);second=subject(path)
    second['G1_retest']={'first':first,'first_files':first_files,'dropped':drop,'count':1};return second

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--derived',required=True);ap.add_argument('--subjects',required=True);ap.add_argument('--bids-root');ap.add_argument('--out',required=True);ap.add_argument('--protocol');a=ap.parse_args(argv)
    subs=a.subjects.split(',');guard(subs,a.protocol);geometry=json.loads((PACKAGE/'fixtures/DEV_GEOMETRY.json').read_text())
    results={s:with_retest(a.derived,s,a.bids_root,geometry['subjects'][s]['sealed_channels']) for s in subs}
    extra=[s for s,r in results.items() if r.get('budget',{}).get('excluded')];atomic_json(a.out,{'subjects':results,'E1':e1(results,extra)})
if __name__=='__main__':main()
