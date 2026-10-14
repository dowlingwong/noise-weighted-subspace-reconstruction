"""R3 CLI (never executed on real data in R2); DESIGN 1 and 2."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from experiments.mlst2026.meg_z_20261013 import preprocess as old,estimators
from experiments.mlst2026.meg_z_20261013.common import vec,atomic_json
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol
from experiments.mlst2026.meg_z_20261013.data_manifest import run_stem
from .protocol import guard,PACKAGE
from .windows import gap_geometry,window_plan
from .signal import EVOKED_T

def assert_channels(channels,expected,drop=()):
    if channels!=[x for x in expected if x not in drop]:raise RuntimeError('STOP channel selection differs from sealed D1 list')

def assemble(runs,cfg,expected_channels,subject,drop=()):
    names=runs[1]['ch_names'];bads=sorted(set().union(*(set(r['bads']) for r in runs.values())))
    good=[i for i,n in enumerate(names) if n not in bads];fs=runs[1]['sfreq']
    if any(r['ch_names']!=names or r['sfreq']!=fs for r in runs.values()):raise ValueError('run channel/sample clock mismatch')
    ons={r:v['events']['stim_onsets_trigger']+cfg['trigger_delay_s'] for r,v in runs.items()}
    trials=[]
    for r in (1,2):
        v=runs[r];X,_=old.epochs(v['data'][good],fs,ons[r][v['events']['is_face']],EVOKED_T,cfg['baseline_s'],v['bad_mask']);trials.append(X)
    evoked_all=np.concatenate(trials).mean(0);selection=old.select_channels(evoked_all,24)
    selected=[names[good[i]] for i in selection['selected']];assert_channels(selected,expected_channels)
    chosen=[good[i] for i in selection['selected'] if names[good[i]] not in drop];channels=[names[i] for i in chosen]
    assert_channels(channels,expected_channels,drop)
    evoked=evoked_all[[i for i in selection['selected'] if names[good[i]] not in drop]]
    windows={ro:[] for ro in ('ref','train','cal','eval')};mm={ro:{k:[] for k in ('run','gap','start','position','rposition','block')} for ro in windows};geometry=[]
    for r,v in sorted(runs.items()):
        # fs=1100/12 is constructed exactly, rather than from rounded binary sfreq.
        geom=gap_geometry(v['data'].shape[1],ons[r],v['events']['resp_onsets'],bad_mask=v['bad_mask']);plans=window_plan(geom,r)
        geometry.append({'run':r,'G':len(geom)})
        for ro,plan in plans.items():
            windows[ro].append(old.cut(v['data'][chosen],[w['start'] for w in plan],10))
            for k in mm[ro]:mm[ro][k].extend(w[k] for w in plan)
    windows={ro:np.concatenate(x) for ro,x in windows.items()};mm={ro:{k:np.asarray(v,dtype=int) for k,v in meta.items()} for ro,meta in mm.items()}
    ref=estimators.moment_stats(vec(windows['ref']));from .controls import effective_reference_count
    ref.n_eff=effective_reference_count(mm['ref'])
    long={'train':'training','cal':'calibration','eval':'evaluation'}
    payload=old.subject_payload({long[ro]:windows[ro] for ro in long},{long[ro]:mm[ro] for ro in long},ref,evoked,{})
    for ro in long:payload['windows'][long[ro]]['block']=mm[ro]['block']
    attrs={'subject':subject,'C':len(chosen),'T':10,'evoked_T':20,'sfreq':fs,'channels':channels,'bads_dropped':bads,'drop_selected':list(drop),
        'proc_history_empty':all(r.get('proc_history_empty',False) for r in runs.values()),'geometry':geometry}
    return payload,attrs,windows,mm

def process_subject(raw,subject,out,expected_channels,drop=(),runs_loader=None):
    guard([subject]);proto=load_protocol();cfg=old.preprocess_config(proto);runs={};loader=runs_loader or old.load_run
    for r in range(1,7):
        stem=Path(raw)/run_stem(subject,r);v=loader(*(Path(str(stem)+x) for x in ('_meg.fif','_events.tsv','_channels.tsv')),cfg)
        if not v['stim_check'].get('ok'):raise old.TriggerCheckError('trigger disagreement; no timing repair')
        runs[r]=v
    payload,attrs,windows,meta=assemble(runs,cfg,expected_channels,subject,drop)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);h=old.write_h5(out/f'{subject}.h5',payload,attrs)
    summary={**h,**attrs,'pool_counts':{ro:len(X) for ro,X in windows.items()},'reference_metric':'sample_full_unshrunk'}
    atomic_json(out/f'{subject}_summary.json',summary);return summary

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bids-root',required=True);ap.add_argument('--subject',required=True);ap.add_argument('--out',required=True);ap.add_argument('--protocol');a=ap.parse_args(argv)
    guard([a.subject],a.protocol)
    expected=json.loads((PACKAGE/'fixtures/DEV_GEOMETRY.json').read_text())['subjects'][a.subject]['sealed_channels']
    return process_subject(a.bids_root,a.subject,a.out,expected)
if __name__=='__main__':main()
