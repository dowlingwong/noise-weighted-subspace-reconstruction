"""Complete-ledger date means, DEV PILOT readings and power inputs; no tuning."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.mlst2026.hcp_rnoise_20261019 import readings
from experiments.mlst2026.meg_z_20261013.runner import interval
from experiments.mlst2026.hcp_rnoise_dev_20261019.pipeline import new_file
from .prediction import PUB,OUT,evidence
from .pilot import ARMS,arm_key

FAMILY=34
GEOMS=('control','A','B')


def aggregate(records,dates,pred,R=20,reference_only=False):
    if len(dates)!=9 or len(set(dates))!=9 or R not in (20,50):
        raise ValueError('exactly nine unique dev dates and R20/R50 required')
    arms=['reference'] if reference_only else [arm_key(*a) for a in ARMS]
    expected={(d,g,N,a,r) for d in dates for g in GEOMS
              for N in pred['units'][d]['geometries'][g]['grid'] for a in arms for r in range(R)}
    grouped=defaultdict(list);seen=set();diagnostics=defaultdict(Counter)
    for row in records:
        key=(row['date'],row['geometry'],row['N'],row['arm_id'],row['replicate'])
        if key not in expected or key in seen:raise ValueError('unexpected/duplicate pilot record')
        vals=[row['D'],row['U']['clean_ref'],row['F']['clean_ref']]
        if not np.isfinite(vals).all():raise ValueError('nonfinite pilot score')
        seen.add(key);grouped[key[:-1]].append(row)
        diag=diagnostics[row['arm_id']]
        diag['fit_pairs']+=1
        for flag in ('optimization_limited','optimization_limited_t25_rule'):
            diag[flag]+=int(row[flag])
        diag['selected_'+row['selected_status']]+=1
        for restart in row['restarts']:diag['restart_'+restart['status']]+=1
    if seen!=expected:raise ValueError('incomplete pilot ledger')
    per={}
    for key,rows in grouped.items():
        per[key]={'D':float(np.mean([r['D'] for r in rows])),
                  'within_date_SD':float(np.std([r['D'] for r in rows],ddof=1)),
                  'U_clean_ref':float(np.mean([r['U']['clean_ref'] for r in rows])),
                  'F_clean_ref':float(np.mean([r['F']['clean_ref'] for r in rows])),
                  'scores':{k:float(np.mean([r['D_scores'][k] for r in rows])) for k in rows[0]['D_scores']},
                  'optimization_limited':sum(r['optimization_limited'] for r in rows),
                  'restart_spread_max':max(r['restart_spread'] for r in rows),
                  'selected_gradient_norm_max':max(r['selected_gradient_norm'] for r in rows)}
    table=[];power=[]
    for g in GEOMS:
        grid=pred['units'][dates[0]]['geometries'][g]['grid']
        if any(pred['units'][d]['geometries'][g]['grid']!=grid for d in dates):raise ValueError('pilot fixed grids differ')
        for i,N in enumerate(grid):
            for a in arms:
                cell=f'{g}/{i}/{a}'
                details={d:per[(d,g,N,a)] for d in dates}
                ledger=[{'date':d,'replicate':r['replicate'],'D':r['D'],'N':N,'arm':a,'theta':g}
                        for d in dates for r in grouped[(d,g,N,a)]]
                ci=readings.date_interval(ledger,FAMILY,expected_dates=dates,resamples=R)
                entry={'cell':cell,'geometry':g,'N':N,'arm':a,'per_date':details,**ci};table.append(entry)
                if a=='reference':
                    pc={}
                    for d in dates:
                        rr=pred['units'][d]['geometries'][g]['row']
                        pc[d]=rr['a_best']+rr['B']/N
                    power.append({'cell':cell,'geometry':g,'N':N,'date_means':ci['date_means'],
                                  'between_date_sd':ci['sd'],'predicted_by_date':pc,
                                  'predicted_contrast':float(np.mean(list(pc.values())))})
    out={'label':'DEV PILOT; not confirmatory','dates':dates,'n_dev':len(dates),'resamples':R,
         'reference_only':reference_only,'family_size':FAMILY,'family_alpha':.05,
         'family':'15 reference cells + 4 PZ4 paired cell-average interactions + 15 isotropic cells',
         'record_count':len(seen),'cells':table,'power_cells':power,'diagnostics':dict(diagnostics),
         'confirmation_access':False,'scientific_changes_from_pilot':None}
    if reference_only:return out
    def find(g,i,a):return next(t for t in table if t['cell']==f'{g}/{i}/{a}')
    pz2={}
    for g in ('A','B'):
        below=[];above=[]
        for i,N in enumerate(pred['units'][dates[0]]['geometries'][g]['grid']):
            intervals=[pred['units'][d]['geometries'][g]['row']['N_star_interval'] for d in dates]
            if all(v is not None and N<v[0] for v in intervals):below.append(find(g,i,'reference'))
            if all(v is not None and N>v[1] for v in intervals):above.append(find(g,i,'reference'))
        applicable=bool(below and above and all(pred['units'][d]['geometries'][g]['row']['PZ2']['in_range'] for d in dates))
        pz2[g]={'applicable':applicable,'below':[r['cell'] for r in below],'above':[r['cell'] for r in above],
                'met':bool(applicable and all(r['decision']=='negative' for r in below) and all(r['decision']=='positive' for r in above)),
                'reason':None if applicable else 'certificate crossing out of fixed grid in one or more dates; no PZ2 claim'}
    def aggregate_ci(values,cell):
        base=interval(list(values.values()),FAMILY,.05)
        return {**base,'cell':cell,'available':True,'date_means':values,'expected_dates':dates,
                'resamples':R,'resamples_per_date':{d:R for d in dates}}
    interactions={}
    for a in ('lw','kron'):
        for m in (256,512):
            vals={d:float(np.mean([per[(d,g,N,arm_key(a,m))]['D']-per[(d,g,N,'reference')]['D']
                                  for g in ('A','B') for N in pred['units'][d]['geometries'][g]['grid']])) for d in dates}
            interactions[arm_key(a,m)]=aggregate_ci(vals,arm_key(a,m))
    control=[find('control',i,'reference') for i in range(5)]
    iso=[r for r in table if r['arm']=='isotropic']
    out['PZ_readings']={
        'label':'DEV PILOT; not confirmatory',
        'PZ1':{'met':all(pred['units'][d]['PZ1']['met'] for d in dates),'per_date':{d:pred['units'][d]['PZ1'] for d in dates}},
        'PZ2':{'geometries':pz2,'reading':'NOT APPLICABLE' if not any(v['applicable'] for v in pz2.values()) else 'MET' if all(v['met'] for v in pz2.values()) else 'NOT MET'},
        'PZ3':readings.pz3(control,[r['cell'] for r in control]),
        'PZ4':{**readings.pz4({a:interactions[arm_key(a,256)] for a in ('lw','kron')},
                             {a:interactions[arm_key(a,512)] for a in ('lw','kron')}),'interactions':interactions},
        'PZ5':readings.pz5(iso,[r['cell'] for r in iso])}
    poprows=[r for d in dates for r in pred['units'][d]['fixed']]+[pred['units'][d]['adaptive'][g]['row'] for d in dates for g in ('A','B')]
    out['population_diagnostics']={'rows':len(poprows),'n_starts':sorted(set(r['n_starts'] for r in poprows)),
        'H_min_eig_min':min(r['H_min_eig'] for r in poprows),'H_max_eig_max':max(r['H_max_eig'] for r in poprows),
        'gradient_norm_max':max(r['gradient_norm_at_optimum'] for r in poprows),
        'all_rows_restart_agreement':all(r['restart_agreement'] for r in poprows),
        'all_rows_all_starts_best':all(r['all_starts_reach_best'] for r in poprows)}
    return out


def summarize(remedy=False):
    q,s=evidence();dates=q['E1']['included'];pred=json.loads((PUB/'DEV_PREDICTIONS.json').read_text())
    if not (OUT/'PILOT_COMPLETED.json').is_file():raise RuntimeError('STOP incomplete pilot campaign')
    if remedy and not (OUT/'REMEDY_COMPLETED.json').is_file():raise RuntimeError('STOP incomplete remedy campaign')
    rows=[];manifest={}
    original_manifest=json.loads((PUB/'PILOT_RECORD_SHA256.json').read_text()) if remedy else None
    for d in dates:
        for g in GEOMS:
            for r in range(50 if remedy else 20):
                path=OUT/('remedy_records' if r>=20 else 'pilot_records')/d/g/f'r{r:02d}.json'
                b=path.read_bytes();record=json.loads(b)
                if (record['date'],record['geometry'],record['replicate'])!=(d,g,r):raise ValueError('group identity mismatch')
                manifest[str(path)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
                if remedy and r<20 and original_manifest.get(str(path))!=manifest[str(path)]:
                    raise RuntimeError('STOP R20 bytes changed before remedy summary')
                rows.extend(v for v in record['records'] if not remedy or v['arm_id']=='reference')
    result=aggregate(rows,dates,pred,50 if remedy else 20,remedy)
    result['DEV_PREDICTIONS_sha256']=hashlib.sha256((PUB/'DEV_PREDICTIONS.json').read_bytes()).hexdigest()
    result['external_records']={'path':str(OUT),'file_count':len(manifest),'total_bytes':sum(v['bytes'] for v in manifest.values()),
                                'manifest_file':'REMEDY_RECORD_SHA256.json' if remedy else 'PILOT_RECORD_SHA256.json'}
    new_file(PUB/('REMEDY_RECORD_SHA256.json' if remedy else 'PILOT_RECORD_SHA256.json'),manifest)
    new_file(PUB/('PILOT_R50_REF_SUMMARY.json' if remedy else 'PILOT_SUMMARY.json'),result)
    return result


if __name__=='__main__':
    import sys
    summarize(remedy='--remedy' in sys.argv)
