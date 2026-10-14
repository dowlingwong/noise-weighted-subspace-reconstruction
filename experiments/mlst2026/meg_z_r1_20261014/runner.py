"""R2 commands: identity, census, measured preflight, gated controls, summary.

R3 preprocess/gates delegate to additive modules. Never launches R3 during R2.
"""
import argparse,json,os,platform,subprocess,time,signal,math
from pathlib import Path
import numpy as np
from experiments.mlst2026.meg_z_20261013.common import sha_file,atomic_json,sha_json
from experiments.mlst2026.meg_z_20261013.runner import kit_hashes,verify_kit as old_verify
from .protocol import PACKAGE,DEV,load
from . import controls
from .partition import census

def verify_kit():
    want=json.loads((PACKAGE/'KIT_SHA256.json').read_text());got=kit_hashes(PACKAGE)
    if got!=want:raise RuntimeError('STOP new kit identity mismatch')
    old=old_verify()
    if old['kit_identity']!=load()['base']['kit_identity']:raise RuntimeError('STOP old kit identity mismatch')
    proto=json.loads((PACKAGE.parent/'meg_z_20261013/protocol_dev.json').read_text())
    root=PACKAGE.parents[2]
    for rel,want_hash in proto['dependencies']['files'].items():
        if sha_file(root/rel)!=want_hash:raise RuntimeError(f'STOP dependency hash mismatch {rel}')
    return {'kit_identity':got['kit_identity'],'old_identity':old['kit_identity'],'files':len(got['files'])}

def census_check():
    ev=json.loads((PACKAGE/'fixtures/R0_EVIDENCE.json').read_text());got=json.loads(json.dumps(census(ev)));want=json.loads((PACKAGE/'fixtures/PARTITION_CENSUS.json').read_text())
    if got!=want:raise RuntimeError('STOP census mismatch')
    from .gates import choose_m_hi
    pools={s:r['10']['cal'] for s,r in got['subjects'].items()}
    return {'census':'exact','T':got['T_rule']['chosen_T'],'m_hi':choose_m_hi(pools),'calibration_pools':pools,'census_excludes_final_gaps':True}

def environment():
    import mne,scipy
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):
        if os.environ.get(key)!='1':raise RuntimeError(f'STOP {key} must be 1')
    if 'PYTHONPYCACHEPREFIX' in os.environ:raise RuntimeError('STOP unset PYTHONPYCACHEPREFIX')
    if (mne.__version__,np.__version__,scipy.__version__)!=('1.8.0','2.2.6','1.15.3'):raise RuntimeError('STOP reference environment differs')
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'mne':mne.__version__,
        'threads':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'bytecode_disabled':True}

def context(derived,gates):
    sources,hashes=controls.source_moments(derived,gates);geometry=json.loads((PACKAGE/'fixtures/DEV_GEOMETRY.json').read_text());shapes={}
    for s in DEV:
        maxL=max(g['length'] for R in geometry['subjects'][s]['runs'] for g in R['gaps']);shapes[s]=controls.ar_shape(sources[s],L=maxL)
    return shapes,geometry,hashes

def budget_projection(controls_cpu,d3_cpu,complete=True):
    """RD-11: controls only at 4 workers/6 h; R4 D3-S at 16 workers/8 h."""
    cfg=load();workers=cfg['controls']['workers'];cap=cfg['controls']['budget_hours']
    # Author-approved numeric values are in the pinned prose CONFIG field.
    assert '16 workers' in cfg['d3']['D3S_budget'] and '8 h' in cfg['d3']['D3S_budget']
    control_hours=controls_cpu/workers/3600;d3_hours=d3_cpu/16/3600
    control_ok=control_hours<=cap;d3_ok=complete and d3_hours<=8
    return {'projected_control_hours_4_workers':control_hours,'controls_workers':workers,
        'controls_cap_hours':cap,'controls_status':'PASS' if control_ok else 'STOP',
        'projected_D3S_hours_16_workers':d3_hours,'D3S_workers':16,'D3S_cap_hours':8,
        'D3S_status':'PASS' if d3_ok else 'STOP','D3S_budget_stage':'R4',
        'status':'PASS' if control_ok and d3_ok else 'STOP'}

def preflight(derived,gates,out,full_timing=False):
    # R2-B F8: returned preflight evidence is never overwritten; each preflight goes to a new path.
    if Path(out).exists():raise RuntimeError(f'STOP preflight output {out} exists; write to a new path')
    env=environment();identity=verify_kit();shapes,geometry,hashes=context(derived,gates)
    rows=[]
    # Bound all subject shapes with representative null/detection/drift/jitter cells.
    for si in range(4):
        for law,cell in ((1,0),(2,0),(4,7),(6,2),(7,9)):
            r=controls.replicate(law,cell,si,0,shapes,geometry);rows.append({'subject':r['subject'],'law':law,'cell':cell,'seconds':r['seconds']})
    from .signal import temporal_bases
    from .population import general_noise
    from experiments.mlst2026.meg_z_20261013.signal import SignalModel,orth
    B1,Cb,geom_check=temporal_bases();timings=[]
    # D3-S has its own R4 wall budget; controls do not consume it (RD-11).
    row_limit=8*16*3600/(4*50*7)
    class CostLimitExceeded(Exception):pass
    def cost_limit(signum,frame):raise CostLimitExceeded()
    old_handler=signal.signal(signal.SIGALRM,cost_limit)
    try:
        for si,s in enumerate(DEV):
            if row_limit<=0 and not full_timing:break
            gen,_=controls.rng(2,0,si,999);shape=shapes[s]
            t0=time.perf_counter();
            if not full_timing:signal.setitimer(signal.ITIMER_REAL,row_limit+.01)
            print(f'D3-S timing {s}; full_timing={full_timing}; cap deadline {row_limit:.3f} s',flush=True)
            try:
                rec=controls.generate(shape,geometry['subjects'][s]['runs'],2,gen);S=controls.reference(rec['ref'][0])
                A=orth(gen.normal(size=(24,2)));v=np.kron([1.,1.],[1.,.5,.25]);m0=SignalModel(A,B1,Cb,0.,v)
                v=v*(206.604/np.trace(np.linalg.solve(S,m0.second_moment())))
                model=SignalModel(A,B1,Cb,.4,v);general_noise(model,S,shape['population'],gen,62)
                timings.append({'subject':s,'seconds_per_reference_and_general_row':time.perf_counter()-t0,'complete':True})
            except CostLimitExceeded:
                timings.append({'subject':s,'seconds_per_reference_and_general_row':time.perf_counter()-t0,'complete':False,'censored_lower_bound':True})
                break
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    per=max(r['seconds'] for r in rows);d3=max((r['seconds_per_reference_and_general_row'] for r in timings),default=0.)
    controls_cpu=72800*per
    # Four fixed theta rows and three fitted geometries: conservative shared-estimate bound.
    d3_cpu=4*50*7*d3
    projected=(controls_cpu+d3_cpu)/4/3600;report={'environment':env,**identity,'source_hashes':hashes,'geometry':geom_check,'controls_timings':rows,
        'seconds_per_replicate_bound':per,'D3S_timings':timings,'D3S_reference_estimates_per_subject':50,'D3S_rows_per_estimate_bound':7,
        'projected_control_CPU_seconds':controls_cpu,'projected_D3S_CPU_seconds':d3_cpu,
        'combined_projected_hours_4_workers_descriptive':projected,
        'projection_kind':'conservative worst-row projection; a censored row provides a lower bound on that projection',
        **budget_projection(controls_cpu,d3_cpu,len(timings)==4 and all(r['complete'] for r in timings)),'fit_checks':{s:z['fit'] for s,z in shapes.items()},'controls_not_run':True}
    atomic_json(out,report)
    if report['status']=='STOP':raise SystemExit('STOP separate controls or R4 D3-S budget failed, or D3-S timing incomplete')
    return report

_CTX=None

def _batch(task):
    law,cell,si=task;shapes,geometry=_CTX;rows=[]
    for rep in range(200):
        r=controls.replicate(law,cell,si,rep,shapes,geometry)
        r['G3'].pop('slots');r['G2'].pop('generalised_eigenvalues',None);rows.append(r)
    return rows

def reviewed(commit,identity,review):
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,env=os.environ|{'GIT_OPTIONAL_LOCKS':'0'}).strip()
    if head!=commit or verify_kit()['kit_identity']!=identity:raise RuntimeError('STOP reviewed HEAD/identity mismatch')
    text=Path(review).read_text()
    import re
    if not re.search(r'(?im)^\s*\*{0,2}(?:Verdict|Status)\s*:\s*\*{0,2}ACCEPTED\b(?! WITH)',text) or commit not in text or identity not in text:raise RuntimeError('STOP R2-B ACCEPTED review with exact commit/identity required')

def campaign(derived,gates,out,commit,identity,review,preflight_path):
    reviewed(commit,identity,review)
    # R2-B F8: the controls-only preflight is named explicitly and must carry the reviewed identity.
    pf=json.loads(Path(preflight_path).read_text())
    if pf.get('controls_status')!='PASS' or pf.get('kit_identity')!=identity or pf.get('projected_control_hours_4_workers',float('inf'))>load()['controls']['budget_hours']:raise RuntimeError('STOP reviewed identity needs a passing controls-only preflight')
    env=environment();shapes,geometry,hashes=context(derived,gates)
    out=Path(out)
    if out.exists():raise RuntimeError('STOP campaign output exists; no rerun/tuning')
    out.mkdir(parents=True);t0=time.perf_counter()
    global _CTX
    _CTX=(shapes,geometry)
    import multiprocessing as mp
    tasks=[(l,c,s) for l in range(1,8) for c in range(13) for s in range(4)];rows=[]
    with mp.get_context('fork').Pool(4) as pool:
        for batch in pool.imap_unordered(_batch,tasks):rows.extend(batch)
    rows.sort(key=lambda r:(r['law'],r['cell'],r['subject'],r['replicate']))
    atomic_json(out/'records.json',rows);summary=controls.summarize(rows)
    summary.update(environment=env,reviewed_commit=commit,kit_identity=identity,source_hashes=hashes,
        generator_fit_checks={s:v['fit'] for s,v in shapes.items()},wall_seconds=time.perf_counter()-t0,seeds='SeedSequence([20261014,law_index,cell_index,subject_index,replicate]), PCG64DXSM; indices 1..7,0..12,0..3,0..199')
    atomic_json(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# MEG R1 controls\n\n'+summary['status']+'\n\n'+ '\n'.join(f"- {c['claim']} L{c['law']} C{c['cell']}: {c['count']}/800 — {'PASS' if c['pass'] else 'FAIL'}" for c in summary['claims'])+'\n')
    if summary['status']=='STOP':raise SystemExit('STOP failed endpoint; change nothing')
    return summary

def main(argv=None):
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='command',required=True)
    for name in ('verify-kit','census-check'):sp.add_parser(name)
    for name in ('preflight','controls'):
        p=sp.add_parser(name);p.add_argument('--derived',required=True);p.add_argument('--gates',required=True);p.add_argument('--out',required=True)
        if name=='preflight':p.add_argument('--full-timing',action='store_true',help='Measure all four complete D3-S rows even when the final projection will STOP')
        if name=='controls':
            p.add_argument('--reviewed-commit',required=True);p.add_argument('--reviewed-identity',required=True);p.add_argument('--review',required=True)
            p.add_argument('--preflight',required=True,help='controls-only PREFLIGHT.json produced at the reviewed identity')
    p=sp.add_parser('summarize-controls');p.add_argument('--out',required=True)
    a=ap.parse_args(argv)
    if a.command=='verify-kit':r=verify_kit()
    elif a.command=='census-check':r=census_check()
    elif a.command=='preflight':r=preflight(a.derived,a.gates,a.out,a.full_timing)
    elif a.command=='controls':r=campaign(a.derived,a.gates,a.out,a.reviewed_commit,a.reviewed_identity,a.review,a.preflight)
    else:
        r=controls.summarize(json.loads((Path(a.out)/'records.json').read_text()));atomic_json(Path(a.out)/'claim_check.json',r)
    print(json.dumps(r,indent=1))
if __name__=='__main__':main()
