import json,os,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest
from experiments.mlst2026.upgrade_20260922 import protocol as p,cresst as c
from experiments.mlst2026.upgrade_20260922.__main__ import verify
MS=Path(os.environ.get('MLST_MS_ROOT',str(p.ROOT.parent/'arXiv-v1-polished-extended-preprint')))

def cli(*args,ok=True):
    e={**os.environ,'PYTHONPATH':str(p.ROOT/'src'),'MLST_MS_ROOT':str(MS)}
    r=subprocess.run([sys.executable,'-m','experiments.mlst2026.upgrade_20260922',*map(str,args)],cwd=p.ROOT,env=e,capture_output=True,text=True)
    assert (r.returncode==0)==ok,r.stdout+r.stderr
    return r

def fixture(root,n=640,d=32):
    root.mkdir();rng=np.random.default_rng(21);x=rng.normal(0,.03,size=(n,d));h=np.exp(-((np.arange(d)-13)/3)**2)
    noise=np.ones(n,dtype=bool);noise[::5]=False;x[~noise]+=h
    x.astype('<f4').tofile(root/'X_train.npy');pd.DataFrame({'run':1,'channel':2,'hours':np.arange(n)/10,'noise':noise,'clean':True}).to_csv(root/'features_train.csv',index=False)

def test_loader_boolean_width_duplicate_and_time_roles(tmp_path):
    raw=tmp_path/'raw';fixture(raw);x,f,_=c.load(raw,width=32)
    shuffled=f.sample(frac=1,random_state=4);allocation=c.allocate(shuffled)['1/2']
    for ids in allocation['roles'].values():assert ids==sorted(ids)
    assert len(set(sum(allocation['roles'].values(),[])))==len(f)
    csv=raw/'features_train.csv';df=pd.read_csv(csv);df['clean']=df['clean'].astype(object);df.loc[0,'clean']='perhaps';df.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='boolean'):c.load(raw,width=32)
    df.loc[0,'clean']='False';df.to_csv(csv,index=False);_,ff,_=c.load(raw,width=32);assert not ff.iloc[0]['clean']
    with pytest.raises(ValueError,match='byte'):c.load(raw,width=16)
    df.loc[1,'hours']=df.loc[0,'hours'];df.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='duplicate'):c.load(raw,width=32)

def test_c1_production_path_and_export_guards(tmp_path):
    cfg=p.PACKAGE/'configs/c1_fixture_protocol.yaml';manifest=tmp_path/'manifest.json';cli('inventory','--config',cfg,'--out',manifest)
    frozen=tmp_path/'pilot.json';run=tmp_path/'pilot_run'
    cli('freeze','--config',cfg,'--manifest',manifest,'--stage','pilot','--ms-root',MS,'--out',frozen)
    cli('run','--freeze',frozen,'--stage','confirm','--run-root',run,'--ms-root',MS,ok=False)
    cli('run','--freeze',frozen,'--stage','pilot','--run-root',run,'--ms-root',MS)
    cli('verify','--run-root',run);cli('reduce','--run-root',run,'--out',tmp_path/'pilot_reduction.json')
    confirm=tmp_path/'confirm.json';crun=tmp_path/'confirm_run'
    cli('freeze','--config',cfg,'--manifest',manifest,'--stage','confirm','--pilot-root',run,'--ms-root',MS,'--out',confirm)
    cli('run','--freeze',confirm,'--stage','confirm','--run-root',crun,'--ms-root',MS)
    cli('reduce','--run-root',crun,'--out',tmp_path/'confirm_reduction.json');cli('export','--run-root',crun,'--out',tmp_path/'badexport',ok=False)
    # Exercise report rendering separately, without promoting development evidence.
    from experiments.mlst2026.upgrade_20260922.reporting import render
    ff,rr=verify(crun);plots=tmp_path/'development_only_c1_plots';plots.mkdir()
    render(plots,ff,rr,p.read(tmp_path/'confirm_reduction.json')['reduction'])
    assert (plots/'summary.pdf').read_bytes().startswith(b'%PDF')
    assert len(p.read(plots/'source_table.json'))==75 and (plots/'source_table.csv').stat().st_size>100
    cli('run','--freeze',confirm,'--stage','confirm','--run-root',crun,'--ms-root',MS,ok=False)
    ledger=crun/'attempts.jsonl';original=ledger.read_text();ledger.write_text(original+original.splitlines()[0]+'\n')
    with pytest.raises(ValueError,match='duplicate'):verify(crun)
    ledger.write_text('\n'.join(original.splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='incomplete'):verify(crun)
    ledger.write_text(original);next(crun.glob('*.npz')).write_bytes(b'broken')
    with pytest.raises(ValueError,match='hash'):verify(crun)

def test_cresst_fixture_preparation_and_role_mutation(tmp_path):
    raw=tmp_path/'raw';fixture(raw);cfg=p.PACKAGE/'configs/cresst_fixture_protocol.yaml';cfig=p.config(cfg)
    audit={'1/2':{'status':'exposed','evidence':'synthetic fixture','sample_spacing_seconds':.00064,'acquisition_terms':'generated fixture'}}
    m=c.qualify(raw,cfig,audit);m['config_sha256']=p.identity(cfig)
    report,arrays=c.prepare(raw,cfig,m)
    spec=p.cells(cfig,'confirm',m)[0];r,z=c.run_cell(spec,cfig,m,report,arrays,raw)
    assert r['paired_bank_sha256'] and 'oracle_joint_template' in r['metrics']
    m['channels']['1/2']['noise_roles']['F'][0]=m['channels']['1/2']['noise_roles']['E'][0]
    with pytest.raises(ValueError,match='manifest'):c.prepare(raw,cfig,m)


def test_cresst_cli_and_mutation(tmp_path):
    raw=tmp_path/'raw';fixture(raw);cfg=p.PACKAGE/'configs/cresst_fixture_protocol.yaml'
    audit=tmp_path/'exposure.json';p.write(audit,{'1/2':{'status':'exposed','evidence':'synthetic','sample_spacing_seconds':.00064,'acquisition_terms':'fixture'}})
    m=tmp_path/'manifest.json';cli('validate','--config',cfg,'--data-root',raw,'--exposure',audit,'--out',m)
    cli('prepare','--config',cfg,'--manifest',m,'--data-root',raw,'--out',tmp_path/'prepared')
    fr=tmp_path/'frozen/freeze.json';cli('freeze','--config',cfg,'--manifest',m,'--data-root',raw,'--ms-root',MS,'--out',fr)
    run=tmp_path/'run';cli('run','--freeze',fr,'--stage','confirm','--data-root',raw,'--ms-root',MS,'--run-root',run)
    cli('verify','--run-root',run)
    from experiments.mlst2026.upgrade_20260922.validation import check
    f,records=verify(run);bad=json.loads(json.dumps(records[0]));bad['result']['primary_contrast']['mean']=-999
    with pytest.raises(ValueError,match='headline'):check(bad,run/(bad['cell']['id']+'.npz'),f)
    bad=json.loads(json.dumps(records[0]));bad['result']['metrics']={}
    with pytest.raises(ValueError,match='headline'):check(bad,run/(bad['cell']['id']+'.npz'),f)
    cli('reduce','--run-root',run,'--out',tmp_path/'reduction.json');cli('export','--run-root',run,'--out',tmp_path/'export',ok=False)
    from experiments.mlst2026.upgrade_20260922.reporting import render
    plots=tmp_path/'development_only_cresst_plots';plots.mkdir()
    render(plots,p.read(fr),records,p.read(tmp_path/'reduction.json')['reduction'])
    assert (plots/'summary.pdf').read_bytes().startswith(b'%PDF')
    assert len(p.read(plots/'source_table.json'))==8*len(records)
    with (raw/'X_train.npy').open('r+b') as f:f.write(b'xxxx')
    cli('run','--freeze',fr,'--stage','confirm','--data-root',raw,'--ms-root',MS,'--run-root',tmp_path/'mutated',ok=False)

def test_unknown_unused_and_freeze_mutation(tmp_path):
    cfg=p.PACKAGE/'configs/c1_fixture_protocol.yaml';bad=tmp_path/'config.json';obj=p.read(cfg);obj['typo']=1;p.write(bad,obj)
    cli('inventory','--config',bad,'--out',tmp_path/'inventory',ok=False)
    cli('inventory','--config',cfg,'--manifest',bad,'--out',tmp_path/'inventory',ok=False)
    m=tmp_path/'m';cli('inventory','--config',cfg,'--out',m);fr=tmp_path/'freeze';cli('freeze','--config',cfg,'--manifest',m,'--stage','pilot','--ms-root',MS,'--out',fr)
    obj=p.read(fr);obj['sources']['files']['EXP']['src/noise_geometry/cresst/loader.py']='0'*64
    obj['freeze_id']=p.identity({k:v for k,v in obj.items() if k!='freeze_id'});fr.write_text(json.dumps(obj))
    cli('run','--freeze',fr,'--stage','pilot','--ms-root',MS,'--run-root',tmp_path/'run',ok=False)

def test_process_termination_charged_and_orphans_rejected(tmp_path):
    import signal,time
    cfg=p.PACKAGE/'configs/c1_fixture_protocol.yaml';m=tmp_path/'m';cli('inventory','--config',cfg,'--out',m);fr=tmp_path/'freeze'
    cli('freeze','--config',cfg,'--manifest',m,'--stage','pilot','--ms-root',MS,'--out',fr)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        run=tmp_path/str(sig);env={**os.environ,'PYTHONPATH':str(p.ROOT/'src')}
        proc=subprocess.Popen([sys.executable,'-m','experiments.mlst2026.upgrade_20260922','run','--freeze',str(fr),'--stage','pilot','--ms-root',str(MS),'--run-root',str(run)],cwd=p.ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and not (run/'attempts.jsonl').exists():time.sleep(.005)
        assert (run/'attempts.jsonl').exists();proc.send_signal(sig);proc.communicate(timeout=10)
        events=[json.loads(l) for l in (run/'attempts.jsonl').read_text().splitlines()];assert events[0]['reserved_fits']==27
        with pytest.raises(ValueError):verify(run)
        cli('run','--freeze',fr,'--stage','pilot','--ms-root',MS,'--run-root',run,ok=False)
