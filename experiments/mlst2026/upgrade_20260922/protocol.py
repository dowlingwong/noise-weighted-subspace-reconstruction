"""Closed, versioned protocols. JSON is used as the YAML-compatible serialization."""
import json,hashlib,subprocess,sys,platform,os
from pathlib import Path
import numpy as np, scipy
from importlib.metadata import version,distributions
ROOT=Path(__file__).resolve().parents[3]
PACKAGE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def identity(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def config(path):
    c=read(path)
    expected={'version','study','development_only','budget','scientific_settings'}
    if set(c)!=expected or c['version']!='upgrade/1' or type(c['development_only']) is not bool:raise ValueError('unknown/missing configuration fields')
    if c['study'] not in ('c1','cresst'):raise ValueError('unknown study')
    canonical=read(PACKAGE/'configs'/f"{c['study']}{'_fixture' if c['development_only'] else ''}_protocol.yaml")
    if c!=canonical:raise ValueError('configuration differs from reviewed protocol; versioned implementation required')
    return c

def sources(ms):
    roots={'EXP':ROOT,'MS':Path(ms).resolve()}; out={};commits={}
    for name,root in roots.items():
        commits[name]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        if name=='EXP':
            files=list((root/'src').rglob('*.py'))+list(PACKAGE.rglob('*.py'))+list((PACKAGE/'configs').glob('*'))
            files += [p for p in (root/'pyproject.toml',root/'uv.lock',root/'requirements.txt') if p.exists()]
        else:
            files=list(root.glob('*.tex'))+[root/'upgrade_plan.md',root/'RELEASE_GATE.json']+list((root/'scripts').glob('*.py'))+list((root/'scripts').glob('*.json'))
        out[name]={str(p.relative_to(root)):sha(p) for p in sorted(files)}
    return {'commits':commits,'files':out,'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.system(),'machine':platform.machine(),'packages':{name:version(name) for name in ('numpy','scipy','pandas','h5py','psutil','matplotlib')},'installed_distributions':dict(sorted((d.metadata['Name'],d.version) for d in distributions()))}}

def cells(c,stage,manifest=None):
    if stage not in ('pilot','confirm'):raise ValueError('unknown stage')
    dev=c['development_only'];settings=c['scientific_settings']
    if c['study']=='c1':
        count=(1 if dev else (8 if stage=='pilot' else 24));dims=[8] if dev else [64,128]
        rows=[{'study':'c1','d':d,'spectrum':s,'seed':seed,'stage':('development_' if dev else '')+stage,'cost_fits':27} for seed in range(count) for d in dims for s in ('S1','S2','S3')]
    else:
        if stage!='confirm':raise ValueError('CRESST uses prepare D/C/F/V, not a scientific pilot')
        if not manifest or manifest.get('qualification')!='QUALIFIED':raise ValueError('data qualification required')
        rows=[]
        for name,ch in manifest['channels'].items():
            nmax=min(128,len(ch['noise_roles']['F']));sizes=sorted(set(n for n in [nmax//4,nmax//2,nmax] if n>=12))
            if not sizes:raise ValueError('no admissible training sizes')
            main={'rank':3,'covariance':'psd','fixed_shape':False,'cal_fraction':1.0}
            variants=[dict(main,n=n,role='primary' if n==nmax else 'learning_curve') for n in sizes]
            variants += [dict(main,n=nmax,role='sensitivity',**{})|change for change in [{'rank':1},{'rank':6},{'covariance':'full'},{'covariance':'identity'},{'fixed_shape':True},{'cal_fraction':.5}]]
            for v in variants:
                if nmax<4*v['rank']:continue
                rows.append({'study':'cresst','channel':name,'stage':'development_confirm' if dev else 'confirm','cost_fits':7,**v})
    for row in rows:row['id']=identity(row)[:24]
    if len(rows)>c['budget']['max_cells'] or sum(r['cost_fits'] for r in rows)>c['budget']['max_fits']:raise ValueError('enumeration exceeds resource envelope')
    return rows
