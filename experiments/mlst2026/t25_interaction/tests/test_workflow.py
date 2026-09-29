"""End-to-end synthetic workflows and mutation boundaries, no scientific exports."""
import os,sys,subprocess,json,shutil
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[4]

def cli(pkg,*args,ok=True):
 env=os.environ.copy();env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')});env['PYTHONPATH']=str(ROOT/'src')
 p=subprocess.run([sys.executable,'-m','experiments.mlst2026.'+pkg,*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True)
 assert (p.returncode==0)==ok,p.stdout+p.stderr
 return p

def test_t25_full_fixture_and_boundaries(tmp_path):
 pkg='t25_interaction';cfg=ROOT/'experiments/mlst2026'/pkg/'fixture.json';f=tmp_path/'freeze.json';run=tmp_path/'run'
 cli(pkg,'freeze','--config',cfg,'--out',f);cli(pkg,'run','--freeze',f,'--run-root',run)
 cli(pkg,'verify','--run-root',run);cli(pkg,'reduce','--run-root',run,'--out',tmp_path/'reduction.json')
 cli(pkg,'export','--run-root',run,'--out',tmp_path/'export',ok=False)
 cli(pkg,'run','--freeze',f,'--run-root',run,ok=False)
 cli(pkg,'freeze','--config',cfg.with_name('production.json'),'--out',tmp_path/'production-freeze.json',ok=False)
 # Coherently rehashed bogus record still rejected numerically.
 from experiments.mlst2026.t25_interaction.workflow import sha
 bad=tmp_path/'tampered';shutil.copytree(run,bad);rec=bad/'cell-0000/record.json';r=json.loads(rec.read_text());r['methods']['iso__restricted']['clean']=-999;rec.write_text(json.dumps(r));done=bad/'cell-0000/COMPLETE.json';d=json.loads(done.read_text());d['record_sha']=sha(rec);done.write_text(json.dumps(d))
 cli(pkg,'verify','--run-root',bad,ok=False)
 interrupted=tmp_path/'interrupted';shutil.copytree(run,interrupted);(interrupted/'FAILED.json').write_text('{}')
 cli(pkg,'verify','--run-root',interrupted,ok=False);cli(pkg,'diagnose','--run-root',interrupted)
 source=tmp_path/'source-mutated';shutil.copytree(run,source);p=source/'freeze.json';d=json.loads(p.read_text());d['sources']['head']='0'*40;p.write_text(json.dumps(d));cli(pkg,'verify','--run-root',source,ok=False)

def test_t24_fixture_closed_schema_and_ledgers(tmp_path):
 pkg='t24_cumulant';f=tmp_path/'f.json';run=tmp_path/'run'
 cli(pkg,'freeze','--fixture','--out',f);cli(pkg,'run','--freeze',f,'--run-root',run);cli(pkg,'verify','--run-root',run)
 cli(pkg,'reduce','--run-root',run,'--out',tmp_path/'reduced.json');cli(pkg,'export','--run-root',run,'--out',tmp_path/'export',ok=False)
 cli(pkg,'freeze','--out',tmp_path/'production.json',ok=False)
 for field,value in [('N',65),('version','fake')]:
  bad=tmp_path/field;shutil.copytree(run,bad);p=bad/'freeze.json';d=json.loads(p.read_text());d[field]=value;p.write_text(json.dumps(d));cli(pkg,'verify','--run-root',bad,ok=False)
 bad=tmp_path/'failed';shutil.copytree(run,bad);(bad/'FAILED.json').write_text('{}');cli(pkg,'verify','--run-root',bad,ok=False)
 bad=tmp_path/'nostart';shutil.copytree(run,bad);(bad/'START.json').rename(bad/'START.missing.json');cli(pkg,'verify','--run-root',bad,ok=False)
