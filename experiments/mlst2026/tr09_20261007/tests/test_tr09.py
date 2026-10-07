import json, shutil
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.tr09_20261007 import science,workflow,guard

def test_accepted_dependency_identity():
    assert guard.closure_identity(guard.load_closure())==guard.ACCEPTED_T25_IDENTITY
    guard.assert_direct();guard.assert_loaded_modules()

def test_pairing_and_nested_prefix():
    c=workflow.config('fixture');th=c['theta'][0]
    a=science.data(c,11,th,1.)[-1];b=science.data(c,11,th,1.)[-1]
    assert np.array_equal(a,b)
    sparse=science.data(c,11,th,.1)[-1]
    assert not np.array_equal(a,sparse)
    x,mu,S=science.t25.moments(a,24)
    assert np.array_equal(x,a[:24]);assert np.allclose(S,(x-mu).T@(x-mu)/24)
    g=science.t25.geometry(c);assert g.k==1
    # The same keyed amplitude/noise records and same activity uniforms underlie p.
    L,Li=science.t25.powers(science.t25.covariance(c));V=g.V(th)
    difference=(a-sparse)@Li@V
    assert np.allclose(difference[:,g.k:],0,atol=1e-10)

def test_primary_and_secondary_overlap():
    c=workflow.config('confirmation');rows=[]
    for seed in c['seeds']:
     for th in c['theta']:
      for n in c['N']:
       for p in c['p']:
        d=float(seed%11)*(p-1)
        rows.append(dict(seed=seed,theta=th,N=n,p=p,R_U=4.,R_F=4.+d,D=d,optimization_limited=False,selected_equals_hosvd=True))
    result=science.reduce(c,rows)
    assert len(result['families']['primary'])==4
    assert len(result['families']['secondary'])==36
    for primary in result['families']['primary']:
        secondary=next(r for r in result['families']['secondary'] if all(r[k]==primary[k] for k in ['theta','N','p','endpoint']))
        assert primary['values']==secondary['values']
        assert primary['ci'][1]-primary['ci'][0]<secondary['ci'][1]-secondary['ci'][0]
    with pytest.raises(ValueError):science.reduce(c,rows+[rows[0]])
    with pytest.raises(ValueError):science.reduce(c,rows[:-1])

def test_authorization_refusal_before_output(tmp_path,monkeypatch):
    monkeypatch.delenv('TR09_CONFIRMATION_AUTHORISED',raising=False)
    out=tmp_path/'run'
    with pytest.raises(ValueError):workflow.run('confirmation',out)
    assert not out.exists()

def test_confirmation_without_acceptance_refused(tmp_path,monkeypatch):
    monkeypatch.setenv('TR09_CONFIRMATION_AUTHORISED','yes')
    out=tmp_path/'run'
    with pytest.raises(ValueError):workflow.run('confirmation',out)
    assert not out.exists()

def test_development_seeds_disjoint_and_config_guard():
    c=workflow.config('benchmark');assert set(c['seeds']).isdisjoint(workflow.config('confirmation')['seeds'])
    workflow.validate(c,'benchmark')
    c['seeds'][0]=202610060300
    with pytest.raises(ValueError):workflow.validate(c,'benchmark')

def test_raw_map_tamper_and_duplicate_census(tmp_path):
    origin=Path('/private/tmp/tr09-fixture-v3')
    if not origin.exists():pytest.skip('full fixture archive provided separately')
    out=tmp_path/'fixture';shutil.copytree(origin,out)
    # Source later gains tests, so bind to current source for a fixture consumer test.
    start=json.loads((out/'START.json').read_text());start['sources']=workflow.sources();workflow.write(out/'START.json',start)
    for rel in start['sources']:
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((workflow.ROOT/rel).read_bytes())
    workflow.audit(out)
    path=next(out.glob('cell_*.json'));row=json.loads(path.read_text())
    row['R_U']+=1.;workflow.write(path,row)
    with pytest.raises(ValueError,match='endpoint'):workflow.audit(out)


def test_receipt_tamper_and_census(tmp_path):
    origin=Path('/private/tmp/tr09-fixture-v3')
    if not origin.exists():pytest.skip('full fixture archive provided separately')
    out=tmp_path/'receipt';shutil.copytree(origin,out)
    workflow.verify_receipt(out)
    (out/'extra.txt').write_text('injected file')
    with pytest.raises(ValueError,match='census'):workflow.verify_receipt(out)
    (out/'extra.txt').unlink()
    p=next(out.glob('cell_*.json'));p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):workflow.verify_receipt(out)


@pytest.mark.parametrize('attack',['intercept','basis','factor','objective','snapshot','training'])
def test_adversarial_portable_consumer(tmp_path,attack):
    origin=Path('/private/tmp/tr09-fixture-v3')
    if not origin.exists():pytest.skip('full fixture archive provided separately')
    out=tmp_path/'altered';shutil.copytree(origin,out)
    start=json.loads((out/'START.json').read_text());start['sources']=workflow.sources();workflow.write(out/'START.json',start)
    for rel in start['sources']:
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((workflow.ROOT/rel).read_bytes())
    path=next(out.glob('cell_*.json'));row=json.loads(path.read_text())
    arrays_path=out/row['arrays'];arrays=dict(np.load(arrays_path,allow_pickle=False))
    if attack=='intercept':arrays['b_F'][0]+=.1
    if attack=='basis':arrays['B_F'][0,0]+=.1
    if attack=='factor':arrays['uc_0'][0,0]+=.1
    if attack=='objective':row['optimization']['restarts'][0]['objective']+=1
    if attack=='training':row['training_sha']='0'*64
    if attack=='snapshot':
        snap=out/'source_snapshot'/next(iter(start['sources']));snap.write_text(snap.read_text()+'\n# altered\n')
    np.savez_compressed(arrays_path,**arrays);workflow.write(path,row)
    with pytest.raises(ValueError):workflow.audit(out)


def test_namespace_claim_blocks_second_attempt_without_drawing(tmp_path,monkeypatch):
    for key in ['TR09_CONFIRMATION_AUTHORISED','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        monkeypatch.setenv(key,'yes' if key=='TR09_CONFIRMATION_AUTHORISED' else '1')
    freeze_dir=tmp_path/'freeze';body=workflow.freeze('confirmation',freeze_dir)
    ledger=tmp_path/'namespace.attempt';body['ledger_path']=str(ledger);workflow.write(freeze_dir/'freeze.json',body)
    acceptance=tmp_path/'acceptance.json';workflow.write(acceptance,dict(verdict='ACCEPTED',registered=True,namespace=body['config']['namespace'],freeze_sha256=workflow.sha(freeze_dir/'freeze.json')))
    ledger.write_text('{"prior_attempt": true}')
    def forbidden(*args,**kwargs):raise AssertionError('confirmation draw forbidden in test')
    monkeypatch.setattr(science,'fit',forbidden)
    out=tmp_path/'confirmation'
    with pytest.raises(FileExistsError):workflow.run('confirmation',out,freeze_dir/'freeze.json',acceptance,ledger)
    assert not out.exists();assert ledger.read_text()=='{"prior_attempt": true}'


@pytest.mark.parametrize('attack',['missing_exit','failed_exit','symlink'])
def test_failed_or_unsafe_receipt_refused(tmp_path,attack):
    origin=Path('/private/tmp/tr09-fixture-v3')
    if not origin.exists():pytest.skip('full fixture archive provided separately')
    out=tmp_path/'bad';shutil.copytree(origin,out)
    if attack=='missing_exit':
        (out/'EXIT.json').unlink()
        with pytest.raises(FileNotFoundError):workflow.audit(out)
    elif attack=='failed_exit':
        workflow.write(out/'EXIT.json',dict(exit_code=1))
        with pytest.raises(ValueError,match='failed'):workflow.audit(out)
    else:
        (out/'injected').symlink_to(tmp_path)
        with pytest.raises(ValueError,match='symlink'):workflow.verify_receipt(out)
