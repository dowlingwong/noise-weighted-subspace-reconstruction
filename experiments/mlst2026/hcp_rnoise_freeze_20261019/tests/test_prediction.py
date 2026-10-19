import json
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_freeze_20261019 import prediction as p

def row(n=100, interval=(100,200)):
    return {'N_star_pred':n,'N_star_interval':interval,'checks':{'H':True},'restart_agreement':True}

def test_range_rules():
    assert p.annotate(row(),[50,100,150,200,400])['PZ2']=={'available':True,'in_range':True,'below_certificate':[50],'above_certificate':[400],'reason':None}
    assert not p.annotate(row(),[100,150,200])['PZ2']['in_range']
    assert not p.annotate(row(None,None),[50,400])['PZ2']['available']
    assert not p.annotate({**row(),'restart_agreement':False},[50,400])['PZ2']['available']
    assert not p.annotate(row(np.nan,(np.nan,200)),[50,400])['PZ2']['available']

def test_exact_prediction_wrapper(monkeypatch):
    fixed=[{'theta':t,**row()} for t in (0,.12,.25,.4)]
    def fake(S,P,i):
        assert P==1111 and i==2
        return {'fixed':fixed,'adaptive':{'A':{'root':True,'theta':.2,'row':row()},'B':{'root':False,'reason':'no root'}},'grids':{'A_grid':[56,111,222,444,888],'B_grid':[28,56,111,222,444]},'pre_fit':True}
    monkeypatch.setattr(p.population,'predictions',fake)
    r=p.predict_unit(np.eye(72),1111,2)
    assert len(r['fixed'])==4 and r['adaptive']['A']['available']
    assert not r['adaptive']['B']['available']
    assert r['geometries']['control']['grid']==[56,111,222,444,888]
    with pytest.raises(ValueError):p.predict_unit(np.eye(4),1111,2)
    with pytest.raises(np.linalg.LinAlgError):p.predict_unit(-np.eye(72),1111,2)

def test_committed_bytes(tmp_path):
    import subprocess,os
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],env=env)
    git('init','-q');git('config','user.email','fixture@example.invalid');git('config','user.name','fixture')
    f=tmp_path/'pred.json';f.write_text('{}\n');git('add','pred.json');git('commit','-qm','fixture')
    c=git('rev-parse','HEAD').decode().strip(); assert p.committed(f,tmp_path,c)
    f.write_text('{"changed":true}')
    with pytest.raises(PermissionError):p.committed(f,tmp_path,c)

def test_confirmation_refuses_before_fit(tmp_path,monkeypatch):
    from experiments.mlst2026.hcp_rnoise_freeze_20261019 import confirmation_guard as g
    from experiments.mlst2026.hcp_rnoise_20261019 import learning
    monkeypatch.setattr(learning,'fit_cell',lambda *a,**k:pytest.fail('fit reached before authority'))
    with pytest.raises(PermissionError):
        g.fit_confirmation(None,None,None,56,256,'reference',0,0,ms=tmp_path,
            registration_record='reviews/hcp_rnoise_registration_20261020/REGISTRATION.md',date='2013-06-24',
            prediction_path=tmp_path/'missing.json',prediction_commit='HEAD')
    with pytest.raises(PermissionError):g.before_fit(tmp_path,'bad','2014-01-23',tmp_path/'x','HEAD')

def test_before_fit_requires_remote_commit(monkeypatch,tmp_path):
    from experiments.mlst2026.hcp_rnoise_freeze_20261019 import confirmation_guard as g
    monkeypatch.setattr(g,'registration',lambda *a:{'status':'PASS'})
    calls=[]
    def commit(*a,**k):calls.append(k);return 'sha'
    monkeypatch.setattr(g,'committed',commit)
    f=tmp_path/'p.json';f.write_text(json.dumps({'date':'2013-06-24','pre_fit':True}))
    assert g.before_fit(tmp_path,'r','2013-06-24',f,'abc')['prediction_sha256']=='sha'
    assert calls==[{'remote':True}]
    f.write_text('{}')
    with pytest.raises(PermissionError):g.before_fit(tmp_path,'r','2013-06-24',f,'abc')

@pytest.mark.parametrize('failure',['url','tag','tag_commit','sums','ots','entry','frozen','signature','auth'])
def test_registration_evidence_guard(monkeypatch,failure):
    import hashlib
    from experiments.mlst2026.hcp_rnoise_freeze_20261019 import confirmation_guard as g
    path='reviews/hcp_rnoise_registration_20261020/REGISTRATION.md'
    frozen=b'{"frozen":true}'
    digest=lambda b:hashlib.sha256(b).hexdigest()
    sums=(digest(frozen)+'  '+g.FREEZE+'/protocol_frozen.json\n').encode()
    record=f'OSF URL: https://osf.io/abc12/\nGit tag: hcp-rnoise-registration-20261020\nTagged commit: abc\nBundle SHA-256: {digest(sums)}\nOTS SHA-256: {digest(b"ots")}\n'
    auth=f'OSF URL: https://osf.io/abc12/\nGit tag: hcp-rnoise-registration-20261020\nOTS SHA-256: {digest(b"ots")}\nAuthor signature: Dowling Wong; 2026-10-20\n'
    monkeypatch.setattr(g,'verify',lambda expected:expected)
    blobs={g.FREEZE+'/FREEZE_KIT_SHA256.json':b'{}',path:record.encode(),str(__import__('pathlib').Path(path).parent/'CONFIRMATION_AUTHORIZATION.md'):auth.encode(),g.FREEZE+'/REGISTRATION_SHA256SUMS':sums,g.FREEZE+'/REGISTRATION_SHA256SUMS.ots':b'ots',g.FREEZE+'/protocol_frozen.json':frozen}
    def fake(ms,*args):
        if args[0]=='rev-parse':return b'abc\n'
        if args[0]=='merge-base':return b''
        return blobs[args[1].split(':',1)[1]]
    monkeypatch.setattr(g,'git',fake)
    assert g.registration('x',path)['status']=='PASS'
    if failure=='url':blobs[path]=record.replace('https://osf.io/abc12/','https://example.org/x').encode()
    elif failure=='tag':blobs[path]=record.replace('Git tag: hcp-rnoise-registration-20261020','Git tag: HEAD').encode()
    elif failure=='tag_commit':blobs[path]=record.replace('Tagged commit: abc','Tagged commit: def').encode()
    elif failure=='sums':blobs[path]=record.replace(digest(sums),'0'*64).encode()
    elif failure=='ots':blobs[g.FREEZE+'/REGISTRATION_SHA256SUMS.ots']=b'changed'
    elif failure=='entry':blobs[g.FREEZE+'/protocol_frozen.json']=b'changed'
    elif failure=='frozen':
        blobs[g.FREEZE+'/protocol_frozen.json']=b'{"frozen":false}'
        newsums=(digest(blobs[g.FREEZE+'/protocol_frozen.json'])+'  '+g.FREEZE+'/protocol_frozen.json\n').encode()
        blobs[g.FREEZE+'/REGISTRATION_SHA256SUMS']=newsums
        blobs[path]=record.replace(digest(sums),digest(newsums)).encode()
    elif failure=='signature':blobs[str(__import__('pathlib').Path(path).parent/'CONFIRMATION_AUTHORIZATION.md')]=auth.replace('Dowling Wong;','UNSIGNED;').encode()
    elif failure=='auth':blobs[str(__import__('pathlib').Path(path).parent/'CONFIRMATION_AUTHORIZATION.md')]=auth.replace('https://osf.io/abc12/','https://osf.io/different/').encode()
    with pytest.raises(PermissionError):g.registration('x',path)


def test_identity_mismatch_and_registered_identity(monkeypatch,tmp_path):
    from experiments.mlst2026.hcp_rnoise_freeze_20261019 import identity as i
    monkeypatch.setattr(i,'PACKAGE',tmp_path)
    monkeypatch.setattr(i,'kit_hashes',lambda p:{'kit_identity':'actual'})
    monkeypatch.setattr(i,'dev_verify',lambda:None)
    (tmp_path/'KIT_SHA256.json').write_text('{"kit_identity":"actual"}')
    assert i.verify({'kit_identity':'actual'})['kit_identity']=='actual'
    with pytest.raises(PermissionError):i.verify({'kit_identity':'changed'})
    (tmp_path/'KIT_SHA256.json').write_text('{"kit_identity":"changed"}')
    with pytest.raises(PermissionError):i.verify()
