import json
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_20261019 import protocol, controls, windows, gates
from experiments.mlst2026.hcp_rnoise_dev_20261019 import access, artifact, pipeline, runner


@pytest.fixture
def dates(): return protocol.load()['access']['dev_dates']


def test_every_nondev_date_refused_before_reader(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline.inherited_pre,'open_header',lambda *a:pytest.fail('reader reached'))
    bad=[u['measurement_date'] for u in protocol.split()['units'] if u['split']=='confirmation']+['2099-01-01','']
    for date in bad:
        with pytest.raises(PermissionError):pipeline.read_native(date,tmp_path,tmp_path/'review')


def test_missing_unsigned_changed_decisions_before_reader(monkeypatch,tmp_path,dates):
    monkeypatch.setattr(pipeline.inherited_pre,'open_header',lambda *a:pytest.fail('reader reached'))
    sheet=tmp_path/'reviews/hcp_rnoise_author_decisions_20261019/DECISIONS.md'
    for content in (None,'Status: UNSIGNED','Status: SIGNED but changed'):
        if content is not None:
            sheet.parent.mkdir(parents=True,exist_ok=True);sheet.write_text(content)
        with pytest.raises(PermissionError,match='DECISIONS'):
            pipeline.read_native(dates[0],tmp_path,tmp_path/'review')


def test_duplicate_missing_and_mixed_dates_refused(tmp_path,dates):
    for bad in (dates[:-1],dates+[dates[0]],[dates[0]]*12,dates[:-1]+['2099-01-01']):
        with pytest.raises(PermissionError):access.authorize(bad,tmp_path,tmp_path/'review')


def test_review_exact_verdict_and_identity(tmp_path):
    p=tmp_path/'review'
    for text in ('Verdict: **CHANGES REQUIRED**\nabc','Verdict: **ACCEPTED**\nwrong','Verdict: **ACCEPTED WITH CHANGES**\nwrong'):
        p.write_text(text)
        with pytest.raises(PermissionError):access.accepted_review(p,['abc'])
    p.write_text('Verdict: **ACCEPTED**\nabc');access.accepted_review(p,['abc'])


def test_complete_fabricated_authority_and_reader_close(tmp_path,monkeypatch,dates):
    # Entire authority tree is fabricated. No recording path is accessed.
    sheet=tmp_path/'reviews/hcp_rnoise_author_decisions_20261019/DECISIONS.md'
    sheet.parent.mkdir(parents=True);sheet.write_text('**Status: SIGNED (fabricated)**\nApproved by: **Dowling Wong**\n')
    add=tmp_path/'reviews/hcp_rnoise_design_addendum_20261019';add.mkdir()
    (add/'ADDENDUM.md').write_text('fabricated addendum')
    (add/'CONFIG_DELTA.json').write_bytes((access.PACKAGE/'CONFIG_DELTA.json').read_bytes())
    monkeypatch.setattr(access,'DECISIONS_SHA',protocol.sha(sheet))
    monkeypatch.setattr(access,'ADDENDUM_SHA',protocol.sha(add/'ADDENDUM.md'))
    monkeypatch.setattr(access,'verify',lambda:{'kit_identity':'fabricated-kit'})
    hashes='\n'.join([access.ADDENDUM_SHA,access.DELTA_SHA,access.DECISIONS_SHA])
    ar=tmp_path/'reviews/hcp_rnoise_design_addendum_review_20261019_01/REPORT.md'
    ar.parent.mkdir();ar.write_text('Verdict: **ACCEPTED**\n'+hashes)
    ir=tmp_path/'reviews/hcp_rnoise_dev_kit_review_20261019_01/REPORT.md'
    ir.parent.mkdir();ir.write_text('Verdict: **ACCEPTED**\n'+hashes+'\nfabricated-kit')
    assert len(access.authorize(dates,tmp_path,ir))==12
    closed=[];data=np.zeros((12,100))
    class Raw:
        def get_data(self,**kw):
            assert kw['picks']==protocol.load()['geometry']['selected_channels'];return data
        def close(self):closed.append(True)
    monkeypatch.setattr(pipeline.inherited_pre,'open_header',lambda date,subject:Raw())
    assert pipeline.read_native(dates[0],tmp_path,ir) is data and closed==[True]
    ar.write_text('Verdict: **CHANGES REQUIRED**\n'+hashes)
    with pytest.raises(PermissionError):access.authorize(dates,tmp_path,ir)


@pytest.mark.parametrize('command',['confirmation','predict','fit','pilot','register'])
def test_future_clis_always_prohibited(command):
    with pytest.raises(PermissionError):runner.main([command])


def test_artifact_constant_offsets_and_medians():
    base=np.tile(np.linspace(-1e-12,1e-12,101),(12,1))
    offset=base+np.arange(1,13)[:,None]*1e-8
    result=artifact.screen(offset)
    assert result['pass'] and result['amplitude_count']==result['step_count']==0
    assert np.allclose([r['median_tesla'] for r in result['channels'].values()],np.arange(1,13)*1e-8,rtol=0,atol=1e-23)
    assert artifact.screen(base)['pass']


def test_amplitude_boundary_even_median_and_local_excursion():
    # A symmetric slow ramp reaches exactly +/- the inclusive amplitude limit;
    # its increments remain below the unchanged step limit.
    x=np.tile(np.linspace(-5e-10,5e-10,101),(12,1));result=artifact.screen(x)
    assert not result['pass'] and result['amplitude_count']==24 and result['step_count']==0
    assert all(r['median_tesla']==0 for r in result['channels'].values())
    x=np.zeros((12,102));x[0,-1]=5e-10
    assert artifact.screen(x)['channels']['MEG 001']['amplitude_count']==1
    x=np.tile([1e-12,3e-12],(12,1));result=artifact.screen(x)
    assert all(r['median_tesla']==2e-12 for r in result['channels'].values())


def test_raw_step_boundary():
    x=np.zeros((12,101));x[:,51:]=5e-11
    result=artifact.screen(x)
    assert not result['pass'] and result['step_count']==12 and result['amplitude_count']==0


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_raw_nonfinite_has_explicit_unavailable_amplitude(value):
    x=np.zeros((12,101));x[0,50]=value;result=artifact.screen(x)
    assert not result['pass'] and result['nonfinite_count']==1
    assert result['amplitude_count'] is None and result['step_unavailable_pairs']==2
    assert result['channels']['MEG 001']['median_tesla'] is None
    assert result['channels']['MEG 002']['amplitude_count']==0
    json.dumps(result,allow_nan=False)


def test_original_fir_input_not_median_subtracted(monkeypatch):
    u=next(u for u in protocol.split()['units'] if u['split']=='dev')
    x=np.ones((12,u['n_times']))*1e-8
    def filtering(data,*args,**kwargs):
        assert data is x and np.array_equal(data,x)
        return data.copy()
    monkeypatch.setattr(pipeline.mne.filter,'filter_data',filtering)
    filtered,roles,p=pipeline.preprocess(x,u)
    assert np.all(filtered==1e-8)
    assert {r:len(X) for r,(X,meta) in roles.items()}=={'ref':3630,'train':1111,'cal':606,'eval':606}


def test_qualification_records_all_dates_and_e1_stops_without_fit(tmp_path,monkeypatch,dates):
    units=[{**protocol.unit(d),'n_times':1000} for d in dates]
    monkeypatch.setattr(pipeline,'authorize',lambda *a:units)
    monkeypatch.setattr(pipeline,'environment',lambda:{'fabricated':True})
    monkeypatch.setattr(pipeline,'verify',lambda:{'kit_identity':'fabricated'})
    monkeypatch.setattr(pipeline,'read_native',lambda *a:np.zeros((12,1000)))
    monkeypatch.setattr(pipeline,'planned_budget',lambda *a:{'pass':True})
    monkeypatch.setattr(pipeline,'preprocess',lambda *a:(np.zeros((12,100)),{},{}))
    def assess(roles,native_artifact,date_index):
        return {'G1':{'pass':date_index<8},'G2':{'pass':date_index<8},'budget':{'pass':True},'artifact':native_artifact}
    monkeypatch.setattr(pipeline.gates,'assess',assess)
    out=tmp_path/'run';report=pipeline.qualify(dates,tmp_path,tmp_path/'review',out)
    assert len(report['units'])==12 and len(report['E1']['included'])==8 and report['status']=='STOP'
    with pytest.raises(RuntimeError,match='E1'):pipeline.seal(units,out)
    with pytest.raises(FileExistsError):pipeline.qualify(dates,tmp_path,tmp_path/'review',out)


def test_seal_requires_exact_committed_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *a,**kw:b'wrong committed bytes')
    (tmp_path/'SEALED_MOMENTS.json').write_text('{}')
    with pytest.raises(RuntimeError,match='not committed'):pipeline.committed_seal(tmp_path,'abc',tmp_path)


def test_generator_instability_no_repair():
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        controls.fit_shape(np.eye(12),np.ones(20))


def test_controls_missing_duplicate_and_exact_cp():
    with pytest.raises(ValueError,match='incomplete'):controls.summarize([])
    row={'law':1,'cell':0,'date_index':0,'replicate':0}
    with pytest.raises(ValueError,match='duplicate'):controls.summarize([row,row])
    assert controls.claim_bound(90,2400,'null')['pass']
    assert not controls.claim_bound(91,2400,'null')['pass']
    assert controls.claim_bound(2202,2400,'detection')['pass']
    assert not controls.claim_bound(2201,2400,'detection')['pass']


def test_campaign_requires_exact_workers_and_refuses_restart(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='16'):pipeline.campaign([],tmp_path,4)
    (tmp_path/'GENERATOR.json').write_text('{}')
    (tmp_path/'COST.json').write_text(json.dumps({'status':'PASS','generator_sha256':protocol.sha(tmp_path/'GENERATOR.json'),'kit_identity':'fixture'}))
    monkeypatch.setattr(pipeline,'verify',lambda:{'kit_identity':'fixture'})
    (tmp_path/'CAMPAIGN_STARTED.json').write_text('{}')
    with pytest.raises(RuntimeError,match='already attempted'):pipeline.campaign([],tmp_path)


def test_diagnostics_nonfinite_values_retained_explicitly():
    assert pipeline.clean({'condition':float('inf'),'pass':np.bool_(False)})=={'condition':'inf','pass':False}


@pytest.fixture
def mocked_campaign(tmp_path,monkeypatch):
    (tmp_path/'GENERATOR.json').write_text('{}')
    (tmp_path/'COST.json').write_text(json.dumps({'status':'PASS','generator_sha256':protocol.sha(tmp_path/'GENERATOR.json'),'kit_identity':'fixture'}))
    monkeypatch.setattr(pipeline,'verify',lambda:{'kit_identity':'fixture'})
    monkeypatch.setattr(pipeline,'shapes_from_seal',lambda *a:{})
    clock={'now':0.,'armed':False,'summary_seen':False}
    monkeypatch.setattr(pipeline.time,'monotonic',lambda:clock['now'])
    monkeypatch.setattr(pipeline.signal,'signal',lambda *a:None)
    def timer(kind,seconds):clock['armed']=seconds>0
    monkeypatch.setattr(pipeline.signal,'setitimer',timer)
    class Pool:
        _processes={}
        def __init__(self,max_workers):assert max_workers==16
        def map(self,*a,**kw):return []
        def shutdown(self,**kw):pass
    monkeypatch.setattr(pipeline,'ProcessPoolExecutor',Pool)
    def reduce(out):
        assert clock['armed'],'deadline must remain active throughout reduction'
        clock['summary_seen']=True
        return {'status':'PASS','records':168000}
    monkeypatch.setattr(pipeline,'summarize_ledger',reduce)
    return tmp_path,clock


def test_campaign_deadline_remains_active_through_publication(mocked_campaign,monkeypatch):
    out,clock=mocked_campaign
    original=pipeline.new_file
    def publication(path,obj):
        if Path(path).name in ('CONTROLS_SUMMARY.json','CAMPAIGN_COMPLETED.json'):
            assert clock['armed'] and clock['summary_seen']
        return original(path,obj)
    monkeypatch.setattr(pipeline,'new_file',publication)
    assert pipeline.campaign([],out)['status']=='PASS'
    assert not clock['armed'] and (out/'CAMPAIGN_COMPLETED.json').exists()


@pytest.mark.parametrize('kind',['signal','elapsed'])
def test_summary_timeout_persists_stop(mocked_campaign,monkeypatch,kind):
    out,clock=mocked_campaign
    def reduction(path):
        assert clock['armed']
        if kind=='signal':pipeline.alarm(None,None)
        clock['now']=21601.
        return {'status':'PASS','records':168000}
    monkeypatch.setattr(pipeline,'summarize_ledger',reduction)
    with pytest.raises(pipeline.Deadline):pipeline.campaign([],out)
    assert (out/'CAMPAIGN_STOP.json').exists() and not (out/'CAMPAIGN_COMPLETED.json').exists()
    with pytest.raises(RuntimeError,match='persisted'):pipeline.summarize(out)


def test_complete_ledger_with_persisted_stop_never_reduces(tmp_path,monkeypatch):
    (tmp_path/'CONTROLS.jsonl').write_text('fabricated complete ledger')
    (tmp_path/'CAMPAIGN_STOP.json').write_text('{"status":"STOP"}')
    monkeypatch.setattr(pipeline,'summarize_ledger',lambda *a:pytest.fail('reducer reached after STOP'))
    with pytest.raises(RuntimeError,match='persisted'):pipeline.summarize(tmp_path)


def test_missing_completion_never_reduces(tmp_path,monkeypatch):
    (tmp_path/'CONTROLS.jsonl').write_text('fabricated complete ledger')
    monkeypatch.setattr(pipeline,'summarize_ledger',lambda *a:pytest.fail('reducer reached without completion'))
    with pytest.raises(RuntimeError,match='completion record missing'):pipeline.summarize(tmp_path)


def test_completed_campaign_hashes_and_disposition_bound(mocked_campaign,monkeypatch):
    out,clock=mocked_campaign;pipeline.campaign([],out)
    monkeypatch.setattr(pipeline,'summarize_ledger',lambda *a:{'status':'PASS','records':168000})
    assert pipeline.summarize(out)['status']=='PASS'
    monkeypatch.setattr(pipeline,'summarize_ledger',lambda *a:{'status':'STOP','records':168000})
    with pytest.raises(RuntimeError,match='outcome differs'):pipeline.summarize(out)
    (out/'CONTROLS.jsonl').write_text('changed')
    with pytest.raises(RuntimeError,match='evidence differs'):pipeline.summarize(out)


def test_effective_config_only_ad2_diff():
    original=protocol.load();effective=access.config()
    a=original['gates'].pop('artifact');b=effective['gates'].pop('artifact')
    assert original==effective
    for key in a:
        if key!='exceedance_rule':assert a[key]==b[key]
    assert b['amplitude_baseline'].startswith('per-channel whole-scan median')
