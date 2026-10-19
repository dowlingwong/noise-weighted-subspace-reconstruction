"""Regression counterexamples and independent fabricated diagnostic calculations."""
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_20261019 import controls,gates,protocol,readings,descriptives
from experiments.mlst2026.hcp_rnoise_20261019.tests.test_contract import _complete_interval


@pytest.mark.parametrize('sd,pred,n,family',[
    ([],[],9,1),([-1.],[1.],9,1),([np.nan],[1.],9,1),([1.],[np.inf],9,1),
    ([1.],[np.nan],9,1),([1.],[1.,2.],9,2),([1.],[1.],8,1),([1.],[1.],9,0),
    ([1.,2.],[1.,2.],9,1)])
def test_invalid_power_evidence_never_passes(sd,pred,n,family):
    with pytest.raises(ValueError):gates.n_min(sd,n,pred,family,cell_ids=[str(i) for i in range(len(sd))],expected_cells=[str(i) for i in range(len(sd))])


def test_date_ledger_complete_default_and_explicit_remedy():
    dates=[str(i) for i in range(8)]
    records=[{'date':d,'replicate':r,'D':1+int(d)*.01} for d in dates for r in range(20)]
    assert readings.date_interval(records,2,expected_dates=dates)['resamples']==20
    with pytest.raises(ValueError,match='incomplete'):readings.date_interval(records[:-1],2,expected_dates=dates)
    with pytest.raises(ValueError,match='duplicate'):readings.date_interval(records+[records[0]],2,expected_dates=dates)
    bad=[dict(r) for r in records];bad[0]['D']=np.nan
    with pytest.raises(ValueError,match='nonfinite'):readings.date_interval(bad,2,expected_dates=dates)
    with pytest.raises(ValueError):readings.date_interval(records,2,expected_dates=dates+['other'])
    short=[{'date':d,'replicate':0,'D':1.} for d in dates]
    with pytest.raises(ValueError,match='incomplete'):readings.date_interval(short,2,expected_dates=dates)
    remedy=[{'date':d,'replicate':r,'D':1+int(d)*.01} for d in dates for r in range(50)]
    assert readings.date_interval(remedy,2,expected_dates=dates,resamples=50)['resamples']==50
    with pytest.raises(ValueError):readings.date_interval(remedy,2,expected_dates=dates)
    with pytest.raises(ValueError):readings.date_interval(records,2,expected_dates=dates,resamples=51)


def test_incomplete_or_nonfinite_pz_evidence_is_unavailable():
    assert not readings.pz1([{'theta':0.,'B_cert':0.}])['met']
    full=[{'theta':t,'B_cert':i} for i,t in enumerate(protocol.load()['signal']['theta_grid'])]
    full[-1]['B_cert']=np.nan
    assert not readings.pz1(full)['available']
    valid=_complete_interval(0,.5)
    bad={**valid,'mean':np.nan,'ci':[np.nan,np.nan],'decision':'undecided'}
    for reader in (readings.pz3,readings.pz5):
        assert not reader([bad],['cell'])['available']
        assert reader([valid],['cell'])['met']  # valid finite undecided keeps reading
        assert not reader([valid])['met']  # no expected frozen family
        assert not reader([valid],['cell','missing'])['met']
        assert not reader([valid,valid],['cell','other'])['met']
    assert not readings.pz4({'lw':valid},{'lw':valid})['met']


def test_lag_pairs_never_cross_original_blocks():
    gen=np.random.default_rng(119);X=gen.normal(size=(8,72))
    X[4:]+=30.
    meta={'block':np.array([2]*4+[5]*4),'start':np.array([100,106,112,118,1000,1006,1012,1018])}
    roles={r:(X.copy(),meta.copy()) for r in ('train','cal','eval')}
    result=descriptives.lag_covariance(roles,np.eye(72))
    Z=X-X.mean(axis=0);left=np.array([0,1,2,4,5,6]);right=left+1
    expected=Z[left].T@Z[right]/6
    assert result['roles']['eval']['n_pairs']==6
    assert np.allclose(result['roles']['eval']['covariance_whitened'],expected)
    assert not np.allclose(expected,Z[:-1].T@Z[1:]/7)


def test_bootstrap_whole_blocks_matches_raw_concatenation():
    gen=np.random.default_rng(20);roles={}
    for role,nb in (('train',11),('cal',6),('eval',6)):
        X=gen.normal(size=(nb*4,72))+np.repeat(np.arange(nb),4)[:,None]
        meta={'block':np.repeat(np.arange(nb),4),'start':np.tile(np.arange(4)*6,nb)+np.repeat(np.arange(nb)*100,4)}
        roles[role]=(X,meta)
    result=descriptives.block_bootstrap(roles,np.eye(72),3)
    assert result['replicates']==200 and result['available']
    assert result['seed_entropy']==[20261019,9,103,3,0]
    sampled={}
    for role,(X,meta) in roles.items():
        ids=result['whole_block_draw_ids'][role][0]
        assert len(ids)==len(np.unique(meta['block']))
        sampled[role]=np.concatenate([X[meta['block']==b] for b in ids])
    expected=gates.role_g2(sampled['train'],sampled['eval'],np.eye(72),False)
    for k,value in result['first_replicate_statistics']['eval'].items():
        assert abs(value-expected[k])<1e-12
    assert result['conditional_on_fixed_Sref'] and result['not_calibrated_CI'] and result['not_ESS']


def test_default_controls_keep_cheap_diagnostics_without_qualification_work(monkeypatch):
    from experiments.mlst2026.hcp_rnoise_20261019 import descriptives
    def forbidden(*args,**kw):pytest.fail('qualification bootstrap called during controls')
    monkeypatch.setattr(descriptives,'block_bootstrap',forbidden)
    u=next(u for u in protocol.split()['units'] if u['split']=='dev')
    row=controls.replicate(controls.synthetic_shape(),u,2,0,0,999)
    assert 'G2_cal' in row and 'scan_third_transport' in row and 'extremes' in row['G2']
    assert 'whole_block_bootstrap' not in row and 'G4' not in row and 'G5' not in row


def test_control_report_retains_descriptions_and_unavailable_denominators():
    cfg=protocol.load()['controls']
    diag={'G2_cal':{'G2S_pass':False},
          'scan_third_transport':{'0':{'G2S_pass':True},'1':{'G2S_pass':False},'2':{'G2S_pass':True}}}
    rows=[{'law':l,'cell':c,'date_index':d,'replicate':r,'G1':{'pass':True},
           'G2':{'pass':c==0,'extremes':[.2,3.]},**diag}
          for l in cfg['laws'] for c in cfg['cells'] for d in range(12) for r in range(200)]
    rows[0]={**rows[0],'G1':{'pass':False},'numerical_failure':'fabricated_singular'}
    result=controls.summarize(rows)
    assert result['status']==result['endpoint_status']=='PASS'
    desc=result['descriptive']
    assert desc['L1/C0']['diagnostics_n_valid']==2399
    assert desc['L1/C0']['unavailable_reason_counts']=={'fabricated_singular':1}
    assert desc['L2/C0']['generalised_extreme_quantiles_025_50_975']==[[.2,3.]]*3
    assert desc['L2/C0']['cal_G2S_exclusion_rate']==1.
    assert desc['L2/C0']['scan_third_G2S_any_exclusion_rate']==1.
    assert desc['L2/C0']['scan_third_G2S_exclusion_counts']=={'0':0,'1':2400,'2':0}
    rows[1]=dict(rows[1]);rows[1].pop('G2_cal')
    missing=controls.summarize(rows)
    assert missing['endpoint_status']=='PASS' and missing['status']=='STOP'
    assert missing['required_diagnostics_missing']==1


def test_qualification_emits_both_dependence_descriptions():
    u=next(u for u in protocol.split()['units'] if u['split']=='dev')
    gen,_=controls.rng(2,99,0,999)
    rec=controls.generate(controls.synthetic_shape(),u,2,gen)
    result=gates.assess(rec,details=True)
    assert result['within_block_lag_covariance']['roles']['eval']['n_pairs']==600
    assert result['whole_block_bootstrap']['blocks_per_role']=={'train':11,'cal':6,'eval':6}
    assert result['whole_block_bootstrap']['available']

def test_power_explicit_frozen_cell_set_cannot_silently_shrink():
    with pytest.raises(ValueError,match='identities'):
        gates.n_min([.001],9,[1.],2,cell_ids=['c1'],expected_cells=['c1','c2'])
    with pytest.raises(ValueError,match='identities'):
        gates.n_min([.001],9,[1.],1)
    assert gates.n_min([.001,.002],9,[1.,1.],2,cell_ids=['c1','c2'],expected_cells=['c1','c2'])['pass']
