import itertools
import json
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_20261019 import protocol,windows,preprocess,gates,controls,signal,readings
from experiments.mlst2026.hcp_rnoise_20261019.runner import census_check,main

@pytest.fixture
def dev():
    return next(u for u in protocol.split()['units'] if u['split']=='dev')

@pytest.fixture
def generated(dev):
    shape=controls.synthetic_shape()
    gen,_=controls.rng(2,99,0,999)
    return controls.generate(shape,dev,2,gen),shape


def test_config_and_split():
    assert protocol.load()['frozen'] is False
    assert protocol.split_check()['confirmation']==60
    assert census_check()['grids']=={'P':1111,'N_max':1010,'A':222,'B':111.,'A_grid':[56,111,222,444,888],'B_grid':[28,56,111,222,444]}


def test_config_tampering_fails_closed(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_bytes((protocol.PACKAGE/'protocol.json').read_bytes()+b' ')
    monkeypatch.setattr(protocol,'PACKAGE',tmp_path)
    with pytest.raises(RuntimeError,match='differs'):
        protocol.load()


def test_all_confirmation_headers_refused_before_reader():
    def forbidden(*a,**kw): pytest.fail('recording reader must not be called')
    for u in protocol.split()['units']:
        if u['split']=='confirmation':
            with pytest.raises(PermissionError,match='frozen registered'):
                preprocess.open_header(u['measurement_date'],reader=forbidden)


@pytest.mark.parametrize('command',['decode-dev','gates','predict','controls','fit','confirmation'])
def test_all_real_clis_refuse_dev_and_confirmation(command,dev):
    for date in (dev['measurement_date'],protocol.load()['access']['confirmation_pool_dates'][0]):
        with pytest.raises(PermissionError):
            main([command,'--date',date])


def test_unknown_and_wrong_subject_before_reader(dev):
    for date,sub in [('2099-01-01',None),(dev['measurement_date'],'100307')]:
        with pytest.raises(PermissionError):
            preprocess.open_header(date,sub,reader=lambda *a,**kw:pytest.fail('reader called'))


def test_dev_header_mock_never_decodes(dev):
    rows=protocol.geometry()['channels']; closed=[]
    info={'chs':[dict(ch_name=r['name'],loc=np.array(r['loc']),coord_frame=r['coord_frame'],coil_type=r['coil_type']) for r in rows],
          'bads':[],'sfreq':dev['sfreq_hz'],'proc_history':[],'projs':[]}
    def reader(*args,**kw):
        assert kw['preload'] is False and kw['head_shape_fname'] is None
        assert args[0].endswith('/'+dev['subject_id']+'/c,rfDC')
        return SimpleNamespace(preload=False,info=info,n_times=dev['n_times'],close=lambda:closed.append(True),
                               get_data=lambda:pytest.fail('sample decode'),load_data=lambda:pytest.fail('sample decode'))
    raw=preprocess.open_header(dev['measurement_date'],reader=reader)
    assert not closed
    raw.close()


def test_geometry_fixed_and_incompatible_excluded():
    names,rows=preprocess.selected_geometry()
    assert len(names)==12 and names==protocol.load()['geometry']['selected_channels']
    ch=[dict(ch_name=r['name'],loc=np.array(r['loc']),coord_frame=r['coord_frame'],coil_type=r['coil_type']) for r in rows]
    ch[0]['loc'][0]+=.001
    with pytest.raises(preprocess.Excluded,match='geometry'):
        preprocess.selected_geometry({'chs':ch,'bads':[]})


@pytest.mark.parametrize('T,train,cal',[(20,330,180),(16,418,228),(12,550,300),(10,671,366),(8,836,456),(6,1111,606)])
def test_counts_and_margins(dev,T,train,cal):
    p=windows.plan(dev['n_times'],dev['sfreq_hz'],T)
    assert len(p['blocks'])==29
    assert len(p['records']['train'])==train
    assert len(p['records']['cal'])==len(p['records']['eval'])==cal
    assert all(b['end']-b['start']==1017 for b in p['blocks'])


def test_vectorization_grid_and_no_shared_fir_support(dev):
    p=windows.plan(dev['n_times'],dev['sfreq_hz'])
    X=np.arange(p['n_decimated'])[None,:]+100000*np.arange(12)[:,None]
    rec=windows.cut(X,p)
    for role,(v,meta) in rec.items():
        for b in np.unique(meta['block']):
            offsets=meta['start'][meta['block']==b]
            assert np.all(np.diff(offsets)==(1 if role=='ref' else 6))
        assert np.array_equal(v[0],X[:,meta['start'][0]:meta['start'][0]+6].T.reshape(-1))
    assert p['guard_samples']/p['sfreq_hz']>2*(len(preprocess.kernel())-1)/(2*dev['sfreq_hz'])


def test_exhaustive_equal_inclusion_and_original_indices():
    B,L=3,4;X=np.arange(B*L)[:,None];meta={'block':np.repeat(np.arange(B),L)}
    class Fixed:
        def __init__(self,order,rot): self.order=order;self.rot=iter(rot)
        def permutation(self,n): return np.array(self.order)
        def integers(self,n): return next(self.rot)
    for N in range(1,B*L+1):
        counts=np.zeros(B*L,int)
        for order in itertools.permutations(range(B)):
            for rot in itertools.product(range(L),repeat=B):
                sample,idx=windows.sample_blocks(X,meta,N,Fixed(order,rot))
                assert np.array_equal(sample,X[idx]) and len(set(idx))==N
                counts[idx]+=1
        assert len(set(counts))==1
    with pytest.raises(ValueError,match='partial'):
        windows.sample_blocks(X[:-1],{'block':meta['block'][:-1]},2,np.random.default_rng(1))


@pytest.mark.parametrize('which',['amplitude','step','nan','inf'])
def test_artifact_thresholds_are_native_and_any_exceedance_excludes(which):
    x=np.zeros((12,100))
    if which=='amplitude': x[0,:]=5e-10
    elif which=='step': x[0,50:]=5e-11
    elif which=='nan': x[0,0]=np.nan
    else: x[0,0]=np.inf
    assert preprocess.artifact_screen(x)['pass'] is False


def test_artifact_below_threshold_passes():
    x=np.tile(np.sin(np.arange(200)/10)*1e-12,(12,1))
    assert preprocess.artifact_screen(x)['pass'] is True


def test_fir_and_array_preprocessing_matches_raw_filter(dev):
    import mne
    n=50000;sf=dev['sfreq_hz']
    x=np.array([(j+1)*1e-13*np.sin(2*np.pi*7*np.arange(n)/sf) for j in range(12)])
    rec,summary=preprocess.preprocess_array(x,sf)
    raw=mne.io.RawArray(x.copy(),mne.create_info(12,sf,ch_types='mag'),verbose='error')
    raw.filter(1,40,method='fir',phase='zero',fir_design='firwin',verbose='error')
    expected=windows.cut(raw.get_data()[:,::20],summary['plan'])
    for role in windows.ROLES:
        assert np.allclose(rec[role][0],expected[role][0],rtol=0,atol=1e-26)
    assert summary['n_decimated']==2500


def test_forward_and_filtered_basis_golden():
    A=signal.spatial_templates();B,C=signal.temporal_bases()
    assert A.shape==(12,2) and abs(abs(A[:,0]@A[:,1])-.8517275500825963)<1e-12
    assert np.allclose(B.T@B,np.eye(3),atol=1e-12)
    assert np.allclose(C.T@C,np.eye(3),atol=1e-12)
    assert np.max(abs(B.T@C))<1e-12


def test_signal_scale_fixed_across_theta_and_exact_product_zero():
    S=np.diag(np.linspace(1,2,72))*1e-24
    m0=signal.build(S,0);m4=signal.build(S,.4)
    assert np.allclose(m0.variances,m4.variances,rtol=1e-14,atol=0)
    assert abs(np.trace(np.linalg.solve(S,m0.second_moment()))-206.604)<1e-10
    assert np.linalg.matrix_rank(m0.templates())==6


def test_sarvas_matches_mne_point_sensor_formula():
    from mne.forward._compute_forward import _do_sphere_field
    _,rows=preprocess.selected_geometry()
    r=np.array([v['loc'][:3] for v in rows]);nn=np.array([v['loc'][9:12] for v in rows]);nn/=np.linalg.norm(nn,axis=1,keepdims=True)
    p=np.array([.03,.015,.04]);q=np.array([1.,0,0])
    expected=_do_sphere_field(p[None,:],r,nn,np.ones(12),np.arange(12),np.zeros(3))
    got=signal.sarvas(r,nn,p,q)
    assert np.allclose(got,q@expected,rtol=1e-12,atol=1e-25)


def test_m_hi_e1_and_power():
    assert gates.choose_m_hi({str(i):606 for i in range(12)})==512
    with pytest.raises(ValueError): gates.choose_m_hi({'one':606})
    dates=protocol.load()['access']['dev_dates']
    results={d:{g:{'pass':i<9} for g in ('G1','G2','artifact','budget')} for i,d in enumerate(dates)}
    assert gates.e1(results)['pass'] is True
    results[dates[8]]['G2']['pass']=False
    assert gates.e1(results)['status']=='STOP'
    assert gates.n_min([.001],9,[1.],1)['n_min']==8
    assert gates.n_min([100.],9,[.1],1,after_remedy=True)['confirmation_access'] is False


def test_shape_ar_and_ref_lags_ignore_guards(dev):
    shape=controls.synthetic_shape()
    assert abs(shape['ar_coefficients'][0]-.7)<1e-10 and max(abs(shape['ar_coefficients'][1:]))<1e-10
    p=windows.plan(dev['n_times'],dev['sfreq_hz']);g=np.random.default_rng(19)
    x=g.normal(size=(12,p['n_decimated']))*1e-12
    Sc,r=controls.ref_moments(x,p)
    changed=x.copy()
    for b in p['blocks']: changed[:,b['start']:b['active_start']]=1e90
    Sc2,r2=controls.ref_moments(changed,p)
    assert np.array_equal(Sc,Sc2) and np.array_equal(r,r2)
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        controls.fit_shape(np.eye(12),np.ones(20))


def test_generator_stationary_first_block_and_role_shapes(dev):
    shape=controls.synthetic_shape()
    a=controls.generate(shape,dev,2,np.random.default_rng(7));b=controls.generate(shape,dev,4,np.random.default_rng(7))
    assert np.array_equal(a['ref'][0][0],b['ref'][0][0])
    assert {r:len(v[0]) for r,v in a.items()}=={'ref':3630,'train':1111,'cal':606,'eval':606}


def test_eval_only_cells_do_not_change_ref_train_cal(generated):
    rec,shape=generated;S=gates.reference(rec['ref'][0])
    for cell in protocol.load()['controls']['cells']:
        changed=controls.apply_cell(rec,S,shape,cell)
        for r in ('ref','train','cal'): assert changed[r] is rec[r]
        if cell==0: assert np.array_equal(changed['eval'][0],rec['eval'][0])
        else: assert not np.array_equal(changed['eval'][0],rec['eval'][0])
    with pytest.raises(ValueError,match='stimulus'):
        controls.apply_cell(rec,S,shape,7)


@pytest.mark.parametrize('g1,g2,null,detect',[(True,True,False,False),(True,False,True,True),(False,False,True,False),(False,True,True,False)])
def test_asymmetric_gate_events(g1,g2,null,detect):
    row={'G1':{'pass':g1},'G2':{'pass':g2}}
    assert controls.event(row,'null')==null
    assert controls.event(row,'detection')==detect


def test_finite_ill_conditioned_and_singular_are_not_detections(generated,monkeypatch,dev):
    rec,shape=generated
    S=np.eye(72);S[0,0]=1e-12
    monkeypatch.setattr(gates,'reference',lambda X:S)
    result=gates.assess(rec,details=False)
    assert not result['G1']['pass'] and not result['G2']['evaluated']
    assert not controls.event(result,'detection')
    monkeypatch.setattr(controls,'generate',lambda *a: {r:(np.zeros_like(X),meta) for r,(X,meta) in rec.items()})
    result=controls.replicate(shape,dev,2,1,0,999)
    assert 'numerical_failure' in result
    assert controls.event(result,'null') and not controls.event(result,'detection')


@pytest.mark.parametrize('count,kind,passed',[(90,'null',True),(91,'null',False),(2201,'detection',False),(2202,'detection',True),(2112,'detection',False)])
def test_exact_cp_boundaries(count,kind,passed):
    assert controls.claim_bound(count,2400,kind)['pass'] is passed


def test_summary_claim_set_and_complete_campaign_contract():
    cfg=protocol.load()['controls'];tuples={(r['type'],r['law'],r['cell']) for r in cfg['claim_tuples']}
    assert len(tuples)==24 and ('null',5,0) in tuples and not any(l==7 for _,l,_ in tuples)
    # Fabricated outcomes only: no generator, campaign, or data loader is run.
    rows=[{'law':l,'cell':c,'date_index':d,'replicate':r,'G1':{'pass':True},'G2':{'pass':c==0}}
          for l in cfg['laws'] for c in cfg['cells'] for d in range(12) for r in range(200)]
    result=controls.summarize(rows)
    assert result['status']=='PASS'
    assert {(v['type'],v['law'],v['cell']) for v in result['claims']}==tuples
    with pytest.raises(ValueError,match='incomplete'):
        controls.summarize(rows[:-1])
    with pytest.raises(ValueError,match='duplicate'):
        controls.summarize(rows+[rows[0]])


def test_date_means_and_pz_signs():
    rec=[{'date':str(d),'D':d+r*.1} for d in range(8) for r in range(20)]
    interval=readings.date_interval(rec,4)
    assert interval['n']==8 and len(interval['date_means'])==8
    assert readings.pz1([{'theta':0,'B_cert':0},{'theta':.12,'B_cert':1},{'theta':.25,'B_cert':1}])['met']
    neg={'decision':'negative','mean':-1};pos={'decision':'positive','mean':1};small={'decision':'undecided','mean':-.1}
    assert readings.pz2(neg,pos,True)['met'] and not readings.pz2(neg,pos,False)['met']
    assert readings.pz3([neg])['met'] and not readings.pz3([pos])['met']
    assert readings.pz4([neg,neg],[small,small])['met']
    assert not readings.pz5([neg])['met']

def test_stationarity_gate_detects_eval_scale():
    gen=np.random.default_rng(1919)
    X=gen.normal(size=(3000,72));Y=gen.normal(size=(3000,72));S=np.eye(72)
    null=gates.role_g2(X,Y,S,False)
    drift=gates.role_g2(X,Y*np.sqrt(1.5),S,False)
    assert null['pass'] is True and drift['pass'] is False
    assert drift['physical_trace_ratio']>1.25


def test_empty_prediction_family_never_passes_by_vacuous_truth():
    assert readings.pz1([])['met'] is False
    assert readings.pz3([])['met'] is False
    assert readings.pz4([],[])['met'] is False
    assert readings.pz5([])['met'] is False
