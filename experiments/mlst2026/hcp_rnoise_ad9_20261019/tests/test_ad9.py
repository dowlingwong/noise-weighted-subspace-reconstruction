"""Fabricated JSON, arrays and git evidence only; no recording access."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_ad9_20261019 import power, protocol, guard, readings, prediction
from experiments.mlst2026.hcp_rnoise_ad9_20261019.build_bundle import make_config, encoded, new


def fabricated():
    dates = [f'd{i}' for i in range(9)]
    grids = {'control': [56, 111, 222, 444, 888],
             'A': [56, 111, 222, 444, 888], 'B': [28, 56, 111, 222, 444]}
    pred = {'dates': dates, 'units': {d: {'geometries': {
        g: {'grid': grid, 'row': {'a_best': 0. if g == 'control' else 1.,
            'B': -1. if g == 'control' else -310.8 if g == 'A' else -155.4,
            'N_star_interval': None if g == 'control' else [222. if g == 'A' else 111., 1000.]}}
        for g, grid in grids.items()}} for d in dates}}
    def summary(R):
        cells, pcs = [], []
        for g, grid in grids.items():
            for i, N in enumerate(grid):
                key = f'{g}/{i}/reference'
                sd = .755 if g == 'A' and i == 1 else .3
                vals = np.arange(9) * sd / np.std(np.arange(9), ddof=1)
                per = {d: {'D': float(np.mean([v] * R)),
                           'resample_records': [{'replicate': r, 'D': float(v)} for r in range(R)]}
                       for d, v in zip(dates, vals)}
                pv = {d: pred['units'][d]['geometries'][g]['row']['a_best'] + pred['units'][d]['geometries'][g]['row']['B'] / N for d in dates}
                means = {d: per[d]['D'] for d in dates}
                cells.append({'cell': key, 'arm': 'reference', 'per_date': per})
                pcs.append({'cell': key, 'geometry': g, 'N': N, 'date_means': means,
                    'predicted_by_date': pv, 'predicted_contrast': float(np.mean(list(pv.values()))),
                    'between_date_sd': float(np.std(list(means.values()), ddof=1)),
                    'certificate_by_date': {d: pred['units'][d]['geometries'][g]['row']['N_star_interval'] for d in dates}})
        return {'label': 'DEV PILOT; not confirmatory', 'n_dev': 9, 'dates': dates,
                'resamples': R, 'family_size': 34, 'cells': cells, 'power_cells': pcs}
    return summary(20), summary(50), pred


def test_power_boundary_and_no_input_mutation():
    inputs = fabricated(); before = copy.deepcopy(inputs)
    out = power.project(*inputs)
    assert inputs == before
    assert out['n_min'] == 18 and out['binding_cells'] == ['A/1/reference']
    assert out['required_cells'] == power.REQUIRED and out['family_size'] == 34
    assert len(out['trials']) == 53 and not out['n17_failure']['all_cells_pass']
    assert out['power_at_60']['all_cells_pass'] and out['power_at_n_min']['all_cells_pass']
    assert out['refits'] == out['new_resamples'] == 0


def test_power_impossible_has_no_freeze():
    s20, s50, pred = fabricated()
    for summary in (s20, s50):
        for c in summary['power_cells']:
            if c['geometry'] != 'control':
                c['date_means'] = {d: v * 100 for d, v in c['date_means'].items()}
                c['between_date_sd'] = float(np.std(list(c['date_means'].values()), ddof=1))
                for detail in next(x for x in summary['cells'] if x['cell'] == c['cell'])['per_date'].values():
                    detail['D'] *= 100
                    for r in detail['resample_records']: r['D'] *= 100
    out = power.project(s20, s50, pred)
    assert out['status'] == 'STOP' and out['n_min'] is None and not out['freeze_eligible']


@pytest.mark.parametrize('bad', ['dates', 'grid', 'prediction', 'certificate', 'replicate', 'mean', 'power_mean', 'prefix', 'sd', 'R', 'family', 'selection'])
def test_power_rejects_changed_source(bad):
    a, b, p = fabricated()
    if bad == 'dates': b['dates'].pop()
    if bad == 'grid': p['units']['d0']['geometries']['control']['grid'] = []
    if bad == 'prediction': b['power_cells'][0]['predicted_by_date']['d0'] += 1
    if bad == 'certificate': b['power_cells'][0]['certificate_by_date']['d0'] = [0, 1]
    if bad == 'replicate': b['cells'][0]['per_date']['d0']['resample_records'].pop()
    if bad == 'mean': b['cells'][0]['per_date']['d0']['D'] += 1
    if bad == 'power_mean': b['power_cells'][0]['date_means']['d0'] += 1
    if bad == 'prefix': b['cells'][0]['per_date']['d0']['resample_records'][0]['flag'] = True
    if bad == 'sd': b['power_cells'][0]['between_date_sd'] += 1
    if bad == 'R': b['resamples'] = 20
    if bad == 'family': b['family_size'] = 1
    if bad == 'selection':
        for d in p['dates']: p['units'][d]['geometries']['A']['row']['N_star_interval'] = [56., 1000.]
    with pytest.raises(ValueError): power.project(a, b, p)


def config():
    return {'access': {'confirmation_analysis_dates': [f'c{i:02}' for i in range(60)]},
            'n_min': 18, 'inference': {'family_alpha': .05}, 'signal': {'theta_grid': [0., .12, .25, .4]}}


def qualification(n=60):
    return {d: {'G1': i < n, 'G2': True, 'artifact': True, 'budget': True}
            for i, d in enumerate(config()['access']['confirmation_analysis_dates'])}


@pytest.mark.parametrize('n', [0, 17, 18, 60])
def test_complete_cohort_stop_boundary(n):
    out = protocol.cohort(qualification(n), config())
    assert out['confirmatory_claim_allowed'] == (n >= 18)
    assert out['n_included'] == n and len(out['excluded']) == 60 - n
    assert out['status'] == ('PASS' if n >= 18 else 'STOP')


@pytest.mark.parametrize('bad', ['missing', 'extra', 'nonboolean'])
def test_qualification_refuses_incomplete_or_implicit(bad):
    q = qualification()
    if bad == 'missing': q.pop('c00')
    if bad == 'extra': q['unknown'] = q['c00']
    if bad == 'nonboolean': q['c00']['artifact'] = None
    with pytest.raises(ValueError): protocol.cohort(q, config())


def ledger(n):
    return [{'date': f'c{i:02}', 'replicate': r, 'D': -2. + i / 100,
             'theta': 'A', 'N': 56, 'arm': 'reference'} for i in range(n) for r in range(50)]


@pytest.fixture
def fakeconfig(monkeypatch):
    monkeypatch.setattr(protocol, 'load', config)
    monkeypatch.setattr(readings, 'load', config)
    monkeypatch.setattr(guard, 'load', config)
    monkeypatch.setattr(prediction, 'load', config)


def cell_ci(n, g='control', i=0, arm='reference', positive=False):
    rows=ledger(n)
    for r in rows:
        r['theta']=g; r['N']=prediction.GRIDS[g][i]; r['arm']=arm
        if positive: r['D'] += 4
    return readings.date_interval(rows,34,expected_dates=config()['access']['confirmation_analysis_dates'][:n], qualification=qualification(n))


def paired_ledger(n):
    rows=[]
    for d in range(n):
        for r in range(50):
            for g in ('A','B'):
                for N in prediction.GRIDS[g]:
                    for arm,m in [('reference',None),('lw',256),('lw',512),('kron',256),('kron',512)]:
                        v=d/200+r/1000 if arm=='reference' else (-2 if m==256 else -.4)+d/100+r/1000
                        rows.append(dict(date=f'c{d:02}',replicate=r,theta=g,N=N,arm=arm,m=m,D=v))
    return rows


def test_intervals_and_pz_rules(fakeconfig):
    for n in (18,60):
        controls=[cell_ci(n,i=i) for i in range(5)]
        iso=[cell_ci(n,g=g,i=i,arm='isotropic',positive=True) for g in prediction.GRIDS for i in range(5)]
        assert readings.pz3(controls)['met']
        assert readings.pz5(iso)['met']
        assert not readings.pz3(controls[:1],[controls[0]['cell']])['available']
        assert not readings.pz5(iso[:1],[iso[0]['cell']])['available']
        assert not readings.pz5(iso+[iso[0]])['available']
        lo=cell_ci(n,g='A',i=0);hi=cell_ci(n,g='A',i=4,positive=True)
        assert readings.pz2(lo,hi,True)['met']
        assert not readings.pz2(lo,hi,False)['met']
    rows=[{'theta':v,'B_cert':i} for i,v in enumerate([0.,.12,.25,.4])]
    assert readings.pz1(rows)['met']
    rows[-1]['B_cert']=0
    assert not readings.pz1(rows)['met']


def test_canonical_families_and_shared_cohort(fakeconfig):
    controls=[cell_ci(18,i=i) for i in range(5)]
    iso=[cell_ci(18,g=g,i=i,arm='isotropic',positive=True) for g in prediction.GRIDS for i in range(5)]
    controls[-1]=cell_ci(19,i=4)
    iso[-1]=cell_ci(19,g='B',i=4,arm='isotropic',positive=True)
    assert not readings.pz3(controls)['available']
    assert not readings.pz5(iso)['available']
    assert not readings.pz3([cell_ci(18,g='A',i=i) for i in range(5)])['available']
    assert not readings.pz2(cell_ci(18,g='A',i=0),cell_ci(19,g='A',i=4,positive=True),True)['available']


def test_complete_paired_pz4(fakeconfig):
    out=readings.paired_interactions(paired_ledger(18),qualification=qualification(18))
    low={e:out[e+'_m256'] for e in ('lw','kron')};high={e:out[e+'_m512'] for e in ('lw','kron')}
    assert readings.pz4(low,high)['met']
    assert out['lw_m256']['mean'] == pytest.approx(-2+np.mean(np.arange(18))/200)
    high['lw']=cell_ci(18,g='A',i=0)
    assert not readings.pz4(low,high)['available']
    out19=readings.paired_interactions(paired_ledger(19),qualification=qualification(19))
    assert not readings.pz4(low,{e:out19[e+'_m512'] for e in ('lw','kron')})['available']


@pytest.mark.parametrize('bad',['missing_pair','missing_cell','duplicate','nonfinite','R20','below18'])
def test_paired_pz4_rejects_incomplete_evidence(fakeconfig,bad):
    n=17 if bad=='below18' else 18;rows=paired_ledger(n)
    if bad=='missing_pair':rows.pop()
    if bad=='missing_cell':rows=[r for r in rows if not(r['theta']=='A' and r['N']==56)]
    if bad=='duplicate':rows.append(rows[0])
    if bad=='nonfinite':rows[0]['D']=float('nan')
    if bad=='R20':rows=[r for r in rows if r['replicate']<20]
    with pytest.raises(ValueError): readings.paired_interactions(rows,qualification=qualification(n))


@pytest.mark.parametrize('bad', ['below18', 'R20', 'missing', 'duplicate', 'mixed', 'nonfinite', 'family', 'omitted_date'])
def test_intervals_fail_closed(fakeconfig, bad):
    n = 17 if bad == 'below18' else 18
    rows = ledger(n); dates = config()['access']['confirmation_analysis_dates'][:n]
    if bad == 'missing': rows.pop()
    if bad == 'duplicate': rows.append(rows[0])
    if bad == 'mixed': rows[0]['N'] = 111
    if bad == 'nonfinite': rows[0]['D'] = float('nan')
    if bad == 'omitted_date': dates = dates[:-1]
    with pytest.raises(ValueError):
        readings.date_interval(rows, 1 if bad == 'family' else 34, expected_dates=dates,
            qualification=qualification(n), resamples=20 if bad == 'R20' else 50)


def test_zero_sd_is_undecided(fakeconfig):
    rows = ledger(18)
    for r in rows: r['D'] = -1.
    ci = readings.date_interval(rows, 34, expected_dates=config()['access']['confirmation_analysis_dates'][:18], qualification=qualification(18))
    assert ci['zero_SE'] and ci['decision'] == 'undecided'


def evidence(monkeypatch):
    digest = lambda b: hashlib.sha256(b).hexdigest()
    f = guard.FREEZE
    names = ['PROTOCOL_FROZEN.md', 'protocol_frozen.json', 'FREEZE_KIT_SHA256.json', 'POWER_AD9.json',
             'DEV_PREDICTIONS.json', 'PILOT_SUMMARY.json', 'PILOT_R50_REF_SUMMARY.json',
             'CONFIRMATION_RUN_CLI.md', 'CONFIRMATION_AUTHORIZATION_TEMPLATE.md']
    cfg = {**config(), 'frozen': True}
    blobs = {f+'/'+n: b'{}' for n in names}
    blobs[f+'/protocol_frozen.json'] = encoded(cfg)
    sums = ''.join(digest(blobs[f+'/'+n])+'  '+f+'/'+n+'\n' for n in names).encode()
    recordpath = 'reviews/hcp_rnoise_registration_20261020/REGISTRATION.md'
    authpath = str(Path(recordpath).parent/'CONFIRMATION_AUTHORIZATION.md')
    record = f'OSF URL: https://osf.io/abc12/\nGit tag: hcp-rnoise-registration-20261020\nTagged commit: abc\nBundle SHA-256: {digest(sums)}\nOTS SHA-256: {digest(b"ots")}\n'
    auth = f'OSF URL: https://osf.io/abc12/\nGit tag: hcp-rnoise-registration-20261020\nOTS SHA-256: {digest(b"ots")}\nAuthor signature: Dowling Wong; 2026-10-20 12:00 CEST\n'
    blobs.update({recordpath: record.encode(), authpath: auth.encode(), f+'/REGISTRATION_SHA256SUMS': sums,
                  f+'/REGISTRATION_SHA256SUMS.ots': b'ots'})
    def git(ms, *args):
        if args[0] == 'rev-parse': return b'abc\n'
        if args[0] == 'merge-base': return b''
        try: return blobs[args[1].split(':', 1)[1]]
        except KeyError: raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(guard, 'git', git)
    monkeypatch.setattr(guard, 'load', lambda: cfg)
    monkeypatch.setattr(guard, 'verify', lambda expected: expected)
    return blobs, recordpath, authpath


@pytest.mark.parametrize('bad', ['none', 'missing_record', 'url', 'tag', 'commit', 'bundle', 'ots', 'content', 'protocol', 'signature', 'authorization', 'incomplete', 'duplicate', 'path', 'malformed'])
def test_registration_gate(monkeypatch, bad):
    blobs, r, a = evidence(monkeypatch)
    f = guard.FREEZE
    if bad == 'missing_record': del blobs[r]
    if bad == 'url': blobs[r] = blobs[r].replace(b'osf.io', b'example.org')
    if bad == 'tag': blobs[r] = blobs[r].replace(b'Git tag: hcp-rnoise-registration-20261020', b'Git tag: HEAD')
    if bad == 'commit': blobs[r] = blobs[r].replace(b'Tagged commit: abc', b'Tagged commit: def')
    if bad == 'bundle': blobs[r] = blobs[r].replace(b'Bundle SHA-256:', b'Bad field:')
    if bad == 'ots': blobs[f+'/REGISTRATION_SHA256SUMS.ots'] = b'changed'
    if bad == 'content': blobs[f+'/POWER_AD9.json'] = b'changed'
    if bad == 'protocol': monkeypatch.setattr(guard, 'load', lambda: {})
    if bad == 'signature': blobs[a] = blobs[a].replace(b'Dowling Wong;', b'UNSIGNED;')
    if bad == 'authorization': blobs[a] = blobs[a].replace(b'abc12', b'other')
    if bad in ('incomplete', 'duplicate', 'path', 'malformed'):
        old = blobs[f+'/REGISTRATION_SHA256SUMS']; lines = old.splitlines(keepends=True)
        if bad == 'incomplete': lines.pop()
        if bad == 'duplicate': lines.append(lines[0])
        if bad == 'path': lines[0] = lines[0].replace(b'reviews/', b'../reviews/')
        if bad == 'malformed': lines[0] = b'bad\n'
        changed = b''.join(lines)
        blobs[f+'/REGISTRATION_SHA256SUMS'] = changed
        blobs[r] = blobs[r].replace(hashlib.sha256(old).hexdigest().encode(), hashlib.sha256(changed).hexdigest().encode())
    if bad == 'none': assert guard.registration('fixture', r)['status'] == 'PASS'
    else:
        with pytest.raises(PermissionError): guard.registration('fixture', r)


def test_cli_refuses_without_registration(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        guard.main(['--ms', str(tmp_path), '--registration-record', 'bad'])
    assert error.value.code == 2 and 'STOP' in capsys.readouterr().err


def complete_prediction(date='c59'):
    row={k:0. for k in prediction.ROW_KEYS}
    row.update(theta=0.,n_starts=64,checks={'H':True},N_star_pred=222.,N_star_interval=[222.,1000.],
        restart_agreement=True,B_dagger=dict.fromkeys(('B_dagger','lambda_min','B_cert_KG','B_cert_KG_raw','zero_cut','Delta_k_KG'),0.))
    fixed=[prediction.annotate({**row,'theta':t},sorted(set(prediction.GRIDS['A']+prediction.GRIDS['B']))) for t in (0.,.12,.25,.4)]
    adaptive={g:dict(root=True,available=True,theta=.2,row=prediction.annotate({**row,'theta':.2},prediction.GRIDS[g])) for g in ('A','B')}
    return dict(date=date,date_index=12+config()['access']['confirmation_analysis_dates'].index(date),pre_fit=True,
        Sref_sha256_f64le=prediction.sref_hash(np.eye(72)),
        fixed=fixed,adaptive=adaptive,PZ1=prediction.inherited_pz1(fixed),
        grids={'P':1111,'A_grid':prediction.GRIDS['A'],'B_grid':prediction.GRIDS['B']},
        geometries={'control':{'theta':0.,'row':fixed[0],'grid':prediction.GRIDS['control']},
            **{g:dict(available=True,theta=.2,row=adaptive[g]['row'],grid=prediction.GRIDS[g]) for g in ('A','B')}})


def test_prediction_order_and_membership(fakeconfig, monkeypatch, tmp_path):
    monkeypatch.setattr(guard, 'registration', lambda *a: {'tagged_commit': 'registered'})
    monkeypatch.setattr(guard, 'git', lambda ms, *a: b'after' if a[0] == 'rev-parse' else b'')
    monkeypatch.setattr(guard, 'committed', lambda *a, **kw: 'hash' if kw['remote'] else pytest.fail('not published'))
    p = tmp_path/'prediction.json'; p.write_text(json.dumps(complete_prediction()))
    assert guard.before_fit(tmp_path, 'r', 'c59', p, 'after')['prediction_sha256'] == 'hash'
    p.write_text('{"date":"c59","pre_fit":true}')
    with pytest.raises(PermissionError): guard.before_fit(tmp_path, 'r', 'c59', p, 'after')
    with pytest.raises(PermissionError): guard.before_fit(tmp_path, 'r', 'dev', p, 'after')
    monkeypatch.setattr(guard, 'git', lambda *a: b'registered')
    with pytest.raises(PermissionError): guard.before_fit(tmp_path, 'r', 'c59', p, 'registered')
    def older(ms, *a):
        if a[0] == 'rev-parse': return b'older'
        raise subprocess.CalledProcessError(1, a)
    monkeypatch.setattr(guard, 'git', older)
    with pytest.raises(PermissionError): guard.before_fit(tmp_path, 'r', 'c59', p, 'older')


@pytest.mark.parametrize('bad',['metadata','date_index','hash','fixed','adaptive','row','grid','theta','certificate','diagnostics'])
def test_prediction_record_rejects_incomplete_or_inconsistent(fakeconfig,monkeypatch,bad):
    monkeypatch.setattr(prediction,'load',config)
    p=complete_prediction()
    if bad=='metadata':p={'date':'c59','pre_fit':True}
    if bad=='date_index':p['date_index']=0
    if bad=='hash':p['Sref_sha256_f64le']='placeholder'
    if bad=='fixed':p['fixed'].pop()
    if bad=='adaptive':p['adaptive'].pop('B')
    if bad=='row':p['fixed'][1].pop('a_best')
    if bad=='grid':p['geometries']['A']['grid']=[]
    if bad=='theta':p['geometries']['A']['theta']=.3
    if bad=='certificate':p['fixed'][1]['N_star_interval']=[2.,1.]
    if bad=='diagnostics':p['fixed'][1]['n_starts']=1
    with pytest.raises(PermissionError):prediction.validate_record(p,'c59')


def test_prediction_adapter_and_explicit_unavailable(fakeconfig,monkeypatch):
    monkeypatch.setattr(prediction,'load',config)
    p=complete_prediction()
    payload={k:v for k,v in p.items() if k not in ('date','date_index','Sref_sha256_f64le')}
    monkeypatch.setattr(prediction,'predict_unit',lambda S,P,i:payload)
    assert prediction.prediction_record(np.eye(72),'c59')['date_index']==71
    for g in ('A','B'):
        p['adaptive'][g]=dict(root=False,available=False,reason='no root')
        p['geometries'][g].update(available=False,theta=None,row=None,reason='no root')
    assert prediction.validate_record(p,'c59')
    p['geometries']['A']['reason']=None
    with pytest.raises(PermissionError):prediction.validate_record(p,'c59')


def test_array_fit_checks_binding_and_order(fakeconfig, monkeypatch):
    from experiments.mlst2026.hcp_rnoise_20261019 import learning,gates,signal
    from types import SimpleNamespace
    record=complete_prediction()
    monkeypatch.setattr(prediction,'load',config)
    calls=[]
    def gate(*args):calls.append('guard');return {'prediction_record':record}
    monkeypatch.setattr(guard,'before_fit',gate)
    monkeypatch.setattr(learning,'fit_cell',lambda *a:calls.append('fit'))
    monkeypatch.setattr(gates,'reference',lambda X:np.eye(72))
    model=SimpleNamespace(C=12,T=6,theta=.2,variances=np.ones((2,3)),templates=lambda:np.ones((72,6)))
    expected=SimpleNamespace(variances=np.ones((2,3)),templates=lambda:np.ones((72,6)))
    monkeypatch.setattr(signal,'build',lambda S,t:expected)
    kw=dict(ms='fixture',registration_record='r',date='c59',prediction_path='p',prediction_commit='c',geometry='A')
    roles={'ref':(np.zeros((100,72)),None)}
    guard.fit_confirmation(roles,model,np.eye(72),56,256,'lw',71,49,**kw)
    assert calls==['guard','fit']
    for idx,rep,N,m,arm,S in [(59,0,56,256,'lw',np.eye(72)),(71,50,56,256,'lw',np.eye(72)),
        (71,0,1,256,'lw',np.eye(72)),(71,0,56,1024,'lw',np.eye(72)),
        (71,0,56,256,'reference',np.eye(72)),(71,0,56,256,'lw',np.eye(72)*2)]:
        with pytest.raises(PermissionError):guard.fit_confirmation(roles,model,S,N,m,arm,idx,rep,**kw)
    model.theta=.3
    with pytest.raises(PermissionError):guard.fit_confirmation(roles,model,np.eye(72),56,256,'lw',71,0,**kw)
    model.theta=.2
    model.templates=lambda:np.zeros((72,6))
    with pytest.raises(PermissionError):guard.fit_confirmation(roles,model,np.eye(72),56,256,'lw',71,0,**kw)
    model.templates=expected.templates
    model.variances=np.ones((2,3))*2
    with pytest.raises(PermissionError):guard.fit_confirmation(roles,model,np.eye(72),56,256,'lw',71,0,**kw)
    model.variances=expected.variances
    monkeypatch.setattr(gates,'reference',lambda X:np.eye(72)*2)
    with pytest.raises(PermissionError):guard.fit_confirmation(roles,model,np.eye(72),56,256,'lw',71,0,**kw)


def test_identity_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(protocol, 'PACKAGE', tmp_path)
    actual = {'kit_identity': 'fixture'}
    (tmp_path/'KIT_SHA256.json').write_bytes(encoded(actual))
    (tmp_path/'INHERITED_KIT_SHA256.json').write_bytes(encoded({'kit_identity': 'old'}))
    monkeypatch.setattr(protocol, 'kit_hashes', lambda *a: actual)
    monkeypatch.setattr(protocol, 'inherited_verify', lambda: {'kit_identity': 'old'})
    assert protocol.verify(actual) == actual
    with pytest.raises(PermissionError): protocol.verify({'kit_identity': 'changed'})
    monkeypatch.setattr(protocol, 'inherited_verify', lambda: {})
    with pytest.raises(PermissionError): protocol.verify(actual)


def test_config_delta_and_stable_bytes():
    cfg = {'access': {'unused_confirmation_dates': ['old']}, 'inference': {},
           'gates': {'artifact': {'max_abs_tesla': 5e-10}, 'E1': {}}}
    split = {'confirmation_dates': ['c0'], 'units': [{'split': 'confirmation', 'measurement_date': 'c0', 'subject_id': '1'}]}
    before = copy.deepcopy(cfg)
    out = make_config(cfg, {'gates': {'artifact': {'amplitude_baseline': 'median'}}}, split, 'disclosure')
    assert cfg == before and out['n_min'] == 18 and out['inference']['resamples'] == 50
    assert out['gates']['artifact']['max_abs_tesla'] == 5e-10
    assert encoded(out) == encoded(copy.deepcopy(out)) and encoded(out).endswith(b'\n')


def test_exclusive_publication(tmp_path):
    p = tmp_path/'out.json'; new(p, b'first')
    with pytest.raises(FileExistsError): new(p, b'second')
    assert p.read_bytes() == b'first'
