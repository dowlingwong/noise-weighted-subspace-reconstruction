"""D3--D5 orchestration. No prediction/learning/confirmation entry point."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
import mne
import numpy as np
from experiments.mlst2026.hcp_rnoise_20261019 import controls, gates, windows
from experiments.mlst2026.hcp_rnoise_20261019 import preprocess as inherited_pre
from experiments.mlst2026.hcp_rnoise_20261019.protocol import sha
from experiments.mlst2026.hcp_rnoise_20261019.runner import environment
from experiments.mlst2026.meg_z_20261013.common import atomic_json
from .access import authorize, config, date_guard, verify, ADDENDUM_SHA, DECISIONS_SHA
from .artifact import screen


def clean(obj):
    """Preserve nonfinite diagnostic values explicitly in portable JSON."""
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)): return [clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else str(float(obj))
    return obj


def write(path, obj):
    atomic_json(path, clean(obj))


def new_file(path, obj):
    path = Path(path)
    if path.exists(): raise FileExistsError('STOP output exists: ' + str(path))
    write(path, obj)


def read_native(date, ms, review):
    unit = date_guard(date)
    authorize(config()['access']['dev_dates'], ms, review)
    raw = inherited_pre.open_header(date, unit['subject_id'])
    try:
        return raw.get_data(picks=config()['geometry']['selected_channels'])
    finally:
        raw.close()


def preprocess(data, unit):
    artifact = screen(data)
    if not artifact['pass']:
        raise inherited_pre.Excluded('artifact: whole date excluded')
    inherited_pre.kernel()
    p = config()['preprocessing']
    # Median belongs ONLY to the screen. Original raw values go into the FIR.
    filtered = mne.filter.filter_data(data, unit['sfreq_hz'], **p['filter'], verbose='error')[:, ::p['decimation_factor']]
    plan = windows.plan(unit['n_times'], unit['sfreq_hz'])
    return filtered, windows.cut(filtered, plan), plan


def planned_budget(unit):
    p = windows.plan(unit['n_times'], unit['sfreq_hz'])
    mock = {r: (np.empty((len(rows), 0)), {'block': np.array([v['block'] for v in rows])}) for r, rows in p['records'].items()}
    return gates.budget(mock)


def qualify(dates, ms, review, out):
    units = authorize(dates, ms, review); env = environment(); started = time.monotonic()
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    report = {'stage': 'D3', 'environment': env, 'kit_identity': verify()['kit_identity'],
              'decisions_sha256': DECISIONS_SHA, 'addendum_sha256': ADDENDUM_SHA,
              'units': {}, 'status': 'IN_PROGRESS', 'confirmation_access': False}
    write(out / 'QUALIFICATION.json', report)
    for index, unit in enumerate(units):
        date = unit['measurement_date']; t0 = time.monotonic()
        result = {'date': date, 'subject_id': unit['subject_id'], 'date_index': index,
                  'budget': planned_budget(unit), 'G3': {'applicable': False}}
        try:
            data = read_native(date, ms, review)
            if data.shape != (12, unit['n_times']):
                raise inherited_pre.Excluded('decoded native shape differs; no shortening')
            artifact = screen(data); result['artifact'] = artifact
            if artifact['pass']:
                filtered, roles, plan = preprocess(data, unit)
                result.update(gates.assess(roles, native_artifact=artifact, date_index=index))
                # Local arrays are never committed. Save even excluded G1/G2 units;
                # the controls contract requires all 12 shapes without selection.
                np.save(out / (date + '_decimated.npy'), filtered, allow_pickle=False)
                result['local_decimated_sha256'] = sha(out / (date + '_decimated.npy'))
                result['cause'] = [g for g in ('G1', 'G2', 'budget') if not result[g]['pass']]
            else:
                result.update({'G1': {'pass': False, 'evaluated': False, 'reason': 'native artifact exclusion'},
                               'G2': {'pass': False, 'evaluated': False, 'reason': 'native artifact exclusion'},
                               'cause': ['artifact']})
        except (inherited_pre.Excluded, FileNotFoundError, OSError, np.linalg.LinAlgError) as exc:
            result.update({'input': {'pass': False, 'reason': str(exc)}, 'cause': [type(exc).__name__, str(exc)]})
            for key in ('artifact', 'G1', 'G2'):
                result.setdefault(key, {'pass': False, 'evaluated': False, 'reason': str(exc)})
        result['seconds'] = time.monotonic() - t0
        report['units'][date] = result
        write(out / 'QUALIFICATION.json', report)
        print(date, 'PASS' if all(result[k]['pass'] for k in ('artifact','G1','G2','budget')) else 'EXCLUDED', result['cause'], flush=True)
    report['E1'] = gates.e1(report['units']); report['status'] = report['E1']['status']
    report['seconds'] = time.monotonic() - started
    write(out / 'QUALIFICATION.json', report)
    return report


def qualified(out):
    out = Path(out); record = json.loads((out / 'QUALIFICATION.json').read_text())
    if record['kit_identity'] != verify()['kit_identity'] or record['E1'] != gates.e1(record['units']) or not record['E1']['pass']:
        raise RuntimeError('STOP E1 not passed by this exact kit')
    return record


def seal(units, out):
    out = Path(out); record = qualified(out); rows = {}
    for unit in units:
        date = unit['measurement_date']; source = out / (date + '_decimated.npy')
        if not source.is_file() or sha(source) != record['units'][date].get('local_decimated_sha256'):
            raise RuntimeError('STOP all 12 immutable dev source arrays required; no selective pooling')
        data = np.load(source, allow_pickle=False); p = windows.plan(unit['n_times'], unit['sfreq_hz'])
        Sc, lags = controls.ref_moments(data, p)
        S = gates.reference(windows.cut(data, p)['ref'][0])
        rows[date] = {'Sref': S.tolist(), 'spatial_covariance': Sc.tolist(), 'temporal_lags': lags.tolist(),
                      'source_sha256': sha(source), 'n_ref': len(p['records']['ref'])}
    result = {'stage': 'D4-seal', 'kit_identity': verify()['kit_identity'],
              'qualification_sha256': sha(out / 'QUALIFICATION.json'), 'moments': rows}
    new_file(out / 'SEALED_MOMENTS.json', result)
    return {'path': str(out / 'SEALED_MOMENTS.json'), 'sha256': sha(out / 'SEALED_MOMENTS.json'),
            'must_commit_before_fit': True}


def committed_seal(ms, commit, out):
    path = 'reviews/hcp_rnoise_dev_20261019/SEALED_MOMENTS.json'
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    blob = subprocess.check_output(['git', '-C', str(ms), 'show', f'{commit}:{path}'], env=env)
    local = Path(out) / 'SEALED_MOMENTS.json'
    if hashlib.sha256(blob).hexdigest() != sha(local): raise RuntimeError('STOP seal is not committed at named commit')
    record = qualified(out); seal_record = json.loads(blob)
    if seal_record['qualification_sha256'] != sha(Path(out) / 'QUALIFICATION.json'):
        raise RuntimeError('STOP qualification changed after seal')
    if set(seal_record['moments']) != set(config()['access']['dev_dates']):
        raise RuntimeError('STOP 12 sealed shapes required')
    return seal_record


def fit_generator(ms, commit, out):
    moments = committed_seal(ms, commit, out); shapes = {}
    for date, row in sorted(moments['moments'].items()):
        try:
            shapes[date] = controls.fit_shape(row['spatial_covariance'], row['temporal_lags'])
        except (ValueError, np.linalg.LinAlgError) as exc:
            new_file(Path(out) / 'GENERATOR_STOP.json', {'status':'STOP','failed_date':date,
                     'reason':str(exc),'seal_commit':commit,
                     'sealed_moments_sha256':sha(Path(out)/'SEALED_MOMENTS.json'),
                     'completed_diagnostics':{d:shape['fit'] for d,shape in shapes.items()}})
            raise
    result = {'stage': 'D4-generator', 'seal_commit': commit,
              'sealed_moments_sha256': sha(Path(out) / 'SEALED_MOMENTS.json'),
              'diagnostics': {d: shape['fit'] for d, shape in shapes.items()}, 'status': 'PASS'}
    new_file(Path(out) / 'GENERATOR.json', result)
    return result


def shapes_from_seal(out):
    out = Path(out); seal_record = json.loads((out / 'SEALED_MOMENTS.json').read_text())
    generator = json.loads((out / 'GENERATOR.json').read_text())
    if generator['status'] != 'PASS' or generator['sealed_moments_sha256'] != sha(out / 'SEALED_MOMENTS.json'):
        raise RuntimeError('STOP generator/seal mismatch')
    return {date: controls.fit_shape(row['spatial_covariance'], row['temporal_lags']) for date, row in seal_record['moments'].items()}


class Deadline(RuntimeError): pass


def alarm(signum, frame):
    raise Deadline('STOP fixed runtime cap reached; no restart')


def cost(units, out):
    """Actual 12-shape control probes and full 64-start population rows."""
    from experiments.mlst2026.hcp_rnoise_20261019.population import general_noise
    from experiments.mlst2026.hcp_rnoise_20261019.signal import build
    out = Path(out); shapes = shapes_from_seal(out); timings = []; d3s = []
    previous = signal.signal(signal.SIGALRM, alarm)
    try:
        for i, unit in enumerate(units):
            date = unit['measurement_date']; shape = shapes[date]
            for law, cell in ((1,0),(2,0),(4,2),(6,5),(7,12)):
                start = time.monotonic(); controls.replicate(shape, unit, law, cell, i, 0, timing=True)
                timings.append({'date':date, 'law':law, 'cell':cell, 'seconds':time.monotonic()-start})
            for theta in (.12, .4):
                start = time.monotonic(); signal.setitimer(signal.ITIMER_REAL, 128.)
                try:
                    gen, _ = controls.rng(2,99,i,int(theta*100))
                    S = gates.reference(controls.generate(shape,unit,2,gen)['ref'][0])
                    general_noise(build(S,theta),S,shape['population'],gen,config()['signal']['population_random_starts'])
                finally: signal.setitimer(signal.ITIMER_REAL,0)
                d3s.append({'date':date, 'theta':theta, 'seconds':time.monotonic()-start})
                print('cost',date,theta,d3s[-1]['seconds'],flush=True)
    except Deadline:
        new_file(out / 'COST.json', {'status':'STOP', 'reason':'real-shape population row exceeded 128s', 'control_timings':timings, 'D3S_timings':d3s})
        raise
    finally: signal.signal(signal.SIGALRM,previous)
    cmax=max(t['seconds'] for t in timings); pmax=max(t['seconds'] for t in d3s)
    projections={str(w):{'controls_hours_ideal':168000*cmax/w/3600, 'controls_hours_25pct':1.25*168000*cmax/w/3600,
                         'D3S_hours_ideal':3600*pmax/w/3600, 'D3S_hours_25pct':1.25*3600*pmax/w/3600} for w in (4,16)}
    result={'status':'PASS' if projections['4']['controls_hours_25pct']<=6 and projections['16']['D3S_hours_25pct']<=8 else 'STOP',
            'control_timings':timings,'D3S_timings':d3s,'projections':projections,
            'generator_sha256':sha(out/'GENERATOR.json'), 'kit_identity':verify()['kit_identity'],
            'uncertainty':'worst observed shape/probe, illustrative 25% overhead; multiworker throughput unmeasured',
            'campaign_workers':16, 'campaign_wall_cap_seconds':21600,
            'worker_count_authority':'current D5 prompt explicitly specifies 16; original 6h/4-worker projection cap retained'}
    new_file(out/'COST.json',result);return result


def campaign_task(task):
    unit, shape, law, cell, index = task
    return [controls.replicate(shape,unit,law,cell,index,r,details=True) for r in range(config()['controls']['replicates_per_shape'])]


def campaign(units, out, workers=16):
    if workers != 16: raise ValueError('STOP D5 requires 16 workers')
    out=Path(out); projection=json.loads((out/'COST.json').read_text())
    if projection['status']!='PASS' or projection['generator_sha256']!=sha(out/'GENERATOR.json') or projection['kit_identity']!=verify()['kit_identity']:
        raise RuntimeError('STOP real-shape cost gate not passed')
    ledger=out/'CONTROLS.jsonl'
    if ledger.exists() or (out/'CAMPAIGN_STARTED.json').exists(): raise RuntimeError('STOP campaign already attempted; no restart')
    shapes=shapes_from_seal(out); cfg=config()['controls']
    tasks=[(u,shapes[u['measurement_date']],law,cell,i) for law in cfg['laws'] for cell in cfg['cells'] for i,u in enumerate(units)]
    new_file(out/'CAMPAIGN_STARTED.json',{'workers':workers,'cap_seconds':21600,'started_unix':time.time(),'kit_identity':verify()['kit_identity']})
    start=time.monotonic();previous=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,21600)
    pool=None
    try:
        pool=ProcessPoolExecutor(max_workers=workers)
        with ledger.open('x') as handle:
            for rows in pool.map(campaign_task,tasks,chunksize=1):
                for row in rows: handle.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
                handle.flush()
        pool.shutdown(wait=True);pool=None
        # The same wall clock covers parsing, statistical reduction and output
        # publication, as well as worker execution and shutdown.
        result=summarize_ledger(out);result['seconds']=time.monotonic()-start;result['workers']=workers
        if result['seconds']>=21600:raise Deadline('STOP complete D5 stage exceeded cap')
        new_file(out/'CONTROLS_SUMMARY.json',result)
        new_file(out/'CAMPAIGN_COMPLETED.json',{
            'status':result['status'],'ledger_sha256':sha(ledger),
            'summary_sha256':sha(out/'CONTROLS_SUMMARY.json'),
            'generator_sha256':sha(out/'GENERATOR.json'),'kit_identity':verify()['kit_identity'],
            'records':result['records'],'seconds':time.monotonic()-start,'workers':workers})
        if time.monotonic()-start>=21600:raise Deadline('STOP D5 publication exceeded cap')
        return result
    except BaseException as exc:
        # Cap/worker failure never resumes. Kill running children rather than
        # waiting past the fixed wall cap in Executor.__exit__.
        signal.setitimer(signal.ITIMER_REAL,0)
        if pool is not None:
            for proc in list((pool._processes or {}).values()): proc.terminate()
            pool.shutdown(wait=True,cancel_futures=True)
        new_file(out/'CAMPAIGN_STOP.json',{'status':'STOP','seconds':time.monotonic()-start,
                 'reason':str(exc),'exception':type(exc).__name__,'no_restart':True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def summarize_ledger(out):
    """Internal reducer; CLI summaries additionally require completion proof."""
    out=Path(out)
    with (out/'CONTROLS.jsonl').open() as handle: rows=[json.loads(line) for line in handle]
    result=controls.summarize(rows)
    result.update({'ledger_sha256':sha(out/'CONTROLS.jsonl'),'records':len(rows),'generator_sha256':sha(out/'GENERATOR.json')})
    return result


def summarize(out):
    out=Path(out)
    if (out/'CAMPAIGN_STOP.json').exists():
        raise RuntimeError('STOP persisted campaign failure; no summary may upgrade it')
    completed=out/'CAMPAIGN_COMPLETED.json'
    if not completed.is_file():raise RuntimeError('STOP campaign completion record missing')
    record=json.loads(completed.read_text())
    if (record['ledger_sha256']!=sha(out/'CONTROLS.jsonl')
            or record['generator_sha256']!=sha(out/'GENERATOR.json')
            or record['summary_sha256']!=sha(out/'CONTROLS_SUMMARY.json')
            or record['kit_identity']!=verify()['kit_identity']
            or not 0<=record['seconds']<21600 or record['workers']!=16):
        raise RuntimeError('STOP campaign completion evidence differs')
    result=summarize_ledger(out)
    if result['status']!=record['status'] or result['records']!=record['records']:
        raise RuntimeError('STOP recomputed outcome differs from completed campaign')
    return result
