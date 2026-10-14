import hashlib,json,os
from pathlib import Path
import pytest
from experiments.mlst2026.meg_z_r1_20261014 import protocol,partition,runner
K=protocol.PACKAGE

def test_config_equals_sha_pinned_MS():
    p=Path(os.environ.get('MS_CONFIG_PATH',K/'fixtures/MS_CONFIG.json'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==protocol.CONFIG_SHA
    assert protocol.load()==json.loads(p.read_text())
    assert protocol.load()==json.loads((K/'fixtures/MS_CONFIG.json').read_text())

def test_partition_source_hash():
    assert hashlib.sha256((K/'partition.py').read_bytes()).hexdigest()==protocol.PARTITION_SHA

def test_census_exact_and_budget():
    r=runner.census_check();assert r['T']==10 and r['m_hi']==768

@pytest.mark.parametrize('G',range(0,170))
def test_partition_guards_final_gap_and_no_cross_role_adjacency(G):
    for run in range(1,7):
        ro=partition.assign_roles(G,run);assert len(ro)==G
        blocks=max(1,G//12)
        for g,r in enumerate(ro):
            b=min(g//12,blocks-1)
            expected='guard' if b>0 and g==12*b else partition.CYCLE[(b+run-1)%6]
            assert r==expected
        for x,y in zip(ro,ro[1:]):assert x==y or 'guard' in (x,y)

@pytest.mark.parametrize('sub',[f'sub-{i:02}' for i in range(5,17)])
def test_confirmation_refused(sub):
    with pytest.raises(SystemExit):protocol.guard([sub])
    with pytest.raises(SystemExit):protocol.guard([sub],'variant.json')

def test_unknown_subject_refused():
    with pytest.raises(SystemExit):protocol.guard(['sub-99'])

def test_identity_mutation_detected(tmp_path,monkeypatch):
    import shutil
    shutil.copytree(K,tmp_path/'kit');monkeypatch.setattr(runner,'PACKAGE',tmp_path/'kit')
    (tmp_path/'kit'/'protocol_r1.json').write_text('{}')
    with pytest.raises(RuntimeError):runner.verify_kit()

def test_geometry_census_and_real_final_gaps():
    geom=json.loads((K/'fixtures/DEV_GEOMETRY.json').read_text());r0=json.loads((K/'fixtures/R0_EVIDENCE.json').read_text())
    for sub,subgeom in geom['subjects'].items():
        assert len(subgeom['sealed_channels'])==24
        for R in subgeom['runs']:
            pg=r0['subjects'][sub]['R0_d']['events']['10']['runs'][str(R['run'])]['per_gap']
            assert R['G']==len(pg)+1 and len(R['gaps'])==R['G']
            for g,p in zip(R['gaps'][:-1],pg):
                stretches=g['stretches'];assert sum(b-a for a,b in stretches)==p['allowed_samples']
                assert sum((b-a)//10 for a,b in stretches)==p['stride_T']
                assert sum(max(0,b-a-9) for a,b in stretches)==p['stride_1']

def test_design_map_all_keys_valid_code_lines_and_tests():
    import ast
    mapping=json.loads((K/'DESIGN_TO_CODE.json').read_text());keys=[]
    def walk(obj,prefix=''):
        if isinstance(obj,dict):
            for key,value in obj.items():
                path=prefix+'.'+key if prefix else key;keys.append(path);walk(value,path)
    walk(protocol.load());assert set(keys)==set(mapping)
    for v in mapping.values():
        file,line=v['code'].split(':');nodes=ast.parse((K/file).read_text()).body
        assert any(isinstance(n,ast.FunctionDef) and n.name==v['function'] and n.lineno==int(line) for n in nodes)
        file,test=v['test'].split('::');assert f'def {test}(' in (K/file).read_text()
