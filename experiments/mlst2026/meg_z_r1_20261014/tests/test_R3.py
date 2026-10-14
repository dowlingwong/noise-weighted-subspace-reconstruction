import copy,json
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.meg_z_r1_20261014 import preprocess as p,gate_cli,protocol
from experiments.mlst2026.meg_z_20261013 import preprocess as old
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol

def test_channel_assertion_sealed_lists_and_single_drop():
    geometry=json.loads((protocol.PACKAGE/'fixtures/DEV_GEOMETRY.json').read_text())
    for s,v in geometry['subjects'].items():
        assert len(v['sealed_channels'])==24
        p.assert_channels(v['sealed_channels'],v['sealed_channels'])
        p.assert_channels(v['sealed_channels'][1:],v['sealed_channels'],v['sealed_channels'][:1])
        with pytest.raises(RuntimeError):p.assert_channels(v['sealed_channels'][::-1],v['sealed_channels'])

@pytest.mark.parametrize('module',[p,gate_cli])
def test_R3_cli_confirmation_before_IO(module):
    args=['--subject','sub-05','--bids-root','/no-raw','--out','/no-out'] if module is p else ['--subjects','sub-05','--derived','/no-derived','--out','/no-out']
    with pytest.raises(SystemExit,match='confirmation'):module.main(args)

def test_evoked_T20_and_R3_synthetic_assembly_roundtrip(tmp_path):
    gen=np.random.default_rng(109);fs=1100/12;names=[f'MEG{i:04}' for i in range(30)];runs={};cfg=old.preprocess_config(load_protocol())
    # Fabricated continuous arrays only, never load_run or real raw files.
    for run in range(1,7):
        on=np.arange(24)*3.+1.;data=gen.normal(size=(30,8000))
        runs[run]={'data':data,'sfreq':fs,'ch_names':names,'bads':[],'bad_mask':np.zeros(8000,bool),
            'events':{'stim_onsets_trigger':on,'resp_onsets':np.array([]),'is_face':np.ones(24,bool)},'proc_history_empty':True}
    trials=[]
    for r in (1,2):
        X,_=old.epochs(runs[r]['data'],fs,runs[r]['events']['stim_onsets_trigger']+cfg['trigger_delay_s'],20,cfg['baseline_s'],runs[r]['bad_mask']);trials.append(X)
    ev=np.concatenate(trials).mean(0);sel=old.select_channels(ev,24);expected=[names[i] for i in sel['selected']]
    payload,attrs,windows,meta=p.assemble(runs,cfg,expected,'sub-01')
    assert attrs['T']==10 and attrs['evoked_T']==20 and attrs['channels']==expected
    assert payload['templates']['evoked_faces'].shape==(24,20)
    for role in windows:
        assert windows[role].shape[1:]==(24,10)
        assert np.all(meta[role]['rposition']==7-meta[role]['position'])
    path=tmp_path/'synthetic.h5';old.write_h5(path,payload,attrs);sd=old.load_subject(path,cache=False)
    assert sd.T==10 and sd.evoked.shape[1]==20 and sd.ref.S.shape==(240,240)
    assert np.allclose(sd.ref.S,(old.vec(windows['ref'])-sd.ref.mean).T@(old.vec(windows['ref'])-sd.ref.mean)/sd.ref.n)
