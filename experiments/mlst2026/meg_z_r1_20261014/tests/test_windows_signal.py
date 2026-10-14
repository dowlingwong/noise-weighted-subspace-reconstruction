from fractions import Fraction as F
import numpy as np
import pytest,mne
from experiments.mlst2026.meg_z_r1_20261014 import windows as w,signal as s

def test_exact_band_sample_boundaries():
    g=w.gap_geometry(400,[F(0)],fs=F(100))[0]
    assert g['start']==150 and g['length']==96 and g['stretches']==[[0,96]]
    for eps in [F(1,1000000000),-F(1,1000000000)]:
        shifted=w.gap_geometry(400,[eps],fs=F(100))[0]
        assert F(shifted['start'],100)-eps>=F(3,2)
        assert F(shifted['start']+shifted['length']-1,100)-eps<=F(49,20)

def test_next_onset_end_margin_and_final_gap():
    g=w.gap_geometry(300,[F(0),F(2)],fs=100)
    assert len(g)==2 and g[0]['start']+g[0]['length']-1==195
    assert g[1]['length']==0
    g=w.gap_geometry(240,[F(0)],fs=100)[0];assert g['start']+g['length']-1==234

def test_response_exclusion_closed_endpoints_and_bad_mask():
    g=w.gap_geometry(400,[0],[F(2)],fs=100)[0]
    assert g['stretches']==[[0,45]] # 1.95 itself excluded
    mask=np.zeros(400,bool);mask[160:170]=True
    g=w.gap_geometry(400,[0],fs=100,bad_mask=mask)[0];assert g['stretches']==[[0,10],[20,96]]

def test_stride_contiguous_and_mirror():
    geom=[{'gap':g,'start':100*g,'length':88,'stretches':[[0,88]]} for g in range(145)]
    plan=w.window_plan(geom,1)
    assert len([r for r in plan['ref'] if r['gap']==0])==79
    for role in ('train','cal','eval'):
        assert plan[role]
        for r in plan[role]:assert r['rposition']==7-r['position']
        gg=plan[role][0]['gap'];offsets=[r['offset'] for r in plan[role] if r['gap']==gg];assert offsets==list(range(0,80,10))

def test_tiling_restarts_at_allowed_stretch():
    geom=[{'gap':g,'start':100*g,'length':88,'stretches':[[17,88]]} for g in range(145)]
    plan=w.window_plan(geom,1);assert plan['train'][0]['offset']==17
    assert plan['train'][1]['offset']==27

def test_fir_equals_raw_filter_and_geometry():
    h=s.kernel();x=np.zeros((1,22001));x[0,11000]=1
    raw=mne.io.RawArray(x,mne.create_info(['impulse'],1100,'misc'),verbose='error');raw.filter(1.,40.,picks='all',method='fir',phase='zero',fir_design='firwin',n_jobs=1,verbose='error')
    start=11000-(len(h)-1)//2;assert np.max(abs(raw.get_data()[0,start:start+len(h)]-h))<1e-12
    B,C,info=s.temporal_bases();assert np.allclose(B.T@B,np.eye(3));assert np.allclose(B.T@C,0,atol=1e-12)
    assert np.allclose(info['principal_cosines'],[.7238,.0867,.0464],atol=1e-3);assert abs(info['projected_norm']-.817)<1e-3
    assert s.EVOKED_T==20 and B.shape==(10,3)
