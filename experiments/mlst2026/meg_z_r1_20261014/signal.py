"""DESIGN 1.4: FIR bump family; evoked_T remains 20 independently of T."""
import hashlib
import numpy as np
import mne
from experiments.mlst2026.meg_z_20261013 import signal as old
from .protocol import load
FIR_SHA='e093562c8cff0e3cef1ca78e11d7021b3ae3e56bb3c4ab8b6e230d237998fa5f'
EVOKED_T=20

def kernel():
    h=mne.filter.create_filter(None,1100.,1.,40.,method='fir',phase='zero',fir_design='firwin',verbose='error')
    got=hashlib.sha256(np.asarray(h,dtype='<f8').tobytes()).hexdigest()
    if got!=FIR_SHA:raise RuntimeError(f'STOP FIR kernel {got} != {FIR_SHA}')
    return h

def family(center,width,T=10):
    tt=np.arange(-2.,2.,1/1100);u=(tt-center)/width;g=np.exp(-.5*u*u)
    F=np.stack([g,-u/width*g,(u*u-1)/width**2*g],1);h=kernel()
    F=np.stack([np.convolve(F[:,j],h,mode='same') for j in range(3)],1)
    return F[2200:2200+12*T:12]

def temporal_bases():
    cfg=load()['signal'];F1=family(cfg['bump_centers_s'][0],cfg['bump_width_s']);F2=family(cfg['bump_centers_s'][1],cfg['bump_width_s'])
    B1=old.orth(F1);P=F2-B1@(B1.T@F2);Cb=old.orth(P)
    cos=np.linalg.svd(B1.T@old.orth(F2),compute_uv=False);norm=np.linalg.svd(P,compute_uv=False)[-1]/np.linalg.svd(F2,compute_uv=False)[-1]
    if not np.allclose(cos,[.7238,.0867,.0464],atol=1e-3,rtol=0) or abs(norm-.817)>1e-3:raise RuntimeError(f'STOP FIR geometry {cos}, {norm}')
    return B1,Cb,{'principal_cosines':cos.tolist(),'projected_norm':float(norm),'FIR_SHA256':FIR_SHA}

def build(evoked,sfreq,Sref,theta,cfg=None):
    cfg=cfg or load()['signal'];assert evoked.shape[1]==EVOKED_T
    tp=old.spatial_templates(evoked,sfreq,[[.08,.13],[.14,.20]],.95)
    B1,Cb,info=temporal_bases();v=old.base_variances(cfg)
    m0=old.SignalModel(tp['A'],B1,Cb,0.,v);scale=206.604/np.trace(np.linalg.solve(Sref,m0.second_moment()))
    return old.SignalModel(tp['A'],B1,Cb,theta,v*scale),{**info,'template_peaks':tp['peak_times_s'],'degenerate':tp['degenerate'],'variance_scale':float(scale),'trace_K_theta0':206.604}
