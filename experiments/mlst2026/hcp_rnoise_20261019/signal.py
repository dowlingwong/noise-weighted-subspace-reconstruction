"""Fixed Sarvas point-sensor forward model, R1 native-FIR bases, fixed latent scale."""
import math
import numpy as np
from scipy.signal import fftconvolve
from experiments.mlst2026.meg_z_20261013.signal import SignalModel, orth, fix_signs, base_variances
from .protocol import load
from .preprocess import selected_geometry, kernel


def sarvas(positions, normals, dipole, orientation, center=(0,0,0)):
    r = np.asarray(positions,float)-center; p = np.asarray(dipole,float)-center
    q = np.asarray(orientation,float); norms = np.linalg.norm(normals,axis=1)
    if np.linalg.norm(p) == 0 or np.any(norms <= 0):
        raise ValueError('degenerate dipole or sensor normal')
    normals = np.asarray(normals,float)/norms[:,None]
    a = np.linalg.norm(r-p,axis=1); rr = np.linalg.norm(r,axis=1)
    if np.any(a <= 0) or np.any(rr <= 0):
        raise ValueError('singular Sarvas geometry')
    ar = rr**2-r@p; F = a*(rr*a+ar)
    grad = (a*a/rr+ar/a+2*(a+rr))[:,None]*r-(a+2*rr+ar/a)[:,None]*p
    cross = np.cross(q,p)
    field = 1e-7*(F[:,None]*cross-(r@cross)[:,None]*grad)/(F[:,None]**2)
    return np.sum(field*normals,axis=1)


def spatial_templates():
    cfg = load()['signal']['forward']; _, rows = selected_geometry()
    r = np.array([c['loc'][:3] for c in rows]); normals = np.array([c['loc'][9:12] for c in rows]); cols=[]
    for p,q in zip(cfg['dipole_positions_device_m'],cfg['dipole_orientations']):
        if np.linalg.norm(np.asarray(p)-cfg['sphere_center_device_m']) >= cfg['sphere_radius_m']:
            raise ValueError('dipole outside fixed sphere')
        z = sarvas(r,normals,p,q,cfg['sphere_center_device_m'])
        if np.linalg.norm(z) <= 0:
            raise ValueError('zero forward template')
        cols.append(fix_signs((z/np.linalg.norm(z))[:,None])[:,0])
    A = np.stack(cols,axis=1)
    if np.linalg.matrix_rank(A)!=2 or abs(A[:,0]@A[:,1])>cfg['maximum_abs_cosine']:
        raise ValueError('degenerate fixed templates; no alternate recipe')
    return A


def temporal_bases():
    cfg = load(); s = cfg['signal']; p = cfg['preprocessing']; T=cfg['windows']['T']
    n = math.ceil(4*p['native_sfreq_hz']); tt=np.arange(-n,n+1)/p['native_sfreq_hz']; h=kernel(); families=[]
    for center in s['bump_centers_s']:
        width=s['bump_width_s']; u=(tt-center)/width; g=np.exp(-.5*u*u)
        F=np.stack([g,-u/width*g,(u*u-1)/width**2*g],axis=1)
        families.append(fftconvolve(F,h[:,None],mode='same',axes=0)[n:n+p['decimation_factor']*T:p['decimation_factor']])
    B1=orth(families[0]); Cb=orth(families[1]-B1@(B1.T@families[1]))
    if not np.allclose(B1.T@Cb,0,atol=1e-10,rtol=0):
        raise RuntimeError('STOP temporal complement rank/orthogonality')
    return B1,Cb


def build(Sref, theta, *, fixed_variances=None):
    cfg=load()['signal']; A=spatial_templates(); B1,Cb=temporal_bases(); v=base_variances(cfg)
    if fixed_variances is None:
        m0=SignalModel(A,B1,Cb,0.,v)
        scale=cfg['trace_K_target']/np.trace(np.linalg.solve(Sref,m0.second_moment()))
        if not np.isfinite(scale) or scale<=0:
            raise ValueError('invalid trace normalization')
        v=v*scale
    else:
        v=np.array(fixed_variances,float,copy=True)
    return SignalModel(A,B1,Cb,float(theta),v)
