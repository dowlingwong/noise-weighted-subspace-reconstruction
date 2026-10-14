"""DESIGN 1.1–1.3: rational sample clock; allowed stretches; onset-locked fixed grid (rev 2.3, RD-12a)."""
from fractions import Fraction as F
import numpy as np
from .partition import assign_roles

def rational(x):
    return x if isinstance(x,F) else F(str(x))

def ceil(x):
    return -((-x.numerator)//x.denominator)

def gap_geometry(n, onsets, responses=(), fs=F(1100,12), bad_mask=None):
    fs=rational(fs); ons=list(map(rational,onsets)); resp=list(map(rational,responses))
    if ons != sorted(ons) or len(set(ons))!=len(ons):
        raise ValueError('onsets must be strictly increasing')
    out=[]
    for g,on in enumerate(ons):
        lo=max(0,ceil((on+F(3,2))*fs))
        hi=min(n-1, ((on+F(49,20))*fs).__floor__(),
               (((ons[g+1]-F(1,20)) if g+1<len(ons) else F(n-1)/fs-F(1,20))*fs).__floor__())
        # floor, including for negative exact times
        hi=min(hi, ((on+F(49,20))*fs).__floor__())
        length=max(0,hi-lo+1)
        allowed=np.ones(length,dtype=bool)
        for r in resp:
            a=max(0,ceil((r-F(1,20))*fs)-lo); b=min(length-1,((r+F(1,2))*fs).__floor__()-lo)
            if a<=b: allowed[a:b+1]=False
        if bad_mask is not None and length: allowed &= ~np.asarray(bad_mask[lo:hi+1],bool)
        idx=np.flatnonzero(allowed)
        stretches=np.split(idx,np.flatnonzero(np.diff(idx)!=1)+1) if len(idx) else []
        out.append({'gap':g,'start':lo,'length':length,'stretches':[[int(z[0]),int(z[-1])+1] for z in stretches if len(z)]})
    return out

def band_slots(T=10,fs=F(1100,12)):
    """Grid slots of the untruncated band [1.50, 2.45] s (R2-B F6).

    An untruncated band holds floor(0.95 fs) or floor(0.95 fs) + 1 samples, depending on the onset phase; both give
    the same slot count at T = 10 (8). rposition mirrors position within this full band, so a final gap cut short by
    the recording end keeps the mirror.
    """
    fs=rational(fs);n=((F(49,20)-F(3,2))*fs).__floor__();slots={n//T,(n+1)//T}
    if len(slots)!=1:raise ValueError('band slot count depends on onset phase at this T; store per-gap band length')
    return slots.pop()

def window_plan(geometry,run,T=10):
    """Onset-locked fixed grid for train/cal/eval; stride 1 inside allowed stretches for the reference.

    DESIGN rev 2.3 / RD-12a (R2-B F1): train, cal and eval windows sit at band offsets 0, T, 2T, ... from the gap's
    first band sample. A grid window that intersects a response exclusion or a BAD sample is dropped and the grid is
    not restarted, so position k is a fixed latency after the onset (registered-kit convention). rposition is the
    mirror within the full band (F6). A gap contributes at most one window at each position.
    """
    roles=assign_roles(len(geometry),run); nb=max(1,len(geometry)//12); full=band_slots(T)
    plans={r:[] for r in ('ref','train','cal','eval')}
    for g,ro in zip(geometry,roles):
        if ro=='guard': continue
        if ro=='ref':
            offsets=[o for lo,hi in g['stretches'] for o in range(lo,hi-T+1)]
        else:
            ok=np.zeros(g['length'],bool)
            for lo,hi in g['stretches']:ok[lo:hi]=True
            offsets=[T*k for k in range(g['length']//T) if ok[T*k:T*k+T].all()]
        for offset in offsets:
            pos=offset//T
            plans[ro].append({'start':g['start']+offset,'offset':offset,'gap':g['gap'],
                'position':pos,'rposition':full-1-pos,'run':run,'block':min(g['gap']//12,nb-1)})
    return plans
