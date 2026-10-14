"""DESIGN 1.1–1.3: rational sample clock; contiguous allowed stretches."""
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
        hi=min(n-1, int((on+F(49,20))*fs),
               int(((ons[g+1]-F(1,20)) if g+1<len(ons) else F(n-1)/fs-F(1,20))*fs))
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

def window_plan(geometry,run,T=10):
    """Full-band positions retain physical offsets; backward alignment is the mirror.

    Exclusions restart contiguous tiling at each admissible stretch, as CONFIG requires.
    A gap contributes at most one window at each forward position (checked).
    """
    roles=assign_roles(len(geometry),run); nb=max(1,len(geometry)//12)
    plans={r:[] for r in ('ref','train','cal','eval')}
    for g,ro in zip(geometry,roles):
        if ro=='guard': continue
        seen=set()
        full_slots=g['length']//T
        for lo,hi in g['stretches']:
            for offset in range(lo,hi-T+1,1 if ro=='ref' else T):
                pos=offset//T; rp=full_slots-1-pos
                if ro!='ref' and pos in seen: raise ValueError('multiple windows per gap/position')
                seen.add(pos)
                plans[ro].append({'start':g['start']+offset,'offset':offset,'gap':g['gap'],
                    'position':pos,'rposition':rp,'run':run,'block':min(g['gap']//12,nb-1)})
    return plans
