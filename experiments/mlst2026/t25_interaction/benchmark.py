"""Production-shaped training-only development benchmark; never exports evidence."""
import json,time,resource
import numpy as np
from .protocol import BASE
from .science import covariance,powers,rng,starts,structured

def main():
 c=BASE;d=c['C']*c['T'];L,_=powers(covariance(c));gen=rng('t25-production-shaped-benchmark-v1')
 x=gen.normal(size=(3200,d))@L;x-=x.mean(0);S=x.T@x/len(x)
 cal=gen.normal(size=(256,d))@L;raw=np.cov(cal,rowvar=False);cov=.9*raw+.1*np.trace(raw)/d*np.eye(d);W=np.linalg.inv(cov)
 initial=starts(S,c['C'],c['T'],c['kc'],c['kt'],'benchmark-only')
 start=time.monotonic();_,diag,_=structured(W,S,initial,150);elapsed=time.monotonic()-start
 print(json.dumps({'development_only':True,'scientific_risk_evaluated':False,'dimension':d,'N':3200,'restarts':4,'iteration_cap':150,'structured_one_metric_seconds':elapsed,'peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'selected_gradient':diag['restarts'][diag['selected']]['gradient_norm'],'statuses':[r['status'] for r in diag['restarts']],'fit_only_extrapolation_seconds':elapsed*8*576,'fit_plus_replay_lower_work_estimate_seconds':elapsed*8*576*2,'not_hardware_qualification':True,'caveat':'one development random-noise covariance fixture, no variance estimate; excludes eigensolves/data I/O/source checks; host benchmark mandatory'},indent=2))
if __name__=='__main__':main()
