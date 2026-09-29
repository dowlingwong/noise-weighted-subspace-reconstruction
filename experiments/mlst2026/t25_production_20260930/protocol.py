"""Closed candidate settings and exact allocation (no production gate bypass)."""
from pathlib import Path
import json
BASE={'version':'t25/1','C':8,'T':32,'kc':2,'kt':3,'spectrum':{'top':[12.,11.,10.,9.,8.,7.],'gap':4.,'tail_min':.02},'geometry_seed':20260913,'rho_c':.4,'rho_t':.7,'theta':[0.,.12,.25,.40],'N':[400,800,1000,1250,1600,3200],'m':[64,256,1024],'seeds':list(range(202609290100,202609290124)),'namespace':'paper1-t25-v1/confirm','development_only':False,'max_iter':6000,'gradient_tolerance':1e-8,'budget':{'max_cells':576,'max_charges':30536,'max_component_records':650000,'rss_bytes':4*1024**3,'scratch_bytes':40*1024**3,'cell_seconds':600,'campaign_seconds':96*3600,'threads':1,'retries':0}}
FIXTURE={**BASE,'C':3,'T':4,'kc':2,'kt':2,'spectrum':{'top':[6.,5.,4.,3.],'gap':1.5,'tail_min':.05},'theta':[0.,.3],'N':[40,80],'m':[16,32,64],'seeds':[900001,900002],'namespace':'paper1-t25-v1/fixture','development_only':True,'max_iter':50}

def config(path):
    c=json.loads(Path(path).read_text())
    if c not in (BASE,FIXTURE):raise ValueError('settings differ from reviewed closed candidate')
    return c

def enumeration(c):
    rows=[{'seed':s,'theta':th,'N':n,'representation_charges':40,'covariance_charges':6,'reference_charges':7} for s in c['seeds'] for th in c['theta'] for n in c['N']]
    charges=len(rows)*53+len(c['theta'])*2
    component_records=len(c['seeds'])*(2*len(c['theta'])*max(c['N'])+max(c['m']))
    if len(rows)>c['budget']['max_cells'] or charges>c['budget']['max_charges'] or component_records>c['budget']['max_component_records']:raise ValueError('allocation exceeds caps')
    return {'cells':rows,'paired_cells':len(rows),'selected_maps':len(rows)*16,'structured_starts':len(rows)*32,'representation_charges':len(rows)*40,'covariance_charges':len(rows)*6,'reference_charges':len(rows)*7+len(c['theta'])*2,'total_charges':charges,'unique_component_records':component_records,'calibration_recomputations':'same per-seed nested pool recomputed for isolated cells; charged each time, not independent replication','shared_baselines':'iso/oracle fit once per paired cell and referenced at each m'}
