"""Bounded full-shape resource sample and conservative full-grid projections."""
import argparse,json,math,os,resource,time
from pathlib import Path
from . import workflow as w

def qualify(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    run_timing={};w.run('resource',out/'sample',timings=run_timing);sample=out/'sample';c=w.config('confirmation');rc=w.config('resource')
    rows=[w.read(p) for p in sample.glob('cell_*.json')]
    if len(rows)!=2 or any(len(r['restarts'])!=1 or r['restarts'][0]['fit']['epochs_run']!=2 for r in rows):raise ValueError('bounded qualification census')
    # Independent external invocation demonstrates portable saved-artifact audit too.
    then=time.monotonic();w.verify_receipt(sample);verify_seconds=time.monotonic()-then
    then=time.monotonic();audit_timing={};w.audit(sample,timings=audit_timing);external_audit_seconds=time.monotonic()-then
    measure={r['k']:r for r in rows};train_seconds=0.
    for k,Ns in c['arms'].items():
        row=measure[int(k)];observed=row['restarts'][0]['train_seconds']/2
        base_steps=math.ceil(row['n_train']/c['recipe']['batch'])
        train_seconds+=sum(math.ceil((N-int(.2*N))/c['recipe']['batch'])/base_steps for N in Ns)*observed*c['recipe']['max_epochs']*len(c['p'])*len(c['seeds'])*c['recipe']['restarts']
    # Charge worst measured full-shape non-training overhead for EVERY restart, not just selected.
    per_restart_overhead=max(r['seconds']-r['restarts'][0]['train_seconds'] for r in rows)
    overhead_seconds=per_restart_overhead*c['budget']['max_restarts']
    # Audit measured includes raw-pool loading, all evaluation/probes/gradients and environment hashes.
    audit_seconds=(external_audit_seconds-audit_timing['setup_seconds'])/2*c['budget']['max_restarts']+audit_timing['setup_seconds']
    file_sizes={str(p.relative_to(sample)):p.stat().st_size for p in sample.rglob('*') if p.is_file()}
    pool_bytes=sum(v for k,v in file_sizes.items() if k.startswith('pool_'))
    source_bytes=sum(v for k,v in file_sizes.items() if k.startswith('source_snapshot/'))
    max_restart_bytes=max(sum(v for name,v in file_sizes.items() if name.startswith(f"cell_{r['seed']}_{r['k']}_{r['N']}_{r['p']}") and (f'_restart0' in name or f'_evaluation0' in name)) for r in rows)
    other_bytes=max(sum(v for name,v in file_sizes.items() if name.startswith(f"cell_{r['seed']}_{r['k']}_{r['N']}_{r['p']}") and '_restart' not in name and '_evaluation' not in name) for r in rows)
    receipt_bytes=pool_bytes*len(c['seeds'])+source_bytes+max_restart_bytes*c['budget']['max_restarts']+other_bytes*c['budget']['max_cells']+16*2**20
    # Uncompressed archive bound, even if high entropy defeats gzip; margin covers container headers.
    archive_bytes=receipt_bytes*1.01+16*2**20;scratch_peak=(receipt_bytes+archive_bytes)*1.5
    total_sample_seconds=time.monotonic()-begin
    size_ratio=max(1.,receipt_bytes/max(1,sum(v for k,v in file_sizes.items() if k!='return.tgz')))
    # Measured phases avoid charging fixed bootstrap/runtime verification per restart or
    # counting the sample's internal model audit twice. Three setup charges conservatively
    # cover initial setup plus production's extra pre-import verification and seal imports.
    setup_seconds=run_timing['setup_seconds']*3
    pools_seconds=run_timing['pool_seconds']*len(c['seeds'])
    archive_seconds=(run_timing['export_seconds']+verify_seconds)*size_ratio
    measured_residual=max(0.,run_timing['total_seconds']-run_timing['setup_seconds']-run_timing['pool_seconds']-sum(r['seconds'] for r in rows)-run_timing['internal_audit']['total_seconds']-run_timing['export_seconds'])
    remaining_metadata_seconds=measured_residual*size_ratio
    run_projection=(train_seconds+overhead_seconds+audit_seconds+setup_seconds+pools_seconds+remaining_metadata_seconds)*1.5+1800
    total_projection=run_projection+archive_seconds*1.5+1800
    vm_status=Path('/proc/self/status').read_text() if Path('/proc/self/status').exists() else ''
    vm_peak_kib=int(next((line.split()[1] for line in vm_status.splitlines() if line.startswith('VmPeak:')),'0'))
    body={'kind':'ENGINEERING_ONLY','source_sha256':w.sources(),'runtime':w.read(sample/'START.json')['environment'],'config':c,'resource_config':rc,'resource_namespace':rc['namespace'],'sample_cells':2,'sample_trainings':2,'sample_epochs':4,'confirmation_namespace_claimed':False,'measured':{'rows':rows,'external_audit_seconds':external_audit_seconds,'audit_timing':audit_timing,'run_timing':run_timing,'verify_seconds':verify_seconds,'sample_wall_seconds':total_sample_seconds,'vm_peak_bytes':vm_peak_kib*1024,'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'sample_file_bytes':file_sizes},'projection':{'full300epoch_train_hours':train_seconds/3600,'nontraining_hours':overhead_seconds/3600,'all1728restart_audit_hours':audit_seconds/3600,'setup_hours':setup_seconds/3600,'all24pool_write_hours':pools_seconds/3600,'metadata_hours':remaining_metadata_seconds/3600,'archive_hash_hours':archive_seconds/3600,'run_with_margin_hours':run_projection/3600,'whole_campaign_with_margin_hours':total_projection/3600,'receipt_bytes':receipt_bytes,'archive_bytes':archive_bytes,'scratch_peak_with_margin_bytes':scratch_peak,'time_size_margin':1.5,'fixed_time_reserve_seconds':3600},'limitations':['Two-epoch full-shape engineering measurement is a conservative planning projection, not a mathematical worst-case runtime guarantee.','Whole campaign is operationally stopped at72h; lack of completion preserves an incomplete permanent attempt, never triggers retry.','No primary confirmation truth outcome was inspected.']}
    body['qualified']=total_projection<c['budget']['campaign_seconds'] and run_projection<c['budget']['run_seconds'] and scratch_peak<c['budget']['scratch_bytes'] and vm_peak_kib*1024<c['budget']['virtual_memory_bytes'] and vm_peak_kib>0
    w.write(out/'QUALIFICATION.json',body)
    if not body['qualified']:raise ValueError('full-grid resource projection does not fit approved budget')
    return body

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);x=a.parse_args();print(json.dumps({'qualified':qualify(x.out)['qualified']}))
if __name__=='__main__':main()
