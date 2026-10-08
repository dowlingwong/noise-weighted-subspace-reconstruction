"""Once-only coordinator with shared whole-campaign and post-science hard deadlines."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from . import workflow as w,deadline as d

def coordinate(out):
    w.no_cache()
    c=w.config('confirmation');out=Path(out).resolve();records=Path(str(out)+'.coordinator')
    if os.getenv('ENL_CONFIRMATION_AUTHORISED')!='yes':raise ValueError('confirmation not authorized')
    if out.exists() or records.exists() or Path(w.LEDGER).exists():raise ValueError('permanent output/coordinator/attempt occupied')
    d.adopt_descendants()
    records.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    initial_sources=w.sources();phase=records/'POST_SCIENCE_START.json'
    w.write(records/'COORDINATOR_START.json',{'time':time.time(),'out':str(out),'sources':initial_sources,'deadline_seconds':c['budget']['campaign_seconds'],'pid':os.getpid(),'campaign_start_monotonic':begin,'seal_export_hash_limit_seconds':d.SEAL_SECONDS})
    env=dict(os.environ,ENL_COORDINATOR_PID=str(os.getpid()),ENL_OUTPUT=str(out),ENL_CAMPAIGN_STARTED_MONOTONIC=str(begin),ENL_POST_SCIENCE_RECORD=str(phase))
    tree=None;code=1;reason=None;post=None;finished=None;source_unchanged=False
    def end():
        nonlocal post
        if phase.exists():post=d.post_start(phase,begin)
        return d.effective_deadline(begin,c['budget']['campaign_seconds'],post,seal_seconds=d.SEAL_SECONDS)
    def scratch():
        used=0
        if out.exists():
            for p in out.rglob('*'):
                if time.monotonic()>=end()-d.GRACE_SECONDS:raise TimeoutError('deadline during scratch census')
                if p.is_file():used+=p.stat().st_size
                if used>c['budget']['scratch_bytes']:raise RuntimeError('scratch cap')
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (records/'console.txt').open('xb') as log:
            process=subprocess.Popen(['bash',str(w.PACKAGE/'supervisor.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            tree=d.Tree(process)
            status=d.wait_tree(tree,end,scratch)
            if status!=0:raise RuntimeError('supervisor exit '+str(status))
            if not phase.is_file():raise ValueError('post-science handoff missing')
            end()
            if type(w.read(out/'EXIT.json').get('exit_code')) is not int or w.read(out/'EXIT.json')['exit_code']!=0 or w.read(out/'SUPERVISOR_EXIT.json').get('exit_code')!=0:raise ValueError('scientific/supervisor seal not successful')
            # Hashing and final source/attempt validation run under the same deadline.
            destination=records/'FINAL_ARCHIVE_RECEIPT.json'
            process=subprocess.Popen([sys.executable,'-B','-m','experiments.mlst2026.enl_20261008.deadline','finalize','--out',str(out),'--destination',str(destination)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            tree=d.Tree(process)
            if d.wait_tree(tree,end)!=0:raise RuntimeError('archive/source finalization failed')
            receipt=w.read(destination);source_unchanged=receipt['sources']==initial_sources
            if not source_unchanged:raise ValueError('coordinator source changed')
            finished=time.monotonic()
            if finished>=end():raise TimeoutError('post-science/campaign deadline exceeded')
            code=0
    except BaseException as error:
        reason=repr(error)
        if tree is not None:
            # Stop all descendants, including adopted children after leader exit.
            try:limit=end()
            except (ValueError,OSError):limit=begin+c['budget']['campaign_seconds']
            try:
                if not tree.terminate(limit):reason+='; process tree termination incomplete'
            except BaseException as cleanup:reason+='; cleanup '+repr(cleanup)
    now=time.monotonic()
    body={'exit_code':code,'error':reason,'seconds':now-begin,'out':str(out),'source_unchanged':source_unchanged,
          'campaign_start_monotonic':begin,'post_science_start_monotonic':post,
          'seal_export_hash_seconds':None if post is None else (finished if finished is not None else now)-post,
          'seal_export_hash_limit_seconds':d.SEAL_SECONDS,'deadline_includes_termination_grace':True}
    if code==0:body.update(archive_sha256=receipt['archive_sha256'],attempt_sha256=receipt['attempt_sha256'],supervisor_exit_sha256=receipt['supervisor_exit_sha256'])
    w.write(records/'COORDINATOR_EXIT.json',body)
    if code:raise SystemExit(code)

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);x=a.parse_args();coordinate(x.out)
if __name__=='__main__':main()
