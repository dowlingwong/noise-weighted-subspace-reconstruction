"""Existing-process science-to-export handoff, with no numerical imports or retry."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from . import workflow as w,deadline as d

def supervise():
    w.no_cache();c=w.config('confirmation');out=Path(os.environ['ENL_OUTPUT'])
    begin=float(os.environ['ENL_CAMPAIGN_STARTED_MONOTONIC']);record=Path(os.environ['ENL_POST_SCIENCE_RECORD'])
    if os.environ.get('ENL_CONFIRMATION_AUTHORISED')!='yes':raise ValueError('confirmation not authorized')
    lock=Path(w.LEDGER).with_suffix('.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if Path(w.LEDGER).exists() or out.exists():raise ValueError('permanent namespace occupied')
        d.adopt_descendants()
        process=None;tree=None;post=None
        def end():
            phase_start=post
            if record.exists():phase_start=d.post_start(record,begin)
            return d.effective_deadline(begin,c['budget']['campaign_seconds'],phase_start,seal_seconds=d.SEAL_SECONDS)
        def stop(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            # Termination grace is inside the unchanged 71-hour scientific limit.
            command=['timeout','--kill-after=60',str(c['budget']['run_seconds']-d.GRACE_SECONDS),sys.executable,'-B','-m','experiments.mlst2026.enl_20261008.workflow','run','--mode','confirmation','--out',str(out),'--freeze',os.environ['ENL_FREEZE'],'--acceptance',os.environ['ENL_ACCEPTANCE'],'--ledger',w.LEDGER]
            science_begin=time.monotonic()
            process=subprocess.Popen(command,start_new_session=True);tree=d.Tree(process)
            status=d.wait_tree(tree,lambda:min(end(),science_begin+c['budget']['run_seconds']))
            # Capture at the transition, inside this already-running supervisory process.
            post=time.monotonic()
            d.start_record(record,begin,post)
            with Path(str(out)+'.supervisor_exit.json').open('x') as f:json.dump({'exit_code':status},f)
            if (out/'ATTEMPT.json').exists():
                remaining=int(end()-time.monotonic()-d.GRACE_SECONDS)
                if remaining<=0:raise TimeoutError('no export time inside termination reserve')
                process=subprocess.Popen(['timeout','--kill-after=60',str(remaining),sys.executable,'-B','-m','experiments.mlst2026.enl_20261008.workflow','seal','--out',str(out),'--exit-code',str(status)],start_new_session=True)
                tree=d.Tree(process)
                if d.wait_tree(tree,end)!=0:raise RuntimeError('seal/export failed')
            if status:raise SystemExit(status)
        except BaseException:
            if tree is not None:tree.terminate(min(end(),science_begin+c['budget']['run_seconds']) if post is None else end())
            raise

def main():supervise()
if __name__=='__main__':main()
