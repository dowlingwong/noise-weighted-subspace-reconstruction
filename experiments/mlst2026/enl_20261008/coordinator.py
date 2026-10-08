"""Once-only process-tree coordinator, whole-campaign deadline and export receipt."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from . import workflow as w

def coordinate(out):
    w.no_cache()
    c=w.config('confirmation');out=Path(out).resolve();records=Path(str(out)+'.coordinator')
    if os.getenv('ENL_CONFIRMATION_AUTHORISED')!='yes':raise ValueError('confirmation not authorized')
    if out.exists() or records.exists() or Path(w.LEDGER).exists():raise ValueError('permanent output/coordinator/attempt occupied')
    records.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    w.write(records/'COORDINATOR_START.json',{'time':time.time(),'out':str(out),'sources':w.sources(),'deadline_seconds':c['budget']['campaign_seconds'],'pid':os.getpid()})
    env=dict(os.environ,ENL_COORDINATOR_PID=str(os.getpid()),ENL_OUTPUT=str(out))
    process=None;code=1;reason=None
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (records/'console.txt').open('xb') as log:
            process=subprocess.Popen(['bash',str(w.PACKAGE/'supervisor.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                # Reserve 60 seconds for graceful tree termination within 72 hours.
                if time.monotonic()-begin>=c['budget']['campaign_seconds']-60:raise TimeoutError('whole-campaign deadline')
                if out.exists() and w.usage(out)>c['budget']['scratch_bytes']:raise RuntimeError('scratch cap')
                time.sleep(1)
        if process.returncode!=0:raise RuntimeError('supervisor exit '+str(process.returncode))
        if w.read(out/'EXIT.json').get('exit_code')!=0 or w.read(out/'SUPERVISOR_EXIT.json').get('exit_code')!=0:raise ValueError('scientific/supervisor seal not successful')
        # Member hashes are sealed; final archive bytes receive their separate host digest.
        archive_hash=w.sha(out/'return.tgz')
        if time.monotonic()-begin>=c['budget']['campaign_seconds']:raise TimeoutError('archive digest exceeded whole-campaign deadline')
        if w.sources()!=w.read(records/'COORDINATOR_START.json')['sources']:raise ValueError('coordinator source changed')
        code=0
    except BaseException as error:
        reason=repr(error)
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=max(0,min(60,c['budget']['campaign_seconds']-(time.monotonic()-begin))))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    body={'exit_code':code,'error':reason,'seconds':time.monotonic()-begin,'out':str(out),'source_unchanged':w.sources()==w.read(records/'COORDINATOR_START.json')['sources']}
    if code==0:body.update(archive_sha256=archive_hash,attempt_sha256=w.sha(out/'ATTEMPT.json'),supervisor_exit_sha256=w.sha(out/'SUPERVISOR_EXIT.json'))
    w.write(records/'COORDINATOR_EXIT.json',body)
    if code:raise SystemExit(code)

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);x=a.parse_args();coordinate(x.out)
if __name__=='__main__':main()
