"""Shared post-science deadline and Linux process-tree supervision; no numerical imports."""
import argparse,ctypes,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
SEAL_SECONDS=3600
GRACE_SECONDS=60

def effective_deadline(campaign_start,campaign_seconds,post_start=None,seal_seconds=SEAL_SECONDS):
    values=[campaign_start,campaign_seconds,seal_seconds]
    if post_start is not None:values.append(post_start)
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in values) or campaign_start<0 or campaign_seconds<=0 or seal_seconds<=0:
        raise ValueError('invalid deadline values')
    if post_start is not None and post_start<campaign_start:raise ValueError('post-science start precedes campaign')
    return min(campaign_start+campaign_seconds,post_start+seal_seconds) if post_start is not None else campaign_start+campaign_seconds

def post_start(record,campaign_start):
    body=json.loads(Path(record).read_text())
    if body.get('campaign_start_monotonic')!=campaign_start:raise ValueError('post-science campaign binding')
    value=body.get('post_science_start_monotonic')
    effective_deadline(campaign_start,259200,value)
    if value is None or value>time.monotonic():raise ValueError('missing/future post-science start')
    return value

def start_record(record,campaign_start,started=None):
    if started is None:started=time.monotonic()
    effective_deadline(campaign_start,259200,started)
    body={'campaign_start_monotonic':campaign_start,'post_science_start_monotonic':started,'utc_epoch':time.time(),'pid':os.getpid(),'seal_export_hash_limit_seconds':SEAL_SECONDS}
    # Fully materialize outside the output; readers must never observe partial JSON.
    record=Path(record);temp=record.with_name(record.name+'.pending')
    with temp.open('x') as stream:json.dump(body,stream)
    os.link(temp,record);temp.unlink()
    return body

def adopt_descendants():
    if not Path('/proc').is_dir():raise ValueError('Linux process-tree supervision required')
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'cannot become child subreaper')

def processes():
    rows={}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            parts=p.read_text().rsplit(')',1)[1].split()
            rows[int(p.parent.name)]=(int(parts[1]),parts[0],parts[19])
        except (OSError,ValueError,IndexError):continue
    return rows

def process_info(pid):
    try:
        parts=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return int(parts[1]),parts[0],parts[19]
    except (OSError,ValueError,IndexError):return None

def child_pids(pid):
    children=set()
    for task in Path(f'/proc/{pid}/task').glob('*'):
        try:children.update(map(int,(task/'children').read_text().split()))
        except (OSError,ValueError):continue
    return children

class Tree:
    """Track adopted descendants too, even after a leader exits or a child changes session."""
    def __init__(self,process):self.process=process;self.known={};self.observe()
    def observe(self):
        # Avoid an unbounded scan of unrelated host processes at a hard deadline.
        pending=[(os.getpid(),None)];seen=set()
        while pending:
            pid,parent=pending.pop()
            if pid in seen:continue
            seen.add(pid)
            info=process_info(pid)
            if info is None or (parent is not None and info[0]!=parent):continue
            if pid in self.known and self.known[pid]!=info[2]:continue
            if pid!=os.getpid():
                self.known.setdefault(pid,info[2])
            pending.extend((child,pid) for child in child_pids(pid)-seen)
        live=set()
        for pid,identity in self.known.items():
            info=process_info(pid)
            if info is not None and info[2]==identity and info[1]!='Z':live.add(pid)
        return live
    def send(self,sig):
        for pid in self.observe():
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
    def terminate(self,deadline,grace=GRACE_SECONDS):
        until=min(deadline-.25,time.monotonic()+max(0,grace-.25));self.send(signal.SIGTERM)
        while self.observe() and time.monotonic()<until:
            time.sleep(min(.05,max(0,until-time.monotonic())))
        self.send(signal.SIGKILL)
        try:self.process.wait(timeout=max(0,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:raise RuntimeError('process leader survived SIGKILL')
        for pid in list(self.known):
            if pid==self.process.pid:continue
            try:os.waitpid(pid,os.WNOHANG)
            except (ChildProcessError,ProcessLookupError):pass
        return not self.observe()

def wait_tree(tree,deadline_fn,check=lambda:None,grace=GRACE_SECONDS):
    while True:
        tree.observe();deadline=deadline_fn()
        if time.monotonic()>=deadline-grace:raise TimeoutError('deadline termination reserve reached')
        check()
        if tree.process.poll() is not None:
            if tree.observe():raise RuntimeError('process leader exited with live descendants')
            return tree.process.returncode
        time.sleep(min(.1,max(0,deadline-grace-time.monotonic())))

def final_receipt(out,destination):
    from . import workflow as w
    out=Path(out);w.no_cache()
    body={'archive_sha256':w.sha(out/'return.tgz'),'attempt_sha256':w.sha(out/'ATTEMPT.json'),'supervisor_exit_sha256':w.sha(out/'SUPERVISOR_EXIT.json'),'sources':w.sources()}
    w.write(destination,body)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','seconds','finalize']);p.add_argument('--record');p.add_argument('--campaign-start',type=float);p.add_argument('--out');p.add_argument('--destination');a=p.parse_args()
    if a.action=='start':start_record(a.record,a.campaign_start)
    elif a.action=='seconds':
        end=effective_deadline(a.campaign_start,259200,post_start(a.record,a.campaign_start))
        seconds=math.floor(end-time.monotonic()-GRACE_SECONDS)
        if seconds<=0:raise TimeoutError('no sealing time remains inside termination reserve')
        print(seconds)
    else:final_receipt(a.out,a.destination)
if __name__=='__main__':main()
