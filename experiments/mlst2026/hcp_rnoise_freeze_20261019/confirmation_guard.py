"""Read-only Git evidence gates. Never constructs a recording path."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse
from experiments.mlst2026.hcp_rnoise_20261019.protocol import load
from .prediction import committed
from .identity import verify

FREEZE='reviews/hcp_rnoise_freeze_20261019'


def git(ms,*args):
    return subprocess.check_output(['git','-C',str(ms),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))


def field(text,name):
    match=re.search(r'^'+re.escape(name)+r':\s*(\S.*?)\s*$',text,re.M)
    if not match: raise PermissionError('STOP missing registration field: '+name)
    return match.group(1)


def registration(ms,record):
    """Require published OSF, tag, bundle and OTS identities plus author authorization."""
    if not re.fullmatch(r'reviews/hcp_rnoise_registration_\d{8}/REGISTRATION.md',record):
        raise PermissionError('STOP registration record path')
    try:
        text=git(ms,'show','origin/main:'+record).decode()
        url=field(text,'OSF URL');parsed=urlparse(url)
        if parsed.scheme!='https' or parsed.hostname!='osf.io' or parsed.path in ('','/'):
            raise PermissionError('STOP OSF registration URL required')
        tag=field(text,'Git tag')
        if not re.fullmatch(r'hcp-rnoise-registration-[0-9]{8}',tag):
            raise PermissionError('STOP registration tag name')
        tagged=git(ms,'rev-parse',f'refs/tags/{tag}^{{commit}}').decode().strip()
        if tagged!=field(text,'Tagged commit'): raise PermissionError('STOP tag commit mismatch')
        git(ms,'merge-base','--is-ancestor',tagged,'origin/main')
        sums=git(ms,'show',tagged+':'+FREEZE+'/REGISTRATION_SHA256SUMS')
        sums_main=git(ms,'show','origin/main:'+FREEZE+'/REGISTRATION_SHA256SUMS')
        if sums!=sums_main or hashlib.sha256(sums).hexdigest()!=field(text,'Bundle SHA-256'):
            raise PermissionError('STOP registered checksum bundle changed')
        # OTS upgrades may change receipt bytes; the current origin/main receipt must
        # match the current registration and authorization hashes. Tag binds original.
        git(ms,'show',tagged+':'+FREEZE+'/REGISTRATION_SHA256SUMS.ots')
        ots=git(ms,'show','origin/main:'+FREEZE+'/REGISTRATION_SHA256SUMS.ots')
        if not ots or hashlib.sha256(ots).hexdigest()!=field(text,'OTS SHA-256'):
            raise PermissionError('STOP OTS receipt identity')
        for line in sums.decode().splitlines():
            digest,path=line.split('  ',1)
            if not re.fullmatch(r'[0-9a-f]{64}',digest) or Path(path).is_absolute() or '..' in Path(path).parts:
                raise PermissionError('STOP invalid checksum entry')
            blob=git(ms,'show','origin/main:'+path)
            if hashlib.sha256(blob).hexdigest()!=digest:
                raise PermissionError('STOP registered file differs: '+path)
        frozen=json.loads(git(ms,'show','origin/main:'+FREEZE+'/protocol_frozen.json'))
        if frozen.get('frozen') is not True: raise PermissionError('STOP protocol not frozen')
        registered_kit=json.loads(git(ms,'show','origin/main:'+FREEZE+'/FREEZE_KIT_SHA256.json'))
        verify(registered_kit)
        auth=git(ms,'show','origin/main:'+str(Path(record).parent/'CONFIRMATION_AUTHORIZATION.md')).decode()
        for key,value in (('OSF URL',url),('Git tag',tag),('OTS SHA-256',field(text,'OTS SHA-256'))):
            if field(auth,key)!=value:raise PermissionError('STOP authorization mismatch')
        if not field(auth,'Author signature').startswith('Dowling Wong;'):
            raise PermissionError('STOP unsigned author authorization')
        return {'OSF_URL':url,'tag':tag,'tagged_commit':tagged,'record':record,'status':'PASS'}
    except (subprocess.CalledProcessError,ValueError,KeyError,UnicodeDecodeError) as exc:
        raise PermissionError('STOP complete registration must exist on origin/main') from exc


def before_fit(ms,registration_record,date,prediction_path,prediction_commit):
    """Mandatory public order gate, called before any confirmation learning kernel."""
    if date not in load()['access']['confirmation_analysis_dates']:
        raise PermissionError('STOP outside fixed 12 confirmation dates')
    reg=registration(ms,registration_record)
    sha=committed(prediction_path,ms,prediction_commit,remote=True)
    record=json.loads(Path(prediction_path).read_text())
    if record.get('date')!=date or record.get('pre_fit') is not True:
        raise PermissionError('STOP prediction is not for this unit before fitting')
    return {**reg,'prediction_sha256':sha,'prediction_commit':prediction_commit}


def fit_confirmation(roles,model,Sref,N,m,arm,date_index,replicate_index,*,
                     ms,registration_record,date,prediction_path,prediction_commit):
    """Array API only. Recording access needs a separately authorized future launcher."""
    before_fit(ms,registration_record,date,prediction_path,prediction_commit)
    from experiments.mlst2026.hcp_rnoise_20261019.learning import fit_cell
    return fit_cell(roles,model,Sref,N,m,arm,date_index,replicate_index)
