"""Record source-bound block-E unit checks. No signal, gates or controls run."""
import argparse,json,re,subprocess,os
from pathlib import Path
from .runner import PACKAGE,environment,verify_kit,census_check,atomic_json

def main(argv=None):
    # R2-B F8: host evidence goes to an explicit new path; returned VALIDATION files are never overwritten.
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);out=Path(ap.parse_args(argv).out)
    if out.exists():raise SystemExit(f'STOP {out} exists; write host validation to a new path')
    env=environment();cmd=['python','-m','pytest',str(PACKAGE/'tests'),'-q','-p','no:cacheprovider']
    test=subprocess.run(cmd,text=True,capture_output=True)
    output=test.stdout+test.stderr
    if test.returncode:print(output);raise SystemExit(test.returncode)
    count=int(re.search(r'(\d+) passed',output).group(1))
    git_env=os.environ|{'GIT_OPTIONAL_LOCKS':'0'}
    subprocess.run(['git','diff','--quiet', 'bfe1139d3923f3b38915c3637696d7afbbf65641','HEAD','--','experiments/mlst2026/meg_z_20261013'],env=git_env,check=True)
    old=subprocess.run(['python','-m','experiments.mlst2026.meg_z_20261013.runner','verify-kit'],text=True,capture_output=True,check=True)
    result={'environment':env,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True,env=git_env).strip(),
        'pytest_count':count,'pytest_command':cmd,'pytest_output':output,'old_runner_verify':json.loads(old.stdout),
        'new_runner_verify':verify_kit(),'census':census_check(),'old_kit_git_diff':'unchanged from bfe1139','controls_not_run':True,'real_R3_not_run':True}
    atomic_json(out,result);print(json.dumps(result,indent=1))
if __name__=='__main__':main()
