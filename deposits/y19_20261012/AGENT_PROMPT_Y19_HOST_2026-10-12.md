# Agent prompt: Y19 confirmation run on the host (stage B2)

**Written 12 October 2026** (stage B0); revised the same day after the B1 independent review
(`reviews/y19_preparation_review_20261012_01/INDEPENDENT_REVIEW.md`, findings F1, F2, F6, F7). Use only after
the gates in §0 hold. Placeholders the author fills in before handing this prompt over:
`<FREEZE_COMMIT>` (the EXP commit that contains the frozen kit), `<MS_FREEZE_COMMIT>` (the MS commit that
contains the Y19 protocol, this prompt and the preparation record), `<ZENODO_DOI>` (the timestamped deposit),
`<HOST>` (chione or kalinka).

```
Role: execution agent for stage B2 (Y19 confirmation run) of
plan/onthefly/RUNBOOK_MLST_MULTICHANNEL_UPGRADE_2026-10-12.md.
Read first: MS AGENTS.md; the runbook §2 and §4; the frozen protocol MS plan/Y19_PROTOCOL_2026-10-12.md at
<MS_FREEZE_COMMIT>; EXP experiments/mlst2026/y19_20261012/README.md at <FREEZE_COMMIT>.
Allowed writes (nothing else): the fresh clone ~/y19_host; the virtual environment ~/y19_venv; the logs
~/y19_pytest.txt, ~/y19_run_stdout.txt, ~/y19_run_stderr.txt; the system temporary directory used by pytest;
inside the clone only experiments/mlst2026/y19_20261012_return/ (including your HOST_REPORT.md there) and one
git branch y19-return-20261012 pushed to origin.
Forbidden: any change to the kit (experiments/mlst2026/y19_20261012/), to any other EXP or MS file, to seeds,
grid, geometry, learners or tolerances; reruns of a completed run; deleting or editing any record or log;
`runner summarize --partial` or opening record/summary contents while the run is in progress; running the
kit on any other generator or namespace; plan/canonical/; frozen records; other review folders; merging
into main.
Stop and report instead of improvising if any gate fails or any hash differs.
```

## 0. Gates that must already hold (check, do not create)

1. The B1 review folder `reviews/y19_preparation_review_20261012_01/` (MS) exists with verdict ACCEPTED or
   ACCEPTED WITH MINOR CHANGES, and the author has confirmed that its fixes are in.
2. The author has committed the kit: `git -C ~/y19_host show --stat <FREEZE_COMMIT>` lists
   `experiments/mlst2026/y19_20261012/` including `KIT_SHA256.json`; the MS Y19 files are committed at
   `<MS_FREEZE_COMMIT>`.
3. The Zenodo deposit `<ZENODO_DOI>` exists and contains the files listed in MS
   `reviews/y19_preparation_20261012/ZENODO_PACKAGE.md`. Expected: `KIT_SHA256.json` SHA-256
   `8d8ac8441feaee4b28cfc46637bfbd06813c0131a1f7578967a21c59274909d9`, kit identity
   `bbf4e8c64bfb1126bcccb8645a18e78196d9e03c5a2ea1f8117e2464eaa2bfe8`. If the author re-froze the kit, the
   Zenodo copy is authoritative and replaces these values.
4. No Y19 return exists yet. After the clone in §1, all three must hold (empty output / true):
   ```bash
   git ls-remote origin 'refs/heads/y19-return*'                                      # empty
   git log --all --oneline -- experiments/mlst2026/y19_20261012_return | wc -l        # 0
   test ! -e experiments/mlst2026/y19_20261012_return && echo absent                  # absent
   ```
   If any fails, STOP: the run has already happened or was started.

## 1. Fresh checkout on `<HOST>`

```bash
git clone https://github.com/dowlingwong/noise-weighted-subspace-reconstruction.git ~/y19_host
cd ~/y19_host
git checkout --detach <FREEZE_COMMIT>
test "$(git rev-parse HEAD)" = "<FREEZE_COMMIT>"
test -z "$(git status --porcelain)"
test "$(find experiments src -name '*.pyc' | wc -l)" = 0
nproc; free -g; df -h .            # need >= 4 cores, >= 4 GB RAM, >= 1 GB disk
```

Then run the gate-4 commands above.

## 2. Environment (record every version in the host report)

```bash
python3 -m venv ~/y19_venv && . ~/y19_venv/bin/activate
pip install numpy==2.2.6 scipy==1.15.3 h5py pytest      # versions used for the predictions and the B1 review
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
unset PYTHONPYCACHEPREFIX
python -c "import numpy, scipy, h5py; print(numpy.__version__, scipy.__version__, h5py.__version__)"
```

If numpy 2.2.6 / scipy 1.15.3 cannot be installed on the host, use the closest available versions, record
them, and rely on the preflight (population recomputation within 1e-6). Do not change anything else.

## 3. Verify the kit (STOP on any mismatch)

```bash
sha256sum experiments/mlst2026/y19_20261012/KIT_SHA256.json      # = Zenodo copy (gate 3)
python -m experiments.mlst2026.y19_20261012.runner verify-kit    # kit_identity = gate 3 value, dependencies ok
python -m pytest experiments/mlst2026/y19_20261012/tests -q -p no:cacheprovider 2>&1 | tee ~/y19_pytest.txt
                                                                 # 14 passed; the tests never draw G2 records
```

The tests write only to the system temporary directory. If a test fails, STOP and report the output.

## 4. Run once

```bash
nohup python -m experiments.mlst2026.y19_20261012.runner run \
      --out experiments/mlst2026/y19_20261012_return --workers 4 \
      > ~/y19_run_stdout.txt 2> ~/y19_run_stderr.txt &
sleep 90
cat ~/y19_run_stderr.txt                                          # must not contain KitMismatch or a traceback
test -f experiments/mlst2026/y19_20261012_return/START.json && echo started
ps aux | grep -c "[y]19_20261012.runner run"                      # >= 1 while running
```

- The preflight re-checks the kit hash, the pinned dependency bytes (accepted T25 closure `01f2f82f…`),
  thread variables, the seed derivation and a population recomputation of `predictions.json` (relative
  tolerance 1e-6) **before any record is drawn**. A `KitMismatch` (exit code 3) or any traceback in the first
  check means STOP and report; `START.json` is written only after the preflight passed.
- Expected: 480 records, roughly 10–30 min wall with 4 workers (worst case a few hours). While it runs, monitor
  progress **only by counting**: `wc -l < experiments/mlst2026/y19_20261012_return/RUN_LOG.txt` and
  `find experiments/mlst2026/y19_20261012_return/records -name '*.json' | wc -l`. Do not open records, do not
  run `summarize` (with or without `--partial`).
- If the process dies (host reboot, OOM), resume with **the identical command**. Existing records are never
  recomputed or overwritten; `ATTEMPTS.jsonl` logs every start. Do not delete anything. Report every interruption.
- Never start a second run after `summary.json` with `"status": "COMPLETE"` exists.

## 5. Check completion and state the reading (no interpretation beyond the frozen rule)

```bash
R=experiments/mlst2026/y19_20261012_return
python - <<'PY'
import json; s = json.load(open('experiments/mlst2026/y19_20261012_return/summary.json'))
print(s['status'], s['records'], s['family_size'], 'machine overall:', s['overall'])
r = s['readings']
for k in ('A', 'B', 'control'): print(k, r[k].get('pattern', r[k]['decisions']), r[k]['reading'])
ab = r['A']['reading'] == 'MET' and r['B']['reading'] == 'MET'
print('reported overall (protocol §7):', 'NOT MET' if not ab else ('MET' if r['control']['reading'] == 'MET' else 'MET WITH FAILED CONTROL'))
print('optimization-limited records:', s['optimization_limited_records_total'])
for k in ('A', 'B'): print(k, 'optimization-limited deciding cells:', r[k]['deciding_cells_optimization_limited'])
PY
sha256sum $R/summary.json $R/results.h5 $R/START.json
python -m experiments.mlst2026.y19_20261012.runner verify-kit     # kit unchanged after the run
```

Required: status COMPLETE, records 480, family_size 20. Copy every printed line verbatim into the host report.

## 6. Logs, host report, commit (results are force-added: EXP's .gitignore ignores *.h5 and *.npz)

```bash
cp ~/y19_pytest.txt     $R/pytest_output.txt
cp ~/y19_run_stdout.txt $R/run_stdout.txt
cp ~/y19_run_stderr.txt $R/run_stderr.txt
find $R -name '*.tmp*'                                           # list any leftover temp file in the report; do not delete
```

Write `$R/HOST_REPORT.md` with: host name, `<FREEZE_COMMIT>`, `<MS_FREEZE_COMMIT>`, `<ZENODO_DOI>`, kit identity,
Python/numpy/scipy/h5py versions, BLAS (`python -c "import numpy; numpy.show_config()"`), start/end time (UTC),
wall time, number of attempts, the §5 output verbatim (machine overall and the reported overall reading,
patterns A/B, control, optimization-limited counts), SHA-256 of `summary.json`, `results.h5`, `START.json`, the
committed file counts below, and any anomaly. No interpretation, no comparison with other studies.

```bash
git switch -c y19-return-20261012
git add -f $R
git status --porcelain --ignored $R | grep -v '^A ' | wc -l      # must be 0: nothing ignored, nothing unstaged
git commit -m "Y19 confirmation return (kit bbf4e8c6, freeze <FREEZE_COMMIT>)"   # add the session's attribution lines
git ls-files $R/records | grep -c '\.json$'                       # 480
git ls-files $R/arrays  | grep -c '\.npz$'                        # 480
for f in START.json ATTEMPTS.jsonl RUN_LOG.txt summary.json results.h5 HOST_REPORT.md \
         pytest_output.txt run_stdout.txt run_stderr.txt; do git ls-files --error-unmatch $R/$f >/dev/null && echo "ok $f"; done
git ls-files $R | wc -l                                           # 969 (+ any listed temp files)
git push origin y19-return-20261012
```

If any count differs, do not push; report. The host report must state the committed counts (480 JSON, 480 NPZ,
9 run files, total). Because `HOST_REPORT.md` is written before the commit, record the counts from
`find $R -type f` there and confirm them with the `git ls-files` output in your final message.
Commit only the return directory. Do not merge into main and do not touch the kit; the receiving stage (B3,
independent reviewer, who re-runs `verify-kit` before reading) and the author handle the merge. Report the
pushed commit hash.
