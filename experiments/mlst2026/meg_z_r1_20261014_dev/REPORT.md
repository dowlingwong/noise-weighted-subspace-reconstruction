# MEG R1 R3 — fresh dev D1/D2 return

**Result: STOP, route A — E1 2/4.** sub-02 and sub-03 pass; sub-01 fails G3; sub-04 fails G2 and G3. No rerun, resume or tuning; R4 not started.

HEAD `2f650bc56d65098f1190d4d6ea2e0ccc7ba2c374`; kit `a1825db94366eb67677818fdeb5e43219de6dd4bfc27acbcafd4472feed45ccf`; old kit `f121776c80c3cd424f1bd14dc1360e8c7930de6f840b34140190f1108e8acdfb`; protocol_r1.json `35a892276b298cc349434b0907bc3e79ab962a3038f9b0ea178c42bde56d1c47`; pytest 264 passed; census exact, T=10, m_hi=768. Python/MNE/NumPy/SciPy 3.12.2/1.8.0/2.2.6/1.15.3; threads 1.

## D1 (preprocessing)

sub-01 66 s, sub-02 38 s, sub-03 35 s, sub-04 35 s; total 174 s. No failures, no G1 retest later, no BAD channels. Derived files in `/ceph/dwong/meg_z/derived_r1_20261018/` (4 h5 + 4 summary JSON). C=24, T=10, evoked_T=20, sfreq 91.6667. Sealed `derived_timingfix_20261013` hashes before = after. No `.pyc`.

| pool | sub-01 | sub-02 | sub-03 | sub-04 | expected (no BAD) |
|---|---:|---:|---:|---:|---:|
| train | 2204 | 2115 | 1972 | 1954 | 2204 / 2115 / 1972 / 1954 |
| cal | 1088 | 1084 | 999 | 987 | 1088 / 1084 / 999 / 987 |
| eval | 1080 | 1069 | 991 | 1000 | 1080 / 1069 / 991 / 1000 |
| ref stride-1 | 21210 | 20997 | 19661 | 19490 | 21210 / 20997 / 19661 / 19490 |

## D2 (gates)

D2 wall 6 s (one gate_cli run; no retest).

| subject | G1 | G2 | G3 | excluded |
|---|---|---|---|---|
| sub-01 | pass | pass (m_train 1.0231) | **fail** (max E_cr 2.4193, eval pos 0, n=135, B=12; f_hat 0.9874) | yes |
| sub-02 | pass | pass (m_train 1.0346) | pass (max E_cr 1.5903; f_hat 1.0009) | no |
| sub-03 | pass | pass (m_train 1.0214) | pass (max E_cr 1.8939; f_hat 1.0080) | no |
| sub-04 | pass | **fail** (m_train 1.4278; m_eval/m_train 0.7175, g_bottom50 0.8466, g_middle40 0.5988, g_top10 0.7495, physical_trace_ratio 0.7360) | **fail** (max E_cr 2.1711, cal pos 0, n=104, B=12; f_hat 1.0007) | yes |

E1: 2 of 4 included (sub-02, sub-03) < 3 → **STOP, route A**.

## Files

`gates_r1.json`, `sub-0{1..4}_summary.json`, `DERIVED_MANIFEST.json`, `INPUT_SHA.txt`, `D1_TIMES.txt`, `D2_TIMES.txt`, `ENVIRONMENT.json`, `RESOLVED_INVOCATION.md`, this `REPORT.md`; hashes in the MS hand-back `ARTIFACT_SHA256.json`. No `.h5`, `.fif`, `.npz` or `.npy`.

## Anomalies and notes

First D2 launcher wrote `START_D2` then exited before starting `gate_cli` because `DER` was unbound in the script; no `gate_cli` process ran and no output file was created. The script was corrected and D2 ran once at 19:29:57Z (6 s). This is a pre-computation launcher fix, not a rerun. The stale tmux session `r1-preflight` (2026-10-14) was left untouched. The D0 `DATA_MANIFEST.json` is the full 449-file/84.7 GB manifest including confirmation subjects; its verify step would hash sub-05..16 and was not run; dev raw files were hashed directly and the 24 FIF + 24 events TSV matched the dev-only R0 artifact (0 mismatches). SSH printed a post-quantum key-exchange warning. No kit file, flag, seed, threshold, path or worker count was changed; nothing was rerun, resumed or tuned.
