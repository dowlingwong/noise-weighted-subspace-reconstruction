# R3 resolved invocation (written before launch)

Date: 2026-10-18 UTC. Host: Deepthought, `/ceph/dwong/megz_r1_host_20261014`.
Head: `2f650bc56d65098f1190d4d6ea2e0ccc7ba2c374` (= origin/codex/megz-r1-kit-20261014); kit `a1825db…` (31 files); old kit `f121776c…` (27); protocol_r1.json `35a89227…`; pytest 264; census exact, T=10, m_hi=768. Kit unchanged vs 93bf6ab (`git diff --quiet 93bf6ab HEAD -- experiments/mlst2026/meg_z_r1_20261014/`).

## CLI resolution (read-only, from --help and source at HEAD)

- `preprocess.py` (`preprocess.py:58-59`): required `--bids-root`, `--subject`, `--out`; optional `--protocol`. No reviewed-commit / identity / review / controls-acceptance flag. `--protocol` reaches only `guard([subject], a.protocol)`; for the four dev subjects the guard returns immediately, so leaving it at its default (None) has no scientific effect.
- `gate_cli.py` (`gate_cli.py:54-55`): required `--derived`, `--subjects`, `--out`; optional `--bids-root` (needed only for the once-only G1 retest) and `--protocol` (same guard-only use). No reviewed-commit / identity / review / controls-acceptance flag.
- Preprocess scientific defaults load from the registered old-kit `experiments/mlst2026/meg_z_20261013/protocol_dev.json` (SHA-256 `d9da3e62f5d0c3158635b68303e05da1286ff1a6f9a320368c25daf8c63adfe0`) because the R1 CONFIG has no `preprocess` block: magnetometers only, FIR 1–40 Hz zero/firwin, decim 12, trigger delay 0.0345 s, C = 24, evoked_T = 20, baseline [−0.1, 0] s, face codes 5/6/7/13/14/15 from runs 1–2, selection RMS 0–207 ms, reference stride 1. These match DESIGN rev 2.3 §1.1–§1.2 and the R2-B re-check.
- New window/T/partition parameters load from `protocol_r1.json` (CONFIG): `noise_windows.band_s` [1.50, 2.45] s, `before_next_s` 0.05, `end_margin_s` 0.05, response exclusion [−0.05, 0.5]; `windows.gap_geometry` hardcodes exactly these; the old `noise_windows.after_onset_s = 1.2` is explicitly listed as replaced. T = 10 (d = 240), interleaved-block partition, reference stride 1, unshrunk reference metric.
- Gate defaults from `protocol_r1.json`: G1 spatial ≤ 1e6 and full ≤ 1e10 on the unshrunk S_ref, once-only retest; G2-R and the five G2-S statistics in [0.8, 1.25]; G3 strict E_cr < 2.06604 with n ≥ 30 and B ≥ 8, screen f̂ ≤ 2.64; E1 ≥ 3/4 and eval budget ≥ 500. All match DESIGN §2–§5 and CONFIG.
- Channel-selection assertion uses `DEV_GEOMETRY.json` `sealed_channels` (`preprocess.py:16` `assert_channels`); a mismatch is a kit STOP.
- No mapping is ambiguous and no CLI demands the reviewed commit, identity, review file or controls-acceptance file; no STOP is raised by the amendment checks.
- `DATA_MANIFEST.json` (D0) exists at `experiments/mlst2026/meg_z_20261013_dev/DATA_MANIFEST.json`, but it is the full 449-file/84.7 GB dataset manifest whose `split` includes the confirmation subjects. Its `verify` step would hash sub-05..sub-16 and is therefore not run. Dev raw files were hashed directly; the 24 FIF and 24 events TSV match the dev-only `reviews/megz_rootcause_r0_20261013/ARTIFACT_SHA256.json` (0 mismatches), and the 24 channels TSV hashes are recorded in `RAW_SHA256.txt`.

## Resolved commands

D1 (one tmux session, subjects in order):
```
python -m experiments.mlst2026.meg_z_r1_20261014.preprocess \
  --bids-root /ceph/dwong/meg_z/raw/ds000117 --subject <sub-01|sub-02|sub-03|sub-04> \
  --out /ceph/dwong/meg_z/derived_r1_20261018
```
D2 (one tmux session):
```
python -m experiments.mlst2026.meg_z_r1_20261014.gate_cli \
  --derived /ceph/dwong/meg_z/derived_r1_20261018 \
  --subjects sub-01,sub-02,sub-03,sub-04 \
  --bids-root /ceph/dwong/meg_z/raw/ds000117 \
  --out experiments/mlst2026/meg_z_r1_20261014_dev/gates_r1.json
```
Leaving every other option at its default; no flag, threshold, seed, worker count or path is changed.
