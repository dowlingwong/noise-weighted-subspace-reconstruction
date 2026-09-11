# Return package — Paper 1 upgrade round, 11 September 2026 (server)

Private transfer package, committed into this repository on 11 September 2026.
**This is not an arXiv release package**: the release gate is still held
(`MS:RELEASE_GATE.json`, `decision_received: false`), `\gatedevidencefalse` in
every entry point, and `MS:scripts/package_arxiv.py` still refuses to write a
tarball.

## The transfer mechanism is the remotes; this package is the fallback

Both branches are on their origins, so the normal way to get this work is:

```bash
git -C <MS>  fetch origin upgrade/remote-2026-09-11
git -C <MS>  checkout upgrade/remote-2026-09-11      # manuscript
git -C <EXP> fetch origin mlst/t22-2026-09-11
git -C <EXP> checkout mlst/t22-2026-09-11            # experiments
```

The bundles and patches below exist for the case where the histories are not
shared. **They are a snapshot taken immediately before the commit that added
this directory**, so each is one commit behind its branch tip; the branches on
the remotes are authoritative.

| item | size | what |
|---|---|---|
| `ms-upgrade-remote.bundle` | 9.0 MB | manuscript, `e0b130e..6fbde0c`, branch `upgrade/remote-2026-09-11` (7 commits) |
| `ms-patches/` | 25 MB | the same seven commits as `git am`-able patches |
| `exp-t22.bundle` | 1.8 MB | experiments, `a7125ea..5d731d0`, branch `mlst/t22-2026-09-11` (3 commits) |
| `exp-patches/` | 6.6 MB | the same three commits as patches |
| `pdfs/` | 2.3 MB | `main.pdf` 116 pp · `main_mlst.pdf` 26 pp · `supplement_mlst.pdf` 50 pp · `supplement_configuration.pdf` 15 pp |
| `REMOTE_UPGRADE_HANDOFF_2026-09-11.md` | 37 kB | the handoff, also at `MS:reviews/` |
| `SHA256SUMS` | — | checksums of everything above |

**The 31.6 MB of patch directories is redundant** — they duplicate the bundles,
which duplicate the repositories, which are now on their remotes. `ms-patches/`
is large because the evidence-migration commit carries the whole 21 MB
`MS:evidence/` tree base64-encoded. Delete both directories if the remotes are
reachable:

```bash
git rm -r transfer/mlst_upgrade_return_2026-09-11/ms-patches \
          transfer/mlst_upgrade_return_2026-09-11/exp-patches
```

## Apply from the fallback

```bash
git -C <MS>  fetch /path/to/ms-upgrade-remote.bundle \
    upgrade/remote-2026-09-11:upgrade/remote-2026-09-11
git -C <EXP> fetch /path/to/exp-t22.bundle \
    mlst/t22-2026-09-11:mlst/t22-2026-09-11
```

or, if the histories are not shared, `git am ms-patches/*.patch` and
`git am exp-patches/*.patch`.

## No result file needs copying

Both repositories track their own results: the T22 run archive
(`EXP:results/t22_2026-09-11/`, 23 files, 6.0 MB, including the two bootstrap
draw `.npz` arrays that `MS:scripts/recompute_t22.py` reads) and the manuscript
evidence root (`MS:evidence/`, 259 files, 21 MB). No raw instrument data is
included: the TIDMAD (171 GB), CRESST (2.3 GB) and GWOSC (21 MB) releases stay on
the server and in their public homes, and the manuscript archives their checksums
and source locations rather than their bytes.

Full instructions, checks and open questions:
`MS:reviews/REMOTE_UPGRADE_HANDOFF_2026-09-11.md` (copied here).
