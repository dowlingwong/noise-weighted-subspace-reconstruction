# Anonymized code link — preparation checklist (author decision pending)

The CFP allows links to "properly anonymized code and/or data repositories." This
paper's identity is *measured, not asserted* — a code link is unusually on-brand.
Whether and where to host is the author's call; this is what the snapshot must
contain and must not leak.

## Include

- `src/tidmad/` (training, PSD, scoring, whitening_range, paths), `src/NFPA/`,
  `experiments/synthetic/nfpa_confirmatory/`, `scripts/make_rps_figure.py`,
  `scripts/rung3_whitening_range.py`, `scripts/train_tidmad_local.sh`
- `results/rps2026/` CSV/JSON records the figure and tables are computed from
- A README stating: TIDMAD data are public (CC-BY-4.0, Fry et al. 2024) with the
  download source; one command per figure/table

## Must NOT leak (verified leak vectors in this repo)

- Git history: **fresh export, never `git clone`** — history contains author
  names, emails, and host paths (`/ceph/dwong/...`, `deepthought`)
- `docs/plans/`, `paper_rps2026/plan/`, `paper_rps2026/notes/`,
  `paper_rps2026/sources/`, run logs: contain names (Junjie, Sjostrom,
  Dowling Wong), DELight material, and this checklist itself
- `artifacts/`, `.venv/`, `wandb/`: run metadata embeds usernames and host paths
- Absolute paths in file contents: grep the export for
  `dwong|wongdowling|dowling|ceph|deepthought|delight|kit\.edu|karlsruhe|imperial`
  — must return nothing
- `scripts/train_tidmad_local.sh` and `src/tidmad/paths.py` contain `/ceph/dwong`
  candidate paths — **edit these two files in the export** (replace with
  `/path/to/data` placeholders) or exclude the launcher

## Hosting options

- anonymous.4open.science (purpose-built, auto-expires, common at NeurIPS) —
  recommended
- A fresh anonymous GitHub account works but is manual and easier to get wrong

## Wiring

Add to §4 or a footnote on page 1: `Code and archived results:
\url{https://anonymous.4open.science/r/<id>}` — costs one line; the donor list in
`round8_contingency.md` applies.
