# R1 DMLR manuscript

This is the journal-formatted source for **Failure Dynamics in Scientific
Agents: A Causal Benchmark for Calibrated Early Warning and Recoverability**.
It follows the official DMLR template at the exact commit in `TEMPLATE.lock`.

## Build

```sh
make setup-template
make pdf
```

The setup target downloads `dmlr2e.sty` from the pinned official repository.
The fetched file is intentionally ignored by Git; the lock file and fetch
script are the reproducible dependency record. `latexmk` and a LaTeX
distribution are required.

## Manuscript states

- During drafting and review, keep the standard `\usepackage{dmlr2e}` line and
  real author metadata because DMLR review is single blind.
- Replace the placeholder dates, paper ID, OpenReview URL, and editor only when
  those values are assigned.
- Quantitative results must enter through generated tables or macros under
  `generated/`; do not type result values into prose.
- Keep artifact access, licensing, maintenance, impact, funding, and competing
  interest disclosures synchronized with the release documentation.

The section files are claim-oriented placeholders. Their bracketed text is not
intended for submission and should be replaced with full scientific prose.

