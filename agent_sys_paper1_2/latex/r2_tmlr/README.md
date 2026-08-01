# R2 TMLR manuscript

This is the journal-formatted source for **Learning When to Verify and How to
Recover: Cost-Sensitive Process Supervision for Scientific Agents**. It follows
the official TMLR template at the exact commit in `TEMPLATE.lock`.

## Build

```sh
make setup-template
make pdf
make anonymity-check
```

The setup target downloads `tmlr.sty`, `tmlr.bst`, and `fancyhdr.sty` from the
pinned official repository. These fetched dependencies are ignored by Git;
the lock file and fetch script are the reproducible record.

## Review and camera-ready modes

- Review uses `\usepackage{tmlr}`. The style hides the author block, but all
  source, PDF metadata, acknowledgments, repositories, supplements, and URLs
  must also be anonymous.
- After acceptance, change the package line to `\usepackage[accepted]{tmlr}`,
  replace the author placeholders, set the month/year and OpenReview URL, and
  restore acknowledgments and author contributions.
- For a de-anonymized preprint, use `\usepackage[preprint]{tmlr}`.
- Quantitative results enter through generated tables or macros under
  `generated/`; never copy result values manually into prose.

`make anonymity-check` is a heuristic guard, not a substitute for inspecting
the source, compiled PDF, metadata, supplemental archive, and anonymous code
release.

