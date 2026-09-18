# Sole active T19-S accounting root

From the first launcher invocation against this root onward, **this root is the
only active T19-S accounting root**.

* Original root `/ceph/dwong/mlst-successor-runs-20260913/t19s` — historical
  artifact, unchanged, **not** an accounting root and never resumed.
* Archival copy (local disk, `NOT_A_PRODUCTION_ROOT.md`) — reference bytes only.
* **This root** — carries the 21 original attempt UUIDs verbatim, plus all new
  work. Those historical UUIDs are one set of charges and are **counted once,
  here**. Never add them to the original root's totals.

Carried in: 21 attempts · 20 completed cells · 1 failed attempt
(`5493e66358a9c0cf`, confirmation, 32 planned fits) · 672 charged fits of a
4,800 cap. Identity transition recorded in `TRANSITION.json`; the superseded
identity is preserved in `LAUNCH_ORIGINAL_ARCHIVED.json`.

Once the launcher has been invoked here, this root, its logs and every attempt —
including failed or interrupted attempts with no completed cell — are preserved.
Accounting is never reverted to the original root.
