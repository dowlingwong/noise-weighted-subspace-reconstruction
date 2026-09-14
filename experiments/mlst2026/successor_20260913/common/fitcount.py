"""Observed estimator-invocation accounting.

The round's accounting rule (``README.md``, "Accounting") is that one *fit* is
one estimator invocation on drawn data: each restart counts, each learner
counts, each rank counts.  Every cell of a run matrix declares the ``n_fits``
it will charge, and ``common.cli`` refuses a cell whose reported count differs
from the declared one.

That guard is only a check if the reported count is **observed at the estimator
call sites** rather than restated from the same constant that produced the
declaration.  This module is that observation.  A cell run is wrapped in a
:func:`counting` region by ``common.cli``; the packages call :func:`count_fit`
(or use the :func:`counted` decorator) where an estimator is actually invoked,
and report the cell's fit count.  ``common.cli`` then compares three numbers:
the count declared in the matrix, the count the cell reports, and the count this
module instrumented.

Only two of those three are independent, and the load-bearing comparison is the
one against the **matrix declaration**.  Packages differ in how they produce the
reported count: ``t19s``, ``t21s`` and ``t22s`` build it from their own per-cell
bookkeeping, so all three numbers have separate sources there; ``t20s`` and
``cresst_s`` return :func:`observed` itself as the reported count, which makes
the reported-versus-instrumented half of the check a tautology in those two
packages.  What still bites in every package is that the instrumented count must
equal the number the matrix declared before the cell ran — that is the number a
scientific claim is priced and capped against, and it comes from a different
source in all five packages.  Do not weaken the reported-versus-instrumented
check on the strength of that: it is what catches a package that starts
restating a constant.

Counting is per-thread and re-entrant.  Outside a :func:`counting` region
:func:`count_fit` is a no-op, so an instrumented estimator called from a test,
a reducer or an interactive session records nothing and raises nothing.

Fits only.  Algebraic/analytic *solves* (which fit no drawn data) keep their own
per-package counters where the packages already increment them.
"""
from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from .results import SchemaError

_state = threading.local()


class FitCount:
    """The estimator invocations observed inside one :func:`counting` region."""

    __slots__ = ("total", "by_label", "limit")

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.total = 0
        self.by_label: dict[str, int] = {}

    def add(self, label: str, n: int = 1) -> None:
        n = int(n)
        if n < 0:
            raise ValueError(f"an estimator invocation count must be non-negative, got {n}")
        if self.limit is not None and self.total + n > self.limit:
            raise SchemaError(f"estimator fit limit exceeded before invocation: {self.total} + {n} > {self.limit}")
        self.total += n
        self.by_label[label] = self.by_label.get(label, 0) + n

    def as_dict(self) -> dict:
        return {"total": int(self.total), "by_label": dict(sorted(self.by_label.items()))}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FitCount(total={self.total}, by_label={self.by_label})"


def current() -> FitCount | None:
    """The innermost active counting region, or ``None`` outside one."""
    return getattr(_state, "current", None)


@contextmanager
def counting(limit: int | None = None):
    """Open a counting region; yields the :class:`FitCount` it fills."""
    prev = current()
    cur = FitCount(limit=limit)
    _state.current = cur
    try:
        yield cur
    finally:
        _state.current = prev


def count_fit(label: str = "fit", n: int = 1) -> int:
    """Record ``n`` estimator invocations at this call site.  Returns ``n``."""
    c = current()
    if c is not None:
        c.add(label, n)
    return int(n)


def count_completed(label: str, n: int) -> int:
    """Record a legacy batch AFTER it returns, retaining actual overruns.

    Unlike count_fit this cannot prevent work already done inside the batch.
    Refuse further work and preserve the full count for failed-attempt charging.
    """
    c = current()
    n = int(n)
    if n < 0:
        raise ValueError("negative completed count")
    if c is not None:
        c.total += n
        c.by_label[label] = c.by_label.get(label, 0) + n
        if c.limit is not None and c.total > c.limit:
            raise SchemaError(f"accounting mismatch: completed legacy batch used {c.total} fits, limit {c.limit}")
    return n


def require() -> FitCount:
    """The active counting region, or a clear error if there is none."""
    c = current()
    if c is None:
        raise RuntimeError("no fitcount counting region is active; cell runners are wrapped "
                           "in one by common.cli._run, so report the observed count from there")
    return c


def observed() -> int:
    """Total invocations recorded in the active region.

    Raises outside a region: a cell runner that reports this value must be run
    inside one, and silently reporting ``0`` would reintroduce exactly the
    defect this module exists to remove.
    """
    return int(require().total)


def counted(label: str | None = None):
    """Decorator: count one estimator invocation per call of ``fn``."""
    def deco(fn):
        lab = label or getattr(fn, "__name__", "fit")

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            count_fit(lab)
            return fn(*args, **kwargs)

        inner.__fitcount_label__ = lab
        return inner
    return deco
