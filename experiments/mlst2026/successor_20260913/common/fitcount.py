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
and report :func:`observed` as the cell's fit count.  ``common.cli`` then
compares three numbers that no longer share a source: the count declared in the
matrix, the count the cell reports, and the count this module instrumented.

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

_state = threading.local()


class FitCount:
    """The estimator invocations observed inside one :func:`counting` region."""

    __slots__ = ("total", "by_label")

    def __init__(self) -> None:
        self.total = 0
        self.by_label: dict[str, int] = {}

    def add(self, label: str, n: int = 1) -> None:
        n = int(n)
        if n < 0:
            raise ValueError(f"an estimator invocation count must be non-negative, got {n}")
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
def counting():
    """Open a counting region; yields the :class:`FitCount` it fills."""
    prev = current()
    cur = FitCount()
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
