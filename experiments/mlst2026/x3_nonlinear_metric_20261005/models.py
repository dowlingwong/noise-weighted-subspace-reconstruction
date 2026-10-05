# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Models and SURE for X3, ported from VERITAS ``src/whitebox_subjects``.

* ``FlatAE`` and ``_mlp``: ``whitebox_subjects/ladder.py`` (untied branch; the tied
  linear AE is fitted in closed form instead, see science.py).
* ``hutchinson_divergence`` and ``sure_terms``: ``whitebox_subjects/geometry.py``
  (``hutchinson_divergence``, ``sure_risk``).

Changes from the source: models act on PHYSICAL traces x (the loss carries the metric),
the Hutchinson generator is passed in so a probe stream can be keyed by seed, and the
residual is measured in whitened units with an explicit Sigma^{-1/2}.  For f acting on x
and g(y) = Sigma^{-1/2} f(Sigma^{1/2} y) acting on whitened y, div g = tr(Sigma^{-1/2}
J_f Sigma^{1/2}) = div f, so the divergence is computed on f directly.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

_ACT = {'silu': nn.SiLU}


def _mlp(dims: Sequence[int], activation: str | None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if activation is not None and i < len(dims) - 2:
            layers.append(_ACT[activation]())
    return nn.Sequential(*layers)


class FlatAE(nn.Module):
    """Encoder D -> hidden... -> d, decoder d -> ...hidden -> D (biases on, untied)."""

    def __init__(self, D: int, d: int, hidden: Sequence[int], activation: str | None):
        super().__init__()
        self.encoder = _mlp([D, *hidden, d], activation)
        self.decoder = _mlp([d, *reversed(hidden), D], activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def hutchinson_divergence(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, n_probes: int,
                          generator: torch.Generator) -> torch.Tensor:
    """div fn(x) = tr d fn / dx per sample (Rademacher probes, forward-mode JVPs)."""
    est = torch.zeros(x.shape[0], dtype=x.dtype)
    for _ in range(n_probes):
        v = (torch.randint(0, 2, x.shape, generator=generator) * 2 - 1).to(x.dtype)
        _, jv = torch.func.jvp(fn, (x,), (v,))
        est += (v * jv).flatten(1).sum(-1).detach()
    return est / n_probes


def sure_terms(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, whiten: torch.Tensor, n_probes: int,
               generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Per-sample SURE of the whitened clean risk E||Sigma^{-1/2}(f(x) - clean)||^2 under
    whitened noise N(0, I): ||Sigma^{-1/2}(x - f(x))||^2 + 2 div f(x) - D."""
    with torch.no_grad():
        resid = (((x - fn(x)) @ whiten) ** 2).sum(-1)
    div = hutchinson_divergence(fn, x, n_probes, generator)
    D = x.shape[1]
    return {'sure': resid + 2 * div - D, 'residual': resid, 'divergence': div}
