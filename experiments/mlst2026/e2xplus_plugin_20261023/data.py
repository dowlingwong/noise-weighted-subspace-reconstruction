"""E2X+ records: nested training prefixes and a calibration noise pool per (generator, geometry, seed).

G1 uses t25.science.make_data unchanged (namespace replaced).  G2 uses the Y19 generator's draw unchanged;
its calibration pool is the one new data-generating line (mirrors the T25 calibration stream).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.mlst2026.t25_production_20260930 import science as t25
from experiments.mlst2026.y19_20261012 import generator as gen
from experiments.mlst2026.y19_20261012 import predictions as Y19

PACKAGE = Path(__file__).resolve().parent
NAMESPACE = 'e2xplus-20261023'
T25_CONFIG = PACKAGE.parent / 't25_production_20260930' / 'production.json'
M_MAX = 1024

G1_THETAS = (0.0, 0.12, 0.25, 0.40)
G2_PHI = {'control': 0.0, 'A': Y19.REGISTERED_PHI['A'], 'B': Y19.REGISTERED_PHI['B']}
Q1_GRID = (50, 100, 200, 400, 800, 1600, 3200, 6400)
G1_Q2_GRID = (400, 800, 1000, 1250, 1600, 3200)
G2_Q2_GRID = {'A': (729, 1458, 2917, 5834, 11668), 'B': (302, 604, 1208, 2415, 4830)}
G2_Q2_GRID['control'] = tuple(sorted(set(G2_Q2_GRID['A']) | set(G2_Q2_GRID['B'])))


def seed_of(i: int, namespace: str = NAMESPACE) -> int:
    return int.from_bytes(hashlib.sha256(f'{namespace}/seed/{i}'.encode()).digest()[:8], 'little') % 10 ** 12


def geometries() -> list[tuple[str, str]]:
    return [('G1', f'{t:g}') for t in G1_THETAS] + [('G2', lab) for lab in G2_PHI]


def q2_grid(generator: str, geometry: str) -> tuple[int, ...]:
    return G1_Q2_GRID if generator == 'G1' else G2_Q2_GRID[geometry]


def n_max(generator: str, geometry: str) -> int:
    return max(max(Q1_GRID), max(q2_grid(generator, geometry)))


_G2 = None


def g2():
    global _G2
    if _G2 is None:
        _G2 = gen.build(Y19.GENERATOR)
    return _G2


def g1_config(namespace: str) -> dict:
    c = json.loads(T25_CONFIG.read_text())
    c = dict(c, namespace=f'{namespace}/G1', N=[n_max('G1', '0')], m=[M_MAX])
    return c


def draw(generator: str, geometry: str, seed: int, namespace: str = NAMESPACE) -> dict:
    """Returns dims, the true Sigma, the training pool (n_max x d, physical) and the calibration pool (M_MAX x d)."""
    if generator == 'G1':
        c = g1_config(namespace)
        g, sigma, L, Li, calibration, training = t25.make_data(c, seed, float(geometry))
        return {'C': g.C, 'T': g.T, 'kc': g.k_c, 'kt': g.k_t, 'sigma': sigma, 'training': training,
                'calibration': calibration}
    g = g2()
    phi = G2_PHI[geometry]
    training = gen.draw(g, f'{namespace}/G2', seed, phi, n_max('G2', geometry))
    calibration = t25.rng(f'{namespace}/G2', seed, 'calibration').normal(size=(M_MAX, g.d)) @ g.L
    return {'C': g.C, 'T': g.T, 'kc': g.k_c, 'kt': g.k_t, 'sigma': g.sigma, 'training': training,
            'calibration': calibration}
