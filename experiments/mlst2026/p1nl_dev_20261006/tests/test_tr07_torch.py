"""TR07 torch parts: exact affine nesting of the Route-A architecture and an end-to-end fixture run."""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from experiments.mlst2026.p1nl_dev_20261006.tr07_pilot import models, pilot   # noqa: E402


@pytest.mark.parametrize('activation', ['silu', 'relu'])
def test_architecture_represents_rank_k_affine_maps_exactly(activation):
    """u = act(u) - act(-u) for ReLU and SiLU, so hidden width >= 2k carries k coordinates exactly."""
    t = models.import_torch()
    D, k, hidden = 12, 2, [8, 6]
    net = models.build(t, D, k, hidden, activation)
    g = np.random.default_rng(0)
    E = g.normal(size=(k, D))          # encoder map
    Dm = g.normal(size=(D, k))         # decoder map
    c0, b0 = g.normal(size=k), g.normal(size=D)

    def pm(n_in, n_out):               # [I; -I] style pass-through blocks
        return n_in, n_out
    with t.no_grad():
        lin = [mod for mod in net.encoder if isinstance(mod, t.nn.Linear)]
        # layer 1: D -> 8: rows [E; -E] (+ zero rows)
        W1 = np.zeros((8, D)); W1[:k] = E; W1[k:2 * k] = -E
        # layer 2: 8 -> 6: [I -I; -I I] pass-through of the k coordinates
        W2 = np.zeros((6, 8)); W2[:k, :k] = np.eye(k); W2[:k, k:2 * k] = -np.eye(k)
        W2[k:2 * k, :k] = -np.eye(k); W2[k:2 * k, k:2 * k] = np.eye(k)
        W3 = np.zeros((k, 6)); W3[:, :k] = np.eye(k); W3[:, k:2 * k] = -np.eye(k)
        for mod, W, b in zip(lin, (W1, W2, W3), (np.zeros(8), np.zeros(6), c0)):
            mod.weight.copy_(t.tensor(W)); mod.bias.copy_(t.tensor(b))
        dec = [mod for mod in net.decoder if isinstance(mod, t.nn.Linear)]
        V1 = np.zeros((6, k)); V1[:k] = np.eye(k); V1[k:2 * k] = -np.eye(k)
        V2 = np.zeros((8, 6)); V2[:k, :k] = np.eye(k); V2[:k, k:2 * k] = -np.eye(k)
        V2[k:2 * k, :k] = -np.eye(k); V2[k:2 * k, k:2 * k] = np.eye(k)
        V3 = np.zeros((D, 8)); V3[:, :k] = Dm; V3[:, k:2 * k] = -Dm
        for mod, W, b in zip(dec, (V1, V2, V3), (np.zeros(6), np.zeros(8), b0)):
            mod.weight.copy_(t.tensor(W)); mod.bias.copy_(t.tensor(b))
    X = g.normal(size=(500, D)) * 5
    want = (X @ E.T + c0) @ Dm.T + b0
    got = models.predict(t, net, X)
    assert np.max(np.abs(got - want)) < 1e-10 * np.max(np.abs(want))


def test_fixture_pilot_end_to_end(tmp_path):
    out = pilot.run('fixture', tmp_path / 'run')
    cells = sorted(out.glob('cell_*.json'))
    assert len(cells) == 3 * 2                                     # 3 seeds x 2 p x 1 N
    rec = json.loads(cells[0].read_text())
    assert rec['ok'] and np.isfinite(rec['R_AE']) and rec['mcse'] > 0
    assert len(rec['interference']) == 4 and len(rec['recovery']['output_r2']) == 4
    summary = pilot.reduce(out)
    assert summary['verdict']['verdict'].split()[0] in ('PASS', 'FAIL')
    assert 'literal_rule_ceiling_for_sparse_oracle' in summary['verdict']
    assert (out / 'GATE.json').is_file()


def test_pilot_mode_refuses_without_recorded_decisions(tmp_path, monkeypatch):
    monkeypatch.delenv('P1NL_TR07_PILOT_AUTHORISED', raising=False)
    with pytest.raises(SystemExit, match='pilot refused'):
        pilot.run('pilot', tmp_path / 'p')
    assert not (tmp_path / 'p').exists()
