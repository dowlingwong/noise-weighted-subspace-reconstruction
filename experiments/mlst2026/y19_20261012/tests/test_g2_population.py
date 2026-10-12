"""G2 population checks (data-free: no record is drawn and no learner is fitted)."""
import json
from pathlib import Path

import numpy as np

from experiments.mlst2026.y19_20261012 import generator as gen, predictions as P, runner

KIT = Path(__file__).resolve().parents[1]


def test_separable_at_phi0_to_1e12():
    g = gen.build(P.GENERATOR)
    K = g.K(0.0)
    w, U = np.linalg.eigh(K)
    Uk = U[:, np.argsort(w)[::-1][:g.k]]
    PU = Uk @ Uk.T
    P4 = PU.reshape(g.C, g.T, g.C, g.T, order='F')
    Uc = np.linalg.eigh(np.einsum('ctdt->cd', P4))[1][:, -g.k_c:]
    Ut = np.linalg.eigh(np.einsum('ctcs->ts', P4))[1][:, -g.k_t:]
    assert np.linalg.norm(PU - np.kron(Ut @ Ut.T, Uc @ Uc.T)) <= 1e-12
    # the phi = 0 product factors of the construction span the same subspace
    Pf = np.kron(g.U_t0 @ g.U_t0.T, g.U_c0 @ g.U_c0.T)
    assert np.linalg.norm(PU - Pf) <= 1e-12
    # physical templates at phi = 0 are exactly vec(q_i p_j^T)
    B = g.templates(0.0)
    assert np.max(np.abs(B - np.kron(g.pulses(0.0), g.Q))) <= 1e-12
    m = gen.PopulationModel(g, 0.0)
    cert = P.certificate(m)
    assert cert['B_cert'] <= 1e-12 and cert['separability_residual_fro'] <= 1e-12


def test_spectrum_invariant_and_nonseparable():
    g = gen.build(P.GENERATOR)
    for phi in (0.0, P.REGISTERED_PHI['A'], P.REGISTERED_PHI['B']):
        assert np.max(np.abs(np.sort(np.linalg.eigvalsh(g.K(phi)))[::-1] - g.mu)) <= 1e-12
    assert g.Delta_k > 0
    for phi in (P.REGISTERED_PHI['A'], P.REGISTERED_PHI['B']):
        assert P.certificate(gen.PopulationModel(g, phi))['B_cert'] > 0.1


def test_predictions_and_protocol_consistent():
    pred = json.loads((KIT / 'predictions.json').read_text())
    proto = json.loads((KIT / 'protocol.json').read_text())
    assert pred['all_checks_pass']
    assert P.build_protocol(pred, proto['dependencies']) == proto
    assert pred['phi_selection']['phi_A'] == P.REGISTERED_PHI['A'] and pred['phi_selection']['phi_B'] == P.REGISTERED_PHI['B']
    for lab in ('A', 'B'):
        r = pred['geometries'][lab]
        assert 300 <= r['N_star_pred'] <= 3000
        assert all(r['checks'].values())
        assert r['n_starts'] >= 50 and r['all_starts_reach_best']
        assert pred['grids'][lab] == [P.round_half_up(r['N_star_pred'] * f) for f in P.FACTORS]
        lo, hi = r['N_star_interval']
        assert abs(lo - r['N_star_pred']) < 1e-9 and hi > lo
    assert pred['ratio_N_star_A_over_B'] >= 2
    assert pred['grids']['control'] == sorted(set(pred['grids']['A']) | set(pred['grids']['B']))
    assert proto['inference']['family_size'] == 20 and proto['inference']['df'] == 23
    assert len(proto['seeds']) == 24


def test_population_recomputation_matches_predictions():
    pred = json.loads((KIT / 'predictions.json').read_text())
    cmp_ = runner.compare_predictions(pred, P.compute(scan=False))
    assert cmp_['ok'], cmp_
