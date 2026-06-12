"""Regression: `import src` must work (README quickstart, paper reproduction)."""


def test_import_src():
    import src

    for name in src.__all__:
        assert callable(getattr(src, name)), name


def test_import_submodules():
    from src import OptimumFilter, PSDCalculator, metrics, make_weights, of  # noqa: F401
    from src.EMPCA import empca_TCY, empca_TCY_optimized, empca_equivalence_utils  # noqa: F401


def test_weight_convention_matches_equivalence_utils():
    import numpy as np

    from src.make_weights import build_of_one_sided_weights as w_pkg
    from src.EMPCA.empca_equivalence_utils import build_of_one_sided_weights as w_utils

    n = 1024
    J = np.linspace(1.0, 2.0, n // 2 + 1)
    np.testing.assert_allclose(w_pkg(J, n), w_utils(J, n))


def test_clip_psd_for_weights():
    import numpy as np

    from src.make_weights import clip_psd_for_weights, make_inverse_psd_weights

    J = np.array([0.0, 1e-30, 1.0, 2.0])
    Jc = clip_psd_for_weights(J)
    assert np.all(Jc > 0)
    w = make_inverse_psd_weights(J)
    assert np.all(np.isfinite(w))
    assert w[0] == 0.0  # DC zeroed
