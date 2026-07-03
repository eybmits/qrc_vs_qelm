import numpy as np

from qrc_vs_qelm.core import dissipators as dsp
from qrc_vs_qelm.core import diagnostics as diag
from qrc_vs_qelm.core import reservoirs as res


def test_rate_normalization_mean():
    g = dsp.loguniform_rates(8, 0.1, 10, np.random.default_rng(0), mean=0.5)
    assert np.isclose(g.mean(), 0.5)


def test_degree_rates_uniform_on_dense_graph():
    # dense coupling graph -> every node degree N-1 -> degree profile == uniform
    rng = np.random.default_rng(0)
    J = res.random_couplings(5, 1.0, rng)
    g = dsp.degree_rates(J, mean=0.3)
    assert np.allclose(g, 0.3)


def test_jump_strength_matching():
    rng = np.random.default_rng(1)
    J = res.random_couplings(5, 1.0, rng)
    n = 5
    target = dsp.jump_strength(dsp.local_loss(n, 0.3))
    for fam in [dsp.dephasing(n, 0.3), dsp.thermal(n, 0.3, 0.1),
                dsp.collective_loss(n, 0.3), dsp.exchange(n, 0.3, dsp.graph_edges(J))]:
        normed = dsp.normalize_jump_strength(fam, target)
        assert np.isclose(dsp.jump_strength(normed), target)


def test_unitality_predicts_dephasing_exchange():
    # dephasing (sigma^z) and exchange (sigma^- sigma^+) are unital -> defect 0
    n = 4
    edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
    assert diag.unitality_defect(dsp.dephasing(n, 0.5)) < 1e-12
    assert diag.unitality_defect(dsp.exchange(n, 0.5, edges)) < 1e-12
    # amplitude damping and collective loss are non-unital
    assert diag.unitality_defect(dsp.local_loss(n, 0.5)) > 1e-6
    assert diag.unitality_defect(dsp.collective_loss(n, 0.5)) > 1e-6
