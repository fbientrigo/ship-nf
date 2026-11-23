"""ESS, Pareto k-hat and coverage on cases with a known answer."""

import numpy as np
import pytest

pytest.importorskip("scipy")
from ship_nf.diagnostics import coverage, kish_ess, psis_khat, tail_mass_ratio  # noqa: E402


def test_kish_ess_limits():
    assert kish_ess(np.ones(100)) == pytest.approx(100)
    assert kish_ess(np.r_[1.0, np.zeros(99)]) == pytest.approx(1)


@pytest.mark.parametrize("shape", [0.3, 0.5, 0.8])
def test_khat_recovers_pareto_shape(shape):
    # Pareto with tail index 1/shape has GPD shape k = shape
    w = np.random.default_rng(0).pareto(1 / shape, 20000) + 1
    assert psis_khat(np.log(w)) == pytest.approx(shape, abs=0.1)


def test_khat_small_for_bounded_weights():
    assert psis_khat(np.log(np.random.default_rng(1).uniform(1, 2, 5000))) < 0


def test_coverage_full_vs_missing_tail():
    rng = np.random.default_rng(2)
    real = rng.normal(size=(2000, 2))
    assert coverage(real, rng.normal(size=(4000, 2))) > 0.9
    shrunk = 0.3 * rng.normal(size=(4000, 2))  # a model that lost the tails
    tail = real[np.linalg.norm(real, axis=1) > 2]
    assert coverage(tail, shrunk) < 0.3


def test_tail_mass_ratio():
    assert tail_mass_ratio(np.r_[True, False, False, False], np.r_[True, True, False, False]) == pytest.approx(0.5)


def test_sliced_wasserstein_shift_joint_and_seed():
    from scipy.stats import wasserstein_distance

    from ship_nf.diagnostics import sliced_wasserstein as sw

    rng = np.random.default_rng(1)
    n, d = 2000, 5
    a, b = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    null = sw(a, b)
    assert sw(a, b) == null  # fixed seed: same directions, same value
    shifted = b + 0.5 * np.eye(d)[0]  # 0.5 sigma shift of one coordinate
    assert sw(a, shifted) > 2 * null
    # joint change with identical marginals: correlation +0.9 vs -0.9 in 2-D
    z = rng.standard_normal((2, n, 2))
    pos = np.c_[z[0, :, 0], 0.9 * z[0, :, 0] + np.sqrt(1 - 0.81) * z[0, :, 1]]
    neg = np.c_[z[1, :, 0], -0.9 * z[1, :, 0] + np.sqrt(1 - 0.81) * z[1, :, 1]]
    ref = np.c_[z[1, :, 0], 0.9 * z[1, :, 0] + np.sqrt(1 - 0.81) * z[1, :, 1]]
    marg = max(wasserstein_distance(pos[:, j], neg[:, j]) for j in range(2))
    assert marg < 0.1  # marginals look the same
    assert sw(pos, neg) > 4 * sw(pos, ref)
    with pytest.raises(ValueError):
        sw(a, b[:-1])
