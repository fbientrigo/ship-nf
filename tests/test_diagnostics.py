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
