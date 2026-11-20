from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from ship_nf import flow
from ship_nf.tail_demo import px_of

DEMO = Path(__file__).parents[1] / "docs/meeting_1121"


def test_t5_base_log_density_matches_scipy_and_has_unit_variance():
    m = flow.Flow(5, base="t5")
    z = torch.linspace(-6, 6, 25).reshape(5, 5)
    s = flow.T_SCALE
    want = stats.t.logpdf(z.numpy() / s, 5).sum(1) - 5 * np.log(s)
    assert np.allclose(m.base_log_prob(z).numpy(), want, atol=1e-5)
    g = torch.Generator().manual_seed(0)
    zs = torch.randn(1, 400000, 1, generator=g)[0]  # same construction as Flow.sample
    zs = zs / torch.sqrt((torch.randn(5, 400000, 1, generator=g) ** 2).sum(0) / 5) * s
    assert abs(zs.var().item() - 1) < 0.05
    assert stats.kstest(zs[:, 0].numpy() / s, stats.t(5).cdf).pvalue > 1e-3


def test_default_base_reproduces_the_meeting_checkpoint_samples():
    m = flow.Flow(5)
    m.load_state_dict(torch.load(DEMO / "flow.pt"))
    g = m.sample(50000, 1001)
    assert np.array_equal(px_of(g[np.isfinite(g).all(1)]), np.load(DEMO / "samples.npz")["px_gen"])


def test_same_seed_gives_same_initial_weights_for_both_bases():
    torch.manual_seed(3)
    a = flow.Flow(5).state_dict()
    torch.manual_seed(3)
    b = flow.Flow(5, base="t5").state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_unknown_base_is_refused():
    with pytest.raises(ValueError):
        flow.Flow(5, base="cauchy")
