import numpy as np

from ship_nf.tail_demo import mass_ratio, px_of, sample_region


def test_mass_ratio_uses_totals_and_leaves_empty_bands_none():
    b = mass_ratio(np.array([0.5, 1.5, 1.5]), np.array([0.5, 0.5, 2.5]), n_ref=100, n_gen=50, edges=[0, 1, 2, np.inf])
    assert [x["ratio"] for x in b] == [(2 / 50) / (1 / 100), 0.0, None]
    assert all(x["ref_total"] == 100 and x["gen_total"] == 50 for x in b)


def test_sample_region_reports_counts_even_when_nothing_is_accepted():
    class M:
        def sample(self, n, seed):
            return np.zeros((n, 5))
    acc, c = sample_region(M(), lambda g: g[:, 0] > 1, n_req=10, seed=0, batch=7, max_batches=3)
    assert len(acc) == 0 and c == {"requested": 10, "generated": 21, "accepted": 0, "max_batches": 3, "batch": 7}


def test_px_of_inverts_features():
    from ship_nf.loop import features
    f = features([{"x": 0, "y": 0, "px": -2.5, "py": 0.3, "pz": 35.0}])
    assert np.isclose(px_of(f)[0], -2.5)
