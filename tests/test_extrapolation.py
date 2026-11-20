"""Systems 1-2 on synthetic data: the two split versions, and the study's report shape."""

import numpy as np
import pytest

pytest.importorskip("scipy")
from ship_nf.extrapolation import pt_z, score, split, study  # noqa: E402


def fake_afterms(n=3000, seed=0):
    r = np.random.default_rng(seed)
    px, py = r.normal(0, 1, n), r.normal(0, 1, n)
    return np.column_stack([px, py, r.uniform(10, 100, n), r.normal(0, 50, n), r.normal(0, 50, n),
                            np.full(n, 28.9), np.full(n, 13.0), np.ones(n)])


def test_box_contains_radial_at_the_same_k():
    z, _ = pt_z(*np.random.default_rng(1).normal(size=(2, 5000)))
    inside_r, inside_b = score(z, "radial") <= 2, score(z, "box") <= 2
    assert np.all(inside_b[inside_r]) and inside_b.sum() > inside_r.sum()  # the corners


def test_split_bands_and_interpolated_holdout():
    s = np.linspace(0, 4, 1000)
    lab = split(s, 2.0, 2.5, 0.1, seed=0)
    assert set(lab[s > 2.5]) == {"test"} and set(lab[(s > 2) & (s <= 2.5)]) == {"val"}
    core = lab[s <= 2]
    assert set(core) == {"train", "val_interp"} and (core == "val_interp").sum() == int(0.1 * len(core))


def test_study_reports_every_split_and_d_bins():
    r = study(fake_afterms(), "radial", seed=0, epochs=20, n_gen=2000)
    assert set(r["splits"]) == {"train", "val_interp", "val", "test"} and len(r["nll_vs_d"]) == 5
    assert r["splits"]["test"]["nll_mean"] > r["splits"]["train"]["nll_mean"]
