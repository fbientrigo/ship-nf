import numpy as np
import torch

from ship_nf.edge_reuse import score
from ship_nf.loop import features


class Fixed:
    """Stands in for a flow: always the same four rows, px = -3, 0, 3, 3 with median 0 and sigma 1."""
    def __init__(self):
        self.rows = features([{"x": 0, "y": 0, "px": px, "py": 0, "pz": 30.0} for px in (-3, 0, 3, 3)])

    def sample(self, n, seed):
        return np.tile(self.rows, (n // 4, 1))

    def log_prob(self, x):
        return torch.zeros(len(x))


def test_score_counts_each_side_over_all_samples():
    d = {"median": 0.0, "sigma": 1.0}
    s = score(Fixed(), d, {"left": 0.25, "right": 0.25}, np.zeros((3, 5)), n_eval=8, seed=0)
    assert (s["left"]["gen"], s["right"]["gen"]) == (2, 4)
    assert np.isclose(s["left"]["R"], 1.0) and np.isclose(s["right"]["R"], 2.0)
