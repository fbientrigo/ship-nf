"""How a normalizing flow degrades when it extrapolates (systems 1 and 2 of the thesis plan).

System 1, the split. Each transverse-momentum component ``px``, ``py`` is treated as roughly Gaussian and scored as
``z_j = (p_j - median_j) / sigma_j`` with a robust sigma (1.4826 x MAD, so the tails being studied do not inflate it).
Two versions of "within k sigma", compared A/B:

* ``box``:    ``max(|z_x|, |z_y|) <= k``  (each component inside its band; the corners of the square count as inside)
* ``radial``: ``sqrt(z_x^2 + z_y^2) <= k`` (the joint p_t distance; corners count as outside)

Rows with score <= ``k_train`` are the core: most train, a random ``interp_frac`` is held out as interpolated
validation. ``k_train < score <= k_val`` is the validation shell, ``score > k_val`` is the test set, seen only at the
end: mostly extrapolated, where the tails are.

System 2, the error. The flow is fitted on train only. For every split it reports the mean negative log-likelihood of
the real points, the coverage of those points by flow samples (:func:`ship_nf.diagnostics.coverage`) and the ratio of
flow mass to data mass in that score band. For validation and test points it also bins the error by ``d(x)``, the
distance to the ``k``-th nearest training row in standardised feature space.

Notes
-----
Data: the afterMS sample of ``ship_adaptive_muon_bg`` (``[px, py, pz, x, y, z, id, w]``), one charge at a time, never
pooled. The stored weight ``w`` is not used: its semantics is OPEN (contract §3).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow
from ship_nf.diagnostics import coverage, tail_mass_ratio

VERSIONS = ("box", "radial")
SPLITS = ("train", "val_interp", "val", "test")


def afterms_states(arr: np.ndarray) -> list[dict]:
    """afterMS rows ``[px, py, pz, x, y, z, id, w]`` -> state dicts as used by :func:`ship_nf.loop.features`."""
    return [{"px": r[0], "py": r[1], "pz": r[2], "x": r[3], "y": r[4]} for r in arr]


def pt_z(px: np.ndarray, py: np.ndarray, ref: tuple[np.ndarray, np.ndarray] | None = None):
    """Robust z-scores of ``px``, ``py``; ``ref`` = (median, sigma) to reuse, else fitted here. Returns z, ref."""
    p = np.column_stack([px, py])
    if ref is None:
        med = np.median(p, 0)
        ref = (med, 1.4826 * np.median(np.abs(p - med), 0))
    return (p - ref[0]) / ref[1], ref


def score(z: np.ndarray, version: str) -> np.ndarray:
    if version == "box":
        return np.abs(z).max(1)
    if version == "radial":
        return np.linalg.norm(z, axis=1)
    raise ValueError(f"version must be one of {VERSIONS}")


def split(s: np.ndarray, k_train: float, k_val: float, interp_frac: float, seed: int) -> np.ndarray:
    lab = np.where(s <= k_train, "train", np.where(s <= k_val, "val", "test")).astype(object)
    core = np.flatnonzero(lab == "train")
    held = np.random.default_rng(seed).choice(core, int(interp_frac * len(core)), replace=False)
    lab[held] = "val_interp"
    return lab


def knn_distance(query: np.ndarray, ref: np.ndarray, k: int = 5) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(ref).query(query, k)[0][:, -1]


def study(arr: np.ndarray, version: str, seed: int, k_train: float = 2.0, k_val: float = 2.5,
          interp_frac: float = 0.1, n_gen: int = 20000, epochs: int = 400, d_bins: int = 5) -> dict:
    """Fit on ``train`` of one split version and report the error per split and versus ``d(x)``."""
    from ship_nf.loop import features, to_state

    f = features(afterms_states(arr))
    z, ref = pt_z(arr[:, 0], arr[:, 1])
    s = score(z, version)
    lab = split(s, k_train, k_val, interp_frac, seed)
    model, losses = flow.fit(f[lab == "train"], seed, epochs=epochs)

    gen = model.sample(n_gen, seed + 1)
    gen = gen[np.isfinite(gen).all(1)]
    g_states = [to_state(g, 0.0, 0.0, 13) for g in gen]
    g_s = score(pt_z(np.array([g["px"] for g in g_states]), np.array([g["py"] for g in g_states]), ref)[0], version)
    band = {"train": (0, k_train), "val_interp": (0, k_train), "val": (k_train, k_val), "test": (k_val, np.inf)}

    scale = f[lab == "train"].std(0)
    with torch.no_grad():
        nll = -model.log_prob(torch.as_tensor(f, dtype=torch.float32)).numpy()
    per_split = {}
    for name in SPLITS:
        m = lab == name
        lo, hi = band[name]
        per_split[name] = {"rows": int(m.sum()), "nll_mean": float(nll[m].mean()),
                           "coverage": coverage(f[m] / scale, gen / scale) if m.sum() > 10 else None,
                           "mass_ratio": tail_mass_ratio((g_s > lo) & (g_s <= hi), (s > lo) & (s <= hi))}

    out_of_train = np.isin(lab, ["val_interp", "val", "test"])
    d = knn_distance(f[out_of_train] / scale, f[lab == "train"] / scale)
    edges = np.quantile(d, np.linspace(0, 1, d_bins + 1))
    which = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, d_bins - 1)
    by_d = [{"d_lo": float(edges[b]), "d_hi": float(edges[b + 1]), "rows": int((which == b).sum()),
             "nll_mean": float(nll[out_of_train][which == b].mean())} for b in range(d_bins)]
    return {"version": version, "seed": seed, "k_train": k_train, "k_val": k_val, "interp_frac": interp_frac,
            "sigma_ref": {"median": ref[0].tolist(), "sigma": ref[1].tolist()}, "final_train_nll": losses[-1],
            "splits": per_split, "nll_vs_d": by_d}


def ab_test(npz: Path, seeds=(1, 2, 3), charges=(13, -13), **kw) -> dict:
    """Both versions, both charges, several seeds on the same data. Charges are never pooled."""
    raw = Path(npz).read_bytes()
    data = np.load(npz)["muons"]
    ok = data[:, 2] > 0  # features need pz > 0; the sample has one all-zero-momentum row
    runs = [{"pdg": int(q), **study(data[ok & (data[:, 6] == q)], v, sd, **kw)}
            for q in charges for v in VERSIONS for sd in seeds]
    return {"data": {"path": str(Path(npz).resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(data),
                     "dropped_pz_le_0": int((~ok).sum())}, "runs": runs}
