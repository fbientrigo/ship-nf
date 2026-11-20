"""System 3: how a flow degrades when it is retrained on its own samples, versus on points FairShip vouched for.

Two arms start from the same flow ``q_0``, fitted on real training rows, and run ``G`` generations:

* ``self``:     ``q_{g+1}`` is fitted on ``N`` samples of ``q_g`` only.
* ``anchored``: ``q_{g+1}`` is fitted on ``N`` samples where a fraction ``rho`` of ``q_g``'s samples is replaced by
  fresh real rows. In the thesis loop those real rows are the states FairShip has simulated; here they are drawn from
  the real training rows, so the study needs no FairShip run.

Each generation is scored on what matters for tails:

* coverage of the real *tail* points (radial p_t score above ``k_tail``) by samples of ``q_g``
  (:func:`ship_nf.diagnostics.coverage`), and the mass ratio of ``q_g`` to the data in that tail;
* how well ``q_g`` still covers ``q_0``: importance weights ``w = q_0(x) / q_g(x)`` for ``x ~ q_g``, summarised by
  Kish ESS / N and the PSIS Pareto ``k``. If ``q_g`` has lost part of ``q_0``'s support, a few samples carry huge
  weights: ESS drops and ``k`` rises above 0.7 (Yao et al. 2018 use this to judge a fitted density).
* the mean NLL of held-out real rows under ``q_g``.

References
----------
Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained
    on recursively generated data. *Nature*, 631, 755-759.
    Tails disappear first when a model is retrained on its own output.
Alemohammad, S., Casco-Rodriguez, J., Luzi, L., Humayun, A. I., Babaei, H., LeJeune, D., Siahkoohi, A., & Baraniuk,
    R. G. (2024). Self-consuming generative models go MAD. *ICLR 2024*.
    Fresh real data in each generation (the "fresh data loop") prevents the collapse.
Diagnostics: see the references in :mod:`ship_nf.diagnostics` (Kish 1965; Vehtari et al. 2024; Yao et al. 2018;
    Naeem et al. 2020). All written from memory on 2026-11-19; to be checked before the thesis.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow
from ship_nf.diagnostics import coverage, kish_ess, psis_khat, tail_mass_ratio
from ship_nf.extrapolation import afterms_states, pt_z, score
from ship_nf.loop import features, to_state

ARMS = ("self", "anchored")


def _log_q(model: flow.Flow, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.log_prob(torch.as_tensor(x, dtype=torch.float32)).numpy().astype(float)


def _tail_score(f: np.ndarray, ref) -> np.ndarray:
    st = [to_state(r, 0.0, 0.0, 13) for r in f]
    return score(pt_z(np.array([s["px"] for s in st]), np.array([s["py"] for s in st]), ref)[0], "radial")


def run(arr: np.ndarray, seed: int, generations: int = 5, n: int = 10000, rho: float = 0.5, k_tail: float = 2.5,
        epochs: int = 200, test_frac: float = 0.2) -> dict:
    """Both arms for one charge of afterMS rows; returns one record per (arm, generation)."""
    rng = np.random.default_rng(seed)
    f = features(afterms_states(arr))
    f = f[np.isfinite(f).all(1)]
    test = rng.random(len(f)) < test_frac
    real_train, real_test = f[~test], f[test]
    _, ref = pt_z(arr[:, 0], arr[:, 1])
    tail = real_test[_tail_score(real_test, ref) > k_tail]
    scale = real_train.std(0)

    q0, _ = flow.fit(real_train, seed, epochs=epochs)
    out = []
    for arm in ARMS:
        q = q0
        for g in range(generations + 1):
            x = q.sample(n, seed + 100 * g)
            x = x[np.isfinite(x).all(1)]
            lw = _log_q(q0, x) - _log_q(q, x)
            s = _tail_score(x, ref)
            out.append({"arm": arm, "generation": g, "ess_over_n": kish_ess(np.exp(lw - lw.max())) / len(x),
                        "khat": psis_khat(lw), "tail_coverage": coverage(tail / scale, x / scale),
                        "tail_mass_ratio": tail_mass_ratio(s > k_tail, _tail_score(real_test, ref) > k_tail),
                        "test_nll": float(-_log_q(q, real_test).mean())})
            if g == generations:
                break
            if arm == "anchored":
                keep = rng.random(len(x)) >= rho
                x = np.vstack([x[keep], real_train[rng.choice(len(real_train), int((~keep).sum()))]])
            q, _ = flow.fit(x, seed + g + 1, epochs=epochs)
    return {"seed": seed, "generations": generations, "n": n, "rho": rho, "k_tail": k_tail, "epochs": epochs,
            "tail_test_rows": int(len(tail)), "records": out}


def study(npz: Path, seeds=(1, 2), pdg: int = 13, **kw) -> dict:
    """Run both arms for one charge and several seeds of the afterMS sample (rows with pz <= 0 dropped)."""
    raw = Path(npz).read_bytes()
    data = np.load(npz)["muons"]
    data = data[(data[:, 2] > 0) & (data[:, 6] == pdg)]
    return {"data": {"path": str(Path(npz).resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "pdg": pdg,
                     "rows": len(data)}, "runs": [run(data, sd, **kw) for sd in seeds]}
