"""One A-side step of the demo loop: U from past FairShip evaluations, tilt, NF fit, new states, selection by U.

::

    TrainingProblem rows  --U(s), h_{alpha,delta}-->  tilted resample  --fit-->  RealNVP  --sample-->  candidates
          ^                                                                                              |
          |                       EvaluationReports of earlier iterations                     keep physical, top-m by U
          +---------------------------------------------- FairShip (repo B) <---- MODEL_SAMPLE delivery <-+

Notes
-----
* ``U(s)`` is the Laplace-smoothed R rate of the ``k`` nearest evaluated proposals over every DIS category except UBT
  (vacuum scoring planes, NOT_ELIGIBLE under ``dis_eligibility_v0``): ``(sum n_r + 1) / (sum n_dis + 2)``.
  Without evaluations ``U = 1`` (no information, uniform tilt). Category is a proxy for eligibility, which B
  decides from the vertex material.
* The tilt is ``pi_i ∝ h(U_i / max U)`` over training rows, with ``h(u) = [delta + (1 - delta) u]^alpha``
  (``delta > 0`` keeps every row's mass positive). The base measure is uniform over training rows, not ``P0``:
  the stored weight's semantics is OPEN (contract §3).
* The training rows are a small empirical measure (19 rows today). The resample gets a Gaussian jitter of
  ``JITTER`` training standard deviations so the flow fits a smooth density instead of spikes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ship_nf import flow
from ship_nf.exchange import load_problem, load_report, write_model_delivery

FEATURES = ("x", "y", "tx", "ty", "log_p")
P_MIN = 2.0  # GeV/c: MuDISProcessor skips muons below it (FairShip 8a1cc53 newMuonDIS/prepareEvents.py:126)
JITTER = 0.05
EXCLUDED = ("UBT",)


def features(states) -> np.ndarray:
    """``(n, 5)`` array of ``x, y, px/pz, py/pz, log|p|`` (cm, cm, 1, 1, log GeV/c)."""
    a = np.array([[s["x"], s["y"], s["px"], s["py"], s["pz"]] for s in states], dtype=np.float64)
    return np.column_stack([a[:, 0], a[:, 1], a[:, 2] / a[:, 4], a[:, 3] / a[:, 4], np.log(np.linalg.norm(a[:, 2:], axis=1))])


def to_state(f: np.ndarray, z: float, t: float, pdg: int) -> dict:
    """Inverse of :func:`features` for one row, on the plane ``z`` at time ``t``."""
    x, y, tx, ty, log_p = map(float, f)
    pz = np.exp(log_p) / np.sqrt(1 + tx * tx + ty * ty)
    return {"x": x, "y": y, "z": z, "t": t, "px": tx * pz, "py": ty * pz, "pz": float(pz), "pdg": int(pdg)}


def utility(query: np.ndarray, evaluated: list[dict], scale: np.ndarray, k: int = 5) -> np.ndarray:
    """``U`` at each query row from the ``min(k, sqrt(n))`` nearest of ``n`` evaluated proposals (distance in ``scale`` units)."""
    if not evaluated:
        return np.ones(len(query))
    e = features(evaluated) / scale
    n_dis = np.array([sum(d for c, (d, _) in r["counts"].items() if c not in EXCLUDED) for r in evaluated], dtype=float)
    n_r = np.array([sum(x for c, (_, x) in r["counts"].items() if c not in EXCLUDED) for r in evaluated], dtype=float)
    d = np.linalg.norm(query[:, None, :] / scale - e[None, :, :], axis=-1)
    k = min(k, max(1, int(np.sqrt(len(evaluated)))))  # k >= n would average everything: no contrast
    near = np.argsort(d, axis=1)[:, :k]
    return (n_r[near].sum(1) + 1) / (n_dis[near].sum(1) + 2)


def h(u: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    return (delta + (1 - delta) * u) ** alpha


def kish_ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())


def propose(problem_dir: Path, history: list[tuple[Path, Path]], out: Path, m: int, seed: int,
            alpha: float = 2.0, delta: float = 0.1, k: int = 5, n_train: int = 512, n_cand: int = 2000) -> dict:
    """Write a ``MODEL_SAMPLE`` delivery of ``m`` states to ``out``.

    Parameters
    ----------
    history : list of (delivery_dir, report_dir)
        Every earlier iteration; all their evaluations inform ``U``.
    m : int
        States to send to FairShip.
    """
    pm, rows = load_problem(problem_dir)
    plane = pm["state_definition"]["plane"]
    (cx, cy, cz), (hx, hy, _) = plane["center_cm"], plane["half_cm"]
    evaluated = [r for d, rep in history for r in load_report(rep, d)]
    f_rows = features(rows)
    scale = f_rows.std(0)

    u_rows = utility(f_rows, evaluated, scale, k)
    pi = h(u_rows / u_rows.max(), alpha, delta)
    pi /= pi.sum()
    rng = np.random.default_rng(seed)
    train = f_rows[rng.choice(len(rows), n_train, p=pi)] + rng.normal(0, JITTER, (n_train, len(FEATURES))) * scale
    model, losses = flow.fit(train, seed)

    cand = model.sample(n_cand, seed)
    ok = (np.isfinite(cand).all(1) & (cand[:, 4] >= np.log(P_MIN))
          & (np.abs(cand[:, 0] - cx) <= hx) & (np.abs(cand[:, 1] - cy) <= hy))
    cand = cand[ok]
    u_cand = utility(cand, evaluated, scale, k)
    best = np.argsort(-u_cand, kind="stable")[:m]
    nearest = np.argmin(np.linalg.norm(cand[best, None, :] / scale - f_rows[None] / scale, axis=-1), axis=1)
    t0 = float(np.median([r["t"] for r in rows]))
    states = [to_state(cand[i], cz, t0, rows[j]["pdg"]) for i, j in zip(best, nearest)]

    reports = [json.loads((Path(rep) / "manifest.json").read_text())["report_id"] for _, rep in history]
    config = {"problem_id": pm["problem_id"], "seed": seed, "alpha": alpha, "delta": delta, "k": k, "n_train": n_train,
              "n_cand": n_cand, "jitter": JITTER, "reports": reports}
    generator = {"method": "realnvp_v0 (ship_nf.flow) on an h-tilted, jittered resample of training rows",
                 "model_id": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
                 "seed": seed, "n": len(states), "config": config, "features": list(FEATURES),
                 "pdg_rule": "pdg of the nearest training row", "t_rule": "median training t", "z_rule": "plane centre z",
                 "tilt_kish_ess": kish_ess(pi), "rows": len(rows), "evaluated_proposals": len(evaluated),
                 "final_nll": losses[-1], "candidates_physical": int(ok.sum()),
                 "u_selected": [float(u_cand[i]) for i in best]}
    delivery_id = write_model_delivery(problem_dir, out, states, generator)
    return {"delivery_id": delivery_id, "states": len(states), "tilt_kish_ess": round(kish_ess(pi), 2),
            "evaluated_proposals": len(evaluated), "candidates_physical": int(ok.sum()), "final_nll": round(losses[-1], 3)}
