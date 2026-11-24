"""One A-side step of the demo loop: fit U on past FairShip results, tilt, train the flow, propose new states.

::

    TrainingProblem rows  --U(s), h_{alpha,delta}-->  tilted resample  --fit-->  RealNVP  --sample-->  candidates
          ^                                                                                              |
          |                       EvaluationReports of earlier iterations                     keep physical, top-m by U
          +---------------------------------------------- FairShip (repo B) <---- MODEL_SAMPLE delivery <-+

How the step works
------------------
1. Every proposal FairShip has already evaluated is one training example for U: its state, the number of DIS events
   it produced (``n_dis``) and how many of them reached the current stage (``n_r``). UBT-category DIS are left out:
   they sit on vacuum scoring planes and are NOT_ELIGIBLE under ``dis_eligibility_v0``. The category is a proxy for
   eligibility, which repo B decides from the vertex material.
2. :func:`fit_u` fits ``U(s) = P(stage reached | state s)`` as a binomial logistic regression on quadratic features of
   the standardised state. Quadratic terms let U describe a region (for example "the path crosses the decay vessel"),
   not only a direction. Evaluating U costs one dot product per state, so it can score thousands of candidates.
   With no evaluations yet U is 1 everywhere: no information, so no tilt.
3. Training rows are tilted with ``pi_i ∝ h(U_i / max U)``, ``h(u) = [delta + (1 - delta) u]^alpha``. ``delta > 0``
   keeps every row's probability positive (the law of ``pi_demo`` and the 1112 note). The base measure is uniform over
   training rows, not ``P0``: the stored weight's semantics is OPEN (contract §3).
4. The tilted resample gets a Gaussian jitter of ``JITTER`` training standard deviations, so 19 distinct rows give a
   smooth density instead of 19 spikes, and :mod:`ship_nf.flow` fits a RealNVP to it.
5. Flow samples that MuDIS would reject (``p < 2 GeV/c``) or that fall off the state plane are dropped, and the ``m``
   candidates with the highest U are written as a ``MODEL_SAMPLE`` delivery.

Where else to look
------------------
``ship_nf.flow`` (the flow), ``ship_nf.diagnostics`` (ESS and tail metrics), ``ship_nf.exchange`` (artifacts),
repo B ``scripts/demo_loop.sh`` (the driver that alternates A and FairShip).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow
from ship_nf.diagnostics import kish_ess
from ship_nf.exchange import load_problem, load_report, write_model_delivery

FEATURES = ("x", "y", "tx", "ty", "log_p")
P_MIN = 2.0  # GeV/c: MuDISProcessor skips muons below it (FairShip 8a1cc53 newMuonDIS/prepareEvents.py:126)
JITTER = 0.05
EXCLUDED = ("UBT",)
L2 = 1.0  # ridge strength of the U regression; the intercept is not penalised


def features(states) -> np.ndarray:
    """``(n, 5)`` array of ``x, y, px/pz, py/pz, log|p|`` (cm, cm, 1, 1, log GeV/c).

    Slopes and log momentum instead of raw momenta: the slopes say where the muon is heading, and ``log p`` spreads the
    2-300 GeV range evenly, which helps both the flow and U.
    """
    a = np.array([[s["x"], s["y"], s["px"], s["py"], s["pz"]] for s in states], dtype=np.float64)
    return np.column_stack([a[:, 0], a[:, 1], a[:, 2] / a[:, 4], a[:, 3] / a[:, 4], np.log(np.linalg.norm(a[:, 2:], axis=1))])


def to_state(f: np.ndarray, z: float, t: float, pdg: int) -> dict:
    """Inverse of :func:`features` for one row, placed on the plane ``z`` at time ``t``."""
    x, y, tx, ty, log_p = map(float, f)
    pz = np.exp(log_p) / np.sqrt(1 + tx * tx + ty * ty)
    return {"x": x, "y": y, "z": z, "t": t, "px": tx * pz, "py": ty * pz, "pz": float(pz), "pdg": int(pdg)}


def _quad(z: np.ndarray) -> np.ndarray:
    """Intercept, linear and all quadratic terms (``z_i z_j``, ``i <= j``) of standardised features."""
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack([np.ones(len(z)), z, z[:, i] * z[:, j]])


def observed(evaluated: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """OBSERVED, per evaluated proposal: ``n`` evaluable realizations and ``y`` positives at the endpoint.

    One realization = one eligible DIS event FairShip produced and replayed for that state; ``Y_k = 1`` when it
    reaches the report's endpoint (``r_demo_v0``: >= 1 goodTracks). A repeated state keeps all its realizations as
    ``y`` out of ``n`` (binomial), never one averaged label. UBT DIS (NOT_ELIGIBLE) are outside ``n``. A technical
    failure never reaches here: ``fs_evaluate.sh`` stops without writing a report, so its proposal has no rows.
    """
    n = np.array([sum(d for c, (d, _) in r["counts"].items() if c not in EXCLUDED) for r in evaluated], dtype=float)
    y = np.array([sum(k for c, (_, k) in r["counts"].items() if c not in EXCLUDED) for r in evaluated], dtype=float)
    return n, y


# PREDICTED: U_k(s) scores a state before FairShip sees it. The model is the pair fit_u / utility and nothing else
# calls it; to swap in a kNN or a tree, replace these two functions with the same signatures. FairShip, the report
# and the tilt do not change.
def fit_u(evaluated: list[dict], center: np.ndarray, scale: np.ndarray, iters: int = 30) -> np.ndarray | None:
    """Coefficients of ``U(s) = sigmoid(beta . quad((f(s) - center) / scale))``, or None without data.

    Binomial logistic regression fitted by Newton's method: each evaluated proposal contributes ``n_r`` successes
    out of ``n_dis`` trials, so a proposal with 100 DIS weighs more than one with 20. A ridge penalty ``L2`` keeps
    the 21 coefficients finite when only a handful of proposals exist.
    """
    if not evaluated:
        return None
    x = _quad((features(evaluated) - center) / scale)
    n, y = observed(evaluated)
    pen = np.full(x.shape[1], L2)
    pen[0] = 0.0
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-x @ beta))
        grad = x.T @ (y - n * p) - pen * beta
        hess = (x * (n * p * (1 - p))[:, None]).T @ x + np.diag(pen) + 1e-9 * np.eye(len(beta))
        beta += np.linalg.solve(hess, grad)
    return beta


def utility(query: np.ndarray, beta: np.ndarray | None, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """U at each row of ``query`` (feature space). Without a fitted model, 1 everywhere."""
    if beta is None:
        return np.ones(len(query))
    return 1 / (1 + np.exp(-_quad((query - center) / scale) @ beta))


def h(u: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """Tilt factor ``[delta + (1 - delta) u]^alpha``; never below ``delta^alpha``."""
    return (delta + (1 - delta) * u) ** alpha


def propose(problem_dir: Path, history: list[tuple[Path, Path]], out: Path, m: int, seed: int,
            alpha: float = 2.0, delta: float = 0.1, n_train: int = 512, n_cand: int = 2000, eps: float = 0.0) -> dict:
    """Write a ``MODEL_SAMPLE`` delivery of ``m`` new states to ``out`` and return a short summary.

    Parameters
    ----------
    problem_dir : Path
        The TrainingProblem (P0).
    history : list of (delivery_dir, report_dir)
        Every earlier iteration; all their evaluations train U.
    m : int
        States to send to FairShip.
    seed : int
        Seeds the resample, the flow and its samples; the same inputs give the same delivery.
    eps : float
        Share of the ``m`` slots used for estimation (R4): drawn at random from a flow fitted to the untilted training
        rows, so their density is the flow's and is stored. The other slots are search: top U from the tilted flow,
        no usable density. ``eps = 0`` (default) is the loop as before, with no second flow.
    """
    pm, rows = load_problem(problem_dir)
    plane = pm["state_definition"]["plane"]
    (cx, cy, cz), (hx, hy, _) = plane["center_cm"], plane["half_cm"]
    evaluated = [r for d, rep in history for r in load_report(rep, d)]
    f_rows = features(rows)
    center, scale = f_rows.mean(0), f_rows.std(0)
    beta = fit_u(evaluated, center, scale)

    u_rows = utility(f_rows, beta, center, scale)
    pi = h(u_rows / u_rows.max(), alpha, delta)
    pi /= pi.sum()
    rng = np.random.default_rng(seed)

    def draw(weights, s):  # fit a flow on a weighted, jittered resample; candidates on the plane with p >= P_MIN
        train = f_rows[rng.choice(len(rows), n_train, p=weights)] + rng.normal(0, JITTER, (n_train, len(FEATURES))) * scale
        model, losses = flow.fit(train, s)
        c = model.sample(n_cand, s)
        ok = (np.isfinite(c).all(1) & (c[:, 4] >= np.log(P_MIN)) & (np.abs(c[:, 0] - cx) <= hx) & (np.abs(c[:, 1] - cy) <= hy))
        return model, losses, c[ok], float(ok.mean())

    model, losses, cand, acc = draw(pi, seed)
    n_est = int(round(eps * m))
    u_cand = utility(cand, beta, center, scale)
    best = np.argsort(-u_cand, kind="stable")[:m - n_est]
    picks, roles, models = [cand[i] for i in best], ["search"] * len(best), [model] * len(best)
    if n_est:
        model_e, _, cand_e, acc_e = draw(None, seed + 1)
        picks += list(cand_e[rng.choice(len(cand_e), n_est, replace=False)])
        roles += ["estimate"] * n_est
        models += [model_e] * n_est
    picks = np.array(picks)
    nearest = np.argmin(np.linalg.norm(picks[:, None, :] / scale - f_rows[None] / scale, axis=-1), axis=1)
    t0 = float(np.median([r["t"] for r in rows]))
    states = [to_state(f, cz, t0, rows[j]["pdg"]) for f, j in zip(picks, nearest)]

    progenitors = {r.get("progenitor_id") for r in rows}
    reports = [json.loads((Path(rep) / "manifest.json").read_text())["report_id"] for _, rep in history]
    config = {"problem_id": pm["problem_id"], "seed": seed, "alpha": alpha, "delta": delta, "u_model": "logistic_quad_v0",
              "l2": L2, "n_train": n_train, "n_cand": n_cand, "jitter": JITTER, "reports": reports, "eps": eps}
    generator = {"method": "realnvp_v0 (ship_nf.flow) on an h-tilted, jittered resample of training rows",
                 "model_id": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
                 "seed": seed, "n": len(states), "config": config, "features": list(FEATURES),
                 "pdg_rule": "pdg of the nearest training row", "t_rule": "median training t", "z_rule": "plane centre z",
                 "origin": "NF", "tilt_kish_ess": kish_ess(pi), "tilt_base": "equiprobable distinct training rows",
                 "rows": len(rows), "distinct_progenitors": len(progenitors), "evaluated_proposals": len(evaluated),
                 "final_nll": losses[-1], "candidates_physical": len(cand),
                 "u_selected": [float(x) for x in utility(picks, beta, center, scale)]}
    # Checkpoints + per-state log density under the flow that produced the state, so q can be recomputed later;
    # flow.sample(n_cand, seed) on a checkpoint regenerates its candidates. Only "estimate" states were drawn from
    # their flow (density = exp(log_q_flow) / acceptance); "search" states are a top-m choice and have no usable q.
    out.mkdir(parents=True, exist_ok=True)
    ck = {"search": model, **({"estimate": model_e} if n_est else {})}
    for role, mdl in ck.items():
        torch.save(mdl.state_dict(), out / f"flow_{role}.pt")
    with torch.no_grad():
        log_q = [float(mdl.log_prob(torch.as_tensor(f[None], dtype=torch.float32))[0]) for f, mdl in zip(picks, models)]
    generator.update({"checkpoints": {r: {"file": f"flow_{r}.pt", "sha256": hashlib.sha256((out / f"flow_{r}.pt").read_bytes()).hexdigest()}
                                      for r in ck},
                      "roles": roles, "log_q_flow": log_q, "log_q_space": list(FEATURES),
                      "acceptance": {"search": acc, **({"estimate": acc_e} if n_est else {})},
                      "selection": {"search": "top_m_by_U", "estimate": "uniform draw among accepted candidates of the untilted flow"}})
    delivery_id = write_model_delivery(problem_dir, out, states, generator)
    return {"delivery_id": delivery_id, "states": len(states), "rows": len(rows), "distinct_progenitors": len(progenitors),
            "tilt_kish_ess": round(kish_ess(pi), 2),
            "evaluated_proposals": len(evaluated), "candidates_physical": len(cand), "final_nll": round(losses[-1], 3),
            "estimate_slots": n_est}


def propose_manual(problem_dir: Path, out: Path, x: float, y: float, px: float, py: float, pz: float, pdg: int) -> dict:
    """Write a one-state delivery for a muon defined by hand on the P2 plane (same z, t rules as :func:`propose`).

    It goes through the same validation and FairShip route as an NF state; ``generator.origin = "MANUAL"`` and the
    ``model_id`` (hash of the state) keep it apart from model draws.
    """
    pm, rows = load_problem(problem_dir)
    state = {"x": x, "y": y, "z": pm["state_definition"]["plane"]["center_cm"][2],
             "t": float(np.median([r["t"] for r in rows])), "px": px, "py": py, "pz": pz, "pdg": pdg}
    mid = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]
    generator = {"method": "manual: state typed by hand", "origin": "MANUAL", "model_id": mid, "seed": None, "n": 1,
                 "state": state}
    return {"delivery_id": write_model_delivery(problem_dir, out, [state], generator), "origin": "MANUAL", "state": state}
