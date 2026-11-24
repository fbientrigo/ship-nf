"""Planted-target benchmark: does the loop find a region it never trained on, and does greed trap it?

The FairShip step of the loop is replaced by a pass probability we choose ourselves (a *planted target*), so the right
answer is known and a run takes minutes instead of days. Everything else is the loop's own code: ``loop.fit_u`` (U),
``loop.h`` (tilt), ``flow.fit`` (RealNVP), on afterMS mu- rows with the tail split of ``tail_demo``.

::

    pool (afterMS |zpx| <= 2 rows) --tilt by U--> resample + jitter --fit--> RealNVP --sample 2000--> choose m by RULE
        ^                                                                                                  |
        |  GROW: states with a stage-1 pass join the pool                                                   v
        +--------------- fit U on stage-1 counts <---- planted target: k1 ~ Bin(n, p1), k2 ~ Bin(k1, p2|1) <-+

Planted target, on the signed robust z of px (``zp``, same reference as the deck's ``|zpx|``):

- DECOY ``1.5 <= zp <= 2`` (inside training): stage 1 often (p1 = 0.5), stage 2 never (p2|1 = 0).
- GOAL ``zp > 2.5`` (held out, never in the pool at start): stage 1 less often (0.3), stage 2 half the time (0.5).
- elsewhere: p1 = 0.02, p2|1 = 0.05.

U is fitted on stage 1 only, as the real loop fits ``>= 1 good track``. Stage 2 plays "a later cut" (Oliver's
question); reaching GOAL plays "rediscovering the held-out tail" (Mathei's question).

Rules for choosing the ``m`` states sent to the target each iteration:

- ``greedy``: the m candidates with the highest U (``loop.propose`` today).
- ``sample``: m candidates at random from the tilted flow (the tilt alone steers).
- ``mixture``: like ``sample``, but each slot comes with probability EPS from a flow fitted to the untilted pool
  (defensive mixture).
- ``thompson``: for each slot, draw coefficients from U's Laplace posterior ``N(beta, H^-1)`` and take the best unused
  candidate under that draw.
- ``thompson_mix``: ``thompson`` with the EPS untilted slots of ``mixture``.

Run: ``python -m ship_nf.planted OUT.json [--seeds 5] [--iters 12] [--workers 8]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow, loop
from ship_nf.tail_demo import load, px_of

DATA = Path("/home/fabian/thesis/ship_adaptive_muon_bg/data/samples/muonsFullMC_afterMS_sample.npz")
RULES = ("greedy", "sample", "mixture", "thompson", "thompson_mix")
EPS, M, N_REAL, N_TRAIN, N_CAND = 0.2, 20, 20, 512, 2000
DECOY, GOAL = (1.5, 2.0), 2.5
P1 = {"decoy": 0.5, "goal": 0.3, "rest": 0.02}
P2 = {"decoy": 0.0, "goal": 0.5, "rest": 0.05}


def region(zp: np.ndarray) -> np.ndarray:
    return np.where(zp > GOAL, "goal", np.where((zp >= DECOY[0]) & (zp <= DECOY[1]), "decoy", "rest"))


def state(f: np.ndarray) -> dict:
    return loop.to_state(f, 0.0, 0.0, 13)


def laplace_cov(evaluated: list[dict], beta: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of the Hessian ``fit_u`` builds at its last Newton step, evaluated at the returned ``beta``."""
    x = loop._quad((loop.features(evaluated) - center) / scale)
    n, _ = loop.observed(evaluated)
    p = 1 / (1 + np.exp(-x @ beta))
    pen = np.full(x.shape[1], loop.L2)
    pen[0] = 0.0
    hess = (x * (n * p * (1 - p))[:, None]).T @ x + np.diag(pen) + 1e-9 * np.eye(len(beta))
    return np.linalg.inv(hess)


def fit_sample(rows: np.ndarray, weights: np.ndarray | None, rng: np.random.Generator, seed: int) -> np.ndarray:
    """Resample ``rows`` with ``weights`` (+ loop.JITTER), fit a flow, return physical candidates (p >= P_MIN)."""
    scale = rows.std(0)
    idx = rng.choice(len(rows), N_TRAIN, p=weights)
    model, _ = flow.fit(rows[idx] + rng.normal(0, loop.JITTER, (N_TRAIN, rows.shape[1])) * scale, seed)
    c = model.sample(N_CAND, seed)
    return c[np.isfinite(c).all(1) & (c[:, 4] >= np.log(loop.P_MIN))]


def trajectory(rule: str, grow: bool, seed: int, iters: int) -> dict:
    """One loop run; returns per-iteration counts by region and stage."""
    torch.set_num_threads(1)
    d = load(DATA, 1)
    zp_of = lambda f: (px_of(f) - d["median"]) / d["sigma"]  # noqa: E731
    pool = d["f"][d["lab"] == "train"]
    center, scale = pool.mean(0), pool.std(0)  # fixed standardisation for U, as loop.propose uses P0's
    rng = np.random.default_rng(seed)
    evaluated, out, untilted = [], [], None
    for it in range(iters):
        s0 = seed * 1000 + it
        beta = loop.fit_u(evaluated, center, scale)
        u_pool = loop.utility(pool, beta, center, scale)
        pi = loop.h(u_pool / u_pool.max(), 2.0, 0.1)
        cand = fit_sample(pool, pi / pi.sum(), rng, s0)
        mix = rule in ("mixture", "thompson_mix")
        if mix and (untilted is None or grow):
            untilted = fit_sample(pool, None, rng, s0 + 500)
        n_mix = int(rng.binomial(M, EPS)) if mix else 0
        k = M - n_mix
        if rule == "greedy":
            pick = cand[np.argsort(-loop.utility(cand, beta, center, scale), kind="stable")[:k]]
        elif rule in ("thompson", "thompson_mix") and beta is not None:
            cov = laplace_cov(evaluated, beta, center, scale)
            free, pick = np.ones(len(cand), bool), []
            for b in rng.multivariate_normal(beta, cov, k):
                u = np.where(free, loop.utility(cand, b, center, scale), -1.0)
                j = int(np.argmax(u))
                free[j] = False
                pick.append(cand[j])
            pick = np.array(pick)
        else:  # sample, mixture, and Thompson before any data (U = 1 everywhere)
            pick = cand[rng.choice(len(cand), k, replace=False)]
        if n_mix:
            pick = np.vstack([pick, untilted[rng.choice(len(untilted), n_mix, replace=False)]])
        reg = region(zp_of(pick))
        k1 = rng.binomial(N_REAL, [P1[r] for r in reg])
        k2 = rng.binomial(k1, [P2[r] for r in reg])
        evaluated += [{**state(f), "counts": {"PLANTED": (N_REAL, int(a))}} for f, a in zip(pick, k1)]
        if grow and (k1 > 0).any():
            pool = np.vstack([pool, pick[k1 > 0]])
        out.append({"iter": it, "n_goal": int((reg == "goal").sum()), "n_decoy": int((reg == "decoy").sum()),
                    "n_untilted": n_mix, "k1": int(k1.sum()), "k2": int(k2.sum()),
                    "k2_goal": int(k2[reg == "goal"].sum()), "zp_max": float(zp_of(pick).max()), "pool": len(pool)})
    return {"rule": rule, "grow": grow, "seed": seed, "iters": out}


def _run(args):
    return trajectory(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(r, g, s, a.iters) for r in RULES for g in (False, True) for s in range(1, a.seeds + 1)]
    t0 = time.time()
    with Pool(a.workers) as p:
        runs = p.map(_run, jobs, chunksize=1)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "commit": commit, "seconds": round(time.time() - t0), "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "config": {"eps": EPS, "m": M, "n_real": N_REAL, "n_train": N_TRAIN, "n_cand": N_CAND, "decoy": DECOY,
                   "goal": GOAL, "p1": P1, "p2_given_1": P2, "tilt": {"alpha": 2.0, "delta": 0.1}, "iters": a.iters},
        "runs": runs}, indent=1))
    print(f"wrote {a.out} ({len(runs)} runs, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
