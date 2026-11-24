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
- GOAL ``2.5 < zp <= 10`` (held out, never in the pool at start; the real tail ends at 9.66): stage 1 less often
  (0.3), stage 2 half the time (0.5).
- OUTSIDE ``zp > 10``: beyond any real muon, passes nothing. Run 1 had no upper edge and counted px of 1 000 sigma
  as GOAL.
- elsewhere: p1 = 0.02, p2|1 = 0.05.

U is fitted on stage 1 only, as the real loop fits ``>= 1 good track``. Stage 2 plays "a later cut" (Oliver's
question); reaching GOAL plays "rediscovering the held-out tail" (Mathei's question).

Rules for choosing the ``m`` states sent to the target each iteration:

- ``greedy``: the m candidates with the highest U (``loop.propose`` today).
- ``greedy_mix``: ``greedy``, but each slot comes with probability EPS from a flow fitted to the untilted pool
  (defensive mixture).
- ``thompson``: for each slot, draw coefficients from U's Laplace posterior ``N(beta, H^-1)`` and take the best unused
  candidate under that draw.
- ``thompson_mix``: ``thompson`` with the EPS untilted slots of ``greedy_mix``.

Run: ``python -m ship_nf.planted OUT.json [--seeds 5] [--iters 12] [--workers 8]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow, loop
from ship_nf.tail_demo import load, px_of

DATA = Path(os.environ.get("SHIP_NF_AFTERMS", "/home/fabian/thesis/ship_adaptive_muon_bg/data/samples/muonsFullMC_afterMS_sample.npz"))
RULES = ("greedy", "greedy_mix", "thompson", "thompson_mix")  # "sample" and "mixture" dropped after run 1, see docs
EPS, M, N_REAL, N_TRAIN, N_CAND = 0.2, 20, 20, 512, 2000
P_MAX = 400.0  # GeV/c, R1: primary proton momentum, FairShip 8a1cc53 shipgen/FixedTargetGenerator.cxx:71
GROWS = ("none", "k1", "jeffreys")  # which evaluated states join the pool, see grow_mask
DECOY, GOAL, ZP_MAX = (1.5, 2.0), 2.5, 10.0  # real afterMS mu- tail: zp max 9.66
P1 = {"decoy": 0.5, "goal": 0.3, "rest": 0.02, "outside": 0.0}
P2 = {"decoy": 0.0, "goal": 0.5, "rest": 0.05, "outside": 0.0}


def region(zp: np.ndarray) -> np.ndarray:
    return np.where(zp > ZP_MAX, "outside", np.where(zp > GOAL, "goal",
                    np.where((zp >= DECOY[0]) & (zp <= DECOY[1]), "decoy", "rest")))


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
    """Resample ``rows`` with ``weights`` (+ loop.JITTER), fit a flow, return candidates with P_MIN <= p <= P_MAX (R1)."""
    scale = rows.std(0)
    idx = rng.choice(len(rows), N_TRAIN, p=weights)
    model, _ = flow.fit(rows[idx] + rng.normal(0, loop.JITTER, (N_TRAIN, rows.shape[1])) * scale, seed)
    c = model.sample(N_CAND, seed)
    return c[np.isfinite(c).all(1) & (c[:, 4] >= np.log(loop.P_MIN)) & (c[:, 4] <= np.log(P_MAX))]


def grow_mask(k1: np.ndarray, base: float, mode: str) -> np.ndarray:
    """R3 growth rule. ``k1``: stage-1 passes out of N_REAL per picked state; ``base``: pooled stage-1 rate so far.

    ``k1``: grow on any pass. ``jeffreys``: grow only if the 16 % Jeffreys bound of k/N is above ``base`` (one pass
    out of 20 in a region at the base rate is then not enough).
    """
    from scipy.stats import beta as B

    if mode == "none":
        return np.zeros(len(k1), bool)
    if mode == "k1":
        return k1 > 0
    return B.ppf(0.16, k1 + 0.5, N_REAL - k1 + 0.5) > base


def trajectory(rule: str, grow: str, seed: int, iters: int, anchor: float | None = None, n0: int | None = None) -> dict:
    """One loop run; returns per-iteration counts by region and stage.

    ``anchor``: cap on the share of the tilted resampling mass carried by grown rows (R3); None = no cap. P0 rows
    are never removed. Grown rows are kept apart from P0 (``grown``) so their share and lineage stay explicit.
    ``n0``: P0 = a random subset of that many training rows (fixed per seed); the real P2 problem has 19 rows.
    """
    torch.set_num_threads(1)
    d = load(DATA, 1)
    zp_of = lambda f: (px_of(f) - d["median"]) / d["sigma"]  # noqa: E731
    p0 = d["f"][d["lab"] == "train"]
    if n0:
        p0 = p0[np.random.default_rng(10_000 + seed).choice(len(p0), n0, replace=False)]
    grown = np.empty((0, p0.shape[1]))
    center, scale = p0.mean(0), p0.std(0)  # fixed standardisation for U, as loop.propose uses P0's
    rng = np.random.default_rng(seed)
    evaluated, out, untilted = [], [], None
    for it in range(iters):
        s0 = seed * 1000 + it
        beta = loop.fit_u(evaluated, center, scale)
        pool = np.vstack([p0, grown])
        u_pool = loop.utility(pool, beta, center, scale)
        pi = loop.h(u_pool / u_pool.max(), 2.0, 0.1)
        pi /= pi.sum()
        g_mass = pi[len(p0):].sum()
        if anchor is not None and g_mass > anchor:  # R3: grown rows carry at most `anchor` of the mass
            pi[len(p0):] *= anchor / g_mass
            pi[:len(p0)] *= (1 - anchor) / (1 - g_mass)
        cand = fit_sample(pool, pi, rng, s0)
        mix = rule.endswith("_mix")
        if mix and untilted is None:  # always P0, never the grown pool (run 2 refit it on the grown pool)
            untilted = fit_sample(p0, None, rng, s0 + 500)
        n_mix = int(rng.binomial(M, EPS)) if mix else 0
        k = M - n_mix
        if rule.startswith("greedy"):
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
        else:  # Thompson before any data (U = 1 everywhere): any candidate is as good
            pick = cand[rng.choice(len(cand), k, replace=False)]
        if n_mix:
            pick = np.vstack([pick, untilted[rng.choice(len(untilted), n_mix, replace=False)]])
        reg = region(zp_of(pick))
        k1 = rng.binomial(N_REAL, [P1[r] for r in reg])
        k2 = rng.binomial(k1, [P2[r] for r in reg])
        evaluated += [{**state(f), "counts": {"PLANTED": (N_REAL, int(a))}} for f, a in zip(pick, k1)]
        base = sum(r["counts"]["PLANTED"][1] for r in evaluated) / (N_REAL * len(evaluated))
        add = grow_mask(k1, base, grow)
        near = (np.linalg.norm((pick[:, None] - grown[None]) / scale, axis=-1).min(1) < 2 * loop.JITTER).mean() if len(grown) else 0.0
        grown = np.vstack([grown, pick[add]])
        out.append({"iter": it, "n_goal": int((reg == "goal").sum()), "n_decoy": int((reg == "decoy").sum()),
                    "n_outside": int((reg == "outside").sum()),
                    "n_untilted": n_mix, "k1": int(k1.sum()), "k2": int(k2.sum()),
                    "k2_goal": int(k2[reg == "goal"].sum()), "zp_max": float(zp_of(pick).max()),
                    "added": int(add.sum()), "added_rest": int((add & (reg == "rest")).sum()),
                    "grown": len(grown), "grown_mass": float(min(g_mass, anchor) if anchor is not None else g_mass),
                    "near_dup": float(near)})
    return {"rule": rule, "grow": grow, "anchor": anchor, "n0": n0, "seed": seed, "iters": out}


def summarize(path: Path, last: int = 5) -> list[dict]:
    """Per (rule, grow): the numbers the CM 39 questions ask for, with plain binomial uncertainty.

    - ``first_goal``: iteration of the first proposal in GOAL, per seed (None = never, censored).
    - ``goal_frac_last``: share of proposals in GOAL over the last ``last`` iterations, pooled over seeds, with a
      Jeffreys 68 % interval ``Beta(k + 1/2, n - k + 1/2)``; same for ``decoy_frac_last``.
    - ``k2_per_real``: stage-2 passes per evaluated realization over the whole run, pooled, Jeffreys 68 %.
    """
    from scipy.stats import beta as B

    jeff = lambda k, n: [round(k / n, 4), *(round(float(v), 4) for v in B.ppf([0.16, 0.84], k + 0.5, n - k + 0.5))]  # noqa: E731
    runs = json.loads(Path(path).read_text())["runs"]
    out = []
    for key in sorted({(r["rule"], str(r["grow"]), str(r.get("anchor")), str(r.get("n0"))) for r in runs}):
        rs = sorted([r for r in runs if (r["rule"], str(r["grow"]), str(r.get("anchor")), str(r.get("n0"))) == key], key=lambda r: r["seed"])
        tail = [i for r in rs for i in r["iters"][-last:]]
        n_last, n_all = M * len(tail), M * N_REAL * sum(len(r["iters"]) for r in rs)
        out.append({"rule": key[0], "grow": key[1], "anchor": key[2], "n0": key[3], "seeds": len(rs),
                    "first_goal": [next((i["iter"] for i in r["iters"] if i["n_goal"]), None) for r in rs],
                    "goal_frac_last": jeff(sum(i["n_goal"] for i in tail), n_last),
                    "decoy_frac_last": jeff(sum(i["n_decoy"] for i in tail), n_last),
                    "outside_frac_last": jeff(sum(i.get("n_outside", 0) for i in tail), n_last),
                    "goal_frac_last_per_seed": [round(sum(i["n_goal"] for i in r["iters"][-last:]) / (M * last), 2) for r in rs],
                    "k2_per_real": jeff(sum(i["k2"] for r in rs for i in r["iters"]), n_all),
                    "k2_goal": sum(i["k2_goal"] for r in rs for i in r["iters"]),
                    "stuck_seeds": sum(1 for r in rs if not any(i["n_goal"] for i in r["iters"][-last:])),
                    "added_rest_share": round(sum(i.get("added_rest", 0) for r in rs for i in r["iters"])
                                              / max(1, sum(i.get("added", 0) for r in rs for i in r["iters"])), 3),
                    "near_dup_last": round(float(np.mean([i.get("near_dup", 0) for i in tail])), 3)})
    return out


def _run(args):
    return trajectory(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rules", nargs="+", default=list(RULES), choices=RULES)
    ap.add_argument("--grow", nargs="+", default=list(GROWS), choices=GROWS)
    ap.add_argument("--anchors", nargs="+", type=float, default=[None], help="R3 caps on the grown-row mass")
    ap.add_argument("--arms", nargs="+", help="explicit rule:grow:anchor list, overrides --rules/--grow/--anchors")
    ap.add_argument("--n0", nargs="+", type=lambda v: None if v == "all" else int(v), default=[None], help="P0 sizes, or all")
    ap.add_argument("--summarize", action="store_true", help="read OUT.json and print the summary instead of running")
    a = ap.parse_args()
    if a.summarize:
        for row in summarize(a.out):
            print(json.dumps(row))
        return
    if a.arms:  # explicit list "rule:grow:anchor", e.g. greedy_mix:k1:0.5 (anchor "-" = none)
        arms = [(r, g, None if an == "-" else float(an)) for r, g, an in (x.split(":") for x in a.arms)]
    else:
        arms = [(r, g, an) for r in a.rules for g in a.grow for an in (a.anchors if g != "none" else [None])]
    jobs = [(r, g, s, a.iters, an, n0) for n0 in a.n0 for r, g, an in arms for s in range(1, a.seeds + 1)]
    t0 = time.time()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    part = a.out.with_suffix(".partial.jsonl")  # one line per finished run: a crash loses nothing already done
    runs = [json.loads(line) for line in part.open()] if part.exists() else []  # resume: keep finished runs
    done = {(r["rule"], r["grow"], r["seed"], r.get("anchor"), r.get("n0")) for r in runs}
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[4], j[5]) not in done]
    with Pool(a.workers) as p, part.open("a") as f:
        for r in p.imap_unordered(_run, jobs):
            runs.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    a.out.write_text(json.dumps({
        "commit": commit, "seconds": round(time.time() - t0), "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "argv": os.sys.argv[1:], "config": {"eps": EPS, "m": M, "n_real": N_REAL, "n_train": N_TRAIN, "n_cand": N_CAND, "decoy": DECOY, "p_max": P_MAX,
                   "goal": GOAL, "zp_max": ZP_MAX, "p1": P1, "p2_given_1": P2, "tilt": {"alpha": 2.0, "delta": 0.1}, "iters": a.iters},
        "runs": runs}, indent=1))
    print(f"wrote {a.out} ({len(runs)} runs, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
