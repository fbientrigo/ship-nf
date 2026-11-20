"""Train in the centre, ask for a tail: demos 2-4 for the meeting of 2026-11-21 (afterMS, one charge).

The variable is ``px`` of the afterMS muons, scored as a robust z (``ship_nf.extrapolation.pt_z``, median and
1.4826 x MAD of all rows of the charge). The rule is two-sided and saved with every result::

    |z| <= K_TRAIN            centre: 90 % train, 10 % interpolated validation (val_interp)
    K_TRAIN < |z| <= K_VAL    validation shell (both tails)
    |z| > K_VAL               test, used only by :func:`run` for the final numbers

The flow is trained on the 5 features of ``ship_nf.loop.features`` (x, y, tx, ty, log p) of train rows; ``px`` is a
1-D projection of them. The median/MAD only *define* the regions. Everything learned (flow standardisation,
distance transform) is fitted on train rows only.

Distances (demo 4): standardise with train mean and std, then take the distance to the 5th nearest train row
(``d_train``) and to the 5th nearest validation-shell row (``d_val``), the same function for real test rows and
for flow samples. Metric: mass ratio per distance band, ``(g_b / G) / (r_b / R)`` with ``G`` all flow samples before
any filter and ``R`` all real rows of the charge, so a flow that loses tail mass shows a ratio below 1.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow
from ship_nf.extrapolation import afterms_states, knn_distance, pt_z, split
from ship_nf.loop import features

K_TRAIN, K_VAL, INTERP, PDG, KNN = 2.0, 2.5, 0.1, 13, 5


def px_of(f: np.ndarray) -> np.ndarray:
    """``px`` (GeV/c) from feature rows ``x, y, tx, ty, log p``."""
    return f[:, 2] * np.exp(f[:, 4]) / np.sqrt(1 + f[:, 2] ** 2 + f[:, 3] ** 2)


def load(npz: Path, seed: int) -> dict:
    """Rows of one charge, their features, |z| of px and the split labels. pz <= 0 rows are dropped (features need it)."""
    raw = Path(npz).read_bytes()
    a = np.load(npz)["muons"]
    a = a[(a[:, 2] > 0) & (a[:, 6] == PDG)]
    f = features(afterms_states(a))
    z, ref = pt_z(a[:, 0], a[:, 1])
    s = np.abs(z[:, 0])
    return {"f": f, "s": s, "lab": split(s, K_TRAIN, K_VAL, INTERP, seed), "median": float(ref[0][0]),
            "sigma": float(ref[1][0]), "data": {"path": str(Path(npz).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                                                "pdg": PDG, "rows": len(a)}}


def z_of(f: np.ndarray, d: dict) -> np.ndarray:
    return np.abs(px_of(f) - d["median"]) / d["sigma"]


def sample_region(model: flow.Flow, inside, n_req: int, seed: int, batch: int = 20000, max_batches: int = 10):
    """Draw from ``model`` in batches and keep rows where ``inside(rows)`` is True, until ``n_req`` or ``max_batches``.

    Returns the accepted rows (at most ``n_req``) and ``{"requested", "generated", "accepted"}``; accepted may be 0.
    """
    kept, gen = [], 0
    for b in range(max_batches):
        g = model.sample(batch, seed + b)
        gen += len(g)
        kept.append(g[np.isfinite(g).all(1) & inside(g)])
        if sum(map(len, kept)) >= n_req:
            break
    acc = np.concatenate(kept)[:n_req]
    return acc, {"requested": n_req, "generated": gen, "accepted": len(acc), "max_batches": max_batches, "batch": batch}


def mass_ratio(d_ref, d_gen, n_ref, n_gen, edges) -> list[dict]:
    """Per band ``[lo, hi)``: real and flow counts, their totals and ``(g/G)/(r/R)``; None when the band has no real row."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        r, g = int(((d_ref >= lo) & (d_ref < hi)).sum()), int(((d_gen >= lo) & (d_gen < hi)).sum())
        out.append({"lo": float(lo), "hi": float(hi), "ref": r, "ref_total": n_ref, "gen": g, "gen_total": n_gen,
                    "ratio": (g / n_gen) / (r / n_ref) if r else None})
    return out


def _commit() -> str:
    here = Path(__file__).parent
    c = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "src"], capture_output=True, text=True).stdout
    return c + ("-dirty" if dirty else "")


def _rule(d: dict) -> dict:
    return {"variable": "px of afterMS muons, pdg %d" % PDG, "z": "|px - median| / (1.4826 MAD), all rows of the charge",
            "median": d["median"], "sigma": d["sigma"], "train": f"|z| <= {K_TRAIN} minus val_interp",
            "val_interp": f"random {INTERP:.0%} of |z| <= {K_TRAIN}", "val": f"{K_TRAIN} < |z| <= {K_VAL}",
            "test": f"|z| > {K_VAL}", "trained_on": "x, y, tx, ty, log p (ship_nf.loop.features)"}


def run(npz: Path, out: Path, seed: int = 1, n_gen: int = 50000, n_req: int = 1000, epochs: int = 400) -> dict:
    """Demos 3-4: one flow on train rows, saved as the common checkpoint; histograms and distance bands to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    d = load(npz, seed)
    f, lab = d["f"], d["lab"]
    tr = f[lab == "train"]
    model, losses = flow.fit(tr, seed, epochs=epochs)
    torch.save(model.state_dict(), out / "flow.pt")
    gen = model.sample(n_gen, seed + 1000)
    fin = np.isfinite(gen).all(1)

    in_test = lambda g: z_of(g, d) > K_VAL
    acc, req = sample_region(model, in_test, n_req, seed + 2000)

    mu, sd = tr.mean(0), tr.std(0)
    t = lambda x: (x - mu) / sd
    test, gtest = f[lab == "test"], gen[fin & in_test(gen)]
    dist = {name: (knn_distance(t(test), t(f[lab == ref]), KNN), knn_distance(t(gtest), t(f[lab == ref]), KNN))
            for name, ref in (("d_train", "train"), ("d_val", "val"))}
    bands = {}
    for name, (dr, dg) in dist.items():
        edges = np.r_[0, np.quantile(dr, [0.2, 0.4, 0.6, 0.8]), np.inf]
        bands[name] = mass_ratio(dr, dg, len(f), n_gen, edges)
    with torch.no_grad():
        nll = -model.log_prob(torch.as_tensor(test, dtype=torch.float32)).numpy()
    for name, (dr, _) in dist.items():
        for b in bands[name]:
            m = (dr >= b["lo"]) & (dr < b["hi"])
            b["ref_nll_median"] = float(np.median(nll[m])) if m.any() else None

    np.savez_compressed(out / "samples.npz", px_ref=px_of(f), lab=lab.astype(str), px_gen=px_of(gen[fin]),
                        px_region=px_of(acc), d_train_ref=dist["d_train"][0], d_train_gen=dist["d_train"][1],
                        d_val_ref=dist["d_val"][0], d_val_gen=dist["d_val"][1])
    res = {"data": d["data"], "rule": _rule(d), "split_rows": {k: int((lab == k).sum()) for k in ("train", "val_interp", "val", "test")},
           "model": {"flow": "ship_nf.flow RealNVP, 6 couplings, 64 hidden", "epochs": epochs, "seed": seed,
                     "final_train_nll": losses[-1], "checkpoint": str((out / "flow.pt").resolve()),
                     "checkpoint_sha256": hashlib.sha256((out / "flow.pt").read_bytes()).hexdigest()},
           "transform": {"fit_on": "train rows only", "mean": mu.tolist(), "std": sd.tolist(), "knn": KNN},
           "generated": {"total_before_filter": n_gen, "finite": int(fin.sum()), "in_test_region": len(gtest)},
           "region_request": {"call": f"sample_region(model, lambda g: z_of(g, d) > {K_VAL}, n_req={n_req}, seed={seed + 2000})", **req},
           "bands": bands, "commit": _commit(), "geometry": "none: afterMS surface, not P2 (#43)"}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return {k: res[k] for k in ("split_rows", "generated", "region_request", "commit")}


def learning_curve(npz: Path, out: Path, sizes=(250, 1000, 4000, 13000), seeds=(1, 2, 3), n_gen: int = 50000,
                   epochs: int = 400) -> dict:
    """Demo 2: same flow, epochs and n_gen; only the number of distinct train rows changes. Scored on validation.

    Rows are equally weighted, so the training ESS = N. Test rows are not touched.
    """
    rows = []
    for sd in seeds:
        d = load(npz, sd)
        f, lab = d["f"], d["lab"]
        idx = np.flatnonzero(lab == "train")
        r_val = float((lab == "val").mean())
        for n in sizes:
            sub = f[np.random.default_rng(sd).choice(idx, min(n, len(idx)), replace=False)]
            model, _ = flow.fit(sub, sd, epochs=epochs)
            g = model.sample(n_gen, sd + 1000)
            zg = z_of(g, d)  # non-finite rows give nan and fall in no band
            with torch.no_grad():
                nll = -model.log_prob(torch.as_tensor(f[lab == "val_interp"], dtype=torch.float32)).numpy()
            g_val = int(((zg > K_TRAIN) & (zg <= K_VAL)).sum())
            rows.append({"seed": sd, "n_train_distinct": len(sub), "ess": float(len(sub)), "n_gen": n_gen,
                         "val_gen": g_val, "val_ref_frac": r_val, "val_mass_ratio": g_val / n_gen / r_val,
                         "val_interp_nll_median": float(np.median(nll))})
    res = {"data": d["data"], "rule": _rule(d), "epochs": epochs, "ess_note": "equal weights: ESS = N",
           "rows": rows, "commit": _commit()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "learning_curve.json").write_text(json.dumps(res, indent=1))
    return {"runs": len(rows), "commit": res["commit"]}
