"""Edge-reuse pilot: does retraining on the flow's own edge samples move mass into the exterior of px?

Starts from the centre-trained checkpoint of ``tail-demo`` (generation 0) and runs two paired arms per seed:

* ``control``: real train rows + ``m`` unfiltered samples of the arm's previous flow.
* ``edge``:    real train rows + ``m`` samples of the arm's previous flow with ``K_TRAIN < |z_px| <= K_VAL``.

Both arms use the same real rows, the same training size, the same ``flow.fit`` seed, optimizer and epochs (full
batch, so the same number of steps). If the edge search accepts fewer than ``m`` within ``max_props`` proposals, both
arms use the number accepted. From round 2 each arm samples from its own checkpoint. Val and test rows never enter
training. Every checkpoint is scored on ``n_eval`` fresh, unfiltered samples:
``R_side = (exterior samples on that side / n_eval) / (real exterior rows on that side / all real rows)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow
from ship_nf.tail_demo import K_TRAIN, K_VAL, _commit, load, px_of, sample_region


def _signed_z(f, d):
    return (px_of(f) - d["median"]) / d["sigma"]


def score(model: flow.Flow, d: dict, ref: dict, val_interp: np.ndarray, n_eval: int, seed: int) -> dict:
    """Exterior counts per side for ``n_eval`` fresh samples (non-finite rows count in the total, in no side)."""
    z = _signed_z(model.sample(n_eval, seed), d)
    with torch.no_grad():
        nll = -model.log_prob(torch.as_tensor(val_interp, dtype=torch.float32)).numpy()
    out = {"n_eval": n_eval, "val_interp_nll_median": float(np.median(nll)), "val_interp_nll_mean": float(nll.mean())}
    for side, mask in (("left", z < -K_VAL), ("right", z > K_VAL)):
        n = int(mask.sum())
        out[side] = {"gen": n, "R": n / n_eval / ref[side]}
    return out


def run(npz: Path, ckpt: Path, out: Path, seeds=(1, 2), rounds: int = 2, m: int | None = None,
        max_props: int = 200_000, batch: int = 20_000, n_eval: int = 200_000, epochs: int = 400) -> dict:
    d = load(npz, 1)  # split seed 1 = the split the checkpoint was trained on
    f, lab = d["f"], d["lab"]
    tr, vi = f[lab == "train"], f[lab == "val_interp"]
    z = _signed_z(f, d)
    ref = {"left": float((z < -K_VAL).mean()), "right": float((z > K_VAL).mean())}
    if m is None:  # edge arm gets the real edge/centre ratio: m / n_train = n_val / n_centre
        m = round(len(tr) * (lab == "val").sum() / np.isin(lab, ["train", "val_interp"]).sum())
    q0 = flow.Flow(f.shape[1])
    q0.load_state_dict(torch.load(ckpt))
    edge = lambda g: (lambda s: (s > K_TRAIN) & (s <= K_VAL))(np.abs(_signed_z(g, d)))
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for sd in seeds:
        rec = {"seed": sd, "gen0": score(q0, d, ref, vi, n_eval, 90_000 + sd), "control": [], "edge": []}
        q = {"control": q0, "edge": q0}
        for g in range(1, rounds + 1):
            s = 10_000 * sd + 1_000 * g
            e_rows, e_req = sample_region(q["edge"], edge, m, s, batch, max_props // batch)
            c_rows, c_req = sample_region(q["control"], lambda x: np.ones(len(x), bool), m, s, batch, 1)
            m_used = min(len(e_rows), len(c_rows))
            for arm, rows, req in (("control", c_rows, c_req), ("edge", e_rows, e_req)):
                q[arm], _ = flow.fit(np.vstack([tr, rows[:m_used]]), 100 * sd + g, epochs=epochs)
                torch.save(q[arm].state_dict(), out / f"seed{sd}_{arm}_g{g}.pt")
                rec[arm].append({"generation": g, "m_target": m, "m_used": m_used, "n_train": len(tr) + m_used,
                                 "request": req, **score(q[arm], d, ref, vi, n_eval, 90_000 + 10 * sd + g)})
        runs.append(rec)
    res = {"data": d["data"], "checkpoint_g0": str(Path(ckpt).resolve()), "ref_exterior_frac": ref,
           "ref_rows": {"total": len(f), "left": int((z < -K_VAL).sum()), "right": int((z > K_VAL).sum())},
           "settings": {"rounds": rounds, "m": m, "max_props": max_props, "batch": batch, "n_eval": n_eval,
                        "epochs": epochs, "edge": f"{K_TRAIN} < |z_px| <= {K_VAL}", "exterior": f"|z_px| > {K_VAL}",
                        "real_rows_in_training": "train only (val_interp, val, test held out)"},
           "runs": runs, "commit": _commit()}
    (out / "edge_reuse.json").write_text(json.dumps(res, indent=1))
    return res
