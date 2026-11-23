"""Slide 8: how much the NF distribution error grows with distance from its training data. Samples; trains nothing.

Same experiment and checkpoint as ``tail_demo.run`` (``result.json``): flow trained on the centre ``|zpx| <= 2`` of
the afterMS mu- sample minus a random 10 % (val_interp); validation shell ``2 < |zpx| <= 2.5``; test ``|zpx| > 2.5``.

- Distance ``d``: to the 5th-nearest train row, features standardised by train mean/std (``d_train`` of
  ``result.json``), the same function for real and NF rows.
- Real rows: every held-out row (val_interp, shell, test). NF rows: 10^7 samples, seeds 3000-3009; those in the
  centre are kept with probability 10 %, the hold-out rate there, so both sides describe the same population.
- Error per equal-count ``d`` band: ``diagnostics.error_vs_distance`` on ``(x, y, px, py, pz)`` scaled by median and
  1.4826 MAD of train rows, with the real-vs-real floor. A distance between distributions in those robust units, not an
  error in m or GeV/c. Rows unweighted (``w`` OPEN; training was unweighted too).
- Five error measures (``METRICS`` and the mass ratio), each drawn twice: marks at the last training / validation
  distance (``max``) or at their 95th percentiles (``p95``). Training rows' distance is to their 5th-nearest *other*
  train row. Writes ``slide8_<metric>_<max|p95>.{png,pdf}`` and ``error_vs_d.json``::

    .venv/bin/python plots/cm_swd_vs_d.py docs/meeting_1121 --out docs/meeting_1121/cm
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ship_nf import flow
from scipy.spatial import cKDTree

from ship_nf.diagnostics import coverage, error_vs_distance, knn_two_sample_accuracy, sliced_wasserstein
from ship_nf.extrapolation import knn_distance
from ship_nf.tail_demo import INTERP, K_TRAIN, KNN, load, z_of

NF, REF, HI = "#1D6B7D", "#6B7280", "#B45309"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.transparent": True, "savefig.dpi": 300})

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
res = json.loads((a.result / "result.json").read_text())
ckpt = a.result / "flow.pt"
assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == res["model"]["checkpoint_sha256"]
model = flow.Flow(5)
model.load_state_dict(torch.load(ckpt))
d = load(Path(res["data"]["path"]), res["model"]["seed"])
f, lab = d["f"], d["lab"]
tr = f[lab == "train"]
mu, sd = tr.mean(0), tr.std(0)


def phys(g):  # x, y, tx, ty, log p -> x, y, px, py, pz
    pz = np.exp(g[:, 4]) / np.sqrt(1 + g[:, 2] ** 2 + g[:, 3] ** 2)
    return np.c_[g[:, 0], g[:, 1], g[:, 2] * pz, g[:, 3] * pz, pz]


raw = np.load(res["data"]["path"])["muons"]
raw = raw[(raw[:, 2] > 0) & (raw[:, 6] == 13)]
assert np.allclose(phys(f), raw[:, [3, 4, 0, 1, 2]], rtol=1e-10, atol=1e-12)
med = np.median(phys(tr), 0)
scale = 1.4826 * np.median(np.abs(phys(tr) - med), 0)
X = lambda g: (phys(g) - med) / scale
dist = lambda g: knn_distance((g - mu) / sd, (tr - mu) / sd, KNN)

held = lab != "train"
real, d_real = f[held], dist(f[held])
assert np.allclose(d_real[lab[held] == "test"], np.load(a.result / "samples.npz")["d_train_ref"])
d_val = d_real[lab[held] != "test"]
d_tr = cKDTree((tr - mu) / sd).query((tr - mu) / sd, KNN + 1)[0][:, -1]  # 5th-nearest *other* train row
MARKS = {"max": (d_tr.max(), d_val.max(), "last"), "p95": (np.quantile(d_tr, 0.95), np.quantile(d_val, 0.95), "95 % of")}

rng, gen = np.random.default_rng(5), []
for i in range(10):
    g = model.sample(1_000_000, 3000 + i)
    g = g[np.isfinite(g).all(1)]
    gen.append(g[(z_of(g, d) > K_TRAIN) | (rng.random(len(g)) < INTERP)])  # thin the centre like the hold-out
gen = np.concatenate(gen)
d_gen = dist(gen)

edges = np.r_[0, np.quantile(d_real, np.linspace(0, 1, 9)[1:-1]), np.inf]
px_gev = lambda u, v: float(np.abs(np.sort(u[:, 2]) - np.sort(v[:, 2])).mean() * scale[2])
METRICS = {  # name: (function on scaled equal-size samples, y label, log y)
    "sw": (sliced_wasserstein, "distribution error, sliced W1\n(x, y, px, py, pz; robust units)", True),
    "classifier": (knn_two_sample_accuracy, "classifier accuracy, NF vs real\n(0.5 = cannot tell apart)", False),
    "px_shift": (px_gev, "px distribution shift, W1 (GeV/c)", True),
    "uncovered": (lambda u, v: 1 - coverage(u, v), "real muons with no NF muon\nnearby (fraction)", False),
}
bands = error_vs_distance(X(real), X(gen), d_real, d_gen, edges, {k: v[0] for k, v in METRICS.items()}, seed=11)
for b in bands:
    m = (d_real >= b["lo"]) & (d_real < b["hi"])
    P, Q = b["real"] / len(real), b["gen"] / len(gen)  # same population on both sides (centre thinned to 10 %)
    s_ = np.sqrt((1 - P) / (len(real) * P) + (1 - Q) / (len(gen) * Q))
    b.update(d_median=float(np.median(d_real[m])), test_share=float((lab[held][m] == "test").mean()),
             mass=[Q / P * np.exp(-s_), Q / P, Q / P * np.exp(s_)])
out = {"marks": {k: [float(v[0]), float(v[1])] for k, v in MARKS.items()}, "nf_kept": len(gen), "real_heldout": len(real),
       "bands": bands, "scale": {"median": med.tolist(), "robust_sigma": scale.tolist()}}
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "error_vs_d.json").write_text(json.dumps(out, indent=1))
METRICS["mass"] = (None, "NF muons / real muons in the band\n(1 = right amount)", True)

x = np.array([b["d_median"] for b in bands])
for name, (_, ylabel, logy) in METRICS.items():
    for mark, (d_t, d_v, word) in MARKS.items():
        fig, ax = plt.subplots(figsize=(8.9, 4.8))
        if name == "mass":
            y = np.array([b["mass"] for b in bands])
            ax.fill_between(x, y[:, 0], y[:, 2], color=NF, alpha=0.2, lw=0)
            ax.plot(x, y[:, 1], "o-", color=NF, lw=2.5, ms=7, label="NF vs held-out real muons")
            ax.axhline(1, color=REF, ls="--", lw=1.5, label="right amount")
        else:
            for key, c, lbl in ((name, NF, "NF vs held-out real muons"), (name + "_floor", REF, "real vs real (statistical floor)")):
                y = np.array([b[key] for b in bands])
                ax.fill_between(x, y[:, 0], y[:, 2], color=c, alpha=0.2, lw=0)
                ax.plot(x, y[:, 1], "o-", color=c, lw=2.5, ms=7, label=lbl)
        tf = ax.get_xaxis_transform()
        for dv, c, ls, txt, k, ha in ((d_t, REF, "--", f"{word}\ntraining", 1.05, "left"), (d_v, HI, "-", f"{word}\nvalidation", 0.95, "right")):
            ax.axvline(dv, color=c, ls=ls, lw=2)  # training lies right of validation here: labels on the outer sides
            ax.text(dv * k, 0.03, txt, transform=tf, ha=ha, va="bottom", color=c, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(x.min() * 0.8, max(x.max(), d_t, d_v) * 1.6)
        ax.set_xlabel("distance to the training muons (5th nearest)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", frameon=False)
        for ext in ("png", "pdf"):
            fig.savefig(a.out / f"slide8_{name}_{mark}.{ext}", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
print(json.dumps({k: v for k, v in out.items() if k != "bands"}, indent=1))
