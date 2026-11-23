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
- Vertical line: 95th percentile of ``d`` over validation rows (the maximum, 9.9, is one isolated centre row)::

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
from ship_nf.diagnostics import error_vs_distance
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
d_val95 = float(np.quantile(d_real[lab[held] != "test"], 0.95))  # max is set by a few isolated centre rows

rng, gen = np.random.default_rng(5), []
for i in range(10):
    g = model.sample(1_000_000, 3000 + i)
    g = g[np.isfinite(g).all(1)]
    gen.append(g[(z_of(g, d) > K_TRAIN) | (rng.random(len(g)) < INTERP)])  # thin the centre like the hold-out
gen = np.concatenate(gen)
d_gen = dist(gen)

edges = np.r_[0, np.quantile(d_real, np.linspace(0, 1, 9)[1:-1]), np.inf]
bands = error_vs_distance(X(real), X(gen), d_real, d_gen, edges, seed=11)
for b in bands:
    m = (d_real >= b["lo"]) & (d_real < b["hi"])
    b.update(d_median=float(np.median(d_real[m])), test_share=float((lab[held][m] == "test").mean()))
out = {"d_val95": d_val95, "nf_kept": len(gen), "real_heldout": len(real), "bands": bands,
       "scale": {"median": med.tolist(), "robust_sigma": scale.tolist()}}
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "swd_vs_d.json").write_text(json.dumps(out, indent=1))

fig, ax = plt.subplots(figsize=(8.9, 4.8))
x = np.array([b["d_median"] for b in bands])
for key, c, lbl in (("error", NF, "NF vs held-out real muons"), ("floor", REF, "real vs real (statistical floor)")):
    y = np.array([b[key] for b in bands])
    ax.fill_between(x, y[:, 0], y[:, 2], color=c, alpha=0.2, lw=0)
    ax.plot(x, y[:, 1], "o-", color=c, lw=2.5, ms=7, label=lbl)
ax.axvline(d_val95, color=HI, lw=2)
ax.axvspan(x.min() * 0.8, d_val95, color=HI, alpha=0.06, lw=0)
ax.text(d_val95 * 0.93, 0.97, "95 % of\nvalidation", transform=ax.get_xaxis_transform(), ha="right", va="top", color=HI)
ax.text(d_val95 * 1.07, 0.97, "mostly\nunseen test", transform=ax.get_xaxis_transform(), ha="left", va="top", color=HI)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x.min() * 0.8, None)
ax.set_xlabel("distance to the training muons (5th nearest)")
ax.set_ylabel("distribution error, sliced W1\n(x, y, px, py, pz; robust units)")
ax.legend(loc="upper left", frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(a.out / f"slide8_swd_vs_distance.{ext}", bbox_inches="tight", pad_inches=0.05)
print(json.dumps(out, indent=1))
