"""Slide 7: does more centre data lower the error on the tail? NLL of real tail muons per distance band, for each N.

Retrains the 12 flows of ``tail_demo.learning_curve`` (N = 250, 1000, 4000, 13000; seeds 1-3) with the same code path
and seeds, and asserts that each reproduces its ``val_gen`` count in ``learning_curve.json`` (same model). Then it
evaluates the NLL of the 1587 real test rows (|zpx| > 2.5; identical for every seed, the seed only picks val_interp).

Distance axis: ``d_train_ref`` from ``samples.npz`` and the band edges of ``result.json`` (distance to the 5th
nearest row of the full train set of split seed 1), the same axis as the slide 8 figure for every N. A flow trained
on a subsample of N rows has seen fewer of those rows, so this is the distance to the pool, not to its own data::

    .venv/bin/python plots/cm_lc_error_vs_d.py docs/meeting_1121 --out docs/meeting_1121/cm
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ship_nf import flow
from ship_nf.tail_demo import K_TRAIN, K_VAL, load, z_of

NF, REF, HI = "#1D6B7D", "#6B7280", "#B45309"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.transparent": True, "savefig.dpi": 300})

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
lc = json.loads((a.result / "learning_curve.json").read_text())
res = json.loads((a.result / "result.json").read_text())
d_ref = np.load(a.result / "samples.npz")["d_train_ref"]
bands = res["bands"]["d_train"]
which = [(d_ref >= b["lo"]) & (d_ref < b["hi"]) for b in bands]

rows = []
for r in lc["rows"]:  # same steps as tail_demo.learning_curve
    sd, n = r["seed"], r["n_train_distinct"]
    d = load(Path(lc["data"]["path"]), sd)
    f, lab = d["f"], d["lab"]
    idx = np.flatnonzero(lab == "train")
    model, _ = flow.fit(f[np.random.default_rng(sd).choice(idx, min(n, len(idx)), replace=False)], sd, epochs=lc["epochs"])
    zg = z_of(model.sample(r["n_gen"], sd + 1000), d)
    val_gen = int(((zg > K_TRAIN) & (zg <= K_VAL)).sum())
    assert val_gen == r["val_gen"], (sd, n, val_gen, r["val_gen"])  # reproduces the saved model
    with torch.no_grad():
        nll = -model.log_prob(torch.as_tensor(f[lab == "test"], dtype=torch.float32)).numpy()
    rows.append({"seed": sd, "n": n, "val_gen": val_gen, "test_nll_median": float(np.median(nll)),
                 "band_nll_median": [float(np.median(nll[m])) for m in which]})
    print(rows[-1], flush=True)

out = {"bands": [{k: b[k] for k in ("lo", "hi", "ref")} for b in bands], "rows": rows}
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "lc_error_vs_d.json").write_text(json.dumps(out, indent=1))

fig, ax = plt.subplots(figsize=(8.9, 4.8))
Ns = sorted({r["n"] for r in rows})
y = np.array([[[r["band_nll_median"][b] for r in rows if r["n"] == n] for n in Ns] for b in range(len(bands))])  # band x N x seed
shades = ["#C9E0E4", "#A9CDD4", "#6FA8B3", "#3A8494", NF]  # near -> far
for b, c in enumerate(shades):
    lab = f"d {bands[b]['lo']:.2f}–{bands[b]['hi']:.2f}" if np.isfinite(bands[b]["hi"]) else f"d > {bands[b]['lo']:.2f}"
    ax.plot(Ns, np.median(y[b], 1), "o-", color=c, lw=2.5, ms=7, label=lab)
    for j in range(y.shape[2]):
        ax.plot(Ns, y[b, :, j], "o", color=c, ms=4, alpha=0.6)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=10)
ax.set_yticks([-10, 0, 10, 100], ["−10", "0", "10", "100"])
ax.set_xticks(Ns, [f"{n:,}".replace(",", " ") for n in Ns])
ax.minorticks_off()
ax.set_xlabel("distinct training states N")
ax.set_ylabel("median NLL of real tail muons\n(nats, higher = worse)")
ax.legend(title="distance to train (band)", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
fig.savefig(a.out / "slide7_error_vs_distance_by_N.png", bbox_inches="tight", pad_inches=0.05)
