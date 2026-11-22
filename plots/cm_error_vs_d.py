"""Slide 8: model error of the real tail points versus their distance to the training data. Evaluates; trains nothing.

Error = NLL (nats) of each real test row (|zpx| > 2.5, 1587 rows) under the saved tail-demo ``flow.pt``; distance =
``d_train`` / ``d_val`` from ``samples.npz`` (same row order as ``f[lab == "test"]``, ``tail_demo.run``). Bands are
the five equal-count bands of ``result.json``; the script asserts that its band medians equal the saved
``ref_nll_median``. Reference line: median NLL of the held-out centre rows (val_interp), same checkpoint::

    .venv/bin/python plots/cm_error_vs_d.py docs/meeting_1121 --out docs/meeting_1121/cm
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ship_nf import flow
from ship_nf.tail_demo import load

NF, REF, HI = "#1D6B7D", "#6B7280", "#B45309"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.transparent": True, "savefig.dpi": 300})
XL = {"d_train": "distance to 5th nearest train row", "d_val": "distance to 5th nearest\nvalidation-shell row"}

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
smp = np.load(a.result / "samples.npz")
with torch.no_grad():
    nll = {k: -model.log_prob(torch.as_tensor(d["f"][d["lab"] == k], dtype=torch.float32)).numpy() for k in ("test", "val_interp")}
base = float(np.median(nll["val_interp"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, key in zip(axes, ("d_train", "d_val")):
    dist = smp[f"{key}_ref"]
    ax.scatter(dist, nll["test"], s=6, color=REF, alpha=0.35, lw=0, label="real tail muons (1 587)")
    for b in res["bands"][key]:
        m = (dist >= b["lo"]) & (dist < b["hi"])
        q25, q50, q75 = np.quantile(nll["test"][m], [0.25, 0.5, 0.75])
        assert np.isclose(q50, b["ref_nll_median"], rtol=1e-5), (key, q50, b["ref_nll_median"])
        xm = np.median(dist[m])
        ax.errorbar(xm, q50, yerr=[[q50 - q25], [q75 - q50]], fmt="o", color=NF, ms=9, lw=2.5, capsize=4,
                    label="band median, 25-75 %" if b["lo"] == 0 else None)
    ax.axhline(base, color=HI, ls="--", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel(XL[key])
axes[0].set_ylim(base - 14, None)
axes[1].text(0.98, base - 2, f"centre rows (val_interp), median {base:.1f}", transform=axes[1].get_yaxis_transform(),
             color=HI, ha="right", va="top")
axes[0].set_ylabel("NLL per muon (nats)\nhigher = worse")
axes[0].legend(loc="upper left", frameon=False, markerscale=1.5)
fig.tight_layout(w_pad=1.5)
fig.savefig(a.out / "slide8_error_vs_distance.png", bbox_inches="tight", pad_inches=0.05)
print({"val_interp_median": base, "test_rows": len(nll["test"]),
       "band_medians": {k: [b["ref_nll_median"] for b in res["bands"][k]] for k in ("d_train", "d_val")}})
