"""Figure B: NF/reference mass ratio in the px exterior per distance band, against d_train (left) and d_val (right).
Reads result.json and samples.npz of ``ship_nf tail-demo``. No training, no distances recomputed.

    .venv/bin/python plots/figB_mass_vs_distance.py docs/meeting_1121 --out docs/meeting_1121/figB_mass_vs_distance
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 9})  # 16:9 slide
FLOOR = 3e-4  # where a zero ratio is drawn on the log axis

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--out", required=True, type=Path, help="path without extension; writes .png and .pdf")
a = ap.parse_args()
res = json.loads((a.result / "result.json").read_text())
smp = np.load(a.result / "samples.npz")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
summary = {}
for ax, key, xlab in ((axes[0], "d_train", "distance to 5th nearest training row"),
                      (axes[1], "d_val", "distance to 5th nearest validation-shell row")):
    bands, d_ref, d_gen = res["bands"][key], smp[f"{key}_ref"], smp[f"{key}_gen"]
    beyond = int((d_gen > d_ref.max()).sum())
    summary[key] = {"max_ref_distance": float(d_ref.max()), "nf_beyond_max_ref": beyond, "nf_in_exterior": len(d_gen)}
    for i, b in enumerate(bands):
        r = b["ratio"]
        if r is None:
            ax.text(i, FLOOR, "no estimate", ha="center", va="bottom", fontsize=8, rotation=90)
            continue
        ax.bar(i, max(r, FLOOR), color="C0" if r > 0 else "none", edgecolor="C0", hatch=None if r > 0 else "//")
        ax.text(i, max(r, FLOOR) * 1.15, f"{r:.3f}\nref {b['ref']}\nNF {b['gen']}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.text(len(bands) - 0.5, 1.1, "R = 1", ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_xticks(range(len(bands)), [f"{b['lo']:.2f}–{b['hi']:.2f}" if np.isfinite(b["hi"]) else
                                      f"≥ {b['lo']:.2f} (open)\nreal max {d_ref.max():.2f}\nNF beyond: {beyond}"
                                      for b in bands], fontsize=8)
    ax.set_xlabel(xlab)
axes[0].set_yscale("log")
axes[0].set_ylim(FLOOR / 1.5, 4)
axes[0].set_ylabel("R = (NF in band / all NF)\n/ (ref in band / all ref)")
b0 = res["bands"]["d_train"][0]
fig.suptitle(f"Exterior |z(px)| > 2.5, μ−, centre-trained NF. Totals before the exterior cut: "
             f"ref {b0['ref_total']}, NF {b0['gen_total']}", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(a.out.with_suffix(f".{ext}"), dpi=150)
a.out.with_suffix(".json").write_text(json.dumps({"bands": res["bands"], "beyond_max": summary}, indent=1))
print(json.dumps(summary, indent=1))
