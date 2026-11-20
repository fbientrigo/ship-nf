"""Figure 4: flow/real mass ratio in the test region per distance band, vs d_train (left) and d_val (right).
Reads result.json of ``ship_nf tail-demo`` (no training, no distances recomputed).

    .venv/bin/python scripts/fig4_mass_vs_distance.py RESULT_DIR --out fig4.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
res = json.loads((a.result / "result.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, key, lab in ((axes[0], "d_train", "distance to 5th nearest train row"),
                     (axes[1], "d_val", "distance to 5th nearest validation-shell row")):
    bands = res["bands"][key]
    x = range(len(bands))
    ratio = [b["ratio"] if b["ratio"] is not None else float("nan") for b in bands]
    ax.bar(x, ratio, color="C0")
    ax.axhline(1, color="grey", ls="--")
    for i, b in enumerate(bands):
        ax.text(i, ratio[i] * 1.3 if ratio[i] > 0 else 1e-4, f"ref {b['ref']}\nNF {b['gen']}\n{ratio[i]:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x), [f"{b['lo']:.2f}-{b['hi']:.2f}" if b["hi"] != float("inf") else f">{b['lo']:.2f}" for b in bands],
                  fontsize=7)
    ax.set_xlabel(lab + " (train-standardised)")
axes[0].set_yscale("log")
axes[0].set_ylim(1e-4, 3)
axes[0].set_ylabel("(NF in band / NF total) / (ref in band / ref total)")
fig.suptitle(f"Test region |z(px)|>2.5: NF recovers mass where ratio = 1. Totals: ref {res['bands']['d_train'][0]['ref_total']}, "
             f"NF {res['bands']['d_train'][0]['gen_total']} (before filter)", fontsize=9)
fig.tight_layout()
fig.savefig(a.out, dpi=150)
