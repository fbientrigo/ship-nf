"""Figure 3: px of real rows vs flow samples, both as a fraction of their own total, regions shaded.
Reads samples.npz + result.json of ``ship_nf tail-demo`` (no training).

    .venv/bin/python scripts/fig3_tail_histogram.py RESULT_DIR --out fig3.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
res, s = json.loads((a.result / "result.json").read_text()), np.load(a.result / "samples.npz")
med, sig = res["rule"]["median"], res["rule"]["sigma"]
ref, gen = s["px_ref"], s["px_gen"]
n_gen_total = res["generated"]["total_before_filter"]
lo, hi = np.quantile(ref, [0.0005, 0.9995])
bins = np.linspace(lo, hi, 121)
fig, ax = plt.subplots(figsize=(8, 4.5))
for (k0, k1), c, name in (((2.5, 99), "#f4cccc", "test |z|>2.5"), ((2.0, 2.5), "#fff2cc", "validation 2<|z|<=2.5")):
    ax.axvspan(max(lo, med - k1 * sig), med - k0 * sig, color=c, label=name)
    ax.axvspan(med + k0 * sig, min(hi, med + k1 * sig), color=c)
ax.axvspan(med - 2 * sig, med + 2 * sig, color="#d9ead3", label="train centre |z|<=2")
ax.hist(ref, bins, weights=np.full(len(ref), 1 / len(ref)), histtype="step", color="k", lw=1.5,
        label=f"reference (all {len(ref)} rows)")
ax.hist(gen, bins, weights=np.full(len(gen), 1 / n_gen_total), histtype="step", color="C0", lw=1.5,
        label=f"NF ({n_gen_total} samples, / total before filter)")
ax.set_yscale("log")
ax.set_xlabel("px (GeV/c), afterMS, pdg %d" % res["data"]["pdg"])
ax.set_ylabel("fraction of total per bin")
r = res["region_request"]
ax.set_title(f"Trained on the centre only. Request in test region: {r['requested']} asked, {r['generated']} generated, "
             f"{r['accepted']} accepted", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(a.out, dpi=150)
