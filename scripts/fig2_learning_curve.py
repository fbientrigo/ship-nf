"""Figure 2: validation tail quality vs distinct training states. Reads learning_curve.json only (no training).

    .venv/bin/python scripts/fig2_learning_curve.py RESULT_DIR [--loop-delivery DIR] --out fig2.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--loop-delivery", type=Path, help="a propose-nf delivery: its tilt ESS goes in the caption box")
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
res = json.loads((a.result / "learning_curve.json").read_text())
rows = res["rows"]
sizes = sorted({r["n_train_distinct"] for r in rows})
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, key, lab in ((axes[0], "val_mass_ratio", "flow mass / real mass in validation shell"),
                     (axes[1], "val_interp_nll_median", "median NLL of val_interp rows (nats)")):
    for sd in sorted({r["seed"] for r in rows}):
        rr = sorted((r for r in rows if r["seed"] == sd), key=lambda r: r["n_train_distinct"])
        ax.plot([r["n_train_distinct"] for r in rr], [r[key] for r in rr], "o-", alpha=0.5, label=f"seed {sd}")
    m = [np.mean([r[key] for r in rows if r["n_train_distinct"] == n]) for n in sizes]
    ax.plot(sizes, m, "k-", lw=2, label="mean")
    ax.set_xscale("log")
    ax.set_xlabel("distinct training states N (equal weights: ESS = N)")
    ax.set_ylabel(lab)
axes[0].axhline(1, color="grey", ls="--")
axes[0].legend(fontsize=8)
note = f"afterMS px, pdg {res['data']['pdg']}; val shell {res['rule']['val']}; same flow, {res['epochs']} epochs, " \
       f"{rows[0]['n_gen']} samples per fit"
if a.loop_delivery:
    g = json.loads((a.loop_delivery / "manifest.json").read_text())["generator"]
    note += f"\nFairShip loop tilt: {g['rows']} distinct states, {g.get('distinct_progenitors', '?')} progenitors, " \
            f"Kish ESS {g['tilt_kish_ess']:.1f} before resampling, {g['config']['n_train']} resampled rows, {g['n']} proposals"
fig.suptitle("Training size vs tail quality on validation (test untouched)", fontsize=11)
fig.text(0.01, 0.01, note, fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(a.out, dpi=150)
