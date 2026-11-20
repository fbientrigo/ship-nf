"""Figure A: exterior mass ratio R per generation, retraining on unfiltered vs edge-selected own samples.
Reads edge_reuse.json of ``ship_nf edge-reuse``. No training.

    .venv/bin/python plots/figA_edge_reuse.py docs/meeting_1121/edge_reuse.json --out docs/meeting_1121/figA_edge_reuse
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 8})  # 16:9 slide
ARMS = {"control": ("C7", "unfiltered own samples"), "edge": ("C3", "edge-selected own samples")}
SEED_LS = ["-", "--", ":", "-."]

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--out", required=True, type=Path, help="path without extension; writes .png and .pdf")
a = ap.parse_args()
res = json.loads(a.result.read_text())
st = res["settings"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, side in zip(axes, ("left", "right")):
    for k, run in enumerate(res["runs"]):
        for arm, (c, lab) in ARMS.items():
            pts = [(0, run["gen0"][side]["R"])] + [(r["generation"], r[side]["R"]) for r in run[arm]]
            ax.plot(*zip(*pts), color=c, ls=SEED_LS[k], marker="o", ms=4,
                    label=f"{lab}, seed {run['seed']}")
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.text(st["rounds"], 1.1, "R = 1", ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2)
    ax.set_xticks(range(st["rounds"] + 1))
    ax.set_xlabel("generation (0 = centre-trained NF)")
    n = res["ref_rows"][side]
    ax.set_title(f"{side} exterior, z(px) {'<' if side == 'left' else '>'} {'-' if side == 'left' else ''}2.5 "
                 f"({n} real rows)")
axes[0].set_ylabel("R = NF exterior fraction\n/ reference exterior fraction")
axes[0].legend(loc="lower right")
fig.suptitle(f"μ−, real centre rows + m = {st['m']} own samples per generation; "
             f"each point scored on {st['n_eval']} fresh unfiltered samples", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(a.out.with_suffix(f".{ext}"), dpi=150)
rows = [(run["seed"], arm, r["generation"], r["m_used"], r["request"]["accepted"], r["request"]["generated"],
         r["left"]["gen"], round(r["left"]["R"], 4), r["right"]["gen"], round(r["right"]["R"], 4),
         round(r["val_interp_nll_median"], 3))
        for run in res["runs"] for arm in ARMS for r in run[arm]]
print("seed arm gen m_used accepted/generated left_n R_left right_n R_right nll_val_interp_median")
for run in res["runs"]:
    g = run["gen0"]
    print(run["seed"], "g0", 0, "-", "-", g["left"]["gen"], round(g["left"]["R"], 4), g["right"]["gen"],
          round(g["right"]["R"], 4), round(g["val_interp_nll_median"], 3))
for r in rows:
    print(*r[:4], f"{r[4]}/{r[5]}", *r[6:])
