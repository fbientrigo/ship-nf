"""Figure: tail mass ratio and interior NLL per design (A Gaussian/affine, B Student-t/affine, C Gaussian/spline),
from saved results only.

    .venv/bin/python scripts/fig_tbase.py results/tbase_1120 [results/nsf_1120] --out FIG.png

Reads ``[ABC]_s*.json`` written by ``exp_tbase.py`` in every given directory; writes the PNG and a PDF next to it.
Validation shell is the main panel, the already inspected test exterior the secondary one. Never trains.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

LABEL = {"A": "A: affine, Gaussian base", "B": "B: affine, Student-t base (nu = 5)",
         "C": "C: spline (RQS, 8 bins, B = 5), Gaussian base"}
COL = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
MARK = {1: "o", 2: "s"}
FLOOR = 3e-4  # where a zero count is drawn


def xpos(r, designs, i=0, width=0.6):
    k = designs.index(r["design"])
    return i + (k - (len(designs) - 1) / 2) * width / len(designs) + (r["seed"] - 1.5) * 0.07


def tails(ax, runs, designs, k, title):
    for i, side in enumerate(("left", "right")):
        for r in runs:
            t, x = r["tails"][f"{k}_{side}"], xpos(r, designs, i)
            zero = t["gen"] == 0
            y = FLOOR if zero else t["R"]
            ax.scatter(x, y, marker="v" if zero else MARK[r["seed"]], s=45, color=COL[r["design"]],
                       edgecolor="k" if zero else "none", zorder=3)
            ax.annotate(str(t["gen"]), (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7,
                        color=COL[r["design"]])
    ax.axhline(1, color="k", lw=1, ls="--")
    ax.text(1.45, 1.1, "R = 1", ha="right", va="bottom", fontsize=8)
    ax.set_yscale("log")
    any_zero = any(r["tails"][f"{k}_{s}"]["gen"] == 0 for r in runs for s in ("left", "right"))
    ax.set_ylim(FLOOR / 2 if any_zero else 1e-3, 3)
    ax.set_xlim(-0.5, 1.5)
    r0 = runs[0]["tails"]
    ax.set_xticks([0, 1], [f"left tail\nreal {r0[f'{k}_left']['ref']}/{r0[f'{k}_left']['ref_total']}",
                           f"right tail\nreal {r0[f'{k}_right']['ref']}/{r0[f'{k}_right']['ref_total']}"])
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", which="major", alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [json.loads(p.read_text()) for d in a.results for p in sorted(d.glob("[ABC]_s*.json"))]
    designs = sorted({r["design"] for r in runs})
    gpus = sorted({json.loads((d / "env.json").read_text())["gpu"] for d in a.results})
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.35, 1, 0.55, 0.55]})
    tails(axes[0], runs, designs, "val", "validation shell, 2 < |z| <= 2.5 (main)")
    tails(axes[1], runs, designs, "test", "test exterior, |z| > 2.5 (exploratory)")
    axes[0].set_ylabel("mass ratio R = (gen frac) / (real frac)")
    for ax, stat in zip(axes[2:], ("median", "mean")):
        for r in runs:
            ax.scatter(xpos(r, designs, 0, 1.2) + 0.5, r["val_interp_nll"][stat], marker=MARK[r["seed"]], s=45,
                       color=COL[r["design"]])
        ax.set_xticks([xpos({"design": d, "seed": 1.5}, designs, 0, 1.2) + 0.5 for d in designs], designs)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"interior {stat} NLL", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[2].set_ylabel("NLL on val_interp, |z| <= 2 (nats, lower is better)")
    h = [plt.Line2D([], [], ls="", marker="o", color=COL[d], label=LABEL[d]) for d in designs]
    h += [plt.Line2D([], [], ls="", marker=MARK[s], color="gray", label=f"seed {s}") for s in MARK]
    h += [plt.Line2D([], [], ls="", marker="v", color="w", markeredgecolor="k", label="0 generated")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8, frameon=False)
    n = runs[0]["generated"]["total"]
    fig.suptitle(f"Flow designs trained on centre only: afterMS mu-, px ({n:,} unfiltered draws per run, "
                 f"numbers = generated counts, {', '.join(gpus)})", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(a.out, dpi=200)
    fig.savefig(a.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
