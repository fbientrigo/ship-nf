"""Figure: tail mass ratio and interior NLL, Gaussian (A) vs Student-t nu=5 (B) base, from saved results only.

    .venv/bin/python scripts/fig_tbase.py results/tbase_1120 --out results/tbase_1120/fig_tbase.png

Reads ``[AB]_s*.json`` written by ``exp_tbase.py``; writes the PNG and a PDF next to it. Never trains.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COL = {"A": "#1f77b4", "B": "#d62728"}
MARK = {1: "o", 2: "s"}
FLOOR = 3e-4  # where a zero count is drawn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(a.results.glob("[AB]_s*.json"))]
    env = json.loads((a.results / "env.json").read_text())
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.7]})
    for ax, k, name in ((axes[0], "val", "validation shell, 2 < |z| <= 2.5"),
                        (axes[1], "test", "test exterior, |z| > 2.5 (exploratory)")):
        r0 = runs[0]["tails"]
        for i, side in enumerate(("left", "right")):
            for r in runs:
                t = r["tails"][f"{k}_{side}"]
                x = i + (-0.2 if r["design"] == "A" else 0.2) + (r["seed"] - 1.5) * 0.16
                zero = t["gen"] == 0
                ax.scatter(x, FLOOR if zero else t["R"], marker="v" if zero else MARK[r["seed"]], s=45,
                           color=COL[r["design"]], edgecolor="k" if zero else "none", zorder=3)
                ax.annotate(str(t["gen"]), (x, FLOOR if zero else t["R"]), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=7, color=COL[r["design"]])
        ax.axhline(1, color="k", lw=1, ls="--")
        ax.text(1.45, 1.1, "R = 1", ha="right", va="bottom", fontsize=8)
        ax.set_yscale("log")
        any_zero = any(r["tails"][f"{k}_{s}"]["gen"] == 0 for r in runs for s in ("left", "right"))
        ax.set_ylim(FLOOR / 2 if any_zero else 2e-3, 3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1], [f"left tail\n(real {r0[f'{k}_left']['ref']} of {r0[f'{k}_left']['ref_total']})",
                               f"right tail\n(real {r0[f'{k}_right']['ref']} of {r0[f'{k}_right']['ref_total']})"])
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", which="major", alpha=0.3)
    axes[0].set_ylabel("mass ratio R = (gen frac) / (real frac)")
    ax = axes[2]
    for r in runs:
        x = (0 if r["design"] == "A" else 1) + (r["seed"] - 1.5) * 0.15
        ax.scatter(x, r["val_interp_nll"]["median"], marker=MARK[r["seed"]], s=45, color=COL[r["design"]])
    ax.set_xticks([0, 1], ["A", "B"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("median NLL, val_interp (nats)")
    ax.set_title("interior (|z| <= 2)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    h = [plt.Line2D([], [], ls="", marker="o", color=COL["A"], label="A: Gaussian base"),
         plt.Line2D([], [], ls="", marker="o", color=COL["B"], label="B: Student-t base, nu = 5"),
         plt.Line2D([], [], ls="", marker="o", color="gray", label="seed 1"),
         plt.Line2D([], [], ls="", marker="s", color="gray", label="seed 2"),
         plt.Line2D([], [], ls="", marker="v", color="w", markeredgecolor="k", label="0 generated")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8, frameon=False)
    n = runs[0]["generated"]["total"]
    fig.suptitle(f"RealNVP base A vs B, trained on centre only: afterMS mu-, px "
                 f"({n:,} unfiltered draws per run, numbers = generated counts, {env['gpu']})", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(a.out, dpi=200)
    fig.savefig(a.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
