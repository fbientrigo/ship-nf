"""Slide versions of figures 2, 3 and 4 for the SHiP CM computing session. Same numbers, new layout.

Reads only saved results; it trains nothing and recomputes nothing except the plotted fractions.

Data (all in ``RESULT_DIR``, normally ``docs/meeting_1121``, added in commit 9f237dd):

- ``learning_curve.json`` from ``ship_nf learning-curve`` (sha256 321148401eaf484b7b6d5427cb5df334a37f7dce858a52b2b6f09a239ed8afbf):
  rows of ``(seed, n_train_distinct, val_mass_ratio, val_interp_nll_median)``, seeds 1-3, N = 250, 1000, 4000,
  13000. Same values as ``plots/fig2_learning_curve.py`` (slide 7).
- ``result.json`` (sha256 208fd24c83fb00da0985da45fd92b8069c40693806c7835dd7a496bc4d758f74) and ``samples.npz``
  (sha256 e9d654061773b9b55f74a35ee7b58a89c05accb90000f9ebc74c499cdc87c6e6) from ``ship_nf tail-demo``:
  ``bands[d_train|d_val]`` (equal-count bands of the 1587 real rows with |z(px)| > 2.5, with NF counts and ratio) and
  the raw distances ``d_*_ref`` / ``d_*_gen`` (standardised features, dimensionless). Same values as
  ``plots/fig4_mass_vs_distance.py`` (slide 8) and ``plots/fig3_tail_histogram.py`` (slide 6).

Writes 300 dpi transparent PNGs sized for 2/3 of a 16:9 slide (8.9 in wide), fonts >= 14 pt, no titles::

    .venv/bin/python plots/cm_slides.py docs/meeting_1121 --out docs/meeting_1121/cm
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NF, REF, HI = "#1D6B7D", "#6B7280", "#B45309"
W, H = 8.9, 4.4  # inches: 2/3 of a 13.33 in wide 16:9 slide
plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.transparent": True, "savefig.dpi": 300})

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
lc = json.loads((a.result / "learning_curve.json").read_text())
res = json.loads((a.result / "result.json").read_text())
smp = np.load(a.result / "samples.npz")


def save(fig, name):
    fig.savefig(a.out / f"{name}.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---- slide 7: validation-shell mass ratio vs N -------------------------------------------------------------------
rows = lc["rows"]
Ns = sorted({r["n_train_distinct"] for r in rows})
seeds = sorted({r["seed"] for r in rows})
pct = np.array([[100 * next(r["val_mass_ratio"] for r in rows if r["seed"] == s and r["n_train_distinct"] == n)
                 for n in Ns] for s in seeds])  # seeds x N, % of the reference mass
nll = np.array([[next(r["val_interp_nll_median"] for r in rows if r["seed"] == s and r["n_train_distinct"] == n)
                 for n in Ns] for s in seeds])


def ratio_panel(ax, band=True):
    ax.axhline(100, color=REF, ls="--", lw=1.5)
    ax.text(Ns[0], 102, "perfect (100 %)", color=REF, va="bottom")
    if band:
        ax.fill_between(Ns, pct.min(0), pct.max(0), color=NF, alpha=0.25, lw=0, label=f"range of {len(seeds)} seeds")
    else:
        for i in range(len(seeds)):
            ax.plot(Ns, pct[i], "o", color=NF, alpha=0.45, ms=7, label=f"{len(seeds)} seeds" if i == 0 else None)
    ax.plot(Ns, pct.mean(0), "o-", color=NF, lw=2.5, label="mean")
    ax.axhspan(pct.min(), pct.max(), color=HI, alpha=0.08, lw=0)
    ax.text(Ns[-1], pct.max() + 3, f"{pct.min():.0f}–{pct.max():.0f} % at every N", color=HI, ha="right", va="bottom",
            weight="bold")
    ax.set_xscale("log")
    ax.set_xticks(Ns, [f"{n:,}".replace(",", " ") for n in Ns])
    ax.minorticks_off()
    ax.set_ylim(0, 115)
    ax.set_xlabel("distinct training states N  (ESS = N)")
    ax.set_ylabel("NF mass in validation shell\n(% of reference)")
    ax.legend(loc="center right", frameon=False)


fig, ax = plt.subplots(figsize=(W, H))  # A: one panel, seed band
ratio_panel(ax)
save(fig, "slide7_A_ratio_only")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [3, 1.15]})  # B: + small NLL panel
ratio_panel(ax)
bx.fill_between(Ns, nll.min(0), nll.max(0), color=REF, alpha=0.25, lw=0)
bx.plot(Ns, nll.mean(0), "o-", color=REF, lw=2)
bx.set_xscale("log")
bx.set_xticks([Ns[0], Ns[-1]], [str(Ns[0]), f"{Ns[-1] // 1000}k"])
bx.minorticks_off()
bx.set_xlabel("N")
bx.set_ylabel("median NLL (nats),\ninterpolated rows")
bx.text(0.95, 0.95, "flat from\nN = 4 000", transform=bx.transAxes, ha="right", va="top", color=REF)
fig.tight_layout(w_pad=1.5)
save(fig, "slide7_B_ratio_plus_nll")

fig, ax = plt.subplots(figsize=(W, H))  # C: seed dots, NLL written under each N
ratio_panel(ax, band=False)
ax.set_xticks(Ns, [f"{n:,}".replace(",", " ") + f"\nNLL {m:.1f}" for n, m in zip(Ns, nll.mean(0))])
ax.set_xlabel("distinct training states N  (ESS = N)")
ax.text(0.02, 0.6, "under each N: median NLL of\ninterpolated rows, mean of seeds", transform=ax.transAxes, color=REF)
save(fig, "slide7_C_nll_as_labels")

# ---- slide 8: tail mass per distance band ------------------------------------------------------------------------
XL = {"d_train": "distance to 5th nearest\ntrain row", "d_val": "distance to 5th nearest\nvalidation-shell row"}


def band_labels(bands):
    return [f"{b['lo']:.2f}–{b['hi']:.2f}" if np.isfinite(b["hi"]) else f"> {b['lo']:.2f}" for b in bands]


def bars(ax, key, log):
    bands = res["bands"][key]
    p = [100 * b["ratio"] for b in bands]
    ax.bar(range(len(p)), p, color=NF, width=0.7)
    ax.axhline(100, color=REF, ls="--", lw=1.5)
    ax.set_xticks(range(len(p)), band_labels(bands), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel(XL[key])
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.05, 250)
        ax.set_yticks([0.1, 1, 10, 100], ["0.1 %", "1 %", "10 %", "100 %"])
        ax.minorticks_off()
    return p


fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)  # A: log %, both distances
for ax, key in zip(axes, ("d_train", "d_val")):
    p = bars(ax, key, log=True)
    ax.text(len(p) - 0.6, 120, "expected", color=REF, ha="right", va="bottom")
allp = [100 * b["ratio"] for k in ("d_train", "d_val") for b in res["bands"][k]]
axes[0].set_ylabel("NF mass in band\n(% of expected)")
axes[0].text(0.03, 0.62, f"every band: {min(allp):.1f}–{max(allp):.1f} %", transform=axes[0].transAxes, color=HI, weight="bold")
fig.tight_layout()
save(fig, "slide8_A_log_percent")

fig, ax = plt.subplots(figsize=(W, H))  # B: linear %, d_train only
p = bars(ax, "d_train", log=False)
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
ax.text(len(p) - 0.6, 102, "expected", color=REF, ha="right", va="bottom")
for i, v in enumerate(p):
    ax.text(i, v + 2, f"{v:.1f} %", ha="center", va="bottom", color=HI, weight="bold")
ax.set_ylabel("NF mass in band\n(% of expected)")
save(fig, "slide8_B_linear_train")

fig, ax = plt.subplots(figsize=(W, H))  # C: where the 41 NF samples sit vs the 1587 real tail muons
d_ref, d_gen = smp["d_train_ref"], smp["d_train_gen"]
bins = np.geomspace(min(d_ref.min(), d_gen.min()) * 0.9, max(d_ref.max(), d_gen.max()) * 1.1, 26)
ax.hist(d_ref, bins, weights=np.full(len(d_ref), 100 / len(d_ref)), color=REF, alpha=0.45,
        label=f"real tail muons ({len(d_ref)})")
ax.hist(d_gen, bins, weights=np.full(len(d_gen), 100 / len(d_gen)), histtype="step", color=NF, lw=2.5,
        label=f"NF samples in the tail ({len(d_gen)})")
beyond = int((d_gen > d_ref.max()).sum())
ax.axvspan(d_ref.max(), bins[-1], color=HI, alpha=0.12, lw=0)
ax.axvline(d_ref.max(), color=HI, lw=1.5)
ax.text(d_ref.max() * 1.05, 20, f"{beyond} of {len(d_gen)} NF samples\nfarther than any\nreal muon",
        color=HI, va="top", weight="bold")
ax.set_xscale("log")
ax.set_xlabel(XL["d_train"].replace("\n", " "))
ax.set_ylabel("% of own total per bin")
ax.set_ylim(0, 21)
ax.legend(loc="upper left", frameon=False)
save(fig, "slide8_C_distance_hist")

# ---- slide 6: figure 3 with a legend that fits -------------------------------------------------------------------
med, sig = res["rule"]["median"], res["rule"]["sigma"]
ref, gen = smp["px_ref"], smp["px_gen"]
n_gen_total = res["generated"]["total_before_filter"]
lo, hi = np.quantile(ref, [0.0005, 0.9995])
bins = np.linspace(lo, hi, 121)
fig, ax = plt.subplots(figsize=(W, H))
for k0, k1, c in ((2.5, 99, "#f4cccc"), (2.0, 2.5, "#fff2cc")):
    ax.axvspan(max(lo, med - k1 * sig), med - k0 * sig, color=c, lw=0)
    ax.axvspan(med + k0 * sig, min(hi, med + k1 * sig), color=c, lw=0)
ax.axvspan(med - 2 * sig, med + 2 * sig, color="#d9ead3", lw=0)
tr = ax.get_xaxis_transform()
for xc, name in ((med, "train"), (med + 2.25 * sig, "val"), (med - 2.25 * sig, "val"), ((hi + med + 2.5 * sig) / 2, "test |z| > 2.5"),
                 ((lo + med - 2.5 * sig) / 2, "test")):
    ax.text(xc, 1.01, name, transform=tr, ha="center", va="bottom", weight="bold")
ax.hist(ref, bins, weights=np.full(len(ref), 1 / len(ref)), histtype="step", color="k", lw=1.5,
        label=f"reference ({len(ref)} rows)")
ax.hist(gen, bins, weights=np.full(len(gen), 1 / n_gen_total), histtype="step", color=NF, lw=1.8,
        label=f"NF ({n_gen_total} samples,\n÷ total before any cut)")
ax.set_yscale("log")
ax.set_xlabel("px (GeV/c), afterMS, pdg %d" % res["data"]["pdg"])
ax.set_ylabel("fraction of own total per bin")
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
save(fig, "slide6_px_tail")
