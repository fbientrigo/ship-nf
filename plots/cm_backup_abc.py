"""Backup slides of the SHiP CM: extrapolation (A), representation (B) and tail scarcity (C). Trains 18 small flows.

One fixed split of the afterMS mu- rows (19 787 distinct rows, no duplicates; each row is its own source unit):
random 30 % test, 10 % validation, 60 % training pool (seed 0). Regions use the robust z of px with median and
1.4826 MAD fitted on the pool only: centre ``|z| <= 2``, outside ``|z| > 2``, tail ``|z| > 2.5`` (the old benchmark).
The test set is a full-distribution sample and is never used for fitting, scaling or selection; nothing is selected
(fixed architecture and budget from ``tail_demo``: RealNVP 6x64, 400 full-batch epochs, lr 3e-3, so every run takes
the same number of optimisation steps). Validation NLL is only reported.

Training sets, all drawn from the pool, natural (unweighted, not oversampled) composition:

- C(f): all centre rows + a random fraction f of the outside rows, f = 0, 0.1, 0.25, 0.5, 1. A = C(0) (centre only),
  B = C(1) (whole pool).
- B_matched: a random subset of the whole pool with the size of A, so B's gain can be split into tail exposure vs size.
Seeds 1-3 (the subset and the flow init). Each flow draws 10^6 samples.

Metrics against the same test set:

- Tail mass: ``q = NF tail / 10^6``, ``p = test tail / test rows``; ``q/p`` (log-delta 68 % interval incl. both counts;
  p's error is common to every run) and ``q - p``. Also ``q`` / tail share of the run's own training set: C keeps the
  natural composition, so its training sets have a lower tail share than the target by construction.
- SW1 (``diagnostics.sliced_wasserstein``) on ``(x, y, px, py, pz)`` scaled by pool median/MAD (``sw``) and by pool
  mean/std (``sw_std``, scaling check): global (all test rows) and inside the tail (test tail vs NF tail), each with a
  real-vs-real floor (``diagnostics.error_vs_distance`` with one band, 50 splits). Too few NF tail samples -> no value.
- Figure 3: SW per band of distance to A's training rows (the same axis for A and B), seed 1.

Unweighted: the stored ``w`` has OPEN semantics, and training is unweighted::

    .venv/bin/python plots/cm_backup_abc.py --out docs/meeting_1121/cm/backup
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ship_nf import flow
from ship_nf.diagnostics import error_vs_distance, sliced_wasserstein
from ship_nf.extrapolation import afterms_states, knn_distance
from ship_nf.loop import features

DATA = Path("/home/fabian/thesis/ship_adaptive_muon_bg/data/samples/muonsFullMC_afterMS_sample.npz")
NF, REF, HI = "#1D6B7D", "#6B7280", "#B45309"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.transparent": True, "savefig.dpi": 300})
FRACS, SEEDS, N_GEN, EPOCHS = (0.0, 0.1, 0.25, 0.5, 1.0), (1, 2, 3), 1_000_000, 400

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

raw = np.load(DATA)["muons"]
raw = raw[(raw[:, 2] > 0) & (raw[:, 6] == 13)]
assert len(np.unique(raw, axis=0)) == len(raw)
f = features(afterms_states(raw))
part = np.random.default_rng(0).permutation(len(f))
n_te, n_va = int(0.3 * len(f)), int(0.1 * len(f))
test, val, pool = part[:n_te], part[n_te:n_te + n_va], part[n_te + n_va:]
px = raw[:, 0]
med_px = np.median(px[pool])
sig_px = 1.4826 * np.median(np.abs(px[pool] - med_px))
zabs = lambda g: np.abs(g[:, 2] * np.exp(g[:, 4]) / np.sqrt(1 + g[:, 2] ** 2 + g[:, 3] ** 2) - med_px) / sig_px
z = zabs(f)
assert np.allclose(z, np.abs(px - med_px) / sig_px)
centre, outside = pool[z[pool] <= 2], pool[z[pool] > 2]


def phys(g):  # x, y, tx, ty, log p -> x, y, px, py, pz
    pz = np.exp(g[:, 4]) / np.sqrt(1 + g[:, 2] ** 2 + g[:, 3] ** 2)
    return np.c_[g[:, 0], g[:, 1], g[:, 2] * pz, g[:, 3] * pz, pz]


P = phys(f[pool])
c0 = np.median(P, 0)
rs, ss = 1.4826 * np.median(np.abs(P - c0), 0), P.std(0)
METRICS = {"sw": lambda u, v: sliced_wasserstein(u / rs, v / rs), "sw_std": lambda u, v: sliced_wasserstein(u / ss, v / ss)}
Y = lambda g: phys(g) - c0
te, te_tail = f[test], f[test][z[test] > 2.5]
p_tail = len(te_tail) / len(te)


def one_band(real, gen):
    return error_vs_distance(Y(real), Y(gen), np.zeros(len(real)), np.zeros(len(gen)), [0, np.inf], METRICS, splits=50, seed=7)[0]


runs, keep = [], {}
configs = [("C", fr, sd) for fr in FRACS for sd in SEEDS] + [("B_matched", None, sd) for sd in SEEDS]
for kind, fr, sd in configs:
    rng = np.random.default_rng(100 + sd)
    if kind == "C":
        idx = np.r_[centre, rng.choice(outside, int(round(fr * len(outside))), replace=False)]
    else:
        idx = rng.choice(pool, len(centre), replace=False)
    model, losses = flow.fit(f[idx], sd, epochs=EPOCHS)
    g = model.sample(N_GEN, 5000 + sd)
    g = g[np.isfinite(g).all(1)]
    zg = zabs(g)
    gt = int((zg > 2.5).sum())
    q = gt / N_GEN
    s = np.sqrt((1 - p_tail) / len(te_tail) + ((1 - q) / gt if gt else np.inf))
    with __import__("torch").no_grad():
        import torch
        val_nll = float(-model.log_prob(torch.as_tensor(f[val], dtype=torch.float32)).mean())
    glob, tail = one_band(te, g), one_band(te_tail, g[zg > 2.5])
    name = {0.0: "A", 1.0: "B"}.get(fr, f"C{fr:g}") if kind == "C" else "B_matched"
    row = {"name": name, "kind": kind, "frac": fr, "seed": sd, "n_train": len(idx), "unique_tail_train": int((z[idx] > 2.5).sum()),
           "unique_outside_train": int((z[idx] > 2).sum()), "train_tail_frac": float((z[idx] > 2.5).mean()),
           "ratio_vs_train": q / float((z[idx] > 2.5).mean()) if (z[idx] > 2.5).any() else None, "steps": EPOCHS, "final_train_nll": losses[-1], "val_nll": val_nll,
           "nf_finite": len(g), "nf_tail": gt, "q": q, "p": p_tail, "q_minus_p": q - p_tail,
           "ratio": q / p_tail if gt else None, "ratio_68": [q / p_tail * np.exp(-s), q / p_tail * np.exp(s)] if gt else None,
           "ratio_upper95": None if gt else 3 / N_GEN / p_tail,
           **{f"global_{k}": glob.get(k) for k in ("sw", "sw_floor", "sw_std", "sw_std_floor")},
           **{f"tail_{k}": tail.get(k) for k in ("sw", "sw_floor", "sw_std", "sw_std_floor")},
           "tail_sw_reliable": "sw" in tail}
    runs.append(row)
    print({k: row[k] for k in ("name", "seed", "n_train", "unique_tail_train", "nf_tail", "ratio")}, flush=True)
    if sd == 1 and name in ("A", "B"):
        keep[name] = (f[idx], g)

# figure 3 data: SW per band of distance to A's training rows, A and B (seed 1)
tr_a = keep["A"][0]
mu, sd_ = tr_a.mean(0), tr_a.std(0)
dist = lambda g: knn_distance((g - mu) / sd_, (tr_a - mu) / sd_, 5)
d_te = dist(te)
edges = np.r_[0, np.quantile(d_te, np.linspace(0, 1, 9)[1:-1]), np.inf]
by_d = {k: error_vs_distance(Y(te), Y(g), d_te, dist(g), edges, {"sw": METRICS["sw"]}, splits=50, seed=7) for k, (_, g) in keep.items()}
for k in by_d:
    for b in by_d[k]:
        b["d_median"] = float(np.median(d_te[(d_te >= b["lo"]) & (d_te < b["hi"])]))

counts = {"rows": len(f), "test": len(test), "val": len(val), "pool": len(pool), "pool_centre": len(centre),
          "pool_outside": len(outside), "pool_tail": int((z[pool] > 2.5).sum()), "test_tail": len(te_tail),
          "test_tail_frac": p_tail, "px_median_pool": med_px, "px_sigma_pool": sig_px,
          "test_tail_weight_frac": float(raw[test][z[test] > 2.5, 7].sum() / raw[test, 7].sum())}
(a.out / "abc.json").write_text(json.dumps({"definition": __doc__, "counts": counts, "runs": runs, "by_distance": by_d}, indent=1))
with open(a.out / "abc_runs.csv", "w", newline="") as fh:
    cols = ["name", "seed", "n_train", "unique_tail_train", "nf_tail", "q", "p", "q_minus_p", "ratio", "val_nll"]
    w = csv.DictWriter(fh, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(runs)
print(json.dumps(counts, indent=1))


def med(name, key, i=1):
    v = [r[key][i] if isinstance(r[key], list) else r[key] for r in runs if r["name"] == name and r[key] is not None]
    return np.median(v) if v else np.nan, v


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---- figure 1: A vs B vs B_matched ----------------------------------------------------------------------------------
names = ["A", "B_matched", "B"]
labels = ["A: centre only", "B, size of A", "B: full pool"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
for i, n in enumerate(names):
    m, v = med(n, "ratio")
    ax.plot([i] * len(v), v, "o", color=NF, alpha=0.5, ms=7)
    ax.plot(i, m, "_", color=NF, ms=30, mew=3)
ax.axhline(1, color=REF, ls="--", lw=1.5)
ax.text(2.4, 1.1, "right mass", color=REF, ha="right", va="bottom")
ax.set_yscale("log")
ax.set_xticks(range(3), labels)
ax.set_xlim(-0.5, 2.5)
ax.set_ylabel(f"NF tail mass / real tail mass\n(|zpx| > 2.5, real = {100 * p_tail:.1f} %)")
for j, (key, c, lbl) in enumerate((("global_sw", NF, "all muons"), ("tail_sw", HI, "tail muons only"))):
    for i, n in enumerate(names):
        m, v = med(n, key)
        bx.plot([i + 0.15 * (2 * j - 1)] * len(v), v, "o", color=c, alpha=0.5, ms=7)
        bx.plot(i + 0.15 * (2 * j - 1), m, "_", color=c, ms=20, mew=3, label=lbl if i == 0 else None)
        fl, _ = med(n, key + "_floor")
        bx.plot(i + 0.15 * (2 * j - 1), fl, "x", color=REF, ms=9, mew=2, label="real vs real" if i == 0 and j == 1 else None)
bx.set_yscale("log")
bx.set_xticks(range(3), labels)
bx.set_xlim(-0.5, 2.5)
bx.set_ylabel("distribution error, sliced W1\n(x, y, px, py, pz; robust units)")
bx.legend(loc="upper right", frameon=False)
fig.tight_layout(w_pad=2)
save(fig, "backup1_A_vs_B")

# ---- figure 2: C, tail coverage and tail shape vs unique tail examples ----------------------------------------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
cn = [n for n in ("A", "C0.1", "C0.25", "C0.5", "B")]
xs = np.array([np.median([r["unique_tail_train"] for r in runs if r["name"] == n]) for n in cn])
CAP = 1e3  # tail SW of single runs reaches 3.6e9 (a few extreme NF samples); those runs are drawn as triangles at the cap
xp = np.maximum(xs, 3)  # A has 0 tail rows; drawn at 3 on the log axis and labelled
for axis, key, ylab in ((ax, "ratio", "NF tail mass / real tail mass"), (bx, "tail_sw", "tail shape error, sliced W1\n(tail muons only, robust units)")):
    for n, x in zip(cn, xp):
        v = np.array([r[key][1] if isinstance(r[key], list) else r[key] for r in runs if r["name"] == n and r[key] is not None])
        axis.plot([x] * (v <= CAP).sum(), v[v <= CAP], "o", color=NF, alpha=0.5, ms=7)
        axis.plot([x] * (v > CAP).sum(), [CAP] * (v > CAP).sum(), "^", color=HI, ms=9)  # runs beyond the axis
    axis.plot(xp, [med(n, key)[0] for n in cn], "-", color=NF, lw=2.5)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(xp, ["0"] + [str(int(v)) for v in xs[1:]])
    axis.minorticks_off()
    axis.set_xlabel("unique tail muons in training")
    axis.set_ylabel(ylab)
cs = cn[1:]  # A has no training tail share
ax.plot(xp[1:], [med(n, "ratio_vs_train")[0] for n in cs], "s--", color=HI, lw=2, ms=7,
        label="NF tail mass / tail share of its own training set")
ax.plot([], [], "o-", color=NF, lw=2.5, label="NF tail mass / real tail mass (test)")
ax.legend(loc="lower right", frameon=False, fontsize=12)
ax.axhline(1, color=REF, ls="--", lw=1.5)
ax.axhspan(*(np.array([1, 1]) * [np.exp(-1.96 * np.sqrt((1 - p_tail) / len(te_tail))), np.exp(1.96 * np.sqrt((1 - p_tail) / len(te_tail)))]),
           color=REF, alpha=0.12, lw=0)
ax.text(xp[0], 1.12, "right mass (95 % band of the test count)", color=REF, va="bottom")
fl = [med(n, "tail_sw_floor")[0] for n in cn]
bx.plot(xp, fl, "x--", color=REF, ms=9, mew=2, label="real vs real")
bx.plot([], [], "^", color=HI, ms=9, label=f"run above {CAP:.0e}")
bx.set_ylim(None, CAP * 2)
bx.legend(loc="center right", frameon=False)
fig.tight_layout(w_pad=2)
save(fig, "backup2_C_tail_scarcity")

# ---- figure 3: SW vs distance to A's training rows, A and B ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.9, 4.8))
for k, c in (("A", NF), ("B", HI)):
    b = [x for x in by_d[k] if "sw" in x]
    x, y = np.array([x["d_median"] for x in b]), np.array([x["sw"] for x in b])
    ax.fill_between(x, y[:, 0], y[:, 2], color=c, alpha=0.2, lw=0)
    ax.plot(x, y[:, 1], "o-", color=c, lw=2.5, ms=7, label={"A": "A: centre only", "B": "B: full pool"}[k])
b = by_d["A"]
x, y = np.array([x["d_median"] for x in b]), np.array([x["sw_floor"] for x in b])
ax.plot(x, y[:, 1], "o--", color=REF, lw=2, ms=6, label="real vs real")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.1, 0.2, 0.4, 0.8], ["0.1", "0.2", "0.4", "0.8"])
ax.minorticks_off()
ax.set_xlabel("distance to the centre training muons (5th nearest)")
ax.set_ylabel("distribution error, sliced W1\n(x, y, px, py, pz; robust units)")
ax.legend(loc="upper left", frameon=False)
save(fig, "backup3_error_vs_distance_A_B")
