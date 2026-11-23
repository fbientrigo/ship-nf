"""Slide 8: how far the NF distribution is from the real one, per distance-to-training band. Samples; trains nothing.

Same experiment, checkpoint and definitions as ``tail_demo.run`` (``result.json``): flow ``flow.pt`` trained on the
centre ``|zpx| <= 2`` (minus a random 10 % val_interp) of the afterMS mu- sample, test = ``|zpx| > 2.5``.

State compared: ``X = (x, y, px, py, pz)``, the physical form of the 5 modeled features ``x, y, tx, ty, log p``
(x, y in m per the afterMS audit; momenta in GeV/c). Each coordinate is scaled by its median and 1.4826 MAD over
train rows only, so the scale never sees val/test rows. Unweighted: the stored ``w`` has OPEN semantics.

Distance: the existing ``d_train`` (5th-nearest train row, features standardised by train mean/std), the same
function for real and NF rows. Groups: the held-out centre (val_interp vs NF with |zpx| <= 2), the validation shell
(2 < |zpx| <= 2.5) and the five fixed equal-count ``d_train`` bands of the 1587 real test rows from ``result.json``.

Per group, two numbers:

* ``SW``: sliced Wasserstein-1 (``diagnostics.sliced_wasserstein``, 500 fixed directions) between half of the real
  rows and an equal-size random subset of the NF rows of that group, over 200 random splits; the baseline is the same
  statistic between the two real halves. It is a distance between distributions in robust-scaled units, not an error
  in m or GeV/c. Also for position (x, y) and momentum (px, py, pz) alone.
* mass: ``P_b`` = real rows of the group / all real rows, ``Q_b`` = NF samples of the group / all NF samples (before
  any cut), with binomial errors; ``Q_b / P_b`` (log-delta 68 % interval) and ``Q_b - P_b``.

NF samples: 10 x 10^6 from the checkpoint (seeds 3000-3009); the 50 000-sample baseline (seed 1001) is re-checked
against ``result.json``::

    .venv/bin/python plots/cm_swd_vs_d.py docs/meeting_1121 --out docs/meeting_1121/cm
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ship_nf import flow
from ship_nf.diagnostics import sliced_wasserstein
from ship_nf.extrapolation import knn_distance
from ship_nf.tail_demo import K_TRAIN, K_VAL, KNN, load, z_of

NF, REF, HI = "#1D6B7D", "#6B7280", "#B45309"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.transparent": True, "savefig.dpi": 300})
BATCHES, BATCH, SPLITS, MIN_ROWS = 10, 1_000_000, 200, 30
BLOCKS = {"joint": [0, 1, 2, 3, 4], "position": [0, 1], "momentum": [2, 3, 4]}

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
assert d["data"]["sha256"] == res["data"]["sha256"]
f, lab = d["f"], d["lab"]
tr = f[lab == "train"]
mu, sd = tr.mean(0), tr.std(0)
assert np.allclose(mu, res["transform"]["mean"]) and np.allclose(sd, res["transform"]["std"])


def phys(g):  # features x, y, tx, ty, log p -> x, y, px, py, pz (inverse of loop.features)
    pz = np.exp(g[:, 4]) / np.sqrt(1 + g[:, 2] ** 2 + g[:, 3] ** 2)
    return np.c_[g[:, 0], g[:, 1], g[:, 2] * pz, g[:, 3] * pz, pz]


raw = np.load(res["data"]["path"])["muons"]
raw = raw[(raw[:, 2] > 0) & (raw[:, 6] == 13)]
assert np.allclose(phys(f), raw[:, [3, 4, 0, 1, 2]], rtol=1e-10, atol=1e-12)  # check 5: same physical representation
med = np.median(phys(tr), 0)
scale = 1.4826 * np.median(np.abs(phys(tr) - med), 0)
X = lambda g: (phys(g) - med) / scale
d_train = lambda g: knn_distance((g - mu) / sd, (tr - mu) / sd, KNN)
d_test = d_train(f[lab == "test"])
assert np.allclose(d_test, np.load(a.result / "samples.npz")["d_train_ref"])
edges = [b["lo"] for b in res["bands"]["d_train"]] + [np.inf]

g0 = model.sample(50_000, 1001)  # the saved baseline
z0 = z_of(g0, d)
b0 = np.digitize(d_train(g0[z0 > K_VAL]), edges) - 1
assert (z0 > K_VAL).sum() == res["generated"]["in_test_region"]
assert np.bincount(b0, minlength=5).tolist() == [b["gen"] for b in res["bands"]["d_train"]]

G, nonfinite, gen = 0, 0, {"centre": [], "shell": [], "tail": []}
n_region = {"centre": 0, "shell": 0}
for i in range(BATCHES):
    g = model.sample(BATCH, 3000 + i)
    G += len(g)
    ok = np.isfinite(g).all(1)
    nonfinite += int((~ok).sum())
    g = g[ok]
    z = z_of(g, d)
    for k, m in (("centre", z <= K_TRAIN), ("shell", (z > K_TRAIN) & (z <= K_VAL))):
        n_region[k] += int(m.sum())
        if i == 0:  # one batch is plenty for these well-populated groups
            gen[k].append(g[m][:100_000])
    gen["tail"].append(g[z > K_VAL])
gen = {k: np.concatenate(v) for k, v in gen.items()}
d_gen_tail = d_train(gen["tail"])
b_gen = np.digitize(d_gen_tail, edges) - 1
b_ref = np.digitize(d_test, edges) - 1
R = len(f)

groups = [("centre (held-out 10 %)", f[lab == "val_interp"], gen["centre"], int(np.isin(lab, ["train", "val_interp"]).sum()), n_region["centre"]),
          ("validation shell", f[lab == "val"], gen["shell"], int((lab == "val").sum()), n_region["shell"])]
for b in range(5):
    groups.append((f"test band {b + 1}", f[lab == "test"][b_ref == b], gen["tail"][b_gen == b], int((b_ref == b).sum()), int((b_gen == b).sum())))



def mass(r_cnt, R, g_cnt, G):
    """``P = r/R``, ``Q = g/G`` with binomial errors, ``Q - P``, and ``Q / P`` with a log-delta 68 % interval; with
    ``g = 0`` the ratio is None and only the 95 % upper bound ``3 / G / P`` is given."""
    P, Q = r_cnt / R, g_cnt / G
    sP, sQ = np.sqrt(P * (1 - P) / R), np.sqrt(Q * (1 - Q) / G)
    out = dict(ref_count=r_cnt, ref_total=R, nf_count=g_cnt, nf_total=G, P=P, Q=Q, P_err=sP, Q_err=sQ,
               Q_minus_P=Q - P, Q_minus_P_err=float(np.hypot(sP, sQ)))
    if not g_cnt:
        return {**out, "ratio": None, "ratio_upper95": 3 / G / P}
    s = np.sqrt((1 - P) / (R * P) + (1 - Q) / (G * Q))
    return {**out, "ratio": Q / P, "ratio_68": [Q / P * np.exp(-s), Q / P * np.exp(s)]}


# check 4: a "generator" that is the real sample without its tail must show the tail mass as missing
no_tail = f[lab != "test"]
miss = mass(int((lab == "test").sum()), R, int((z_of(no_tail, d) > K_VAL).sum()), len(no_tail))
assert miss["ratio"] is None and miss["ratio_upper95"] < 0.01, miss
rng = np.random.default_rng(11)
rows = []
for name, real, nfr, r_cnt, g_cnt in groups:
    n = len(real) // 2
    row = {"group": name, "real_rows": len(real), "nf_rows": len(nfr), "half": n, "d_train_median": float(np.median(d_train(real)))}
    reliable = n >= MIN_ROWS and len(nfr) >= n
    row["reliable"] = bool(reliable)
    if reliable:
        xr, xg = X(real), X(nfr)
        sw = {k: ([], []) for k in BLOCKS}
        for _ in range(SPLITS):
            p = rng.permutation(len(real))
            A, B, Gs = xr[p[:n]], xr[p[n:2 * n]], xg[rng.choice(len(xg), n, replace=False)]
            for k, c in BLOCKS.items():
                sw[k][0].append(sliced_wasserstein(A[:, c], Gs[:, c]))
                sw[k][1].append(sliced_wasserstein(A[:, c], B[:, c]))
        for k, (nf_v, rr_v) in sw.items():
            row[f"sw_{k}_nf"] = np.quantile(nf_v, [0.16, 0.5, 0.84]).tolist()
            row[f"sw_{k}_real_vs_real"] = np.quantile(rr_v, [0.16, 0.5, 0.84]).tolist()
    pn = phys(nfr)
    row.update(nf_p_above_400=int((np.linalg.norm(pn[:, 2:], axis=1) > 400).sum()),
               nf_p_max=float(np.linalg.norm(pn[:, 2:], axis=1).max()),
               nf_xy_beyond_7p5m=int((np.abs(pn[:, :2]) > 7.5).any(1).sum()), **mass(r_cnt, R, g_cnt, G))
    rows.append(row)

P_t, Q_t = (lab == "test").mean(), len(gen["tail"]) / G
out = {"definition": __doc__, "scale": {"coords": ["x", "y", "px", "py", "pz"], "median": med.tolist(), "robust_sigma": scale.tolist(),
                                       "fit_on": "train rows"},
       "nf_total": G, "nf_nonfinite": nonfinite, "nf_tail": len(gen["tail"]), "real_rows": R, "real_tail": int((lab == "test").sum()),
       "tail_P": P_t, "tail_Q": Q_t, "tail_ratio": Q_t / P_t,
       "nf_tail_beyond_max_real_d": int((d_gen_tail > d_test.max()).sum()), "d_train_max_real_test": float(d_test.max()),
       "baseline_50k_checked": {"tail": int((z0 > K_VAL).sum()), "per_band": np.bincount(b0, minlength=5).tolist()},
       "groups": rows}
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "swd_vs_d.json").write_text(json.dumps(out, indent=1))

# ---- figure ---------------------------------------------------------------------------------------------------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
xs = np.array([r["d_train_median"] for r in rows])
test = np.array([r["group"].startswith("test") for r in rows])
for r, x, t in zip(rows, xs, test):
    if not r["reliable"]:
        ax.annotate("too few NF\nsamples", (x, 0), xycoords=("data", "axes fraction"), ha="center", va="bottom", color=HI)
        continue
    for key, c, dx, lbl in (("sw_joint_nf", NF, 1.0, "NF vs real"), ("sw_joint_real_vs_real", REF, 1.0, "real vs real (finite-sample floor)")):
        lo, m, hi = r[key]
        ax.errorbar(x * dx, m, yerr=[[m - lo], [hi - m]], fmt="o" if t else "s", color=c, ms=8, lw=2, capsize=3,
                    mfc=c if t else "white", label=lbl if r is rows[0] else None)
ax.set_xscale("log")
ax.set_xlabel("distance to the training muons (5th nearest)")
ax.set_ylabel("sliced Wasserstein-1, (x, y, px, py, pz)\n(robust-scaled units)")
ax.set_yscale("log")
far = rows[-1]
ax.annotate(f"NF samples up to p = {far['nf_p_max']:.0e} GeV/c\n({far['nf_p_above_400']} of {far['nf_rows']} above 400 GeV/c)",
            (xs[-1], far["sw_joint_nf"][1]), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", color=HI)
for r, x, t in zip(rows, xs, test):
    if r["ratio"] is None:
        bx.plot(x, r["ratio_upper95"], "v", color=HI, ms=9)
        continue
    lo, hi = r["ratio_68"]
    bx.errorbar(x, r["ratio"], yerr=[[r["ratio"] - lo], [hi - r["ratio"]]], fmt="o" if t else "s", color=NF, ms=8, lw=2,
                capsize=3, mfc=NF if t else "white")
for r, x in zip(rows[:2], xs[:2]):
    bx.annotate(r["group"].split(" (")[0], (x, r["ratio"]), xytext=(10, 0), textcoords="offset points", va="center", color=REF)
bx.axhline(1, color=REF, ls="--", lw=1.5)
bx.text(2, 1.15, "correct mass", color=REF, ha="right", va="bottom")
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlabel("distance to the training muons (5th nearest)")
bx.set_ylabel("NF mass / real mass in the group")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="s", ls="", color="k", mfc="white", ms=8, label="centre, shell"),
     Line2D([], [], marker="o", ls="", color="k", ms=8, label="tail bands |zpx| > 2.5")]
for axis in (ax, bx):
    axis.set_xticks([0.15, 0.2, 0.3, 0.5, 1, 2], ["0.15", "0.2", "0.3", "0.5", "1", "2"])
    axis.minorticks_off()
bx.text(0.98, 0.02, "unweighted rows", transform=bx.transAxes, ha="right", va="bottom", color=REF)
fig.tight_layout(w_pad=2)
fig.legend(handles=ax.get_legend_handles_labels()[0] + h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(a.out / f"slide8_swd_vs_distance.{ext}", bbox_inches="tight", pad_inches=0.05)
print(json.dumps({k: v for k, v in out.items() if k != "definition"}, indent=1))
