"""Can the flow be steered into the px tail by choosing where to sample its latent? Slides 7-8 of the SHiP CM.

Samples the saved tail-demo checkpoint; trains nothing. Notation: ``zpx = (px - median) / (1.4826 MAD)`` is the split
variable (test region ``|zpx| > 2.5``), ``z_lat`` is the 5-D latent of the flow (x, y, tx, ty, log p after the flow).

A. truncated base: ``z_lat ~ N(0, I5)`` given ``|z_lat| > r`` (rejection), pushed through the flow.
B. tilted base: ``z_lat ~ N(beta u, I5)``, ``u`` the normalised least-squares slope of ``zpx`` on ``z_lat`` (50 000 NF
   samples), with weights ``w = exp(-beta u.z_lat + beta^2 / 2)`` back to the untilted base.
For each case: fraction of samples with ``|zpx| > 2.5``, the share of those with ``d_train`` above the largest real
test-row ``d_train`` (4.26, figure 8C), and px quantiles of the tail samples (unweighted) vs the 1587 real test rows.

Inputs: ``RESULT_DIR/flow.pt`` (sha256 1739854586...6137, = ``result.json`` checkpoint), ``samples.npz`` (d_train_ref),
``result.json``, and the afterMS npz named in ``result.json`` (split seed 1, as in the checkpoint)::

    .venv/bin/python plots/cm_base_tail.py docs/meeting_1121 --out docs/meeting_1121/cm
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ship_nf import flow
from ship_nf.extrapolation import knn_distance
from ship_nf.tail_demo import K_VAL, KNN, load, px_of

NF, REF, HI = "#1D6B7D", "#6B7280", "#B45309"
N, SEED = 50_000, 7_000
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.transparent": True, "savefig.dpi": 300})

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
tr = d["f"][d["lab"] == "train"]
mu, sd = tr.mean(0), tr.std(0)
d_max = float(np.load(a.result / "samples.npz")["d_train_ref"].max())
real_px = px_of(d["f"][d["lab"] == "test"])


@torch.no_grad()
def push(z):  # z_lat -> features, the same steps as flow.Flow.sample
    z = torch.as_tensor(z, dtype=torch.float32)
    for layer in reversed(model.layers):
        z = layer.inverse(z)
    return (z * model.std + model.mean).numpy().astype(np.float64)


def zpx(f):
    return (px_of(f) - d["median"]) / d["sigma"]


TAIL_PX = {}  # px of the tail samples per case, for the shape figure


def measure(name, f, draws, w=None):
    zp = zpx(f)
    tail = np.abs(zp) > K_VAL  # nan (non-finite rows) is False
    TAIL_PX[name] = px_of(f[tail])
    dt = knn_distance((f[tail] - mu) / sd, (tr - mu) / sd, KNN)
    row = {"case": name, "n": len(f), "base_draws": draws, "tail": int(tail.sum()), "tail_frac": float(tail.mean()),
           "unsupported": int((dt > d_max).sum()), "unsupported_frac": float((dt > d_max).mean()) if tail.any() else None,
           "right_share": float((zp[tail] > 0).mean()) if tail.any() else None,
           "px_q50_90_99": np.quantile(px_of(f[tail]), [.5, .9, .99]).tolist() if tail.any() else None}
    if w is not None:  # Kish ESS/N of all weights, and the weighted tail fraction (estimate for the untilted flow)
        row.update(ess_frac=float(w.sum() ** 2 / (w ** 2).sum() / len(w)), tail_frac_weighted=float((w * tail).sum() / w.sum()))
    return row


rng = np.random.default_rng(SEED)
rows = [measure("base (sample seed 1001)", model.sample(N, 1001), N)]  # reproduces result.json: 41 in the tail
for r in (2, 3, 4):  # A: rejection until N accepted
    kept, draws, need = [], 0, N
    while need:
        z = rng.standard_normal((1_000_000, 5))
        ok = np.flatnonzero(np.linalg.norm(z, axis=1) > r)[:need]
        draws += int(ok[-1]) + 1 if len(ok) == need else len(z)  # base draws up to the N-th accepted one
        kept.append(z[ok])
        need -= len(ok)
    rows.append(measure(f"r={r}", push(np.concatenate(kept)), draws))

z0 = rng.standard_normal((N, 5))
coef, *_ = np.linalg.lstsq(np.c_[z0, np.ones(N)], np.nan_to_num(zpx(push(z0))), rcond=None)
u = coef[:5] / np.linalg.norm(coef[:5])
for b in (0, 1, 2, 3):  # B
    z = rng.standard_normal((N, 5)) + b * u
    rows.append(measure(f"beta={b}", push(z), N, np.exp(-b * z @ u + b * b / 2)))

out = {"u": u.tolist(), "d_train_max_real": d_max, "real_test_rows": len(real_px),
       "real_px_q50_90_99": np.quantile(real_px, [.5, .9, .99]).tolist(),
       "real_right_share": float((zpx(d["f"][d["lab"] == "test"]) > 0).mean()), "rows": rows}
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "base_tail.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

# ---- figure: left = tail fraction, right = unsupported share; A and B on one categorical axis ------------------------
lab = [r["case"].split(" ")[0].replace("beta", "β") for r in rows[1:]]
x = np.r_[0:3, 3.6:7.6]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6))
for axis, key, base, fmt in ((ax, "tail_frac", rows[0]["tail_frac"], 100), (bx, "unsupported_frac", rows[0]["unsupported_frac"], 100)):
    y = np.array([fmt * (r[key] or 0) for r in rows[1:]])
    axis.plot(x[:3], y[:3], "o-", color=NF, lw=2.5, ms=8, label="A: |z_lat| > r")
    axis.plot(x[3:], y[3:], "s-", color=HI, lw=2.5, ms=8, label="B: tilt β·u")
    axis.axhline(fmt * base, color=REF, ls="--", lw=1.5)
    axis.set_xticks(x, lab)
ax.set_yscale("log")
ax.text(x[-1], 100 * rows[0]["tail_frac"] * 1.15, f"untilted: {rows[0]['tail']} of {N}", color=REF, ha="right", va="bottom")
ax.set_ylabel("samples with |zpx| > 2.5 (%)")
ax.legend(loc="upper left", frameon=False)
bx.text(x[-1], 100 * rows[0]["unsupported_frac"] + 1.5, f"untilted: {rows[0]['unsupported']} of {rows[0]['tail']}",
        color=REF, ha="right", va="bottom")
bx.set_ylim(0, 100)
bx.set_ylabel(f"of those: d_train > {d_max:.2f} (%)\n(farther than any real muon)")
fig.tight_layout(w_pad=2)
fig.savefig(a.out / "slide78_base_tail.png", bbox_inches="tight", pad_inches=0.05)

# ---- slide 8: px shape of the tail, real vs r = 4 vs beta = 3 -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.9, 4.4))
bins = np.linspace(4, 12, 33)
for px, kw, lab in ((real_px, dict(color=REF, alpha=0.45), "real tail"),
                    (TAIL_PX["beta=3"], dict(histtype="step", color=NF, lw=2.5), "NF, tilt β = 3"),
                    (TAIL_PX["r=4"], dict(histtype="step", color=HI, lw=2.5, ls="--"), "NF, |z_lat| > 4")):
    ax.hist(px, bins, weights=np.full(len(px), 1 / len(px) / np.diff(bins)[0]), label=f"{lab} ({len(px):_})".replace("_", " "), **kw)
    ax.axvline(np.median(px), color=kw["color"], lw=1, ls=kw.get("ls", "-"))
ax.set_xlim(4, 12)
ax.set_xlabel("px (GeV/c), samples with |zpx| > 2.5")
ax.set_ylabel("fraction per GeV/c\n(of own total)")
ax.legend(loc="upper right", frameon=False)
fig.savefig(a.out / "slide8_px_shape.png", bbox_inches="tight", pad_inches=0.05)
