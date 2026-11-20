"""Student-t base vs Gaussian base, trained on the centre only (afterMS mu-, px tails). Runs on CPU or one GPU.

    PYTHONPATH=src python scripts/exp_tbase.py AFTERMS.npz --out DIR [--device cuda] [--seeds 1 2] [--epochs 400]

Same split as figures 3-4 (``tail_demo.load`` with split seed 1). Per training seed, A (``base="normal"``) and B
(``base="t5"``) share the initial weights (``flow.fit`` seeds torch before building the flow) and the full-batch
order. Checkpoint rule, as in the benchmark: the weights after the last of ``epochs`` full-batch Adam steps, no
early stopping. Writes ``env.json``, ``<design>_s<seed>.pt`` and ``<design>_s<seed>.json``.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ship_nf import flow
from ship_nf.tail_demo import K_TRAIN, K_VAL, load, px_of

SPLIT_SEED, N_GEN = 1, 200000
BASES = {"A": "normal", "B": "t5"}


def env(device: str) -> dict:
    """Versions, device, and one real forward/backward with parameters and batch on ``device``."""
    m = flow.Flow(5, base="t5").to(device)
    x = torch.randn(256, 5, device=device)
    (-m.log_prob(x).mean()).backward()
    p = next(m.parameters())
    assert x.device.type == p.device.type == p.grad.device.type == torch.device(device).type
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
            "python": platform.python_version(), "device": device, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "forward_backward": {"param_device": str(p.device), "grad_device": str(p.grad.device),
                                 "batch_device": str(x.device), "grad_finite": bool(torch.isfinite(p.grad).all())}}


def regions(z: np.ndarray) -> dict:
    """Masks on signed robust z of px; nan (invalid conversions) falls in no region."""
    return {"val_left": (z >= -K_VAL) & (z < -K_TRAIN), "val_right": (z > K_TRAIN) & (z <= K_VAL),
            "test_left": z < -K_VAL, "test_right": z > K_VAL}


def one(d: dict, design: str, seed: int, epochs: int, device: str, out: Path) -> dict:
    f, lab = d["f"], d["lab"]
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    model, losses = flow.fit(f[lab == "train"], seed, epochs=epochs, base=BASES[design], device=device)
    sync()
    t_train = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        nll = -model.log_prob(torch.as_tensor(f[lab == "val_interp"], dtype=torch.float32, device=device)).cpu().numpy()
    g = model.sample(N_GEN, seed + 1000)
    fin5 = np.isfinite(g).all(1)
    with np.errstate(all="ignore"):
        px = px_of(g)
    ok = fin5 & np.isfinite(px)
    zg = np.where(ok, (px - d["median"]) / d["sigma"], np.nan)
    zr = (px_of(f) - d["median"]) / d["sigma"]
    n_ref, rg, rr = len(f), regions(zg), regions(zr)
    tails = {}
    for k in ("val", "test"):
        for side in ("left", "right", "both"):
            mg = rg[f"{k}_left"] | rg[f"{k}_right"] if side == "both" else rg[f"{k}_{side}"]
            mr = rr[f"{k}_left"] | rr[f"{k}_right"] if side == "both" else rr[f"{k}_{side}"]
            gc, rc = int(mg.sum()), int(mr.sum())
            tails[f"{k}_{side}"] = {"gen": gc, "gen_total": N_GEN, "ref": rc, "ref_total": n_ref,
                                    "R": (gc / N_GEN) / (rc / n_ref) if rc else None}
    ck = out / f"{design}_s{seed}.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ck)
    res = {"design": design, "base": BASES[design], "seed": seed, "split_seed": SPLIT_SEED, "epochs": epochs,
           "checkpoint_rule": f"last of {epochs} full-batch Adam steps (lr 3e-3), no early stopping",
           "device": device, "train_seconds": t_train, "final_train_nll": losses[-1],
           "val_interp_nll": {"median": float(np.median(nll)), "mean": float(np.mean(nll)), "n": len(nll),
                              "finite": int(np.isfinite(nll).sum())},
           "generated": {"total": N_GEN, "sample_seed": seed + 1000, "nonfinite_features": int((~fin5).sum()),
                         "nonfinite_px": int((fin5 & ~np.isfinite(px)).sum()), "valid": int(ok.sum())},
           "tails": tails, "checkpoint": ck.name, "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}
    (out / f"{design}_s{seed}.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = load(a.npz, SPLIT_SEED)
    e = env(a.device)
    e |= {"data": d["data"], "rule": {"median": d["median"], "sigma": d["sigma"], "k_train": K_TRAIN, "k_val": K_VAL},
          "split_rows": {k: int((d["lab"] == k).sum()) for k in ("train", "val_interp", "val", "test")},
          "labels_sha256": hashlib.sha256("".join(d["lab"].astype(str)).encode()).hexdigest()}
    (a.out / "env.json").write_text(json.dumps(e, indent=1))
    print(json.dumps({k: e[k] for k in ("gpu", "torch", "cuda", "forward_backward", "split_rows")}))
    for seed in a.seeds:
        for design in BASES:
            r = one(d, design, seed, a.epochs, a.device, a.out)
            print(design, seed, f"{r['train_seconds']:.1f}s", "nll", round(r["val_interp_nll"]["median"], 3),
                  {k: v["R"] for k, v in r["tails"].items()}, flush=True)


if __name__ == "__main__":
    main()
