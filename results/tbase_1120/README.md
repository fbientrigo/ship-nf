# Gaussian vs Student-t base, trained on centre only (2026-11-20)

Question: does changing only the RealNVP base distribution help the px tails of afterMS mu- when training sees |z| <= 2?

- Data: `ship_adaptive_muon_bg/data/samples/muonsFullMC_afterMS_sample.npz`, sha256 `a45fc522…6f16` (checked on the VM).
- Split: `tail_demo.load(npz, seed=1)`, same as figures 3-4: train 15 806, val_interp 1 756, val 638, test 1 587.
- A: `flow.Flow(base="normal")` (unchanged; `tests/test_flow_base.py` reproduces the meeting checkpoint samples bit for bit).
  B: `base="t5"`, independent Student-t per latent coordinate, nu = 5, scale sqrt(3/5); used in `sample` and `log_prob`.
- Pairing: per seed, same initial weights (torch seeded before the flow is built, on CPU), full batch (no minibatch
  order), Adam lr 3e-3, 400 steps. Checkpoint rule: last step, no early stopping (benchmark rule).
- Measure: 200 000 unfiltered draws per run (sample seed = seed + 1000). R = (gen in region / 200 000) /
  (real in region / 19 787). Non-finite conversions are counted and stay in the denominator (there were none).
- Hardware: Colab Tesla T4, torch 2.14.1+cu130 (`env.json`). The checkpoints re-sampled on a local CPU give the same
  tail counts.

VERIFIED: numbers in `[AB]_s*.json`, figure regenerates from them byte for byte (`scripts/fig_tbase.py`).
PROVISIONAL: two seeds only; the left tail has 29 (val) and 38 (test) real rows. The test exterior was already
inspected in the earlier demo, so its comparison is exploratory. OPEN: whether nu should be learned; whether the effect
survives with tail rows in training.

```sh
PYTHONPATH=src python scripts/exp_tbase.py AFTERMS.npz --out results/tbase_1120 --device cuda --seeds 1 2
.venv/bin/python scripts/fig_tbase.py results/tbase_1120 --out results/tbase_1120/fig_tbase.png
```
