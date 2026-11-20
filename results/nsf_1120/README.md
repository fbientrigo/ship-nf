# Arm C: neural spline flow with Gaussian base, trained on centre only (2026-11-20)

Question: with the base fixed (Gaussian), does replacing the affine coupling of A with monotonic rational-quadratic
spline coupling change the px tails of afterMS mu-? A and B are reused from `../tbase_1120` (not rerun).

- C = `flow.Flow(coupling="rqs")`: same 6 couplings, alternating masks (no permutations, as in A), same conditioner
  (5 -> 64 -> 64, ReLU); only its last layer grows from 2x5 to 23x5 outputs (8 widths, 8 heights, 7 inner derivatives).
  Spline: zuko 1.6.0 `MonotonicRQSTransform`, 8 bins on [-5, 5] of each coupling input (standardised features for the
  first coupling), linear tails = identity outside, boundary derivative 1. Fixed before any evaluation, not tuned.
- Trainable parameters: A 31 164, C 72 114.
- Same rows (labels sha256 `a99e6bbb…` = A/B), transform, full batch, Adam lr 3e-3, 400 steps, last-step checkpoint,
  200 000 unfiltered draws, sample seed = seed + 1000. Seeds 1, 2 are shared with A/B, initial weights are not
  (different last layer).
- Pre-training checks per run (`pre_train_checks` in each JSON): inverse round trip < 4e-5, log-det vs autograd
  Jacobian < 5e-6, forward + inverse log-dets cancel < 4e-6, log_prob finite on all 15 806 train rows.
- Hardware: Colab Tesla T4, torch 2.14.1+cu130; checkpoints re-sampled on local CPU give the same tail counts.
  `session.log` is the remote stdout (it also shows a lost connection during `colab install`; zuko 1.6.0 was confirmed
  on the VM afterwards).

Caveat seen after the run, not acted on: C ends with a worse mean train NLL (-3.94 / -4.03) than A (-4.59 / -4.66), so at
400 steps C is probably not converged. The fixed step budget was kept as specified.

```sh
PYTHONPATH=src python scripts/exp_tbase.py AFTERMS.npz --out results/nsf_1120 --device cuda --seeds 1 2 --designs C
.venv/bin/python scripts/fig_tbase.py results/tbase_1120 results/nsf_1120 --out results/nsf_1120/fig_abc.png
```
