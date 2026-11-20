# Four demos for the meeting of 2026-11-21

Figures, data and checkpoint are in `docs/meeting_1121/` (this repo) and
`../ship-fs-pipeline/docs/evidence/meeting_1121/` (figure 1). Run directories are under `~/thesis/scratch/demos_1120/`.
Every figure script reads saved results; none trains a flow or runs FairShip.

## 1. One muon round trip (figure 1)

| step | where | environment |
|---|---|---|
| hand-typed muon on the P2 plane | `ship_nf.loop.propose_manual` (`loop.py:184`), CLI `propose-manual` | ship-nf venv |
| NF muon | `ship_nf.loop.propose` with the four `demo_loop_v0` reports as history, `--m 1 --seed 104` | ship-nf venv |
| FairShip route, same for both | `ship-fs-pipeline/scripts/fs_evaluate.sh` | B venv for export/report, FairShip pixi (`FairShip-main-8a1cc53`) for ROOT, MuDIS, transport, reco, replay |
| counts usable by U | `ship_nf.loop.observed` (`loop.py:76`) via `load_report` (`exchange.py:189`) | ship-nf venv |

Endpoint: `r_demo_v0`, a DIS event with at least one FairShip `goodTracks` entry (`evaluate.py:31`). UBT DIS are not
eligible and stay outside `n`.

| panel | origin | proposal | delivery | report | evaluable n / positive y | provenance |
|---|---|---|---|---|---|---|
| left | MANUAL, x=-200 cm, y=0, p=(-2.5, 0, 35) GeV/c, mu+ | `ms:ee8a7917…:0` | `pd-59a8aec01b44790b` | `er@…6f688ef5` | 20 / 0 (AIR only) | new run 2026-11-20, FairShip seed 2100, 88 s |
| middle | NF, U=0.44, tilt ESS 3.6 of 19 | `ms:84a8b087…:0` | `pd-f24028359658d163` | `er@…6ced638b` | 120 / 45 | new run 2026-11-20, FairShip seed 2104, 185 s |
| right | NF, iteration 2 of `demo_loop_v0` | `ms:a5f5eb6d…:0` | `pd-a9b920f29ae36552` | `er@…a3ec99cb` | 120 / 46 | replay of the saved 2026-11-19 run |

The hand-typed muon crosses only AIR and UBT and gives zero positives. That still shows the route: it was accepted,
simulated, reconstructed and joined back to its proposal. The NF muon from the logistic U crossed the decay vessel.

```sh
P=~/thesis/scratch/demo_loop_v0/problem/training_problem
.venv/bin/python -m ship_nf propose-manual $P --out manual/delivery --state -200 0 -2.5 0 35 --pdg -13
../ship-fs-pipeline/scripts/fs_evaluate.sh manual/delivery $P manual/eval 2100      # needs pixi + FairShip worktree
cd ../ship-fs-pipeline && .venv/bin/python scripts/fig1_muon_trace.py RUN_DIR... --out fig1.png
```

Docker was not used. The deliveries were written with the code of commit `9d5b61a` before it was committed, so their
`created_by` names `3f3d2e0`.

## 2. U, ESS and training size (figure 2)

Observed and predicted are separate functions in `loop.py`. `observed()` (line 76) returns, per evaluated proposal,
`n` eligible DIS realizations and `y` of them at the endpoint; a repeated state keeps all its realizations. A failed
FairShip run stops `fs_evaluate.sh` before a report exists, so it never enters `n`. `fit_u` / `utility` (lines 92, 114)
are the predicted `U_k(s)`: the existing binomial logistic on quadratic features. Swapping in kNN or a tree means
replacing those two functions; nothing in repo B changes.

Tilt ESS in the FairShip loop: `1 / sum(pi_i^2)` over the 19 distinct training rows before resampling, base measure
equiprobable rows (stored weight not used, OPEN). For the figure 1 NF delivery: 19 states, 7 distinct progenitors,
ESS 3.6, then 512 resampled rows, 1 proposal. These three numbers are kept apart in the delivery manifest.

Learning curve (`tail_demo.learning_curve`): afterMS, mu-, same flow, 400 epochs, 50 000 samples per fit, sizes
250, 1000, 4000, 13 000 distinct train rows, seeds 1-3, equal weights so ESS = N. Scored on validation only.

| N | val-shell mass ratio (seeds 1/2/3) | median NLL of val_interp |
|---|---|---|
| 250 | 0.22 / 0.28 / 0.37 | -3.8 / -4.6 / -4.9 |
| 1000 | 0.36 / 0.12 / 0.08 | -5.4 / -5.4 / -5.5 |
| 4000 | 0.15 / 0.18 / 0.15 | -5.7 / -5.7 / -5.6 |
| 13 000 | 0.23 / 0.15 / 0.13 | -5.7 / -5.7 / -5.7 |

More centre rows help the interpolated fit until about 4000 rows and do nothing for the 2-2.5 sigma shell, which stays
at 10-35 % of the real mass. So these data fix a size for the centre, not for the tail. What is missing: the same
curve with tail rows in training, and a FairShip-side curve (U quality vs evaluated proposals) with more than 21
proposals.

## 3. Train in the centre, ask for a tail (figure 3)

Variable: `px` of afterMS mu- (`muonsFullMC_afterMS_sample.npz`, sha256 `a45fc522…`). Rule saved in
`result.json`: robust `z = |px - median| / (1.4826 MAD)`; `|z| <= 2` centre (90 % train, 10 % val_interp),
`2 < |z| <= 2.5` validation shell, `|z| > 2.5` test. Rows: 15 806 / 1 756 / 638 / 1 587. The flow is trained on
x, y, tx, ty, log p of train rows; `px` is a projection. Both histograms are divided by their own total (NF: 50 000
samples before any filter). Common checkpoint for figures 3 and 4: `meeting_1121/flow.pt`.

Region request, as you would call it:

```python
acc, counts = sample_region(model, lambda g: z_of(g, d) > 2.5, n_req=1000, seed=2001)
# counts = {"requested": 1000, "generated": 200000, "accepted": 121, "max_batches": 10, "batch": 20000}
```

## 4. Mass ratio vs two distances (figure 4)

For every real test row and every flow sample in the test region: distance to the 5th nearest train row and to the
5th nearest validation-shell row, both after standardising with train mean and std. Bands are quintiles of the real
test distances, the last one open. Ratio per band = `(NF in band / 50 000) / (real in band / 19 787)`, with both
counts and totals in `result.json`. Median NLL of the real rows per band is stored as the extra diagnostic.

The flow puts 41 of 50 000 samples in the test region, against 1 587 of 19 787 real rows, so every band sits between
0.001 and 0.04. The ratio does not fall with distance. It is highest in the last, open band, because that band also
collects flow samples that land far from any real row. The NLL of real test rows rises steadily with distance to train
(median 4 to 62 nats). That makes NLL the clearer distance trend here, and the mass ratio the clearer statement of how
much tail is lost.

Why these choices: the 5th-neighbour distance is the d(x) already used in `density_studies.md` and grows smoothly with
the error there. The mass ratio is `tail_mass_ratio` of `diagnostics.py`, the generative-coverage idea of Naeem et al.
(2020), and it answers "how much of the real mass does the NF give back" in the units a sampler cares about.

## Checks done

- Join: the report names its delivery, `fig1` refuses a mismatch; existing tests `test_evaluate.py`, `test_interop.py`.
- Failures vs negatives: `observed()` test, a proposal without rows adds no trials (`tests/test_loop.py`).
- Partitions: one label per row; the four counts add up to 19 787.
- No test leakage: flow and distance transform fit on train rows only; median/MAD define regions and use all rows.
- Denominators: `tests/test_tail_demo.py` (totals kept, empty band gives None, zero accepted still reported).
- Figures regenerate from saved files (commands below). ship-nf 42 tests pass, ship-fs-pipeline 102 pass, 5 skipped.

```sh
.venv/bin/python -m ship_nf learning-curve AFTERMS.npz --out demo2   # ~5 min, writes learning_curve.json
.venv/bin/python -m ship_nf tail-demo AFTERMS.npz --out demo34       # ~2 min, flow.pt, samples.npz, result.json
.venv/bin/python plots/fig2_learning_curve.py docs/meeting_1121 --loop-delivery ~/thesis/scratch/demos_1120/demo1/nf/delivery --out fig2.png
.venv/bin/python plots/fig3_tail_histogram.py docs/meeting_1121 --out fig3.png
.venv/bin/python plots/fig4_mass_vs_distance.py docs/meeting_1121 --out fig4.png
```

## VERIFIED / PROVISIONAL / OPEN

- VERIFIED: both muons went through FairShip 8a1cc53 today and came back joined to their proposals; the counts above
  are read from the reports; the figures come from the saved JSON.
- PROVISIONAL: k = 2 / 2.5, quintile bands, k = 5 neighbours, sizes and seeds of the learning curve, the hand-typed
  state. afterMS is not P2 (#43).
- OPEN: which ESS threshold moves the ladder; the tail-size question; stored weights.

## What you should check

1. Open `~/thesis/scratch/demos_1120/demo1/nf/eval/report/manifest.json` and confirm the delivery id matches figure 1.
2. Decide whether `px` alone is the right variable for the tail story, or whether to split on the radial score.
3. Figure 4: decide whether the last band should be closed at the largest real distance.
