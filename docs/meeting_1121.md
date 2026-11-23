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

## A. Reusing the flow's own edge samples (figure A, new training)

`selfconsume.py` (system 3 of `density_studies.md`) does not answer this question. Its arms are "self" (own samples
only) and "anchored" (half real rows). Neither selects the edge, and its tail is the radial score on a random 20 % test
split, not the px rule. So this needed a new pilot: `ship_nf.edge_reuse`.

Data trace: generation 0 is `meeting_1121/flow.pt` (sha256 `173985…`, the figure 3/4 checkpoint) with the split of
`tail_demo.load(npz, seed=1)`. In each round both arms keep the 15 806 real train rows and add m = 574 samples drawn
from their own previous flow. Control takes the first m finite samples. Edge keeps only samples with 2 < |z_px| ≤ 2.5,
searching at most 200 000 proposals (10 × 20 000). m = 574 gives the edge arm the real edge/centre ratio
(638 / 17 562). Both arms fit with the same `flow.fit` seed, Adam at 3e-3, 400 full-batch epochs and
16 380 rows. Each checkpoint is scored on 200 000 fresh unfiltered samples:
`R = (exterior samples on that side / 200 000) / (real exterior rows on that side / 19 787)`. val_interp, val and test
never enter training. The real exterior counts were already known from figure 4.

| seed | arm | gen | edge accepted / generated | left n, R | right n, R | median NLL val_interp |
|---|---|---|---|---|---|---|
| 1 | g0 | 0 | | 24, 0.063 | 89, 0.006 | −5.65 |
| 1 | control | 1 / 2 | | 50, 0.13 / 61, 0.16 | 49, 0.003 / 155, 0.010 | −5.73 / −5.69 |
| 1 | edge | 1 / 2 | 574/120 000, 574/40 000 | 94, 0.24 / 96, 0.25 | 491, 0.031 / 1 520, 0.097 | −5.67 / −5.70 |
| 2 | g0 | 0 | | 43, 0.11 | 77, 0.005 | −5.65 |
| 2 | control | 1 / 2 | | 80, 0.21 / 30, 0.08 | 434, 0.028 / 172, 0.011 | −5.70 / −5.72 |
| 2 | edge | 1 / 2 | 574/120 000, 574/40 000 | 190, 0.49 / 57, 0.15 | 1 125, 0.072 / 1 035, 0.066 | −5.70 / −5.67 |

The left exterior has 38 real rows and the right 1 549. On the right, the edge arm ends both rounds above control in
both seeds, by 2.6 to 10 times, but R stays at or below 0.1. On the left the edge arm is above control in all four
seed-round pairs, at R up to 0.5. With 38 reference rows and 24 to 190 NF samples, that side is noisy: the same
generation-0 flow gives 0.063 or 0.11 depending only on the evaluation stream. Edge acceptance rose from 0.48 % to
1.4 % between rounds, so the edge arm puts more mass on the edge each round. The median NLL of val_interp moves by
at most 0.06 nats between arms, with no consistent sign. At this size there is no visible cost in the centre.
This is an exploratory result from two seeds and two rounds.

```sh
.venv/bin/python -m ship_nf edge-reuse AFTERMS.npz --ckpt docs/meeting_1121/flow.pt --out ~/thesis/scratch/demos_1120/edge_reuse  # ~6 min
.venv/bin/python plots/figA_edge_reuse.py docs/meeting_1121/edge_reuse.json --out docs/meeting_1121/figA_edge_reuse
```

## B. Mass deficit against distance from known data (figure B, saved results)

This is figure 4 redrawn from the same `result.json` and `samples.npz`, with no training. A zero ratio is drawn as a
hatched bar, a band with no reference row says "no estimate", and the open last band shows the largest real
distance and how many NF exterior samples lie past it.

The largest real distance is 4.26 to training and 6.81 to the shell. 11 of the 41 NF exterior samples lie beyond it
in both panels: 11 of the 26 samples in the last d_train band and 11 of the 29 in the last d_val band. Those samples
sit far from every real row. The last band's R (0.032 and 0.036) is the highest because of them, not because the flow
reproduces the far tail. The ratio does not fall with distance. Every band is between 0.001 and 0.04, and the counts
per band are 1 to 7 NF samples outside the last band. The NLL of real exterior rows does rise with distance to
training (median 4.3 to 62 nats per row, `ref_nll_median` in `result.json`). That is a separate statement, about how
unlikely the real rows are under the flow.

```sh
.venv/bin/python plots/figB_mass_vs_distance.py docs/meeting_1121 --out docs/meeting_1121/figB_mass_vs_distance
```

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
- Figure A: VERIFIED that the arms differ only in the edge filter (same rows, size, fit seed, epochs) and that the
  numbers come from `edge_reuse.json`. PROVISIONAL: m = 574, two rounds, two seeds, the 200 000 proposal cap. OPEN:
  whether the gain survives more seeds and rounds, and the left side, where the reference has only 38 rows.
- Figure B: VERIFIED from the saved files, no new training. The beyond-max count reuses the saved distances.

## What you should check

1. Open `~/thesis/scratch/demos_1120/demo1/nf/eval/report/manifest.json` and confirm the delivery id matches figure 1.
2. Decide whether `px` alone is the right variable for the tail story, or whether to split on the radial score.
3. Figure 4: decide whether the last band should be closed at the largest real distance.
4. Figure A: whether m should follow the real edge/centre ratio (574) or be larger, and whether the next run should
   accumulate synthetic rows across rounds instead of replacing them.
5. Figure B: 11 of 41 NF exterior samples lie beyond every real distance. Look at them in 5D before reading the last
   band as tail coverage.

## C. Distribution error against distance (CM slide 8, replaces the NLL figure)

Script `plots/cm_swd_vs_d.py`, output `cm/slide8_swd_vs_distance.{png,pdf}` and `cm/swd_vs_d.json`. Samples the saved
`flow.pt` (10^7 draws, seeds 3000-3009); trains nothing. NLL is a likelihood, not a density error, so it is dropped.

Data trace: afterMS npz (mu-, 19 787 rows, no duplicates) -> features `x, y, tx, ty, log p` -> physical
`(x, y, px, py, pz)` (asserted equal to the raw columns) -> robust scale (median, 1.4826 MAD of train rows) -> groups:
held-out centre, validation shell, and the five fixed `d_train` bands of the real test rows from `result.json`.

- SW: sliced Wasserstein-1 over 500 fixed directions of the 5-D scaled state, half of the real rows vs an equal-size
  NF subset, 200 random splits; floor = the same statistic between the two real halves. Robust-scaled units, not m or
  GeV/c. Position-only and momentum-only versions in the JSON.
- Mass: `Q/P` (NF fraction / real fraction of the group) with binomial log-delta 68 % intervals, and `Q - P`.

- VERIFIED: 41 of 50 000 and the per-band counts of `result.json` are reproduced; physical coordinates match the raw
  file; a generator without tail is flagged as missing mass; SW detects a shift and a sign flip of a correlation with
  unchanged marginals (`tests/test_diagnostics.py`); two runs give the same JSON.
- PROVISIONAL: robust scale, 500 directions, 200 splits; x, y in m comes from the afterMS audit
  (`ship_adaptive_muon_bg/docs/investigations/muon_weight_provenance.md`), while the `loop.features` docstring says cm.
- OPEN: all rows are unweighted, in training and in this comparison. The stored `w` has five values; the 817 rows with
  w > 300 carry 76 % of the weight and the weighted ESS is 1 221 of 19 787 (71 in the test region).
- Check: the last band's SW is driven by NF samples with p up to 1e7 GeV/c and |x| up to 56 m; look at them before
  quoting a number for that band.
