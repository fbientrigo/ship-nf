# Density studies: extrapolation (systems 1-2) and self-consumption (system 3), 2026-11-19

Both studies use the afterMS sample of `ship_adaptive_muon_bg`
(`data/samples/muonsFullMC_afterMS_sample.npz`, sha256 `a45fc522...6f16`, 40 000 rows, one row with p = 0 dropped).
μ+ and μ− are run separately and never pooled. The stored weight `w` is not used: its meaning is still OPEN. No
FairShip run is involved. The flow is `ship_nf.flow` (RealNVP, 6 couplings, 64 hidden units), trained on CPU.

Reproduce:

```sh
.venv/bin/python -m ship_nf extrapolation AFTERMS.npz --out ab_extrapolation.json   # ~7 min
.venv/bin/python -m ship_nf selfconsume   AFTERMS.npz --out selfconsume.json        # ~6 min
```

Outputs of this run: `~/thesis/scratch/density_v0/{ab_extrapolation,selfconsume}.json`.

## Systems 1-2: box vs radial sigma splits

`px` and `py` get robust z-scores (median and 1.4826 × MAD). Rows within 2σ are the core: 90 % train the flow and
10 % are kept as interpolated validation. Rows between 2σ and 2.5σ are the validation shell, and the rest is the test
set. Version A (`box`) asks each component to be inside its band. Version B (`radial`) uses
√(z_x² + z_y²), so the corners of the square move from core to shell or test. Three seeds per cell. NLL is in nats,
per row, in feature space.

| charge | version | split | rows | NLL (mean ± sd over seeds) | coverage | flow mass / data mass |
|---|---|---|---|---|---|---|
| μ+ (−13) | box | train | 15 236 | −4.80 ± 0.00 | 0.98 | 1.17 |
| | | val_interp | 1 692 | −4.72 ± 0.05 | 1.00 | 1.17 |
| | | val (shell) | 1 130 | −2.88 ± 0.55 | 0.81 | 0.28 |
| | | test | 2 154 | 24.3 ± 12.2 | 0.17 | 0.012 |
| | radial | train | 14 620 | −4.83 ± 0.02 | 0.98 | 1.23 |
| | | val_interp | 1 624 | −4.69 ± 0.15 | 1.00 | 1.23 |
| | | val (shell) | 1 348 | −2.33 ± 0.59 | 0.80 | 0.21 |
| | | test | 2 620 | 35.5 ± 27.6 | 0.14 | 0.008 |
| μ− (13) | box | train | 14 748 | −4.81 ± 0.01 | 0.98 | 1.18 |
| | | val_interp | 1 638 | −4.75 ± 0.03 | 1.00 | 1.18 |
| | | val (shell) | 1 199 | −2.61 ± 0.75 | 0.85 | 0.30 |
| | | test | 2 202 | 41.1 ± 34.0 | 0.18 | 0.016 |
| | radial | train | 14 106 | −4.85 ± 0.01 | 0.98 | 1.24 |
| | | val_interp | 1 567 | −4.71 ± 0.09 | 1.00 | 1.24 |
| | | val (shell) | 1 441 | −2.39 ± 0.88 | 0.80 | 0.23 |
| | | test | 2 673 | 63 ± 91 | 0.16 | 0.014 |

NLL of validation and test rows, binned by d(x) (distance to the 5th nearest training row, standardised features,
quintiles):

| charge, version | d upper edges | NLL per bin |
|---|---|---|
| μ+ box | 0.14, 0.20, 0.32, 0.62, 7.0 | −6.9, −4.5, −0.7, 9.3, 44 |
| μ+ radial | 0.14, 0.21, 0.33, 0.63, 9.6 | −6.8, −4.1, 1.2, 16.8, 66 |
| μ− box | 0.14, 0.20, 0.31, 0.60, 6.8 | −6.8, −4.3, −0.2, 12.7, 78 |
| μ− radial | 0.15, 0.22, 0.34, 0.64, 14.4 | −6.6, −3.7, 2.5, 24.7, 122 |

What the numbers say:

- Interpolation costs almost nothing. The held-out core rows score like the training rows (NLL within 0.15) and are
  fully covered.
- The flow degrades in the 2-2.5σ shell and fails beyond it. In the shell it puts 20-30 % of the mass the data has
  there. In the test set it puts 1-2 %, and only 14-18 % of the real test points have a flow sample nearby.
- The error grows smoothly with d(x) in every cell, so d(x) is a usable extrapolation score. The last quintile
  carries most of the error.
- Box against radial: the pattern is the same. Radial sends the corners to the test set (about 450 more test rows),
  and its test NLL is higher and much noisier between seeds. The two test sets differ, so their NLLs cannot be
  compared directly: the comparison is about which rows each version calls "extrapolated", not about which flow is
  better.
- The test NLL mean is driven by a few extreme rows (seed sd up to 91). Report the median or quantiles next to the
  mean in the next run.

## System 3: retrained on own samples vs anchored on real rows

μ− only, two seeds, five generations, 10 000 samples per generation, 200 epochs. The anchored arm replaces half of each
generation's samples with real training rows. Tail = test rows with radial score above 2.5 (554 and 527 rows).

| seed | arm | generation | ESS/N of q0/qg | k-hat | tail coverage | tail mass ratio | held-out NLL |
|---|---|---|---|---|---|---|---|
| 1 | self | 0 → 5 | 1.00 → 0.31 | 0.54-0.66 | 0.99 → 0.81 | 0.88 → 0.96 | −4.58 → −3.83 |
| 1 | anchored | 0 → 5 | 1.00 → 0.35 | 0.43-0.82 | 0.99 → 0.95 | 0.88 → 1.04 | −4.58 → −4.14 |
| 2 | self | 0 → 5 | 1.00 → 0.20 | 0.40-0.71 | 0.99 → 0.88 | 1.00 → 0.92 | −4.49 → −3.95 |
| 2 | anchored | 0 → 5 | 1.00 → 0.08 | 0.22-0.57 | 0.99 → 0.98 | 1.00 → 1.03 | −4.49 → −4.35 |

(per-generation values in `selfconsume.json`)

What the numbers say:

- Retraining on its own samples loses the tail and the fit: tail coverage falls from 0.99 to 0.81-0.88 and the
  held-out NLL rises by 0.5-0.75 nats in five generations.
- Half real rows per generation stop most of it: tail coverage stays at 0.95-0.98 and NLL rises by 0.1-0.4.
- ESS/N of q0/qg falls in both arms and does not separate them. A refit flow drifts away from q0 even when it is
  good, so this ESS measures distance from generation 0 rather than quality. k-hat stays mostly between 0.4 and 0.7
  ("usable with care") in both arms. Tail coverage and held-out NLL are the metrics that separate the arms here.
- Two seeds and five generations show a direction, not a rate. More seeds are needed before quoting numbers.

## VERIFIED / PROVISIONAL / OPEN

- VERIFIED
  - The code computes what the docstrings say. Tests recover a known Pareto shape for k-hat, full vs missing-tail
    coverage, and the split bands.
  - The numbers above come from the recorded JSON outputs.
- PROVISIONAL
  - k_train = 2, k_val = 2.5, robust σ, 5 d-bins, flow size, 400 / 200 epochs, ρ = 0.5. All are defaults chosen by the
    agent.
  - The afterMS surface and its coordinates (#43). The density results do not depend on the coordinate transform,
    but the surface is not P2.
- OPEN
  - Which split version system 1 should use.
  - The ESS definition and threshold for the staircase.
  - Whether the "real rows" of the anchored arm should be the states FairShip evaluated in the loop. They are in the
    design, but this study draws them from the sample.

## References (from memory, to be checked)

Kish (1965) *Survey Sampling*. Elvira, Martino & Robert (2022) *Int. Stat. Rev.* 90(3):525-550. Vehtari, Simpson,
Gelman, Yao & Gabry (2024) *JMLR* 25(72):1-58. Zhang & Stephens (2009) *Technometrics* 51(3):316-325. Yao, Vehtari,
Simpson & Gelman (2018) *ICML*. Naeem, Oh, Uh, Choi & Yoo (2020) *ICML*. Shumailov et al. (2024) *Nature* 631:755-759.
Alemohammad et al. (2024) *ICLR*.
