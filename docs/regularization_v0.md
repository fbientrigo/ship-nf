# Regularization rules for the loop: audit before building (2026-11-24)

Context: the planted-target runs (ship-fs-pipeline `docs/cm39_feedback.md`, section 2.3) showed the loop proposing
muons with px of thousands of GeV, and greedy selection getting stuck in 3 seeds of 10. Four rules were proposed
to stop that. This note checks each one against data before any of them is built, and lists how each can fail.

Labels: MEASURED = computed here, with the command's inputs named; VERIFIED = read in code; INFERENCE; OPEN.
Data for the measurements: afterMS mu- (`tail_demo.load`, 15 806 training rows `|zpx| <= 2`, 1 587 tail rows
`|zpx| > 2.5`) and the P2 TrainingProblem (`scratch/demo_loop_v0/problem/training_problem`, 19 rows).

## Summary

| rule | as first proposed | verdict | revised |
|---|---|---|---|
| R1 physical support | p <= max p of P0 (x 1.2) | **wrong**: cuts 18.6 % (10.3 % with x 1.2) of the real afterMS tail | physics only: p <= 400 GeV/c, the beam momentum |
| R2 trust region | kNN-5 distance to the pool <= 0.69 | **metric error**: 0.69 was slide 8's robust units; in the loop's units the validation p95 is 0.48. Idea sound, threshold must be measured in the loop's own metric | keep, threshold = p95 of held-out distances in the loop's metric; on P2 OPEN (19 rows) |
| R3 anchored GROW | P0 always in the pool, grown rows <= 50 % of the weight | plausible; three failure modes untested | test in the planted harness before use |
| R4 ε untilted | 20 % of proposals from the untilted flow, so `w <= 1/ε` | the bound **does not hold** with top-m selection, and the harness's "untilted" flow was not P0's | keep for search; do not use it to claim bounded weights |

## R1. Physical support

MEASURED (afterMS): training max p = 111.4 GeV/c, tail max p = 300.6 GeV/c. A cap at the training max removes 18.6 %
of the real tail rows; at 1.2 x the training max, 10.3 %. On P2 the 19 rows reach only 59.4 GeV/c. A cap taken from
data removes the region the loop is meant to find, so it is the wrong kind of rule.

MEASURED: slopes do not need a cap. Training `|tx|` reaches 3.53 and `|ty|` 1.30; the tail stays inside (0.32, 0.18).

Revised R1, physics only: `p <= 400 GeV/c`. VERIFIED: the primary proton momentum is 400 GeV/c
(FairShip 8a1cc53 `shipgen/FixedTargetGenerator.cxx:71`, `fMom = 400`); no muon from it can carry more.

How it can still fail:
- It is loose: px of 100 GeV/c passes R1 although no real muon has it. R1 stops impossible states, not implausible
  ones; R2 has to do the rest.
- Units: features hold `log p` in GeV/c. A unit slip (MeV) would pass or reject everything silently. One assert in
  the test on a known row.

## R2. Trust region (distance to known data)

MEASURED (afterMS, kNN-5 on features standardised by the training std, `extrapolation.knn_distance`): validation p95
= 0.48; real tail rows have median distance 0.49 and p95 1.55; 50 % of them lie within 0.48. So at the start R2
admits half of the real tail, and the rest only after GROW moves the frontier. That is the intended behaviour: the
slide 8 study showed the flow's error growing with exactly this kind of distance.

How it can fail:
- **Metric mismatch.** Already happened once (0.69 vs 0.48). The threshold must be computed by the same function and
  the same standardisation that applies it, in the same run, and written in the manifest.
- **Frontier starvation.** With GROW on positives only, if the pass rate just outside the frontier is near zero,
  no positives come back, the pool never grows and the frontier never moves. The loop then looks converged while it
  is stuck. Visible as: the share of candidates rejected by R2 stays high and the pool size stops changing. Both
  should be logged per iteration.
- **Small P0.** With 19 P2 rows, the 5th neighbour is a quarter of the data, and there is no held-out set to take a
  p95 from. Leave-one-out on 19 rows gives a very noisy threshold. On P2 the threshold is OPEN until the nightly job
  has produced more rows.
- **Scale choice.** Distance depends on how features are standardised (`log p` versus slopes). Changing the
  standardisation changes which directions count as far. Fix it once (training std, as here) and record it.
- **Distance is in feature space, not physics.** Two states close in `(x, y, tx, ty, log p)` can cross different
  material. R2 limits the flow's extrapolation error, not the physics difference.

## R3. Anchored GROW

GROW = states FairShip evaluated with at least one pass join the flow's training pool for the next iteration.
The planted runs showed it is required to reach a held-out region (without it, at most 0.20 of proposals reach GOAL).

Evidence for anchoring, from ship-nf `docs/density_studies.md:76-79`: five generations of training on own samples
dropped tail coverage from 0.99 to 0.81-0.88; anchored to real rows it stayed at 0.95-0.98.

How it can fail (none tested yet):
- **Spiky pool.** Grown rows are few (tens) and clustered. Giving them up to 50 % of the weight puts half the flow's
  mass on a handful of points; the flow then fits spikes and proposes near-duplicates. Measure: share of proposals
  within a small distance of a grown row.
- **Luck as signal.** `k = 1 of 20` can be a fluctuation where the true rate is low. Growing from it moves the
  frontier on noise. The simple binomial answer: grow only when the Jeffreys lower bound of `k / n` is above the
  base rate of the run. One line, no new model.
- **Anchor pulls back.** P0 at >= 50 % keeps pulling mass to the centre and can slow discovery. This is a trade-off
  to measure (planted harness, anchor share 0 / 0.5), not to guess.
- **Lineage.** Grown rows are proposals, not P0 muons. If they were ever counted as P0 in an estimate, the estimate
  would be biased. Each grown row must carry its origin (`GROWN`, iteration, proposal id) and the estimate code must
  refuse them.

## R4. ε untilted proposals

VERIFIED (`loop.py:161`): the main part of the selection is top-m by U, which is deterministic given the candidates.
The probability that a state is selected is therefore not the flow's density. Consequences:

- The bound `w = p0 / q <= 1 / ε` holds only if states are **drawn** from the mixture `ε p0 + (1 - ε) q_tilted`.
  With top-m, `q` of the selection is not available in closed form, so the bound is nominal.
- The `log_q_flow` now stored in each delivery (`a576426`) is the flow's density, and the manifest says so
  (`selection: top_m_by_U`). It is the right quantity for the ε slots, which are drawn, and not for the top-m slots.
- "Untilted" is a model of P0, not P0. Its own tail mass is about 1 % of the real one (slides 7-9), so ε slots rarely
  reach the tail; any weight bound is relative to that model.
- **Bug in my harness:** with GROW on, `planted.trajectory` refits the "untilted" flow on the grown pool
  (`planted.py:108`), so it is not P0's model. Run 2's `greedy_mix` result with GROW is for "ε from the current
  pool", not "ε from P0". The no-stuck-seeds finding still stands for that rule; it is not the same rule as R4.

INFERENCE, the clean way out: separate the two jobs. **Search** (find states that pass) may use top-m, Thompson,
GROW, anything; it needs no weights. **Estimation** (a background rate) uses only states drawn by sampling from a
recorded density, for example the ε slots drawn from the untilted P0 flow, each with its `log q`. The ε slots then do
two things at once: keep search from sticking, and form an unbiased (relative to the P0 model) sample.

## Cross-cutting finding: P0 weights are dropped

VERIFIED: stage 1 ran with `--kaon-pion-splits 10` (`scratch/gen-from-scratch-v0/launch_stage1_v2.sh`), and the
lossless P2 note lists clone weights of 0.1 for some of the 19 rows (`worktrees/ship-p2-extraction-v0/notes/
p2_lossless_v0.md`). MEASURED: the TrainingProblem has no weight values (all NaN), and `loop.propose` treats every row
as one muon (`tilt_base: equiprobable distinct training rows`). So a clone of weight 0.1 counts as much as an unsplit
muon. This does not break search, but any statement about P0 (and the "untilted" flow) inherits it. OPEN, and the
nightly job must store weights from the start.

## What to check

- R1: that 400 GeV/c is the right physical ceiling for the muons at P2 (beam momentum, `FixedTargetGenerator.cxx:71`).
- R4: whether the thesis estimate will be importance weighted. If yes, the search / estimation split above is needed.
- The clone-weight finding: whether split clones should enter P0 with their weight.

## Next steps, in order

1. R1 as revised (one line, one test).
2. Fix the harness so "untilted" is always fitted on P0 rows; rerun `greedy_mix` with GROW.
3. Planted A/B for R2 + R3 (anchor share, Jeffreys growth threshold), logging the R2 rejection share and pool size.
4. Only then move R2 to R4 into `loop.propose`.
