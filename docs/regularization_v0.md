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
| R1 physical support | p <= max p of P0 (x 1.2) | **wrong**: cuts 18.6 % (10.3 % with x 1.2) of the real afterMS tail | only limits FairShip defines: P2 plane x, y (already applied), 2 <= p <= 400 GeV/c; sources in the R1 section |
| R2 trust region | kNN-5 distance to the pool <= 0.69 | **metric error**: 0.69 was slide 8's robust units; in the loop's units the validation p95 is 0.48. Idea sound, threshold must be measured in the loop's own metric | keep, threshold = p95 of held-out distances in the loop's metric; on P2 OPEN (19 rows) |
| R3 anchored GROW | P0 always in the pool, grown rows <= 50 % of the weight | plausible; three failure modes untested | test in the planted harness before use |
| R4 ε untilted | 20 % of proposals from the untilted flow, so `w <= 1/ε` | the bound **does not hold** with top-m selection, and the harness's "untilted" flow was not P0's | split search and estimate slots: done, `79b23ef`, `eps = 0` by default |

## R1. Physical support: what FairShip itself defines

Searched in FairShip 8a1cc53 (`worktrees/FairShip-main-8a1cc53`) for limits on a muon state at P2. Only code is
cited; no SHiP document was found in the repository that defines an acceptance at P2, so none is claimed.

| quantity | limit | where it comes from (VERIFIED in code) | kind |
|---|---|---|---|
| x at P2 | within ±456.6 cm (`ms.X = 2 * ECN3WallDistance` = 9.132 m, centred at 0) | `python/geometry_config.py:112-123`, plane 2 "ShieldGap downstream of the muon shield" | geometry: the plane ends at the ECN3 walls |
| y at P2 | from the ECN3 floor (-336.4 cm) to `Top` = 520 cm (8.564 m, centred at 91.8 cm) | same lines | geometry |
| z of P2 | `muShield.z + muShield.length + 10 cm` | same lines | geometry |
| species | muons only on plane 2 (`ms.MuonsOnly = True`) | same lines | geometry |
| p, lower | 2 GeV/c: MuDIS skips the muon below it (`fMinPythiaP = 2`) | `newMuonDIS/MuDISProcessor.cxx:45,282,458` | simulation support, not physics |
| p, upper | 400 GeV/c, the primary proton momentum (`fMom = 400`) | `shipgen/FixedTargetGenerator.cxx:71` | kinematics: a secondary muon cannot carry more than the beam |
| p, upper (other place) | trajectories stored only for E in [0, 400] GeV | `macro/run_simScript.py:1117` (`trajFilter.SetEnergyCut`) | storage filter of run_simScript; consistent with the beam, not a separate source |
| direction | none found | - | OPEN: no angular acceptance at P2 in the code |

The loop already enforces the plane's x, y and the 2 GeV/c floor (`loop.py`, `draw`). The 400 GeV/c ceiling is the
only new constraint, and it is kinematics from FairShip's own generator setting, not a number chosen by us.

MEASURED, why a cap taken from data is wrong: afterMS training max p = 111.4 GeV/c, tail max p = 300.6 GeV/c; a cap at
the training max removes 18.6 % of the real tail (10.3 % at 1.2 x). On P2 the 19 rows reach only 59.4 GeV/c.
Slopes need no cap: training `|tx|` reaches 3.53 and `|ty|` 1.30, the tail stays within 0.32 and 0.18.

How R1 can still fail:
- It is loose: px of 100 GeV/c passes R1 although no real muon has it. R1 removes impossible states, not implausible
  ones; that is R2's job.
- Units: features hold `log p` in GeV/c. A unit slip (MeV) would pass or reject everything silently; a test on a
  known row guards it.
- The 400 GeV/c holds for the current beam setting. If `FixedTargetGenerator` changes `fMom`, R1 must follow; read it
  from the same place, do not copy the number elsewhere.

## R2. Trust region (distance to known data)

In one paragraph: slide 8 showed that the flow's error grows with the distance from a state to the nearest training
muons; close to the data the flow is about 3x the noise floor, far away 12x. R2 says: do not send FairShip a state
that is farther from known data than the flow has been checked to be reliable. "Known data" grows as the loop runs
(GROW), so the allowed region grows too, one step at a time. The problem it solves: without it, the flow invents states
far outside anything it has seen (run 1: px of thousands of GeV), FairShip spends time on them, and nothing in the
loop notices. The problem it can create: if nothing passes near the edge, the edge never moves (below).

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

How it can fail (tested in run 3, results in the next section):
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

## R3 results: planted run 3 (2026-11-24)

Data trace: `~/thesis/scratch/planted_v0/run3_r3.json` (210 runs, 6422 s, 5 workers, commit and argv inside the
file). 7 arms × P0 size n0 ∈ {all = 15 806, 200, 19} × 10 seeds × 20 iterations, 20 states per iteration, 20
realizations per state. Untilted flow always fitted on P0 (fix `5a2042d`). Rate = stage-2 passes per
(state × realization) over the last 5 iterations, mean ± sd across seeds (sd across seeds is the honest error; the
pooled Jeffreys interval is 10× narrower). Stuck = seed with no GOAL proposal in the last 5 iterations (`planted.py:201`).

| arm (rule:grow:anchor) | n0 | rate mean ± sd | stuck | GOAL share | luck share | near-dup |
|---|---|---|---|---|---|---|
| greedy_mix:none | all | 0.028 ± 0.019 | 0 | 0.18 | 0 | 0 |
| greedy:k1 | all | 0.087 ± 0.065 | 3 | 0.58 | 0.31 | 0.11 |
| greedy_mix:k1 | all | 0.106 ± 0.024 | 0 | 0.67 | 0.26 | 0.04 |
| greedy_mix:jeffreys | all | 0.092 ± 0.019 | 0 | 0.62 | 0.05 | 0.03 |
| thompson_mix:k1:0.5 | all | 0.086 ± 0.017 | 0 | 0.56 | 0.31 | 0.02 |
| greedy_mix:none | 200 | 0.024 ± 0.034 | 0 | 0.15 | 0 | 0 |
| greedy:k1 | 200 | 0.113 ± 0.060 | 2 | 0.76 | 0.16 | 0.38 |
| greedy_mix:k1 | 200 | 0.104 ± 0.016 | 0 | 0.69 | 0.25 | 0.10 |
| greedy_mix:jeffreys | 200 | 0.113 ± 0.009 | 0 | 0.75 | 0.06 | 0.10 |
| thompson_mix:k1:0.5 | 200 | 0.086 ± 0.034 | 1 | 0.56 | 0.23 | 0.00 |
| greedy_mix:none | 19 | 0.016 ± 0.032 | 6 | 0.11 | 0 | 0 |
| greedy:k1 | 19 | 0.072 ± 0.076 | 5 | 0.48 | 0.25 | 0.38 |
| greedy_mix:k1 | 19 | 0.062 ± 0.054 | 2 | 0.42 | 0.36 | 0.18 |
| greedy_mix:jeffreys | 19 | 0.048 ± 0.057 | 5 | 0.32 | 0.16 | 0.21 |
| thompson_mix:k1:0.5 | 19 | 0.111 ± 0.014 | 0 | 0.71 | 0.18 | 0.01 |

Luck share = fraction of grown rows from the REST region (true cut-1 rate 0.02), i.e. growth from fluctuations
(`planted.py:170`). Near-dup = share of proposals within 2 × jitter (0.1 σ) of a grown row, last 5 iterations
(`planted.py:163`).

VERIFIED (from the table):
- **GROW is the main effect.** Without it the rate is 3-4× lower at every n0.
- **Greedy gets stuck (Oliver).** 2-5 of 10 seeds stuck at every n0. The ε mix removes it at n0 ≥ 200, not at
  n0 = 19 (2-6 stuck). Two kinds of stuck seed (33 stuck runs, listed per seed from the JSON): 16 never leave the
  centre (largest zp in the last 5 iterations < 1.5; all greedy at n0 ≥ 200 are this kind) and 17 left it; 14 of
  those put > 50 % of proposals in DECOY, all at n0 = 19. DECOY is Oliver's case: it passes cut 1 and makes cut 2
  impossible. One Thompson seed at n0 = 200 split between DECOY (48 %) and OUTSIDE (zp up to 26).
- **Jeffreys growth threshold cuts luck growth 5×** (0.25 → 0.05) at n0 ≥ 200, with rate equal at 200 and 13 % lower
  at all. At n0 = 19 it is worse (5 stuck): with few pooled passes the bound rarely clears the base rate, so growth
  stalls.
- **Anchor 0.5 never binds at n0 = all** (grown mass max 0.26-0.35; results identical with and without). At 200 and 19
  it binds and changes little (rate within one sd). It does not help.
- **Spiky pool** is real for greedy (near-dup 0.38) and small for the mixes (≤ 0.10, Thompson ≤ 0.01).
- **Thompson side test.** Only arm with 0 stuck and the best rate at n0 = 19, where the U posterior is wide. At
  n0 ≥ 200 it is 15-20 % below greedy_mix and once stuck at 200. Seen once per n0 with 10 seeds.

INFERENCE: n0 = 19 is the regime of the real loop today (19 P2 rows). There, the posterior width is what keeps
search alive, and Thompson uses it directly. With hundreds of P0 rows the plain ε mix is enough.

PROVISIONAL recommendation (to discuss): `greedy_mix` + GROW k1 as default; drop the anchor (no measurable benefit,
one parameter less); keep the Jeffreys threshold out until the pool is ≥ 200 rows; rerun Thompson vs greedy_mix at
n0 = 19 with 30 seeds before deciding, because that is the only cell where Thompson wins and the real loop sits there.

OPEN: the planted target is one geometry (one decoy, one goal along px). A decoy in another direction, or two goals,
may change the ranking.

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

## Correction: P0 weights are stored, and ignored on purpose

An earlier version of this note said the weights were lost. That was my error: I read a column `weight`, which does
not exist. VERIFIED: the TrainingProblem stores `stored_mctrack_weight` per row (2 rows at 1.0, 17 split clones at
0.1, sum 3.7), declared with `semantics_id: OPEN:stored-mctrack-weight-v0` and `usable_as_nominal_measure: false`
(ship-fs-pipeline `src/ship_fs_pipeline/exchange.py:28,144-145`). `loop.propose` tilts over equiprobable rows by that
decision (DEC-WEIGHT-02), not by accident. Whether the clone weight (0.1 = 1/K, K = `--kaon-pion-splits 10`) should
become the base measure of P0 is a scientific decision for the user. The nightly job uses the same extraction, so
its rows carry the same column.

## What to check

- R1: that 400 GeV/c is the right physical ceiling for the muons at P2 (beam momentum, `FixedTargetGenerator.cxx:71`).
- R4: whether the thesis estimate will be importance weighted. If yes, the search / estimation split above is needed.
- The clone-weight finding: whether split clones should enter P0 with their weight.

## Next steps, in order

1. R1 as revised (one line, one test).
2. Fix the harness so "untilted" is always fitted on P0 rows; rerun `greedy_mix` with GROW.
3. Planted A/B for R2 + R3 (anchor share, Jeffreys growth threshold), logging the R2 rejection share and pool size.
4. Only then move R2 to R4 into `loop.propose`.

## Harness errors found and fixed (planted.py)

| # | error | effect | fix |
|---|---|---|---|
| 1 | GOAL had no upper edge | run 1 counted px of 1 000 sigma as success; run 1 invalid | `92e85be`: GOAL <= zp 10, OUTSIDE passes nothing |
| 2 | "mixture" changed two things at once (random choice and ε) | could not test the simple rule | `92e85be`: `greedy_mix` |
| 3 | the "untilted" flow was refitted on the grown pool | run 2 `greedy_mix` + GROW was not "ε from P0" | `5a2042d` |
| 4 | no momentum ceiling on candidates | absurd states reached the target | `917d496`: R1, p <= 400 GeV/c |
| 5 | results written only at the end | a crash after hours would lose everything | `917d496`: one JSON line per finished run |
| 6 | grown rows mixed into the P0 array | no way to cap or audit their share | `917d496`: kept apart, share logged |
| 7 | P0 was always the full 15 806 rows | grown rows then carry < 0.5 % of the mass; R3's anchor never binds, unlike P2 (19 rows) | `917d496`: `--n0` |

Known limits, not bugs: U is quadratic in the features, so it cannot isolate the thin DECOY slab; pooled Jeffreys
intervals are much narrower than the spread between seeds, so seed spread is what results report.
