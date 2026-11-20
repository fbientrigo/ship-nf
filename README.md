# ship-nf (repo A)

The normalizing-flow side of the SHiP thesis. It never imports FairShip, ROOT or repo B
(`../ship-fs-pipeline`). The two repos talk only through the artifacts of `contract/ship_exchange_v0.md`
(protocol 0.3), and each repo keeps an identical copy of that file.

```sh
uv sync --extra flow                     # numpy, scipy, torch (CPU wheels)
.venv/bin/python -m pytest -q
```

## Where things are

Look here before writing a new function; most building blocks already exist.

| Module | What it does | Main functions |
|---|---|---|
| `exchange.py` | Reads a TrainingProblem, writes ProposalDeliveries (`EMPIRICAL_RESAMPLE`, `MODEL_SAMPLE`), reads EvaluationReports | `load_problem`, `propose_resample`, `write_model_delivery`, `load_report` |
| `flow.py` | Small RealNVP: fit by maximum likelihood, sample, log-density in feature space | `fit`, `Flow.sample`, `Flow.log_prob` |
| `loop.py` | One A step of the demo loop: U classifier from FairShip counts, tilt, flow, top-m proposals | `features`, `to_state`, `fit_u`, `utility`, `h`, `propose` |
| `diagnostics.py` | ESS and tail metrics, with references | `kish_ess`, `psis_khat`, `coverage`, `tail_mass_ratio` |
| `extrapolation.py` | Systems 1-2: sigma-band splits (box vs radial) and flow error per split and vs kNN distance | `pt_z`, `score`, `split`, `study`, `ab_test` |
| `selfconsume.py` | System 3: flow retrained on its own samples vs anchored on real rows | `run`, `study` |

## Commands

```sh
.venv/bin/python -m ship_nf inspect TRAINING_PROBLEM_DIR
.venv/bin/python -m ship_nf propose TRAINING_PROBLEM_DIR --out D --n 50 --seed 0          # protocol baseline
.venv/bin/python -m ship_nf propose-nf TRAINING_PROBLEM_DIR --out D --m 5 --seed 0 \
    [--history DELIVERY REPORT ...]                                                       # demo loop A step
.venv/bin/python -m ship_nf extrapolation AFTERMS.npz --out ab.json                       # systems 1-2
.venv/bin/python -m ship_nf selfconsume AFTERMS.npz --out sc.json                         # system 3
```

The demo loop that alternates A and FairShip lives in repo B: `scripts/demo_loop.sh`. Results and decisions are in
`../ship-fs-pipeline/docs/slice4_demo_loop.md` and `docs/density_studies.md` (this repo).

`tests/fixtures/training_problem` is a TrainingProblem published by repo B (FairShip 8a1cc53, seeds 201-203); B's
interop test fails if it drifts. The afterMS sample used by the density studies is
`ship_adaptive_muon_bg/data/samples/muonsFullMC_afterMS_sample.npz`; each result records its path and sha256.
