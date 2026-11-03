# ship-nf (repo A)

Normalizing-flow side of the SHiP thesis (Torch/GPU will live here). Never imports FairShip, ROOT or repo B; talks to
repo B only through `contract/ship_exchange_v0.md` artifacts.

```sh
uv sync
.venv/bin/python -m ship_nf inspect TRAINING_PROBLEM_DIR          # validate + dependence summary
.venv/bin/python -m ship_nf propose TRAINING_PROBLEM_DIR --out D --n 50 --seed 0
.venv/bin/python -m pytest -q
```

`propose` writes the only 0.1 delivery kind, `EMPIRICAL_RESAMPLE`: a uniform resample of training rows (discrete mass,
no density, not `P0`). It exercises the protocol; it is not a model. `tests/fixtures/training_problem` is a
TrainingProblem published by repo B (FairShip 8a1cc53ec, seeds 201-203); B's interop test fails if it drifts.
