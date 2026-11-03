# ship-exchange protocol 0.2 (PROVISIONAL)

The only interface between repo A (`ship-nf`) and repo B (`ship-fs-pipeline`). Each repo carries a byte-identical copy
of this file and its own reader/writer; no Python module is shared. The B interoperability test fails if the two copies
differ. Status words follow `EMPIRICAL_FAIRSHIP_CONTRACT.md`: VERIFIED, PROJECT DECISION, PROVISIONAL, OPEN.

## 1. Artifacts

An artifact is a directory: `manifest.json` + one Parquet table. ROOT files are never part of an artifact; they stay
where they were produced and are referenced by path and sha256 (evidence, never rewritten).

| artifact | direction | table | status in 0.2 |
|---|---|---|---|
| `TrainingProblem` | B -> A | `states.parquet` | defined |
| `ProposalDelivery` | A -> B | `proposals.parquet` | defined, `EMPIRICAL_RESAMPLE` only |
| `EvaluationReport` | B -> analysis | - | not defined; B's results model stays internal to B |

## 2. Common manifest keys (all required)

`protocol = "ship-exchange"`, `protocol_version = "0.2"`, `artifact_type`, an artifact id, `columns` (ordered list of
`{name, dtype, unit, role}`), `files = {<table>: {sha256, rows}}`, `created_by = {repo, commit}`.

Compatibility rule: a reader accepts exactly the `protocol_version` values it lists (0.2 only; 0.1 is rejected). Any change to a column,
unit, dtype, role, enum value or rule below requires a new version; there is no "compatible minor" in 0.x.
A reader rejects: unknown version or type, missing key, missing/extra column, wrong dtype or unit, file sha256 or row
count mismatch, non-finite value in a float column.

## 3. TrainingProblem (B -> A)

Manifest keys besides §2: `problem_id`, `source_dataset_id`, `source_dataset_label`, `permitted_use`, `state_definition_id`, `state_definition`, `state_status`,
`admission_policy_id`, `nominal_measure_id`, `weights`, `runs`, `open_questions`.

`states.parquet` columns, in this order:

| name | dtype | unit | role |
|---|---|---|---|
| `state_id` | string | - | id |
| `x`, `y`, `z` | float64 | cm | state |
| `t` | float64 | ns | state |
| `px`, `py`, `pz` | float64 | GeV/c | state |
| `pdg` | int32 | - | state |
| `stored_mctrack_weight` | float64 | 1 | weight |
| `run_id`, `progenitor_id`, `split_group_id`, `trajectory_id` | string | - | lineage |
| `provenance_status`, `admission_status` | string | - | status |

Rules:

- Frame and point semantics come from `state_definition` (0.2: FairShip global cave frame, P2 entry point).
  `state_definition_id == "p2_crossing_state_v0@sha256:" + sha256(canonical_json(state_definition))`, canonical JSON
  = sorted keys, separators `(",", ":")`. Every `runs[*].geometry_id` equals `state_definition.geometry_id`.
- `source_dataset_id` is the `population_id` of a verifiable population manifest (stage-1 input sha256 + production
  record: FairShip version, stage-1 geometry, field maps from the log, declared launcher); `source_dataset_label` is
  for humans and never compared. Every `runs[*].source_dataset_id` equals it: one problem = one population (a
  synthetic DERIVED input is never pooled with its parent, even under the same geometry).
- Configurations are recorded separately and never inferred: `geometry_id` (geometry content), per-run
  `stage2_config.field.field_id` (map files announced in the log, by sha256), `stage2_config.fairship`, and
  unknowns as strings starting with `OPEN`. Production (stage 1 + stage 2, which made the P2 states) is distinct from
  any downstream FS Sim, which a TrainingProblem does not describe.
- `permitted_use` ⊆ {`PROTOCOL_TEST`, `PHYSICS_TRAINING`, `PHYSICS_BENCHMARK`}. While `state_status = PROVISIONAL` or
  `nominal_measure_id = null`, it must be exactly `["PROTOCOL_TEST"]`: the artifact authorizes protocol tests only.
- `state_id` unique; `trajectory_id` unique (one state per trajectory under this definition; a repeated replay of the
  same stage-1 trajectory is a repeated realization, not a new state).
- Lineage columns are non-null. Rows sharing `progenitor_id` / `split_group_id` are dependent (split clones), not
  independent draws (LINEAGE-REPLICATION-01).
- `provenance_status` is `RESOLVED` in every row.
- `state_status` is `PROVISIONAL` and equals every row's `admission_status`: the rows are *candidate* source states,
  not nominal `SourceState`s. The promotion is a project decision (OPEN).
- Every `weight` column has an entry in `weights` with a non-empty `semantics_id`. `usable_as_nominal_measure` may be
  true only if `nominal_measure_id` is non-null. 0.2: `nominal_measure_id = null`, the stored weight's semantics is
  OPEN, so it must not be used as the nominal measure `P0` (DEC-WEIGHT-02).

## 4. ProposalDelivery (A -> B)

Manifest keys besides §2: `delivery_id`, `training_problem_id`, `state_definition_id`, `use`, `proposal_kind`,
`base_measure`, `density`, `generator = {method, seed, n}`.

`proposals.parquet` columns, in this order: `proposal_id` (string, id), `parent_state_id` (string, lineage), then
`x, y, z, t, px, py, pz, pdg` with the dtypes and units of §3. No weight column: a proposal carries no physical weight.

Rules (0.2):

- `training_problem_id` and `state_definition_id` equal those of the problem it answers; `use` is in its `permitted_use`.
- Three measure objects stay separate: the stored weight (a TrainingProblem column, semantics OPEN), a discrete mass
  (`EMPIRICAL_RESAMPLE`) and a future NF density (not defined). A delivery carries no weight column.
- `proposal_kind = "EMPIRICAL_RESAMPLE"` is the only defined kind: each row is an atom of the training table,
  `parent_state_id` names it and every state column equals the parent's exactly. It is a discrete mass;
  `density` must be `null` and no `log_q` column exists. A continuous-density kind is not defined in 0.2 and is
  rejected (it needs a declared reference measure and Jacobian: next protocol version).
- `base_measure = "UNIFORM_OVER_TRAINING_ROWS"` is the only defined value. It is NOT `P0`: the nominal measure is
  undefined while `nominal_measure_id` is null. Any other value is rejected.
- `proposal_id` unique.

## 5. OPEN, carried by every 0.2 artifact

Physical meaning of the stored weight; P2 promotion to nominal `SourceState`; whether post-shield P2 states suffice for
the downstream endpoint (upstream MS-DIS); vacuum scoring-plane volumes classified `UBT` by name in `newMuonDIS`
(physical interaction eligibility); stage-2 run arguments not recorded by the driver; field identity limited to
log-announced map files.

## 6. Changelog

- 0.2 (2026-11-03, after review GO WITH CHANGES): verifiable `source_dataset_id`, `source_dataset_label`,
  per-run production config, `permitted_use` / `use`. 0.1 artifacts are rejected.
- 0.1 (2026-11-03): first slice.
