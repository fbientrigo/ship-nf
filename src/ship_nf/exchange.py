"""A side of ship-exchange 0.1 (``contract/ship_exchange_v0.md``): read a TrainingProblem, write a ProposalDelivery.

Written independently of repo B's code (no shared module); the B interoperability test catches drift.
The only delivery A can make in 0.1 is ``EMPIRICAL_RESAMPLE``: a discrete resample of training rows, declared as such.
It is a protocol exercise and a baseline, not a model: no density, no claim about ``P0`` or ``PU``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROTOCOL, VERSIONS = "ship-exchange", {"0.1"}
STATE = [("x", "float64", "cm"), ("y", "float64", "cm"), ("z", "float64", "cm"), ("t", "float64", "ns"),
         ("px", "float64", "GeV/c"), ("py", "float64", "GeV/c"), ("pz", "float64", "GeV/c"), ("pdg", "int32", "-")]
LINEAGE = ("run_id", "progenitor_id", "split_group_id", "trajectory_id")
STATES_COLUMNS = ([{"name": "state_id", "dtype": "string", "unit": "-", "role": "id"}]
                  + [{"name": n, "dtype": d, "unit": u, "role": "state"} for n, d, u in STATE]
                  + [{"name": "stored_mctrack_weight", "dtype": "float64", "unit": "1", "role": "weight"}]
                  + [{"name": n, "dtype": "string", "unit": "-", "role": "lineage"} for n in LINEAGE]
                  + [{"name": n, "dtype": "string", "unit": "-", "role": "status"} for n in ("provenance_status", "admission_status")])
PROPOSALS_COLUMNS = ([{"name": "proposal_id", "dtype": "string", "unit": "-", "role": "id"},
                      {"name": "parent_state_id", "dtype": "string", "unit": "-", "role": "lineage"}]
                     + [{"name": n, "dtype": d, "unit": u, "role": "state"} for n, d, u in STATE])
ARROW = {"string": pa.string(), "float64": pa.float64(), "int32": pa.int32()}
PROBLEM_KEYS = ("protocol", "protocol_version", "artifact_type", "columns", "files", "created_by", "problem_id",
                "source_dataset_id", "state_definition_id", "state_definition", "state_status", "admission_policy_id", "nominal_measure_id",
                "weights", "runs", "open_questions")


class ContractError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _schema(columns) -> pa.Schema:
    return pa.schema([(c["name"], ARROW[c["dtype"]]) for c in columns])


def load_problem(dirpath: Path) -> tuple[dict, list[dict]]:
    """Validated ``(manifest, rows)``; raises ``ContractError`` listing every violation found."""
    dirpath, errors = Path(dirpath), []
    mpath = dirpath / "manifest.json"
    if not mpath.exists():
        raise ContractError([f"missing {mpath}"])
    m = json.loads(mpath.read_text())
    missing = [k for k in PROBLEM_KEYS if k not in m]
    if missing:
        raise ContractError([f"manifest misses {missing}"])
    if m["protocol"] != PROTOCOL or m["protocol_version"] not in VERSIONS:
        raise ContractError([f"incompatible protocol {m['protocol']}/{m['protocol_version']} (accepted: {sorted(VERSIONS)})"])
    if m["artifact_type"] != "TrainingProblem":
        raise ContractError([f"not a TrainingProblem: {m['artifact_type']!r}"])
    if m["columns"] != STATES_COLUMNS:
        errors.append("column spec (name/dtype/unit/role/order) differs from protocol 0.1")
    f, tpath = m["files"].get("states.parquet"), dirpath / "states.parquet"
    if f is None or not tpath.exists():
        raise ContractError(errors + ["missing states.parquet"])
    if _sha(tpath) != f["sha256"]:
        errors.append("states.parquet sha256 mismatch")
    t = pq.read_table(tpath)
    if t.schema != _schema(STATES_COLUMNS):
        raise ContractError(errors + [f"states.parquet schema differs from protocol 0.1: {t.schema.names}"])
    rows = t.to_pylist()
    if len(rows) != f["rows"]:
        errors.append(f"{len(rows)} rows, manifest says {f['rows']}")
    sdef = m["state_definition"]
    if "p2_crossing_state_v0@sha256:" + hashlib.sha256(_canonical(sdef).encode()).hexdigest() != m["state_definition_id"]:
        errors.append("state_definition_id does not hash its state_definition")
    if any(r.get("source_dataset_id") != m["source_dataset_id"] for r in m["runs"]):
        errors.append("runs from a source dataset other than the problem's")
    if any(r.get("geometry_id") != sdef.get("geometry_id") for r in m["runs"]):
        errors.append("runs with a geometry different from the state definition")
    for c in STATES_COLUMNS:
        vals = [r[c["name"]] for r in rows]
        if c["role"] != "weight" and any(v is None for v in vals):
            errors.append(f"null in {c['name']}")
        if c["dtype"] == "float64" and any(v is not None and not math.isfinite(v) for v in vals):
            errors.append(f"non-finite value in {c['name']}")
    for key in ("state_id", "trajectory_id"):
        if len({r[key] for r in rows}) != len(rows):
            errors.append(f"duplicate {key}")
    if any(r["provenance_status"] != "RESOLVED" for r in rows):
        errors.append("non-RESOLVED provenance")
    if m["state_status"] != "PROVISIONAL" or any(r["admission_status"] != m["state_status"] for r in rows):
        errors.append("state_status/admission_status must be PROVISIONAL in 0.1")
    for c in (c for c in STATES_COLUMNS if c["role"] == "weight"):
        w = m["weights"].get(c["name"]) or {}
        if not w.get("semantics_id"):
            errors.append(f"weight {c['name']} without declared semantics")
        if w.get("usable_as_nominal_measure") and m["nominal_measure_id"] is None:
            errors.append(f"weight {c['name']} claims nominal use without nominal_measure_id")
    if errors:
        raise ContractError(errors)
    return m, rows


def summary(m: dict, rows: list[dict]) -> dict:
    """What A must know before fitting anything: how many rows are actually independent, and what the weight is not."""
    return {"problem_id": m["problem_id"], "rows": len(rows), "state_status": m["state_status"],
            **{"n_" + k.removesuffix("_id"): len({r[k] for r in rows}) for k in LINEAGE},
            "weight_semantics": {k: v["semantics_id"] for k, v in m["weights"].items()},
            "nominal_measure_id": m["nominal_measure_id"], "open_questions": len(m["open_questions"])}


def _commit() -> str:
    r = subprocess.run(["git", "-C", str(Path(__file__).parent), "describe", "--always", "--dirty", "--abbrev=40"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def propose_resample(problem_dir: Path, out: Path, n: int, seed: int) -> dict:
    """Uniform resample with replacement of training rows (atoms copied exactly). Not P0: the stored weight is OPEN."""
    m, rows = load_problem(problem_dir)
    rng = random.Random(seed)
    picks = [rng.randrange(len(rows)) for _ in range(n)]
    props = [{"proposal_id": f"rs:{seed}:{i}", "parent_state_id": rows[k]["state_id"], **{c: rows[k][c] for c, _, _ in STATE}}
             for i, k in enumerate(picks)]
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(props, schema=_schema(PROPOSALS_COLUMNS)), out / "proposals.parquet")
    manifest = {"protocol": PROTOCOL, "protocol_version": "0.1", "artifact_type": "ProposalDelivery",
                "training_problem_id": m["problem_id"], "state_definition_id": m["state_definition_id"],
                "proposal_kind": "EMPIRICAL_RESAMPLE", "base_measure": "UNIFORM_OVER_TRAINING_ROWS", "density": None,
                "generator": {"method": "uniform resample with replacement (stdlib random.Random)", "seed": seed, "n": n},
                "columns": PROPOSALS_COLUMNS,
                "files": {"proposals.parquet": {"sha256": _sha(out / "proposals.parquet"), "rows": len(props)}}}
    manifest["delivery_id"] = "pd-" + hashlib.sha256(_canonical(manifest).encode()).hexdigest()[:16]
    manifest["created_by"] = {"repo": "ship-nf", "commit": _commit()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {"delivery_id": manifest["delivery_id"], "proposals": n, "distinct_parents": len(set(picks))}
