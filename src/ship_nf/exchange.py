"""A side of ship-exchange 0.3 (``contract/ship_exchange_v0.md``): read a TrainingProblem, write a ProposalDelivery,
read an EvaluationReport.

Written independently of repo B's code (no shared module); the B interoperability test catches drift.
Deliveries: ``EMPIRICAL_RESAMPLE`` (a discrete resample of training rows, the protocol baseline) and, from 0.3,
``MODEL_SAMPLE`` (new states drawn from a fitted model). Neither carries a density B may use, nor a claim about ``P0``.
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

PROTOCOL, VERSION, MODEL_VERSION = "ship-exchange", "0.2", "0.3"
VERSIONS = {VERSION, MODEL_VERSION}
USES = ("PROTOCOL_TEST", "PHYSICS_TRAINING", "PHYSICS_BENCHMARK")
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
                "source_dataset_id", "source_dataset_label", "permitted_use", "state_definition_id", "state_definition", "state_status", "admission_policy_id", "nominal_measure_id",
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
        errors.append("state_status/admission_status must be PROVISIONAL in 0.2")
    if not m["permitted_use"] or not set(m["permitted_use"]) <= set(USES):
        errors.append(f"unknown permitted_use {m['permitted_use']}")
    elif (m["state_status"] == "PROVISIONAL" or m["nominal_measure_id"] is None) and m["permitted_use"] != ["PROTOCOL_TEST"]:
        errors.append("PROVISIONAL states / null nominal measure permit PROTOCOL_TEST only")
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
            "permitted_use": m["permitted_use"], "source_dataset_label": m["source_dataset_label"],
            **{"n_" + k.removesuffix("_id"): len({r[k] for r in rows}) for k in LINEAGE},
            "weight_semantics": {k: v["semantics_id"] for k, v in m["weights"].items()},
            "nominal_measure_id": m["nominal_measure_id"], "open_questions": len(m["open_questions"])}


def _commit() -> str:
    r = subprocess.run(["git", "-C", str(Path(__file__).parent), "describe", "--always", "--dirty", "--abbrev=40"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def require_use(m: dict, use: str) -> None:
    """A refuses any use the problem does not permit (0.2: PROVISIONAL states permit PROTOCOL_TEST only)."""
    if use not in m["permitted_use"]:
        raise ContractError([f"use {use!r} not permitted by problem {m['problem_id']} ({m['permitted_use']})"])


def propose_resample(problem_dir: Path, out: Path, n: int, seed: int, use: str = "PROTOCOL_TEST") -> dict:
    """Uniform resample with replacement of training rows (atoms copied exactly). Not P0: the stored weight is OPEN."""
    m, rows = load_problem(problem_dir)
    require_use(m, use)
    rng = random.Random(seed)
    picks = [rng.randrange(len(rows)) for _ in range(n)]
    props = [{"proposal_id": f"rs:{seed}:{i}", "parent_state_id": rows[k]["state_id"], **{c: rows[k][c] for c, _, _ in STATE}}
             for i, k in enumerate(picks)]
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(props, schema=_schema(PROPOSALS_COLUMNS)), out / "proposals.parquet")
    delivery_id = _write_delivery(out, m, use, props, VERSION, "EMPIRICAL_RESAMPLE", "UNIFORM_OVER_TRAINING_ROWS",
                                  {"method": "uniform resample with replacement (stdlib random.Random)", "seed": seed, "n": n})
    return {"delivery_id": delivery_id, "proposals": n, "distinct_parents": len(set(picks))}


def _write_delivery(out: Path, m: dict, use: str, props: list[dict], version: str, kind: str, base_measure, generator: dict) -> str:
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(props, schema=_schema(PROPOSALS_COLUMNS)), out / "proposals.parquet")
    manifest = {"protocol": PROTOCOL, "protocol_version": version, "artifact_type": "ProposalDelivery",
                "training_problem_id": m["problem_id"], "state_definition_id": m["state_definition_id"], "use": use,
                "proposal_kind": kind, "base_measure": base_measure, "density": None, "generator": generator,
                "columns": PROPOSALS_COLUMNS,
                "files": {"proposals.parquet": {"sha256": _sha(out / "proposals.parquet"), "rows": len(props)}}}
    manifest["delivery_id"] = "pd-" + hashlib.sha256(_canonical(manifest).encode()).hexdigest()[:16]
    manifest["created_by"] = {"repo": "ship-nf", "commit": _commit()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest["delivery_id"]


def write_model_delivery(problem_dir: Path, out: Path, states: list[dict], generator: dict, use: str = "PROTOCOL_TEST") -> str:
    """``MODEL_SAMPLE`` delivery (0.3): ``states`` are new model draws, each a dict of the STATE columns.

    ``generator`` must name ``model_id``; it becomes every row's lineage (``parent_state_id = "model:<model_id>"``).
    """
    m, _ = load_problem(problem_dir)
    require_use(m, use)
    parent = "model:" + generator["model_id"]
    props = [{"proposal_id": f"ms:{generator['model_id']}:{i}", "parent_state_id": parent, **{c: s[c] for c, _, _ in STATE}}
             for i, s in enumerate(states)]
    return _write_delivery(out, m, use, props, MODEL_VERSION, "MODEL_SAMPLE", None, generator)


EVALUATION_COLUMNS = [{"name": "proposal_id", "dtype": "string", "unit": "-", "role": "id"},
                      {"name": "category", "dtype": "string", "unit": "-", "role": "label"},
                      {"name": "n_dis", "dtype": "int32", "unit": "1", "role": "count"},
                      {"name": "n_r", "dtype": "int32", "unit": "1", "role": "count"}]


def load_report(report_dir: Path, delivery_dir: Path) -> list[dict]:
    """One dict per proposal: its state columns + ``counts = {category: (n_dis, n_r)}``. Refuses a report of another delivery."""
    report_dir, delivery_dir = Path(report_dir), Path(delivery_dir)
    m = json.loads((report_dir / "manifest.json").read_text())
    d = json.loads((delivery_dir / "manifest.json").read_text())
    errors = []
    if (m["protocol"], m["protocol_version"], m["artifact_type"]) != (PROTOCOL, MODEL_VERSION, "EvaluationReport"):
        errors.append(f"not an 0.3 EvaluationReport: {m['protocol_version']}/{m['artifact_type']}")
    if m["delivery_id"] != d["delivery_id"]:
        errors.append(f"report answers {m['delivery_id']}, not {d['delivery_id']}")
    if m["columns"] != EVALUATION_COLUMNS or _sha(report_dir / "evaluations.parquet") != m["files"]["evaluations.parquet"]["sha256"]:
        errors.append("evaluations.parquet columns or sha256 differ from the manifest")
    if errors:
        raise ContractError(errors)
    props = {r["proposal_id"]: r for r in pq.read_table(delivery_dir / "proposals.parquet").to_pylist()}
    rows = pq.read_table(report_dir / "evaluations.parquet").to_pylist()
    if not {r["proposal_id"] for r in rows} <= set(props) or any(not 0 <= r["n_r"] <= r["n_dis"] for r in rows):
        raise ContractError(["report rows name unknown proposals or have n_r outside [0, n_dis]"])
    out = {pid: {**p, "counts": {}, "endpoint_definition_id": m["endpoint_definition_id"]} for pid, p in props.items()}
    for r in rows:
        out[r["proposal_id"]]["counts"][r["category"]] = (r["n_dis"], r["n_r"])
    return list(out.values())
