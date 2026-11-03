"""A reads a TrainingProblem (fixture published by repo B from FairShip 8a1cc53ec seeds 201-203) and writes a delivery.
No ROOT, no FairShip, no import of repo B."""

import json
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ship_nf.exchange import STATE, ContractError, _sha, load_problem, propose_resample, summary

FIX = Path(__file__).parent / "fixtures" / "training_problem"
SRC = Path(__file__).parents[1] / "src"


@pytest.fixture
def tp(tmp_path):
    shutil.copytree(FIX, tmp_path / "tp")
    return tmp_path / "tp"


def edit(d, manifest=None, rows=None, schema=None, reseal=True):
    m = json.loads((d / "manifest.json").read_text())
    if rows is not None:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema or pq.read_schema(d / "states.parquet")), d / "states.parquet")
        if reseal:
            m["files"]["states.parquet"] = {"sha256": _sha(d / "states.parquet"), "rows": len(rows)}
    if manifest:
        manifest(m)
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ContractError) as e:
        load_problem(d)
    return " | ".join(e.value.errors)


def rows(d):
    return pq.read_table(d / "states.parquet").to_pylist()


def test_no_root_or_fairship_anywhere_in_a():
    with pytest.raises(ImportError):
        import ROOT  # noqa: F401
    pat = re.compile(r"^\s*(import|from)\s+(ROOT|ship_fs_pipeline|ship_muon_bg)\b", re.M)
    assert not [p for p in SRC.rglob("*.py") if pat.search(p.read_text())]


def test_consumes_problem_and_knows_rows_are_not_independent():
    s = summary(*load_problem(FIX))
    assert (s["rows"], s["n_progenitor"], s["n_split_group"], s["n_run"]) == (19, 7, 7, 3)
    assert s["state_status"] == "PROVISIONAL" and s["nominal_measure_id"] is None
    assert s["weight_semantics"]["stored_mctrack_weight"].startswith("OPEN")


def test_resample_is_deterministic_and_atoms_are_exact(tmp_path):
    a, b = propose_resample(FIX, tmp_path / "a", 30, 3), propose_resample(FIX, tmp_path / "b", 30, 3)
    assert a["delivery_id"] == b["delivery_id"]
    m = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert m["proposal_kind"] == "EMPIRICAL_RESAMPLE" and m["density"] is None and m["base_measure"] == "UNIFORM_OVER_TRAINING_ROWS"
    par = {r["state_id"]: r for r in rows(FIX)}
    props = pq.read_table(tmp_path / "a" / "proposals.parquet").to_pylist()
    assert all(p[c] == par[p["parent_state_id"]][c] for p in props for c, _, _ in STATE)
    assert "weight" not in pq.read_schema(tmp_path / "a" / "proposals.parquet").names


@pytest.mark.parametrize("v", ["0.2", "1.0"])
def test_rejects_incompatible_version(tp, v):
    assert "incompatible protocol" in edit(tp, manifest=lambda m: m.update(protocol_version=v))


def test_rejects_wrong_unit(tp):
    def f(m):
        m["columns"][5]["unit"] = "MeV/c"
    assert "column spec" in edit(tp, manifest=f)


def test_rejects_wrong_dtype(tp):
    s = pq.read_schema(tp / "states.parquet")
    s = s.set(s.get_field_index("x"), pa.field("x", pa.float32()))
    assert "schema differs" in edit(tp, rows=rows(tp), schema=s)


def test_rejects_duplicates_and_clones_without_lineage(tp):
    r = rows(tp)
    r[2]["progenitor_id"] = None
    msg = edit(tp, rows=r + r[:1])
    assert "duplicate state_id" in msg and "null in progenitor_id" in msg


def test_rejects_ambiguous_provenance(tp):
    r = rows(tp)
    r[0]["provenance_status"] = "AMBIGUOUS"
    assert "non-RESOLVED" in edit(tp, rows=r)


def test_rejects_weight_without_semantics_or_as_nominal(tp):
    assert "without declared semantics" in edit(tp, manifest=lambda m: m["weights"].pop("stored_mctrack_weight"))


def test_rejects_weight_claimed_nominal(tp):
    assert "claims nominal" in edit(tp, manifest=lambda m: m["weights"]["stored_mctrack_weight"].update(usable_as_nominal_measure=True))


def test_rejects_geometry_mismatch(tp):
    assert "geometry different" in edit(tp, manifest=lambda m: m["runs"][1].update(geometry_id="other"))


def test_rejects_run_from_other_dataset(tp):
    assert "source dataset other" in edit(tp, manifest=lambda m: m["runs"][0].update(source_dataset_id="synthetic:x"))


def test_rejects_incomplete(tp):
    assert "sha256 mismatch" in edit(tp, rows=rows(tp)[:5], reseal=False)
    (tp / "states.parquet").unlink()
    with pytest.raises(ContractError, match="missing states.parquet"):
        load_problem(tp)


def test_rejects_wrong_artifact_type(tp):
    assert "not a TrainingProblem" in edit(tp, manifest=lambda m: m.update(artifact_type="ProposalDelivery"))
