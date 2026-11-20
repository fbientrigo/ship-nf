"""Demo-loop pieces on A's side: features, U, tilt, the flow, the 0.3 report reader. No FairShip."""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")
from ship_nf import flow  # noqa: E402
from ship_nf.exchange import EVALUATION_COLUMNS, ContractError, _sha, load_problem, load_report  # noqa: E402
from ship_nf.loop import features, h, kish_ess, propose, to_state, utility  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "training_problem"


def test_features_round_trip():
    _, rows = load_problem(FIX)
    back = [to_state(f, r["z"], r["t"], r["pdg"]) for f, r in zip(features(rows), rows)]
    for r, b in zip(rows, back):
        assert np.allclose([b[c] for c in ("x", "y", "px", "py", "pz")], [r[c] for c in ("x", "y", "px", "py", "pz")])


def test_utility_without_data_is_flat_and_with_data_prefers_the_hot_neighbour():
    q = np.array([[0.0, 0, 0, 0, 1], [100.0, 0, 0, 0, 1]])
    assert np.all(utility(q, [], np.ones(5)) == 1)
    ev = [{"x": x, "y": 0.0, "px": 0.0, "py": 0.0, "pz": np.e, "counts": {"AIR": (40, hits), "UBT": (40, 40)}}
          for x, hits in [(0.0, 10), (1.0, 10), (100.0, 0), (101.0, 0)]]
    u = utility(q, ev, np.ones(5))
    assert u[0] > u[1] > 0  # UBT (excluded) would make both high
    assert u[0] == pytest.approx(21 / 82)


def test_tilt_keeps_support_and_ess_is_n_when_flat():
    assert h(np.zeros(3), 4, 0.1).min() == pytest.approx(0.1 ** 4)
    assert kish_ess(np.ones(19)) == pytest.approx(19)


def test_flow_fits_and_samples_finite():
    x = np.random.default_rng(0).normal(size=(200, 5))
    model, losses = flow.fit(x, seed=0, epochs=50)
    s = model.sample(100, seed=1)
    assert s.shape == (100, 5) and np.isfinite(s).all() and losses[-1] < losses[0]


def test_propose_writes_states_on_the_plane(tmp_path):
    out = propose(FIX, [], tmp_path / "d", m=4, seed=3, n_train=128, n_cand=300)
    m = json.loads((tmp_path / "d" / "manifest.json").read_text())
    rows = pq.read_table(tmp_path / "d" / "proposals.parquet").to_pylist()
    plane = m["generator"]
    assert out["states"] == 4 and m["proposal_kind"] == "MODEL_SAMPLE" and m["protocol_version"] == "0.3"
    assert all(r["parent_state_id"] == "model:" + plane["model_id"] for r in rows)
    assert all(np.linalg.norm([r["px"], r["py"], r["pz"]]) >= 2.0 and r["pz"] > 0 for r in rows)


def test_load_report_refuses_another_delivery(tmp_path):
    propose(FIX, [], tmp_path / "d", m=2, seed=3, n_train=64, n_cand=100)
    ids = [r["proposal_id"] for r in pq.read_table(tmp_path / "d" / "proposals.parquet").to_pylist()]
    rep = tmp_path / "r"
    rep.mkdir()
    schema = pa.schema([(c["name"], pa.string() if c["dtype"] == "string" else pa.int32()) for c in EVALUATION_COLUMNS])
    pq.write_table(pa.Table.from_pylist([{"proposal_id": ids[0], "category": "AIR", "n_dis": 20, "n_r": 1}], schema=schema), rep / "evaluations.parquet")
    d_id = json.loads((tmp_path / "d" / "manifest.json").read_text())["delivery_id"]
    m = {"protocol": "ship-exchange", "protocol_version": "0.3", "artifact_type": "EvaluationReport", "report_id": "er-x",
         "delivery_id": d_id, "endpoint_definition_id": "r_demo_v0@x", "columns": EVALUATION_COLUMNS,
         "files": {"evaluations.parquet": {"sha256": _sha(rep / "evaluations.parquet"), "rows": 1}}}
    (rep / "manifest.json").write_text(json.dumps(m))
    got = load_report(rep, tmp_path / "d")
    assert {r["proposal_id"]: r["counts"] for r in got} == {ids[0]: {"AIR": (20, 1)}, ids[1]: {}}
    (rep / "manifest.json").write_text(json.dumps({**m, "delivery_id": "pd-other"}))
    with pytest.raises(ContractError, match="answers pd-other"):
        load_report(rep, tmp_path / "d")
