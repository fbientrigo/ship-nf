"""System 3 on synthetic data: both arms run and generation 0 is the same flow in both."""

import pytest

pytest.importorskip("scipy")
from ship_nf.selfconsume import run  # noqa: E402
from test_extrapolation import fake_afterms  # noqa: E402


def test_both_arms_share_generation_zero_and_report_metrics():
    r = run(fake_afterms(2000), seed=0, generations=1, n=1000, epochs=10)
    recs = {(x["arm"], x["generation"]): x for x in r["records"]}
    assert set(recs) == {("self", 0), ("self", 1), ("anchored", 0), ("anchored", 1)}
    assert recs[("self", 0)]["ess_over_n"] == pytest.approx(1.0)  # q_g = q_0: all weights equal
    same = lambda r: {k: v for k, v in r.items() if k not in ("arm", "khat")}  # noqa: E731  (khat is nan: no tail)
    assert same(recs[("self", 0)]) == same(recs[("anchored", 0)])
