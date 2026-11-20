"""``python -m ship_nf {inspect,propose,propose-nf,extrapolation}``."""

import argparse
import json
import sys
from pathlib import Path

from ship_nf.exchange import ContractError, load_problem, propose_resample, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ship_nf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect").add_argument("problem", type=Path)
    p = sub.add_parser("propose", help="EMPIRICAL_RESAMPLE delivery (protocol baseline, not a model)")
    p.add_argument("problem", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use", default="PROTOCOL_TEST", help="must be in the problem's permitted_use")
    p = sub.add_parser("propose-nf", help="MODEL_SAMPLE delivery: tilt by U from past reports, fit RealNVP, keep top-m by U")
    p.add_argument("problem", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--history", nargs=2, action="append", default=[], type=Path, metavar=("DELIVERY", "REPORT"))
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--delta", type=float, default=0.1)
    p = sub.add_parser("extrapolation", help="systems 1-2: A/B of box vs radial sigma splits, NF error per split and vs d(x)")
    p.add_argument("npz", type=Path, help="afterMS sample, columns [px, py, pz, x, y, z, id, w]")
    p.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "extrapolation":
            from ship_nf.extrapolation import ab_test
            res = ab_test(a.npz)
            a.out.write_text(json.dumps(res, indent=1))
            out = {"out": str(a.out), "runs": len(res["runs"]), "data": res["data"]}
        elif a.cmd == "propose-nf":
            from ship_nf.loop import propose  # torch only when needed
            out = propose(a.problem, a.history, a.out, a.m, a.seed, a.alpha, a.delta)
        else:
            out = summary(*load_problem(a.problem)) if a.cmd == "inspect" else propose_resample(a.problem, a.out, a.n, a.seed, a.use)
    except ContractError as e:
        print(json.dumps({"valid": False, "errors": e.errors}, indent=1))
        return 1
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
