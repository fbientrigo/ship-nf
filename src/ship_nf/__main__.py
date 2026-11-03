"""``python -m ship_nf {inspect,propose}``."""

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
    a = ap.parse_args(argv)
    try:
        out = summary(*load_problem(a.problem)) if a.cmd == "inspect" else propose_resample(a.problem, a.out, a.n, a.seed, a.use)
    except ContractError as e:
        print(json.dumps({"valid": False, "errors": e.errors}, indent=1))
        return 1
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
