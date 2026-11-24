"""``python -m ship_nf {inspect,propose,propose-nf,extrapolation,selfconsume}``."""

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
    p.add_argument("--eps", type=float, default=0.0, help="share of slots drawn untilted for estimation (R4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--delta", type=float, default=0.1)
    p = sub.add_parser("propose-manual", help="one-state MODEL_SAMPLE delivery for a muon typed by hand (origin MANUAL)")
    p.add_argument("problem", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--state", required=True, nargs=5, type=float, metavar=("X", "Y", "PX", "PY", "PZ"), help="cm, GeV/c")
    p.add_argument("--pdg", type=int, default=13)
    p = sub.add_parser("extrapolation", help="systems 1-2: A/B of box vs radial sigma splits, NF error per split and vs d(x)")
    p.add_argument("npz", type=Path, help="afterMS sample, columns [px, py, pz, x, y, z, id, w]")
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("selfconsume", help="system 3: flow retrained on own samples vs anchored on real rows")
    p.add_argument("npz", type=Path)
    p.add_argument("--out", required=True, type=Path)
    for name, h in (("tail-demo", "demos 3-4: flow on the px centre, region request, mass ratio vs d_train / d_val"),
                    ("learning-curve", "demo 2: validation tail quality vs number of distinct train rows")):
        p = sub.add_parser(name, help=h)
        p.add_argument("npz", type=Path)
        p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("edge-reuse", help="pilot: retrain on own edge samples vs unfiltered, R in the px exterior")
    p.add_argument("npz", type=Path)
    p.add_argument("--ckpt", required=True, type=Path, help="centre-trained flow.pt of tail-demo (generation 0)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args(argv)
    try:
        if a.cmd in ("tail-demo", "learning-curve"):
            from ship_nf import tail_demo
            out = (tail_demo.run if a.cmd == "tail-demo" else tail_demo.learning_curve)(a.npz, a.out)
        elif a.cmd == "edge-reuse":
            from ship_nf.edge_reuse import run as er_run
            res = er_run(a.npz, a.ckpt, a.out, epochs=a.epochs)
            out = {"out": str(a.out / "edge_reuse.json"), "m": res["settings"]["m"], "commit": res["commit"]}
        elif a.cmd == "selfconsume":
            from ship_nf.selfconsume import study as sc_study
            res = sc_study(a.npz)
            a.out.write_text(json.dumps(res, indent=1))
            out = {"out": str(a.out), "runs": len(res["runs"]), "data": res["data"]}
        elif a.cmd == "extrapolation":
            from ship_nf.extrapolation import ab_test
            res = ab_test(a.npz)
            a.out.write_text(json.dumps(res, indent=1))
            out = {"out": str(a.out), "runs": len(res["runs"]), "data": res["data"]}
        elif a.cmd == "propose-manual":
            from ship_nf.loop import propose_manual
            out = propose_manual(a.problem, a.out, *a.state, a.pdg)
        elif a.cmd == "propose-nf":
            from ship_nf.loop import propose  # torch only when needed
            out = propose(a.problem, a.history, a.out, a.m, a.seed, a.alpha, a.delta, eps=a.eps)
        else:
            out = summary(*load_problem(a.problem)) if a.cmd == "inspect" else propose_resample(a.problem, a.out, a.n, a.seed, a.use)
    except ContractError as e:
        print(json.dumps({"valid": False, "errors": e.errors}, indent=1))
        return 1
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
