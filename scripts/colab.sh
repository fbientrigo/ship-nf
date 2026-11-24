#!/bin/bash
# Run one ship_nf module on a Colab VM and bring its JSON output back; the VM is always stopped at the end.
#   scripts/colab.sh [--gpu T4] MODULE LOCAL_OUT [module args...]
#   e.g. scripts/colab.sh ship_nf.planted ~/thesis/scratch/planted_v0/run3.json --seeds 20 --workers 2
# The code shipped is `git archive HEAD` (commit recorded in the output), so commit first. The afterMS sample is
# uploaded to the VM and read through SHIP_NF_AFTERMS. One-time setup by the user: `colab sessions` and log in.
# When to use it instead of the server: a bigger flow, more seeds than 8 cores finish in an hour, or a busy night.
set -euo pipefail
GPU=(); [ "${1:-}" = --gpu ] && { GPU=(--gpu "$2"); shift 2; }
MOD=$1; OUT=$2; shift 2
DATA=${SHIP_NF_AFTERMS:-/home/fabian/thesis/ship_adaptive_muon_bg/data/samples/muonsFullMC_afterMS_sample.npz}
REPO=$(cd "$(dirname "$0")/.." && pwd); S=shipnf-$(date +%H%M%S); TMP=$(mktemp -d)
git -C "$REPO" archive --prefix=ship-nf/ -o "$TMP/code.tar" HEAD
colab new -s "$S" "${GPU[@]}"
trap 'colab stop -s "$S"; rm -rf "$TMP"' EXIT
colab upload -s "$S" "$TMP/code.tar" /content/code.tar
colab upload -s "$S" "$DATA" /content/afterms.npz
printf '%s\n' '!tar xf /content/code.tar -C /content && pip install -q --no-deps -e /content/ship-nf' | colab exec -s "$S"
colab exec -s "$S" --env SHIP_NF_AFTERMS=/content/afterms.npz <<PY
import subprocess, sys
sys.exit(subprocess.run([sys.executable, "-m", "$MOD", "/content/out.json", *$(printf '%s\n' "$@" | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read().split()))')], cwd="/content/ship-nf").returncode)
PY
colab download -s "$S" /content/out.json "$OUT"
echo "wrote $OUT (commit $(git -C "$REPO" rev-parse --short HEAD), session $S stopped)"
