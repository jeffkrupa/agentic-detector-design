#!/bin/bash
# HTCondor worker wrapper for the E2 ceiling structure-test experiment.
#
# Runs on a CERN batch node (no shared cwd with the submit host). It:
#   1. sources the project venv via its absolute path,
#   2. cd's to the repo (so config.yaml + relative imports resolve),
#   3. exec's the experiment module, passing through all condor arguments.
#
# All experiment args (--k, --n-layers, --out-jsonl, ...) are forwarded verbatim,
# so the single submit file drives both the tiny test and the real ceiling runs.
set -euo pipefail

REPO=/eos/user/j/jeffkrup/agentic/agentic-detector-design
VENV="$REPO/.venv"

echo "[wrapper] host=$(hostname) date=$(date -u +%FT%TZ)"
echo "[wrapper] whoami=$(whoami) pwd=$(pwd)"
echo "[wrapper] args: $*"

# --- environment ----------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO"

echo "[wrapper] python=$(command -v python)"
python -c "import sys; print('[wrapper] sys.executable=', sys.executable)"

# Sanity: the differentiable sim binaries + data must be reachable from here.
REVERSE_BIN=/eos/user/j/jeffkrup/agentic/hepemshow/build_reverse/HepEmShow
FORWARD_BIN=/eos/user/j/jeffkrup/agentic/hepemshow/build/HepEmShow
HEPEM_DATA=/eos/user/j/jeffkrup/agentic/hepemshow/data/hepem_data.json
for f in "$REVERSE_BIN" "$FORWARD_BIN" "$HEPEM_DATA"; do
    if [[ ! -e "$f" ]]; then
        echo "[wrapper][FATAL] missing required path: $f" >&2
        exit 42
    fi
done
echo "[wrapper] binaries + data present"

# --- run ------------------------------------------------------------------
echo "[wrapper] launching: python -u -m experiments.e2_discriminate_containment $*"
exec python -u -m experiments.e2_discriminate_containment "$@"
