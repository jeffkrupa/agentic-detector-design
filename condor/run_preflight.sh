#!/bin/bash
# HTCondor worker wrapper for the WAVE 15 Fisher-info optimization pre-flight.
#
# Mirrors run_perlayer_adfd.sh: sources the project venv by absolute path, cd's
# to the EOS repo (config.yaml + relative imports), exec's the experiment module
# with all condor arguments forwarded verbatim. One job = one (profile, role,
# seed) unit of fidelity/preflight.py, idempotent via per-unit JSONL + dump
# sidecar existence checks.
#
# Canonical AFS copy (the .sub executable) lives at
#   /afs/cern.ch/user/j/jekrupa/condor_bin/run_preflight.sh
# Re-copy this file there after editing.
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

# Sanity: the gov forward sim binary + data must be reachable from here.
GOV_BIN=/eos/user/j/jeffkrup/agentic/hepemshow-gov/build_gov_fwd/HepEmShow
HEPEM_DATA=/eos/user/j/jeffkrup/agentic/hepemshow/data/hepem_data.json
for f in "$GOV_BIN" "$HEPEM_DATA"; do
    if [[ ! -e "$f" ]]; then
        echo "[wrapper][FATAL] missing required path: $f" >&2
        exit 42
    fi
done
echo "[wrapper] gov binary + data present"

# --- run ------------------------------------------------------------------
echo "[wrapper] launching: python -u -m fidelity.preflight $*"
exec python -u -m fidelity.preflight "$@"
