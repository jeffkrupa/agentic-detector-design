#!/bin/bash
# HTCondor worker wrapper for the wave-2 knob validation (fidelity loop).
#
# Mirrors run_wave1_validation.sh: sources the project venv by absolute path,
# cd's to the EOS repo (config.yaml + relative imports), exec's the runner
# with all condor arguments forwarded verbatim. One job = one (config, seed)
# unit of fidelity/wave2_validation.py, idempotent via the sim cache and
# per-unit JSONL files.
#
# Canonical AFS copy (the .sub executable) lives at
#   /afs/cern.ch/user/j/jekrupa/condor_bin/run_wave2_validation.sh
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

# Sanity: the wave-2 KNOB binary (build_agent_fwd, branch
# knob/prefix-anchor @ 61c7bb5) + data must be reachable from here.
KNOB_BIN=/eos/user/j/jeffkrup/agentic/hepemshow/build_agent_fwd/HepEmShow
HEPEM_DATA=/eos/user/j/jeffkrup/agentic/hepemshow/data/hepem_data.json
for f in "$KNOB_BIN" "$HEPEM_DATA"; do
    if [[ ! -e "$f" ]]; then
        echo "[wrapper][FATAL] missing required path: $f" >&2
        exit 42
    fi
done
echo "[wrapper] knob binary + data present"

# --- run ------------------------------------------------------------------
echo "[wrapper] launching: python -u -m fidelity.wave2_validation $*"
exec python -u -m fidelity.wave2_validation "$@"
