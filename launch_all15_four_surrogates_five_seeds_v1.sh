#!/usr/bin/env bash
# Resource-contained campaign launcher. --preflight performs no evaluations.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
UNIT="agentic-all15-four-surrogates-five-seeds-v1"
PYTHON="$ROOT/.venv/bin/python"
RUNNER="$ROOT/run_all15_four_surrogates_five_seeds_reproduction_v1.py"
EXECUTOR="$ROOT/execute_frozen_python.py"
LOG="$ROOT/runs/reproduction_all15_four_surrogates_five_seeds_v1.log"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing project venv: %s (install requirements first)\n' "$PYTHON" >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--preflight' ) ]]; then
  printf 'usage: %s [--preflight]\n' "$0" >&2
  exit 2
fi
cd -- "$ROOT"
"$PYTHON" "$EXECUTOR" "$RUNNER" --preflight
if [[ ${1:-} == '--preflight' ]]; then
  exit 0
fi
if ! command -v systemd-run >/dev/null 2>&1; then
  printf 'systemd-run is required for resource-contained launch\n' >&2
  exit 2
fi
if [[ "$(systemctl --user is-active "$UNIT.service" 2>/dev/null || true)" == 'active' ]]; then
  printf '%s is already active\n' "$UNIT.service" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$LOG")"
exec systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --property=Type=exec \
  --property=MemoryHigh=16G \
  --property=MemoryMax=24G \
  --property=MemorySwapMax=4G \
  --property=TasksMax=128 \
  --property=CPUQuota=1600% \
  --property=OOMPolicy=kill \
  --property=RuntimeMaxSec=7d \
  --property="StandardOutput=append:$LOG" \
  --property="StandardError=append:$LOG" \
  --working-directory="$ROOT" \
  --setenv=PYTHONUNBUFFERED=1 \
  "$PYTHON" "$EXECUTOR" "$RUNNER"
