#!/usr/bin/env bash
set -euo pipefail
RUN_NUMBER="$1"; MACHINE_CONFIG="$2"; ACTION="${3:-run}"; ATTEMPT_ID="${4:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT=$(printf "%s/runs/run_%03d.py" "$SCRIPT_DIR" "$RUN_NUMBER")
ARGS=("$SCRIPT" --machine-config "$MACHINE_CONFIG" --action "$ACTION")
if [[ -n "$ATTEMPT_ID" ]]; then ARGS+=(--attempt-id "$ATTEMPT_ID"); fi
python "${ARGS[@]}"
