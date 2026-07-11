#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/prepare_experiment.py" --machine-config "$1"
