from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stage1_gapvalue240.run_engine import prepare_run, train_run, evaluate_run, validate_run, run_all, run_entry_cli

RUN_SLOT = "RUN_154"

def prepare(machine_config: str | Path, attempt_id: str | None = None, *, allow_new_attempt_after_validated: bool = False):
    return prepare_run(RUN_SLOT, machine_config, attempt_id, allow_new_attempt_after_validated)

def train(machine_config: str | Path, attempt_id: str):
    return train_run(RUN_SLOT, machine_config, attempt_id)

def evaluate(machine_config: str | Path, attempt_id: str):
    return evaluate_run(RUN_SLOT, machine_config, attempt_id)

def validate(machine_config: str | Path, attempt_id: str):
    return validate_run(RUN_SLOT, machine_config, attempt_id)

def run(machine_config: str | Path, attempt_id: str | None = None, *, allow_new_attempt_after_validated: bool = False):
    return run_all(RUN_SLOT, machine_config, attempt_id, allow_new_attempt_after_validated)

def main(argv=None):
    return run_entry_cli(RUN_SLOT, argv)

if __name__ == "__main__":
    raise SystemExit(main())
