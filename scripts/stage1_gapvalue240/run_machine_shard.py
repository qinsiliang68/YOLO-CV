from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from stage1_gapvalue240.aiops import EXIT_RETRYABLE, EXIT_SUCCESS, EXIT_TERMINAL, exit_code_for_exception
from stage1_gapvalue240.errors import ExternalCommandError
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.locks import RunLock
from stage1_gapvalue240.runtime_contract import load_runtime_contract, validate_runtime_links
from stage1_gapvalue240.subprocesses import run_logged
from stage1_gapvalue240.util import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen machine shard through isolated run-entry subprocesses.")
    parser.add_argument("--machine-config", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-runs", type=int)
    return parser


def _ordered_slots(machine) -> list[str]:
    repo = machine.path_value("repo_root")
    runtime = load_runtime_contract(repo / "configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml")
    links = validate_runtime_links(runtime, repo)
    machine_id = str(machine.data["machine_id"])
    shard_record = links["queue"]["machine_shards"].get(machine_id)
    if shard_record is None:
        raise RuntimeError(f"Machine ID is not bound by the runtime contract: {machine_id}")
    shard = Path(shard_record["path"]).resolve()
    jobs = pd.read_csv(shard, dtype={"run_slot": "string"})
    return jobs.run_slot.astype(str).tolist()


def _write_controller_status(path: Path, *, machine_id: str, run_slot: str | None,
                             index: int, total: int, state: str, retryable: bool,
                             error_code: str | None = None) -> None:
    atomic_write_json(path, {
        "schema_version": "stage1_gapvalue240.shard_controller.v1",
        "machine_id": machine_id,
        "run_slot": run_slot,
        "phase": "dispatch",
        "pid": os.getpid(),
        "index": index,
        "total": total,
        "state": state,
        "retryable": retryable,
        "error_code": error_code,
        "updated_at_unix": time.time(),
    }, overwrite=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    machine = load_machine_config(args.machine_config)
    slots = _ordered_slots(machine)
    if args.max_runs is not None:
        if args.max_runs < 0:
            raise ValueError("--max-runs must be nonnegative")
        slots = slots[: args.max_runs]
    output = machine.path_value("output_root")
    controller_dir = output / "shard_controller"
    controller_dir.mkdir(parents=True, exist_ok=True)
    controller_lock = RunLock(
        output / "locks/shard_controller.lock",
        {"machine_id": str(machine.data["machine_id"]), "role": "shard_controller"},
        reclaim_dead_local=True,
    )
    controller_lock.acquire()
    atexit.register(controller_lock.release)
    status_path = controller_dir / "status.json"
    machine_id = str(machine.data["machine_id"])
    if not slots:
        _write_controller_status(
            status_path, machine_id=machine_id, run_slot=None, index=0, total=0,
            state="NO_TASKS", retryable=False,
        )
        print(json.dumps({"machine_id": machine_id, "runs": 0, "role": "reserve"}))
        controller_lock.release()
        return EXIT_SUCCESS

    results = []
    aggregate_exit = EXIT_SUCCESS
    for index, slot in enumerate(slots, start=1):
        _write_controller_status(
            status_path, machine_id=machine_id, run_slot=slot, index=index, total=len(slots),
            state="RUNNING", retryable=False,
        )
        entry = Path(__file__).resolve().parent / "runs" / f"run_{int(slot.split('_')[1]):03d}.py"
        command = [
            str(machine.data.get("python_executable") or sys.executable),
            str(entry),
            "--machine-config", str(Path(args.machine_config).resolve()),
            "--action", "run",
        ]
        dispatch_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        log = controller_dir / "logs" / f"{slot}_{dispatch_id}.log"
        try:
            subprocess_result = run_logged(
                command,
                machine.path_value("repo_root"),
                log,
                # Whole-run stall detection belongs to AIOps. Phase workers enforce their own timeout.
                timeout=None,
            )
            return_code = int(subprocess_result["returncode"])
        except ExternalCommandError:
            result_path = log.with_suffix(log.suffix + ".result.json")
            subprocess_result = json.loads(result_path.read_text(encoding="utf-8"))
            return_code = int(subprocess_result.get("returncode") or EXIT_RETRYABLE)
        results.append({"run_slot": slot, "exit_code": return_code, "subprocess": subprocess_result})
        if return_code == EXIT_SUCCESS:
            continue
        if return_code not in {EXIT_RETRYABLE, EXIT_TERMINAL}:
            return_code = EXIT_RETRYABLE
        aggregate_exit = max(aggregate_exit, return_code)
        _write_controller_status(
            status_path, machine_id=machine_id, run_slot=slot, index=index, total=len(slots),
            state="FAILED", retryable=return_code == EXIT_RETRYABLE,
            error_code="RUN_RETRYABLE" if return_code == EXIT_RETRYABLE else "RUN_TERMINAL",
        )
        if not args.continue_on_error:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            controller_lock.release()
            return return_code

    final_state = "COMPLETED" if aggregate_exit == EXIT_SUCCESS else "COMPLETED_WITH_FAILURES"
    _write_controller_status(
        status_path, machine_id=machine_id, run_slot=None, index=len(slots), total=len(slots),
        state=final_state, retryable=aggregate_exit == EXIT_RETRYABLE,
        error_code=None if aggregate_exit == EXIT_SUCCESS else "SHARD_HAS_FAILURES",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    controller_lock.release()
    return aggregate_exit


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        exit_code = exit_code_for_exception(exc)
        print(json.dumps({
            "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
            "retryable": exit_code == EXIT_RETRYABLE, "exit_code": exit_code,
        }, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(exit_code)
