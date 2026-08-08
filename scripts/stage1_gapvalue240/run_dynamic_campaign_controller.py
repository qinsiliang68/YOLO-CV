from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import pandas as pd
import psutil

from stage1_gapvalue240.aiops import EXIT_RETRYABLE, EXIT_SUCCESS, EXIT_TERMINAL
from stage1_gapvalue240.campaign_controller import (
    CampaignControllerError,
    load_campaign_release,
    plan_controller_iteration,
    run_worker_process,
)
from stage1_gapvalue240.campaign_assignment import load_campaign_assignment
from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID, active_run_queue_dir
from stage1_gapvalue240.locks import RunLock
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.util import atomic_write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one release-gated dynamic campaign machine queue.")
    parser.add_argument("--machine-config", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument(
        "--campaign-root",
        default=str(
            _BootstrapPath(__file__).resolve().parents[2]
            / "artifacts/stage1_sample_value_experiments/experiments"
            / CAMPAIGN_ID
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=300.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="Dispatch at most one physical job.")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--allow-dirty-code",
        action="store_true",
        help="Development smoke only; never use for a formal release.",
    )
    return parser.parse_args(argv)


def _state_path(campaign_output: Path, job_id: str) -> Path:
    return campaign_output / "09_aiops/job_states" / f"{job_id}.json"


def _expected_worker_is_alive(payload: dict, job_id: str) -> bool:
    try:
        process = psutil.Process(int(payload.get("pid")))
        command = " ".join(process.cmdline())
        return "dynamic_campaign_train_worker.py" in command and job_id in command
    except (TypeError, ValueError, psutil.Error):
        return False


def _load_job_states(campaign_output: Path, job_ids: list[str]) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for job_id in job_ids:
        path = _state_path(campaign_output, job_id)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CampaignControllerError(f"job state is unreadable: {path}") from exc
        if payload.get("job_id") != job_id:
            raise CampaignControllerError(f"job state identity mismatch: {path}")
        if payload.get("state") == "RUNNING" and not _expected_worker_is_alive(payload, job_id):
            payload.update(
                {
                    "state": "FAILED",
                    "retryable": True,
                    "error": "STALE_RUNNING_STATE: registered worker process is not alive",
                    "pid": None,
                    "updated_at_unix": time.time(),
                }
            )
            atomic_write_json(path, payload, overwrite=True)
        states[job_id] = payload
    return states


def _write_status(
    path: Path,
    *,
    machine_id: str,
    release,
    assignment,
    plan,
    current_job_id: str | None,
    worker_pid: int | None,
    dispatched_jobs: int,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": "stage1.dynamic_campaign_controller.v1",
            "campaign_id": CAMPAIGN_ID,
            "machine_id": machine_id,
            "controller_pid": os.getpid(),
            "release_id": release.release_id,
            "release_sha256": release.sha256,
            "release_scope": release.scope,
            "assignment_id": assignment.assignment_id,
            "assignment_sha256": assignment.sha256,
            "overall_state": plan.overall_state,
            "current_job_id": current_job_id,
            "worker_pid": worker_pid,
            "next_job_id": plan.next_job_id,
            "dispatched_jobs_this_session": dispatched_jobs,
            "job_state_counts": plan.counts,
            "updated_at_unix": time.time(),
        },
        overwrite=True,
    )


def _record_abnormal_worker_exit(
    campaign_output: Path,
    row: pd.Series,
    returncode: int,
    *,
    machine_id: str,
    assignment,
) -> None:
    job_id = str(row.job_id)
    path = _state_path(campaign_output, job_id)
    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if payload.get("state") in {"COMPLETE", "FAILED"}:
        return
    normalized = returncode if returncode in {EXIT_RETRYABLE, EXIT_TERMINAL} else EXIT_RETRYABLE
    atomic_write_json(
        path,
        {
            "schema_version": "stage1.dynamic_campaign_job_state.v1",
            "campaign_id": CAMPAIGN_ID,
            "job_id": job_id,
            "machine_id": machine_id,
            "planned_machine_slot": str(row.machine_id),
            "assignment_id": assignment.assignment_id,
            "assignment_sha256": assignment.sha256,
            "logical_run_id": str(row.logical_run_id),
            "logical_arm_id": str(row.logical_arm_id),
            "state": "FAILED",
            "action": payload.get("action"),
            "completed_epoch": payload.get("completed_epoch"),
            "attempt_count": max(1, int(payload.get("attempt_count", 0))),
            "pid": None,
            "updated_at_unix": time.time(),
            "retryable": normalized == EXIT_RETRYABLE,
            "error": f"WORKER_PROCESS_EXIT_{returncode}",
        },
        overwrite=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_attempts <= 0 or args.retry_delay_seconds < 0 or args.heartbeat_seconds <= 0:
        raise CampaignControllerError("invalid controller retry/heartbeat policy")
    if args.max_jobs is not None and args.max_jobs <= 0:
        raise CampaignControllerError("--max-jobs must be positive")
    machine_config = Path(args.machine_config).resolve()
    machine = load_machine_config(machine_config)
    if bool(machine.data.get("dry_run", False)):
        raise CampaignControllerError("formal dynamic campaign controller refuses dry_run configs")
    machine_id = str(machine.data["machine_id"])
    campaign_root = Path(args.campaign_root).resolve()
    queue_dir = active_run_queue_dir(campaign_root)
    release = load_campaign_release(
        queue_dir,
        args.release,
        expected_campaign_id=CAMPAIGN_ID,
    )
    assignment = load_campaign_assignment(
        queue_dir,
        args.release,
        args.assignment,
        expected_campaign_id=CAMPAIGN_ID,
        expected_machine_id=machine_id,
        repo_root=machine.path_value("repo_root"),
    )
    registry = pd.read_csv(queue_dir / "JOB_EXECUTION_REGISTRY.csv", keep_default_na=False)
    queue_validation = json.loads(
        (queue_dir / "RUN_QUEUE_VALIDATION.json").read_text(encoding="utf-8")
    )
    canonical_lock_sha = str(queue_validation.get("canonical_lock_file_sha256", "")).upper()
    if len(canonical_lock_sha) != 64:
        raise CampaignControllerError("campaign queue has no canonical lock SHA-256")
    assigned_job_ids = set(
        assignment.rows.loc[
            assignment.rows.assigned_machine_id.astype(str).eq(machine_id), "job_id"
        ].astype(str)
    )
    jobs = registry.loc[
        registry.job_id.astype(str).isin(release.job_ids)
        & registry.job_id.astype(str).isin(assigned_job_ids)
    ].sort_values("queue_order", kind="stable")
    campaign_output = (machine.path_value("output_root") / CAMPAIGN_ID).resolve()
    controller_dir = campaign_output / f"09_aiops/controllers/{machine_id}"
    controller_dir.mkdir(parents=True, exist_ok=True)
    status_path = controller_dir / "status.json"
    if jobs.empty:
        atomic_write_json(
            status_path,
            {
                "schema_version": "stage1.dynamic_campaign_controller.v1",
                "campaign_id": CAMPAIGN_ID,
                "machine_id": machine_id,
                "release_id": release.release_id,
                "release_sha256": release.sha256,
                "assignment_id": assignment.assignment_id,
                "assignment_sha256": assignment.sha256,
                "overall_state": "NO_RELEASED_JOBS",
                "updated_at_unix": time.time(),
            },
            overwrite=True,
        )
        return EXIT_SUCCESS

    local_job_ids = jobs.job_id.astype(str).tolist()
    local_set = set(local_job_ids)
    cross_machine_dependencies = sorted(
        {
            str(value)
            for value in jobs.dependency_job_id.astype(str)
            if value and value not in local_set
        }
    )
    if cross_machine_dependencies:
        raise CampaignControllerError(
            f"machine release has non-local dependencies: {cross_machine_dependencies}"
        )

    lock = RunLock(
        campaign_output / f"09_aiops/locks/controller_{machine_id}.lock",
        {"campaign_id": CAMPAIGN_ID, "machine_id": machine_id, "role": "campaign_controller"},
        reclaim_dead_local=True,
    )
    dispatched = 0
    with lock:
        while True:
            states = _load_job_states(campaign_output, local_job_ids)
            plan = plan_controller_iteration(
                jobs,
                states,
                max_attempts=args.max_attempts,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            _write_status(
                status_path,
                machine_id=machine_id,
                release=release,
                assignment=assignment,
                plan=plan,
                current_job_id=None,
                worker_pid=None,
                dispatched_jobs=dispatched,
            )
            if args.plan_only:
                print(json.dumps({"machine_id": machine_id, "plan": plan.__dict__}, ensure_ascii=False, indent=2))
                return EXIT_SUCCESS
            if plan.next_job_id is None:
                if plan.overall_state == "WAITING_RETRY":
                    time.sleep(min(args.heartbeat_seconds, max(1.0, args.retry_delay_seconds)))
                    continue
                if plan.overall_state == "RUNNING_EXTERNAL":
                    time.sleep(args.heartbeat_seconds)
                    continue
                return EXIT_SUCCESS if plan.overall_state == "COMPLETE" else EXIT_TERMINAL

            job_id = plan.next_job_id
            row = jobs.loc[jobs.job_id.astype(str).eq(job_id)].iloc[0]
            dispatch_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
            log_path = controller_dir / "logs" / f"{job_id}_{dispatch_id}.log"
            worker_script = Path(__file__).resolve().parent / "dynamic_campaign_train_worker.py"
            command = [
                str(machine.data.get("python_executable") or sys.executable),
                str(worker_script),
                "--machine-config",
                str(machine_config),
                "--campaign-root",
                str(campaign_root),
                "--job-id",
                job_id,
                "--release",
                str(Path(args.release).resolve()),
                "--assignment",
                str(Path(args.assignment).resolve()),
                "--expected-release-id",
                release.release_id,
                "--expected-canonical-lock-sha256",
                canonical_lock_sha,
            ]
            if args.allow_dirty_code:
                command.append("--allow-dirty-code")

            def heartbeat(worker_pid: int) -> None:
                _write_status(
                    status_path,
                    machine_id=machine_id,
                    release=release,
                    assignment=assignment,
                    plan=plan,
                    current_job_id=job_id,
                    worker_pid=worker_pid,
                    dispatched_jobs=dispatched,
                )

            result = run_worker_process(
                command,
                cwd=machine.path_value("repo_root"),
                log_path=log_path,
                heartbeat=heartbeat,
                poll_seconds=args.heartbeat_seconds,
            )
            dispatched += 1
            if int(result["returncode"]) != EXIT_SUCCESS:
                _record_abnormal_worker_exit(
                    campaign_output,
                    row,
                    int(result["returncode"]),
                    machine_id=machine_id,
                    assignment=assignment,
                )
            if args.once or (args.max_jobs is not None and dispatched >= args.max_jobs):
                return EXIT_SUCCESS if int(result["returncode"]) == EXIT_SUCCESS else int(result["returncode"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "exit_code": EXIT_TERMINAL,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(EXIT_TERMINAL)
