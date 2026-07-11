from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psutil
import pytest

from stage1_gapvalue240.errors import ExternalCommandError
from stage1_gapvalue240.subprocesses import run_logged


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not psutil.pid_exists(pid):
            return True
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def test_run_logged_writes_pass_result(tmp_path):
    log = tmp_path / "ok.log"
    result = run_logged([sys.executable, "-c", "print('ok')"], tmp_path, log, timeout=10)
    persisted = json.loads((tmp_path / "ok.log.result.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert persisted == result


def test_run_logged_timeout_terminates_complete_process_tree(tmp_path):
    pid_file = tmp_path / "child.pid"
    helper = tmp_path / "parent.py"
    helper.write_text(
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid),encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    log = tmp_path / "timeout.log"

    with pytest.raises(ExternalCommandError, match="timed out"):
        run_logged([sys.executable, str(helper), str(pid_file)], tmp_path, log, timeout=1)

    assert pid_file.exists()
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    assert _gone(child_pid), f"child process survived timeout: {child_pid}"
    persisted = json.loads((tmp_path / "timeout.log.result.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "TIMEOUT"
    assert persisted["timed_out"] is True
    terminated = set(persisted["termination"]["terminated_pids"] + persisted["termination"]["killed_pids"])
    assert child_pid in terminated


def test_run_logged_nonzero_exit_has_machine_readable_result(tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(ExternalCommandError, match="failed \(7\)"):
        run_logged([sys.executable, "-c", "raise SystemExit(7)"], tmp_path, log, timeout=10)
    result = json.loads((tmp_path / "failed.log.result.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert result["returncode"] == 7
    assert result["timed_out"] is False
