from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_python_compatibility_audit.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_python_compatibility_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_log(root: Path, relative: str, content: str) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    path.write_bytes(payload)
    return {
        "stdout_path": relative,
        "stdout_bytes": len(payload),
        "stdout_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "stderr_path": relative.replace("stdout", "stderr"),
        "stderr_bytes": 0,
        "stderr_sha256": hashlib.sha256(b"").hexdigest().upper(),
        "exit_code": 0,
    }


def _fixture(root: Path) -> dict:
    commands = []
    for version, patch in (("3.11", "14"), ("3.12", "12")):
        test = _write_log(root, f"commands/full_v4_python{version.replace('.', '')}_isolated/stdout.log", "340 passed in 1.00s\n")
        probe = _write_log(
            root,
            f"commands/runtime_probe_python{version.replace('.', '')}_isolated/stdout.log",
            json.dumps(
                {
                    "python": f"{version}.{patch}",
                    "pyarrow": "21.0.0",
                    "torch": "2.11.0+cu128",
                    "zstd_codec_available": True,
                }
            )
            + "\n",
        )
        compileall = _write_log(root, f"commands/compileall_python{version.replace('.', '')}_isolated/stdout.log", "")
        for record in (test, probe, compileall):
            stderr = root / record["stderr_path"]
            stderr.parent.mkdir(parents=True, exist_ok=True)
            stderr.write_bytes(b"")
        test.update(name=f"full_v4_python{version.replace('.', '')}_isolated", argv=["uv", "run", "--isolated", "--python", version, "--extra", "dev", "pytest"])
        probe.update(name=f"runtime_probe_python{version.replace('.', '')}_isolated", argv=["uv", "run", "--isolated", "--python", version, "--extra", "dev", "python"])
        compileall.update(name=f"compileall_python{version.replace('.', '')}_isolated", argv=["uv", "run", "--isolated", "--python", version, "--extra", "dev", "python", "-m", "compileall"])
        commands.extend((test, probe, compileall))
    return {"commands": commands}


def test_dual_python_audit_requires_tests_probe_compileall_and_zstd(tmp_path):
    module = _load_module()

    report = module.build_report(tmp_path, _fixture(tmp_path), required_passed=340)

    assert report["status"] == "PASS"
    assert set(report["versions"]) == {"3.11", "3.12"}
    assert all(row["test_summary"]["passed"] == 340 for row in report["versions"].values())
    assert all(row["runtime_probe"]["zstd_codec_available"] is True for row in report["versions"].values())


def test_log_hash_drift_fails_closed(tmp_path):
    module = _load_module()
    index = _fixture(tmp_path)
    (tmp_path / "commands/full_v4_python311_isolated/stdout.log").write_text("tampered\n", encoding="utf-8")

    report = module.build_report(tmp_path, index, required_passed=340)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "LOG_IDENTITY_MISMATCH" for error in report["errors"])


def test_wrong_python_or_missing_zstd_fails_closed(tmp_path):
    module = _load_module()
    index = _fixture(tmp_path)
    probe = next(row for row in index["commands"] if row["name"] == "runtime_probe_python312_isolated")
    path = tmp_path / probe["stdout_path"]
    payload = json.dumps({"python": "3.13.0", "pyarrow": "21.0.0", "torch": "x", "zstd_codec_available": False}) + "\n"
    raw = payload.encode("utf-8")
    path.write_bytes(raw)
    probe["stdout_bytes"] = len(raw)
    probe["stdout_sha256"] = hashlib.sha256(raw).hexdigest().upper()

    report = module.build_report(tmp_path, index, required_passed=340)

    assert report["status"] == "FAIL"
    codes = {error["code"] for error in report["errors"]}
    assert "PYTHON_VERSION_MISMATCH" in codes
    assert "ZSTD_UNAVAILABLE" in codes
