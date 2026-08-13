from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


VERSIONS = ("3.11", "3.12")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _error(errors: list[dict], code: str, *, command: str, detail: str) -> None:
    errors.append({"code": code, "command": command, "detail": detail})


def _verify_log(root: Path, receipt: dict, stream: str, errors: list[dict]) -> bytes | None:
    relative = receipt.get(f"{stream}_path")
    if not isinstance(relative, str):
        _error(errors, "RECEIPT_FIELD_MISSING", command=receipt.get("name", "UNKNOWN"), detail=f"{stream}_path")
        return None
    path = root / relative
    if not path.is_file():
        _error(errors, "LOG_MISSING", command=receipt.get("name", "UNKNOWN"), detail=relative)
        return None
    payload = path.read_bytes()
    expected_bytes = receipt.get(f"{stream}_bytes")
    expected_sha = str(receipt.get(f"{stream}_sha256", "")).upper()
    observed_sha = _sha256(payload)
    if len(payload) != expected_bytes or observed_sha != expected_sha:
        _error(
            errors,
            "LOG_IDENTITY_MISMATCH",
            command=receipt.get("name", "UNKNOWN"),
            detail=(
                f"{relative}: expected bytes={expected_bytes} sha={expected_sha}; "
                f"observed bytes={len(payload)} sha={observed_sha}"
            ),
        )
        return None
    return payload


def _command_has_isolated_version(argv: list[str], version: str) -> bool:
    try:
        python_index = argv.index("--python")
        extra_index = argv.index("--extra")
    except ValueError:
        return False
    return (
        "--isolated" in argv
        and python_index + 1 < len(argv)
        and argv[python_index + 1] == version
        and extra_index + 1 < len(argv)
        and argv[extra_index + 1] == "dev"
    )


def _parse_test_summary(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for count, kind in re.findall(r"(\d+)\s+(passed|failed|skipped|error|errors)", output):
        key = "errors" if kind in {"error", "errors"} else kind
        counts[key] = int(count)
    return counts


def _parse_probe(output: str) -> dict | None:
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def build_report(
    evidence_root: Path,
    command_index: dict,
    *,
    required_passed: int,
    expected_implementation_source_commit: str | None = None,
) -> dict:
    root = evidence_root.resolve()
    rows = command_index.get("commands", [])
    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    errors: list[dict] = []
    versions: dict[str, dict] = {}
    for version in VERSIONS:
        compact = version.replace(".", "")
        names = {
            "tests": f"full_v4_python{compact}_isolated",
            "probe": f"runtime_probe_python{compact}_isolated",
            "compileall": f"compileall_python{compact}_isolated",
        }
        version_record: dict = {"commands": names}
        for role, name in names.items():
            receipt = by_name.get(name)
            if receipt is None:
                _error(errors, "COMMAND_RECEIPT_MISSING", command=name, detail=role)
                continue
            argv = receipt.get("argv", [])
            if not isinstance(argv, list) or not _command_has_isolated_version(argv, version):
                _error(errors, "COMMAND_NOT_ISOLATED_OR_VERSION_BOUND", command=name, detail=str(argv))
            if receipt.get("exit_code") != 0:
                _error(errors, "COMMAND_EXIT_NONZERO", command=name, detail=str(receipt.get("exit_code")))
            if (
                expected_implementation_source_commit is not None
                and receipt.get("implementation_source_commit") != expected_implementation_source_commit
            ):
                _error(
                    errors,
                    "IMPLEMENTATION_SOURCE_MISMATCH",
                    command=name,
                    detail=str(receipt.get("implementation_source_commit")),
                )
            stdout = _verify_log(root, receipt, "stdout", errors)
            _verify_log(root, receipt, "stderr", errors)
            if stdout is None:
                continue
            decoded = stdout.decode("utf-8", errors="strict")
            if role == "tests":
                summary = _parse_test_summary(decoded)
                version_record["test_summary"] = summary
                if summary["passed"] < required_passed:
                    _error(errors, "V4_PASS_COUNT_TOO_LOW", command=name, detail=str(summary))
                if summary["failed"] or summary["errors"] or summary["skipped"]:
                    _error(errors, "V4_SUITE_NOT_ALL_GREEN", command=name, detail=str(summary))
            elif role == "probe":
                probe = _parse_probe(decoded)
                version_record["runtime_probe"] = probe
                if probe is None:
                    _error(errors, "RUNTIME_PROBE_INVALID", command=name, detail=decoded)
                else:
                    if not str(probe.get("python", "")).startswith(f"{version}."):
                        _error(errors, "PYTHON_VERSION_MISMATCH", command=name, detail=str(probe.get("python")))
                    if probe.get("zstd_codec_available") is not True:
                        _error(errors, "ZSTD_UNAVAILABLE", command=name, detail=str(probe))
                    if not probe.get("pyarrow"):
                        _error(errors, "PYARROW_VERSION_MISSING", command=name, detail=str(probe))
        versions[version] = version_record
    return {
        "schema_version": "stage1.sctsr.python_compatibility_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "required_passed_per_version": required_passed,
        "required_versions": list(VERSIONS),
        "expected_implementation_source_commit": expected_implementation_source_commit,
        "versions": versions,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate isolated Python 3.11/3.12 SCTSR evidence")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--command-index", type=Path, required=True)
    parser.add_argument("--required-passed", type=int, default=340)
    parser.add_argument("--implementation-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.command_index.read_text(encoding="utf-8"))
    report = build_report(
        args.evidence_root,
        index,
        required_passed=args.required_passed,
        expected_implementation_source_commit=args.implementation_source_commit,
    )
    _atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "versions": {
                    key: value.get("test_summary", {}) for key, value in report["versions"].items()
                },
                "error_count": len(report["errors"]),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
