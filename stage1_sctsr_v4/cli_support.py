from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import atomic_write_json, stable_digest


def add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="Explicit JSON receipt path")


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution-mode", choices=("synthetic", "formal"), default="synthetic")
    parser.add_argument("--release-authorization", type=Path, default=None)
    parser.add_argument("--release-trust-policy", type=Path, default=None)
    parser.add_argument("--execution-token", type=Path, default=None)
    parser.add_argument("--execution-claim-root", type=Path, default=None)
    parser.add_argument("--source-tree-manifest", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--arms", type=Path, default=None)
    parser.add_argument("--asset-registry", type=Path, default=None)
    parser.add_argument("--runtime-config", type=Path, default=None)
    parser.add_argument("--seed-registry", type=Path, default=None)
    parser.add_argument("--run-intent-acknowledgement", type=Path, default=None)
    parser.add_argument("--runbook-manifest", type=Path, default=None)


def require_receipt_outside_artifact_root(receipt_path: str | Path, artifact_root: str | Path) -> None:
    receipt = Path(receipt_path).resolve()
    root = Path(artifact_root).resolve()
    try:
        receipt.relative_to(root)
    except ValueError:
        return
    raise SctsrError(
        ErrorCode.ARTIFACT_VALIDATION_FAILED,
        "CLI receipt must remain outside the immutable artifact root it validates or creates",
        artifact_path=str(receipt),
        required_action="Choose a sibling receipt path so the run artifact index remains exhaustive.",
    )


def receipt_base(command: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": "stage1.sctsr.cli_receipt.v1",
        "command": command,
        "cwd": Path.cwd().as_posix(),
        "python": sys.version,
        "pid": os.getpid(),
        **extra,
    }


def error_receipt(command: str, exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, SctsrError):
        detail = exc.to_dict()
    else:
        detail = {
            "code": ErrorCode.INTERNAL_ERROR.value,
            "message": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "recoverable": False,
            "required_action": "Inspect the traceback and correct the implementation or invocation.",
        }
    result = receipt_base(command, status="FAIL", error=detail)
    result["receipt_digest"] = stable_digest(result)
    return result


def success_receipt(command: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = receipt_base(command, status="PASS", result=dict(payload or {}))
    result["receipt_digest"] = stable_digest(result)
    return result


def run_cli(command: str, output: Path, action: Callable[[], Mapping[str, Any] | None]) -> int:
    try:
        payload = action()
        receipt = success_receipt(command, payload)
        code = 0
    except BaseException as exc:  # CLI boundary must turn all failures into a stable receipt.
        receipt = error_receipt(command, exc)
        code = 2 if isinstance(exc, SctsrError) else 1
    atomic_write_json(output, receipt)
    print(output.as_posix())
    return code
