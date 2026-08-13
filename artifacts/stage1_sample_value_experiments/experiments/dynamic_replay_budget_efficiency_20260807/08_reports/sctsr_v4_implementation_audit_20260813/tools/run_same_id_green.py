from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence


HEX64 = re.compile(r"^[0-9A-F]{64}$")


class SameIdGreenError(ValueError):
    """Raised before publication when a historical-test replay is invalid."""


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest().upper()


def _inside(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise SameIdGreenError("decoded source path must be relative")
    target = (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SameIdGreenError("decoded source path escapes the materialization root") from exc
    return target


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": _sha(data),
    }


def materialize_and_run(
    *,
    sources: Sequence[Mapping[str, Any]],
    test_ids: Sequence[str],
    repository_root: str | Path,
    output_root: str | Path,
    compatibility_plugin_path: str | Path | None = None,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise SameIdGreenError("same-ID green output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    materialized = output / "materialized"
    materialized.mkdir()
    if not sources:
        raise SameIdGreenError("same-ID green run has no historical sources")
    source_records = []
    for index, row in enumerate(sources):
        if not isinstance(row, Mapping) or set(row) != {"encoded_path", "decoded_relative_path", "decoded_sha256"}:
            raise SameIdGreenError(f"historical source row {index} schema is invalid")
        encoded = Path(row["encoded_path"]).resolve()
        expected = str(row["decoded_sha256"])
        if not encoded.is_file() or not HEX64.fullmatch(expected):
            raise SameIdGreenError("historical source path or decoded source SHA is invalid")
        try:
            compact = "".join(encoded.read_text(encoding="ascii").split())
            decoded = base64.b64decode(compact, validate=True)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise SameIdGreenError("historical source base64 is invalid") from exc
        if _sha(decoded) != expected:
            raise SameIdGreenError("decoded source SHA does not match the registered historical source")
        destination = _inside(materialized, str(row["decoded_relative_path"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(decoded)
        source_records.append(_file_record(destination, materialized))

    plugin_record = None
    if compatibility_plugin_path is not None:
        plugin = Path(compatibility_plugin_path).resolve()
        if not plugin.is_file():
            raise SameIdGreenError("compatibility plugin is missing")
        plugin_bytes = plugin.read_bytes()
        plugin_target = materialized / "conftest.py"
        plugin_target.write_bytes(plugin_bytes)
        plugin_record = _file_record(plugin_target, materialized)

    exact_ids = tuple(test_ids)
    if not exact_ids or any(not isinstance(test_id, str) or "::" not in test_id for test_id in exact_ids):
        raise SameIdGreenError("same-ID green run requires exact pytest node IDs")
    if len(set(exact_ids)) != len(exact_ids):
        raise SameIdGreenError("same-ID green run repeats a pytest node ID")
    junit = output / "GREEN.junit.xml"
    command = [sys.executable, "-m", "pytest", *exact_ids, "-q", f"--junitxml={junit}"]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repository) if not existing else str(repository) + os.pathsep + existing
    environment["SCTSR_AUDIT_REPOSITORY_ROOT"] = str(repository)
    completed = subprocess.run(
        command,
        cwd=materialized,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = output / "GREEN.stdout.log"
    stderr_path = output / "GREEN.stderr.log"
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    result = {
        "schema_version": "stage1.sctsr.same_id_green_receipt.v1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "exit_code": completed.returncode,
        "command": command,
        "repository_root": repository.as_posix(),
        "test_ids": list(exact_ids),
        "source_records": source_records,
        "compatibility_plugin": plugin_record,
        "stdout": _file_record(stdout_path, output),
        "stderr": _file_record(stderr_path, output),
        "junit": _file_record(junit, output) if junit.is_file() else None,
    }
    receipt_path = output / "GREEN_RECEIPT.json"
    receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        raise SameIdGreenError(f"same-ID green pytest run failed with exit {completed.returncode}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode SHA-bound historical red tests and rerun their exact node IDs green")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compatibility-plugin", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = []
    for row in manifest["sources"]:
        source = dict(row)
        encoded = Path(source["encoded_path"])
        if not encoded.is_absolute():
            source["encoded_path"] = manifest_path.parent / encoded
        sources.append(source)
    result = materialize_and_run(
        sources=sources,
        test_ids=manifest["test_ids"],
        repository_root=args.repository_root,
        output_root=args.output_root,
        compatibility_plugin_path=args.compatibility_plugin,
    )
    print(json.dumps({key: result[key] for key in ("status", "exit_code", "test_ids")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
