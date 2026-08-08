"""JSON-schema validation for versioned dynamic-campaign reports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator
from .errors import ValidationError


def validate_report_schema(report_path: str | Path, schema_path: str | Path) -> dict[str, Any]:
    report_file = Path(report_path).resolve()
    schema_file = Path(schema_path).resolve()
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"unreadable report/schema: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.path))
    if errors:
        detail = [f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors]
        raise ValidationError(f"JSON schema validation failed: {detail}")
    return report
