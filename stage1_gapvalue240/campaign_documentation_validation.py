"""Validate UTF-8 campaign handoff documentation and required operational topics."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Iterable, Any
from .errors import ValidationError
from .util import atomic_write_json, sha256_file

SCHEMA="stage1.documentation_handoff_validation.v1"
REQUIRED_TOKENS=(
    "控制面",
    "单物理 job",
    "assignment",
    "coordination root",
    "lease",
    "fencing",
    "controller",
    "full-block",
    "engineering gate",
    "十机",
    "旧",
    "v2",
)

def validate_documentation_handoff(
    document_paths: Iterable[str|Path], *, output_path: str|Path
) -> dict[str, Any]:
    paths=[Path(path).resolve() for path in document_paths]
    issues=[]
    rows=[]
    combined=[]
    for path in paths:
        if not path.is_file():
            issues.append(f"missing documentation: {path}")
            continue
        raw=path.read_bytes()
        try:
            text=raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            issues.append(f"documentation is not UTF-8: {path}: {exc}")
            continue
        if '\ufffd' in text:
            issues.append(f"documentation contains replacement characters: {path}")
        combined.append(text.lower())
        rows.append({'path':str(path),'size_bytes':len(raw),'sha256':sha256_file(path)})
    corpus='\n'.join(combined)
    missing_tokens=[token for token in REQUIRED_TOKENS if token.lower() not in corpus]
    if missing_tokens:
        issues.append(f"documentation topics missing: {missing_tokens}")
    report={
        'schema_version':SCHEMA,
        'status':'PASS' if not issues else 'FAIL',
        'created_at_unix':time.time(),
        'issues':issues,
        'documents':rows,
        'document_count':len(rows),
        'utf8_valid':not any('UTF-8' in issue or 'replacement' in issue for issue in issues),
        'required_topics':list(REQUIRED_TOKENS),
        'missing_topics':missing_tokens,
    }
    atomic_write_json(output_path,report,overwrite=True)
    if issues:
        raise ValidationError(f"documentation handoff validation failed; see {output_path}")
    return report
