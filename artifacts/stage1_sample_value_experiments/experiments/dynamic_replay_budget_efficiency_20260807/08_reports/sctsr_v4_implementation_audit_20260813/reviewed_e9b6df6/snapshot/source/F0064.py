from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import load_json, stable_digest


REVIEW_IDS = tuple(f"SA-{value}" for value in range(280, 290))
REVIEWER_IDENTITY = "SELF_REVIEW_NOT_INDEPENDENT_REVIEW"
_TOP_FIELDS = {
    "schema_version",
    "implementation_source_commit",
    "reviewer_identity",
    "generated_at_utc",
    "reviews",
    "review_digest",
}
_REVIEW_FIELDS = {"review_id", "status", "finding", "residual_risk", "anchors"}
_ANCHOR_FIELDS = {"relative_path", "start_line", "end_line", "line_sha256", "reviewed_symbols"}


def validate_manual_line_review(
    report: Mapping[str, Any] | str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    raw: Any = load_json(report) if not isinstance(report, Mapping) else report
    if not isinstance(raw, Mapping) or set(raw) != _TOP_FIELDS:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review schema is invalid")
    core = {key: value for key, value in raw.items() if key != "review_digest"}
    if raw.get("schema_version") != "stage1.sctsr.manual_line_review.v1" or raw.get("review_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review digest is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(raw.get("implementation_source_commit", ""))):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line review has an invalid source commit")
    if raw.get("reviewer_identity") != REVIEWER_IDENTITY:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line review improperly claims reviewer independence")
    try:
        datetime.fromisoformat(str(raw.get("generated_at_utc", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review timestamp is invalid") from exc
    reviews = raw.get("reviews")
    if not isinstance(reviews, list) or [row.get("review_id") if isinstance(row, Mapping) else None for row in reviews] != list(REVIEW_IDS):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line review must contain SA-280 through SA-289 exactly once and in order")
    root = Path(repository_root).resolve()
    failures: list[str] = []
    anchor_count = 0
    for row in reviews:
        if set(row) != _REVIEW_FIELDS or row["status"] not in {"PASS", "FAIL"}:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review row is invalid", failing_field=str(row.get("review_id")))
        if not str(row["finding"]).strip() or not str(row["residual_risk"]).strip():
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review finding or risk is empty", failing_field=str(row["review_id"]))
        anchors = row["anchors"]
        if not isinstance(anchors, list) or not anchors:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line review has no line anchor", failing_field=str(row["review_id"]))
        for anchor in anchors:
            if not isinstance(anchor, Mapping) or set(anchor) != _ANCHOR_FIELDS:
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review anchor schema is invalid", failing_field=str(row["review_id"]))
            relative = Path(str(anchor["relative_path"]))
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review anchor escapes repository") from exc
            if not path.is_file():
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review source is missing", artifact_path=str(path))
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review source is not strict UTF-8") from exc
            start, end = anchor["start_line"], anchor["end_line"]
            if type(start) is not int or type(end) is not int or not (1 <= start <= end <= len(lines)):
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review range is invalid", failing_field=str(row["review_id"]))
            line_bytes = ("\n".join(lines[start - 1 : end]) + "\n").encode("utf-8")
            observed = hashlib.sha256(line_bytes).hexdigest().upper()
            if observed != anchor["line_sha256"]:
                raise SctsrError(
                    ErrorCode.CLOSEOUT_NOT_VALIDATED,
                    "Manual line-review source lines changed after review",
                    failing_field=str(row["review_id"]),
                    observed=observed,
                    expected=anchor["line_sha256"],
                )
            symbols = anchor["reviewed_symbols"]
            if not isinstance(symbols, list) or not symbols or any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Manual line-review symbols are empty", failing_field=str(row["review_id"]))
            anchor_count += 1
        if row["status"] == "FAIL":
            failures.append(str(row["review_id"]))
    return {
        "status": "PASS" if not failures else "VALID_REVIEW_WITH_FAILURES",
        "review_count": len(reviews),
        "anchor_count": anchor_count,
        "failed_review_ids": failures,
        "implementation_source_commit": raw["implementation_source_commit"],
        "review_digest": raw["review_digest"],
    }
