from __future__ import annotations

from hashlib import sha256

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.manual_line_review import REVIEW_IDS, validate_manual_line_review
from stage1_sctsr_v4.serialization import canonical_json_bytes, stable_digest


def _report(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("first\nsecond\nthird\n", encoding="utf-8")
    line_bytes = "second\n".encode("utf-8")
    reviews = []
    for review_id in REVIEW_IDS:
        reviews.append(
            {
                "review_id": review_id,
                "status": "PASS",
                "finding": "The reviewed boundary implements the registered fail-closed contract.",
                "residual_risk": "Formal paired training remains unperformed and is outside this line review.",
                "anchors": [
                    {
                        "relative_path": "module.py",
                        "start_line": 2,
                        "end_line": 2,
                        "line_sha256": sha256(line_bytes).hexdigest().upper(),
                        "reviewed_symbols": ["synthetic_symbol"],
                    }
                ],
            }
        )
    core = {
        "schema_version": "stage1.sctsr.manual_line_review.v1",
        "implementation_source_commit": "a" * 40,
        "reviewer_identity": "SELF_REVIEW_NOT_INDEPENDENT_REVIEW",
        "generated_at_utc": "2026-08-13T00:00:00+00:00",
        "reviews": reviews,
    }
    return {**core, "review_digest": stable_digest(core)}


def test_manual_line_review_requires_exact_sa280_to_sa289_and_line_digests(tmp_path):
    result = validate_manual_line_review(_report(tmp_path), repository_root=tmp_path)
    assert result["status"] == "PASS"
    assert result["review_count"] == 10


def test_manual_line_review_rejects_source_line_drift(tmp_path):
    report = _report(tmp_path)
    (tmp_path / "module.py").write_text("first\nreplaced\nthird\n", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_manual_line_review(report, repository_root=tmp_path)
    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED
