import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stage1_dynamic_replay_v3.literature_manual_screening_v2 import (
    ManualScreeningError,
    blind_order_queue,
    merge_and_validate_manual_screening,
)


FIELDS = [
    "queue_id",
    "decision",
    "canonical_title",
    "primary_url_checked",
    "source_authority",
    "checked_at",
    "reading_scope",
    "direct_rq_ids",
    "relevance_class",
    "problem_summary_zh",
    "method_overview_zh",
    "conclusion_summary_zh",
    "critical_review_zh",
    "stage1_transfer_zh",
    "cannot_infer_zh",
    "exclusion_reason",
    "reviewer",
]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _queue(path: Path) -> None:
    _write(
        path,
        [
            {"queue_id": "LQ0001", "title": "Paper One", "primary_url": "https://p.example/1"},
            {"queue_id": "LQ0002", "title": "Paper Two", "primary_url": "https://p.example/2"},
        ],
    )


def _decision(queue_id: str, decision: str) -> dict[str, str]:
    included = decision == "ELIGIBLE_BROAD"
    return {
        "queue_id": queue_id,
        "decision": decision,
        "canonical_title": "Paper One" if queue_id == "LQ0001" else "Paper Two",
        "primary_url_checked": f"https://p.example/{1 if queue_id == 'LQ0001' else 2}",
        "source_authority": "PRIMARY_PUBLISHER",
        "checked_at": "2026-08-09T12:00:00+08:00",
        "reading_scope": "TITLE;ABSTRACT;PROBLEM;METHOD_OVERVIEW;CONCLUSION",
        "direct_rq_ids": "RQ2;RQ6" if included else "NOT_APPLICABLE_WITH_REASON:excluded",
        "relevance_class": "DIRECT_MECHANISM" if included else "EXCLUDED",
        "problem_summary_zh": "论文研究有限训练预算下如何识别仍可学习的样本。",
        "method_overview_zh": "方法比较当前损失和独立参考损失，并在固定预算内选择样本。",
        "conclusion_summary_zh": "结果显示可约损失在部分设置优于单独使用当前损失。",
        "critical_review_zh": "证据来自特定数据集，且随机种子数量限制了稳定性结论。",
        "stage1_transfer_zh": "可迁移为当前状态减去交叉拟合参考损失的独立消融。",
        "cannot_infer_zh": "不能推出该代理在 Stage1 或 FN95 工作点必然优于随机回流。",
        "exclusion_reason": (
            "NOT_APPLICABLE_WITH_REASON:included candidate"
            if included
            else "NO_DIRECT_SAMPLE_UTILITY_MECHANISM"
        ),
        "reviewer": "codex_primary_review",
    }


def test_manual_screening_requires_exact_queue_coverage_and_full_broad_scope(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    _write(
        decision_dir / "batch_001.csv",
        [_decision("LQ0001", "ELIGIBLE_BROAD"), _decision("LQ0002", "EXCLUDE")],
    )

    result = merge_and_validate_manual_screening(queue_path=queue, decision_dir=decision_dir)

    assert result.status == "PASS"
    assert result.reviewed_count == 2
    assert result.eligible_count == 1
    assert result.excluded_count == 1


def test_manual_screening_can_validate_one_explicit_batch_without_reading_siblings(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    selected = decision_dir / "batch_002.csv"
    _write(
        selected,
        [_decision("LQ0001", "ELIGIBLE_BROAD"), _decision("LQ0002", "EXCLUDE")],
    )
    _write(decision_dir / "batch_001.csv", [_decision("LQ0001", "ELIGIBLE_BROAD")])

    result = merge_and_validate_manual_screening(
        queue_path=queue,
        decision_dir=decision_dir,
        decision_paths=[selected],
    )

    assert result.status == "PASS"
    assert result.reviewed_count == 2


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "scope", "rq", "prose"])
def test_manual_screening_fails_closed_on_incomplete_or_template_decisions(
    tmp_path: Path, mutation: str
) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    first = _decision("LQ0001", "ELIGIBLE_BROAD")
    second = _decision("LQ0002", "EXCLUDE")
    rows = [first, second]
    if mutation == "missing":
        rows = [first]
    elif mutation == "duplicate":
        rows = [first, first, second]
    elif mutation == "scope":
        first["reading_scope"] = "TITLE;ABSTRACT"
    elif mutation == "rq":
        first["direct_rq_ids"] = "RQ99"
    else:
        first["critical_review_zh"] = first["problem_summary_zh"]

    _write(decision_dir / "batch_001.csv", rows)
    with pytest.raises(ManualScreeningError):
        merge_and_validate_manual_screening(queue_path=queue, decision_dir=decision_dir)


def test_manual_screening_rejects_unreviewed_placeholders(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    rows = [_decision("LQ0001", "ELIGIBLE_BROAD"), _decision("LQ0002", "EXCLUDE")]
    rows[0]["method_overview_zh"] = "TODO"
    _write(decision_dir / "batch_001.csv", rows)

    with pytest.raises(ManualScreeningError, match="placeholder"):
        merge_and_validate_manual_screening(queue_path=queue, decision_dir=decision_dir)


@pytest.mark.parametrize("placeholder", ["待补", "待确认", "同上", "未阅读", "未核对"])
def test_manual_screening_rejects_chinese_unreviewed_placeholders(
    tmp_path: Path, placeholder: str
) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    rows = [_decision("LQ0001", "ELIGIBLE_BROAD"), _decision("LQ0002", "EXCLUDE")]
    rows[0]["critical_review_zh"] = f"该字段仍然{placeholder}，不得获得人工阅读信用。"
    _write(decision_dir / "batch_001.csv", rows)

    with pytest.raises(ManualScreeningError, match="placeholder"):
        merge_and_validate_manual_screening(queue_path=queue, decision_dir=decision_dir)


def test_excluded_candidate_may_stop_after_primary_abstract_without_fake_full_read(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    included = _decision("LQ0001", "ELIGIBLE_BROAD")
    excluded = _decision("LQ0002", "EXCLUDE")
    excluded["reading_scope"] = "TITLE;ABSTRACT;PROBLEM"
    excluded["method_overview_zh"] = (
        "NOT_APPLICABLE_WITH_REASON:摘要已证明研究对象不是训练样本选择或回流机制。"
    )
    excluded["conclusion_summary_zh"] = (
        "NOT_APPLICABLE_WITH_REASON:候选在标题摘要层即按预注册范围排除，未冒充全文阅读。"
    )
    excluded["stage1_transfer_zh"] = (
        "NOT_APPLICABLE_WITH_REASON:问题对象不属于模型训练中的样本效用或有限预算回流。"
    )
    _write(decision_dir / "batch_001.csv", [included, excluded])

    result = merge_and_validate_manual_screening(queue_path=queue, decision_dir=decision_dir)

    assert result.status == "PASS"
    assert result.excluded_count == 1


def test_eligible_candidate_cannot_claim_broad_read_without_conclusion(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    included = _decision("LQ0001", "ELIGIBLE_BROAD")
    included["reading_scope"] = "TITLE;ABSTRACT;PROBLEM;METHOD_OVERVIEW"
    excluded = _decision("LQ0002", "EXCLUDE")
    _write(decision_dir / "batch_001.csv", [included, excluded])

    with pytest.raises(ManualScreeningError, match="incomplete reading scope"):
        merge_and_validate_manual_screening(queue_path=queue, decision_dir=decision_dir)


def test_blind_order_is_invariant_to_input_order_year_and_legacy_depth() -> None:
    rows = [
        {
            "queue_id": "LQ0001",
            "title": "First Study",
            "authors": "A. Author",
            "doi": "10.1/first",
            "year": "2026",
            "legacy_depth": "DEEP_READ",
        },
        {
            "queue_id": "LQ0002",
            "title": "Second Study",
            "authors": "B. Author",
            "doi": "10.1/second",
            "year": "2001",
            "legacy_depth": "ABSTRACT_SCREEN",
        },
    ]
    mutated = [dict(rows[1]), dict(rows[0])]
    mutated[0]["year"] = "1900"
    mutated[0]["legacy_depth"] = "DEEP_READ"
    mutated[1]["year"] = "2099"
    mutated[1]["legacy_depth"] = "NOT_APPLICABLE"

    first = blind_order_queue(rows, frozen_seed="stage1-screen-v1")
    second = blind_order_queue(mutated, frozen_seed="stage1-screen-v1")

    assert [row["queue_id"] for row in first] == [row["queue_id"] for row in second]
    assert [row["blind_order_key"] for row in first] == [row["blind_order_key"] for row in second]


def test_manual_screening_cli_publishes_atomic_non_credit_receipt(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    _queue(queue)
    decision_dir = tmp_path / "decisions"
    selected_decision = decision_dir / "batch_001.csv"
    _write(
        selected_decision,
        [_decision("LQ0001", "ELIGIBLE_BROAD"), _decision("LQ0002", "EXCLUDE")],
    )
    _write(decision_dir / "unrelated_batch.csv", [_decision("LQ0001", "ELIGIBLE_BROAD")])
    output_csv = tmp_path / "validation" / "manual_screened_candidates.csv"
    output_json = tmp_path / "validation" / "manual_screening_receipt.json"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "stage1_dynamic_replay_v3"
        / "validate_literature_manual_screening_v2.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--queue",
            str(queue),
            "--decision-dir",
            str(decision_dir),
            "--decision-file",
            str(selected_decision),
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(list(csv.DictReader(output_csv.open(encoding="utf-8-sig")))) == 2
    receipt = json.loads(output_json.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["reviewed_count"] == 2
    assert receipt["eligible_candidate_count"] == 1
    assert receipt["excluded_count"] == 1
    assert receipt["reading_credit_granted"] is False
    assert receipt["formal_broad_corpus_count_increment"] == 0
    assert receipt["queue_sha256"]
    assert receipt["merged_output_sha256"]
    assert receipt["decision_files"] == [
        {
            "file_name": "batch_001.csv",
            "row_count": 2,
            "sha256": receipt["decision_files"][0]["sha256"],
        }
    ]
