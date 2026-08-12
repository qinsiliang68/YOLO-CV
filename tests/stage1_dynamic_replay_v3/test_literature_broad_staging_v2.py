from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stage1_dynamic_replay_v3.literature_broad_staging_v2 import (
    BroadStagingError,
    build_broad_staging,
)
from stage1_dynamic_replay_v3.literature_evidence_v2 import (
    TierCounts,
    validate_corpus,
    validate_source_acquisitions,
)
from stage1_dynamic_replay_v3.literature_tier_freeze_v2 import TierSelectionPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    batches = root / "discovery" / "manual_screen_batches_v2"
    decisions = root / "discovery" / "manual_screen_decisions_v2"
    sources = root / "sources" / "fixture"
    sources.mkdir(parents=True)
    queue_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for index in range(1, 13):
        queue_id = f"RG{index:04d}"
        title = f"Unique replay evidence study {index}"
        rq = f"RQ{((index - 1) % 8) + 1}"
        source = sources / f"{queue_id}.html"
        source.write_text(
            f"<html><title>{title}</title><p>method experiment conclusion {index}</p></html>",
            encoding="utf-8",
        )
        digest = hashlib.sha256(source.read_bytes()).hexdigest().upper()
        receipt = source.with_suffix(".html.receipt.json")
        receipt.write_text(
            json.dumps(
                {
                    "ledger_row": {
                        "url": f"https://example.org/{queue_id.lower()}",
                        "retrieved_at": "2026-08-10T08:00:00+08:00",
                        "http_status": 200,
                        "content_type": "text/html",
                        "retrieval_method": "HTTP_DOWNLOAD",
                        "source_authority": "PRIMARY_PUBLISHER",
                    }
                }
            ),
            encoding="utf-8",
        )
        queue_rows.append(
            {
                "queue_id": queue_id,
                "title": title,
                "authors": f"Author {index}",
                "year": 2024,
                "venue": "Fixture Venue",
                "primary_url": f"https://example.org/{queue_id.lower()}",
                "doi": f"10.1234/{queue_id.lower()}",
                "candidate_version_ids": f"CV{index:04d}",
            }
        )
        decision_rows.append(
            {
                "queue_id": queue_id,
                "decision": "ELIGIBLE_BROAD",
                "canonical_title": title,
                "primary_url_checked": f"https://example.org/{queue_id.lower()}",
                "source_authority": "PRIMARY_PUBLISHER",
                "checked_at": "2026-08-10T09:00:00+08:00",
                "reading_scope": "TITLE;ABSTRACT;PROBLEM;METHOD_OVERVIEW;CONCLUSION",
                "direct_rq_ids": rq,
                "relevance_class": "TRANSFER_COMPONENT" if index in {11, 12} else "DIRECT_INTERVENTION",
                "problem_summary_zh": f"研究问题 {index} 是有限训练预算下如何判断样本是否仍然值得学习并避免错误选择。",
                "method_overview_zh": f"方法 {index} 根据当前训练状态选择候选样本，并在固定预算下训练模型后比较结果。",
                "conclusion_summary_zh": f"结论 {index} 报告该任务中选择机制有条件地改善表现，但效果依赖数据和训练设置。",
                "critical_review_zh": f"批判 {index} 指出随机基线、累计曝光和跨种子证据仍需单独核对，不能只看平均结果。",
                "stage1_transfer_zh": f"Stage1 映射 {index} 是把该机制作为独立析因臂并保持基础顺序和总曝光一致。",
                "cannot_infer_zh": f"不能由论文 {index} 推出 CCTV 的固定阈值、任意权重或跨未见种子的稳定收益。",
                "exclusion_reason": "NOT_APPLICABLE_WITH_REASON:included candidate",
                "reviewer": "fixture_reviewer",
            }
        )
        validation_rows.append(
            {
                "paper_id": queue_id,
                "title": title,
                "path": source.relative_to(root).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": digest,
                "source_format": "HTML",
                "page_count": "NOT_APPLICABLE_WITH_REASON:HTML source",
                "title_token_coverage": "1.000000",
                "source_authority": "PRIMARY_PUBLISHER",
                "source_url": f"https://example.org/{queue_id.lower()}",
                "receipt_path": receipt.relative_to(root).as_posix(),
                "probe_tool": "HTML_TEXT_IDENTITY",
                "source_superseded": "False",
                "superseded_sha256": "NOT_APPLICABLE_WITH_REASON:no source supersession",
                "supersession_reason": "NOT_APPLICABLE_WITH_REASON:no source supersession",
                "reading_credit_granted": "False",
            }
        )
    _write_csv(batches / "review_input_001.csv", queue_rows)
    _write_csv(decisions / "batch_001.csv", decision_rows)
    _write_csv(batches / "source_validation_001.csv", validation_rows)
    _write_csv(
        root / "discovery" / "CANONICAL_MERGES_v2.csv",
        [
            {
                "alias_queue_id": "NOT_APPLICABLE_WITH_REASON:no aliases in fixture",
                "canonical_queue_id": "NOT_APPLICABLE_WITH_REASON:no aliases in fixture",
                "evidence": "NOT_APPLICABLE_WITH_REASON:no duplicate versions in fixture",
                "adjudicated_at": "2026-08-10T09:00:00+08:00",
                "adjudicator": "fixture_reviewer",
            }
        ],
    )
    return root


def test_build_broad_staging_is_self_contained_and_strictly_validated(tmp_path: Path) -> None:
    root = _fixture_corpus(tmp_path)
    policy_source = root / "discovery" / "MANDATORY_SELECTION_POLICY.csv"
    policy_source.write_text("policy_id\nfixture-policy\n", encoding="utf-8")
    policy = TierSelectionPolicy(
        total=10,
        minimum_per_rq=1,
        maximum_per_rq=3,
        maximum_transfer=2,
        frozen_seed="broad-staging-fixture",
    )

    result = build_broad_staging(
        root,
        batch_numbers=(1,),
        policy=policy,
        policy_source_paths=(policy_source,),
        output_relative=Path("staging/broad_freeze_v2"),
    )

    assert result.status == "PASS"
    assert result.selected_count == 10
    assert result.reserve_count == 2
    assert result.formal_broad_increment == 0
    report = validate_corpus(
        result.output_root,
        expected=TierCounts(10, 0, 0),
        inspect_pdf_pages=False,
    )
    assert report.status == "PASS"
    assert validate_source_acquisitions(result.output_root, report.papers) == 10

    input_manifest = result.output_root / "BUILD_INPUT_MANIFEST.csv"
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    assert {row["root_scope"] for row in input_rows} == {
        "CORPUS",
        "REPOSITORY_CODE",
    }
    assert all(not Path(row["path"]).is_absolute() for row in input_rows)
    assert any(row["input_role"] == "CANONICAL_MERGES" for row in input_rows)
    assert any(row["input_role"] == "BUILDER_CODE" for row in input_rows)
    assert any(
        row["input_role"] == "SELECTION_POLICY_SOURCE"
        and row["path"] == "discovery/MANDATORY_SELECTION_POLICY.csv"
        for row in input_rows
    )
    for row in input_rows:
        source_root = root if row["root_scope"] == "CORPUS" else REPO_ROOT
        source = source_root / row["path"]
        assert source.is_file()
        assert source.stat().st_size == int(row["bytes"])
        assert hashlib.sha256(source.read_bytes()).hexdigest().upper() == row["sha256"]

    receipt = json.loads(
        (result.output_root / "FREEZE_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert receipt["build_input_manifest_sha256"] == hashlib.sha256(
        input_manifest.read_bytes()
    ).hexdigest().upper()
    assert receipt["build_input_count"] == len(input_rows)
    assert receipt["selection_policy"]["mandatory_canonical_work_ids"] == []
    assert receipt["selection_policy"]["total"] == 10


def test_build_broad_staging_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    root = _fixture_corpus(tmp_path)
    source = root / "sources" / "fixture" / "RG0001.html"
    source.write_bytes(b"X" * source.stat().st_size)

    with pytest.raises(BroadStagingError, match="RG0001.*SHA"):
        build_broad_staging(
            root,
            batch_numbers=(1,),
            policy=TierSelectionPolicy(
                total=10,
                minimum_per_rq=1,
                maximum_per_rq=3,
                maximum_transfer=2,
                frozen_seed="tamper-fixture",
            ),
            output_relative=Path("staging/tamper"),
        )


def test_build_broad_staging_binds_explicit_versioned_merge_ledger(
    tmp_path: Path,
) -> None:
    root = _fixture_corpus(tmp_path)
    merge_ledger = root / "discovery" / "CANONICAL_MERGES_v4.csv"
    merge_ledger.write_text(
        (
            "alias_queue_id,canonical_queue_id,evidence,adjudicated_at,adjudicator\n"
            "NOT_APPLICABLE_WITH_REASON:no aliases in fixture,"
            "NOT_APPLICABLE_WITH_REASON:no aliases in fixture,"
            "NOT_APPLICABLE_WITH_REASON:no duplicate versions in fixture,"
            "2026-08-10T09:00:00+08:00,fixture_reviewer\n"
        ),
        encoding="utf-8",
    )

    result = build_broad_staging(
        root,
        batch_numbers=(1,),
        policy=TierSelectionPolicy(
            total=10,
            minimum_per_rq=1,
            maximum_per_rq=3,
            maximum_transfer=2,
            frozen_seed="versioned-merge-ledger-fixture",
        ),
        merge_ledger_path=Path("discovery/CANONICAL_MERGES_v4.csv"),
        output_relative=Path("staging/versioned-merge-ledger"),
    )

    with (result.output_root / "BUILD_INPUT_MANIFEST.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        inputs = list(csv.DictReader(handle))
    merge_inputs = [row for row in inputs if row["input_role"] == "CANONICAL_MERGES"]
    assert len(merge_inputs) == 1
    assert merge_inputs[0]["path"] == "discovery/CANONICAL_MERGES_v4.csv"
    assert merge_inputs[0]["sha256"] == hashlib.sha256(
        merge_ledger.read_bytes()
    ).hexdigest().upper()


def test_broad_staging_cli_starts_without_manual_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "stage1_dynamic_replay_v3"
                / "build_literature_broad_staging_v2.py"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--corpus-root" in result.stdout
