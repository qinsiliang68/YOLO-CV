from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stage1_dynamic_replay_v3.literature_screened_review_v3 import (
    ScreenedReviewError,
    validate_screened_review_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _inputs(tmp_path: Path) -> tuple[Path, list[dict[str, str]], list[dict[str, str]]]:
    root = tmp_path / "corpus"
    source = root / "sources" / "P0001.pdf"
    text = root / "text" / "P0001.txt"
    source.parent.mkdir(parents=True)
    text.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
    text.write_text(
        "Abstract\nWe study sample selection.\n"
        "\f3 Method\nWe rank reducible loss within a fixed candidate batch.\n"
        "Algorithm 1 selects a subset and updates the model.\n"
        "\f4 Experiments\nWe compare against uniform random selection on three seeds.\n"
        "Table 1 reports 81.2 accuracy versus 80.1 for random.\n"
        "\f5 Ablation Study\nRemoving the reference loss reduces accuracy.\n"
        "\f6 Limitations\nOnly image classification was evaluated.\n",
        encoding="utf-8",
    )
    queue = [
        {
            "paper_id": "P0001",
            "canonical_work_id": "CW0001",
            "title": "Reducible Selection Study",
            "selection_role": "PRIMARY",
            "quota_rq": "RQ2",
            "method_source_path": "sources/P0001.pdf",
            "method_source_sha256": _sha256(source),
            "method_source_bytes": str(source.stat().st_size),
        }
    ]
    extraction = [
        {
            "paper_id": "P0001",
            "title": "Reducible Selection Study",
            "source_sha256": _sha256(source),
            "source_path": "sources/P0001.pdf",
            "source_bytes": str(source.stat().st_size),
            "text_path": "text/P0001.txt",
            "text_sha256": _sha256(text),
            "text_bytes": str(text.stat().st_size),
        }
    ]
    return root, queue, extraction


def test_screened_review_uses_corpus_relative_extraction_source_path(
    tmp_path: Path,
) -> None:
    root, queue, extraction = _inputs(tmp_path)
    relocated = root / "staging" / "broad" / "sources" / "P0001.pdf"
    relocated.parent.mkdir(parents=True)
    (root / "sources" / "P0001.pdf").replace(relocated)
    extraction[0]["source_path"] = "staging/broad/sources/P0001.pdf"
    record = _record()
    record["method_source"]["path"] = extraction[0]["source_path"]
    _bind_artifacts(root, record)

    result = validate_screened_review_records(
        corpus_root=root,
        queue_rows=queue,
        extraction_rows=extraction,
        records=[record],
    )

    assert result.status == "PASS"


def _anchor(page: int, line: int, quote: str, paraphrase: str) -> dict[str, object]:
    return {
        "page": page,
        "line": line,
        "quote": quote,
        "paraphrase_zh": paraphrase,
    }


def _record() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "paper_id": "P0001",
        "canonical_work_id": "CW0001",
        "title": "Reducible Selection Study",
        "decision": "SCREENED_ELIGIBLE",
        "reviewed_at": "2026-08-10T08:00:00+08:00",
        "reviewer": "codex_primary_fulltext_review",
        "method_source": {
            "path": "sources/P0001.pdf",
            "bytes": 0,
            "sha256": "SET_BY_TEST",
        },
        "text_source": {
            "path": "text/P0001.txt",
            "bytes": 0,
            "sha256": "SET_BY_TEST",
        },
        "rq_ids": ["RQ2", "RQ7"],
        "section_evidence": {
            "METHODS": {
                "status": "READ",
                "pages": "3",
                "anchors": [
                    _anchor(
                        2,
                        2,
                        "We rank reducible loss within a fixed candidate batch.",
                        "方法在固定候选批次内计算可约损失并据此排序。",
                    )
                ],
            },
            "EXPERIMENTS": {
                "status": "READ",
                "pages": "4",
                "anchors": [
                    _anchor(
                        3,
                        2,
                        "We compare against uniform random selection on three seeds.",
                        "实验使用三个训练种子并设置均匀随机选样对照。",
                    )
                ],
            },
            "ABLATIONS": {
                "status": "READ",
                "pages": "5",
                "anchors": [
                    _anchor(
                        4,
                        2,
                        "Removing the reference loss reduces accuracy.",
                        "去掉参考损失会降低结果，隔离了可约项的贡献。",
                    )
                ],
            },
            "LIMITATIONS": {
                "status": "READ",
                "pages": "6",
                "anchors": [
                    _anchor(
                        5,
                        2,
                        "Only image classification was evaluated.",
                        "作者只验证图像分类，未覆盖高召回有限预算回流。",
                    )
                ],
            },
        },
        "formulas": ["rho_i = current_loss_i - reference_loss_i"],
        "algorithm_steps": [
            "从固定候选批次计算当前损失与独立参考损失。",
            "按可约差排序后在冻结预算内选择子集并更新模型。",
        ],
        "variables": ["current_loss", "reference_loss", "candidate_batch", "selection_budget"],
        "selection_timing": "每个训练步骤先形成候选批次，再在反向传播前完成选择。",
        "refresh_rule": "每个训练步骤重新评分，论文没有使用永久离线 Top-K。",
        "budget": {
            "unit": "optimizer-visible selected examples per step",
            "denominator": "candidate batch size",
            "unique_sample_definition": "论文按候选样本身份计数，但未报告整轮唯一身份数。",
            "repeat_definition": "同一训练样本在后续候选批次再次被选中构成重复曝光。",
            "cumulative_exposure_definition": "所有训练步骤实际被选中并参与反向传播的样本次数总和。",
            "compute_cost": "每步需对候选批次计算当前模型和参考模型损失。",
        },
        "random_baselines": ["同候选批次与反向传播预算的 uniform random selection。"],
        "datasets": ["Fixture image classification dataset"],
        "models": ["Fixture classifier"],
        "seed_count": 3,
        "checkpoint_selection": "固定训练终点模型；未按测试集挑选 checkpoint。",
        "results": [
            {
                "claim": "可约损失选择在该设定高于均匀随机。",
                "locator": "page 4, Table 1",
                "value": "81.2 versus 80.1 accuracy",
                "anchor": _anchor(
                    3,
                    3,
                    "Table 1 reports 81.2 accuracy versus 80.1 for random.",
                    "表一给出方法与随机基线的数值差异。",
                ),
            }
        ],
        "ablations": ["移除 reference loss 后准确率下降。"],
        "negative_results": ["NOT_REPORTED_BY_PAPER"],
        "failure_conditions": ["参考模型不能提供有效可约性信息时，差值排序可能退化。"],
        "limitations": ["只验证普通图像分类，没有 FN95 局部目标。"],
        "transfer_class": "INSPIRED_ADAPTATION",
        "stage1_mechanism_zh": (
            "该方法支持把当前样本损失减去交叉拟合参考损失作为剩余可学习性候选信号，"
            "但 Stage1 必须另设 current-loss 与严格随机对照，并冻结累计实际曝光。"
        ),
        "stage1_non_inference_zh": (
            "论文没有证明离线 OOF 差值能够跨训练种子改善 FN95，也没有证明单样本分数可替代"
            "集合覆盖或真实 replay 干预，因此只能归为受启发适配。"
        ),
        "exclusion_reason": "NOT_APPLICABLE_WITH_REASON:eligible for SCREENED evidence tier",
    }


def _bind_artifacts(root: Path, record: dict[str, object]) -> None:
    for field in ("method_source", "text_source"):
        artifact = record[field]
        path = root / artifact["path"]
        artifact["bytes"] = path.stat().st_size
        artifact["sha256"] = _sha256(path)


def test_screened_review_binds_full_text_sections_and_exact_anchors(tmp_path: Path) -> None:
    root, queue, extraction = _inputs(tmp_path)
    record = _record()
    _bind_artifacts(root, record)

    result = validate_screened_review_records(
        corpus_root=root,
        queue_rows=queue,
        extraction_rows=extraction,
        records=[record],
    )

    assert result.status == "PASS"
    assert result.reviewed_count == 1
    assert result.eligible_count == 1
    assert result.excluded_count == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("title", "title identity"),
        ("hash", "SHA"),
        ("anchor", "anchor quote"),
        ("placeholder", "placeholder"),
        ("missing_section", "section_evidence"),
    ],
)
def test_screened_review_fails_closed_on_identity_or_reading_evidence_drift(
    tmp_path: Path, mutation: str, match: str
) -> None:
    root, queue, extraction = _inputs(tmp_path)
    record = _record()
    _bind_artifacts(root, record)
    if mutation == "title":
        record["title"] = "Another Paper"
    elif mutation == "hash":
        record["text_source"]["sha256"] = "0" * 64
    elif mutation == "anchor":
        record["section_evidence"]["METHODS"]["anchors"][0]["quote"] = "invented quote"
    elif mutation == "placeholder":
        record["stage1_mechanism_zh"] = "TODO"
    else:
        record["section_evidence"].pop("EXPERIMENTS")

    with pytest.raises(ScreenedReviewError, match=match):
        validate_screened_review_records(
            corpus_root=root,
            queue_rows=queue,
            extraction_rows=extraction,
            records=[record],
        )


def test_canonical_title_may_contain_unknown_as_scientific_subject(tmp_path: Path) -> None:
    root, queue, extraction = _inputs(tmp_path)
    title = "Learning with Unknown Label Noise"
    queue[0]["title"] = title
    extraction[0]["title"] = title
    record = _record()
    record["title"] = title
    _bind_artifacts(root, record)

    result = validate_screened_review_records(
        corpus_root=root,
        queue_rows=queue,
        extraction_rows=extraction,
        records=[record],
    )

    assert result.status == "PASS"


def test_not_reported_ablation_requires_a_specific_full_text_reason(tmp_path: Path) -> None:
    root, queue, extraction = _inputs(tmp_path)
    record = _record()
    _bind_artifacts(root, record)
    record["section_evidence"]["ABLATIONS"] = {
        "status": "NOT_REPORTED_BY_PAPER",
        "pages": "1-6",
        "anchors": [],
        "absence_reason_zh": "全文实验节和附录均未设置组件移除或敏感性比较。",
    }

    result = validate_screened_review_records(
        corpus_root=root,
        queue_rows=queue,
        extraction_rows=extraction,
        records=[record],
    )
    assert result.status == "PASS"

    record["section_evidence"]["ABLATIONS"]["absence_reason_zh"] = "未报告"
    with pytest.raises(ScreenedReviewError, match="absence_reason_zh"):
        validate_screened_review_records(
            corpus_root=root,
            queue_rows=queue,
            extraction_rows=extraction,
            records=[record],
        )


def test_seed_evidence_can_preserve_different_counts_for_scoring_and_retraining(
    tmp_path: Path,
) -> None:
    root, queue, extraction = _inputs(tmp_path)
    record = _record()
    _bind_artifacts(root, record)
    record["seed_count"] = {
        "counts": ["10 independent runs for score averaging", "4 independent subset retraining runs"],
        "scope": "page 5, data-pruning experiments",
        "aggregation": "score mean over 10; final accuracy mean and 16th-84th percentile over 4",
    }

    result = validate_screened_review_records(
        corpus_root=root,
        queue_rows=queue,
        extraction_rows=extraction,
        records=[record],
    )

    assert result.status == "PASS"


def test_screened_review_cli_starts_without_manual_pythonpath(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "stage1_dynamic_replay_v3"
                / "validate_literature_screened_reviews_v3.py"
            ),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--review-dir" in result.stdout
    assert "--screening-queue" in result.stdout
