import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stage1_dynamic_replay_v3.literature_evidence_v2 import (
    LiteratureEvidenceError,
    TierCounts,
    audit_completion,
    deterministic_audit_ids,
    validate_discovery_evidence,
    validate_corpus,
    validate_random_audit,
    validate_second_pass,
    validate_source_acquisitions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _long_zh(label: str, purpose: str) -> str:
    return (
        f"{label}：本文围绕{purpose}建立明确研究问题，核对原始摘要、方法概览和结论后，"
        "区分作者实际证明的范围与尚未验证的外推。该证据只用于形成可证伪机制，"
        "不能把静态相关性直接称为样本回流效用，也不能替代跨训练种子的随机干预。"
    )


def _artifact(path: Path, *, kind: str) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _screened(source: Path, label: str) -> dict[str, object]:
    return {
        "sections_checked": ["METHODS", "EXPERIMENTS", "ABLATIONS", "LIMITATIONS"],
        "method_source": _artifact(source, kind="PRIMARY_FULL_TEXT_HTML"),
        "formulas": [f"{label} 的目标函数与变量定义已按原文方法节核对。"],
        "algorithm_steps": ["构造候选集合。", "按预注册预算执行选择并训练。"],
        "variables": ["候选样本", "训练状态", "有限预算"],
        "selection_timing": "训练前或作者指定检查点，见方法节。",
        "refresh_rule": "不刷新；这是该测试论文的明确合同。",
        "budget": {
            "unit": "unique samples",
            "denominator": "training set",
            "unique_sample_definition": "不同 sample identity 的数量。",
            "repeat_definition": "同一 identity 再次出现计为 repeat。",
            "cumulative_exposure_definition": "所有训练步骤可见的额外样本出现次数。",
            "compute_cost": "按候选打分和一次选择报告。",
        },
        "random_baselines": ["同候选池、同预算 global random。"],
        "datasets": ["Synthetic fixture dataset"],
        "models": ["Fixture classifier"],
        "seed_count": 3,
        "checkpoint_selection": "固定末轮 checkpoint。",
        "results": [
            {
                "claim": f"{label} 报告了与随机对照的配对差异。",
                "locator": "Section 4, Table 1, page 4",
                "value": "direction reported; exact value not reused in fixture",
            }
        ],
        "ablations": ["移除选择信号的消融，见 Section 4.2。"],
        "negative_results": ["某一数据集上未稳定超过随机，见 Section 4.3。"],
        "failure_conditions": ["候选池覆盖不足时方法失效。"],
        "limitations": ["作者未验证高召回 FN95 场景。"],
        "transfer_class": "INSPIRED_ADAPTATION",
    }


def _deep(pdf: Path, label: str) -> dict[str, object]:
    return {
        "first_read_at": "2026-08-09T08:00:00+08:00",
        "full_text": _artifact(pdf, kind="PRIMARY_FULL_TEXT_PDF"),
        "page_count": 6,
        "section_coverage": [
            {"section": "Introduction", "pages": "1-2", "status": "READ_FULLY"},
            {"section": "Method", "pages": "2-3", "status": "READ_FULLY"},
            {"section": "Experiments", "pages": "3-5", "status": "READ_FULLY"},
            {"section": "Limitations and conclusion", "pages": "5-6", "status": "READ_FULLY"},
        ],
        "anchors": [
            {"page": 2, "locator": "Eq. 1", "paraphrase": f"{label} 定义候选价值目标。"},
            {"page": 4, "locator": "Table 1", "paraphrase": f"{label} 比较同预算随机对照。"},
            {"page": 5, "locator": "Ablation", "paraphrase": f"{label} 报告失败条件。"},
        ],
        "formula_assumptions": ["一阶局部近似只在小更新下解释方向。"],
        "algorithm_complexity": "候选打分为线性扫描，集合选择另计。",
        "randomness": "记录训练 seed、选择 seed 和数据顺序。",
        "data_roles": ["train", "selection validation", "held-out evaluation"],
        "leakage_risks": ["选择目标与最终评价必须使用不同数据角色。"],
        "budget_fairness": "目标组和随机组匹配实际 optimizer-visible exposure。",
        "seed_variation": "逐 seed 报告方向，不只报告均值。",
        "worst_case": "最差 seed 允许出现反转，因此不能声称确定性效用。",
        "key_ablations": ["去掉可学习性；去掉方向；去掉覆盖。"],
        "limitations": ["外部任务并非 Stage1，迁移必须重新干预。"],
        "stage1_mapping": {
            "fields": ["sample_id", "epoch", "loss", "selection_reason"],
            "interfaces": ["selector", "exposure ledger", "paired evaluator"],
            "cost": "只在冻结检查点计算候选信号。",
            "code_mapping": ["future selector module; no efficacy claim"],
        },
        "counter_check": "反向检查发现原文并未证明跨未见 seed 稳压随机，因此仅保留机制线索。",
    }


def _metadata(
    paper_id: str,
    tier: str,
    title: str,
    source: Path,
    *,
    pdf: Path | None = None,
) -> dict[str, object]:
    label = f"{paper_id}-{title}"
    screened: object = "NOT_ASSESSED_AT_BROAD_LEVEL"
    deep: object = "NOT_ASSESSED_AT_BROAD_LEVEL"
    if tier in {"SCREENED", "DEEP"}:
        screened = _screened(source, label)
    if tier == "DEEP":
        assert pdf is not None
        deep = _deep(pdf, label)
    return {
        "schema_version": "2.0",
        "paper_id": paper_id,
        "tier": tier,
        "identity": {
            "canonical_work_id": f"WORK-{paper_id}",
            "title": title,
            "authors": [f"Author {paper_id}"],
            "year": 2024,
            "venue": "Fixture Venue",
            "primary_url": f"https://example.org/{paper_id.lower()}",
            "doi": f"10.1234/{paper_id.lower()}",
            "arxiv_id": "NOT_APPLICABLE_WITH_REASON:no arXiv version",
            "openreview_id": "NOT_APPLICABLE_WITH_REASON:not an OpenReview paper",
            "merged_versions": ["official publication"],
        },
        "source_artifact": _artifact(source, kind="OFFICIAL_LANDING_HTML"),
        "rq_ids": ["RQ1", "RQ6"],
        "relation": "MIXED",
        "reading": {
            "read_at": "2026-08-09T08:00:00+08:00",
            "scopes": ["TITLE", "ABSTRACT", "PROBLEM", "METHOD_OVERVIEW", "CONCLUSION"],
            "sections_checked": ["Abstract", "Introduction", "Method overview", "Conclusion"],
            "summary_zh": _long_zh(label, "训练过程中样本是否值得继续学习"),
            "critical_review_zh": _long_zh(label, "证据强度、随机对照和跨种子边界"),
            "direct_relevance_chain": _long_zh(label, "论文变量如何映射到 Stage1 的 Q/R/A/D"),
            "supported_or_refuted": _long_zh(label, "论文支持与反驳的具体命题"),
            "transferable_mechanisms": [_long_zh(label, "可以迁移的机制")],
            "unsupported_inferences": [_long_zh(label, "不能从论文推出的结论")],
            "stage1_boundary": _long_zh(label, "迁移到 FN95 有限预算 replay 的边界"),
        },
        "screened": screened,
        "deep": deep,
    }


def _write_note(path: Path, metadata: dict[str, object]) -> None:
    text = (
        "<!-- STAGE1_EVIDENCE_V2 -->\n"
        "```json\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        + f"# {metadata['paper_id']} - {metadata['identity']['title']}\n\n"
        + "## 独立摘要\n\n"
        + metadata["reading"]["summary_zh"]
        + "\n\n## 批判性小综述\n\n"
        + metadata["reading"]["critical_review_zh"]
        + "\n"
    )
    path.write_text(text, encoding="utf-8")


def _build_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "review_500_300_100_v2"
    notes = root / "notes"
    sources = root / "sources"
    notes.mkdir(parents=True)
    sources.mkdir()
    tiers = ["BROAD", "BROAD", "SCREENED", "SCREENED", "DEEP"]
    rows: list[dict[str, object]] = []
    for index, tier in enumerate(tiers, start=1):
        paper_id = f"P{index:04d}"
        title = f"Unique Evidence Paper {index}"
        source = sources / f"{paper_id}.html"
        source.write_text(f"<html>{title} methods experiments conclusion</html>", encoding="utf-8")
        pdf = None
        if tier == "DEEP":
            pdf = sources / f"{paper_id}.pdf"
            pdf.write_bytes(b"%PDF-1.4\nfixture full text bytes\n%%EOF\n")
        metadata = _metadata(paper_id, tier, title, source, pdf=pdf)
        # Artifacts are expressed relative to the corpus root, but hashes need real files.
        metadata["source_artifact"] = _artifact(source, kind="OFFICIAL_LANDING_HTML") | {"path": source.relative_to(root).as_posix()}
        if tier in {"SCREENED", "DEEP"}:
            metadata["screened"]["method_source"] = _artifact(source, kind="PRIMARY_FULL_TEXT_HTML") | {"path": source.relative_to(root).as_posix()}
        if tier == "DEEP":
            metadata["deep"]["full_text"] = _artifact(pdf, kind="PRIMARY_FULL_TEXT_PDF") | {"path": pdf.relative_to(root).as_posix()}
        note = notes / f"{paper_id}.md"
        _write_note(note, metadata)
        identity = metadata["identity"]
        rows.append(
            {
                "paper_id": paper_id,
                "tier": tier,
                "canonical_work_id": identity["canonical_work_id"],
                "title": title,
                "authors": "; ".join(identity["authors"]),
                "year": identity["year"],
                "venue": identity["venue"],
                "primary_url": identity["primary_url"],
                "doi": identity["doi"],
                "arxiv_id": identity["arxiv_id"],
                "openreview_id": identity["openreview_id"],
                "note_path": note.relative_to(root).as_posix(),
                "source_path": source.relative_to(root).as_posix(),
                "source_sha256": _sha256(source),
                "source_bytes": source.stat().st_size,
            }
        )
    with (root / "CANONICAL_WORKS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_valid_corpus_enforces_exact_nested_tiers_and_artifact_identity(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)

    report = validate_corpus(
        root,
        expected=TierCounts(broad=5, screened=3, deep=1),
        inspect_pdf_pages=False,
    )

    assert report.status == "PASS"
    assert report.counts == {"broad": 5, "screened": 3, "deep": 1}
    assert report.note_count == 5
    assert report.errors == ()


def test_broad_only_staging_can_require_zero_screened_and_deep(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    registry = root / "CANONICAL_WORKS.csv"
    with registry.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["tier"] = "BROAD"
        note = root / row["note_path"]
        metadata = json.loads(note.read_text(encoding="utf-8").split("```json\n", 1)[1].split("\n```", 1)[0])
        metadata["tier"] = "BROAD"
        metadata["screened"] = "NOT_ASSESSED_AT_BROAD_LEVEL"
        metadata["deep"] = "NOT_ASSESSED_AT_BROAD_LEVEL"
        _write_note(note, metadata)
    with registry.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = validate_corpus(
        root,
        expected=TierCounts(broad=5, screened=0, deep=0),
        inspect_pdf_pages=False,
    )

    assert report.status == "PASS"
    assert report.counts == {"broad": 5, "screened": 0, "deep": 0}


def test_corpus_rejects_wrong_counts_missing_note_and_non_contiguous_ids(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    (root / "notes" / "P0002.md").unlink()

    with pytest.raises(LiteratureEvidenceError) as caught:
        validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)

    message = str(caught.value)
    assert "P0002" in message
    assert "note" in message.lower()


@pytest.mark.parametrize("field", ["title", "doi", "canonical_work_id"])
def test_corpus_rejects_duplicate_canonical_identity(tmp_path: Path, field: str) -> None:
    root = _build_corpus(tmp_path)
    registry = root / "CANONICAL_WORKS.csv"
    with registry.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1][field] = rows[0][field]
    with registry.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(LiteratureEvidenceError, match="duplicate"):
        validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)


def test_corpus_rejects_tampered_source_and_placeholder_text(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    (root / "sources" / "P0001.html").write_text("tampered", encoding="utf-8")
    note = root / "notes" / "P0002.md"
    note.write_text(note.read_text(encoding="utf-8").replace("MIXED", "TODO", 1), encoding="utf-8")

    with pytest.raises(LiteratureEvidenceError) as caught:
        validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)

    message = str(caught.value)
    assert "P0001" in message and "sha256" in message.lower()
    assert "P0002" in message and "TODO" in message


def test_legitimate_unknown_term_in_bibliographic_title_is_not_a_placeholder(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    registry = root / "CANONICAL_WORKS.csv"
    with registry.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["title"] = "Robust Learning with Unknown Label Noise"
    with registry.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    note = root / rows[0]["note_path"]
    metadata = json.loads(note.read_text(encoding="utf-8").split("```json\n", 1)[1].split("\n```", 1)[0])
    metadata["identity"]["title"] = rows[0]["title"]
    _write_note(note, metadata)

    report = validate_corpus(
        root,
        expected=TierCounts(5, 3, 1),
        inspect_pdf_pages=False,
    )

    assert report.status == "PASS"


def test_corpus_rejects_reused_per_paper_review_text(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    first = root / "notes" / "P0001.md"
    second = root / "notes" / "P0002.md"
    first_text = first.read_text(encoding="utf-8")
    second_text = second.read_text(encoding="utf-8")
    first_meta = json.loads(first_text.split("```json\n", 1)[1].split("\n```", 1)[0])
    second_meta = json.loads(second_text.split("```json\n", 1)[1].split("\n```", 1)[0])
    second_meta["reading"]["summary_zh"] = first_meta["reading"]["summary_zh"]
    _write_note(second, second_meta)

    with pytest.raises(LiteratureEvidenceError, match="reused"):
        validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)


def test_screened_and_deep_evidence_fail_closed(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    screened_note = root / "notes" / "P0003.md"
    text = screened_note.read_text(encoding="utf-8")
    metadata = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    metadata["screened"].pop("random_baselines")
    _write_note(screened_note, metadata)

    deep_note = root / "notes" / "P0005.md"
    text = deep_note.read_text(encoding="utf-8")
    metadata = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    metadata["deep"]["anchors"] = metadata["deep"]["anchors"][:2]
    _write_note(deep_note, metadata)

    with pytest.raises(LiteratureEvidenceError) as caught:
        validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)

    message = str(caught.value)
    assert "P0003.screened.random_baselines" in message
    assert "P0005.deep.anchors" in message


def test_deterministic_audit_sample_is_nested_and_reproducible(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    report = validate_corpus(
        root,
        expected=TierCounts(5, 3, 1),
        inspect_pdf_pages=False,
    )

    first = deterministic_audit_ids(report.papers, seed="stage1-literature-v2")
    second = deterministic_audit_ids(report.papers, seed="stage1-literature-v2")

    assert first == second
    assert len(first["broad"]) == 1
    assert len(first["screened"]) == 1
    assert len(first["deep"]) == 1
    assert set(first["deep"]).issubset(first["screened"])
    assert set(first["screened"]).issubset(first["broad"])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_discovery_evidence_binds_queries_candidates_and_exclusions(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    report = validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)
    snapshot = root / "discovery" / "raw" / "Q001.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('{"results": ["primary metadata"]}', encoding="utf-8")
    _write_csv(
        root / "discovery" / "QUERY_LOG.csv",
        [
            {
                "query_id": "Q001",
                "database": "OpenAlex",
                "exact_query": "sample learnability training dynamics random baseline",
                "searched_at": "2026-08-09T09:00:00+08:00",
                "result_start": 1,
                "result_end": 6,
                "raw_result_count": 6,
                "snapshot_path": snapshot.relative_to(root).as_posix(),
                "snapshot_sha256": _sha256(snapshot),
                "snapshot_bytes": snapshot.stat().st_size,
            }
        ],
    )
    candidate_rows = [
        {
            "candidate_id": f"C{index:04d}",
            "title": paper.title,
            "primary_url": paper.metadata["identity"]["primary_url"],
            "source_database": "OpenAlex",
            "query_ids": "Q001",
            "decision": "INCLUDED",
            "canonical_paper_id": paper.paper_id,
            "exclusion_reason": "NOT_APPLICABLE_WITH_REASON:included in canonical corpus",
        }
        for index, paper in enumerate(report.papers, start=1)
    ]
    candidate_rows.append(
        {
            "candidate_id": "C9999",
            "title": "Irrelevant excluded survey",
            "primary_url": "https://example.org/excluded",
            "source_database": "OpenAlex",
            "query_ids": "Q001",
            "decision": "EXCLUDED",
            "canonical_paper_id": "NOT_APPLICABLE_WITH_REASON:excluded candidate",
            "exclusion_reason": "Survey with no finite-budget intervention or training-dynamics evidence.",
        }
    )
    _write_csv(root / "discovery" / "CANDIDATE_LEDGER.csv", candidate_rows)

    result = validate_discovery_evidence(root, report.papers)
    assert result == {"queries": 1, "candidates": 6, "included": 5, "excluded": 1}

    candidate_rows[-1]["exclusion_reason"] = "TODO"
    _write_csv(root / "discovery" / "CANDIDATE_LEDGER.csv", candidate_rows)
    with pytest.raises(LiteratureEvidenceError, match="C9999"):
        validate_discovery_evidence(root, report.papers)


def test_source_acquisition_ledger_covers_every_counted_artifact(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    report = validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)
    rows: list[dict[str, object]] = []
    for paper in report.papers:
        artifacts = [("BROAD_SOURCE", paper.metadata["source_artifact"])]
        if paper.tier in {"SCREENED", "DEEP"}:
            artifacts.append(("METHOD_SOURCE", paper.metadata["screened"]["method_source"]))
        if paper.tier == "DEEP":
            artifacts.append(("DEEP_FULL_TEXT", paper.metadata["deep"]["full_text"]))
        for role, artifact in artifacts:
            rows.append(
                {
                    "paper_id": paper.paper_id,
                    "artifact_role": role,
                    "path": artifact["path"],
                    "url": paper.metadata["identity"]["primary_url"],
                    "retrieved_at": "2026-08-09T09:30:00+08:00",
                    "http_status": 200,
                    "content_type": "application/pdf" if role == "DEEP_FULL_TEXT" else "text/html",
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                    "retrieval_method": "HTTP_DOWNLOAD",
                    "source_authority": "PRIMARY_PUBLISHER",
                }
            )
    _write_csv(root / "SOURCE_ACQUISITION.csv", rows)

    assert validate_source_acquisitions(root, report.papers) == len(rows)

    rows.pop()
    _write_csv(root / "SOURCE_ACQUISITION.csv", rows)
    with pytest.raises(LiteratureEvidenceError, match="DEEP_FULL_TEXT"):
        validate_source_acquisitions(root, report.papers)


def test_fixed_random_audit_requires_exact_nested_sample_and_pass_evidence(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    report = validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)
    seed = "stage1-literature-v2"
    expected = deterministic_audit_ids(report.papers, seed=seed)
    rows: list[dict[str, object]] = []
    for audit_tier, paper_ids in expected.items():
        for paper_id in paper_ids:
            rows.append(
                {
                    "paper_id": paper_id,
                    "audit_tier": audit_tier.upper(),
                    "audited_at": "2026-08-10T10:00:00+08:00",
                    "identity_pass": "PASS",
                    "source_hash_pass": "PASS",
                    "relevance_pass": "PASS",
                    "reading_depth_pass": "PASS",
                    "locator_pass": "PASS",
                    "outcome": "PASS",
                    "audit_note": f"{paper_id} 逐项复核身份、原始来源、阅读范围和 Stage1 迁移边界，未以文件数量代替阅读。",
                }
            )
    _write_csv(root / "validation" / "RANDOM_AUDIT.csv", rows)

    result = validate_random_audit(root, report.papers, seed=seed)
    assert result["rows"] == 3

    rows[0]["outcome"] = "FAIL"
    _write_csv(root / "validation" / "RANDOM_AUDIT.csv", rows)
    with pytest.raises(LiteratureEvidenceError, match=rows[0]["paper_id"]):
        validate_random_audit(root, report.papers, seed=seed)


def test_second_pass_requires_deep_pdf_identity_elapsed_time_and_rq_coverage(tmp_path: Path) -> None:
    root = _build_corpus(tmp_path)
    report = validate_corpus(root, expected=TierCounts(5, 3, 1), inspect_pdf_pages=False)
    paper = next(item for item in report.papers if item.tier == "DEEP")
    pdf = paper.metadata["deep"]["full_text"]
    rows = [
        {
            "rank": 1,
            "paper_id": paper.paper_id,
            "priority_reason": "同时覆盖可学习性与严格随机对照，是 Q/R/A/D 合同的关键反证节点。",
            "first_read_at": paper.metadata["deep"]["first_read_at"],
            "second_read_at": "2026-08-10T10:00:00+08:00",
            "pdf_sha256": pdf["sha256"],
            "sections_rechecked": "Method p2-3; Experiments p3-5; Limitations p5-6",
            "claims_confirmed": "确认原文只证明其任务和预算内结果，并未证明 Stage1 跨 seed 效用。",
            "claims_revised": "把原来的直接迁移表述降级为受启发适配，并明确要求 Stage1 重新执行真实干预。",
            "contradictions": "随机对照和最差 seed 证据不足，不能写成稳压随机。",
            "stage1_effect": "保留机制 arm，同时加入严格匹配随机与 no-replay。",
            "same_reviewer_disclosed": "true",
            "outcome": "PASS",
        }
    ]
    _write_csv(root / "validation" / "SECOND_PASS_30.csv", rows)

    result = validate_second_pass(
        root,
        report.papers,
        minimum=1,
        required_rqs={"RQ1", "RQ6"},
        min_elapsed_hours=24,
    )
    assert result == {"papers": 1, "covered_rqs": ["RQ1", "RQ6"]}

    rows[0]["second_read_at"] = "2026-08-09T09:00:00+08:00"
    _write_csv(root / "validation" / "SECOND_PASS_30.csv", rows)
    with pytest.raises(LiteratureEvidenceError, match="24"):
        validate_second_pass(
            root,
            report.papers,
            minimum=1,
            required_rqs={"RQ1", "RQ6"},
            min_elapsed_hours=24,
        )


def test_literature_validator_cli_starts_without_manual_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "stage1_dynamic_replay_v3" / "validate_literature_evidence_v2.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--corpus-root" in result.stdout


def test_completion_audit_fails_closed_without_counted_evidence(tmp_path: Path) -> None:
    root = tmp_path / "review_500_300_100_v2"
    root.mkdir()

    report = audit_completion(
        root,
        expected=TierCounts(5, 3, 1),
        inspect_pdf_pages=False,
        second_pass_minimum=1,
        second_pass_required_rqs={"RQ1"},
    )

    assert report.status == "INCOMPLETE"
    assert report.formal_training_started is False
    assert report.engineering_gate_generated is False
    assert report.blind_holdout_opened is False
    assert report.gates["corpus"] == "FAIL"
    assert report.gates["discovery"] == "BLOCKED_BY_CORPUS"
    assert any("registry missing" in error for error in report.errors)
