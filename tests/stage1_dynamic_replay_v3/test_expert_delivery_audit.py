from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from stage1_dynamic_replay_v3.expert_delivery_audit import (
    ExpertAuditError,
    _remove_extraction_tree,
    audit_expert_deliveries,
    extract_manifested_tar_subset,
    parse_sha256_ledger,
    validate_manifested_tar,
    validate_tar_archive,
    validate_zip_archive,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _add_tar_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _write_manifested_tar(path: Path, *, corrupt_hash: bool = False) -> None:
    payload = b"verified payload\n"
    expected_hash = "0" * 64 if corrupt_hash else _sha256(payload)
    manifest_buffer = io.StringIO(newline="")
    writer = csv.writer(manifest_buffer, lineterminator="\n")
    writer.writerow(
        ["relative_path", "size_bytes", "sha256", "asset_class", "source_or_generated"]
    )
    writer.writerow(["payload.txt", len(payload), expected_hash, "test", "generated"])
    manifest_csv = manifest_buffer.getvalue().encode("utf-8")
    manifest_json = json.dumps(
        {
            "schema_version": "test.manifest.v1",
            "package_id": "fixture",
            "manifest_entry_count": 1,
            "manifested_bytes": len(payload),
        }
    ).encode("utf-8")
    with tarfile.open(path, "w:gz") as tf:
        _add_tar_bytes(tf, "fixture/payload.txt", payload)
        _add_tar_bytes(tf, "fixture/FILE_MANIFEST.csv", manifest_csv)
        _add_tar_bytes(tf, "fixture/FILE_MANIFEST.json", manifest_json)


def test_parse_sha256_ledger_rejects_conflicting_duplicate(tmp_path: Path) -> None:
    ledger = tmp_path / "SHA256SUMS.txt"
    ledger.write_text(
        f"{'A' * 64}  artifact.zip\n{'B' * 64}  artifact.zip\n",
        encoding="utf-8",
    )

    with pytest.raises(ExpertAuditError, match="conflicting sha256 entries"):
        parse_sha256_ledger(ledger)


def test_manifested_tar_is_fully_extracted_and_verified(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    extract_root = tmp_path / "extract"
    _write_manifested_tar(archive)

    result = validate_manifested_tar(archive, extract_root=extract_root)

    assert result.summary["status"] == "PASS"
    assert result.summary["manifest_member_count"] == 1
    assert result.summary["regular_file_count"] == 3
    assert result.summary["extraction_result"] == "FULL_EXTRACT_PASS"
    assert (extract_root / "fixture" / "payload.txt").read_bytes() == b"verified payload\n"
    payload_row = next(row for row in result.members if row["member_path"].endswith("payload.txt"))
    assert payload_row["manifest_hash_match"] is True


def test_manifested_tar_rejects_wrong_payload_hash(tmp_path: Path) -> None:
    archive = tmp_path / "corrupt.tar.gz"
    _write_manifested_tar(archive, corrupt_hash=True)

    with pytest.raises(ExpertAuditError, match="manifest hash mismatch"):
        validate_manifested_tar(archive, extract_root=tmp_path / "extract")


def test_manifested_tar_subset_extracts_only_requested_verified_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "fixture.tar.gz"
    output = tmp_path / "subset"
    _write_manifested_tar(archive)

    result = extract_manifested_tar_subset(
        archive,
        output,
        relative_paths=["payload.txt"],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["selected_file_count"] == 1
    assert (output / "payload.txt").read_bytes() == b"verified payload\n"
    assert not (output / "FILE_MANIFEST.csv").exists()
    assert result.members[0]["manifest_hash_match"] is True


def test_manifested_tar_subset_is_fail_closed(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    _write_manifested_tar(archive)

    with pytest.raises(ExpertAuditError, match="not in package manifest"):
        extract_manifested_tar_subset(
            archive,
            tmp_path / "missing",
            relative_paths=["absent.py"],
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        extract_manifested_tar_subset(
            archive,
            existing,
            relative_paths=["payload.txt"],
        )


def test_tar_extraction_supports_long_nested_windows_paths(tmp_path: Path) -> None:
    archive = tmp_path / "long-path.tar.gz"
    long_member = "/".join(["nested_" + "x" * 72] * 4) + "/payload.txt"
    with tarfile.open(archive, "w:gz") as tf:
        _add_tar_bytes(tf, long_member, b"long path payload")

    result = validate_tar_archive(archive, extract_root=tmp_path / "extract")

    assert result.summary["status"] == "PASS"
    assert result.summary["extraction_result"] == "FULL_EXTRACT_PASS"
    _remove_extraction_tree(tmp_path / "extract")
    assert not (tmp_path / "extract").exists()


def test_archive_validation_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "unsafe")

    with pytest.raises(ExpertAuditError, match="unsafe archive member"):
        validate_zip_archive(archive, extract_root=tmp_path / "extract")


def test_zip_member_hash_can_satisfy_review_evidence_but_not_source_archive(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    output = tmp_path / "audit"
    downloads.mkdir()
    review_body = b"review evidence\n"
    with zipfile.ZipFile(downloads / "Stage1_BudgetedReplay_v1.0.0_Review_Evidence.zip", "w") as zf:
        zf.writestr("Stage1_BudgetedReplay_v1.0.0_Code_Evidence.txt", review_body)
    (downloads / "Stage1_BudgetedReplay_v1.0.0_Review_SHA256SUMS.txt").write_text(
        f"{_sha256(review_body)}  Stage1_BudgetedReplay_v1.0.0_Code_Evidence.txt\n"
        f"{_sha256((downloads / 'Stage1_BudgetedReplay_v1.0.0_Review_Evidence.zip').read_bytes())}  "
        "Stage1_BudgetedReplay_v1.0.0_Review_Evidence.zip\n",
        encoding="utf-8",
    )
    (downloads / "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0_SHA256SUMS.txt").write_text(
        f"{'A' * 64}  Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.tar.gz\n"
        f"{'B' * 64}  Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.zip\n"
        f"{'C' * 64}  stage1_budgeted_replay-1.0.0-py3-none-any.whl\n",
        encoding="utf-8",
    )

    receipt = audit_expert_deliveries(downloads, output)

    inventory = list(csv.DictReader((output / "expert_v1_inventory.csv").open(encoding="utf-8")))
    code_evidence = next(
        row for row in inventory if row["expected_filename"].endswith("Code_Evidence.txt")
    )
    source_tar = next(
        row
        for row in inventory
        if row["expected_filename"].endswith("Learnability_20260809_v1.0.0.tar.gz")
    )
    assert code_evidence["status"] == "PRESENT_AS_ARCHIVE_MEMBER_VERIFIED"
    assert source_tar["status"] == "REPORT_ONLY_SOURCE_MISSING"
    assert receipt["status"] == "INCOMPLETE_SOURCE_MISSING"
    assert receipt["source_level_audit_ready"] is False
    assert (output / "expert_v1_hash_validation.json").is_file()
    assert (output / "expert_archive_member_manifest.csv").is_file()
    assert "INCOMPLETE_SOURCE_MISSING" in (output / "README.md").read_text(encoding="utf-8")
    manifest = list(
        csv.DictReader((output / "expert_audit_output_manifest.csv").open(encoding="utf-8"))
    )
    assert {row["relative_path"] for row in manifest} == {
        "README.md",
        "expert_archive_member_manifest.csv",
        "expert_v1_hash_validation.json",
        "expert_v1_inventory.csv",
    }
    for row in manifest:
        artifact = output / row["relative_path"]
        assert int(row["size_bytes"]) == artifact.stat().st_size
        assert row["sha256"] == _sha256(artifact.read_bytes())


def test_audit_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads"
    output = tmp_path / "audit"
    downloads.mkdir()
    output.mkdir()
    (output / "expert_v1_inventory.csv").write_text("immutable\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit_expert_deliveries(downloads, output)
