from __future__ import annotations

import csv
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.asset_identity import (
    AssetIdentityError,
    build_live_dataset_identity,
    load_manifest_records,
    validate_live_dataset_identity,
)


def _manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["canonical_image_relpath", "Filename", "Defect"])
        writer.writeheader()
        writer.writerows(rows)


def test_live_image_bytes_are_bound_and_revalidated(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "defect").mkdir(parents=True)
    (root / "normal").mkdir(parents=True)
    (root / "defect" / "d.png").write_bytes(b"defect-bytes")
    (root / "normal" / "n.png").write_bytes(b"normal-bytes")
    defect = tmp_path / "defect.csv"
    normal = tmp_path / "normal.csv"
    _manifest(defect, [{"canonical_image_relpath": "defect/d.png", "Filename": "d.png", "Defect": 1}])
    _manifest(normal, [{"canonical_image_relpath": "normal/n.png", "Filename": "n.png", "Defect": 0}])

    records = load_manifest_records(
        dataset_root=root,
        defect_manifest=defect,
        normal_manifest=normal,
        expected_defect=1,
        expected_normal=1,
    )
    identity = build_live_dataset_identity(records)
    assert validate_live_dataset_identity(records, expected_digest=identity.image_content_digest).status == "PASS"

    (root / "normal" / "n.png").write_bytes(b"changed-bytes")
    with pytest.raises(AssetIdentityError, match="image content"):
        validate_live_dataset_identity(records, expected_digest=identity.image_content_digest)


def test_manifest_path_escape_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.csv"
    _manifest(manifest, [{"canonical_image_relpath": "../outside.png", "Filename": "outside.png", "Defect": 1}])
    with pytest.raises(AssetIdentityError, match="escapes"):
        load_manifest_records(
            dataset_root=tmp_path / "data",
            defect_manifest=manifest,
            normal_manifest=tmp_path / "empty.csv",
            expected_defect=1,
            expected_normal=0,
        )
