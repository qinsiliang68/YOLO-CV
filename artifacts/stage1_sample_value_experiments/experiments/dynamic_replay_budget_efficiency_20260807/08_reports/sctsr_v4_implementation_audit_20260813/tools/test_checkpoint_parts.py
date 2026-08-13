from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("checkpoint_parts.py")


def _module():
    spec = importlib.util.spec_from_file_location("checkpoint_parts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_parts_round_trip_and_detect_corruption(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "checkpoint.pt"
    payload = bytes(range(251)) * 97
    source.write_bytes(payload)

    manifest = module.split_file(source, tmp_path / "parts", part_size=7000)
    assert manifest["original_bytes"] == len(payload)
    assert len(manifest["parts"]) == 4
    assert all(row["bytes"] <= 7000 for row in manifest["parts"])
    verification = module.verify_parts(tmp_path / "parts/CHECKPOINT_PARTS_MANIFEST.json")
    assert verification["status"] == "PASS"
    assert verification["sha256"] == manifest["original_sha256"]

    restored = tmp_path / "restored.pt"
    receipt = module.reassemble_file(tmp_path / "parts/CHECKPOINT_PARTS_MANIFEST.json", restored)
    assert receipt["status"] == "PASS"
    assert restored.read_bytes() == payload

    first = tmp_path / "parts" / manifest["parts"][0]["filename"]
    first.write_bytes(b"corrupt" + first.read_bytes()[7:])
    rejected = tmp_path / "rejected.pt"
    with pytest.raises(ValueError, match="part SHA-256 mismatch"):
        module.reassemble_file(tmp_path / "parts/CHECKPOINT_PARTS_MANIFEST.json", rejected)
    assert not rejected.exists()
