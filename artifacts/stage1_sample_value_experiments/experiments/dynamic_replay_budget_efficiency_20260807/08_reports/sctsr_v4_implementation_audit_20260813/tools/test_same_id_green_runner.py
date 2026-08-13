from __future__ import annotations

import base64
import sys
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_same_id_green import SameIdGreenError, materialize_and_run


def test_materializes_sha_bound_red_source_and_runs_same_node_id_green(tmp_path):
    source = b"def test_exact_identity():\n    assert True\n"
    encoded = tmp_path / "test_exact.py.b64"
    encoded.write_text(base64.b64encode(source).decode("ascii") + "\n", encoding="ascii")
    output = tmp_path / "green"
    result = materialize_and_run(
        sources=(
            {
                "encoded_path": encoded,
                "decoded_relative_path": "tests/test_exact.py",
                "decoded_sha256": sha256(source).hexdigest().upper(),
            },
        ),
        test_ids=("tests/test_exact.py::test_exact_identity",),
        repository_root=tmp_path,
        output_root=output,
    )
    assert result["status"] == "PASS"
    assert result["exit_code"] == 0
    assert result["test_ids"] == ["tests/test_exact.py::test_exact_identity"]
    assert "1 passed" in (output / "GREEN.stdout.log").read_text(encoding="utf-8")
    assert (output / "GREEN.junit.xml").is_file()


def test_rejects_decoded_source_sha_mismatch_before_pytest(tmp_path):
    source = b"def test_exact_identity():\n    assert True\n"
    encoded = tmp_path / "test_exact.py.b64"
    encoded.write_text(base64.b64encode(source).decode("ascii"), encoding="ascii")
    with pytest.raises(SameIdGreenError, match="decoded source SHA"):
        materialize_and_run(
            sources=(
                {
                    "encoded_path": encoded,
                    "decoded_relative_path": "tests/test_exact.py",
                    "decoded_sha256": "0" * 64,
                },
            ),
            test_ids=("tests/test_exact.py::test_exact_identity",),
            repository_root=tmp_path,
            output_root=tmp_path / "green",
        )
