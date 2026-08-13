from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_published_evidence_manifest.py")
VERIFY = Path(__file__).with_name("verify_staged_evidence_bytes.py")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load_module():
    spec = importlib.util.spec_from_file_location("published_evidence_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_uses_git_index_and_excludes_local_only_bytes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    evidence = repository / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "reports").mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    (evidence / "tracked.txt").write_text("old\n", encoding="utf-8")
    (evidence / "EVIDENCE_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (evidence / "reports/STAGED_EVIDENCE_BYTES_VALIDATION.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "evidence"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "base"], check=True)

    (evidence / "tracked.txt").write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "evidence/tracked.txt"], check=True)
    (evidence / "tracked.txt").write_text("unstaged drift\n", encoding="utf-8")
    (evidence / "local-checkpoint.pt").write_bytes(b"local-only")

    module = _load_module()
    payload = module.build_published_manifest(
        repository_root=repository,
        evidence_root=evidence,
        implementation_source_commit="a" * 40,
        output=evidence / "EVIDENCE_MANIFEST.json",
    )

    assert payload["publication_scope"] == "GIT_INDEX_ONLY"
    assert [row["relative_path"] for row in payload["files"]] == ["tracked.txt"]
    assert payload["files"][0]["bytes"] == len(b"staged\n")
    assert "local-checkpoint.pt" not in json.dumps(payload)
    assert _git(repository, "show", ":evidence/tracked.txt") == "staged"

    subprocess.run(["git", "-C", str(repository), "add", "evidence/EVIDENCE_MANIFEST.json"], check=True)
    validation = evidence / "reports/STAGED_EVIDENCE_BYTES_VALIDATION.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--repository-root",
            str(repository),
            "--evidence-root",
            str(evidence),
            "--manifest",
            str(evidence / "EVIDENCE_MANIFEST.json"),
            "--output",
            str(validation),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(validation.read_text(encoding="utf-8"))["status"] == "PASS"
