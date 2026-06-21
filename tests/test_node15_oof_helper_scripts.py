from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_prepare_script_keeps_existing_staged_files_until_replacement_is_ready() -> None:
    script = read_script("prepare_stage1_oof_node15_upload_20260621.ps1")

    assert "function New-StagedHardLink" in script
    assert "function Move-StagedFile" in script
    assert ".backup-" in script
    assert "Assert-Sha256SidecarMatches" in script


def test_prepare_manifest_does_not_hash_itself_while_being_written() -> None:
    script = read_script("prepare_stage1_oof_node15_upload_20260621.ps1")

    assert "Where-Object { $_.Name -ne 'UPLOAD_MANIFEST.csv' }" in script


def test_cleanup_script_requires_remote_sha_marker_before_deleting() -> None:
    script = read_script("cleanup_stage1_oof_node15_after_upload_20260621.ps1")

    assert "ConfirmedRemoteSha256" in script
    assert "REMOTE_UPLOAD_SHA256_VERIFIED=YES" in script
    assert "Assert-RemoteUploadVerified" in script


def test_cleanup_script_uses_root_bounded_path_checks() -> None:
    script = read_script("cleanup_stage1_oof_node15_after_upload_20260621.ps1")

    assert "function Assert-PathInsideRoot" in script
    assert "Get-FullPathWithTrailingSeparator" in script
    assert "StartsWith($archiveFull" not in script
    assert "StartsWith($base" not in script
