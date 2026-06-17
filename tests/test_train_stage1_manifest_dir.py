from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_stage1_cls_sweep import build_paths  # noqa: E402


def test_build_paths_allows_manifest_dir_override(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_dir = tmp_path / "fold_00" / "manifests"
    args = argparse.Namespace(
        yolo_root=None,
        dataset_root=str(dataset_root),
        manifest_dir=str(manifest_dir),
        work_root=None,
        runs_root=None,
    )

    paths = build_paths(args)

    assert paths.dataset_root == dataset_root.resolve()
    assert paths.manifest_dir == manifest_dir.resolve()
