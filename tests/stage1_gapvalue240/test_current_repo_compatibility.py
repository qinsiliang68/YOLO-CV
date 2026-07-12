from __future__ import annotations

import sys
import subprocess
from pathlib import Path

from stage1_gapvalue240.contract import load_contract
from stage1_gapvalue240.integration import trainer_command
from stage1_gapvalue240.machine import MachineConfig
from stage1_gapvalue240.predictor import _defect_index
from stage1_gapvalue240.runtime import ensure_ultralytics_assets, ensure_ultralytics_runtime
from stage1_gapvalue240.validation import _repository_audit


ROOT = Path(__file__).resolve().parents[2]


def test_trainer_adapter_targets_current_manifest_directory_cli(tmp_path):
    contract = load_contract(ROOT / "configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml")
    machine = MachineConfig(
        path=tmp_path / "machine.yaml",
        data={
            "machine_id": "local",
            "repo_root": str(ROOT),
            "dataset_root": str(ROOT / "data/final_sewerml_dataset"),
            "artifact_root": str(tmp_path / "artifact"),
            "output_root": str(tmp_path / "outputs"),
            "cache_root": str(tmp_path / "cache"),
            "gpu_id": 0,
            "num_workers": 0,
            "python_executable": sys.executable,
            "base_checkpoint": str(ROOT / "yolo11l-cls.pt"),
        },
    )
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    train = manifest_dir / "train_manifest.csv"
    normal = manifest_dir / "normal_train_manifest.csv"
    train.touch()
    normal.touch()
    (manifest_dir / "val_model_manifest.csv").touch()
    (manifest_dir / "normal_val_model_manifest.csv").touch()

    cmd = trainer_command(contract, machine, train, normal, tmp_path / "trainer", 123)

    assert "--manifest-dir" in cmd
    assert "--runs-root" in cmd
    assert "--work-root" in cmd
    assert "--dataset-root" in cmd
    assert "--yolo-root" in cmd
    assert "--train-manifest" not in cmd
    assert "--normal-train-manifest" not in cmd
    assert "--weights" not in cmd


def test_current_stage1_class_name_is_accepted_as_defect():
    assert _defect_index({0: "no_target", 1: "target_defect"}, ["defect", "1"]) == 1


def test_documented_direct_cli_entrypoints_can_import_overlay_package():
    scripts = [
        ROOT / "scripts/stage1_gapvalue240/validate_package.py",
        ROOT / "scripts/stage1_gapvalue240/prepare_experiment.py",
        ROOT / "scripts/stage1_gapvalue240/runs/run_001.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout


def test_ultralytics_runtime_font_is_valid(tmp_path):
    from matplotlib.ft2font import FT2Font

    config_dir = ensure_ultralytics_runtime(tmp_path / "config")
    font = config_dir / "Arial.ttf"
    assert font.exists()
    FT2Font(str(font))


def test_ultralytics_assets_bootstrap_from_pinned_venv(tmp_path):
    from PIL import Image

    yolo_root = tmp_path / "AI/repos/current/YOLOv11"
    source = tmp_path / "AI/venvs/yolo-cv/Lib/site-packages/ultralytics/assets"
    source.mkdir(parents=True)
    for filename, color in (("bus.jpg", "red"), ("zidane.jpg", "blue")):
        Image.new("RGB", (8, 8), color=color).save(source / filename)
    report = ensure_ultralytics_assets(yolo_root)
    assert set(report) == {"bus.jpg", "zidane.jpg"}
    assert all(Path(item["path"]).is_file() for item in report.values())


def test_repository_audit_accepts_overlay_descendant_of_frozen_base():
    contract = load_contract(ROOT / "configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml")
    audit = _repository_audit(ROOT, contract)
    assert audit["commit_ok"]
    assert audit["contract_base_commit"].startswith(str(contract.data["repository"]["commit"]))
