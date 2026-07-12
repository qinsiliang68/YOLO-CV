from pathlib import Path
import pytest,yaml
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.errors import ConfigurationError


ROOT = Path(__file__).resolve().parents[2]

def test_science_keys_rejected(tmp_path):
    p=tmp_path/'m.yaml'; p.write_text(yaml.safe_dump({'machine_id':'x','repo_root':'.','dataset_root':'.','artifact_root':'.','output_root':'.','cache_root':'.','gpu_id':0,'num_workers':1,'method':'GapCritical'}))
    with pytest.raises(ConfigurationError): load_machine_config(p)


def test_all_formal_machine_templates_lock_four_training_workers():
    paths = sorted((ROOT / "configs/stage1_gapvalue240/machines").glob("machine_*.yaml"))
    assert len(paths) == 12
    assert all(int(yaml.safe_load(path.read_text(encoding="utf-8"))["num_workers"]) == 4 for path in paths)
