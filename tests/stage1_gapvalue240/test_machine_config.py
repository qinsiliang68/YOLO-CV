from pathlib import Path
import pytest,yaml
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.errors import ConfigurationError

def test_science_keys_rejected(tmp_path):
    p=tmp_path/'m.yaml'; p.write_text(yaml.safe_dump({'machine_id':'x','repo_root':'.','dataset_root':'.','artifact_root':'.','output_root':'.','cache_root':'.','gpu_id':0,'num_workers':1,'method':'GapCritical'}))
    with pytest.raises(ConfigurationError): load_machine_config(p)
