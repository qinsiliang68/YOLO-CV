from __future__ import annotations

import importlib.util
from pathlib import Path

from stage1_gapvalue240.machine import MachineConfig


ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/stage1_gapvalue240/local_resource_smoke.py"
    spec = importlib.util.spec_from_file_location("gapvalue_local_resource_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resource_smoke_uses_machine_output_and_staging_volumes(tmp_path):
    module = _module()
    machine = MachineConfig(
        path=tmp_path / "machine.yaml",
        data={
            "output_root": str(tmp_path / "output"),
            "staging_root": str(tmp_path / "staging/machine_01"),
        },
    )
    assert module._default_output_root(machine) == (tmp_path / "output/runtime_validation/local_resource_smoke").resolve()
    assert module._temporary_parent(machine) == (tmp_path / "staging/.resource_smoke_tmp").resolve()
