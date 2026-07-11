from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_long_lived_shard_controller_does_not_import_torch_or_ultralytics():
    script = ROOT / "scripts/stage1_gapvalue240/run_machine_shard.py"
    code = (
        "import runpy,sys; "
        f"runpy.run_path({str(script)!r}, run_name='stage1_shard_import_test'); "
        "print(int('torch' in sys.modules), int('ultralytics' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.strip().endswith("0 0"), result.stdout
