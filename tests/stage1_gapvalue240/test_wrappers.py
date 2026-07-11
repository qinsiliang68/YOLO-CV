from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[2]

def test_240_independent_run_modules():
    files=sorted((ROOT/'scripts/stage1_gapvalue240/runs').glob('run_*.py'))
    assert len(files)==240
    for i in [1,2,120,240]:
        p=ROOT/f'scripts/stage1_gapvalue240/runs/run_{i:03d}.py'
        spec=importlib.util.spec_from_file_location(f'r{i}',p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        assert mod.RUN_SLOT==f'RUN_{i:03d}'
        for name in ['prepare','train','evaluate','validate','run','main']: assert callable(getattr(mod,name))
