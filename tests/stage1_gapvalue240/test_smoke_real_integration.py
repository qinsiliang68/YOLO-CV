from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/stage1_gapvalue240/smoke_real_integration.py"
    spec = importlib.util.spec_from_file_location("gapvalue_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_plan_is_five_runs_and_includes_formal_l():
    module = _module()
    assert len(module.SMOKE_SPECS) == 5
    assert [item[1] for item in module.SMOKE_SPECS] == ["n", "n", "n", "n", "l"]
    assert {item[2][0] for item in module.SMOKE_SPECS} == {"n", "guard"}
