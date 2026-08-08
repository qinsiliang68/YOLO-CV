from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import pytest

from scripts.stage1_gapvalue240 import build_dynamic_replay_preregistration

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=(
    'aggregate_coordination_root_canary.py',
    'aggregate_ten_machine_real_data_canary.py',
    'bind_dynamic_campaign_validation_evidence.py',
    'build_coordination_root_canary_commands.py',
    'build_daily_campaign_status.py',
    'build_dynamic_campaign_engineering_gate_v2.py',
    'build_epoch_resource_log.py',
    'build_ten_machine_real_data_canary_commands.py',
    'dry_generate_dynamic_campaign_v2.py',
    'run_coordination_root_canary.py',
    'run_disk_gpu_preflight.py',
    'validate_all_epoch_telemetry.py',
    'validate_campaign_failure_recovery.py',
    'validate_campaign_leases.py',
    'validate_cycle_closeout.py',
    'validate_documentation_handoff.py',
    'validate_dynamic_campaign_contracts.py',
    'validate_dynamic_campaign_report_schema.py',
)

@pytest.mark.parametrize('name',SCRIPTS)
def test_new_cli_can_run_directly_from_repository(name: str) -> None:
    result=subprocess.run(
        [sys.executable,str(ROOT/'scripts/stage1_gapvalue240'/name),'--help'],
        cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30,
    )
    assert result.returncode==0, result.stdout
    assert 'usage:' in result.stdout.lower()


def test_preregistration_cli_defaults_to_the_full_frozen_gapcritical_ranking() -> None:
    args = build_dynamic_replay_preregistration.parse_args([])
    source = Path(args.treatment_ranking_source)

    assert source.name == "GapCritical-Strict.csv"
    assert source.parent.name == "rankings"
    assert "precomputed_direct_assets" in source.parts


def test_active_campaign_clis_do_not_hardcode_legacy_v1_stage_paths() -> None:
    active_clis = (
        "build_dynamic_replay_preregistration.py",
        "build_dynamic_campaign_run_queue.py",
        "build_dynamic_campaign_release_gates.py",
        "build_dynamic_campaign_assignment.py",
        "activate_dynamic_campaign_assignment.py",
        "build_dynamic_campaign_gradient_candidates.py",
        "run_dynamic_campaign_controller.py",
    )

    for name in active_clis:
        source = (ROOT / "scripts/stage1_gapvalue240" / name).read_text(encoding="utf-8")
        assert 'campaign / "03_preregistration"' not in source, name
        assert 'campaign / "04_run_queue"' not in source, name
        assert 'campaign_root / "04_run_queue"' not in source, name
        assert '"03_preregistration/frozen' not in source, name
        assert '"04_run_queue/' not in source, name
