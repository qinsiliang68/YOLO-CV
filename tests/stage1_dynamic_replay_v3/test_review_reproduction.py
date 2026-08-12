from __future__ import annotations

import json
from pathlib import Path

from stage1_dynamic_replay_v3.review_reproduction import build_v3_p0_reproduction


def test_v3_p0_reproduction_records_resolved_and_remaining_findings(tmp_path: Path) -> None:
    output = tmp_path / "V3_P0_REPRODUCTION.json"

    payload = build_v3_p0_reproduction(Path.cwd(), output)

    assert payload["status"] == "PASS_AUDIT_REPRODUCTION"
    assert payload["executed_expert_code"] is False
    assert payload["p0_01_oof"]["fold_is_not_a_global_trajectory_axis"] is True
    assert payload["p0_01_oof"]["path_fold_mismatch_accepted"] is True
    assert payload["p0_02_checkpoint"]["best_pt_references"] == []
    assert payload["p0_03_data_roles"]["val_target_references"] == []
    assert payload["p0_04_test_oracle"]["test_oracle_references"] == []
    assert payload["p0_05_dual_threshold"]["independent_metric_demonstrated"] is True
    assert payload["p0_05_dual_threshold"]["fn_at_target_tn"] != payload["p0_05_dual_threshold"][
        "fn_at_fn_limit_point"
    ]
    assert payload["p0_06_critical_cld"]["critical_cld_references"] == []
    assert payload["p0_07_matrix"]["run_count"] == 236
    assert payload["p0_07_matrix"]["all_release_states_held"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_v3_p0_reproduction_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "V3_P0_REPRODUCTION.json"
    output.write_text("immutable\n", encoding="utf-8")

    try:
        build_v3_p0_reproduction(Path.cwd(), output)
    except FileExistsError as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        raise AssertionError("audit reproduction overwrote immutable evidence")
