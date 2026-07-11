from pathlib import Path

import pandas as pd

from stage1_gapvalue240.queue_validation import validate_frozen_queues, write_frozen_queue_manifest
from stage1_gapvalue240.util import sha256_file


def test_all_selection_csvs_and_machine_shards_are_frozen_before_training(tmp_path):
    artifact = tmp_path / "artifact"
    selections = artifact / "generated/selections"
    rows = []
    index = []
    for triad in range(1, 3):
        for arm_index, arm in enumerate(("T", "R1", "R2"), start=1):
            run_number = (triad - 1) * 3 + arm_index
            run_slot = f"RUN_{run_number:03d}"
            rows.append({"run_slot": run_slot, "triad_id": f"TRIAD_{triad:03d}", "arm": arm, "budget": 1})
            path = selections / run_slot / "selection_manifest.csv"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [{"run_slot": run_slot, "rank": 1, "sample_id": f"sample_{run_number}", "y_true": 0,
                  "replay_role": "normal_replay"}]
            ).to_csv(path, index=False)
            index.append({"run_slot": run_slot, "selection_manifest": path.relative_to(artifact).as_posix(), "sha256": sha256_file(path)})
    matrix = artifact / "generated/frozen_experiment_matrix.csv"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(matrix, index=False)
    index_path = artifact / "generated/selection_index.csv"
    pd.DataFrame(index).to_csv(index_path, index=False)
    shards = artifact / "generated/machine_shards"
    shards.mkdir()
    pd.DataFrame(rows[:3]).to_csv(shards / "machine_01_jobs.csv", index=False)
    pd.DataFrame(rows[3:]).to_csv(shards / "machine_02_jobs.csv", index=False)
    pd.DataFrame(rows).iloc[:0].to_csv(shards / "machine_03_jobs.csv", index=False)

    report = validate_frozen_queues(
        matrix, index_path, artifact, shards, tmp_path / "queue_validation.json",
        expected_runs=6, expected_triads=2, main_machine_count=2, reserve_machine_count=1,
    )

    assert report["status"] == "PASS"
    assert report["selection_manifest_count"] == 6
    assert report["shard_run_count"] == 6
    manifest_path = write_frozen_queue_manifest(artifact / "generated", artifact / "generated/FROZEN_QUEUE_FILE_MANIFEST.csv")
    manifest = pd.read_csv(manifest_path)
    assert len(manifest[manifest.relative_path.str.endswith("selection_manifest.csv")]) == 6
    assert "oof_cache" not in "\n".join(manifest.relative_path.astype(str))
