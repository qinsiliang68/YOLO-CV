from __future__ import annotations

from copy import deepcopy

from stage1_sctsr_v4.formal_training import _resume_stable_trainer_binding_value
from stage1_sctsr_v4.run_validation import _resume_trainer_binding_field_matches
from stage1_sctsr_v4.serialization import stable_digest


def _dataset_binding() -> dict[str, object]:
    result: dict[str, object] = {
        "dataset_binding_digest": "A" * 64,
        "train_rows": 120_000,
        "val_model_rows": 23_996,
    }
    for role, rows, content_sha in (
        ("train", 120_000, "B" * 64),
        ("val_model", 23_996, "C" * 64),
    ):
        core = {
            "schema_version": "stage1.sctsr.materialized_dataset_binding.v4",
            "status": "PASS",
            "role": role,
            "row_count": rows,
            "materialized_content_digest": content_sha,
            "evidence": {
                "path": f"D:/formal/original/{role}_materialized_files.parquet",
                "bytes": rows * 17,
                "sha256": content_sha,
                "row_count": rows,
                "schema_version": "stage1.sctsr.materialized_dataset_rows.v3",
                "schema_digest": "D" * 64,
                "compression": "ZSTD",
            },
        }
        result[f"{role}_materialized_content_binding"] = {
            **core,
            "binding_digest": stable_digest(core),
        }
    return result


def test_resume_dataset_identity_allows_only_relocated_evidence_ledgers() -> None:
    original = _dataset_binding()
    resumed = deepcopy(original)
    for field in (
        "train_materialized_content_binding",
        "val_model_materialized_content_binding",
    ):
        binding = resumed[field]
        assert isinstance(binding, dict)
        evidence = binding["evidence"]
        assert isinstance(evidence, dict)
        evidence["path"] = evidence["path"].replace("/original/", "/10_resume_setup/epoch_0004.generation_1/")
        binding["binding_digest"] = stable_digest({key: value for key, value in binding.items() if key != "binding_digest"})

    assert _resume_stable_trainer_binding_value("dataset_binding", original) == _resume_stable_trainer_binding_value(
        "dataset_binding", resumed
    )
    assert _resume_trainer_binding_field_matches(
        {"dataset_binding": original},
        {"dataset_binding": resumed},
        "dataset_binding",
    )


def test_resume_dataset_identity_still_rejects_changed_evidence_or_content() -> None:
    original = _dataset_binding()
    changed_evidence = deepcopy(original)
    changed_train = changed_evidence["train_materialized_content_binding"]
    assert isinstance(changed_train, dict)
    changed_train_evidence = changed_train["evidence"]
    assert isinstance(changed_train_evidence, dict)
    changed_train_evidence["sha256"] = "E" * 64
    changed_train["binding_digest"] = stable_digest(
        {key: value for key, value in changed_train.items() if key != "binding_digest"}
    )
    assert _resume_stable_trainer_binding_value("dataset_binding", original) != _resume_stable_trainer_binding_value(
        "dataset_binding", changed_evidence
    )
    assert not _resume_trainer_binding_field_matches(
        {"dataset_binding": original},
        {"dataset_binding": changed_evidence},
        "dataset_binding",
    )

    changed_content = deepcopy(original)
    changed_val = changed_content["val_model_materialized_content_binding"]
    assert isinstance(changed_val, dict)
    changed_val["materialized_content_digest"] = "F" * 64
    changed_val["binding_digest"] = stable_digest(
        {key: value for key, value in changed_val.items() if key != "binding_digest"}
    )
    assert _resume_stable_trainer_binding_value("dataset_binding", original) != _resume_stable_trainer_binding_value(
        "dataset_binding", changed_content
    )
