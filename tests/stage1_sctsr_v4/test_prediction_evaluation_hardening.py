from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json

import pytest
import torch

from stage1_sctsr_v4.asset_registry import AssetRegistry, build_split_identity_bundle
from stage1_sctsr_v4.checkpointing import build_checkpoint_payload, save_checkpoint_atomic
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.evaluation import compute_tie_safe_frontier, validate_checkpoint_for_evaluation, write_frontier_artifacts
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.prediction_artifact import PredictionArtifactBinding, validate_prediction_rows, write_prediction_artifact
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file


class _Scaler:
    def state_dict(self):
        return {}


def _bound_formal_prediction(tmp_path, prediction_rows):
    selected = prediction_rows[:12]
    base_manifest = tmp_path / "base.csv"
    val_op_manifest = tmp_path / "val_op.csv"
    for path, rows, split_role in (
        (base_manifest, (("BASE_ONLY", 0),), "train"),
        (val_op_manifest, tuple((row.sample_id, row.y_true) for row in selected), "val_op"),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "y_true", "split"))
            writer.writeheader()
            writer.writerows({"sample_id": sample_id, "y_true": label, "split": split_role} for sample_id, label in rows)

    def digest(rows):
        hasher = hashlib.sha256()
        for sample_id, label in sorted(rows):
            hasher.update(f"{sample_id}|{label}\n".encode("utf-8"))
        return hasher.hexdigest().upper()

    def record(path, asset_id, role, rows, split_role, universe):
        return {
            "asset_id": asset_id,
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
            "required": True,
            "row_count": len(rows),
            "identity_column": "sample_id",
            "label_column": "y_true",
            "identity_digest": digest(rows),
            "identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
            "split_column": "split",
            "allowed_split_values": [split_role],
            "identity_universe": universe,
            "mutual_exclusion_group": "TRAIN_VALIDATION_DISJOINT",
        }

    base_rows = (("BASE_ONLY", 0),)
    val_op_rows = tuple((row.sample_id, row.y_true) for row in selected)
    raw_registry = {
        "schema_version": "stage1.sctsr.asset_registry.v1",
        "base_denominator": 1,
        "base_identity_digest": digest(base_rows),
        "base_identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
        "base_asset_ids": ["base"],
        "split_asset_ids": {"val_op": ["val_op"]},
        "val_target_available": False,
        "assets": [
            record(base_manifest, "base", "CANONICAL_BASE_MANIFEST", base_rows, "train", "CANONICAL_BASE_FULL"),
            record(val_op_manifest, "val_op", "VAL_OP_ENDPOINT_ONLY", val_op_rows, "val_op", "VAL_OP_COMPONENT"),
        ],
    }
    registry_path = tmp_path / "asset_registry.json"
    atomic_write_json(registry_path, raw_registry)
    registry = AssetRegistry.from_mapping(raw_registry)
    split = tmp_path / "VAL_OP_SPLIT_BUNDLE.json"
    build_split_identity_bundle(
        registry,
        repository_root=tmp_path,
        registry_path=registry_path,
        split_role="val_op",
        output_path=split,
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    payload = build_checkpoint_payload(
        model=model,
        ema=ExponentialMovingAverage.from_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=_Scaler(),
        epoch=200,
        global_step=187_600,
        base_sampler_generation=200,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        training_seed=1,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest=registry.digest,
    )
    checkpoint = tmp_path / "epoch_0200.pt"
    checkpoint_sha = save_checkpoint_atomic(checkpoint, payload)
    split_sha = sha256_file(split)
    rows = tuple(
        replace(
            row,
            split_role="val_op",
            split_manifest_path=split.as_posix(),
            split_manifest_sha256=split_sha,
            checkpoint_sha256=checkpoint_sha,
            source_tree_digest="D" * 64,
            model_variant="MODEL",
        )
        for row in selected
    )
    binding = PredictionArtifactBinding(
        checkpoint_path=checkpoint.as_posix(),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_epoch=200,
        model_variant="MODEL",
        source_tree_digest="D" * 64,
        training_seed=1,
        split_role="val_op",
        split_manifest_path=split.as_posix(),
        split_manifest_sha256=split_sha,
        evaluation_mode="formal",
        selection_semantic="ENDPOINT_ONLY_NOT_FOR_SELECTION",
    )
    return rows, binding, checkpoint, split


def test_prediction_rows_require_one_bound_checkpoint_and_split(prediction_rows):
    mixed = list(prediction_rows)
    mixed[-1] = replace(
        mixed[-1],
        checkpoint_sha256="D" * 64,
        split_manifest_sha256="E" * 64,
        source_tree_digest="F" * 64,
    )
    with pytest.raises(SctsrError) as caught:
        validate_prediction_rows(tuple(mixed))
    assert caught.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def test_prediction_probability_must_match_two_class_logits(prediction_rows):
    inconsistent = (replace(prediction_rows[0], p_defect_raw=0.999),) + prediction_rows[1:]
    with pytest.raises(SctsrError) as caught:
        validate_prediction_rows(inconsistent)
    assert caught.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def test_formal_prediction_write_rejects_unverified_manifest_and_checkpoint(tmp_path, prediction_rows):
    with pytest.raises(SctsrError) as caught:
        write_prediction_artifact(
            prediction_rows,
            tmp_path / "run_id=R" / "epoch=0200" / "predictions.parquet",
            formal_endpoint=True,
        )
    assert caught.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def test_checkpoint_evaluation_requires_existing_loadable_bound_checkpoint(tmp_path):
    with pytest.raises(SctsrError) as caught:
        validate_checkpoint_for_evaluation(
            tmp_path / "epoch_0200.pt",
            epoch=200,
            mode="formal",
        )
    assert caught.value.code in {ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, ErrorCode.PARENT_SHA_MISMATCH}


def test_frontier_rejects_unregistered_artifact_bindings(prediction_rows):
    with pytest.raises(SctsrError) as caught:
        compute_tie_safe_frontier(prediction_rows)
    assert caught.value.code is ErrorCode.FRONTIER_INVALID


def test_prediction_test_role_is_always_forbidden(prediction_rows):
    rows = tuple(replace(row, split_role="test") for row in prediction_rows)
    with pytest.raises(SctsrError) as caught:
        validate_prediction_rows(rows)
    assert caught.value.code is ErrorCode.TEST_ACCESS_FORBIDDEN


def test_formal_prediction_is_bound_to_loadable_checkpoint_and_complete_split(tmp_path, prediction_rows):
    rows, binding, _, _ = _bound_formal_prediction(tmp_path, prediction_rows)
    output = tmp_path / "run_id=R" / "epoch=0200" / "predictions.parquet"
    manifest, summary = write_prediction_artifact(
        rows,
        output,
        formal_endpoint=True,
        binding=binding,
        expected_sample_labels={row.sample_id: row.y_true for row in rows},
        repository_root=tmp_path,
    )
    assert manifest.canonical_parquet
    assert summary["row_count"] == len(rows)
    assert summary["selection_semantic"] == "ENDPOINT_ONLY_NOT_FOR_SELECTION"


def test_formal_prediction_rejects_missing_extra_or_wrong_split_rows(tmp_path, prediction_rows):
    rows, binding, _, _ = _bound_formal_prediction(tmp_path, prediction_rows)
    for mutated in (
        rows[:-1],
        rows + (replace(rows[-1], sample_id="EXTRA"),),
        (replace(rows[0], y_true=1 - rows[0].y_true),) + rows[1:],
    ):
        with pytest.raises(SctsrError) as caught:
            write_prediction_artifact(
                mutated,
                tmp_path / "run_id=R" / "epoch=0200" / f"{len(mutated)}-{mutated[0].sample_id}.parquet",
                formal_endpoint=True,
                binding=binding,
                repository_root=tmp_path,
            )
        assert caught.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def test_formal_prediction_rejects_arbitrary_json_claiming_val_op(tmp_path, prediction_rows):
    rows, binding, _, _ = _bound_formal_prediction(tmp_path, prediction_rows)
    fake = tmp_path / "fake_val_op.json"
    fake.write_text(json.dumps({"rows": [{"sample_id": row.sample_id, "y_true": row.y_true} for row in rows]}), encoding="utf-8")
    fake_binding = replace(binding, split_manifest_path=fake.as_posix(), split_manifest_sha256=sha256_file(fake))
    fake_rows = tuple(
        replace(row, split_manifest_path=fake.as_posix(), split_manifest_sha256=sha256_file(fake))
        for row in rows
    )
    with pytest.raises(SctsrError) as caught:
        write_prediction_artifact(
            fake_rows,
            tmp_path / "run_id=R" / "epoch=0200" / "fake.parquet",
            formal_endpoint=True,
            binding=fake_binding,
            repository_root=tmp_path,
        )
    assert caught.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def test_checkpoint_binding_rejects_wrong_sha(tmp_path, prediction_rows):
    _, binding, checkpoint, _ = _bound_formal_prediction(tmp_path, prediction_rows)
    with pytest.raises(SctsrError) as caught:
        validate_checkpoint_for_evaluation(
            checkpoint,
            epoch=200,
            mode="formal",
            expected_sha256="0" * 64,
        )
    assert caught.value.code is ErrorCode.PARENT_SHA_MISMATCH


def test_tie_safe_frontier_golden_points_and_independent_anchors(prediction_rows):
    probabilities = (0.9, 0.9, 0.8, 0.7, 0.1)
    labels = (1, 0, 1, 0, 0)
    rows = tuple(
        replace(
            prediction_rows[index],
            y_true=labels[index],
            logit_normal=0.0,
            logit_defect=__import__("math").log(probability / (1.0 - probability)),
            p_defect_raw=probability,
        )
        for index, probability in enumerate(probabilities)
    )
    points, summary = compute_tie_safe_frontier(
        rows,
        max_fn=2,
        target_tn=3,
        checkpoint_sha256="B" * 64,
        prediction_artifact_sha256="D" * 64,
    )
    assert [(point.fn_budget, point.actual_fn, point.tn, point.tie_size) for point in points] == [
        (0, 0, 2, 1),
        (1, 1, 2, 2),
        (2, 2, 3, 0),
    ]
    assert summary.raw_frontier_normalized_auc == pytest.approx(0.75)
    assert summary.fn_at_tn68253 == 2
    assert summary.threshold_at_tn68253 != points[1].threshold


def test_formal_frontier_writer_requires_all_96_points(tmp_path, prediction_rows):
    points, summary = compute_tie_safe_frontier(
        prediction_rows,
        max_fn=10,
        target_tn=20,
        checkpoint_sha256="B" * 64,
        prediction_artifact_sha256="D" * 64,
    )
    with pytest.raises(SctsrError) as caught:
        write_frontier_artifacts(
            points,
            summary,
            frontier_path=tmp_path / "run_id=R" / "epoch=0200" / "frontier.parquet",
            summary_path=tmp_path / "summary.json",
            evaluation_mode="formal",
            split_role="val_op",
            checkpoint_epoch=200,
        )
    assert caught.value.code is ErrorCode.FRONTIER_INVALID
