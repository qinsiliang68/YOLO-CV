from __future__ import annotations

from dataclasses import replace
import csv
import hashlib

import pytest
import torch
from PIL import Image
import numpy as np

from stage1_sctsr_v4.checkpointing import build_checkpoint_payload, save_checkpoint_atomic
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.prediction_artifact import PredictionArtifactBinding
from stage1_sctsr_v4.asset_registry import AssetRegistry
from stage1_sctsr_v4.prediction_runtime import (
    PredictionBatch,
    infer_prediction_rows,
    load_registered_image_records,
    publish_formal_endpoint,
)
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file


class _Scaler:
    def state_dict(self):
        return {}


class _BinaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.names = {0: "no_target", 1: "target_defect"}

    def forward(self, images):
        logits = self.linear(images)
        return logits.softmax(dim=1), logits


def _checkpoint_and_binding(tmp_path, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
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
        training_seed=42,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )
    checkpoint = tmp_path / "epoch_0200.pt"
    checkpoint_sha = save_checkpoint_atomic(checkpoint, payload)
    return PredictionArtifactBinding(
        checkpoint_path=checkpoint.as_posix(),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_epoch=200,
        model_variant="MODEL",
        source_tree_digest="D" * 64,
        training_seed=42,
        split_role="val_op",
        split_manifest_path=(tmp_path / "split_identity_bundle.json").as_posix(),
        split_manifest_sha256="9" * 64,
        evaluation_mode="formal",
        selection_semantic="ENDPOINT_ONLY_NOT_FOR_SELECTION",
    )


def _batches():
    return (
        PredictionBatch(
            images=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            sample_ids=("S0", "S1"),
            labels=(0, 1),
        ),
        PredictionBatch(
            images=torch.tensor([[1.0, 1.0]]),
            sample_ids=("S2",),
            labels=(1,),
        ),
    )


def test_inference_publishes_raw_logits_and_probability_from_bound_model(tmp_path):
    torch.manual_seed(7)
    model = _BinaryModel()
    binding = _checkpoint_and_binding(tmp_path, model)
    rows = infer_prediction_rows(
        model=model,
        batches=_batches(),
        binding=binding,
        run_id="RUN_42_T_U",
        arm_id="T_U",
        prediction_generation=1,
    )
    assert [row.sample_id for row in rows] == ["S0", "S1", "S2"]
    assert [row.y_true for row in rows] == [0, 1, 1]
    expected = model(torch.cat([batch.images for batch in _batches()]))[1]
    assert [row.logit_normal for row in rows] == pytest.approx(expected[:, 0].tolist())
    assert [row.logit_defect for row in rows] == pytest.approx(expected[:, 1].tolist())
    assert [row.p_defect_raw for row in rows] == pytest.approx(expected.softmax(1)[:, 1].tolist())


def test_inference_rejects_model_state_drift_or_ambiguous_outputs(tmp_path):
    model = _BinaryModel()
    binding = _checkpoint_and_binding(tmp_path, model)
    with torch.no_grad():
        model.linear.weight.add_(1.0)
    with pytest.raises(SctsrError) as drift:
        infer_prediction_rows(
            model=model,
            batches=_batches(),
            binding=binding,
            run_id="RUN_42_T_U",
            arm_id="T_U",
            prediction_generation=1,
        )
    assert drift.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH

    fresh = _BinaryModel()
    fresh_binding = _checkpoint_and_binding(tmp_path / "fresh", fresh)

    class _ProbabilityOnly(_BinaryModel):
        def forward(self, images):
            return self.linear(images).softmax(dim=1)

    probability_only = _ProbabilityOnly()
    probability_only.load_state_dict(fresh.state_dict())
    with pytest.raises(SctsrError) as ambiguous:
        infer_prediction_rows(
            model=probability_only,
            batches=_batches(),
            binding=fresh_binding,
            run_id="RUN_42_T_U",
            arm_id="T_U",
            prediction_generation=1,
        )
    assert ambiguous.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def test_inference_rejects_class_mapping_and_duplicate_identity(tmp_path):
    model = _BinaryModel()
    binding = _checkpoint_and_binding(tmp_path, model)
    model.names = {0: "target_defect", 1: "no_target"}
    with pytest.raises(SctsrError) as mapping:
        infer_prediction_rows(
            model=model,
            batches=_batches(),
            binding=binding,
            run_id="RUN_42_T_U",
            arm_id="T_U",
            prediction_generation=1,
        )
    assert mapping.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH

    model.names = {0: "no_target", 1: "target_defect"}
    duplicate = _batches() + (replace(_batches()[1], sample_ids=("S0",)),)
    with pytest.raises(SctsrError) as duplicate_error:
        infer_prediction_rows(
            model=model,
            batches=duplicate,
            binding=binding,
            run_id="RUN_42_T_U",
            arm_id="T_U",
            prediction_generation=1,
        )
    assert duplicate_error.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def _image_registry(tmp_path, rows):
    manifest = tmp_path / "val_op.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("canonical_image_relpath", "split"))
        writer.writeheader()
        writer.writerows(rows)
    identities = sorted((str(row["canonical_image_relpath"]).replace("\\", "/"), 1) for row in rows)
    hasher = hashlib.sha256()
    for sample_id, label in identities:
        hasher.update(f"{sample_id}|{label}\n".encode("utf-8"))
    raw = {
        "schema_version": "stage1.sctsr.asset_registry.v1",
        "base_denominator": 1,
        "split_asset_ids": {"val_op": ["op"]},
        "val_target_available": False,
        "assets": [
            {
                "asset_id": "op",
                "relative_path": manifest.name,
                "sha256": sha256_file(manifest),
                "bytes": manifest.stat().st_size,
                "role": "VAL_OP_COMPONENT_DEFECT_ENDPOINT_ONLY_NOT_SELECTION",
                "required": True,
                "row_count": len(rows),
                "identity_column": "canonical_image_relpath",
                "constant_label": 1,
                "identity_digest": hasher.hexdigest().upper(),
                "identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
                "split_column": "split",
                "allowed_split_values": ["val_op"],
                "identity_universe": "VAL_OP_COMPONENT",
            }
        ],
    }
    return AssetRegistry.from_mapping(raw)


def test_registered_image_records_follow_manifest_identity_and_stay_in_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    rows = [
        {"canonical_image_relpath": "Det/images/val_op/b.png", "split": "val_op"},
        {"canonical_image_relpath": "Det/images/val_op/a.png", "split": "val_op"},
    ]
    for row in rows:
        path = dataset / row["canonical_image_relpath"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
    records = load_registered_image_records(
        _image_registry(tmp_path, rows),
        repository_root=tmp_path,
        dataset_root=dataset,
        split_role="val_op",
    )
    assert [record.sample_id for record in records] == [
        "Det/images/val_op/a.png",
        "Det/images/val_op/b.png",
    ]
    assert all(record.y_true == 1 and record.image_path.is_file() for record in records)


def test_registered_image_records_reject_path_escape(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"PNG")
    rows = [{"canonical_image_relpath": "../outside.png", "split": "val_op"}]
    with pytest.raises(SctsrError) as caught:
        load_registered_image_records(
            _image_registry(tmp_path, rows),
            repository_root=tmp_path,
            dataset_root=tmp_path / "dataset",
            split_role="val_op",
        )
    assert caught.value.code is ErrorCode.ASSET_VALIDATION_FAILED


def test_formal_endpoint_publisher_runs_real_images_and_writes_complete_evidence(tmp_path):
    repository = tmp_path / "repository"
    manifest_root = repository / "manifests"
    dataset = repository / "data" / "final_sewerml_dataset"
    manifest_root.mkdir(parents=True)
    components = []
    all_rows = []
    for label, name in ((0, "normal"), (1, "defect")):
        rows = []
        for index in range(2):
            relative = f"Det/images/val_op/{name}_{index}.png"
            image_path = dataset / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((4, 4, 3), 40 + label * 160 + index, dtype=np.uint8)).save(image_path)
            rows.append({"canonical_image_relpath": relative, "split": f"{'' if label else 'normal_'}val_op"})
            all_rows.append((relative, label))
        manifest = manifest_root / f"{name}.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("canonical_image_relpath", "split"))
            writer.writeheader()
            writer.writerows(rows)
        hasher = hashlib.sha256()
        for sample_id, row_label in sorted((row["canonical_image_relpath"], label) for row in rows):
            hasher.update(f"{sample_id}|{row_label}\n".encode("utf-8"))
        components.append(
            {
                "asset_id": f"op_{name}",
                "relative_path": manifest.relative_to(repository).as_posix(),
                "sha256": sha256_file(manifest),
                "bytes": manifest.stat().st_size,
                "role": f"VAL_OP_COMPONENT_{name.upper()}_ENDPOINT_ONLY_NOT_SELECTION",
                "required": True,
                "row_count": len(rows),
                "identity_column": "canonical_image_relpath",
                "constant_label": label,
                "identity_digest": hasher.hexdigest().upper(),
                "identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
                "split_column": "split",
                "allowed_split_values": [f"{'' if label else 'normal_'}val_op"],
                "identity_universe": "VAL_OP_COMPONENT",
            }
        )
    registry_raw = {
        "schema_version": "stage1.sctsr.asset_registry.v1",
        "base_denominator": 1,
        "split_asset_ids": {"val_op": ["op_defect", "op_normal"]},
        "val_target_available": False,
        "assets": components,
    }
    registry_path = repository / "asset_registry.json"
    atomic_write_json(registry_path, registry_raw)
    registry = AssetRegistry.from_mapping(registry_raw)

    class _ImageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)
            self.names = {0: "no_target", 1: "target_defect"}

        def forward(self, images):
            logits = self.conv(images).mean(dim=(2, 3))
            return logits.softmax(dim=1), logits

    model = _ImageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
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
        training_seed=42,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest=registry.digest,
    )
    checkpoint = repository / "epoch_0200.pt"
    save_checkpoint_atomic(checkpoint, payload)

    def transform(image):
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    report = publish_formal_endpoint(
        model=model,
        transform=transform,
        run_root=repository / "run",
        repository_root=repository,
        dataset_root=dataset,
        asset_registry_path=registry_path,
        checkpoint_path=checkpoint,
        run_id="RUN_42_T_U",
        arm_id="T_U",
        model_variant="MODEL",
        batch_size=2,
    )
    assert report["status"] == "PASS"
    assert report["prediction_rows"] == len(all_rows)
    assert report["dataset_root"] == dataset.resolve().as_posix()
    assert report["frontier_points"] == 96
    assert (repository / "run/08_receipts/FORMAL_ENDPOINT_RECEIPT.json").is_file()
