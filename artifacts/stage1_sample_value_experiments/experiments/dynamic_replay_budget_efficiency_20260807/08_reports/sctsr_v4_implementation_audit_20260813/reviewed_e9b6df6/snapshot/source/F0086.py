from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .baseline_reference import SOURCE_TREE_INCLUDE_PATHS, source_external_references
from .arm_spec import ArmId
from .branch_lineage import BranchLineage
from .checkpointing import build_checkpoint_payload, load_checkpoint, save_checkpoint_atomic
from .fixed_step_runtime import run_fixed_step_epoch
from .errors import ErrorCode, SctsrError
from .replay_step_plan import build_replay_step_plan
from .serialization import atomic_write_json, sha256_file, stable_digest
from .source_identity import build_source_tree_manifest
from .synthetic_canary import _build_all_schedules, _new_training_stack, _restore_training_stack
from .synthetic_fixture import build_synthetic_fixture, make_base_loader, make_replay_provider

SYNTHETIC = "SYNTHETIC_NOT_SCIENTIFIC_RESULT"


def run_synthetic_common_parent(output_root: str | Path, *, repository_root: str | Path, training_seed: int, overwrite: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    if overwrite:
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Synthetic parent evidence is immutable; choose a new output root")
    if root.exists():
        raise FileExistsError(f"Output exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    source = build_source_tree_manifest(
        repository_root,
        SOURCE_TREE_INCLUDE_PATHS,
        external_references=source_external_references(),
    )
    fixture = build_synthetic_fixture(training_seed=training_seed)
    model, optimizer, scheduler, scaler, ema = _new_training_stack(training_seed)
    loader = make_base_loader(fixture, epoch=120)
    sizes = [int(batch["labels"].shape[0]) for batch in loader]
    plan = build_replay_step_plan(
        run_id=f"SYNTH_PARENT_{training_seed}", arm_id="COMMON_PARENT_NR", training_seed=training_seed,
        epoch=120, schedule_family="NR", sample_ids=(), base_batch_sizes=sizes,
    )
    result = run_fixed_step_epoch(
        model=model, optimizer=optimizer, base_loader=loader, replay_plan=plan,
        replay_batch_provider=make_replay_provider(fixture), training_seed=training_seed, epoch=120,
        ema=ema, scheduler=scheduler, scaler=scaler, clip_max_norm=10.0,
    )
    source_digest = source["source_tree_digest"]
    payload = build_checkpoint_payload(
        model=model, ema=ema, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        epoch=120, global_step=result.optimizer_steps, base_sampler_generation=1,
        canonical_training_lock_sha256=stable_digest({"synthetic_lock": True}),
        initial_checkpoint_sha256=stable_digest({"synthetic_initial": True}),
        base_manifest_sha256=stable_digest(fixture.base_ids), training_seed=training_seed,
        source_tree_digest=source_digest, runtime_config_digest=stable_digest({"runtime": "synthetic_parent_v1"}),
        asset_registry_digest=stable_digest({"fixture": "synthetic_v1"}),
    )
    checkpoint = root / "epoch_0120.pt"
    checkpoint_sha = save_checkpoint_atomic(checkpoint, payload)
    receipt = {
        "schema_version": "stage1.sctsr.synthetic_parent_receipt.v1",
        "status": "PASS", "semantic": SYNTHETIC, "scientific_result": False,
        "training_seed": training_seed, "logical_epoch": 120,
        "mechanism_epochs_executed": 1, "compressed_timeline": True,
        "optimizer_steps": result.optimizer_steps, "base_occurrences": result.base_occurrences,
        "replay_occurrences": result.replay_occurrences, "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha, "source_tree_digest": source_digest,
        "formal_training_started": False,
    }
    atomic_write_json(root / "PARENT_RECEIPT.json", receipt)
    atomic_write_json(root / "SOURCE_TREE_MANIFEST.json", source)
    return receipt


def run_synthetic_branch(
    output_root: str | Path, *, repository_root: str | Path, parent_checkpoint: str | Path,
    arm_id: ArmId, training_seed: int, epoch: int = 121, overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    if overwrite:
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Synthetic branch evidence is immutable; choose a new output root")
    if root.exists():
        raise FileExistsError(f"Output exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    fixture = build_synthetic_fixture(training_seed=training_seed)
    schedules = _build_all_schedules(fixture)
    schedule = schedules[arm_id]
    epoch_plan = schedule.epoch(epoch)
    payload = load_checkpoint(parent_checkpoint, expected_epoch=120)
    model, optimizer, scheduler, scaler, ema = _restore_training_stack(payload)
    loader = make_base_loader(fixture, epoch=epoch)
    sizes = [int(batch["labels"].shape[0]) for batch in loader]
    plan = build_replay_step_plan(
        run_id=f"SYNTH_{arm_id.value}_{training_seed}", arm_id=arm_id.value, training_seed=training_seed,
        epoch=epoch, schedule_family=epoch_plan.schedule_family,
        sample_ids=epoch_plan.sample_ids, base_batch_sizes=sizes,
    )
    result = run_fixed_step_epoch(
        model=model, optimizer=optimizer, base_loader=loader, replay_plan=plan,
        replay_batch_provider=make_replay_provider(fixture), training_seed=training_seed, epoch=epoch,
        ema=ema, scheduler=scheduler, scaler=scaler, clip_max_norm=10.0,
    )
    source = build_source_tree_manifest(
        repository_root,
        SOURCE_TREE_INCLUDE_PATHS,
        external_references=source_external_references(),
    )
    contract_digest = stable_digest({"synthetic": True, "anchors": [120,160,200]})
    lineage = BranchLineage.create(
        logical_run_id=f"SYNTH_{arm_id.value}_{training_seed}", parent_id=f"SYNTH_PARENT_{training_seed}",
        parent_checkpoint_path=Path(parent_checkpoint).as_posix(), parent_checkpoint_sha256=sha256_file(parent_checkpoint),
        training_seed=training_seed, arm_id=arm_id.value, child_source_tree_digest=source["source_tree_digest"],
        child_contract_digest=contract_digest,
    )
    checkpoint_payload = build_checkpoint_payload(
        model=model, ema=ema, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        epoch=epoch, global_step=int(payload["global_step"]) + result.optimizer_steps, base_sampler_generation=1,
        canonical_training_lock_sha256=payload["canonical_training_lock_sha256"],
        initial_checkpoint_sha256=payload["initial_checkpoint_sha256"], base_manifest_sha256=payload["base_manifest_sha256"],
        training_seed=training_seed, source_tree_digest=source["source_tree_digest"],
        runtime_config_digest=stable_digest({"runtime": "synthetic_branch_v1", "arm": arm_id.value}),
        asset_registry_digest=payload["asset_registry_digest"],
    )
    checkpoint = root / f"epoch_{epoch:04d}.pt"
    checkpoint_sha = save_checkpoint_atomic(checkpoint, checkpoint_payload)
    receipt = {
        "schema_version": "stage1.sctsr.synthetic_branch_receipt.v1", "status": "PASS",
        "semantic": SYNTHETIC, "scientific_result": False, "arm_id": arm_id.value,
        "training_seed": training_seed, "logical_epoch": epoch, "mechanism_epochs_executed": 1,
        "compressed_timeline": True, "optimizer_steps": result.optimizer_steps,
        "base_occurrences": result.base_occurrences, "replay_occurrences": result.replay_occurrences,
        "checkpoint_path": checkpoint.as_posix(), "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint), "lineage": asdict(lineage),
        "formal_training_started": False,
    }
    atomic_write_json(root / "BRANCH_LINEAGE.json", asdict(lineage))
    atomic_write_json(root / "BRANCH_RECEIPT.json", receipt)
    return receipt
