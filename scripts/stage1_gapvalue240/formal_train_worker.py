from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.contract import load_contract
from stage1_gapvalue240.formal_trainer import FormalTrainingSpec, run_formal_training
from stage1_gapvalue240.hardlink_staging import prepare_base_cache, staged_replay_session, storage_preflight
from stage1_gapvalue240.util import atomic_write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated Stage1 GapValue formal training worker (hardlink-only staging)."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--base-train-defect-manifest", required=True)
    parser.add_argument("--base-train-normal-manifest", required=True)
    parser.add_argument("--base-val-defect-manifest", required=True)
    parser.add_argument("--base-val-normal-manifest", required=True)
    parser.add_argument("--run-train-defect-manifest", required=True)
    parser.add_argument("--run-train-normal-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yolo-root", required=True)
    parser.add_argument("--run-slot", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--budget", required=True, choices=(600, 3000, 6000), type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--segment-id")
    parser.add_argument("--minimum-staging-free-gib", type=float, default=2.0)
    parser.add_argument("--minimum-output-free-gib", type=float, default=20.0)
    parser.add_argument("--maximum-staging-files", type=int, default=151_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_contract(args.contract)
    manifests = {
        "train_defect": Path(args.base_train_defect_manifest),
        "train_normal": Path(args.base_train_normal_manifest),
        "val_defect": Path(args.base_val_defect_manifest),
        "val_normal": Path(args.base_val_normal_manifest),
    }
    gib = 1024 ** 3
    storage_report = storage_preflight(
        dataset_root=args.dataset_root,
        staging_root=args.staging_root,
        output_root=args.output_dir,
        hardlink_probe_manifest=manifests["train_defect"],
        expected_staging_files=144_000 + int(args.budget) + 10,
        maximum_staging_files=int(args.maximum_staging_files),
        minimum_staging_free_bytes=int(float(args.minimum_staging_free_gib) * gib),
        minimum_output_free_bytes=int(float(args.minimum_output_free_gib) * gib),
    )
    atomic_write_json(Path(args.output_dir) / "storage_preflight.json", storage_report, overwrite=True)
    cache = prepare_base_cache(args.dataset_root, args.staging_root, manifests)
    with staged_replay_session(
        cache,
        args.run_train_defect_manifest,
        args.run_train_normal_manifest,
        run_slot=args.run_slot,
        expected_replay_rows=args.budget,
    ) as staged:
        spec = FormalTrainingSpec.from_contract(
            contract,
            dataset_dir=staged.dataset_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            yolo_root=args.yolo_root,
            training_seed=args.training_seed,
            budget=args.budget,
            device=args.device,
            workers=args.workers,
            resume_checkpoint=args.resume_checkpoint,
            segment_id=args.segment_id,
        )
        result = run_formal_training(spec)
    print(
        json.dumps(
            {
                "run_slot": args.run_slot,
                "base_cache_snapshot_id": cache.snapshot_id,
                "base_cache_reused": cache.reused,
                "trainer_dir": str(result.trainer_dir),
                "stable_last": str(result.stable_last),
                "training_execution_audit": str(result.audit_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
