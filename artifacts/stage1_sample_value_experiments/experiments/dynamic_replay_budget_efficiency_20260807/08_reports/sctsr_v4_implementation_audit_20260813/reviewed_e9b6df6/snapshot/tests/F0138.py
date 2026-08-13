from types import SimpleNamespace
from pathlib import Path
import sys
import json

import torch

from stage1_sctsr_v4.replay_step_plan import build_replay_step_plan
from stage1_sctsr_v4.serialization import sha256_file
from stage1_sctsr_v4.ultralytics_overlay import run_ultralytics_classification_epoch


EXPECTED_WEIGHT_SHA = "6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C"


def test_sa_115_real_frozen_yolo11l_classification_trainer_step(repository_root):
    yolo_root = repository_root / "YOLOv11"
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))

    from ultralytics.models.yolo.classify.train import ClassificationTrainer
    from ultralytics.nn.tasks import ClassificationModel, load_checkpoint
    from ultralytics.utils.torch_utils import ModelEMA

    assert Path(sys.modules[ClassificationTrainer.__module__].__file__).resolve().is_relative_to(yolo_root.resolve())

    weight = repository_root / "yolo11l-cls.pt"
    assert weight.is_file()
    assert sha256_file(weight) == EXPECTED_WEIGHT_SHA
    model, _ = load_checkpoint(str(weight), device="cpu", inplace=False, fuse=False)
    ClassificationModel.reshape_outputs(model, 2)
    model.float()

    # Construct a prepared instance of the exact frozen trainer class without
    # invoking its full filesystem/data orchestration.  The exercised model,
    # preprocess_batch and optimizer_step methods are real YOLO-CV code.
    trainer = object.__new__(ClassificationTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
    trainer.ema = ModelEMA(model)
    trainer.device = torch.device("cpu")
    trainer.amp = False
    trainer.batch_size = 128
    trainer.world_size = 1
    trainer.accumulate = 1
    trainer.args = SimpleNamespace(
        resume="",
        warmup_epochs=0,
        nbs=64,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        momentum=0.937,
        compile=False,
    )
    trainer.train_loader = [
        {
            "img": torch.zeros(128, 3, 64, 64),
            "cls": torch.arange(128) % 2,
            "sample_ids": [f"BASE_{index:03d}" for index in range(128)],
            "augmentation_digests": ["A" * 64] * 128,
        }
    ]
    optimizer_step = trainer.optimizer_step
    calls = {"count": 0}

    def counted_optimizer_step():
        calls["count"] += 1
        return optimizer_step()

    trainer.optimizer_step = counted_optimizer_step
    plan = build_replay_step_plan(
        run_id="REAL_YOLO_CANARY",
        arm_id="T_U",
        training_seed=31,
        epoch=121,
        schedule_family="U",
        sample_ids=("REPLAY_000",),
        base_batch_sizes=(128,),
    )

    def replay_provider(ids, epoch, step, seed):
        del epoch, step, seed
        return {
            "img": torch.ones(len(ids), 3, 64, 64),
            "cls": torch.zeros(len(ids), dtype=torch.long),
            "sample_ids": tuple(ids),
            "augmentation_digests": ("B" * 64,) * len(ids),
        }

    result = run_ultralytics_classification_epoch(
        trainer=trainer,
        replay_plan=plan,
        replay_batch_provider=replay_provider,
        training_seed=31,
        epoch=121,
        global_step_start=112_560,
    )

    assert type(trainer).__module__ == "ultralytics.models.yolo.classify.train"
    assert type(trainer.model).__name__ == "ClassificationModel"
    assert calls["count"] == 1
    assert result["optimizer_steps"] == 1
    assert result["base_occurrences"] == 128
    assert result["replay_occurrences"] == 1
    assert result["global_step_end"] == 112_561
    print(json.dumps({
        "schema_version": "stage1.sctsr.real_yolo_engineering_canary.v1",
        "status": "PASS",
        "scientific_role": "SYNTHETIC_NOT_SCIENTIFIC_RESULT",
        "trainer_class": f"{type(trainer).__module__}.{type(trainer).__name__}",
        "model_class": f"{type(trainer.model).__module__}.{type(trainer.model).__name__}",
        "initial_weight_sha256": EXPECTED_WEIGHT_SHA,
        "base_occurrences": result["base_occurrences"],
        "replay_occurrences": result["replay_occurrences"],
        "optimizer_steps": result["optimizer_steps"],
        "upstream_optimizer_step_calls": calls["count"],
        "global_step_start": 112_560,
        "global_step_end": result["global_step_end"],
        "blind_or_test_data_accessed": False,
        "formal_training_started": False,
    }, sort_keys=True))
