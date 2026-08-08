from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .util import read_yaml

ALLOWED_KEYS = {
    "machine_id", "repo_root", "dataset_root", "oof_raw_root", "artifact_root", "output_root",
    "cache_root", "local_scratch_root", "gpu_id", "num_workers", "python_executable",
    "val_cal_defect_manifest", "val_cal_normal_manifest", "val_op_defect_manifest",
    "val_op_normal_manifest", "development_benchmark_defect_manifest",
    "val_model_defect_manifest", "val_model_normal_manifest",
    "development_benchmark_normal_manifest", "blind_holdout_defect_manifest",
    "blind_holdout_normal_manifest", "external_test_defect_manifest", "external_test_normal_manifest",
    "prediction_batch_size", "prediction_workers", "nvidia_smi_path", "base_checkpoint",
    "train_manifest", "normal_train_manifest", "trainer_output_root", "evaluator_output_root",
    "dry_run", "command_timeout_seconds", "staging_root", "machine_asset_report",
    "minimum_staging_free_gib", "minimum_output_free_gib", "maximum_staging_files",
    "gpu_memory_release_threshold_mib", "coordination_root", "job_lease_ttl_seconds",
    "job_lease_heartbeat_seconds"
}
FORBIDDEN_SCIENCE_KEYS = {
    "method", "budget", "epochs", "training_seed", "selection_seed", "guard_ratio", "batch_size",
    "learning_rate", "optimizer", "augmentation", "condition_id", "arm"
}

@dataclass(frozen=True)
class MachineConfig:
    path: Path
    data: dict[str, Any]

    def path_value(self, key: str, required: bool = True) -> Path | None:
        value = self.data.get(key)
        if value in (None, ""):
            if required: raise ConfigurationError(f"Missing machine path: {key}")
            return None
        return Path(str(value)).expanduser().resolve()


def load_machine_config(path: str | Path) -> MachineConfig:
    path = Path(path).resolve()
    data = read_yaml(path)
    unknown = set(data) - ALLOWED_KEYS
    forbidden = set(data) & FORBIDDEN_SCIENCE_KEYS
    if forbidden:
        raise ConfigurationError(f"Machine config contains scientific keys: {sorted(forbidden)}")
    if unknown:
        raise ConfigurationError(f"Unknown machine config keys: {sorted(unknown)}")
    required = ["machine_id", "repo_root", "dataset_root", "artifact_root", "output_root", "cache_root", "gpu_id", "num_workers"]
    if not bool(data.get("dry_run", False)):
        required += ["staging_root", "machine_asset_report"]
    for key in required:
        if key not in data: raise ConfigurationError(f"Missing machine config key: {key}")
    return MachineConfig(path=path, data=data)
