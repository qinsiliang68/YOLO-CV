from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


FAMILY_ROOT = Path("artifacts") / "stage1_sample_value_experiments"
ACTIVE_EXPERIMENT_ID = "oof_dynamics_gap_value_20260708"

STAGE_DIRS: dict[str, str] = {
    "00_registry": "00_registry",
    "01_oof_dynamics": "01_oof_dynamics",
    "02_sample_value_tables": "02_sample_value_tables",
    "03_replay_manifests": "03_replay_manifests",
    "04_training_runs": "04_training_runs",
    "05_eval": "05_eval",
    "06_reports": "06_reports",
}

METHOD_IDS: tuple[str, ...] = (
    "confidence_fp",
    "boundary_fp",
    "persistent_fp",
    "grad_mag",
    "grad_align",
    "grad_mag_align",
    "diverse_grad_align",
    "gap_critical",
    "grad_align_guard",
    "gap_critical_guard",
)

BUDGETS: tuple[int, ...] = (600, 3000, 6000)


def manifest_build_status(method_id: str) -> str:
    if method_id.endswith("_guard"):
        return "requires_defect_guard_budget"
    return "default_ready"


@dataclass(frozen=True)
class ExperimentRow:
    experiment_id: str
    experiment_group: str
    status: str
    physical_root: str
    artifact_mode: str
    note: str

    def as_dict(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_group": self.experiment_group,
            "status": self.status,
            "physical_root": self.physical_root,
            "artifact_mode": self.artifact_mode,
            "note": self.note,
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def family_root(repo: Path | None = None, override: Path | None = None) -> Path:
    if override is not None:
        return override.resolve()
    root = repo or repo_root()
    return (root / FAMILY_ROOT).resolve()


def experiments_root(root: Path) -> Path:
    return root / "experiments"


def experiment_root(root: Path, experiment_id: str) -> Path:
    return experiments_root(root) / experiment_id


def stage_root(active_root: Path, stage_key: str) -> Path:
    try:
        stage_dir = STAGE_DIRS[stage_key]
    except KeyError as exc:
        raise KeyError(f"Unknown stage key: {stage_key}") from exc
    return active_root / stage_dir


def method_dir(active_root: Path, stage_key: str, method_id: str) -> Path:
    if method_id not in METHOD_IDS:
        raise ValueError(f"Unknown method_id: {method_id}")
    return stage_root(active_root, stage_key) / method_id


def budget_dir_name(budget: int) -> str:
    if budget <= 0:
        raise ValueError(f"Budget must be positive: {budget}")
    return f"budget_{budget:05d}"


def method_budget_dir(active_root: Path, stage_key: str, method_id: str, budget: int) -> Path:
    return method_dir(active_root, stage_key, method_id) / budget_dir_name(budget)


def run_dir(active_root: Path, stage_key: str, method_id: str, budget: int, run_id: str) -> Path:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    return method_budget_dir(active_root, stage_key, method_id, budget) / run_id


def default_experiment_rows() -> list[dict[str, str]]:
    rows = [
        ExperimentRow(
            experiment_id="hn_rn_formal_manifest_review_20260623",
            experiment_group="confidence_only_replay",
            status="legacy_review",
            physical_root="artifacts/stage1_phase1_hn_rn_20260623_formal_manifest_review",
            artifact_mode="legacy_pointer",
            note="Original HN/RN manifest review. Kept in place; registered for experiment lineage.",
        ),
        ExperimentRow(
            experiment_id="hn_rn_40runs_20260630",
            experiment_group="confidence_only_replay",
            status="legacy_result",
            physical_root="artifacts/stage1_phase1_hn_rn_40runs_20260630_redownload",
            artifact_mode="legacy_pointer",
            note="Exploratory 40-run HN/RN result package. Kept in place; do not move.",
        ),
        ExperimentRow(
            experiment_id="hn_band_120runs_20260707",
            experiment_group="confidence_band_replay",
            status="legacy_result",
            physical_root="artifacts/stage1_phase1_hn_band_20260628_120runs_20260707_review",
            artifact_mode="legacy_pointer",
            note="Exploratory 120-run HN band/RN review package. Kept in place; do not move.",
        ),
        ExperimentRow(
            experiment_id=ACTIVE_EXPERIMENT_ID,
            experiment_group="oof_training_dynamics_sample_value",
            status="active",
            physical_root=f"{FAMILY_ROOT.as_posix()}/experiments/{ACTIVE_EXPERIMENT_ID}",
            artifact_mode="native_grouped_tree",
            note="New grouped OOF dynamics and score-gap sample-value experiment.",
        ),
    ]
    return [row.as_dict() for row in rows]


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_legacy_pointer(pointer_root: Path, row: dict[str, str]) -> None:
    pointer_root.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"# {row['experiment_id']}",
            "",
            "This is a lightweight pointer for the grouped experiment registry.",
            f"legacy_artifact_root: `{row['physical_root']}`",
            f"status: `{row['status']}`",
            "",
            "The original artifact directory is intentionally not moved, so old citations and scripts remain valid.",
            "",
        ]
    )
    (pointer_root / "README.md").write_text(text, encoding="utf-8")


def initialize_active_experiment_dirs(active_root: Path) -> None:
    for stage_dir in STAGE_DIRS.values():
        (active_root / stage_dir).mkdir(parents=True, exist_ok=True)

    registry_root = stage_root(active_root, "00_registry")
    method_rows = [
        {
            "method_id": method_id,
            "manifest_build_status": manifest_build_status(method_id),
            "stage_02_value_table_dir": (stage_root(active_root, "02_sample_value_tables") / method_id).as_posix(),
            "stage_03_replay_manifest_dir": (stage_root(active_root, "03_replay_manifests") / method_id).as_posix(),
            "stage_04_training_run_dir": (stage_root(active_root, "04_training_runs") / method_id).as_posix(),
            "stage_05_eval_dir": (stage_root(active_root, "05_eval") / method_id).as_posix(),
        }
        for method_id in METHOD_IDS
    ]
    write_csv(
        registry_root / "methods.csv",
        method_rows,
        (
            "method_id",
            "manifest_build_status",
            "stage_02_value_table_dir",
            "stage_03_replay_manifest_dir",
            "stage_04_training_run_dir",
            "stage_05_eval_dir",
        ),
    )

    run_matrix_rows: list[dict[str, str]] = []
    for method_id in METHOD_IDS:
        (stage_root(active_root, "02_sample_value_tables") / method_id).mkdir(parents=True, exist_ok=True)
        for budget in BUDGETS:
            for stage_key in ("03_replay_manifests", "04_training_runs", "05_eval"):
                method_budget_dir(active_root, stage_key, method_id, budget).mkdir(parents=True, exist_ok=True)
            run_matrix_rows.append(
                {
                    "experiment_id": ACTIVE_EXPERIMENT_ID,
                    "method_id": method_id,
                    "manifest_build_status": manifest_build_status(method_id),
                    "budget": str(budget),
                    "run_id": f"{method_id}_b{budget:05d}_r001",
                    "replay_manifest_dir": method_budget_dir(
                        active_root, "03_replay_manifests", method_id, budget
                    ).as_posix(),
                    "training_run_dir": run_dir(
                        active_root,
                        "04_training_runs",
                        method_id,
                        budget,
                        f"{method_id}_b{budget:05d}_r001",
                    ).as_posix(),
                    "eval_dir": run_dir(
                        active_root,
                        "05_eval",
                        method_id,
                        budget,
                        f"{method_id}_b{budget:05d}_r001",
                    ).as_posix(),
                }
            )
    write_csv(
        registry_root / "run_matrix.csv",
        run_matrix_rows,
        (
            "experiment_id",
            "method_id",
            "manifest_build_status",
            "budget",
            "run_id",
            "replay_manifest_dir",
            "training_run_dir",
            "eval_dir",
        ),
    )
    write_json(
        registry_root / "experiment_plan.json",
        {
            "experiment_id": ACTIVE_EXPERIMENT_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stage_dirs": STAGE_DIRS,
            "method_ids": METHOD_IDS,
            "method_build_status": {method_id: manifest_build_status(method_id) for method_id in METHOD_IDS},
            "budgets": BUDGETS,
            "layout_rule": "family/experiments/experiment_id/stage/method/budget/run",
        },
    )


def initialize_experiment_family_layout(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    registry_root = root / "00_registry"
    rows = default_experiment_rows()
    write_csv(
        registry_root / "experiments.csv",
        rows,
        ("experiment_id", "experiment_group", "status", "physical_root", "artifact_mode", "note"),
    )
    write_json(
        registry_root / "experiments.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "family_root": root.as_posix(),
            "experiments": rows,
        },
    )

    for row in rows:
        pointer_root = experiment_root(root, row["experiment_id"])
        if row["artifact_mode"] == "legacy_pointer":
            write_legacy_pointer(pointer_root, row)
        else:
            initialize_active_experiment_dirs(pointer_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize grouped Stage-1 sample-value experiment layout.")
    parser.add_argument("--family-root", type=Path, default=None, help="Override experiment family root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = family_root(override=args.family_root)
    initialize_experiment_family_layout(root)
    print(f"initialized_experiment_family={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
