"""Freeze a bounded candidate/target panel for gradient diagnostics.

The Treatment pool is the candidate set. Fixed OOF-only monitor groups supply
the difficult-normal and weak-defect target gradients. The artifact records two
direction axes and never invents a scalar sample-value score.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import uuid

import pandas as pd

from .errors import ValidationError
from .util import atomic_write_bytes, atomic_write_json, sha256_file


class GradientCandidateManifestError(ValidationError):
    """Raised when fixed gradient candidates or targets are invalid."""


NORMAL_TARGET_GROUP = "NORMAL_REPLAY_CORE_0P5_CLASS_PERCENT"
DEFECT_TARGET_GROUP = "WEAK_DEFECT_CORE_0P5_CLASS_PERCENT"
_TREATMENT_REQUIRED = {
    "selection_rank",
    "role_rank",
    "sample_id",
    "y_true",
    "replay_role",
}
_MONITOR_REQUIRED = {"sample_id", "y_true", "monitor_group"}


@dataclass(frozen=True)
class GradientCandidateManifestResult:
    status: str
    candidate_manifest: Path
    validation_path: Path
    union_sample_count: int
    normal_target_count: int
    defect_target_count: int


def _read(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise GradientCandidateManifestError(f"{name} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_id"] = frame.sample_id.astype(str).str.strip()
    if frame.sample_id.eq("").any() or frame.sample_id.duplicated().any():
        raise GradientCandidateManifestError(f"{name} has empty or duplicate sample_id")
    frame["y_true"] = pd.to_numeric(frame.y_true, errors="raise").astype(int)
    if not set(frame.y_true.unique()) <= {0, 1}:
        raise GradientCandidateManifestError(f"{name} labels must be binary")
    return frame


def _load_sources(treatment_path: Path, monitor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment = _read(treatment_path, _TREATMENT_REQUIRED, "Treatment selection")
    monitor = _read(monitor_path, _MONITOR_REQUIRED, "causal monitor")
    if not treatment.y_true.eq(0).all() or not treatment.replay_role.astype(str).eq(
        "normal_replay"
    ).all():
        raise GradientCandidateManifestError("Treatment gradient candidates must be normal replay")
    ranks = pd.to_numeric(treatment.selection_rank, errors="raise").astype(int)
    if ranks.tolist() != list(range(1, len(treatment) + 1)):
        raise GradientCandidateManifestError("Treatment selection_rank must be contiguous")
    treatment["selection_rank"] = ranks
    treatment["role_rank"] = pd.to_numeric(treatment.role_rank, errors="raise").astype(int)
    if "monitor_rank" in monitor:
        monitor = monitor.sort_values("monitor_rank", kind="stable").reset_index(drop=True)
    overlap = treatment.set_index("sample_id").y_true.to_dict()
    conflicts = [
        sample_id
        for sample_id, label in monitor.set_index("sample_id").y_true.items()
        if sample_id in overlap and int(overlap[sample_id]) != int(label)
    ]
    if conflicts:
        raise GradientCandidateManifestError(
            f"cross-source label conflict for samples: {conflicts[:3]}"
        )
    groups = set(monitor.monitor_group.astype(str))
    if not {NORMAL_TARGET_GROUP, DEFECT_TARGET_GROUP} <= groups:
        raise GradientCandidateManifestError("both registered gradient target groups are required")
    normal_target = monitor.monitor_group.astype(str).eq(NORMAL_TARGET_GROUP)
    defect_target = monitor.monitor_group.astype(str).eq(DEFECT_TARGET_GROUP)
    if not monitor.loc[normal_target, "y_true"].eq(0).all():
        raise GradientCandidateManifestError("normal target group contains non-normal labels")
    if not monitor.loc[defect_target, "y_true"].eq(1).all():
        raise GradientCandidateManifestError("defect target group contains non-defect labels")
    return treatment, monitor


def _context(row: pd.Series) -> dict[str, object]:
    aliases = {
        "oof_fold": ("oof_fold",),
        "dynamic_bucket": ("dynamic_bucket",),
        "mean_p_defect": ("mean_p_defect", "oof_mean_p_defect"),
        "correct_rate": ("correct_rate", "oof_correct_rate"),
        "std_p_defect": ("std_p_defect", "oof_std_p_defect"),
    }
    values: dict[str, object] = {}
    for output, candidates in aliases.items():
        for candidate in candidates:
            if candidate in row.index and row[candidate] != "":
                values[output] = row[candidate]
                break
    return values


def _build_union(treatment: pd.DataFrame, monitor: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, dict[str, object]] = {}
    ordered: list[str] = []
    for row in treatment.itertuples(index=False):
        sample_id = str(row.sample_id)
        ordered.append(sample_id)
        series = treatment.loc[treatment.sample_id.eq(sample_id)].iloc[0]
        records[sample_id] = {
            "sample_id": sample_id,
            "image_path": sample_id,
            "y_true": int(row.y_true),
            "treatment_member": True,
            "treatment_selection_rank": int(row.selection_rank),
            "treatment_role_rank": int(row.role_rank),
            "monitor_groups": [],
            **_context(series),
        }
    for row in monitor.itertuples(index=False):
        sample_id = str(row.sample_id)
        series = monitor.loc[monitor.sample_id.eq(sample_id)].iloc[0]
        if sample_id not in records:
            ordered.append(sample_id)
            records[sample_id] = {
                "sample_id": sample_id,
                "image_path": sample_id,
                "y_true": int(row.y_true),
                "treatment_member": False,
                "treatment_selection_rank": "",
                "treatment_role_rank": "",
                "monitor_groups": [],
                **_context(series),
            }
        groups = records[sample_id]["monitor_groups"]
        assert isinstance(groups, list)
        groups.append(str(row.monitor_group))

    rows: list[dict[str, object]] = []
    for rank, sample_id in enumerate(ordered, start=1):
        record = records[sample_id]
        groups = list(record.pop("monitor_groups"))
        candidate_groups = (["TREATMENT_GAPCRITICAL_NESTED"] if record["treatment_member"] else []) + groups
        rows.append(
            {
                "candidate_rank": rank,
                **record,
                "normal_target_member": NORMAL_TARGET_GROUP in groups,
                "defect_target_member": DEFECT_TARGET_GROUP in groups,
                "candidate_groups": ";".join(candidate_groups),
                "monitor_groups": ";".join(groups),
            }
        )
    return pd.DataFrame(rows)


def _validate_target_counts(
    candidates: pd.DataFrame,
    expected_target_counts: dict[str, int] | None,
) -> tuple[int, int]:
    normal = int(candidates.normal_target_member.sum())
    defect = int(candidates.defect_target_member.sum())
    if normal <= 0 or defect <= 0:
        raise GradientCandidateManifestError("normal and defect gradient targets must be non-empty")
    if (candidates.normal_target_member & candidates.defect_target_member).any():
        raise GradientCandidateManifestError("normal and defect gradient targets overlap")
    if expected_target_counts is not None:
        expected = {
            "normal": int(expected_target_counts["normal"]),
            "defect": int(expected_target_counts["defect"]),
        }
        actual = {"normal": normal, "defect": defect}
        if actual != expected:
            raise GradientCandidateManifestError(
                f"gradient target count mismatch: {actual} != {expected}"
            )
    return normal, defect


def _validate_existing(
    output: Path,
    *,
    treatment_sha: str,
    monitor_sha: str,
    lock_sha: str,
    expected_target_counts: dict[str, int] | None,
) -> GradientCandidateManifestResult | None:
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        return None
    candidate = output / "GRADIENT_CANDIDATE_SAMPLES.csv"
    validation_path = output / "GRADIENT_CANDIDATE_VALIDATION.json"
    if not candidate.is_file() or not validation_path.is_file():
        raise GradientCandidateManifestError(f"gradient candidate output is half-published: {output}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "candidate_manifest_sha256": sha256_file(candidate),
        "treatment_selection_sha256": treatment_sha,
        "monitor_manifest_sha256": monitor_sha,
        "canonical_lock_file_sha256": lock_sha,
    }
    mismatch = {
        key: (validation.get(key), value)
        for key, value in expected.items()
        if validation.get(key) != value
    }
    if mismatch:
        raise GradientCandidateManifestError(
            f"existing gradient candidate identity mismatch: {mismatch}"
        )
    frame = pd.read_csv(candidate, keep_default_na=False)
    normal, defect = _validate_target_counts(frame, expected_target_counts)
    if len(frame) != int(validation.get("union_sample_count", -1)):
        raise GradientCandidateManifestError("existing gradient candidate row count mismatch")
    return GradientCandidateManifestResult(
        "PASS", candidate, validation_path, len(frame), normal, defect
    )


def build_gradient_candidate_manifest(
    treatment_selection: str | Path,
    monitor_manifest: str | Path,
    output_dir: str | Path,
    *,
    canonical_lock_file_sha256: str,
    expected_target_counts: dict[str, int] | None = None,
) -> GradientCandidateManifestResult:
    """Publish the Treatment/monitor union and explicit gradient target masks."""

    treatment_path = Path(treatment_selection).resolve()
    monitor_path = Path(monitor_manifest).resolve()
    lock_sha = str(canonical_lock_file_sha256).upper()
    if len(lock_sha) != 64 or any(character not in "0123456789ABCDEF" for character in lock_sha):
        raise GradientCandidateManifestError("canonical lock SHA must be 64 hexadecimal characters")
    treatment_sha = sha256_file(treatment_path)
    monitor_sha = sha256_file(monitor_path)
    output = Path(output_dir).resolve()
    existing = _validate_existing(
        output,
        treatment_sha=treatment_sha,
        monitor_sha=monitor_sha,
        lock_sha=lock_sha,
        expected_target_counts=expected_target_counts,
    )
    if existing is not None:
        return existing
    treatment, monitor = _load_sources(treatment_path, monitor_path)
    candidates = _build_union(treatment, monitor)
    normal_count, defect_count = _validate_target_counts(candidates, expected_target_counts)

    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmpdir"
    try:
        staging.mkdir(parents=True)
        candidate_path = staging / "GRADIENT_CANDIDATE_SAMPLES.csv"
        atomic_write_bytes(
            candidate_path,
            candidates.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )
        validation_path = staging / "GRADIENT_CANDIDATE_VALIDATION.json"
        atomic_write_json(
            validation_path,
            {
                "schema_version": "stage1.gradient_candidate_manifest.v2",
                "status": "PASS",
                "definition": {
                    "candidate_pool": "frozen Treatment pool plus fixed OOF monitor samples",
                    "normal_target": NORMAL_TARGET_GROUP,
                    "defect_target": DEFECT_TARGET_GROUP,
                    "composite_value_score": None,
                },
                "union_sample_count": len(candidates),
                "treatment_candidate_count": len(treatment),
                "monitor_sample_count": len(monitor),
                "normal_target_count": normal_count,
                "defect_target_count": defect_count,
                "treatment_selection_sha256": treatment_sha,
                "monitor_manifest_sha256": monitor_sha,
                "canonical_lock_file_sha256": lock_sha,
                "candidate_manifest_sha256": sha256_file(candidate_path),
            },
        )
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return GradientCandidateManifestResult(
        "PASS",
        output / "GRADIENT_CANDIDATE_SAMPLES.csv",
        output / "GRADIENT_CANDIDATE_VALIDATION.json",
        len(candidates),
        normal_count,
        defect_count,
    )


__all__ = [
    "DEFECT_TARGET_GROUP",
    "GradientCandidateManifestError",
    "GradientCandidateManifestResult",
    "NORMAL_TARGET_GROUP",
    "build_gradient_candidate_manifest",
]
