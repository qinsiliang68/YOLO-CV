from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import stable_digest

ALLOWED_FIELDS = (
    "sample_id",
    "y_true",
    "label",
    "replay_role",
    "historical_dynamic_bucket",
    "dynamic_bucket",
    "oof_fold",
    "oof_group_id",
    "group_source",
    "base_manifest_membership",
    "source_manifest_identity",
)

FORBIDDEN_FIELDS = (
    "rank", "GapCritical", "gapcritical", "loss", "current_loss", "confidence",
    "mean_probability", "probability_std", "correct_rate", "rho", "RHO",
    "gradient", "forgetting", "AUM", "aum", "feature_distance", "future_epoch_outcome",
    "val_model_metric", "val_cal_metric", "val_op_metric", "test_metric",
)


@dataclass(frozen=True, slots=True)
class TerminalFieldGuard:
    allowed_fields: tuple[str, ...] = ALLOWED_FIELDS
    forbidden_fields: tuple[str, ...] = FORBIDDEN_FIELDS

    @property
    def digest(self) -> str:
        return stable_digest({"allowed_fields": self.allowed_fields, "forbidden_fields": self.forbidden_fields})

    def project_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        # Critical design: explicit key lookup only.  We never enumerate values,
        # call dict(row), access row.items(), or inspect forbidden columns.
        result: dict[str, Any] = {}
        for field in self.allowed_fields:
            try:
                result[field] = row[field]
            except KeyError:
                continue
        required_aliases = {
            "sample_id": ("sample_id",),
            "label": ("y_true", "label"),
            "dynamic_bucket": ("historical_dynamic_bucket", "dynamic_bucket"),
            "oof_fold": ("oof_fold",),
            "oof_group_id": ("oof_group_id",),
        }
        missing = [name for name, aliases in required_aliases.items() if not any(alias in result for alias in aliases)]
        if missing:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "R2 pre-terminal projection is missing required fields", observed=missing)
        return result

    def project_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.project_row(row) for row in rows]

    def reject_if_config_mentions_forbidden(self, fields: Sequence[str]) -> None:
        forbidden = sorted(set(fields) & set(self.forbidden_fields))
        if forbidden:
            raise SctsrError(ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN, "R2 matching configuration references terminal fields", observed=forbidden)
