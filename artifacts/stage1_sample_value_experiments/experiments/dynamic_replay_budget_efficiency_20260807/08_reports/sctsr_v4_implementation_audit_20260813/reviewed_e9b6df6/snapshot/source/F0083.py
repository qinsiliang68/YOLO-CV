from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import sha256_file, stable_digest


FROZEN_CONTRAST_IDS = tuple(f"C{index:02d}" for index in range(1, 9))


@dataclass(frozen=True, slots=True)
class ContrastFamilySpec:
    schema_version: str
    alpha: float
    contrasts: tuple[str, ...]
    efficacy_early_stop_allowed: bool
    method: str
    state: str
    file_sha256: str
    contract_digest: str

    def validate(self, *, require_release_frozen: bool = False) -> None:
        if self.schema_version != "stage1.sctsr.contrast_family.v1":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown contrast family schema")
        if self.alpha != 0.05 or self.contrasts != FROZEN_CONTRAST_IDS:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Contrast family must contain ordered C01-C08 at familywise alpha 0.05")
        if self.efficacy_early_stop_allowed or self.method != "HOLM_FAMILYWISE":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Contrast family permits a forbidden method or efficacy early stop")
        allowed_states = {"TO_BE_FROZEN_BY_RELEASE_AUTHORITY", "FROZEN_BY_SIGNED_RELEASE"}
        if self.state not in allowed_states:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Contrast family state is unregistered")
        if require_release_frozen and self.state != "FROZEN_BY_SIGNED_RELEASE":
            raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal statistics require a contrast family frozen by the signed release")
        if not all(len(value) == 64 and value == value.upper() and all(char in "0123456789ABCDEF" for char in value) for value in (self.file_sha256, self.contract_digest)):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Contrast family identity digest is invalid")


def load_contrast_family_spec(path: str | Path, *, require_release_frozen: bool = False) -> ContrastFamilySpec:
    source = Path(path)
    if not source.is_file():
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Contrast family JSON is missing", artifact_path=str(source))
    with source.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    expected_fields = {"schema_version", "alpha", "contrasts", "efficacy_early_stop_allowed", "method", "state"}
    if set(payload) != expected_fields:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Contrast family fields do not exactly match the registered schema",
            observed={"missing": sorted(expected_fields - set(payload)), "extra": sorted(set(payload) - expected_fields)},
        )
    contract_payload = {
        "schema_version": payload["schema_version"],
        "alpha": float(payload["alpha"]),
        "contrasts": tuple(payload["contrasts"]),
        "efficacy_early_stop_allowed": payload["efficacy_early_stop_allowed"],
        "method": payload["method"],
        "state": payload["state"],
    }
    spec = ContrastFamilySpec(
        **contract_payload,
        file_sha256=sha256_file(source),
        contract_digest=stable_digest(contract_payload),
    )
    spec.validate(require_release_frozen=require_release_frozen)
    return spec


@dataclass(frozen=True, slots=True)
class Endpoint:
    primary_nauc: float
    tn_at_fn95: int
    fn_at_tn68253: int | None

    def validate(self) -> None:
        if not math.isfinite(float(self.primary_nauc)) or not 0.0 <= float(self.primary_nauc) <= 1.0:
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "Endpoint nAUC is not finite or in [0,1]")
        if type(self.tn_at_fn95) is not int or self.tn_at_fn95 < 0:
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "Endpoint TN_at_FN95 is invalid")
        if self.fn_at_tn68253 is not None and (type(self.fn_at_tn68253) is not int or self.fn_at_tn68253 < 0):
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "Endpoint FN_at_TN68253 is invalid")


@dataclass(frozen=True, slots=True)
class PairedContrastResult:
    contrast_id: str
    treatment_id: str
    comparator_id: str
    paired_seed_ids: tuple[int, ...]
    treatment_endpoints: tuple[Mapping[str, object], ...]
    comparator_endpoints: tuple[Mapping[str, object], ...]
    primary_deltas: tuple[float, ...]
    mean_delta: float
    median_delta: float
    positive_count: int
    zero_count: int
    negative_count: int
    win_rate: float
    worst_seed: int
    worst_delta: float
    tn_at_fn95_deltas: tuple[int, ...]
    fn_at_tn68253_deltas: tuple[int | None, ...]
    dual_end_degradation_count: int
    unreachable_target_tn_seed_ids: tuple[int, ...]
    exact_sign_flip_p: float
    missing_pair_status: str
    scientific_state: str


def exact_paired_sign_flip(deltas: Sequence[float], *, alternative: str = "greater") -> float:
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be greater or two-sided")
    values = [float(value) for value in deltas]
    if any(not math.isfinite(value) for value in values):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Sign-flip input contains a non-finite delta")
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0
    observed = sum(nonzero) / len(nonzero)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        permutation = sum(sign * abs(delta) for sign, delta in zip(signs, nonzero)) / len(nonzero)
        if alternative == "greater" and permutation >= observed - 1e-15:
            extreme += 1
        elif alternative == "two-sided" and abs(permutation) >= abs(observed) - 1e-15:
            extreme += 1
    return extreme / (2 ** len(nonzero))


def paired_contrast(
    contrast_id: str,
    treatment: Mapping[int, Endpoint],
    comparator: Mapping[int, Endpoint],
    *,
    treatment_id: str = "TREATMENT",
    comparator_id: str = "COMPARATOR",
) -> PairedContrastResult:
    if not contrast_id.strip() or not treatment_id.strip() or not comparator_id.strip():
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Contrast and arm identities must be non-empty")
    if set(treatment) != set(comparator):
        raise SctsrError(
            ErrorCode.STATISTICS_PAIR_MISSING,
            "Paired contrast has a missing seed member",
            observed={"treatment": sorted(treatment), "comparator": sorted(comparator)},
        )
    seeds = tuple(sorted(treatment))
    if not seeds:
        raise SctsrError(ErrorCode.STATISTICS_PAIR_MISSING, "No paired seeds")
    for seed in seeds:
        if type(seed) is not int:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Training seed IDs must be integers")
        treatment[seed].validate()
        comparator[seed].validate()

    primary = tuple(float(treatment[seed].primary_nauc) - float(comparator[seed].primary_nauc) for seed in seeds)
    tn = tuple(treatment[seed].tn_at_fn95 - comparator[seed].tn_at_fn95 for seed in seeds)
    fn: list[int | None] = []
    dual_degradation = 0
    unreachable: list[int] = []
    for seed, tn_delta in zip(seeds, tn):
        treatment_fn = treatment[seed].fn_at_tn68253
        comparator_fn = comparator[seed].fn_at_tn68253
        if treatment_fn is None or comparator_fn is None:
            fn.append(None)
            unreachable.append(seed)
            continue
        delta = treatment_fn - comparator_fn
        fn.append(delta)
        if tn_delta < 0 and delta > 0:
            dual_degradation += 1
    worst_index = min(range(len(primary)), key=primary.__getitem__)
    return PairedContrastResult(
        contrast_id=contrast_id,
        treatment_id=treatment_id,
        comparator_id=comparator_id,
        paired_seed_ids=seeds,
        treatment_endpoints=tuple({"training_seed": seed, **asdict(treatment[seed])} for seed in seeds),
        comparator_endpoints=tuple({"training_seed": seed, **asdict(comparator[seed])} for seed in seeds),
        primary_deltas=primary,
        mean_delta=mean(primary),
        median_delta=median(primary),
        positive_count=sum(delta > 0 for delta in primary),
        zero_count=sum(delta == 0 for delta in primary),
        negative_count=sum(delta < 0 for delta in primary),
        win_rate=sum(delta > 0 for delta in primary) / len(primary),
        worst_seed=seeds[worst_index],
        worst_delta=primary[worst_index],
        tn_at_fn95_deltas=tn,
        fn_at_tn68253_deltas=tuple(fn),
        dual_end_degradation_count=dual_degradation,
        unreachable_target_tn_seed_ids=tuple(unreachable),
        exact_sign_flip_p=exact_paired_sign_flip(primary),
        missing_pair_status="PASS_COMPLETE_PAIRING",
        scientific_state="NOT_EVALUATED",
    )


def holm_adjust(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
    expected_contrast_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Holm alpha must be finite and in (0,1)")
    if not p_values:
        raise SctsrError(ErrorCode.STATISTICS_PAIR_MISSING, "Holm family is empty")
    if expected_contrast_ids is not None:
        expected = tuple(expected_contrast_ids)
        observed = tuple(sorted(p_values))
        if len(expected) != len(set(expected)) or set(observed) != set(expected):
            raise SctsrError(
                ErrorCode.STATISTICS_PAIR_MISSING,
                "Holm input is not the complete frozen contrast family",
                observed=sorted(observed),
                expected=sorted(expected),
            )
    normalized: dict[str, float] = {}
    for contrast_id, raw in p_values.items():
        value = float(raw)
        if not contrast_id.strip() or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Holm family contains an invalid contrast ID or raw p value", observed={contrast_id: raw})
        normalized[contrast_id] = value
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    active = True
    running_adjusted = 0.0
    output: list[dict[str, object]] = []
    for rank, (contrast_id, raw_p) in enumerate(ordered, start=1):
        divisor = family_size - rank + 1
        threshold = float(alpha) / divisor
        reject = active and raw_p <= threshold
        if not reject:
            active = False
        running_adjusted = max(running_adjusted, min(1.0, raw_p * divisor))
        output.append(
            {
                "contrast_id": contrast_id,
                "raw_p": raw_p,
                "holm_rank": rank,
                "holm_threshold": threshold,
                "holm_adjusted_p": running_adjusted,
                "reject": reject,
                "decision": "REJECT" if reject else "DO_NOT_REJECT",
                "familywise_alpha": float(alpha),
                "family_size": family_size,
            }
        )
    return output


def validate_discovery(result: PairedContrastResult) -> str:
    if len(result.paired_seed_ids) != 8 or result.missing_pair_status != "PASS_COMPLETE_PAIRING":
        return "FAIL_INCOMPLETE_DISCOVERY_PAIRING"
    if result.unreachable_target_tn_seed_ids:
        return "FAIL_UNREACHABLE_TARGET_TN"
    if result.dual_end_degradation_count > 0:
        return "MIXED_SAFETY_DEGRADATION"
    if result.positive_count >= 7:
        return "SUPPORTED_FOR_CONFIRMATION_NOT_EFFICACY"
    if result.negative_count >= 7:
        return "CONTRADICTED"
    return "MIXED"


def validate_confirmation(result: PairedContrastResult) -> str:
    if len(result.paired_seed_ids) != 14 or result.missing_pair_status != "PASS_COMPLETE_PAIRING":
        return "FAIL_INCOMPLETE_CONFIRMATION_PAIRING"
    if result.unreachable_target_tn_seed_ids:
        return "FAIL_UNREACHABLE_TARGET_TN"
    if result.positive_count < 12 or result.worst_delta < 0 or result.dual_end_degradation_count != 0:
        return "CONTRADICTED"
    return "SUPPORTED"
