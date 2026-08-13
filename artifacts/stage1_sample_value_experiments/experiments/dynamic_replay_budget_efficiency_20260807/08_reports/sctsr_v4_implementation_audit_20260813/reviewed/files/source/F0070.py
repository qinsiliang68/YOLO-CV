from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import gcd

from .errors import ErrorCode, SctsrError


class RateSemantic(str, Enum):
    IDENTITY_POOL_RATE = "IDENTITY_POOL_RATE"
    PER_EPOCH_REPLAY_RATE = "PER_EPOCH_REPLAY_RATE"


class DenominatorRole(str, Enum):
    CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE = "CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE"


@dataclass(frozen=True, slots=True)
class ReplayRateSpec:
    numerator: int
    denominator: int
    semantic: RateSemantic
    denominator_role: DenominatorRole = DenominatorRole.CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE

    def __post_init__(self) -> None:
        if isinstance(self.numerator, bool) or isinstance(self.denominator, bool):
            raise SctsrError(ErrorCode.FLOAT_RATE_FORBIDDEN, "Boolean values are not valid rates")
        if not isinstance(self.numerator, int) or not isinstance(self.denominator, int):
            raise SctsrError(ErrorCode.FLOAT_RATE_FORBIDDEN, "Replay rates must be integer rational values")
        if self.numerator < 0 or self.denominator <= 0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Invalid replay rate numerator or denominator")
        if self.denominator_role is not DenominatorRole.CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE:
            raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Replay rate denominator role is not canonical base exposure")

    def reduced(self) -> "ReplayRateSpec":
        divisor = gcd(self.numerator, self.denominator)
        return ReplayRateSpec(self.numerator // divisor, self.denominator // divisor, self.semantic, self.denominator_role)

    def canonical_token(self) -> str:
        reduced = self.reduced()
        return f"{reduced.numerator}/{reduced.denominator}"

    def derive_count(self, base_denominator: int) -> int:
        if isinstance(base_denominator, bool) or not isinstance(base_denominator, int) or base_denominator <= 0:
            raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Base denominator must be a positive integer")
        product = base_denominator * self.numerator
        count, remainder = divmod(product, self.denominator)
        if remainder:
            raise SctsrError(
                ErrorCode.RATE_NOT_INTEGRAL,
                "Replay rate does not derive an integral count from the frozen denominator",
                observed={"base_denominator": base_denominator, "rate": self.canonical_token()},
                expected="integral count without rounding",
            )
        return count

    def as_dict(self) -> dict[str, str | int]:
        return {"numerator": self.numerator, "denominator": self.denominator, "semantic": self.semantic.value, "denominator_role": self.denominator_role.value, "canonical_token": self.canonical_token()}

    def to_basis_points(self) -> int:
        value = Fraction(self.numerator, self.denominator) * 10_000
        if value.denominator != 1:
            raise SctsrError(ErrorCode.RATE_NOT_INTEGRAL, "Rate cannot be represented as integral basis points")
        return int(value)


IDENTITY_POOL_RATE = ReplayRateSpec(25, 1000, RateSemantic.IDENTITY_POOL_RATE)
U_RATE = ReplayRateSpec(5, 1000, RateSemantic.PER_EPOCH_REPLAY_RATE)
F_RATE = ReplayRateSpec(10, 1000, RateSemantic.PER_EPOCH_REPLAY_RATE)
F_ACTIVE_RATE = F_RATE
ZERO_RATE = ReplayRateSpec(0, 1000, RateSemantic.PER_EPOCH_REPLAY_RATE)
