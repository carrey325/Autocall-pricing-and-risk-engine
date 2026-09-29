"""Composable, immutable structured-note specifications."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import BasketRule, CouponRule, SameDayPriority
from .schedules import validate_days


@dataclass(frozen=True)
class KnockInFeature:
    barrier_ratio: float
    monitoring_start_day: int = 1
    initially_knocked_in: bool = False

    def __post_init__(self) -> None:
        if self.barrier_ratio <= 0:
            raise ValueError("knock-in barrier ratio must be positive")
        if self.monitoring_start_day <= 0:
            raise ValueError("knock-in monitoring start day must be positive")


@dataclass(frozen=True)
class AutocallFeature:
    observation_days: tuple[int, ...]
    barrier_ratios: tuple[float, ...]
    coupon_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.observation_days)
        if size == 0:
            raise ValueError("autocall requires at least one observation")
        if len(self.barrier_ratios) != size or len(self.coupon_rates) != size:
            raise ValueError("autocall schedules must align with observation days")
        if any(value <= 0 for value in self.barrier_ratios):
            raise ValueError("autocall barriers must be positive")


@dataclass(frozen=True)
class CouponFeature:
    rule: CouponRule
    payment_days: tuple[int, ...] = ()
    rate_schedule: tuple[float, ...] = ()
    maturity_rate: float = 0.0
    survives_knock_in: bool = False

    def __post_init__(self) -> None:
        if self.maturity_rate < 0 or any(rate < 0 for rate in self.rate_schedule):
            raise ValueError("coupon rates cannot be negative")
        if self.rule is CouponRule.FIXED_PERIODIC:
            if not self.payment_days or len(self.payment_days) != len(self.rate_schedule):
                raise ValueError("periodic coupon rates must align with payment days")
        elif self.payment_days or self.rate_schedule:
            raise ValueError("contingent coupons cannot have periodic payment schedules")


@dataclass(frozen=True)
class RedemptionFeature:
    strike_ratio: float = 1.0
    downside_participation: float = 1.0
    include_principal: bool = True
    principal_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.strike_ratio <= 0:
            raise ValueError("strike ratio must be positive")
        if self.downside_participation < 0:
            raise ValueError("downside participation cannot be negative")
        if not 0 <= self.principal_floor <= 1:
            raise ValueError("principal floor must be in [0, 1]")


@dataclass(frozen=True)
class StructuredNote:
    reference_spots: tuple[float, ...]
    notional: float
    maturity_days: int
    coupon: CouponFeature
    redemption: RedemptionFeature
    issue_price: float | None = None
    day_count: int = 252
    basket_rule: BasketRule = BasketRule.SINGLE
    knock_in: KnockInFeature | None = None
    autocall: AutocallFeature | None = None
    same_day_priority: SameDayPriority = SameDayPriority.KNOCK_OUT_FIRST
    metadata: tuple[tuple[str, str], ...] = ()
    elapsed_days: int = 0
    coupon_accrued_days: int = 0

    def __post_init__(self) -> None:
        refs = tuple(float(value) for value in self.reference_spots)
        object.__setattr__(self, "reference_spots", refs)
        if not refs or any(value <= 0 for value in refs):
            raise ValueError("reference spots must be positive")
        if self.notional <= 0 or self.maturity_days <= 0 or self.day_count <= 0:
            raise ValueError("notional, maturity_days, and day_count must be positive")
        if (
            not isinstance(self.elapsed_days, int)
            or not isinstance(self.coupon_accrued_days, int)
            or not 0 <= self.coupon_accrued_days <= self.elapsed_days
        ):
            raise ValueError("accrual days must be integers with 0 <= coupon_accrued_days <= elapsed_days")
        if self.issue_price is None:
            object.__setattr__(self, "issue_price", float(self.notional))
        elif self.issue_price <= 0:
            raise ValueError("issue price must be positive")
        if self.basket_rule is BasketRule.SINGLE and len(refs) != 1:
            raise ValueError("single-asset products require exactly one reference spot")
        if self.knock_in and self.knock_in.monitoring_start_day > self.maturity_days:
            raise ValueError("knock-in monitoring cannot start after maturity")
        if self.autocall:
            validate_days(
                self.autocall.observation_days,
                self.maturity_days,
                name="autocall observation",
            )
        if self.coupon.rule is CouponRule.FIXED_PERIODIC:
            validate_days(self.coupon.payment_days, self.maturity_days, name="coupon payment")

    @property
    def n_assets(self) -> int:
        return len(self.reference_spots)

    @property
    def maturity_years(self) -> float:
        return self.maturity_days / self.day_count

    @property
    def coupon_maturity_years(self) -> float:
        """Full issue-to-maturity accrual, including elapsed contractual days."""
        return (self.elapsed_days + self.maturity_days) / self.day_count

    @property
    def product_name(self) -> str:
        return dict(self.metadata).get("template", "structured_note")
