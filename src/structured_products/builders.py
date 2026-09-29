"""User-facing templates that only assemble composable product features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .enums import BasketRule, CouponRule
from .products import (
    AutocallFeature,
    CouponFeature,
    KnockInFeature,
    RedemptionFeature,
    StructuredNote,
)
from .schedules import (
    flat_schedule,
    linear_schedule,
    monthly_observation_days,
    quarterly_payment_days,
)


def _metadata(name: str) -> tuple[tuple[str, str], ...]:
    return (("template", name),)


def _snowball(
    *,
    name: str,
    reference_spots: Sequence[float],
    basket_rule: BasketRule,
    notional: float,
    issue_price: float | None,
    maturity_days: int,
    day_count: int,
    knock_in_ratio: float,
    observation_days: Sequence[int] | None,
    barrier_ratios: Sequence[float],
    coupon_rates: Sequence[float],
    maturity_coupon_rate: float,
) -> StructuredNote:
    days = tuple(
        observation_days
        if observation_days is not None
        else monthly_observation_days(maturity_days, day_count=day_count)
    )
    barriers = tuple(float(value) for value in barrier_ratios)
    coupons = tuple(float(value) for value in coupon_rates)
    if len(days) != len(barriers) or len(days) != len(coupons):
        raise ValueError("observation, barrier, and coupon schedules must align")
    return StructuredNote(
        reference_spots=tuple(reference_spots),
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        basket_rule=basket_rule,
        knock_in=KnockInFeature(knock_in_ratio),
        autocall=AutocallFeature(days, barriers, coupons),
        coupon=CouponFeature(
            rule=CouponRule.CONTINGENT_AT_REDEMPTION,
            maturity_rate=maturity_coupon_rate,
            survives_knock_in=False,
        ),
        redemption=RedemptionFeature(),
        metadata=_metadata(name),
    )


def make_classic_snowball(
    *,
    reference_spot: float = 100.0,
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.80,
    knock_out_ratio: float = 1.00,
    coupon_rate: float = 0.18,
    maturity_coupon_rate: float | None = None,
    observation_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        observation_days
        if observation_days is not None
        else monthly_observation_days(maturity_days, day_count=day_count)
    )
    maturity_rate = coupon_rate if maturity_coupon_rate is None else maturity_coupon_rate
    return _snowball(
        name="classic_snowball",
        reference_spots=(reference_spot,),
        basket_rule=BasketRule.SINGLE,
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        knock_in_ratio=knock_in_ratio,
        observation_days=days,
        barrier_ratios=flat_schedule(knock_out_ratio, len(days)),
        coupon_rates=flat_schedule(coupon_rate, len(days)),
        maturity_coupon_rate=maturity_rate,
    )


def make_wide_snowball(
    *,
    reference_spot: float = 100.0,
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.70,
    knock_out_ratio: float = 1.02,
    coupon_rate: float = 0.12,
    observation_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        observation_days
        if observation_days is not None
        else monthly_observation_days(maturity_days, day_count=day_count)
    )
    return _snowball(
        name="wide_snowball",
        reference_spots=(reference_spot,),
        basket_rule=BasketRule.SINGLE,
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        knock_in_ratio=knock_in_ratio,
        observation_days=days,
        barrier_ratios=flat_schedule(knock_out_ratio, len(days)),
        coupon_rates=flat_schedule(coupon_rate, len(days)),
        maturity_coupon_rate=coupon_rate,
    )


def make_butterfly_snowball(
    *,
    reference_spot: float = 100.0,
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.80,
    knock_out_ratio: float = 1.00,
    front_coupon_rate: float = 0.22,
    back_coupon_rate: float = 0.10,
    observation_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        observation_days
        if observation_days is not None
        else monthly_observation_days(maturity_days, day_count=day_count)
    )
    coupons = linear_schedule(front_coupon_rate, back_coupon_rate, len(days))
    return _snowball(
        name="butterfly_snowball",
        reference_spots=(reference_spot,),
        basket_rule=BasketRule.SINGLE,
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        knock_in_ratio=knock_in_ratio,
        observation_days=days,
        barrier_ratios=flat_schedule(knock_out_ratio, len(days)),
        coupon_rates=coupons,
        maturity_coupon_rate=coupons[-1],
    )


def make_dividend_snowball(
    *,
    reference_spot: float = 100.0,
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.80,
    knock_out_ratio: float = 1.00,
    knock_out_coupon_rate: float = 0.20,
    maturity_coupon_rate: float = 0.08,
    observation_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        observation_days
        if observation_days is not None
        else monthly_observation_days(maturity_days, day_count=day_count)
    )
    return _snowball(
        name="dividend_snowball",
        reference_spots=(reference_spot,),
        basket_rule=BasketRule.SINGLE,
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        knock_in_ratio=knock_in_ratio,
        observation_days=days,
        barrier_ratios=flat_schedule(knock_out_ratio, len(days)),
        coupon_rates=flat_schedule(knock_out_coupon_rate, len(days)),
        maturity_coupon_rate=maturity_coupon_rate,
    )


def make_stepdown_snowball(
    *,
    reference_spot: float = 100.0,
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.80,
    first_knock_out_ratio: float = 1.03,
    last_knock_out_ratio: float = 0.92,
    coupon_rate: float = 0.18,
    observation_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        observation_days
        if observation_days is not None
        else monthly_observation_days(maturity_days, day_count=day_count)
    )
    return _snowball(
        name="stepdown_snowball",
        reference_spots=(reference_spot,),
        basket_rule=BasketRule.SINGLE,
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        knock_in_ratio=knock_in_ratio,
        observation_days=days,
        barrier_ratios=linear_schedule(
            first_knock_out_ratio, last_knock_out_ratio, len(days)
        ),
        coupon_rates=flat_schedule(coupon_rate, len(days)),
        maturity_coupon_rate=coupon_rate,
    )


def make_standard_fcn(
    *,
    reference_spot: float = 100.0,
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.80,
    strike_ratio: float = 1.00,
    coupon_rate: float = 0.12,
    payment_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        payment_days
        if payment_days is not None
        else quarterly_payment_days(maturity_days, day_count=day_count)
    )
    return StructuredNote(
        reference_spots=(reference_spot,),
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        knock_in=KnockInFeature(knock_in_ratio),
        coupon=CouponFeature(
            rule=CouponRule.FIXED_PERIODIC,
            payment_days=days,
            rate_schedule=flat_schedule(coupon_rate, len(days)),
            survives_knock_in=True,
        ),
        redemption=RedemptionFeature(strike_ratio=strike_ratio),
        metadata=_metadata("standard_fcn"),
    )


def make_discount_entry_fcn(**kwargs: object) -> StructuredNote:
    product = make_standard_fcn(**kwargs)
    assert product.knock_in is not None
    return replace(
        product,
        knock_in=replace(product.knock_in, initially_knocked_in=True),
        metadata=_metadata("discount_entry_fcn"),
    )


def make_worst_of_snowball(
    *,
    reference_spots: Sequence[float] = (100.0, 100.0),
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.65,
    knock_out_ratio: float = 1.00,
    coupon_rate: float = 0.20,
    observation_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        observation_days
        if observation_days is not None
        else monthly_observation_days(maturity_days, day_count=day_count)
    )
    return _snowball(
        name="worst_of_snowball",
        reference_spots=reference_spots,
        basket_rule=BasketRule.WORST_OF,
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        knock_in_ratio=knock_in_ratio,
        observation_days=days,
        barrier_ratios=flat_schedule(knock_out_ratio, len(days)),
        coupon_rates=flat_schedule(coupon_rate, len(days)),
        maturity_coupon_rate=coupon_rate,
    )


def make_worst_of_fcn(
    *,
    reference_spots: Sequence[float] = (100.0, 100.0),
    notional: float = 1_000.0,
    issue_price: float | None = None,
    maturity_days: int = 252,
    day_count: int = 252,
    knock_in_ratio: float = 0.70,
    strike_ratio: float = 1.00,
    coupon_rate: float = 0.15,
    payment_days: Sequence[int] | None = None,
) -> StructuredNote:
    days = tuple(
        payment_days
        if payment_days is not None
        else quarterly_payment_days(maturity_days, day_count=day_count)
    )
    return StructuredNote(
        reference_spots=tuple(reference_spots),
        notional=notional,
        issue_price=issue_price,
        maturity_days=maturity_days,
        day_count=day_count,
        basket_rule=BasketRule.WORST_OF,
        knock_in=KnockInFeature(knock_in_ratio),
        coupon=CouponFeature(
            rule=CouponRule.FIXED_PERIODIC,
            payment_days=days,
            rate_schedule=flat_schedule(coupon_rate, len(days)),
            survives_knock_in=True,
        ),
        redemption=RedemptionFeature(strike_ratio=strike_ratio),
        metadata=_metadata("worst_of_fcn"),
    )
