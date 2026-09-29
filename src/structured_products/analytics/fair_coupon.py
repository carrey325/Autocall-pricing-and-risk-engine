"""Bracket-preserving fair-coupon solving."""

from __future__ import annotations

from dataclasses import replace
import sys

from scipy.optimize import brentq

from ..config import MCConfig, PDEConfig
from ..enums import CouponRule, PricingMethod
from ..market import MarketData
from ..pricing import price
from ..products import StructuredNote
from ..results import FairCouponResult


def with_flat_coupon(product: StructuredNote, coupon_rate: float) -> StructuredNote:
    if coupon_rate < 0:
        raise ValueError("coupon rate cannot be negative")
    coupon = product.coupon
    autocall = product.autocall
    if coupon.rule is CouponRule.FIXED_PERIODIC:
        coupon = replace(
            coupon,
            rate_schedule=tuple(coupon_rate for _ in coupon.rate_schedule),
        )
    else:
        coupon = replace(coupon, maturity_rate=coupon_rate)
    if autocall:
        autocall = replace(
            autocall,
            coupon_rates=tuple(coupon_rate for _ in autocall.coupon_rates),
        )
    return replace(product, coupon=coupon, autocall=autocall)


def solve_fair_coupon(
    product: StructuredNote,
    market: MarketData,
    target_pv: float,
    method: PricingMethod | str,
    engine_config: MCConfig | PDEConfig,
    lower_coupon: float,
    upper_coupon: float,
    *,
    max_iterations: int = 100,
    tolerance: float = 1.0e-8,
) -> FairCouponResult:
    if lower_coupon < 0 or upper_coupon <= lower_coupon:
        raise ValueError("coupon bracket must satisfy 0 <= lower < upper")
    selected = method if isinstance(method, PricingMethod) else PricingMethod(method.lower())

    def objective(rate: float) -> float:
        candidate = with_flat_coupon(product, rate)
        return price(candidate, market, selected, engine_config).present_value - target_pv

    lower_value = objective(lower_coupon)
    upper_value = objective(upper_coupon)
    if lower_value == 0:
        return FairCouponResult(
            selected.value, lower_coupon, target_pv, 0.0, 0, (lower_coupon, upper_coupon)
        )
    if upper_value == 0:
        return FairCouponResult(
            selected.value, upper_coupon, target_pv, 0.0, 0, (lower_coupon, upper_coupon)
        )
    if lower_value * upper_value > 0:
        raise ValueError("coupon interval does not bracket the target present value")
    root, details = brentq(
        objective,
        lower_coupon,
        upper_coupon,
        xtol=tolerance,
        rtol=max(tolerance, 4.0 * sys.float_info.epsilon),
        maxiter=max_iterations,
        full_output=True,
        disp=False,
    )
    if not details.converged:
        raise RuntimeError("fair-coupon solver did not converge")
    residual = objective(float(root))
    return FairCouponResult(
        method=selected.value,
        coupon_rate=float(root),
        target_present_value=float(target_pv),
        residual=float(residual),
        iterations=int(details.iterations),
        bracket=(float(lower_coupon), float(upper_coupon)),
    )
