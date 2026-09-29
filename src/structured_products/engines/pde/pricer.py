"""Generic two-state single-asset Crank-Nicolson pricer."""

from __future__ import annotations

from math import exp

import numpy as np

from ...config import PDEConfig
from ...enums import CouponRule
from ...market import MarketData
from ...payoffs import _period_accrual, evaluate_path
from ...products import StructuredNote
from ...results import PricingResult
from .grid import build_spot_grid
from .operators import CrankNicolsonStepper, build_stepper


def support_matrix() -> dict[str, tuple[str, ...]]:
    return {
        "pde": (
            "classic_snowball",
            "wide_snowball",
            "butterfly_snowball",
            "dividend_snowball",
            "stepdown_snowball",
            "standard_fcn",
            "discount_entry_fcn",
        ),
        "pde_rejected": ("worst_of_snowball", "worst_of_fcn"),
    }


def _validate(product: StructuredNote, market: MarketData) -> None:
    market.validate_for(product)
    if product.n_assets != 1:
        raise ValueError("PDE supports single-asset products only")


def _periodic_coupon(product: StructuredNote, index: int) -> float:
    accrual = _period_accrual(product, index)
    return product.notional * product.coupon.rate_schedule[index] * accrual


def _redemption_values(
    product: StructuredNote, performance: np.ndarray, knocked_in: np.ndarray
) -> np.ndarray:
    values = np.zeros_like(performance)
    if product.redemption.include_principal:
        ratios = np.ones_like(performance)
        loss = knocked_in & (performance < product.redemption.strike_ratio)
        ratios[loss] = np.maximum(
            product.redemption.principal_floor,
            1.0
            + product.redemption.downside_participation
            * (
                performance[loss] / product.redemption.strike_ratio
                - 1.0
            ),
        )
        values += product.notional * ratios
    if product.coupon.rule is CouponRule.CONTINGENT_AT_REDEMPTION:
        eligible = (~knocked_in) | product.coupon.survives_knock_in
        values[eligible] += (
            product.notional
            * product.coupon.maturity_rate
            * product.coupon_maturity_years
        )
    return values


def _terminal_values(
    product: StructuredNote, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    performance = grid / product.reference_spots[0]
    state_not_ki = np.zeros_like(performance, dtype=bool)
    state_ki = np.ones_like(performance, dtype=bool)
    if product.knock_in and product.maturity_days >= product.knock_in.monitoring_start_day:
        state_not_ki |= performance <= product.knock_in.barrier_ratio

    not_ki = _redemption_values(product, performance, state_not_ki)
    ki = _redemption_values(product, performance, state_ki)

    if (
        product.coupon.rule is CouponRule.FIXED_PERIODIC
        and product.maturity_days in product.coupon.payment_days
    ):
        index = product.coupon.payment_days.index(product.maturity_days)
        amount = _periodic_coupon(product, index)
        not_ki += np.where(
            (~state_not_ki) | product.coupon.survives_knock_in, amount, 0.0
        )
        if product.coupon.survives_knock_in:
            ki += amount

    if product.autocall and product.maturity_days in product.autocall.observation_days:
        index = product.autocall.observation_days.index(product.maturity_days)
        mask = performance >= product.autocall.barrier_ratios[index]
        principal = product.notional if product.redemption.include_principal else 0.0
        coupon = (
            product.notional
            * product.autocall.coupon_rates[index]
            * product.coupon_maturity_years
        )
        not_ki[mask] = principal + coupon
        ki[mask] = principal + coupon
    return not_ki, ki


def _apply_events(
    product: StructuredNote,
    grid: np.ndarray,
    day: int,
    not_ki: np.ndarray,
    ki: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    performance = grid / product.reference_spots[0]
    not_ki = not_ki.copy()
    ki = ki.copy()

    if (
        product.coupon.rule is CouponRule.FIXED_PERIODIC
        and day in product.coupon.payment_days
    ):
        index = product.coupon.payment_days.index(day)
        amount = _periodic_coupon(product, index)
        not_ki += amount
        if product.coupon.survives_knock_in:
            ki += amount

    if product.knock_in and day >= product.knock_in.monitoring_start_day:
        mask = performance <= product.knock_in.barrier_ratio
        not_ki[mask] = ki[mask]

    if product.autocall and day in product.autocall.observation_days:
        index = product.autocall.observation_days.index(day)
        mask = performance >= product.autocall.barrier_ratios[index]
        principal = product.notional if product.redemption.include_principal else 0.0
        coupon = (
            product.notional
            * product.autocall.coupon_rates[index]
            * (product.elapsed_days + day)
            / product.day_count
        )
        not_ki[mask] = principal + coupon
        ki[mask] = principal + coupon
    return not_ki, ki


def _deterministic_price(
    product: StructuredNote, market: MarketData, config: PDEConfig
) -> PricingResult:
    times = np.arange(product.maturity_days + 1, dtype=float) / product.day_count
    path = (
        market.spots[0]
        * np.exp((market.rate - market.dividend_yields[0]) * times)
    )[:, None]
    evaluation = evaluate_path(path, product)
    value = evaluation.present_value(market.rate, product.day_count)
    return PricingResult(
        method="pde",
        present_value=value,
        net_value=value - float(product.issue_price),
        diagnostics={
            "deterministic_zero_volatility": True,
            "spot_grid_points": config.spot_grid_points,
        },
        warnings=("deterministic zero-volatility shortcut",),
    )


def price_pde(
    product: StructuredNote,
    market: MarketData,
    config: PDEConfig,
) -> PricingResult:
    _validate(product, market)
    if market.volatilities[0] == 0:
        return _deterministic_price(product, market, config)

    grid = build_spot_grid(product, market, config)
    not_ki, ki = _terminal_values(product, grid)
    regular_dt = 1.0 / (product.day_count * config.time_steps_per_day)
    cache: dict[tuple[float, float], CrankNicolsonStepper] = {}

    def step(values: np.ndarray, dt: float, theta: float) -> np.ndarray:
        key = (dt, theta)
        if key not in cache:
            cache[key] = build_stepper(
                grid,
                rate=market.rate,
                dividend_yield=market.dividend_yields[0],
                volatility=market.volatilities[0],
                dt=dt,
                theta=theta,
            )
        return cache[key].step(values)

    discontinuities = {product.maturity_days}
    if product.autocall:
        discontinuities.update(product.autocall.observation_days)
    discontinuities.update(product.coupon.payment_days)

    for day in range(product.maturity_days - 1, -1, -1):
        for substep in range(config.time_steps_per_day):
            first_after_event = substep == 0 and day + 1 in discontinuities
            if config.rannacher_smoothing and first_after_event:
                half_dt = regular_dt / 2.0
                not_ki = step(step(not_ki, half_dt, 1.0), half_dt, 1.0)
                ki = step(step(ki, half_dt, 1.0), half_dt, 1.0)
            else:
                not_ki = step(not_ki, regular_dt, config.theta)
                ki = step(ki, regular_dt, config.theta)
        if day > 0:
            not_ki, ki = _apply_events(product, grid, day, not_ki, ki)

    initial_values = (
        ki
        if product.knock_in and product.knock_in.initially_knocked_in
        else not_ki
    )
    value = float(np.interp(market.spots[0], grid, initial_values))
    return PricingResult(
        method="pde",
        present_value=value,
        net_value=value - float(product.issue_price),
        diagnostics={
            "spot_grid_points": config.spot_grid_points,
            "time_steps_per_day": config.time_steps_per_day,
            "upper_spot": float(grid[-1]),
            "theta": config.theta,
            "rannacher_smoothing": config.rannacher_smoothing,
            "interpolation": config.interpolation,
            "state": (
                "knocked_in"
                if product.knock_in and product.knock_in.initially_knocked_in
                else "not_knocked_in"
            ),
        },
    )
