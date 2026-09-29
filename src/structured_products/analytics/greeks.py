"""Common-random-number bump-and-revalue Greeks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..config import GreekConfig, MCConfig, PDEConfig
from ..engines.pde.grid import build_spot_grid
from ..enums import CouponRule, PricingMethod
from ..market import MarketData
from ..payoffs import VectorizedPayoffState, basket_performance
from ..pricing import price
from ..products import CouponFeature, StructuredNote
from ..results import GreeksResult


def _roll_product(
    product: StructuredNote, shift_days: int
) -> StructuredNote:
    if shift_days <= 0 or shift_days >= product.maturity_days:
        raise ValueError("theta shift must be inside the remaining maturity")
    autocall = product.autocall
    if autocall:
        keep = [
            day > shift_days for day in autocall.observation_days
        ]
        autocall = replace(
            autocall,
            observation_days=tuple(
                day - shift_days
                for day, include in zip(autocall.observation_days, keep)
                if include
            ),
            barrier_ratios=tuple(
                value
                for value, include in zip(autocall.barrier_ratios, keep)
                if include
            ),
            coupon_rates=tuple(
                value
                for value, include in zip(autocall.coupon_rates, keep)
                if include
            ),
        ) if any(keep) else None

    coupon = product.coupon
    accrued = product.coupon_accrued_days + shift_days
    if coupon.payment_days:
        paid = [day for day in coupon.payment_days if day <= shift_days]
        if paid:
            accrued = shift_days - paid[-1]
        keep_coupon = [day > shift_days for day in coupon.payment_days]
        coupon = replace(
            coupon,
            payment_days=tuple(
                day - shift_days
                for day, include in zip(coupon.payment_days, keep_coupon)
                if include
            ),
            rate_schedule=tuple(
                rate
                for rate, include in zip(coupon.rate_schedule, keep_coupon)
                if include
            ),
        ) if any(keep_coupon) else CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION)
    knock_in = product.knock_in
    if knock_in:
        knock_in = replace(
            knock_in,
            monitoring_start_day=max(1, knock_in.monitoring_start_day - shift_days),
        )
    return replace(
        product,
        maturity_days=product.maturity_days - shift_days,
        autocall=autocall,
        coupon=coupon,
        knock_in=knock_in,
        elapsed_days=product.elapsed_days + shift_days,
        coupon_accrued_days=accrued if coupon.payment_days else 0,
    )


def _roll_scenario(
    product: StructuredNote, market: MarketData, days: int
) -> tuple[StructuredNote | None, dict[str, Any]]:
    """Advance daily events along a flat-spot path; future bumps hold history fixed."""
    rolled = _roll_product(product, days)
    state = VectorizedPayoffState(product, 1, rate=0.0)
    performance = basket_performance(np.asarray([market.spots]), product)
    for day in range(1, days + 1):
        state.process_day(day, performance)
        if not state.alive[0]:
            break
    knocked_in = bool(state.knocked_in[0])
    redeemed = not bool(state.alive[0])
    if rolled.knock_in:
        rolled = replace(
            rolled,
            knock_in=replace(rolled.knock_in, initially_knocked_in=knocked_in),
        )
    return (None if redeemed else rolled), {
        "roll_convention": "constant_spot_daily_close_ex_cash",
        "roll_days": days,
        "roll_knocked_in": knocked_in,
        "roll_redeemed": redeemed,
        "roll_cashflow": float(state.present_values[0]),
    }


def _bump_correlation(
    market: MarketData, bump: float
) -> MarketData:
    matrix = np.asarray(market.correlation, dtype=float).copy()
    mask = ~np.eye(len(matrix), dtype=bool)
    matrix[mask] += bump
    return replace(market, correlation=tuple(tuple(row) for row in matrix))


def calculate_greeks(
    product: StructuredNote,
    market: MarketData,
    method: PricingMethod | str,
    engine_config: MCConfig | PDEConfig,
    greek_config: GreekConfig | None = None,
) -> GreeksResult:
    """Per-asset finite differences; Charm/Veta are per elapsed trading day.

    Volatility is decimal (0.01 = one vol point). Vanna and Vomma are
    own-asset derivatives, not cross-asset matrices. Lambda uses total PV.
    """
    base = price(product, market, method, engine_config).present_value
    return _calculate_greeks(product, market, method, engine_config, greek_config, base)


def _calculate_greeks(
    product: StructuredNote,
    market: MarketData,
    method: PricingMethod | str,
    engine_config: MCConfig | PDEConfig,
    greek_config: GreekConfig | None,
    base: float,
) -> GreeksResult:
    """Shared calculation, reusing a PV already obtained by the pricing workflow."""
    selected = method if isinstance(method, PricingMethod) else PricingMethod(method.lower())
    config = greek_config or GreekConfig()
    market.validate_for(product)
    if any(vol <= config.volatility_bump for vol in market.volatilities):
        raise ValueError("volatility bump requires base volatility above the bump")
    rolled, diagnostics = _roll_scenario(product, market, config.theta_days)
    if selected is PricingMethod.PDE:
        # Bumps must change market inputs, not the finite-difference domain.
        upper = float(build_spot_grid(product, market, engine_config)[-1])
        engine_config = replace(engine_config, spot_grid_upper=upper)
        diagnostics["spot_grid_upper"] = upper
    cache = {(product, market): base}

    def value(
        candidate_product: StructuredNote | None, candidate_market: MarketData
    ) -> float:
        if candidate_product is None:
            return 0.0
        key = (candidate_product, candidate_market)
        if key not in cache:
            cache[key] = price(
                candidate_product, candidate_market, selected, engine_config
            ).present_value
        return cache[key]

    deltas: list[float] = []
    gammas: list[float] = []
    vegas: list[float] = []
    vannas: list[float] = []
    vommas: list[float] = []
    charms: list[float] = []
    vetas: list[float] = []

    for index, spot in enumerate(market.spots):
        bump = spot * config.relative_spot_bump
        up_spots = list(market.spots)
        down_spots = list(market.spots)
        up_spots[index] += bump
        down_spots[index] -= bump
        up_market = replace(market, spots=tuple(up_spots))
        down_market = replace(market, spots=tuple(down_spots))
        up = value(product, up_market)
        down = value(product, down_market)
        deltas.append((up - down) / (2.0 * bump))
        gammas.append((up - 2.0 * base + down) / bump**2)

        up_vols = list(market.volatilities)
        down_vols = list(market.volatilities)
        up_vols[index] += config.volatility_bump
        down_vols[index] -= config.volatility_bump
        high_vol = replace(market, volatilities=tuple(up_vols))
        low_vol = replace(market, volatilities=tuple(down_vols))
        up_vega = value(product, high_vol)
        down_vega = value(product, low_vol)
        vegas.append((up_vega - down_vega) / (2.0 * config.volatility_bump))
        vommas.append((up_vega - 2.0 * base + down_vega) / config.volatility_bump**2)
        vannas.append(
            (
                value(product, replace(high_vol, spots=tuple(up_spots)))
                - value(product, replace(low_vol, spots=tuple(up_spots)))
                - value(product, replace(high_vol, spots=tuple(down_spots)))
                + value(product, replace(low_vol, spots=tuple(down_spots)))
            ) / (4.0 * bump * config.volatility_bump)
        )
        rolled_delta = (
            value(rolled, up_market) - value(rolled, down_market)
        ) / (2.0 * bump)
        rolled_vega = (
            value(rolled, high_vol) - value(rolled, low_vol)
        ) / (2.0 * config.volatility_bump)
        charms.append((rolled_delta - deltas[-1]) / config.theta_days)
        vetas.append((rolled_vega - vegas[-1]) / config.theta_days)

    up_rate = value(product, replace(market, rate=market.rate + config.rate_bump))
    down_rate = value(product, replace(market, rate=market.rate - config.rate_bump))
    rho = (up_rate - down_rate) / (2.0 * config.rate_bump)

    theta = value(rolled, market) - base

    correlation_sensitivity: float | None = None
    if product.n_assets > 1:
        try:
            up_correlation = _bump_correlation(market, config.correlation_bump)
            down_correlation = _bump_correlation(market, -config.correlation_bump)
        except ValueError as exc:
            raise ValueError(
                "correlation bump does not preserve a positive-semidefinite matrix"
            ) from exc
        correlation_sensitivity = (
            value(product, up_correlation) - value(product, down_correlation)
        ) / (2.0 * config.correlation_bump)

    warnings = [
        "Finite-difference risk: check bump and numerical convergence "
        "near barriers and event dates."
    ]
    if selected is PricingMethod.MC:
        warnings.append(
            "MC Greeks use common random numbers; price confidence intervals "
            "are not Greek confidence intervals."
        )
    near_zero = abs(base) <= product.notional * 1.0e-12
    if near_zero:
        warnings.append(
            "PV is near zero; elasticity (Lambda/Omega) is undefined and returned as null."
        )
    if diagnostics["roll_cashflow"] or diagnostics["roll_knocked_in"] != bool(
        product.knock_in and product.knock_in.initially_knocked_in
    ):
        warnings.append(
            "Time roll crosses a cashflow or knock-in event; time risk uses "
            "remaining ex-cash value and fixed rolled history."
        )
    diagnostics.update({
        "elasticity_basis": "total_present_value",
        "cross_asset_derivatives": False,
        "unique_valuations": len(cache),
        "time_direction": "elapsed_time",
    })
    return GreeksResult(
        method=selected.value,
        base_price=base,
        delta=tuple(deltas),
        gamma=tuple(gammas),
        vega=tuple(vegas),
        theta=theta,
        rho=rho,
        correlation_sensitivity=correlation_sensitivity,
        lambda_=tuple(
            None if near_zero else d * s / base for d, s in zip(deltas, market.spots)
        ),
        vanna=tuple(vannas),
        vomma=tuple(vommas),
        charm=tuple(charms),
        veta=tuple(vetas),
        units={
            "delta": "PV / spot",
            "gamma": "PV / spot^2",
            "vega": "PV / decimal_volatility",
            "vanna": "PV / (spot * decimal_volatility)",
            "vomma": "PV / decimal_volatility^2",
            "volga": "PV / decimal_volatility^2",
            "lambda": "dimensionless",
            "omega": "dimensionless",
            "charm": "delta / trading_day",
            "veta": "vega / trading_day",
            "theta": "PV change over roll_days (ex-cash)",
            "rho": "PV / decimal_rate",
            "correlation_sensitivity": "PV / uniform_off_diagonal_correlation_shift",
        },
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        bumps={
            "relative_spot": config.relative_spot_bump,
            "volatility": config.volatility_bump,
            "rate": config.rate_bump,
            "theta_days": float(config.theta_days),
            "correlation": config.correlation_bump,
        },
    )
