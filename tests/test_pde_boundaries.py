"""Independent checks of the absorbing endpoint and its contractual cashflows."""

from dataclasses import replace
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.stats import norm

from structured_products import (
    CouponFeature, CouponRule, KnockInFeature, MarketData, PDEConfig,
    RedemptionFeature, StructuredNote, price,
)
from structured_products.engines.pde.operators import build_stepper


@pytest.mark.parametrize("rate", [0.0, 0.03, -0.01])
@pytest.mark.parametrize("theta", [0.5, 1.0])
@pytest.mark.parametrize("dt", [1 / 252, 1 / 504])
def test_lower_endpoint_discounts_input_value(rate, theta, dt):
    grid = np.linspace(0, 400, 101)
    values = 17 + 0.5 * grid
    stepper = build_stepper(grid, rate=rate, dividend_yield=0.01,
                            volatility=0.25, dt=dt, theta=theta)
    expected = 17 * exp(-rate * dt)
    # This tolerance tests the endpoint equation, not global PDE accuracy.
    assert stepper.step(values)[0] == pytest.approx(
        expected, rel=0, abs=1e-11 * max(1, abs(expected))
    )


def test_zero_endpoint_is_not_copied_from_neighbor():
    grid = np.linspace(0, 400, 101)
    stepper = build_stepper(grid, rate=0.03, dividend_yield=0.0,
                            volatility=0.25, dt=1 / 252, theta=0.5)
    result = stepper.step(10 * grid)
    assert result[0] == pytest.approx(0, abs=1e-11)
    assert result[1] > 0


def call_value(spot, strike, rate, dividend, volatility, time):
    if strike == 0:
        return spot * exp(-dividend * time)
    d1 = (log(spot / strike) + (rate - dividend + volatility**2 / 2) * time) / (
        volatility * sqrt(time)
    )
    d2 = d1 - volatility * sqrt(time)
    return spot * exp(-dividend * time) * norm.cdf(d1) - strike * exp(-rate * time) * norm.cdf(d2)


def principal_oracle(note, market):
    """Independent bond plus call-spread value; no production payoff helpers."""
    if not note.redemption.include_principal:
        return 0.0
    time = note.maturity_days / note.day_count
    participation = note.redemption.downside_participation
    if participation == 0:
        return note.notional * exp(-market.rate * time)
    strike = note.reference_spots[0] * note.redemption.strike_ratio
    floor = note.redemption.principal_floor
    lower = max(0, strike * (floor + participation - 1) / participation)
    calls = [call_value(market.spots[0], k, market.rate, market.dividend_yields[0],
                        market.volatilities[0], time) for k in (lower, strike)]
    return note.notional * (
        max(floor, 1 - participation) * exp(-market.rate * time)
        + participation / strike * (calls[0] - calls[1])
    )


def low_spot_note(**updates):
    note = StructuredNote(
        reference_spots=(100.0,), notional=1000.0, maturity_days=126,
        coupon=CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION),
        redemption=RedemptionFeature(),
        knock_in=KnockInFeature(0.8, initially_knocked_in=True),
    )
    return replace(note, **updates)


def low_market(spot=0.01):
    return MarketData((spot,), 0.03, (0.0,), (0.25,))


@pytest.mark.parametrize("points", [401, 801])
def test_low_spot_initial_ki_note_matches_call_spread_oracle(points):
    note, market = low_spot_note(), low_market()
    expected = principal_oracle(note, market)
    result = price(note, market, "pde", PDEConfig(spot_grid_points=points, spot_grid_upper=400))
    assert expected == pytest.approx(0.1, rel=0, abs=1e-12)
    # Locally affine payoff, zero dividends, and a remote cap: a fixture-specific bound.
    assert result.present_value == pytest.approx(expected, rel=0, abs=1e-6)


@pytest.mark.parametrize("points", [401, 801])
def test_low_spot_delta_remains_ten(points):
    config = PDEConfig(spot_grid_points=points, spot_grid_upper=400)
    values = [price(low_spot_note(), low_market(spot), "pde", config).present_value
              for spot in (0.009, 0.010, 0.011)]
    assert values[1] == pytest.approx(0.1, rel=0, abs=1e-6)
    assert (values[2] - values[0]) / 0.002 == pytest.approx(10, rel=0, abs=1e-5)


@pytest.mark.parametrize("redemption", [
    RedemptionFeature(principal_floor=0.2),
    RedemptionFeature(principal_floor=0.2, downside_participation=0.4),
    RedemptionFeature(include_principal=False),
    RedemptionFeature(downside_participation=0),
])
def test_low_spot_principal_variants_match_analytic_value(redemption):
    note, market = low_spot_note(redemption=redemption), low_market()
    result = price(note, market, "pde", PDEConfig(spot_grid_points=801, spot_grid_upper=400))
    assert result.present_value == pytest.approx(principal_oracle(note, market), rel=0, abs=1e-6)


@pytest.fixture(params=[(1, False), (1, True), (2, False), (2, True)],
                ids=["daily-cn", "daily-rannacher", "twice-cn", "twice-rannacher"])
def endpoint_value(request, monkeypatch):
    steps, smoothing = request.param
    interpolate = np.interp

    def evaluate(note):
        endpoints = []

        def capture(spot, grid, values):
            assert grid[0] == 0.0
            endpoints.append(float(values[0]))
            return interpolate(spot, grid, values)

        # Observe the final interpolation input without accepting S=0 in MarketData.
        monkeypatch.setattr(np, "interp", capture)
        price(note, low_market(), "pde", PDEConfig(
            spot_grid_points=401, spot_grid_upper=400,
            time_steps_per_day=steps, rannacher_smoothing=smoothing,
        ))
        value, = endpoints
        return value

    return evaluate


@pytest.mark.parametrize("redemption,principal", [
    (RedemptionFeature(principal_floor=0.2), 200),
    (RedemptionFeature(principal_floor=0.2, downside_participation=0.4), 600),
    (RedemptionFeature(include_principal=False), 0),
    (RedemptionFeature(downside_participation=0), 1000),
])
def test_initial_ki_endpoint_uses_effective_principal_floor(endpoint_value, redemption, principal):
    note = low_spot_note(redemption=redemption)
    expected = principal * exp(-0.03 * 126 / 252)
    assert endpoint_value(note) == pytest.approx(expected, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("initial_ki,start_day,survives,include_principal,has_ki,cashflows", [
    (True, 1, True, True, True, ((1, 1), (3, 2), (4, 201))),
    (False, 3, False, True, True, ((1, 1), (4, 200))),
    (True, 3, False, True, True, ((4, 200),)),
    (False, 3, False, False, True, ((1, 1),)),
    (True, 1, True, False, True, ((1, 1), (3, 2), (4, 1))),
    (True, 1, False, False, True, ()),
    (False, 1, False, True, False, ((1, 1), (3, 2), (4, 1001))),
    (False, 1, False, False, False, ((1, 1), (3, 2), (4, 1))),
    # Maturity KI suppresses only the same-day coupon, after earlier payments.
    (False, 4, False, True, True, ((1, 1), (3, 2), (4, 200))),
])
def test_fixed_coupon_endpoint_cashflows(
    endpoint_value, initial_ki, start_day, survives, include_principal, has_ki, cashflows,
):
    note = low_spot_note(
        maturity_days=4,
        coupon=CouponFeature(CouponRule.FIXED_PERIODIC, (1, 3, 4), (.252,) * 3,
                             survives_knock_in=survives),
        redemption=RedemptionFeature(principal_floor=.2, include_principal=include_principal),
        knock_in=KnockInFeature(.8, monitoring_start_day=start_day,
                                initially_knocked_in=initial_ki) if has_ki else None,
    )
    expected = sum(amount * exp(-.03 * day / 252) for day, amount in cashflows)
    assert endpoint_value(note) == pytest.approx(expected, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("initial_ki,survives,has_ki,maturity_amount", [
    (True, False, True, 200),
    (True, True, True, 204.5),
    (False, False, True, 200),
    (False, True, True, 204.5),
    (False, False, False, 1004.5),
    (False, True, False, 1004.5),
])
def test_contingent_coupon_endpoint_includes_elapsed_accrual(
    endpoint_value, initial_ki, survives, has_ki, maturity_amount,
):
    note = low_spot_note(
        maturity_days=4, elapsed_days=5,
        coupon=CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION,
                             maturity_rate=.126, survives_knock_in=survives),
        redemption=RedemptionFeature(principal_floor=.2),
        knock_in=KnockInFeature(.8, monitoring_start_day=4,
                                initially_knocked_in=initial_ki) if has_ki else None,
    )
    assert endpoint_value(note) == pytest.approx(
        maturity_amount * exp(-.03 * 4 / 252), rel=1e-11, abs=1e-11
    )


def test_fixed_coupon_endpoint_includes_only_remaining_accrual(endpoint_value):
    note = low_spot_note(
        maturity_days=4, elapsed_days=5, coupon_accrued_days=2,
        coupon=CouponFeature(CouponRule.FIXED_PERIODIC, (1, 3, 4), (.252,) * 3,
                             survives_knock_in=True),
        redemption=RedemptionFeature(principal_floor=.2),
    )
    expected = 3 * exp(-.03 / 252) + 2 * exp(-.03 * 3 / 252) + 201 * exp(-.03 * 4 / 252)
    assert endpoint_value(note) == pytest.approx(expected, rel=1e-11, abs=1e-11)
