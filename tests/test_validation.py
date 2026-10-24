from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from structured_products import (
    AutocallFeature,
    CouponFeature,
    CouponRule,
    GreekConfig,
    MCConfig,
    PDEConfig,
    KnockInFeature,
    MarketData,
    RedemptionFeature,
    StructuredNote,
    make_classic_snowball,
    make_standard_fcn,
)
from structured_products.analytics.fair_coupon import with_flat_coupon
from structured_products.analytics.greeks import _roll_product, calculate_greeks
from structured_products.config import load_config
from structured_products.schedules import flat_schedule, linear_schedule, monthly_observation_days


@pytest.mark.parametrize(
    "factory",
    [
        lambda: KnockInFeature(0.0),
        lambda: KnockInFeature(0.8, monitoring_start_day=0),
        lambda: AutocallFeature((), (), ()),
        lambda: AutocallFeature((1,), (1.0, 0.9), (0.1,)),
        lambda: AutocallFeature((1,), (0.0,), (0.1,)),
        lambda: CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION, maturity_rate=-0.1),
        lambda: CouponFeature(CouponRule.FIXED_PERIODIC, (1,), ()),
        lambda: CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION, (1,), (0.1,)),
        lambda: RedemptionFeature(strike_ratio=0.0),
        lambda: RedemptionFeature(downside_participation=-1.0),
        lambda: RedemptionFeature(principal_floor=1.1),
    ],
)
def test_invalid_features_fail(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def _note(**updates) -> StructuredNote:
    values = {
        "reference_spots": (100.0,),
        "notional": 100.0,
        "maturity_days": 10,
        "coupon": CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION),
        "redemption": RedemptionFeature(),
    }
    values.update(updates)
    return StructuredNote(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"reference_spots": ()},
        {"notional": 0.0},
        {"issue_price": 0.0},
        {"reference_spots": (100.0, 100.0)},
        {"knock_in": KnockInFeature(0.8, monitoring_start_day=11)},
        {"autocall": AutocallFeature((11,), (1.0,), (0.1,))},
        {
            "coupon": CouponFeature(
                CouponRule.FIXED_PERIODIC, payment_days=(11,), rate_schedule=(0.1,)
            )
        },
    ],
)
def test_invalid_products_fail(updates) -> None:
    with pytest.raises(ValueError):
        _note(**updates)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "spots": (),
            "rate": 0.0,
            "dividend_yields": (),
            "volatilities": (),
            "correlation": (),
        },
        {
            "spots": (100.0,),
            "rate": 0.0,
            "dividend_yields": (),
            "volatilities": (0.2,),
        },
        {
            "spots": (100.0,),
            "rate": 0.0,
            "dividend_yields": (0.0,),
            "volatilities": (-0.1,),
        },
        {
            "spots": (100.0, 100.0),
            "rate": 0.0,
            "dividend_yields": (0.0, 0.0),
            "volatilities": (0.2, 0.2),
            "correlation": ((1.0,),),
        },
        {
            "spots": (100.0, 100.0),
            "rate": 0.0,
            "dividend_yields": (0.0, 0.0),
            "volatilities": (0.2, 0.2),
            "correlation": ((1.0, 0.2), (0.3, 1.0)),
        },
        {
            "spots": (100.0, 100.0),
            "rate": 0.0,
            "dividend_yields": (0.0, 0.0),
            "volatilities": (0.2, 0.2),
            "correlation": ((0.9, 0.2), (0.2, 1.0)),
        },
    ],
)
def test_invalid_market_inputs_fail(kwargs) -> None:
    with pytest.raises(ValueError):
        MarketData(**kwargs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MCConfig(n_paths=1),
        lambda: MCConfig(n_paths=3, antithetic=True),
        lambda: MCConfig(confidence_level=1.0),
        lambda: PDEConfig(spot_grid_points=100),
        lambda: PDEConfig(time_steps_per_day=0),
        lambda: PDEConfig(upper_spot_multiple=1.0),
        lambda: PDEConfig(theta=0.4),
        lambda: PDEConfig(interpolation="cubic"),
        lambda: GreekConfig(theta_days=0),
    ],
)
def test_invalid_engine_configs_fail(factory) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("n_paths", [-1, 0, 1])
def test_mc_rejects_fewer_than_two_paths(antithetic, n_paths):
    with pytest.raises(ValueError):
        MCConfig(n_paths=n_paths, antithetic=antithetic)


@pytest.mark.parametrize("n_paths,antithetic", [(2, True), (4, True), (2, False), (3, False)])
def test_mc_accepts_supported_small_sample_configs(n_paths, antithetic):
    config = MCConfig(n_paths=n_paths, antithetic=antithetic)
    assert config.n_paths == n_paths


def test_schedule_edge_cases() -> None:
    with pytest.raises(ValueError):
        monthly_observation_days(0)
    with pytest.raises(ValueError):
        monthly_observation_days(10, start_month=0)
    with pytest.raises(ValueError):
        flat_schedule(1.0, -1)
    assert linear_schedule(1.0, 0.5, 0) == ()
    assert linear_schedule(1.0, 0.5, 1) == (1.0,)


def test_coupon_replacement_and_greek_guardrails() -> None:
    fcn = make_standard_fcn(maturity_days=63, payment_days=(21, 42, 63))
    replaced = with_flat_coupon(fcn, 0.2)
    assert replaced.coupon.rate_schedule == (0.2, 0.2, 0.2)
    with pytest.raises(ValueError):
        with_flat_coupon(fcn, -0.1)
    with pytest.raises(ValueError):
        _roll_product(fcn, 63)

    market = MarketData(
        spots=(100.0,),
        rate=0.03,
        dividend_yields=(0.0,),
        volatilities=(0.005,),
    )
    with pytest.raises(ValueError, match="volatility bump"):
        calculate_greeks(
            make_classic_snowball(maturity_days=21, observation_days=(21,)),
            market,
            "mc",
            MCConfig(n_paths=100),
        )


def test_strict_json_failure_modes(tmp_path: Path) -> None:
    cases = [
        [],
        {"market": {}},
        {"product": {"template": "unknown"}, "market": {}},
        {
            "product": {"template": "classic_snowball", "extra": 1},
            "market": {},
        },
        {
            "product": {"template": "classic_snowball"},
            "market": {
                "spots": [100.0],
                "rate": 0.0,
                "dividend_yields": [0.0],
                "volatilities": [0.2],
                "extra": 1,
            },
        },
        {
            "product": {"template": "classic_snowball"},
            "market": {
                "spots": [100.0],
                "rate": 0.0,
                "dividend_yields": [0.0],
                "volatilities": [0.2],
            },
            "engines": {"unknown": {}},
        },
    ]
    for index, payload in enumerate(cases):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)
