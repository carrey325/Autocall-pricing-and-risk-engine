"""Independent European-payoff checks and lifecycle regression tests."""

from dataclasses import replace
import json
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.stats import norm

from structured_products import (
    CouponFeature, CouponRule, GreekConfig, KnockInFeature, MCConfig,
    MarketData, PDEConfig, RedemptionFeature, StructuredNote,
    calculate_greeks, make_classic_snowball, make_standard_fcn,
    make_worst_of_snowball, price,
)
from structured_products.analytics.greeks import _roll_product
from structured_products.payoffs import evaluate_path


def european_note():
    # At maturity: 1000 - 10 * max(100 - S, 0), with no path-dependent events.
    return StructuredNote(
        reference_spots=(100.0,), notional=1000.0, maturity_days=126,
        coupon=CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION),
        redemption=RedemptionFeature(),
        knock_in=KnockInFeature(0.8, initially_knocked_in=True),
    )


def market():
    return MarketData((90.0,), 0.03, (0.01,), (0.25,))


def analytic_note(t):
    # Bond minus a European put; expectations are independent of our pricers.
    s, k, r, q, vol, scale = 90.0, 100.0, 0.03, 0.01, 0.25, 10.0
    d1 = (log(s / k) + (r - q + vol**2 / 2) * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    pv = 1000 * exp(-r*t) - scale * (
        k * exp(-r*t) * norm.cdf(-d2) - s * exp(-q*t) * norm.cdf(-d1)
    )
    delta = scale * exp(-q*t) * norm.cdf(-d1)
    vega = -scale * s * exp(-q*t) * norm.pdf(d1) * sqrt(t)
    return dict(pv=pv, delta=delta, vega=vega,
                vanna=scale * exp(-q*t) * norm.pdf(d1) * d2 / vol,
                vomma=vega * d1 * d2 / vol)


def test_extended_pde_greeks_against_european_closed_form():
    config = GreekConfig(relative_spot_bump=0.005, volatility_bump=0.005,
                         theta_days=2)
    result = calculate_greeks(european_note(), market(), "pde",
                              PDEConfig(spot_grid_points=4001), config)
    now, later = analytic_note(0.5), analytic_note(0.5 - 2/252)
    assert result.vanna[0] == pytest.approx(now["vanna"], rel=0.02)
    assert result.vomma[0] == pytest.approx(now["vomma"], rel=0.02)
    assert result.lambda_[0] == pytest.approx(now["delta"] * 90 / now["pv"], rel=0.002)
    assert result.charm[0] == pytest.approx((later["delta"]-now["delta"])/2, rel=0.03)
    assert result.veta[0] == pytest.approx((later["vega"]-now["vega"])/2, rel=0.03)
    assert result.omega == result.lambda_
    assert result.volga == result.vomma
    payload = result.to_dict()
    assert payload["lambda"] == payload["omega"]
    assert payload["vomma"] == payload["volga"]
    assert "lambda_" not in payload
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("method,engine", [("mc", MCConfig(n_paths=200)),
                                          ("pde", PDEConfig())])
@pytest.mark.parametrize("kind", ["autocall", "maturity", "periodic"])
def test_time_roll_preserves_contractual_coupon_amounts(method, engine, kind):
    if kind == "periodic":
        note = make_standard_fcn(maturity_days=42, payment_days=(21,42), coupon_rate=0.12)
    else:
        note = make_classic_snowball(maturity_days=42, observation_days=(21,42),
                                    coupon_rate=0.12, knock_out_ratio=1 if kind == "autocall" else 2)
    fixed = MarketData((100.0,), 0.0, (0.0,), (0.0,))
    rolled = _roll_product(note, 5)
    assert price(rolled, fixed, method, engine).present_value == pytest.approx(
        price(note, fixed, method, engine).present_value)
    assert rolled.reference_spots == note.reference_spots


def test_roll_after_coupon_preserves_remaining_accrual_and_history():
    note = make_standard_fcn(maturity_days=63, payment_days=(21,42,63), coupon_rate=0.12)
    rolled = _roll_product(note, 25)
    path = np.full((rolled.maturity_days+1,1), 100.0)
    amounts = [c.amount for c in evaluate_path(path, rolled).cashflows if c.kind == "periodic_coupon"]
    assert amounts == pytest.approx([10.0, 10.0])
    assert rolled.elapsed_days == 25
    assert rolled.coupon_accrued_days == 4
    twice = _roll_product(_roll_product(note, 10), 15)
    assert twice == rolled


def test_mc_extended_risk_reproducible_per_asset_and_batch_invariant():
    note = make_worst_of_snowball(maturity_days=21, observation_days=(21,))
    m = MarketData((90.0, 95.0), 0.03, (0.0,0.0), (0.2,0.25), ((1,0.4),(0.4,1)))
    engine = MCConfig(n_paths=1000, batch_size=100)
    a = calculate_greeks(note,m,"mc",engine)
    b = calculate_greeks(note,m,"mc",replace(engine,batch_size=256))
    for name in ("lambda_", "vanna", "vomma", "charm", "veta"):
        assert len(getattr(a,name)) == 2
        assert getattr(a,name) == getattr(b,name)


def test_price_can_include_all_risk_and_json_aliases():
    result = price(european_note(),market(),"mc",MCConfig(n_paths=200),include_greeks=True)
    assert result.greeks.base_price == result.present_value
    payload = result.to_dict()
    assert payload["greeks"]["lambda"] == payload["greeks"]["omega"]
    assert len(payload["greeks"]["vanna"]) == 1
    json.dumps(payload, allow_nan=False)


def test_zero_value_elasticity_is_null_with_warning():
    note = replace(european_note(),redemption=RedemptionFeature(include_principal=False))
    result = calculate_greeks(note,market(),"mc",MCConfig(n_paths=100))
    assert result.lambda_ == (None,)
    assert any("elasticity" in warning for warning in result.warnings)
    assert result.to_dict()["omega"] == [None]


def test_charm_and_veta_after_autocall_use_zero_remaining_exposure():
    note = make_classic_snowball(maturity_days=21,observation_days=(1,21),knock_out_ratio=0.9)
    m = MarketData((100.0,),0.03,(0.0,),(0.2,))
    result = calculate_greeks(note,m,"mc",MCConfig(n_paths=1000))
    assert result.charm[0] == pytest.approx(-result.delta[0])
    assert result.veta[0] == pytest.approx(-result.vega[0])
    assert result.diagnostics["roll_redeemed"] is True
    assert result.diagnostics["roll_cashflow"] > 1000


def test_time_roll_records_knock_in_at_fixed_spot():
    note = make_classic_snowball(maturity_days=21,observation_days=(21,),knock_in_ratio=0.8)
    m = MarketData((79.0,),0.03,(0.0,),(0.2,))
    result = calculate_greeks(note,m,"mc",MCConfig(n_paths=1000))
    assert result.diagnostics["roll_knocked_in"] is True


@pytest.mark.parametrize("changes", [
    {"relative_spot_bump": 1.0}, {"relative_spot_bump": float("nan")},
    {"volatility_bump": float("inf")}, {"rate_bump": float("nan")},
    {"correlation_bump": float("nan")}, {"theta_days": 1.5},
])
def test_greek_bumps_cannot_produce_invalid_scenarios(changes):
    with pytest.raises(ValueError):
        GreekConfig(**changes)


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_pde_second_order_risk_stable_across_bump_sizes(factor):
    result = calculate_greeks(european_note(), market(), "pde",
                              PDEConfig(spot_grid_points=4001),
                              GreekConfig(relative_spot_bump=0.005*factor,
                                          volatility_bump=0.005*factor))
    expected = analytic_note(0.5)
    assert result.vanna[0] == pytest.approx(expected["vanna"], rel=0.025)
    assert result.vomma[0] == pytest.approx(expected["vomma"], rel=0.025)


def test_roll_past_final_coupon_and_autocall_keeps_valid_remaining_note():
    note = make_standard_fcn(maturity_days=42, payment_days=(10,), coupon_rate=0.12)
    rolled = _roll_product(note, 15)
    result = price(rolled, MarketData((100.0,),0.0,(0.0,),(0.0,)),"pde")
    assert result.present_value == 1000.0
    note = make_classic_snowball(maturity_days=42, observation_days=(10,),
                                 knock_out_ratio=2, coupon_rate=0.12)
    rolled = _roll_product(note, 15)
    assert rolled.autocall is None
    assert price(rolled, MarketData((100.0,),0.0,(0.0,),(0.0,)),"pde").present_value == 1020.0


def test_time_risk_preserves_initial_ki_and_defers_future_ki_monitoring():
    note = european_note()
    a = calculate_greeks(note,market(),"mc",MCConfig(n_paths=100))
    assert a.diagnostics["roll_knocked_in"] is True
    note = replace(note,knock_in=KnockInFeature(0.95,monitoring_start_day=5))
    b = calculate_greeks(note,market(),"mc",MCConfig(n_paths=100))
    assert b.diagnostics["roll_knocked_in"] is False


@pytest.mark.parametrize("changes", [{"elapsed_days": -1}, {"elapsed_days": 1.5},
                                    {"elapsed_days": 1, "coupon_accrued_days": 2}])
def test_invalid_accrual_history_is_rejected(changes):
    with pytest.raises(ValueError, match="accrual days"):
        replace(european_note(), **changes)


def test_pde_fixed_domain_survives_spot_bumps():
    from structured_products.engines.pde.grid import build_spot_grid

    note = european_note()
    config = PDEConfig(spot_grid_points=401, spot_grid_upper=400.0)
    down = build_spot_grid(note, replace(market(),spots=(99.0,)), config)
    up = build_spot_grid(note, replace(market(),spots=(101.0,)), config)
    np.testing.assert_array_equal(down, np.arange(401, dtype=float))
    np.testing.assert_array_equal(up, down)
    with pytest.raises(ValueError, match="upper"):
        build_spot_grid(note, replace(market(),spots=(400.0,)), config)


@pytest.mark.parametrize("upper", [0.0, float("nan"), float("inf")])
def test_pde_rejects_invalid_fixed_domain(upper):
    with pytest.raises(ValueError, match="upper"):
        PDEConfig(spot_grid_upper=upper)


def test_snowball_vanna_stability_requires_a_common_pde_domain():
    from structured_products import load_config

    config = load_config("configs/classic_snowball.json")
    values = [calculate_greeks(
        config.product, config.market, "pde", PDEConfig(spot_grid_points=1601),
        GreekConfig(relative_spot_bump=bump, volatility_bump=bump),
    ).vanna[0] for bump in (0.005, 0.01)]
    # Moving the grid with each bumped spot introduces a >20% discrepancy.
    assert values[0] == pytest.approx(values[1], rel=0.01)


def test_roll_respects_ko_priority_over_same_day_ki():
    note = make_classic_snowball(maturity_days=21, observation_days=(1,21),
                                 knock_out_ratio=0.9, knock_in_ratio=1.1)
    result = calculate_greeks(note, MarketData((100.0,),0.03,(0.0,),(0.2,)),
                              "mc", MCConfig(n_paths=100))
    assert result.diagnostics["roll_redeemed"] is True
    assert result.diagnostics["roll_knocked_in"] is False
