from __future__ import annotations

from dataclasses import replace
import json
from math import exp, isfinite, sqrt
from statistics import NormalDist

import numpy as np
import pytest

from structured_products.builders import (
    make_classic_snowball,
    make_worst_of_snowball,
)
from structured_products.config import MCConfig, load_example
from structured_products.engines.mc import price_mc
from structured_products.market import MarketData
from structured_products.payoffs import VectorizedPayoffState


def single_market(volatility: float = 0.2) -> MarketData:
    return MarketData(
        spots=(100.0,),
        rate=0.03,
        dividend_yields=(0.0,),
        volatilities=(volatility,),
    )


def test_fixed_seed_is_reproducible_and_batch_invariant() -> None:
    product = make_classic_snowball(maturity_days=63)
    first = price_mc(product, single_market(), MCConfig(n_paths=2_000, batch_size=128))
    second = price_mc(product, single_market(), MCConfig(n_paths=2_000, batch_size=777))
    assert first.present_value == second.present_value
    assert first.standard_error == second.standard_error
    assert first.confidence_interval == second.confidence_interval


def price_samples(monkeypatch, values, *, antithetic):
    # Replace only the payoff samples; exercise the real estimator and CI wiring.
    original = VectorizedPayoffState.result

    def result(state):
        return replace(original(state), present_values=np.asarray(values, dtype=float))

    monkeypatch.setattr(VectorizedPayoffState, "result", result)
    return price_mc(
        make_classic_snowball(maturity_days=1, observation_days=(1,)),
        single_market(), MCConfig(n_paths=len(values), antithetic=antithetic),
    )


@pytest.mark.parametrize("antithetic", [True, False])
def test_antithetic_standard_error_uses_pair_means(monkeypatch, antithetic):
    result = price_samples(monkeypatch, [1, 4, 3, 8], antithetic=antithetic)
    expected = 2.0 if antithetic else np.std([1, 4, 3, 8], ddof=1) / 2
    assert result.present_value == 4.0
    assert result.standard_error == pytest.approx(expected)


def test_constant_pair_means_have_zero_standard_error(monkeypatch):
    result = price_samples(monkeypatch, [1, 2, 3, 2], antithetic=True)
    assert result.present_value == 2.0
    assert result.standard_error == 0.0
    assert result.confidence_interval == (2.0, 2.0)


@pytest.mark.parametrize("volatility", [0.0, 0.2])
def test_one_antithetic_pair_has_unknown_uncertainty(volatility):
    result = price_mc(make_classic_snowball(maturity_days=21),
                      single_market(volatility), MCConfig(n_paths=2))
    assert isfinite(result.present_value)
    assert result.standard_error is None
    assert result.confidence_interval is None
    assert any("one independent antithetic pair" in warning for warning in result.warnings)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("values", [[1, 3], [1, 3, 8]])
def test_ordinary_mc_small_samples_use_iid_standard_error(monkeypatch, values):
    result = price_samples(monkeypatch, values, antithetic=False)
    assert result.standard_error == pytest.approx(np.std(values, ddof=1) / sqrt(len(values)))
    assert result.confidence_interval is not None


def test_classic_uncertainty_matches_captured_payoff_pairs(monkeypatch):
    samples = []
    original = VectorizedPayoffState.result

    def capture(state):
        evaluation = original(state)
        samples.append(evaluation.present_values.copy())
        return evaluation

    monkeypatch.setattr(VectorizedPayoffState, "result", capture)
    config = load_example("classic_snowball")
    result = price_mc(config.product, config.market, config.mc)
    values, = samples
    pairs = (values[:10_000] + values[10_000:]) / 2
    expected = np.std(pairs, ddof=1) / sqrt(10_000)
    assert result.present_value == float(np.mean(values))
    assert result.standard_error == pytest.approx(expected)
    width = NormalDist().inv_cdf(0.975) * expected
    assert result.confidence_interval == pytest.approx(
        (result.present_value - width, result.present_value + width)
    )


def test_zero_volatility_is_deterministic() -> None:
    product = make_classic_snowball(
        maturity_days=21,
        observation_days=(21,),
        coupon_rate=0.12,
    )
    result = price_mc(product, single_market(0.0), MCConfig(n_paths=100))
    expected = 1_000.0 * (1.0 + 0.12 * 21 / 252) * exp(-0.03 * 21 / 252)
    assert result.present_value == pytest.approx(expected)
    assert result.standard_error == pytest.approx(0.0, abs=1.0e-12)


def test_standard_error_decreases_with_more_paths() -> None:
    product = make_classic_snowball(maturity_days=63)
    small = price_mc(product, single_market(), MCConfig(n_paths=1_000))
    large = price_mc(product, single_market(), MCConfig(n_paths=8_000))
    assert large.standard_error < small.standard_error


def test_knock_in_is_not_recorded_after_knock_out() -> None:
    product = make_classic_snowball(
        maturity_days=2,
        day_count=252,
        observation_days=(1,),
        knock_out_ratio=0.5,
        knock_in_ratio=2.0,
    )
    result = price_mc(product, single_market(0.0), MCConfig(n_paths=100))
    assert result.knock_out_probability == 1.0
    assert result.knock_in_probability == 0.0


def test_worst_of_uses_correlated_multi_asset_market() -> None:
    product = make_worst_of_snowball(
        reference_spots=(100.0, 100.0),
        maturity_days=21,
        observation_days=(21,),
    )
    market = MarketData(
        spots=(100.0, 80.0),
        rate=0.0,
        dividend_yields=(0.0, 0.0),
        volatilities=(0.0, 0.0),
        correlation=((1.0, 0.5), (0.5, 1.0)),
    )
    result = price_mc(product, market, MCConfig(n_paths=100))
    assert result.knock_out_probability == 0.0
    assert result.knock_in_probability == 0.0
    assert result.present_value == pytest.approx(1_000.0 * (1 + 0.20 * 21 / 252))
