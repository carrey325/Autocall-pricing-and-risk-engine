"""Batched, reproducible risk-neutral Monte Carlo pricing."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np

from ..config import MCConfig
from ..market import MarketData
from ..models import PricingModel, checked_local_vol, validate_model
from ..payoffs import VectorizedPayoffState, basket_performance
from ..products import StructuredNote
from ..results import PricingResult


def _daily_independent_normals(
    *,
    seed: int,
    stream_index: int,
    n_paths: int,
    n_assets: int,
    antithetic: bool,
) -> np.ndarray:
    base_paths = n_paths // 2 if antithetic else n_paths
    sequence = np.random.SeedSequence((seed, stream_index))
    generator = np.random.Generator(np.random.Philox(sequence))
    base = generator.standard_normal((base_paths, n_assets))
    if antithetic:
        return np.concatenate((base, -base), axis=0)
    return base


def price_mc(
    product: StructuredNote,
    market: MarketData,
    config: MCConfig,
    *,
    model: PricingModel | None = None,
) -> PricingResult:
    market.validate_for(product)
    if model is not None:
        validate_model(model, product, market)
    spots = np.broadcast_to(
        np.asarray(market.spots, dtype=float), (config.n_paths, product.n_assets)
    ).copy()
    volatilities = np.asarray(market.volatilities, dtype=float)
    dividends = np.asarray(market.dividend_yields, dtype=float)
    correlation_factor = market.correlation_factor()
    dt = 1.0 / (product.day_count * config.steps_per_day)
    drift = (market.rate - dividends - 0.5 * volatilities**2) * dt
    diffusion = volatilities * sqrt(dt)
    state = VectorizedPayoffState(product, config.n_paths, market.rate,
                                 discount=None if model is None else model.discount)

    for day in range(1, product.maturity_days + 1):
        for step in range(config.steps_per_day):
            stream_index = (day - 1) * config.steps_per_day + step
            independent = _daily_independent_normals(
                seed=config.seed,
                stream_index=stream_index,
                n_paths=config.n_paths,
                n_assets=product.n_assets,
                antithetic=config.antithetic,
            )
            correlated = independent @ correlation_factor.T
            if model is not None:
                t = stream_index * dt
                growth = float(model.forward_growth(t, t + dt))
                if not np.isfinite(growth) or growth <= 0:
                    raise ValueError('forward growth must be finite and positive')
            for start in range(0, config.n_paths, config.batch_size):
                stop = min(config.n_paths, start + config.batch_size)
                if model is None:
                    spots[start:stop] *= np.exp(
                        drift + diffusion * correlated[start:stop]
                    )
                else:
                    sigma = checked_local_vol(model, t, spots[start:stop, 0])
                    spots[start:stop, 0] *= np.exp(np.log(growth) - .5*sigma**2*dt
                        + sigma*sqrt(dt)*correlated[start:stop, 0])
                    if np.any(~np.isfinite(spots[start:stop])) or np.any(spots[start:stop] <= 0):
                        raise ValueError('local-vol evolution left finite positive spot domain')
        state.process_day(day, basket_performance(spots, product))

    evaluation = state.result()
    values = evaluation.present_values
    present_value = float(np.mean(values))
    samples = (values[:config.n_paths//2] + values[config.n_paths//2:])/2 if config.antithetic else values
    standard_error = float(np.std(samples, ddof=1) / sqrt(len(samples))) if len(samples)>1 else None
    quantile = NormalDist().inv_cdf(0.5 + config.confidence_level / 2.0)
    confidence_interval = None if standard_error is None else (
        present_value - quantile * standard_error,
        present_value + quantile * standard_error,
    )
    warnings: list[str] = []
    if model is None and np.all(volatilities == 0):
        warnings.append("deterministic zero-volatility market")
    if standard_error is None:
        warnings.append('one antithetic pair: sampling uncertainty is undefined')
    return PricingResult(
        method="mc",
        present_value=present_value,
        net_value=present_value - float(product.issue_price),
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        knock_in_probability=float(np.mean(evaluation.knocked_in)),
        knock_out_probability=float(np.mean(evaluation.knocked_out)),
        expected_redemption_time=float(
            np.mean(evaluation.redemption_days) / product.day_count
        ),
        expected_coupon_cashflow=float(np.mean(evaluation.coupon_cashflows)),
        diagnostics={
            "n_paths": config.n_paths,
            "seed": config.seed,
            "antithetic": config.antithetic,
            "batch_size": config.batch_size,
            "steps_per_day": config.steps_per_day,
            "confidence_level": config.confidence_level,
            "random_generator": "Philox",
            "independent_samples": len(samples),
            "uncertainty_estimator": 'antithetic pair averages' if config.antithetic else 'independent paths',
            "model": {"kind": "flat_market", "time_offset": 0.} if model is None else model.diagnostics(),
        },
        warnings=tuple(warnings),
    )
