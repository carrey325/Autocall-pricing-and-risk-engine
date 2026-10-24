"""Independent checks of the absorbing endpoint and its contractual cashflows."""

from math import exp

import numpy as np
import pytest

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
