# Autocall Pricing and Risk Engine

[![Tests](https://github.com/carrey325/Autocall-pricing-and-risk-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/carrey325/Autocall-pricing-and-risk-engine/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular Python framework for pricing and risk analysis of autocallable
structured products. It combines reproducible Monte Carlo pricing for single-
and multi-asset worst-of structures with an independent two-state
Crank–Nicolson PDE benchmark for single-asset products.

## Project Highlights

- Prices classic, wide, butterfly, dividend, and step-down Snowballs.
- Prices standard, discount-entry, and worst-of FCNs.
- Calculates fair coupon, Delta, Gamma, Vega, Theta, Rho, and correlation risk.
- Adds per-underlying Lambda/Omega, Vanna, Vomma/Volga, Charm, and Veta
  to pricing-time risk reports.
- Reports Monte Carlo confidence intervals, KI/KO probabilities, and expected life.
- Models product lifecycles with reusable knock-in, autocall, coupon, and
  redemption features instead of product-specific payoff scripts.
- Provides deterministic JSON configuration, Python API, and command-line workflows.

## Pricing Methods

**Monte Carlo**

- Correlated risk-neutral GBM paths for single- and multi-asset products.
- Philox random streams, fixed-seed reproducibility, and antithetic variates.
- Batch-size-invariant simulations for stable fair-coupon and Greek calculations.

**Crank–Nicolson PDE**

- Independent two-state knocked-in / not-knocked-in valuation.
- Exact application of knock-in, autocall, coupon, and maturity events.
- One-dimensional benchmark for supported single-asset products.

## Numerical Validation

The project includes **128 automated tests**, **96% test coverage**, and CI across
Python 3.10–3.13. Tests cover hand-calculated payoff paths, barrier-touch and
same-day event rules, Monte Carlo reproducibility, PDE grid refinement, and
cross-engine comparisons.

For the benchmark classic Snowball on a 1,000 notional:

| Method | Present Value |
| --- | ---: |
| Monte Carlo | 1000.693524 |
| PDE | 1000.487825 |

The difference is approximately **2.06 basis points**, with the PDE estimate
inside the Monte Carlo 95% confidence interval. See
[VALIDATION.md](VALIDATION.md) for the methodology and full numerical results.

## Quick Start

~~~bash
python -m pip install -e ".[dev]"
~~~

Price a Snowball and calculate its Greeks:

~~~bash
python main.py price --config configs/classic_snowball.json --method mc
~~~

Both `price` and `run` return valuation and all Greeks as JSON. Use `--no-greeks`
for valuation only. In Python, call `price(..., include_greeks=True)` and read
`result.greeks`. See [RISK_ANALYTICS.md](RISK_ANALYTICS.md) for definitions,
units, time-roll conventions, and a complete example.

Compare Monte Carlo and PDE valuations:

~~~bash
python main.py compare --config configs/classic_snowball.json
~~~

Solve for the fair coupon:

~~~bash
python main.py fair-coupon --config configs/classic_snowball.json --method pde \
  --target-pv 1000 --lower 0.01 --upper 0.40
~~~

## Model Scope

The current implementation assumes risk-neutral geometric Brownian motion with
flat rates, dividend yields, volatilities, and correlations. Barrier monitoring
uses daily closes and ACT/252 year fractions. Multi-asset worst-of products are
supported through Monte Carlo only.

See [MODEL_SPEC.md](MODEL_SPEC.md) for contract conventions and
[VALIDATION.md](VALIDATION.md) for numerical tests and limitations.
