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

The project includes numerical and API tests, a **90% minimum coverage gate**,
and CI across Python 3.10–3.13. Tests cover hand-calculated payoff paths, barrier-touch and
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

Install from this GitHub repository (Python 3.10–3.13):

~~~bash
python -m pip install "git+https://github.com/carrey325/Autocall-pricing-and-risk-engine.git@main"
~~~

Price a Snowball and calculate its Greeks from any directory:

~~~bash
python -m structured_products price --example classic_snowball --method mc
~~~

Both `price` and `run` return valuation and all Greeks as JSON. Use `--no-greeks`
for valuation only. The `structured-products-pricer` console command provides
the same interface; `--config path/to/config.json` accepts your own inputs.

~~~python
from structured_products import load_example, price

config = load_example("classic_snowball")
result = price(config.product, config.market, "pde", config.pde, include_greeks=True)
print(result.present_value)
print(result.greeks.vanna)
~~~

See the [API guide](docs/API.md), [tutorials](docs/TUTORIALS.md), and
[risk conventions](RISK_ANALYTICS.md). Version **0.3.0rc1** is a GitHub development
candidate; this project is not being published to PyPI.

Compare Monte Carlo and PDE valuations:

~~~bash
python -m structured_products compare --example classic_snowball
~~~

Solve for the fair coupon:

~~~bash
python -m structured_products fair-coupon --example classic_snowball --method pde \
  --target-pv 1000 --lower 0.01 --upper 0.40
~~~

## Market Surface Scripts

SSVI calibration, Dupire local volatility, risk grids, stress revaluation and
coupon/barrier comparisons are available through the Python API and CLI.

~~~bash
python -m pip install ".[plot]"
python examples/run_market_research.py --output results/market-research-new
~~~

Check calibration gates and grid/bump sensitivity before using results. The bundled
synthetic example does not establish real-market calibration or Greek convergence.

## Model Scope

The default uses geometric Brownian motion with flat rates and volatilities.
Single-asset notes also support SSVI/Dupire local volatility with forward and
discount curves. Barrier monitoring uses daily closes and ACT/252 year fractions.
Multi-asset worst-of products use flat-model Monte Carlo.

See [MODEL_SPEC.md](MODEL_SPEC.md) for contract conventions and
[VALIDATION.md](VALIDATION.md) for numerical tests and limitations.
