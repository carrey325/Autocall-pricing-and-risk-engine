# Four short workflows

These examples run after installing the package from GitHub. They do not need
the source checkout or local JSON files. Input rates and volatility are decimals.

## 1. Price your first Snowball

```python
from structured_products import MarketData, MCConfig, make_classic_snowball, price

note = make_classic_snowball(reference_spot=100, notional=1000,
                             maturity_days=252, coupon_rate=0.18)
market = MarketData(spots=(100,), rate=0.03,
                    dividend_yields=(0.0,), volatilities=(0.20,))
value = price(note, market, "mc", MCConfig(n_paths=20_000, seed=7))
print(value.present_value, value.confidence_interval)
```

The reference spot belongs to the contract; changing today's spot in
`MarketData` does not move barriers or the strike.

## 2. Price a multi-asset worst-of FCN

```python
from structured_products import load_example, price

config = load_example("worst_of_fcn")
value = price(config.product, config.market, "mc", config.mc)
print(value.present_value, value.knock_in_probability)
```

Worst-of products use MC. Change the immutable market with `dataclasses.replace`
to supply your spots, volatilities, dividends, and valid correlation matrix.

## 3. Solve a fair coupon

```python
from structured_products import load_example, solve_fair_coupon

config = load_example("classic_snowball")
solution = solve_fair_coupon(
    config.product, config.market, 1000.0, "pde", config.pde, 0.01, 0.40,
)
print(solution.coupon_rate, solution.residual)
```

The target is total PV. Supply a bracket containing the target; an unreachable
target raises an error instead of returning a misleading coupon.

## 4. Calculate price and risk together

```python
from structured_products import load_example, price

config = load_example("classic_snowball")
result = price(config.product, config.market, "pde", config.pde,
               include_greeks=True, greek_config=config.greeks)
print(result.present_value)
print(result.greeks.delta, result.greeks.vega)
print(result.greeks.vanna, result.greeks.vomma)
print(result.greeks.charm, result.greeks.veta)
```

Charm and Veta are changes per elapsed trading day under the documented
constant-spot roll. MC higher-order Greeks can be noisy: inspect warnings and
vary numerical settings. See [the risk guide](../RISK_ANALYTICS.md).

Equivalent installed CLI commands:

```bash
structured-products-pricer --version
structured-products-pricer price --example classic_snowball --method pde
python -m structured_products price --example worst_of_fcn --paths 20000
python -m structured_products compare --example classic_snowball
python -m structured_products price --example classic_snowball --no-greeks
```
