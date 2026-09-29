# Pricing and Risk Analytics

`price` and `run` in the CLI calculate valuation and risk together:

```bash
python main.py price --config configs/classic_snowball.json --method pde
python main.py run --config configs/worst_of_snowball.json --method mc
```

The JSON output has sibling `price` and `greeks` objects. `run` also includes the
product name. `greeks` calculates the same risk report on its own. For price-only
work, including zero-volatility scenarios, use `--no-greeks` with `price` or `run`.
Risk revaluation costs more than one valuation; the baseline PV is reused.

## Python API

```python
from structured_products import GreekConfig, PDEConfig, load_config, price

config = load_config("configs/classic_snowball.json")
result = price(
    config.product, config.market, "pde", PDEConfig(spot_grid_points=1601),
    include_greeks=True,
    greek_config=GreekConfig(
        relative_spot_bump=0.01, volatility_bump=0.01, theta_days=1,
    ),
)
print(result.present_value)
print(result.greeks.to_dict())
```

The Python `price` API defaults to valuation only, so fair-coupon solving and
scenario loops do not recursively calculate risk. Set `include_greeks=True` for
the combined result, or use `calculate_greeks` directly. Python serialization
places the risk report inside `result.to_dict()["greeks"]`.

## Definitions and units

`V` is total PV, including principal; `S_i` is current spot, not contractual
reference spot. Volatility is decimal: changing 0.20 to 0.21 is one vol point.
All per-underlying arrays follow the order of `market.spots`.

| JSON field | Definition | Unit |
| --- | --- | --- |
| `delta` | dV/dS_i | PV / spot unit |
| `gamma` | d²V/dS_i² | PV / spot unit² |
| `vega` | dV/dsigma_i | PV / decimal volatility |
| `lambda`, `omega` | Delta_i * S_i / V | dimensionless elasticity |
| `vanna` | d²V/(dS_i dsigma_i) | PV / (spot unit * decimal volatility) |
| `vomma`, `volga` | d²V/dsigma_i² | PV / decimal volatility² |
| `charm` | (rolled Delta_i - Delta_i) / h | Delta / trading day |
| `veta` | (rolled Vega_i - Vega_i) / h | Vega / trading day |
| `theta` | rolled PV - PV | PV change over h trading days |
| `rho` | dV/dr | PV / decimal rate |
| `correlation_sensitivity` | uniform shift of all off-diagonal correlations | PV / unit correlation shift |

`omega` aliases `lambda`; `volga` aliases `vomma`. In Python use `lambda_`
because `lambda` is reserved. Lambda is null, with a warning, when
`abs(PV) <= 1e-12 * notional`. Its denominator is never PV minus issue price.
Multiplying each Lambda by a small fractional spot move approximates its
contribution to the fractional change in total note value.

Vanna and Vomma use centered finite differences. Vanna uses the four spot/vol
corner valuations; Vomma reuses the two volatility bumps used for Vega.
For multi-asset notes these are own-asset derivatives, not cross-asset matrices.
All Greek bumps preserve contractual reference prices, barriers, and strikes.

To express Vega or Vanna per one vol point, multiply the reported value by 0.01.
For Vomma in squared vol-point units, multiply by 0.01². For Rho per basis point,
multiply by 0.0001. Veta per vol point per day is the reported value times 0.01.

## Time roll: Charm, Veta, and Theta

`h = GreekConfig.theta_days`, default 1, is a positive integer smaller than the
remaining maturity. Time runs forward in elapsed trading days. The sign is
therefore opposite to a derivative with respect to remaining maturity.
Charm/Veta divide by h; the existing Theta field remains the total h-day change.

The roll assumes each underlying stays at its current spot on every intervening
daily close. It applies the contract's same-day priority, updates knock-in state,
processes coupons, and stops at autocall. Future spot/volatility bumps hold that
realized scenario history fixed. Once redeemed, remaining PV, Delta, and Vega
are zero. The resulting Charm/Veta describe the change in remaining exposure.

Rolled PV is **ex-cash**: intervening cashflows are reported separately as
`diagnostics.roll_cashflow` (undiscounted, without reinvestment). Theta is not
total holding-period P&L. Near a coupon or autocall date the roll includes an
event jump, so Charm/Veta are scenario differences rather than smooth local
time derivatives. This is a constant-spot scenario, not an expectation of the
future Greek over simulated market paths.

Remaining schedules move backward by h days while original coupon cash amounts
retain their contractual accrual. `StructuredNote.elapsed_days` records time
since issue and `coupon_accrued_days` records days since the last periodic
coupon. `initially_knocked_in` carries historical KI state; rolled notes retain
it and add any KI caused by the constant-spot scenario. Repeated rolls retain
the same accrual origin.

## Using the factors

For small single-asset shocks, a local approximation is:

```text
dV ≈ Delta*dS + 0.5*Gamma*dS² + Vega*dsigma
     + Vanna*dS*dsigma + 0.5*Vomma*dsigma² + Rho*dr

dDelta ≈ Gamma*dS + Vanna*dsigma + Charm*days
dVega  ≈ Vanna*dS + Vomma*dsigma + Veta*days
```

These are local risk decompositions, not reliable approximations for large
shocks or barrier crossings. Reprice those scenarios. Multi-asset decompositions
also need cross-Gamma/cross-Vanna/cross-vol terms, which are not provided here.

## Numerical controls and limitations

Configure `relative_spot_bump`, `volatility_bump`, `theta_days`, `rate_bump`, and
`correlation_bump` under the existing JSON `greeks` section. Relative spot bumps
must be below 1; all bumps must be finite and positive. Centered volatility
differences require every base volatility to exceed `volatility_bump`.
Correlation bumps must preserve positive semidefiniteness in both directions.

MC uses the same seed and Philox innovations across bumps and relative-day
rolls. This reduces noise but does not eliminate it. Price confidence intervals
are not Greek confidence intervals. Risk reports include warnings and the
applied bumps; compare multiple bump sizes, path counts/seeds, or PDE grids
before interpreting high-order estimates. PDE supports single-asset notes;
multi-asset notes use MC. See [VALIDATION.md](VALIDATION.md).

Within a PDE risk report, all revaluations use the baseline grid upper bound,
also returned in `diagnostics.spot_grid_upper`. You may explicitly set
`engines.pde.spot_grid_upper`; it must exceed all scenario spots and contract
levels. A fixed domain avoids spurious sensitivities caused by grid movement.
