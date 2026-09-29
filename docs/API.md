# Public Python API

Install from GitHub as shown in the README. The distribution name is
`structured-products-pricer`; the Python import is `structured_products`.
The supported user-facing objects are exported from that namespace. Modules
under `engines` and names prefixed with `_` are implementation details.

## Calculations

| Function | Required arguments | Optional arguments | Result |
| --- | --- | --- | --- |
| `price` | product, market, method | engine_config=None; keyword include_greeks=False, greek_config=None | `PricingResult` |
| `calculate_greeks` | product, market, method, engine_config | greek_config=None | `GreeksResult` |
| `solve_fair_coupon` | product, market, target_pv, method, engine_config, lower_coupon, upper_coupon | max_iterations=100, tolerance=1e-8 | `FairCouponResult` |

`method` accepts `"mc"`, `"pde"`, or `PricingMethod`. Use `MCConfig` with MC and
`PDEConfig` with PDE. Passing the wrong configuration type raises `TypeError`.
PDE rejects multi-asset products with `ValueError`; it never silently switches
engines. Failed coupon brackets and invalid contract/market parameters also
raise `ValueError`. The CLI reports errors to stderr and returns a nonzero code.

## Contracts, markets, and templates

`StructuredNote` is immutable and holds reference spots, notional, maturity in
trading days, issue price, and features (`KnockInFeature`, `AutocallFeature`,
`CouponFeature`, `RedemptionFeature`). Market inputs are separate in `MarketData`:
current `spots`, continuously compounded `rate`, `dividend_yields`,
`volatilities`, and a correlation matrix. Values such as 0.20 mean 20% per year.
Asset ordering must match the contract; correlation must be positive semidefinite.

Builders are keyword-based templates that produce the same `StructuredNote`:

- `make_classic_snowball`, `make_wide_snowball`, `make_butterfly_snowball`,
  `make_dividend_snowball`, `make_stepdown_snowball`.
- `make_standard_fcn`, `make_discount_entry_fcn`.
- `make_worst_of_snowball`, `make_worst_of_fcn`.

Use `help(make_classic_snowball)` or an IDE for the complete typed parameter
list. Existing builder signatures are retained in 0.3.0rc1. For custom terms,
compose features or use `dataclasses.replace`; do not mutate a note in place.
See [MODEL_SPEC.md](../MODEL_SPEC.md) for cashflows and event ordering.

## Configuration and packaged examples

- `load_config(path)` reads a user-supplied JSON file.
- `config_from_dict(payload)` validates the same template-based input schema.
- `list_examples()` returns six bundled preset names.
- `load_example(name)` reads a preset using package resources, from any directory.

All loaders return `ResolvedConfig`, with `.product`, `.market`, `.mc`, `.pde`,
and `.greeks`. Input sections are `product`, `market`, `engines`, and `greeks`;
unknown fields are rejected. `product` contains `template` and `parameters`.
`ResolvedConfig.to_dict()` describes resolved objects for inspection; it is not
a round-trip input configuration. `PricingResult.to_dict()` and risk/solver
result serialization produce JSON-compatible values.

## Results and risk

`PricingResult.present_value` includes principal. `net_value` subtracts the
undiscounted issue price. MC additionally reports standard error, confidence
interval, KI/KO probabilities, expected redemption time, and expected coupons.
PDE leaves unavailable MC diagnostics as `None`.

Call `price(..., include_greeks=True)` to populate `.greeks`; otherwise it is
`None`. Greeks are sensitivities of total PV. Array entries follow asset order.
Lambda is exposed as `.lambda_` in Python and `"lambda"` in JSON; `.omega` is an
alias. `.volga` aliases `.vomma`. Undefined elasticity returns `None`/JSON null.
Units, bumps, warnings, and time-roll diagnostics are included in the result.
See [RISK_ANALYTICS.md](../RISK_ANALYTICS.md) for all definitions and limitations.

The CLI defaults to price plus risk, while the Python pricing function defaults
to price only to make computational cost explicit. Both interfaces use the same
calculation code. Runtime dependencies remain NumPy and SciPy; development and
build tools are optional extras.
