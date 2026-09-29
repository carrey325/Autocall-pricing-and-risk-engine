"""Compute PV and per-underlying risk in one public API call."""

import json

from structured_products import GreekConfig, PDEConfig, load_example, price


config = load_example("classic_snowball")
result = price(
    config.product,
    config.market,
    "pde",
    PDEConfig(spot_grid_points=1601),
    include_greeks=True,
    greek_config=GreekConfig(relative_spot_bump=0.01, volatility_bump=0.01),
)
print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
