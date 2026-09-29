# Numerical Validation

## Test strategy

The suite starts with deterministic, hand-calculated paths for every payoff
branch: early autocall, knock-in then autocall, knock-in loss, recovery,
no-event maturity, exact barrier touches, same-day priority, fixed coupons,
discount entry, and worst-of aggregation.

Monte Carlo tests cover fixed-seed reproducibility, batch-size invariance,
antithetic handling, the zero-volatility limit, confidence intervals, standard
error behavior, and the rule that knock-in cannot occur after redemption.

PDE tests cover the deterministic limit, spot-grid refinement, all supported
single-asset schedule variants, and explicit rejection of multi-asset products.
Cross-engine acceptance uses
`abs(MC - PDE) <= max(4 * MC standard error, 0.002 * notional)`.

This tolerance combines simulation uncertainty with finite-difference
discretization and replaces brittle equality at twelve decimal places.

## Reference comparison

For `configs/classic_snowball.json` with 20,000 antithetic paths and a
401-point PDE grid:

| Method | Present value | Standard error / interval |
| --- | ---: | --- |
| Monte Carlo | 1000.693524 | SE 0.703762; 95% CI [999.314176, 1002.072872] |
| PDE | 1000.487825 | Not applicable |

The absolute difference is 0.205699 on a 1,000 notional and the PDE value lies
inside the MC confidence interval. These values are regression references with
documented tolerances, not exact-equality requirements.

## Reproducibility

- MC innovations use NumPy Philox streams keyed by seed and relative day.
- Processing batch size does not change the generated innovations.
- Fair-coupon trials and Greeks restart the same stream.
- JSON output rejects NaN and NumPy-only scalar types.

## Known limitations

- Flat Black-Scholes market inputs only.
- Daily-close monitoring; no intraday bridge correction.
- No local/stochastic volatility, rate curves, discrete dividends, credit risk,
  business-day calendars, or physical-settlement inventory accounting.
- PDE supports one underlying only. Worst-of products are Monte Carlo only.

## Extended risk validation

The extended suite compares Lambda, Vanna, Vomma, Charm, and Veta against an
independent Black-Scholes reference: a note paying
`1000 - 10 * max(100 - S_T, 0)`. It is constructed with initial KI, no autocall,
and no coupons. At S=90, r=3%, q=1%, sigma=25%, and 126 trading days, a 4,001-point
PDE grid is checked against analytical derivatives. Vanna/Vomma are tested at
half and twice the central bump (0.5% relative spot; 0.005 absolute volatility).
Charm/Veta are checked against analytical two-day forward changes, divided by
two. This tests units, signs, and normalization independently of the pricers.

Other tests cover aliases in standalone and nested JSON, Lambda at zero PV,
per-asset MC replay and batch invariance, default CLI risk output, valuation-only
opt-out, accrual before/after coupons, repeated rolls, KI history, future KI
monitoring, same-day KO priority, and zero remaining exposure after autocall.

### Snowball bump study (2026-09-29)

Using `configs/classic_snowball.json`, seed 7, a 1,601-point PDE grid, and 100,000
antithetic MC paths. Each row uses `GreekConfig(relative_spot_bump=h,
volatility_bump=h, theta_days=1)`; PDE scenarios share a fixed spatial domain.
The table is a numerical diagnostic, not an accuracy guarantee for every Greek.

| Method | h | Lambda | Vanna | Vomma | Charm / day | Veta / day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PDE | 0.005 | 0.283235 | 47.063628 | 2385.922871 | -0.028155 | 3.733493 |
| PDE | 0.010 | 0.283900 | 47.096394 | 2377.725572 | -0.028005 | 3.739955 |
| PDE | 0.020 | 0.286536 | 47.153597 | 2344.580832 | -0.027333 | 3.765100 |
| MC | 0.005 | 0.248962 | 35.226599 | 4007.333866 | 0.263211 | -3.340770 |
| MC | 0.010 | 0.271752 | 40.752865 | 3992.799879 | 0.067382 | -1.027823 |
| MC | 0.020 | 0.280060 | 47.391285 | 2723.344782 | -0.006949 | -0.501542 |

The PDE estimates are relatively stable across these bumps. The MC time Greeks
are visibly noisy, including sign changes, despite common random numbers.
These MC estimates are not converged hedge quantities. Increase paths, repeat
independent seeds, vary bumps, and compare with PDE where applicable. The price
SE acceptance rule above does not apply to Greeks. No Greek confidence interval
is currently estimated, and PDE Greeks still require grid/time-step convergence.
