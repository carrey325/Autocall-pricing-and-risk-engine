# Model and Contract Specification

## Market model

Under the risk-neutral measure, each underlying follows
`dS_i / S_i = (r - q_i) dt + sigma_i dW_i`.

Rates, dividend yields, volatilities, and correlations are flat. Time is measured
in integer trading days using ACT/252 unless a product supplies another positive
day count. Barrier monitoring uses daily closes.

For asset `i`, performance is `S_i(t) / reference_i`. A single-asset product
uses its only performance; a worst-of product uses the minimum performance.

## Lifecycle

The default same-day order is:

1. Test autocall. If hit, pay configured principal and the annualized redemption
   coupon accrued from issue to that observation day, then terminate.
2. If still alive, update the persistent knock-in state on an eligible day.
3. If still alive, pay a scheduled periodic coupon when its survival rule allows.
4. At maturity, settle principal and any contingent maturity coupon exactly once.

Barrier touches are inclusive. Knock-in is never observed after redemption.

## Redemption

Without a prior knock-in, maturity principal is the notional. After knock-in,
when final performance is below the strike ratio, the principal ratio is
`max(floor, 1 + participation * (performance / strike_ratio - 1))`.

Otherwise principal is returned in full. Settlement is valued as its cash
equivalent. Result PV includes principal; net value equals PV minus issue price.

Snowball coupons are contingent at redemption. Fixed-coupon notes pay annualized
periodic coupons over each scheduled accrual interval. Discount-entry FCNs use
the same rules with the initial knock-in state set to true.

## Numerical methods

Monte Carlo simulates correlated risk-neutral paths with Philox streams,
antithetic variates, and daily streaming payoff state. A fixed seed identifies
the same innovations for pricing, fair-coupon trials, and Greek bumps.

The PDE engine solves two value states (alive/not knocked in and alive/knocked
in) backward on a one-dimensional spot grid. It applies generic KI, KO, coupon,
and maturity transforms on exact event days. Multi-asset PDE pricing is rejected.

## Valuation-time risk

Pricing can include Lambda/Omega, Vanna, Vomma/Volga, Charm, and Veta alongside
the existing Greeks. These are finite-difference estimates on total PV, reported
per underlying. PDE risk scenarios share the baseline spot grid upper bound;
changing a market spot does not move the numerical domain.

Time risk uses an explicit constant-spot daily-close scenario. Knock-in and
autocall events are applied before shifting future schedules. Remaining value
excludes cash already paid; a redeemed contract has no remaining exposure.
`elapsed_days` preserves issue-to-redemption coupon accrual and
`coupon_accrued_days` preserves the first remaining periodic coupon interval.
Discounting still uses only remaining days. Charm and Veta are changes per
elapsed trading day; Theta retains the total change over the configured horizon.
See [RISK_ANALYTICS.md](RISK_ANALYTICS.md) for formulas and output conventions.
