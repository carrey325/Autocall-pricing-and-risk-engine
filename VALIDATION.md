# Numerical Validation

## Package candidate verification

The 2026-10-24 local run at numerical code commit `6278ed2` passes **261 tests**
with **96% combined statement/branch coverage**, no failures, and no skips.
Environment: Windows 11, Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1,
pytest 9.1.1, coverage 7.17.0. The existing minimum remains 90%.
CI is configured for source tests on Python 3.10–3.13 and artifact installations
on Windows, Linux, and macOS with Python 3.10 and 3.13. This local run does not
claim to have executed that matrix.

Artifact checks run in fresh virtual environments outside the checkout. They
exercise all six bundled presets, MC/PDE pricing, extended risk, fair coupon,
both CLI entry points, version metadata, typing resources, and dependency
consistency. Archive audits exclude local archives, notebooks, and caches.
These are GitHub build checks only; no package-index publication is configured.

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

For `configs/classic_snowball.json` / built-in `classic_snowball`, code `6278ed2`,
20,000 antithetic paths, seed 7, daily steps, batch size 4,096, 95% confidence,
and a 401-point PDE grid on [0,400] with one time step per day, theta=0.5,
Rannacher smoothing and linear interpolation:

| Method | Present value | Standard error / interval |
| --- | ---: | --- |
| Monte Carlo | 1000.693524 | SE 0.698839; 95% CI [999.323825, 1002.063223] |
| PDE | 1000.487825 | Not applicable |

The absolute difference is 0.205699 on a 1,000 notional and the PDE value lies
inside the MC confidence interval. These values are regression references with
documented tolerances, not exact-equality requirements. The approximately
2.06bp difference is not an accuracy bound against a known true price.

## Antithetic uncertainty

Independent samples are the pair means `Y_i=(PV_i+PV_(i+N/2))/2`, because paths
are generated as `base` followed by `-base`. SE is `std(Y,ddof=1)/sqrt(N/2)`;
PV remains the original mean of all paths. Ordinary MC retains iid SE, including
N=2 and N=3. Antithetic N=2 remains priceable but returns SE=None and CI=None,
with an explicit warning. Odd antithetic N and all N<2 remain rejected.
Constant pair means yield zero SE even when individual paths vary.

The 95% interval uses a normal approximation, not a small-sample coverage
guarantee. When CI is absent, `compare` retains its existing boolean field:
`pde_inside_mc_confidence_interval=false` means membership was not verified,
not that consistency was rejected. JSON preserves null and rejects NaN.

Against baseline `9a8ecb5`, all three ordered path-PV arrays had identical
SHA-256 hashes and exactly unchanged means in this environment:

| Seed | PV before and after | Legacy iid SE | Pair SE |
| --- | ---: | ---: | ---: |
| 7 | 1000.693523797 | 0.703761791 | 0.698838724 |
| 17 | 1000.539228942 | 0.701487410 | 0.696416889 |
| 29 | 1001.142088639 | 0.696849942 | 0.692251622 |

These seeds diagnose reproducibility, not empirical 95% coverage. The direction
of the old SE error depends on pair covariance; it was not always too small.

## Absorbing boundary and independent analytic checks

At S=0, each state discounts its own continuation value by `exp(-r*dt)` between
events. Terminal and daily event transforms retain responsibility for cashflows.
This replaces the incorrect neighbor-copy/zero-Delta lower boundary; the upper
zero-gamma boundary is unchanged.

The regression note has reference 100, notional 1,000, 126 remaining days,
initial KI, no autocall or coupons, strike ratio 1, participation 1, floor 0.
Market: spot .01, r=.03, q=0, sigma=.25. Its payoff is `10*min(S_T,100)`.
The independent discounted-stock-minus-call formula gives PV .1 and Delta 10.

| Grid points (upper=400) | Old PDE PV | Corrected PDE PV | Corrected central Delta |
| --- | ---: | ---: | ---: |
| 401 | 10.0797347265 | 0.1000000000 | 10.0000000000 |
| 801 | 5.0398673633 | 0.1000000000 | 10.0000000000 |

Delta uses spots .009/.010/.011 on the same domain. The absolute PV tolerance
1e-6 and Delta tolerance 1e-5 apply only to this locally affine, zero-dividend,
remote-cap fixture. Endpoint equations/cashflows use floating-point tolerances
of 1e-11, not a global PDE error claim.

Additional tests use an independent bond-plus-call-spread oracle, including
floor=.2, participation=.4, zero participation and excluded principal. The
zero-spot principal is `N*max(floor,1-participation)`, not always `N*floor`.
Hand-calculated four-day coupon tests cover amounts (1,2,1) on days (1,3,4),
initial/delayed/maturity-day KI, coupon survival, no KI, excluded principal,
and same-day maturity coupons. With elapsed_days=5, the contingent coupon is
4.5; with coupon_accrued_days=2, fixed coupons become (3,2,1).
Each endpoint case runs with 1/2 daily steps and Rannacher on/off at nonzero
rate and volatility. No change to event ordering or half-step wiring was needed.

## Finite convergence and bump diagnostics (2026-10-24)

Run from an installed development checkout:

```bash
python tools/numerical_diagnostics.py --output results/numerical-diagnostics.json
```

JSON records code HEAD, dirty state, runtime versions, the complete classic
product/market configuration, each engine/bump configuration, PVs, optional
SE/CI, sample hashes, analytic errors, and signed adjacent differences.
The code above was run from clean commit `6278ed2`.
The classic contract is the bundled 252-day preset: reference/spot 100,
notional 1,000, KI=.8, KO=1, annual coupon=.18, r=.03, q=0, sigma=.2,
monthly observations, daily KI, ACT/252. All PDE rows use theta=.5,
Rannacher smoothing and linear interpolation. Sweeps are independent, not
a Cartesian product; all market values except spot are held fixed.

Space sweep: upper=400, two time steps/day; columns are grid points.

| Spot | 401 | 801 | 1601 |
| --- | ---: | ---: | ---: |
| 100.0 | 1000.512763 | 1000.690407 | 1000.932188 |
| 79.9 | 818.345607 | 817.990973 | 817.716285 |
| 80.1 | 820.882823 | 820.580851 | 820.532920 |
| 99.9 | 1000.198490 | 1000.393358 | 1000.646026 |
| 100.1 | 1000.774484 | 1000.961338 | 1001.205367 |

Time sweep: upper=400, 1601 grid points; columns are time steps/day.

| Spot | 1 | 2 | 4 |
| --- | ---: | ---: | ---: |
| 100.0 | 1000.721415 | 1000.932188 | 1000.868126 |
| 79.9 | 818.141990 | 817.716285 | 817.865585 |
| 80.1 | 820.554878 | 820.532920 | 820.499281 |
| 99.9 | 1000.431569 | 1000.646026 | 1000.580860 |
| 100.1 | 1000.998127 | 1001.205367 | 1001.142388 |

Domain sweep: upper=400/600/800, grid points=801/1201/1601, so spacing stays
.5, with two time steps/day. For each spot all three domains produced identical
PVs in this run: 1000.690407 / 817.990973 / 820.580851 / 1000.393358 /
1000.961338 for spots 100 / 79.9 / 80.1 / 99.9 / 100.1, respectively.
This isolates domain extension at fixed spacing and is not a general bound
on truncation error.

Existing PDE Greeks on upper=400, 1601 points, two time steps/day;
only relative_spot_bump changes (volatility_bump=.01, rate_bump=.0001,
theta_days=1, correlation_bump=.01):

| Spot | Relative bump | Delta | Gamma |
| --- | ---: | ---: | ---: |
| 100.0 | .005 | 2.798390 | -.519254 |
| 100.0 | .010 | 2.805113 | -.518995 |
| 100.0 | .020 | 2.831812 | -.517965 |
| 80.1 | .005 | 13.022669 | 1.173155 |
| 80.1 | .010 | 13.152877 | 1.096802 |
| 80.1 | .020 | 13.029571 | .825961 |

These barrier prices have no independent analytic truth in this study.
Space changes do not consistently shrink, time changes can reverse sign, and
near-KI Gamma varies materially with bump. No convergence guarantee is claimed.
Linear interpolation, barrier nonsmoothness and finite differences remain
relevant. Existing cross-engine tolerances and independent European high-order
Greek tests were retained without relaxation.

## Verification commands (2026-10-24)

- `python -m coverage run -m pytest`: 261 passed, no skips or failures.
- `python -m coverage report`: 96%; the 90% gate passed.
- All four existing `examples/*.py`: passed; outputs serialize with `allow_nan=False`.
- Classic `compare`, `price --no-greeks`, and PDE `fair-coupon` CLI: passed.
- Classic `price --paths 2 --no-greeks` and `compare --paths 2`: passed with null SE/CI.
- `python -m build --outdir <fresh-dist-dir>`: wheel and sdist built successfully.
- `python -m twine check <fresh-dist-dir>/*`: both artifacts passed.
- `python tools/verify_distribution.py --dist <fresh-dist-dir>`: both archive
  audits and fresh installations outside the checkout passed, including all
  six presets, pricing/risk/fair-coupon and both CLIs; `pip check` passed.

Network access for declared build/install dependencies required the runner's
approved unsandboxed execution. An initial sandbox TEMP permission failure
was resolved with a writable temporary directory. No package rules, version,
dependencies, CI matrix or notebook exclusions were changed. These are local
Windows results; remote CI and other operating systems were not run here.

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
