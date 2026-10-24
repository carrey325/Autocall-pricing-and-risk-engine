"""Run finite numerical diagnostics against the installed package, emitting strict JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

import numpy as np
import scipy
from scipy.stats import norm

from structured_products import (
    CouponFeature, CouponRule, GreekConfig, KnockInFeature, MarketData,
    PDEConfig, RedemptionFeature, StructuredNote, calculate_greeks, load_example, price,
)
from structured_products.payoffs import VectorizedPayoffState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        result = subprocess.run(["git", "-C", str(root), *arguments],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    config = load_example("classic_snowball")
    status = git("status", "--porcelain")
    report = {
        "metadata": {"code_head": git("rev-parse", "HEAD"),
                     "working_tree_dirty": None if status is None else bool(status),
                     "platform": platform.platform(), "python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "classic_configuration": config.to_dict(),
        "mc": [], "analytic_checks": [], "pde_sweeps": {}, "greek_bumps": [],
        "interpretation": {
            "mc": "Three seeds diagnose reproducibility, not confidence-interval coverage.",
            "analytic": "The 1e-6 bound applies only to this remote-cap, low-spot fixture.",
            "sweeps": "No analytic truth for the barrier cases; signed adjacent differences only.",
            "greeks": "Bump sensitivity is not a convergence or accuracy guarantee.",
        },
    }

    original = VectorizedPayoffState.result
    for seed in (7, 17, 29):
        samples = []

        def capture(state):
            evaluation = original(state)
            samples.append(evaluation.present_values.copy())
            return evaluation

        mc_config = replace(config.mc, seed=seed, n_paths=20_000)
        with patch.object(VectorizedPayoffState, "result", capture):
            result = price(config.product, config.market, "mc", mc_config)
        values, = samples
        pairs = (values[:10_000] + values[10_000:]) / 2
        report["mc"].append({
            "config": asdict(mc_config), "result": result.to_dict(),
            "independent_pair_se": float(np.std(pairs, ddof=1) / np.sqrt(10_000)),
            "legacy_iid_se": float(np.std(values, ddof=1) / np.sqrt(20_000)),
            "sample_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        })
    report["classic_pde_reference"] = {
        "config": asdict(config.pde),
        "result": price(config.product, config.market, "pde", config.pde).to_dict(),
    }

    note = StructuredNote(
        reference_spots=(100.0,), notional=1000.0, maturity_days=126,
        coupon=CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION),
        redemption=RedemptionFeature(),
        knock_in=KnockInFeature(.8, initially_knocked_in=True),
    )
    market = MarketData((.01,), .03, (0.0,), (.25,))
    # Independent S-discounted minus call value avoids bond-minus-put cancellation.
    spot, strike, time, rate, vol = .01, 100.0, .5, .03, .25
    d1 = (np.log(spot / strike) + (rate + vol**2 / 2) * time) / (vol * np.sqrt(time))
    d2 = d1 - vol * np.sqrt(time)
    call = spot * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)
    expected = float(10 * (spot - call))
    for points in (401, 801):
        engine = PDEConfig(spot_grid_points=points, spot_grid_upper=400)
        result = price(note, market, "pde", engine)
        bumped = [price(note, replace(market, spots=(s,)), "pde", engine).present_value
                  for s in (.009, .011)]
        delta = (bumped[1] - bumped[0]) / .002
        error = result.present_value - expected
        report["analytic_checks"].append({
            "product": asdict(note), "market": asdict(market), "config": asdict(engine),
            "result": result.to_dict(), "analytic_pv": expected, "pv_error": error,
            "delta_spots": [.009, .01, .011], "delta": delta, "analytic_delta": 10.0,
            "pv_tolerance": 1e-6, "delta_tolerance": 1e-5,
            "passed": abs(error) <= 1e-6 and abs(delta - 10) <= 1e-5,
        })

    base_pde = PDEConfig(spot_grid_points=1601, spot_grid_upper=400, time_steps_per_day=2)
    sweeps = {
        "space": [replace(base_pde, spot_grid_points=n) for n in (401, 801, 1601)],
        "time": [replace(base_pde, time_steps_per_day=n) for n in (1, 2, 4)],
        "domain": [replace(base_pde, spot_grid_upper=u, spot_grid_points=n)
                   for u, n in ((400, 801), (600, 1201), (800, 1601))],
    }
    for name, engines in sweeps.items():
        rows = []
        for spot in (100.0, 79.9, 80.1, 99.9, 100.1):
            market = replace(config.market, spots=(spot,))
            previous = None
            for engine in engines:
                result = price(config.product, market, "pde", engine)
                rows.append({"spot": spot, "market": asdict(market), "config": asdict(engine),
                             "result": result.to_dict(),
                             "change_from_previous": None if previous is None
                             else result.present_value - previous})
                previous = result.present_value
        report["pde_sweeps"][name] = rows

    for spot in (100.0, 80.1):
        market = replace(config.market, spots=(spot,))
        for bump in (.005, .01, .02):
            greek_config = GreekConfig(relative_spot_bump=bump)
            risk = calculate_greeks(config.product, market, "pde", base_pde, greek_config)
            report["greek_bumps"].append({
                "spot": spot, "market": asdict(market), "config": asdict(base_pde),
                "greek_config": asdict(greek_config),
                "present_value": risk.base_price, "delta": risk.delta, "gamma": risk.gamma,
                "warnings": risk.warnings,
            })

    # Enums are str subclasses; optional SE/CI remain JSON null.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote numerical diagnostics to {args.output}")
    return 0 if all(row["passed"] for row in report["analytic_checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
