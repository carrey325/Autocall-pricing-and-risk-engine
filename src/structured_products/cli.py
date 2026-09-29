"""Argument-driven command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .analytics import calculate_greeks, solve_fair_coupon
from .config import ResolvedConfig, list_examples, load_config, load_example
from .pricing import price


def _add_run_controls(parser: argparse.ArgumentParser, *, method: bool = True) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="path to a JSON configuration")
    source.add_argument("--example", choices=list_examples(), help="built-in configuration")
    if method:
        parser.add_argument("--method", choices=("mc", "pde"), default="mc")
    parser.add_argument("--paths", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="structured-products-pricer")
    from ._version import __version__

    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "price", "greeks"):
        command = subparsers.add_parser(name)
        _add_run_controls(command)
        if name in ("run", "price"):
            command.add_argument("--no-greeks", action="store_true",
                                 help="return valuation only, without risk revaluations")

    fair = subparsers.add_parser("fair-coupon")
    _add_run_controls(fair)
    fair.add_argument("--target-pv", required=True, type=float)
    fair.add_argument("--lower", required=True, type=float)
    fair.add_argument("--upper", required=True, type=float)

    compare = subparsers.add_parser("compare")
    _add_run_controls(compare, method=False)
    return parser


def _overrides(config: ResolvedConfig, args: argparse.Namespace) -> ResolvedConfig:
    mc_updates: dict[str, Any] = {}
    pde_updates: dict[str, Any] = {}
    if args.paths is not None:
        mc_updates["n_paths"] = args.paths
    if args.seed is not None:
        mc_updates["seed"] = args.seed
    if args.grid_size is not None:
        pde_updates["spot_grid_points"] = args.grid_size
    return replace(
        config,
        mc=replace(config.mc, **mc_updates),
        pde=replace(config.pde, **pde_updates),
    )


def _engine(config: ResolvedConfig, method: str):
    return config.mc if method == "mc" else config.pde


def _write_output(payload: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def _execute(config: ResolvedConfig, args: argparse.Namespace) -> dict[str, Any]:
    if args.command in ("price", "run"):
        result = price(
            config.product,
            config.market,
            args.method,
            _engine(config, args.method),
            include_greeks=not args.no_greeks,
            greek_config=config.greeks,
        )
        valuation = result.to_dict()
        risk = valuation.pop("greeks")
        payload = {"price": valuation}
        if risk is not None:
            payload["greeks"] = risk
        if args.command == "run":
            payload["product"] = config.product.product_name
        return payload
    if args.command == "greeks":
        result = calculate_greeks(
            config.product,
            config.market,
            args.method,
            _engine(config, args.method),
            config.greeks,
        )
        return {"greeks": result.to_dict()}
    if args.command == "fair-coupon":
        result = solve_fair_coupon(
            config.product,
            config.market,
            args.target_pv,
            args.method,
            _engine(config, args.method),
            args.lower,
            args.upper,
        )
        return {"fair_coupon": result.to_dict()}
    if args.command == "compare":
        mc = price(config.product, config.market, "mc", config.mc)
        pde = price(config.product, config.market, "pde", config.pde)
        absolute = abs(mc.present_value - pde.present_value)
        relative = absolute / max(abs(pde.present_value), 1.0e-15)
        interval = mc.confidence_interval
        return {
            "mc": mc.to_dict(),
            "pde": pde.to_dict(),
            "comparison": {
                "absolute_difference": absolute,
                "relative_difference": relative,
                "pde_inside_mc_confidence_interval": bool(
                    interval and interval[0] <= pde.present_value <= interval[1]
                ),
            },
        }
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        config = _overrides(
            load_example(args.example) if args.example else load_config(args.config), args
        )
        _write_output(_execute(config, args), args.output)
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
