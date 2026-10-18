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
from .workflow_io import load_model, read_json, write_json, write_csv


def _add_model_controls(parser,required=False):
    source=parser.add_mutually_exclusive_group(required=required)
    source.add_argument('--model',help='serialized LocalVolModel or TimeVolModel JSON')
    source.add_argument('--calibration',help='accepted calibration JSON; revalidated once')
    source.add_argument('--surface',help='theoretical SSVI JSON')
    parser.add_argument('--curves',help='curve JSON or CSV, required with --surface')


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
    for command in (subparsers.choices['price'],subparsers.choices['run'],fair):
        _add_model_controls(command)

    compare = subparsers.add_parser("compare")
    _add_run_controls(compare, method=False)
    risk = subparsers.add_parser('risk-surface', help='single-asset PDE surface risk with reconstructable scenarios')
    _add_run_controls(risk, method=False)
    _add_model_controls(risk,required=True)
    risk.add_argument('--spot-ratios', help='comma-separated spot/reference ratios')
    risk.add_argument('--rho-shifts', help='comma-separated rho shifts; use --rho-shifts=-0.1,0,0.1')
    risk.add_argument('--roll-days', help='comma-separated elapsed trading days')
    risk.add_argument('--initial-ki', help='comma-separated false,true histories')
    risk.add_argument('--metrics', help='comma-separated present_value,delta,gamma,atm_level_sensitivity,rho_skew_sensitivity')
    risk.add_argument('--mc-checkpoints', help='JSON list of ScenarioShock objects')
    risk.add_argument('--csv-output', help='optional additional CSV output')
    for name in ('stress','compare-terms'):
        command=subparsers.add_parser(name)
        _add_run_controls(command)
        command.set_defaults(method='pde')
        _add_model_controls(command,required=True)
        command.add_argument('--scenarios',help='JSON StressSpec list; default 46 mark/observed-close stresses')
        command.add_argument('--csv-output')
        if name=='compare-terms':
            command.add_argument('--coupons',default='.1,.15,.2')
            command.add_argument('--ki-ratios',default='.6,.7,.8')
            command.add_argument('--ko-ratios',default='.95,1,1.05')
    calibrate=subparsers.add_parser('calibrate')
    calibrate.add_argument('--market-input',required=True,help='canonical four-file directory')
    calibrate.add_argument('--quality-config',help='CalibrationConfig JSON')
    calibrate.add_argument('--quote-policy',help='QuotePolicy JSON, already in normalized price units')
    calibrate.add_argument('--output',required=True)
    adapter=subparsers.add_parser('adapt-cboe')
    adapter.add_argument('--raw-csv',required=True)
    adapter.add_argument('--schedule',required=True)
    adapter.add_argument('--timestamp',required=True)
    adapter.add_argument('--root',choices=('SPX','SPXW'),default='SPX')
    adapter.add_argument('--expiries',help='comma-separated ISO dates')
    adapter.add_argument('--output',required=True,help='canonical output directory, local vendor-derived data')
    research=subparsers.add_parser('research',help='complete offline synthetic or real-input research report')
    research.add_argument('--output',required=True,help='output directory with checkpoints, plots and HTML')
    research.add_argument('--analysis-config',help='analysis JSON; omitted uses full approved default grid')
    research.add_argument('--config',help='product JSON config')
    research.add_argument('--market-input',help='canonical market input directory; omitted is bundled synthetic')
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
    model=None
    if getattr(args,'model',None) or getattr(args,'calibration',None) or getattr(args,'surface',None):
        model=load_model(model=args.model,calibration=args.calibration,surface=args.surface,curves=args.curves)
    if args.command in ('stress','compare-terms'):
        from .analytics.stress import StressSpec,stress_test
        from .analytics.terms import compare_terms
        kwargs=dict(scenarios=None if args.scenarios is None else tuple(StressSpec.from_dict(s) for s in read_json(args.scenarios)),
                    method=args.method,pde_config=config.pde,mc_config=config.mc)
        if args.command=='compare-terms':
            kwargs.update(coupons=tuple(map(float,args.coupons.split(','))),ki_ratios=tuple(map(float,args.ki_ratios.split(','))),
                          ko_ratios=tuple(map(float,args.ko_ratios.split(','))))
        result=(stress_test if args.command=='stress' else compare_terms)(config.product,config.market,model,**kwargs)
        if args.csv_output: result.write_csv(args.csv_output)
        return result.to_dict()
    if args.command == 'risk-surface':
        from .analytics.surface_risk import ScenarioShock, risk_surface
        def axis(text, convert):
            return None if text is None else tuple(convert(v.strip()) for v in text.split(','))
        def boolean(text):
            if text.lower() not in ('true','false'):
                raise ValueError('--initial-ki accepts true/false')
            return text.lower() == 'true'
        result = risk_surface(config.product,config.market,model,
            spot_ratios=axis(args.spot_ratios,float),rho_shifts=axis(args.rho_shifts,float),
            roll_days=axis(args.roll_days,int),initial_ki=axis(args.initial_ki,boolean),
            metrics=axis(args.metrics,str),pde_config=config.pde,mc_config=config.mc,
            mc_checkpoints=() if args.mc_checkpoints is None else tuple(ScenarioShock(**s) for s in read_json(args.mc_checkpoints)))
        if args.csv_output:
            Path(args.csv_output).parent.mkdir(parents=True,exist_ok=True)
            result.write_csv(args.csv_output)
        if args.output and Path(args.output).suffix.lower() == '.csv':
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            result.write_csv(args.output)
            args.output = None
        return result.to_dict()
    if args.command in ("price", "run"):
        if model is not None and not args.no_greeks:
            raise ValueError('explicit model price requires --no-greeks; use risk-surface for model sensitivities')
        result = price(
            config.product,
            config.market,
            args.method,
            _engine(config, args.method),
            include_greeks=not args.no_greeks,
            greek_config=config.greeks,
            model=model,
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
            model=model,
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
        if args.command=='research':
            from .research import run_research
            def progress(stage,current,total,cell):
                if current==0 or total is None or current%10==0 or current==total:
                    print(f'{stage}: {current}/{total or "?"}',file=sys.stderr,flush=True)
            result=run_research(args.output,config=load_config(args.config) if args.config else None,
                analysis=read_json(args.analysis_config) if args.analysis_config else None,
                market_input=args.market_input,progress=progress)
            print(json.dumps(dict(status=result['status'],summary=result['summary'],input_hash=result['input_hash']),allow_nan=False))
            return 0
        if args.command=='adapt-cboe':
            from .cboe import adapt_cboe_eod
            result=adapt_cboe_eod(args.raw_csv,read_json(args.schedule),args.timestamp,root=args.root,
                expiries=None if args.expiries is None else tuple(args.expiries.split(',')),output_directory=args.output)
            print(json.dumps(dict(provenance=result.provenance,parity=result.parity),allow_nan=False))
            return 0
        if args.command=='calibrate':
            from .marketdata import load_market_data,validate_quotes,QuotePolicy
            from .volatility import calibrate_ssvi,CalibrationConfig
            data=load_market_data(args.market_input)
            validation=validate_quotes(data.rows,data.snapshot,data.calendar,data.curves,
                QuotePolicy(**read_json(args.quote_policy)) if args.quote_policy else None)
            fit=calibrate_ssvi(validation,CalibrationConfig(**read_json(args.quality_config)) if args.quality_config else None)
            write_json(args.output,fit.to_dict())
            write_csv(Path(args.output).with_suffix('.cleaning.csv'),[r.to_dict() for r in validation.records])
            print(json.dumps(dict(accepted=fit.accepted,real_data_acceptance=fit.to_dict()['real_data_acceptance'],
                gates=[g.to_dict() for g in fit.gates]),allow_nan=False))
            return 0
        config = _overrides(
            load_example(args.example) if args.example else load_config(args.config), args
        )
        _write_output(_execute(config, args), args.output)
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
