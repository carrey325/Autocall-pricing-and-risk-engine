"""Engine configuration dataclasses and strict JSON loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from inspect import signature
import json
from math import isfinite
from pathlib import Path
from typing import Any, Callable

from . import builders
from .market import MarketData
from .products import StructuredNote


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 20_000
    seed: int = 7
    antithetic: bool = True
    batch_size: int = 4_096
    confidence_level: float = 0.95
    steps_per_day: int = 1

    def __post_init__(self) -> None:
        if self.n_paths < 2 or self.batch_size <= 0 or self.steps_per_day <= 0:
            raise ValueError("MC path count, batch size, and steps per day must be positive")
        if self.antithetic and self.n_paths % 2:
            raise ValueError("antithetic Monte Carlo requires an even path count")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence level must be in (0, 1)")


@dataclass(frozen=True)
class PDEConfig:
    spot_grid_points: int = 401
    time_steps_per_day: int = 1
    upper_spot_multiple: float = 4.0
    theta: float = 0.5
    interpolation: str = "linear"
    rannacher_smoothing: bool = True
    spot_grid_upper: float | None = None

    def __post_init__(self) -> None:
        if self.spot_grid_points < 101 or self.spot_grid_points % 2 == 0:
            raise ValueError("PDE spot grid points must be an odd integer >= 101")
        if self.time_steps_per_day <= 0:
            raise ValueError("PDE time steps per day must be positive")
        if self.upper_spot_multiple <= 1.25:
            raise ValueError("PDE upper spot multiple must exceed 1.25")
        if not 0.5 <= self.theta <= 1.0:
            raise ValueError("PDE theta must be in [0.5, 1.0]")
        if self.interpolation != "linear":
            raise ValueError("only linear PDE interpolation is supported")
        if self.spot_grid_upper is not None and (
            not isfinite(self.spot_grid_upper) or self.spot_grid_upper <= 0
        ):
            raise ValueError("fixed spot grid upper bound must be finite and positive")


@dataclass(frozen=True)
class GreekConfig:
    relative_spot_bump: float = 0.01
    volatility_bump: float = 0.01
    rate_bump: float = 0.0001
    theta_days: int = 1
    correlation_bump: float = 0.01

    def __post_init__(self) -> None:
        bumps = (self.relative_spot_bump, self.volatility_bump,
                 self.rate_bump, self.correlation_bump)
        if any(not isfinite(bump) or bump <= 0 for bump in bumps):
            raise ValueError("Greek bumps must be finite and positive")
        if self.relative_spot_bump >= 1:
            raise ValueError("relative spot bump must be less than one")
        if isinstance(self.theta_days, bool) or not isinstance(self.theta_days, int) or self.theta_days <= 0:
            raise ValueError("theta_days must be a positive integer")


@dataclass(frozen=True)
class ResolvedConfig:
    product: StructuredNote
    market: MarketData
    mc: MCConfig
    pde: PDEConfig
    greeks: GreekConfig

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if hasattr(value, "value"):
                return value.value
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(
            {
                "product": asdict(self.product),
                "market": asdict(self.market),
                "engines": {"mc": asdict(self.mc), "pde": asdict(self.pde)},
                "greeks": asdict(self.greeks),
            }
        )


BUILDERS: dict[str, Callable[..., StructuredNote]] = {
    "classic_snowball": builders.make_classic_snowball,
    "wide_snowball": builders.make_wide_snowball,
    "butterfly_snowball": builders.make_butterfly_snowball,
    "dividend_snowball": builders.make_dividend_snowball,
    "stepdown_snowball": builders.make_stepdown_snowball,
    "standard_fcn": builders.make_standard_fcn,
    "discount_entry_fcn": builders.make_discount_entry_fcn,
    "worst_of_snowball": builders.make_worst_of_snowball,
    "worst_of_fcn": builders.make_worst_of_fcn,
}


def _strict_dataclass(cls: type[Any], payload: dict[str, Any], name: str) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return cls(**payload)


def _strict_builder(template: str, parameters: dict[str, Any]) -> StructuredNote:
    try:
        builder = BUILDERS[template]
    except KeyError as exc:
        raise ValueError(f"unknown product template: {template}") from exc
    allowed = set(signature(builder).parameters)
    if template == "discount_entry_fcn":
        allowed = set(signature(builders.make_standard_fcn).parameters)
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unknown product parameters: {sorted(unknown)}")
    return builder(**parameters)


def load_config(path: str | Path) -> ResolvedConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    allowed_top = {"product", "market", "engines", "greeks"}
    unknown_top = set(payload) - allowed_top
    if unknown_top:
        raise ValueError(f"unknown top-level fields: {sorted(unknown_top)}")
    if "product" not in payload or "market" not in payload:
        raise ValueError("configuration requires product and market sections")

    product_payload = dict(payload["product"])
    unknown_product = set(product_payload) - {"template", "parameters"}
    if unknown_product:
        raise ValueError(f"unknown product fields: {sorted(unknown_product)}")
    if "template" not in product_payload:
        raise ValueError("product template is required")
    product = _strict_builder(
        str(product_payload["template"]), dict(product_payload.get("parameters", {}))
    )

    market_payload = dict(payload["market"])
    allowed_market = {"spots", "rate", "dividend_yields", "volatilities", "correlation"}
    unknown_market = set(market_payload) - allowed_market
    if unknown_market:
        raise ValueError(f"unknown market fields: {sorted(unknown_market)}")
    n_assets = len(market_payload.get("spots", ()))
    market_payload.setdefault(
        "correlation",
        [[1.0 if row == column else 0.0 for column in range(n_assets)] for row in range(n_assets)],
    )
    market = MarketData(**market_payload)
    market.validate_for(product)

    engine_payload = dict(payload.get("engines", {}))
    unknown_engines = set(engine_payload) - {"mc", "pde"}
    if unknown_engines:
        raise ValueError(f"unknown engine fields: {sorted(unknown_engines)}")
    mc = _strict_dataclass(MCConfig, dict(engine_payload.get("mc", {})), "MC")
    pde = _strict_dataclass(PDEConfig, dict(engine_payload.get("pde", {})), "PDE")
    greek_config = _strict_dataclass(
        GreekConfig, dict(payload.get("greeks", {})), "Greek"
    )
    return ResolvedConfig(product, market, mc, pde, greek_config)
