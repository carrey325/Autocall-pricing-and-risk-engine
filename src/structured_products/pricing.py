"""Public pricing dispatcher with explicit capability validation."""

from __future__ import annotations

from dataclasses import replace

from .config import GreekConfig, MCConfig, PDEConfig
from .engines.mc import price_mc
from .engines.pde import price_pde
from .enums import PricingMethod
from .market import MarketData
from .products import StructuredNote
from .results import PricingResult


def price(
    product: StructuredNote,
    market: MarketData,
    method: PricingMethod | str,
    engine_config: MCConfig | PDEConfig | None = None,
    *,
    include_greeks: bool = False,
    greek_config: GreekConfig | None = None,
) -> PricingResult:
    """Price a note, optionally attaching per-asset risk to the same result."""
    try:
        selected = method if isinstance(method, PricingMethod) else PricingMethod(method.lower())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"unsupported pricing method: {method}") from exc
    if selected is PricingMethod.MC:
        if engine_config is not None and not isinstance(engine_config, MCConfig):
            raise TypeError("MC pricing requires MCConfig")
        resolved = engine_config or MCConfig()
        result = price_mc(product, market, resolved)
    else:
        if engine_config is not None and not isinstance(engine_config, PDEConfig):
            raise TypeError("PDE pricing requires PDEConfig")
        resolved = engine_config or PDEConfig()
        result = price_pde(product, market, resolved)
    if include_greeks:
        from .analytics.greeks import _calculate_greeks

        risk = _calculate_greeks(
            product, market, selected, resolved, greek_config, result.present_value
        )
        result = replace(result, greeks=risk)
    return result


__all__ = ["price"]
