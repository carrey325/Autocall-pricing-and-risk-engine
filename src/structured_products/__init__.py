"""Public API for autocallable pricing and risk analytics."""

from .builders import (
    make_butterfly_snowball,
    make_classic_snowball,
    make_discount_entry_fcn,
    make_dividend_snowball,
    make_standard_fcn,
    make_stepdown_snowball,
    make_wide_snowball,
    make_worst_of_fcn,
    make_worst_of_snowball,
)
from .analytics import calculate_greeks, solve_fair_coupon
from .analytics.fair_coupon import with_flat_coupon
from .analytics.stress import StressSpec, default_stresses, stress_test
from .analytics.terms import with_terms, compare_terms
from .analytics.surface_risk import (ScenarioShock, ScenarioCashflow, Scenario, RiskSurfaceResult,
    transform_scenario, advance_scenario, price_scenario, scenario_risk, risk_surface, default_risk_axes, shock_model)
from .enums import BasketRule, CouponRule, PricingMethod, SameDayPriority
from .config import (
    GreekConfig, MCConfig, PDEConfig, ResolvedConfig,
    config_from_dict, list_examples, load_config, load_example,
)
from ._version import __version__
from .market import MarketData
from .models import PricingModel, LocalVolModel, TimeVolModel
from .cboe import CboeAdapterResult, adapt_cboe_eod
from .marketdata import (
    MarketCurves, MarketSnapshot, TradingCalendar, MarketInput, QuotePolicy,
    ValidatedQuote, QuoteRecord, QuoteValidationResult, load_market_data, validate_quotes,
)
from .volatility import (
    SSVISurface, VarianceDerivatives, CalibrationConfig, CalibrationResult, QualityGate,
    black_price, implied_volatility, calibrate_ssvi,
)
from .pricing import price
from .products import (
    AutocallFeature,
    CouponFeature,
    KnockInFeature,
    RedemptionFeature,
    StructuredNote,
)
from .results import FairCouponResult, GreeksResult, PricingResult


__all__ = [
    "CboeAdapterResult", "adapt_cboe_eod",
    "with_flat_coupon", "StressSpec", "default_stresses", "stress_test", "with_terms", "compare_terms",
    "ScenarioShock", "ScenarioCashflow", "Scenario", "RiskSurfaceResult", "transform_scenario",
    "advance_scenario", "price_scenario", "scenario_risk", "risk_surface", "default_risk_axes", "shock_model",
    "AutocallFeature",
    "BasketRule",
    "CouponFeature",
    "CouponRule",
    "calculate_greeks",
    "FairCouponResult",
    "GreeksResult",
    "GreekConfig",
    "KnockInFeature",
    "MarketData",
    "PricingModel", "LocalVolModel", "TimeVolModel",
    "MarketCurves", "MarketSnapshot", "TradingCalendar", "MarketInput", "QuotePolicy",
    "ValidatedQuote", "QuoteRecord", "QuoteValidationResult", "load_market_data", "validate_quotes",
    "SSVISurface", "VarianceDerivatives", "CalibrationConfig", "CalibrationResult", "QualityGate",
    "black_price", "implied_volatility", "calibrate_ssvi",
    "MCConfig",
    "PDEConfig",
    "PricingMethod",
    "PricingResult",
    "RedemptionFeature",
    "ResolvedConfig",
    "SameDayPriority",
    "StructuredNote",
    "make_butterfly_snowball",
    "make_classic_snowball",
    "make_discount_entry_fcn",
    "make_dividend_snowball",
    "make_standard_fcn",
    "make_stepdown_snowball",
    "make_wide_snowball",
    "make_worst_of_fcn",
    "make_worst_of_snowball",
    "load_config",
    "load_example",
    "list_examples",
    "config_from_dict",
    "price",
    "solve_fair_coupon",
]
