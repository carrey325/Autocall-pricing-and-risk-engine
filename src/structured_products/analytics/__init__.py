"""Public analytics built on the unified pricing API."""

from .fair_coupon import solve_fair_coupon
from .greeks import calculate_greeks

__all__ = ["calculate_greeks", "solve_fair_coupon"]
