"""Typed, JSON-serializable public result objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonify(item) for item in value]
    return value


class SerializableResult:
    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(frozen=True)
class PricingResult(SerializableResult):
    method: str
    present_value: float
    net_value: float
    standard_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    knock_in_probability: float | None = None
    knock_out_probability: float | None = None
    expected_redemption_time: float | None = None
    expected_coupon_cashflow: float | None = None
    diagnostics: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    greeks: GreeksResult | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.greeks is not None:
            payload["greeks"] = self.greeks.to_dict()
        return payload


@dataclass(frozen=True)
class GreeksResult(SerializableResult):
    method: str
    base_price: float
    delta: tuple[float, ...]
    gamma: tuple[float, ...]
    vega: tuple[float, ...]
    theta: float
    rho: float
    correlation_sensitivity: float | None
    bumps: dict[str, float]
    lambda_: tuple[float | None, ...] = ()
    vanna: tuple[float, ...] = ()
    vomma: tuple[float, ...] = ()
    charm: tuple[float, ...] = ()
    veta: tuple[float, ...] = ()
    units: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def omega(self) -> tuple[float | None, ...]:
        return self.lambda_

    @property
    def volga(self) -> tuple[float, ...]:
        return self.vomma

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["lambda"] = payload.pop("lambda_")
        payload["omega"] = payload["lambda"]
        payload["volga"] = payload["vomma"]
        return payload


@dataclass(frozen=True)
class FairCouponResult(SerializableResult):
    method: str
    coupon_rate: float
    target_present_value: float
    residual: float
    iterations: int
    bracket: tuple[float, float]
