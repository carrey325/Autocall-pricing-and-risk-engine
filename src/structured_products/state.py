"""Lifecycle state and deterministic cashflow records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cashflow:
    day: int
    amount: float
    kind: str


@dataclass(frozen=True)
class PathEvaluation:
    cashflows: tuple[Cashflow, ...]
    knocked_in: bool
    knocked_out: bool
    redemption_day: int
    coupon_cashflow: float

    def present_value(self, rate: float, day_count: int) -> float:
        from math import exp

        return sum(
            cashflow.amount * exp(-rate * cashflow.day / day_count)
            for cashflow in self.cashflows
        )


@dataclass(frozen=True)
class BatchEvaluation:
    present_values: np.ndarray
    knocked_in: np.ndarray
    knocked_out: np.ndarray
    redemption_days: np.ndarray
    coupon_cashflows: np.ndarray
