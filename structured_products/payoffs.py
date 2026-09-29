"""Canonical event ordering and deterministic structured-note cashflows."""

from __future__ import annotations

from math import exp

import numpy as np

from .enums import BasketRule, CouponRule, SameDayPriority
from .products import StructuredNote
from .state import BatchEvaluation, Cashflow, PathEvaluation


def basket_performance(spots: np.ndarray, product: StructuredNote) -> np.ndarray:
    ratios = spots / np.asarray(product.reference_spots, dtype=float)
    if product.basket_rule is BasketRule.WORST_OF:
        return np.min(ratios, axis=-1)
    return ratios[..., 0]


def _period_accrual(product: StructuredNote, index: int) -> float:
    previous = -product.coupon_accrued_days if index == 0 else product.coupon.payment_days[index - 1]
    return (product.coupon.payment_days[index] - previous) / product.day_count


def _principal_cash(product: StructuredNote, performance: float, knocked_in: bool) -> float:
    if not product.redemption.include_principal:
        return 0.0
    ratio = 1.0
    if knocked_in and performance < product.redemption.strike_ratio:
        ratio = max(
            product.redemption.principal_floor,
            1.0
            + product.redemption.downside_participation
            * (performance / product.redemption.strike_ratio - 1.0),
        )
    return product.notional * ratio


def evaluate_path(path: np.ndarray, product: StructuredNote) -> PathEvaluation:
    values = np.asarray(path, dtype=float)
    if values.shape != (product.maturity_days + 1, product.n_assets):
        raise ValueError("path shape must be (maturity_days + 1, n_assets)")
    if np.any(values <= 0):
        raise ValueError("path spots must be positive")

    knocked_in = bool(product.knock_in and product.knock_in.initially_knocked_in)
    knocked_out = False
    redemption_day = product.maturity_days
    coupon_cashflow = 0.0
    cashflows: list[Cashflow] = []
    observation_index = (
        {day: index for index, day in enumerate(product.autocall.observation_days)}
        if product.autocall
        else {}
    )
    payment_index = {
        day: index for index, day in enumerate(product.coupon.payment_days)
    }

    for day in range(1, product.maturity_days + 1):
        performance = float(basket_performance(values[day], product))

        def process_knock_in() -> None:
            nonlocal knocked_in
            if (
                product.knock_in
                and day >= product.knock_in.monitoring_start_day
                and performance <= product.knock_in.barrier_ratio
            ):
                knocked_in = True

        def process_knock_out() -> bool:
            nonlocal knocked_out, redemption_day, coupon_cashflow
            if not product.autocall or day not in observation_index:
                return False
            index = observation_index[day]
            if performance < product.autocall.barrier_ratios[index]:
                return False
            principal = product.notional if product.redemption.include_principal else 0.0
            coupon = (
                product.notional
                * product.autocall.coupon_rates[index]
                * (product.elapsed_days + day)
                / product.day_count
            )
            if principal:
                cashflows.append(Cashflow(day, principal, "principal"))
            if coupon:
                cashflows.append(Cashflow(day, coupon, "autocall_coupon"))
            knocked_out = True
            redemption_day = day
            coupon_cashflow += coupon
            return True

        if product.same_day_priority is SameDayPriority.KNOCK_OUT_FIRST:
            if process_knock_out():
                break
            process_knock_in()
        else:
            process_knock_in()
            if process_knock_out():
                break

        if product.coupon.rule is CouponRule.FIXED_PERIODIC and day in payment_index:
            index = payment_index[day]
            if not knocked_in or product.coupon.survives_knock_in:
                coupon = (
                    product.notional
                    * product.coupon.rate_schedule[index]
                    * _period_accrual(product, index)
                )
                cashflows.append(Cashflow(day, coupon, "periodic_coupon"))
                coupon_cashflow += coupon

        if day == product.maturity_days:
            principal = _principal_cash(product, performance, knocked_in)
            if principal:
                cashflows.append(Cashflow(day, principal, "principal"))
            if (
                product.coupon.rule is CouponRule.CONTINGENT_AT_REDEMPTION
                and (not knocked_in or product.coupon.survives_knock_in)
            ):
                coupon = (
                    product.notional
                    * product.coupon.maturity_rate
                    * product.coupon_maturity_years
                )
                if coupon:
                    cashflows.append(Cashflow(day, coupon, "maturity_coupon"))
                    coupon_cashflow += coupon

    return PathEvaluation(
        cashflows=tuple(cashflows),
        knocked_in=knocked_in,
        knocked_out=knocked_out,
        redemption_day=redemption_day,
        coupon_cashflow=coupon_cashflow,
    )


class VectorizedPayoffState:
    """Streaming payoff state shared by deterministic and MC path evaluation."""

    def __init__(self, product: StructuredNote, n_paths: int, rate: float) -> None:
        self.product = product
        self.rate = rate
        self.present_values = np.zeros(n_paths, dtype=float)
        self.coupon_cashflows = np.zeros(n_paths, dtype=float)
        self.alive = np.ones(n_paths, dtype=bool)
        initially_ki = bool(product.knock_in and product.knock_in.initially_knocked_in)
        self.knocked_in = np.full(n_paths, initially_ki, dtype=bool)
        self.knocked_out = np.zeros(n_paths, dtype=bool)
        self.redemption_days = np.full(n_paths, product.maturity_days, dtype=int)
        self._observations = (
            {day: index for index, day in enumerate(product.autocall.observation_days)}
            if product.autocall
            else {}
        )
        self._payments = {
            day: index for index, day in enumerate(product.coupon.payment_days)
        }

    def _discount(self, day: int) -> float:
        return exp(-self.rate * day / self.product.day_count)

    def _knock_in(self, day: int, performance: np.ndarray) -> None:
        feature = self.product.knock_in
        if feature and day >= feature.monitoring_start_day:
            self.knocked_in |= self.alive & (performance <= feature.barrier_ratio)

    def _knock_out(self, day: int, performance: np.ndarray) -> None:
        feature = self.product.autocall
        if feature is None or day not in self._observations:
            return
        index = self._observations[day]
        mask = self.alive & (performance >= feature.barrier_ratios[index])
        if not np.any(mask):
            return
        principal = (
            self.product.notional if self.product.redemption.include_principal else 0.0
        )
        coupon = (
            self.product.notional
            * feature.coupon_rates[index]
            * (self.product.elapsed_days + day)
            / self.product.day_count
        )
        self.present_values[mask] += (principal + coupon) * self._discount(day)
        self.coupon_cashflows[mask] += coupon
        self.knocked_out[mask] = True
        self.redemption_days[mask] = day
        self.alive[mask] = False

    def _periodic_coupon(self, day: int) -> None:
        if (
            self.product.coupon.rule is not CouponRule.FIXED_PERIODIC
            or day not in self._payments
        ):
            return
        index = self._payments[day]
        mask = self.alive.copy()
        if not self.product.coupon.survives_knock_in:
            mask &= ~self.knocked_in
        coupon = (
            self.product.notional
            * self.product.coupon.rate_schedule[index]
            * _period_accrual(self.product, index)
        )
        self.present_values[mask] += coupon * self._discount(day)
        self.coupon_cashflows[mask] += coupon

    def _maturity(self, performance: np.ndarray) -> None:
        mask = self.alive
        if not np.any(mask):
            return
        ratio = np.ones_like(performance)
        loss = (
            mask
            & self.knocked_in
            & (performance < self.product.redemption.strike_ratio)
        )
        ratio[loss] = np.maximum(
            self.product.redemption.principal_floor,
            1.0
            + self.product.redemption.downside_participation
            * (
                performance[loss] / self.product.redemption.strike_ratio
                - 1.0
            ),
        )
        principal = (
            self.product.notional * ratio
            if self.product.redemption.include_principal
            else np.zeros_like(ratio)
        )
        self.present_values[mask] += principal[mask] * self._discount(
            self.product.maturity_days
        )
        if self.product.coupon.rule is CouponRule.CONTINGENT_AT_REDEMPTION:
            eligible = mask & (
                (~self.knocked_in) | self.product.coupon.survives_knock_in
            )
            coupon = (
                self.product.notional
                * self.product.coupon.maturity_rate
                * self.product.coupon_maturity_years
            )
            self.present_values[eligible] += coupon * self._discount(
                self.product.maturity_days
            )
            self.coupon_cashflows[eligible] += coupon
        self.alive[mask] = False

    def process_day(self, day: int, performance: np.ndarray) -> None:
        if self.product.same_day_priority is SameDayPriority.KNOCK_OUT_FIRST:
            self._knock_out(day, performance)
            self._knock_in(day, performance)
        else:
            self._knock_in(day, performance)
            self._knock_out(day, performance)
        self._periodic_coupon(day)
        if day == self.product.maturity_days:
            self._maturity(performance)

    def result(self) -> BatchEvaluation:
        return BatchEvaluation(
            present_values=self.present_values,
            knocked_in=self.knocked_in,
            knocked_out=self.knocked_out,
            redemption_days=self.redemption_days,
            coupon_cashflows=self.coupon_cashflows,
        )


def evaluate_paths(
    paths: np.ndarray,
    product: StructuredNote,
    *,
    rate: float = 0.0,
) -> BatchEvaluation:
    values = np.asarray(paths, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (
        product.maturity_days + 1,
        product.n_assets,
    ):
        raise ValueError(
            "paths shape must be (n_paths, maturity_days + 1, n_assets)"
        )
    state = VectorizedPayoffState(product, values.shape[0], rate)
    for day in range(1, product.maturity_days + 1):
        state.process_day(day, basket_performance(values[:, day, :], product))
    return state.result()
