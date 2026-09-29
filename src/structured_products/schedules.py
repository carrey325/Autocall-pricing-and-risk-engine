"""Schedule construction and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence


def validate_days(days: Sequence[int], maturity_days: int, *, name: str) -> tuple[int, ...]:
    values = tuple(int(day) for day in days)
    if any(day <= 0 or day > maturity_days for day in values):
        raise ValueError(f"{name} days must be in [1, maturity_days]")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} days must be strictly increasing")
    return values


def monthly_observation_days(
    maturity_days: int,
    *,
    day_count: int = 252,
    start_month: int = 1,
    step_months: int = 1,
) -> tuple[int, ...]:
    if maturity_days <= 0 or day_count <= 0:
        raise ValueError("maturity_days and day_count must be positive")
    if start_month <= 0 or step_months <= 0:
        raise ValueError("start_month and step_months must be positive")
    total_months = max(1, round(maturity_days * 12 / day_count))
    result: list[int] = []
    for month in range(start_month, total_months + 1, step_months):
        day = min(maturity_days, round(month * day_count / 12))
        if day > 0 and (not result or day != result[-1]):
            result.append(day)
    if not result:
        result.append(maturity_days)
    return tuple(result)


def quarterly_payment_days(maturity_days: int, *, day_count: int = 252) -> tuple[int, ...]:
    return monthly_observation_days(
        maturity_days, day_count=day_count, start_month=3, step_months=3
    )


def flat_schedule(value: float, count: int) -> tuple[float, ...]:
    if count < 0:
        raise ValueError("count cannot be negative")
    return tuple(float(value) for _ in range(count))


def linear_schedule(start: float, end: float, count: int) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (float(start),)
    step = (end - start) / (count - 1)
    return tuple(float(start + step * index) for index in range(count))
