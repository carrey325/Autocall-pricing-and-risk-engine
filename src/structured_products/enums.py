"""Stable enumerations used by the public domain model."""

from __future__ import annotations

from enum import Enum


class PricingMethod(str, Enum):
    MC = "mc"
    PDE = "pde"


class BasketRule(str, Enum):
    SINGLE = "single"
    WORST_OF = "worst_of"


class CouponRule(str, Enum):
    CONTINGENT_AT_REDEMPTION = "contingent_at_redemption"
    FIXED_PERIODIC = "fixed_periodic"


class SameDayPriority(str, Enum):
    KNOCK_OUT_FIRST = "knock_out_first"
    KNOCK_IN_FIRST = "knock_in_first"
