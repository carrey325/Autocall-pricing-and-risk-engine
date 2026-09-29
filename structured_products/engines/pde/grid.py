"""PDE grid construction."""

from __future__ import annotations

import numpy as np

from ...config import PDEConfig
from ...market import MarketData
from ...products import StructuredNote


def build_spot_grid(
    product: StructuredNote,
    market: MarketData,
    config: PDEConfig,
) -> np.ndarray:
    reference = product.reference_spots[0]
    scales = [reference, market.spots[0], reference * product.redemption.strike_ratio]
    if product.knock_in:
        scales.append(reference * product.knock_in.barrier_ratio)
    if product.autocall:
        scales.extend(reference * value for value in product.autocall.barrier_ratios)
    upper = config.spot_grid_upper
    if upper is None:
        upper = config.upper_spot_multiple * max(scales)
    elif upper <= max(scales):
        raise ValueError("fixed spot grid upper bound must exceed spots and contract levels")
    return np.linspace(0.0, upper, config.spot_grid_points)
