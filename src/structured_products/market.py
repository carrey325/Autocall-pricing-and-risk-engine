"""Validated flat market inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .products import StructuredNote


@dataclass(frozen=True)
class MarketData:
    spots: tuple[float, ...]
    rate: float
    dividend_yields: tuple[float, ...]
    volatilities: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...] = ((1.0,),)

    def __post_init__(self) -> None:
        spots = tuple(float(value) for value in self.spots)
        dividends = tuple(float(value) for value in self.dividend_yields)
        volatilities = tuple(float(value) for value in self.volatilities)
        correlation = tuple(tuple(float(value) for value in row) for row in self.correlation)
        object.__setattr__(self, "spots", spots)
        object.__setattr__(self, "dividend_yields", dividends)
        object.__setattr__(self, "volatilities", volatilities)
        object.__setattr__(self, "correlation", correlation)
        size = len(spots)
        if size == 0 or any(value <= 0 for value in spots):
            raise ValueError("market spots must be positive")
        if len(dividends) != size or len(volatilities) != size:
            raise ValueError("spot, dividend, and volatility dimensions must match")
        if any(value < 0 for value in volatilities):
            raise ValueError("volatilities cannot be negative")
        matrix = np.asarray(correlation, dtype=float)
        if matrix.shape != (size, size):
            raise ValueError("correlation matrix dimension does not match assets")
        if not np.allclose(matrix, matrix.T, atol=1.0e-12):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(matrix), 1.0, atol=1.0e-12):
            raise ValueError("correlation matrix must have unit diagonal")
        if np.min(np.linalg.eigvalsh(matrix)) < -1.0e-10:
            raise ValueError("correlation matrix must be positive semidefinite")

    def validate_for(self, product: StructuredNote) -> None:
        if len(self.spots) != product.n_assets:
            raise ValueError("product and market asset dimensions do not match")

    def correlation_factor(self) -> np.ndarray:
        matrix = np.asarray(self.correlation, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues))
