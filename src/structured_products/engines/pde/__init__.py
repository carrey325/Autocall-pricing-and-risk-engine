"""Clean-room one-dimensional Crank-Nicolson pricing."""

from .pricer import price_pde, support_matrix

__all__ = ["price_pde", "support_matrix"]
