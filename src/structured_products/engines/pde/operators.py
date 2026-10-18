"""Sparse Black-Scholes finite-difference operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, diags
from scipy.sparse.linalg import SuperLU, splu


@dataclass
class CrankNicolsonStepper:
    left_factor: SuperLU
    right: csc_matrix

    def step(self, values: np.ndarray) -> np.ndarray:
        rhs = self.right @ values
        rhs[0] = 0.0
        rhs[-1] = 0.0
        return self.left_factor.solve(rhs)


def build_stepper(
    grid: np.ndarray,
    *,
    rate: float,
    dividend_yield: float,
    volatility: float,
    dt: float,
    theta: float,
) -> CrankNicolsonStepper:
    size = len(grid)
    spacing = grid[1] - grid[0]
    left = lil_matrix((size, size), dtype=float)
    right = lil_matrix((size, size), dtype=float)

    # S=0 is absorbing (zero delta); the distant upper edge uses zero gamma.
    left[0, 0] = 1.0
    left[0, 1] = -1.0
    left[-1, -3] = 1.0
    left[-1, -2] = -2.0
    left[-1, -1] = 1.0

    for index in range(1, size - 1):
        spot = grid[index]
        variance = 0.5 * volatility**2 * spot**2 / spacing**2
        drift = (rate - dividend_yield) * spot / (2.0 * spacing)
        lower = variance - drift
        diagonal = -2.0 * variance - rate
        upper = variance + drift

        left[index, index - 1] = -theta * dt * lower
        left[index, index] = 1.0 - theta * dt * diagonal
        left[index, index + 1] = -theta * dt * upper
        right[index, index - 1] = (1.0 - theta) * dt * lower
        right[index, index] = 1.0 + (1.0 - theta) * dt * diagonal
        right[index, index + 1] = (1.0 - theta) * dt * upper

    left_csc = csc_matrix(left)
    return CrankNicolsonStepper(splu(left_csc), csc_matrix(right))


def build_variable_stepper(grid, *, rate, carry, volatility, dt, theta):
    """Vectorized interior operator; identical boundary equations to the flat solver."""
    size = len(grid)
    spacing = grid[1]-grid[0]
    spots = grid[1:-1]
    variance = .5*np.asarray(volatility)**2*spots**2/spacing**2
    drift = carry*spots/(2*spacing)
    lower, diagonal, upper = variance-drift, -2*variance-rate, variance+drift

    lo, diag, hi = np.zeros(size-1), np.zeros(size), np.zeros(size-1)
    weight = -theta*dt
    lo[:-1], diag[1:-1], hi[1:] = weight*lower, 1+weight*diagonal, weight*upper
    diag[0], hi[0] = 1., -1.
    lo[-1], diag[-1] = -2., 1.
    edge = np.zeros(size-2)
    edge[-1] = 1.
    left = diags((edge, lo, diag, hi), (-2, -1, 0, 1), format='csc')
    # Identity belongs only to interior rows; right boundary rows remain zero.
    lo, diag, hi = np.zeros(size-1), np.zeros(size), np.zeros(size-1)
    weight = (1-theta)*dt
    lo[:-1], diag[1:-1], hi[1:] = weight*lower, 1+weight*diagonal, weight*upper
    right = diags((lo, diag, hi), (-1, 0, 1), format='csc')
    return CrankNicolsonStepper(splu(left), right)
