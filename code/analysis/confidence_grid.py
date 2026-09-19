"""Validate a discrete inverted-test confidence set before reporting interval bounds."""

import numpy as np


def checked_log_bounds(grid, accepted):
    grid = np.asarray(grid, dtype=float)
    accepted = np.asarray(accepted, dtype=float)
    if len(accepted) == 0:
        raise ValueError("Exact-test inversion accepted no grid points")
    spacing = grid[1] - grid[0]
    if np.isclose(accepted[0], grid[0], atol=abs(spacing) * 1e-6, rtol=0) or np.isclose(
        accepted[-1], grid[-1], atol=abs(spacing) * 1e-6, rtol=0
    ):
        raise ValueError(
            "Confidence set touches the grid boundary. Expand the range at the same spacing and document the change."
        )
    if np.any(np.diff(accepted) > abs(spacing) * 1.001):
        raise ValueError(
            "Disconnected confidence set. Report its components, not an interval spanning an unaccepted gap."
        )
    return float(np.exp(accepted[0])), float(np.exp(accepted[-1]))
