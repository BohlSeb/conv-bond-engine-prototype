from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scipy.sparse as sp


def step_theta(
        time_mass: sp.spmatrix,
        time_step: float,
        space_lhs: sp.spmatrix,
        w_previous: np.ndarray,
        theta: float
) -> tuple[sp.spmatrix, np.ndarray]:
    """
    Build (LHS, RHS) for:
      (M/dt + theta A) w_{n+1} = (M/dt - (1-theta) A) w_n
    """
    if time_step <= 0:
        raise ValueError('time_step must be > 0')
    if not (0.0 <= theta <= 1.0):
        raise ValueError('theta must be in [0,1]')

    t_m = (1.0 / time_step) * time_mass.tocsr(copy=False)
    lhs_temp = space_lhs.tocsr(copy=False)
    lhs = t_m + theta * lhs_temp
    rhs = (t_m - (1.0 - theta) * lhs_temp) @ w_previous
    return lhs, rhs
