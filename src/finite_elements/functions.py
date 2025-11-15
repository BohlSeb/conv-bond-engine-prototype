from __future__ import annotations

import numpy as np

from typing import Callable, TYPE_CHECKING

# Typical code in danger of overengineering

if TYPE_CHECKING:
    from numpy.typing import NDArray

    # 2D scalar field input function type
    ScalarFunctionT = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


class Function:

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        # points have shape (n_points, n_dimensions), e.g. (n, 2) for 2D
        raise NotImplementedError


class Scalar(Function):  # Cachable 2D scalar field

    def __init__(self, fct: ScalarFunctionT):
        self._fct = fct

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._fct(points[:, 0], points[:, 1])


class Constant(Function):

    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._value * np.ones(points.shape[0], dtype=np.float64)
