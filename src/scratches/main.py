from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def initialize_linear(x_ones: NDArray, x_twos: NDArray, x_1_0: float = 0.0, x_2_0: float = 0.0, m_1: float = 1.0,
                      m_2: float = 1.0) -> NDArray:
    return x_1_0 + m_1 * x_ones + x_2_0 + m_2 * x_twos


def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


if __name__ == "__main__":
    _times = np.linspace(0, 1, 2)
    _x_ones = np.linspace(0, 1, 20)
    _x_twos = np.linspace(0, 1, 10)

    _X, _Y = np.meshgrid(_x_ones, _x_twos)

    print(_X.shape)
    print(_Y.shape)

    # _test = initialize_linear(_X, _Y)

    # print(_test)

    # print(test)
