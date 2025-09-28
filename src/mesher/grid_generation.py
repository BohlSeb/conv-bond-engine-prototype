from __future__ import annotations

import math
from typing import TypeVar, TYPE_CHECKING

import numpy as np
import QuantLib as ql


def bisect_direction_impl(grid: list[float], a: float, b: float, left: bool) -> tuple[float, float]:
    mid = (a + b) / 2
    if mid in grid:
        raise Exception('Should not happen')
    grid.append(mid)
    if left:
        return a, mid
    return mid, b


def bisect_outer_intervals(a: float, b: float, min_step: float, max_step: float) -> list[float]:
    if b - a <= min_step:
        return [a, b]

    n_start = math.ceil(math.log((b - a) / max_step, 2))
    if n_start > 1:
        grid = np.linspace(a, b, 2 ** n_start).tolist()
    else:
        _mid = (a + b) / 2
        grid = [a, _mid, b]
    left_interval = (grid[0], grid[1])
    right_interval = (grid[-2], grid[-1])

    for (_a, _b), is_left in zip([left_interval, right_interval], [True, False]):
        intervals = [(_a, _b)]
        while intervals:
            new_intervals = []
            for a, b in intervals:
                length = b - a
                if length > 2 * min_step:
                    new_a, new_b = bisect_direction_impl(grid, a, b, is_left)
                    new_intervals.append((new_a, new_b))
            intervals = new_intervals

    return sorted(grid)


KnotKeyT = TypeVar('KnotKeyT')


class BisectingGridFiller:

    def __init__(self, knots: dict[KnotKeyT, float], min_step: float, max_step: float):
        assert len(knots) > 1, 'At least two grid points are needed'

        # Knot position indices in the final grid
        self._indexes: dict[KnotKeyT, int] = {}

        j = 0
        keys = [key for key, _ in sorted(knots.items(), key=lambda x: x[1])]
        knots = [knots[key] for key in keys]

        grid = [knots[0]]
        for i in range(len(knots) - 1):
            a, b = knots[i], knots[i + 1]

            bisection = bisect_outer_intervals(a, b, min_step, max_step)[1:-1]
            if bisection:
                j += len(bisection)
                grid.extend(bisection)

            j += 1
            grid.append(b)
            self._indexes[keys[i]] = j
        self._grid = sorted(grid)

    def indexes(self) -> dict[KnotKeyT, int]:
        return self._indexes

    def grid(self) -> list[float]:
        return self._grid


if __name__ == "__main__":
    # times = [0.0, 0.5, 2.0]
    # avg_step = 1
    # max_step = 1 / 6
    # alpha = 0.001
    #
    # test = MinDistanceMesher(times, avg_step, max_step, alpha)
    # print(test.grid())
    #
    from matplotlib import pyplot as plt

    # t_1, t_2 = 0.0, 0.5
    # test = bisect_outer_intervals(t_1, t_2, min_step=0.5 / 365, max_step=0.1)
    # print(test)

    _times = {0: 0.0, 1: 1.0}
    gen = BisectingGridFiller(_times, min_step=0.01, max_step=0.2)
    y = [1.0] * len(gen.grid())
    print(gen.grid())

    plt.plot(gen.grid(), y, '|')
    plt.grid(True)
    plt.show()
