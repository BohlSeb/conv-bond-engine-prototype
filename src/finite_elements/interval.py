from __future__ import annotations

import math

from typing import TypeVar

import numpy as np
import QuantLib as ql


# Sketchy implementation of crude concentrating bisection around the provided knots on a 1D interval


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


KeyT = TypeVar('KeyT')


class IntervalsBisectFill:

    def __init__(self, knots: dict[int, float], min_step: float, max_step: float):
        assert len(knots) > 1, 'At least two grid points are needed'

        # Knot position indices within the final grid
        self._indexes: dict[int, int] = {}

        j = 0
        keys = [key for key, _ in sorted(knots.items(), key=lambda x: x[1])]
        knots = [knots[key] for key in keys]
        self._indexes[keys[0]] = j

        grid = [knots[0]]
        for i in range(len(knots) - 1):
            a, b = knots[i], knots[i + 1]

            bisection = bisect_outer_intervals(a, b, min_step, max_step)[1:-1]
            if bisection:
                j += len(bisection)
                grid.extend(bisection)

            j += 1
            grid.append(b)
            self._indexes[keys[i + 1]] = j
        self._grid = sorted(grid)

    def indexes(self) -> dict[int, int]:
        return self._indexes

    def grid(self) -> list[float]:
        return self._grid


class ConcentratingInterval:

    def __init__(self,
                 start: float,
                 end: float,
                 size: int,
                 point: float,
                 beta: float,
                 require_point: bool = False) -> None:
        intervals = ql.Concentrating1dMesher(start, end, size, (point, beta), require_point)
        self._grid = np.array(intervals.locations())
        if not require_point:
            self._indexes = None
        else:
            i = None
            for i in range(len(self._grid)):
                if math.isclose(self._grid[i], point):
                    self._indexes = {i: i}
                    break
            if i is None:
                raise Exception('Point not found in grid')

    def grid(self) -> np.ndarray:
        return self._grid

    def indexes(self) -> dict[int, int]:
        if self._indexes is not None:
            return self._indexes
        else:
            raise Exception('No indexes available, grid was not required to contain a specific point')


if __name__ == '__main__':
    interval = ConcentratingInterval(0.0, 10.0, 9, 5.0, 0.5)
    print(interval.grid())
