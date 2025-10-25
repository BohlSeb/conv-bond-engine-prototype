from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LinearTriangularElements:

    def __init__(self, points: NDArray[np.float64], triangles: NDArray[np.int32], areas: NDArray[np.float64]) -> None:
        x0 = points[triangles[:, 0], 0]
        x1 = points[triangles[:, 1], 0]
        x2 = points[triangles[:, 2], 0]

        y0 = points[triangles[:, 0], 1]
        y1 = points[triangles[:, 1], 1]
        y2 = points[triangles[:, 2], 1]

        self._b = np.stack([y1 - y2, y2 - y0, y0 - y1], axis=1)
        self._c = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1)

        self._points = points
        self._triangles = triangles
        self._areas = areas

    def points(self) -> NDArray[np.float64]:
        return self._points

    def triangles(self) -> NDArray[np.int32]:
        return self._triangles

    def areas(self) -> NDArray[np.float64]:
        return self._areas

    def local_stiffness(self, tri_index: int, const_factor: float = 1.0) -> NDArray[np.float64]:
        b = self._b[tri_index]
        c = self._c[tri_index]
        a = self._areas[tri_index]
        return const_factor * (np.outer(b, b) + np.outer(c, c)) / (4 * a)

    def local_mass(self, tri_index: int, const_factor: float = 1.0) -> NDArray[np.float64]:
        a = self._areas[tri_index]
        return const_factor * a / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
