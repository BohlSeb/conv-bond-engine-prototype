from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


# 2D Triangle Elements interface
class TriangleElements:

    def __init__(self,points: NDArray[np.float64], triangles: NDArray[np.int32], areas: NDArray[np.float64]) -> None:
        self._points = points
        self._triangles = triangles
        self._areas = areas

    def points(self) -> NDArray[np.float64]:
        return self._points

    def triangles(self) -> NDArray[np.int32]:
        return self._triangles

    def stiffness(self, tri_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        raise NotImplementedError

    def mass(self, tri_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        raise NotImplementedError

    def convection(self, tri_index: int, weight_x: float = 1.0, weight_y: float = 1.0) -> NDArray[np.float64]:
        raise NotImplementedError



class LinearTriElements(TriangleElements):

    def __init__(self, points: NDArray[np.float64], triangles: NDArray[np.int32], areas: NDArray[np.float64]) -> None:
        super().__init__(points, triangles, areas)
        x0 = points[triangles[:, 0], 0]
        x1 = points[triangles[:, 1], 0]
        x2 = points[triangles[:, 2], 0]

        y0 = points[triangles[:, 0], 1]
        y1 = points[triangles[:, 1], 1]
        y2 = points[triangles[:, 2], 1]

        # "shape"-function gradient coefficients
        self._b = np.stack([y1 - y2, y2 - y0, y0 - y1], axis=1)
        self._c = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1)

    def stiffness(self, tri_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        b = self._b[tri_index]
        c = self._c[tri_index]
        a = self._areas[tri_index]
        return weight / (4 * a) * np.outer(b, b) + np.outer(c, c)

    def mass(self, tri_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        a = self._areas[tri_index]
        return weight * a / 12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

    def convection(self, tri_index: int, weight_x: float = 1.0, weight_y: float = 1.0) -> NDArray[np.float64]:
        b = self._b[tri_index]
        c = self._c[tri_index]
        return 1 / 6.0 * np.outer(np.ones(3), weight_x * b + weight_y * c)
