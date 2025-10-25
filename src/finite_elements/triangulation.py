from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike


class RectangleMesher:

    def __init__(self, x_knots: ArrayLike, y_knots: ArrayLike) -> None:
        n = len(x_knots)
        m = len(y_knots)
        self._points: NDArray[np.float64] = np.empty((n * m, 2), dtype=np.float64)

        # locates for each node up to (n-1, m-1) the two triangles
        # spanning the rectangle that has that node at its lower left

        triangles: list[list[float]] = []
        areas: list[float] = []

        h_x = np.diff(x_knots)
        h_y = np.diff(y_knots)

        for i in range(n):
            for j in range(m):

                p_1 = [x_knots[i], y_knots[j]]

                k = i * m + j
                self._points[k, :] = p_1

                if i < n - 1 and j < m - 1:
                    # counter clock wise orientation
                    right = k + m
                    top = k + 1
                    diag = right + 1
                    triangles.append([k, right, diag])
                    triangles.append([k, diag, top])
                    area = float(h_x[i] * h_y[j] / 2)
                    areas.extend([area, area])


        self._triangles: NDArray[np.int32] = np.array(triangles)
        assert self._triangles.shape == ((n - 1) * (m - 1) * 2, 3)
        self._areas: NDArray[np.float64] = np.array(areas)

    def points(self) -> NDArray[np.float64]:
        return self._points

    # indexes of the 3 vertices of the triangles
    def triangles(self) -> NDArray[np.int32]:
        return self._triangles

    def areas(self) -> NDArray[np.float64]:
        return self._areas

    def plot(self) -> None:
        # Swap x/y for plotting IJ as Cartesian
        x = self._points[:, 1]  # j index → x-axis
        y = self._points[:, 0]  # i index → y-axis

        plt.figure()
        plt.triplot(x, y, self._triangles, color='gray')
        plt.plot(x, y, 'ro')
        plt.gca().set_aspect('equal')
        plt.xlabel('j (column index)')
        plt.ylabel('i (row index)')
        plt.title('Triangular mesh in "ij" coordinates')
        plt.show()





if __name__ == "__main__":
    from matplotlib import pyplot as plt

    _range_x = np.linspace(0, 1, 10)
    _range_y = np.linspace(0, 1, 12)
    x_range = list(_range_x)
    y_range = list(_range_y)

    triangulation = RectangleMesher(x_range, y_range)
    triangulation.plot()

