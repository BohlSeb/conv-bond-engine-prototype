from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.spatial import Delaunay


class Mesh2D:

    def points(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def triangles(self) -> NDArray[np.int32]:
        raise NotImplementedError

    def areas(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def plot(self) -> None:
        from matplotlib import pyplot as plt
        # Swap x/y for plotting IJ as Cartesian
        x = self.points()[:, 1]  # j index → x-axis
        y = self.points()[:, 0]  # i index → y-axis

        plt.figure()
        plt.triplot(x, y, self.triangles(), color='gray')
        plt.plot(x, y, 'ro')
        plt.gca().set_aspect('equal')
        plt.xlabel('j (column index)')
        plt.ylabel('i (row index)')
        plt.title('Triangular mesh in "ij" coordinates')
        plt.show()


class CrudeMesh2D(Mesh2D):

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

    def triangles(self) -> NDArray[np.int32]:
        return self._triangles

    def areas(self) -> NDArray[np.float64]:
        return self._areas


class DelaunayMesh2D(Mesh2D):

    def __init__(self, x_knots: ArrayLike, y_knots: ArrayLike) -> None:

        n = len(x_knots)
        m = len(y_knots)
        points: NDArray[np.float64] = np.empty((n * m, 2), dtype=np.float64)

        for i in range(n):
            for j in range(m):
                k = i * m + j
                points[k, :] = [x_knots[i], y_knots[j]]

        delaunay = Delaunay(points)
        self._points = points
        self._triangles = delaunay.simplices

        areas: list[float] = []
        for tri in self._triangles:
            x0, y0 = points[tri[0]]
            x1, y1 = points[tri[1]]
            x2, y2 = points[tri[2]]
            area = abs((x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) / 2.0)
            areas.append(area)

        self._areas = np.array(areas)

    def points(self) -> NDArray[np.float64]:
        return self._points

    def triangles(self) -> NDArray[np.int32]:
        return self._triangles

    def areas(self) -> NDArray[np.float64]:
        return self._areas


if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    #
    # _range_x = np.linspace(0, 1, 10)
    # _range_y = np.linspace(0, 1, 12)
    # x_range = list(_range_x)
    # y_range = list(_range_y)
    #
    # triangulation = RectangleMesher(x_range, y_range)
    # triangulation.plot()

    from interval import ConcentratingInterval
    kappa = 0.1
    _x1, _x2 = 0.0, 1.0
    _y1, _y2 = 0.0, 1.0
    _x = 1.0 - kappa
    _beta = 2 * kappa
    _size = 10

    x_range_c = ConcentratingInterval(_x1, _x2, _size, _x,  _beta, False).grid()
    y_range_c = np.linspace(_y1, _y2, _size).tolist()

    mesh = DelaunayMesh2D(x_range_c, y_range_c)
    mesh.plot()

    # print(x_range_c.grid())

