from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import lil_matrix


class RectangleHelper:

    def __init__(self, points: NDArray[np.float64]) -> None:
        tol = 1e-12
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        self._x_min = np.where(np.abs(points[:, 0] - x_min) < tol)[0]
        self._x_max = np.where(np.abs(points[:, 0] - x_max) < tol)[0]
        self._y_min = np.where(np.abs(points[:, 1] - y_min) < tol)[0]
        self._y_max = np.where(np.abs(points[:, 1] - y_max) < tol)[0]

        self._boundary = np.where(
            (np.abs(points[:, 0] - x_min) < tol) |
            (np.abs(points[:, 0] - x_max) < tol) |
            (np.abs(points[:, 1] - y_min) < tol) |
            (np.abs(points[:, 1] - y_max) < tol)
        )[0]

    def boundary(self) -> NDArray[np.int32]:
        return self._boundary

    def x_min(self) -> NDArray[np.int32]:
        return self._x_min

    def x_max(self) -> NDArray[np.int32]:
        return self._x_max

    def y_min(self) -> NDArray[np.int32]:
        return self._y_min

    def y_max(self) -> NDArray[np.int32]:
        return self._y_max


class TimeIndependentCondition:

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError


class XYCallBack(TimeIndependentCondition):

    def __init__(self, callback: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]) -> None:
        self._callback = callback

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._callback(points[:, 0], points[:, 1])


class ConstDirichletCondition:

    def __init__(self,
                 where: NDArray[np.int32],
                 condition: TimeIndependentCondition,
                 points: NDArray[np.float64]) -> None:
        self._where = where
        self._condition = condition
        self._values = self._condition(points[self._where])

    # modifies input
    def apply(self, matrix: lil_matrix[np.float64], rhs: NDArray[np.float64], zero_diagonal: bool = False) -> None:
        for idx, value in zip(self._where, self._values):
            matrix.rows[idx] = [idx]
            matrix.data[idx] = [1.0]
            rhs[idx] = value
            if zero_diagonal:
                for j in range(matrix.shape[0]):
                    if j != idx:
                        matrix[j, idx] = 0.0


class ConstNeumannCondition:

    def __init__(self,
                 where: NDArray[np.int32],
                 condition: TimeIndependentCondition,
                 points: NDArray[np.float64]) -> None:
        self._where = where
        self._condition = condition
        self._values = self._condition(points[self._where]) * self._approx_edge_lengths(points)

    # modifies input
    def apply(self, rhs: NDArray[np.float64]) -> None:
        rhs[self._where] += self._values

    def _approx_edge_lengths(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        pts = points[self._where]
        deltas = np.diff(pts, axis=0)
        dists = np.sqrt(np.sum(deltas ** 2, axis=1))
        weights = np.empty_like(pts[:, 0])
        weights[0] = dists[0] / 2
        weights[-1] = dists[-1] / 2
        if pts.shape[0] > 1:
            weights[1:-1] = (dists[:-1] + dists[1:]) / 2
        return weights
