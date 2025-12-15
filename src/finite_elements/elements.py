from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


# 2D Triangle Elements interface
class TriangleElements:

    def __init__(self, points: NDArray[np.float64], triangles: NDArray[np.int32], areas: NDArray[np.float64]) -> None:
        self._points = points
        self._triangles = triangles
        self._areas = areas

    def points(self) -> NDArray[np.float64]:
        return self._points

    def triangles(self) -> NDArray[np.int32]:
        return self._triangles

    def mass(self, tri_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        # weight: scalar weight for mass matrix
        raise NotImplementedError

    def stiffness(self, tri_index: int, weight: float, diffusion_tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        # assert diffusion_tensor.shape == (2,2)
        # 2x2 diffusion tensor, examples:
        # np.eye(2) => u_xx + u_yy
        # matrix [0.0, 1.0; 1.0, 0.0] => u_xy + u_yx = 2 u_xy
        raise NotImplementedError

    def convection(self, tri_index: int, weights: tuple[float, float]) -> NDArray[np.float64]:
        # weights: tuple of (weight_x, weight_y) for convection in x and y directions
        # Examples:
        # weights = (1.0, 0.0) <=> u_x
        # weights = (0.0, 1.0) <=> u_y
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

        # shape-function gradient coefficients
        self._b = np.stack([y1 - y2, y2 - y0, y0 - y1], axis=1)
        self._c = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1)

    def stiffness(self, tri_index: int, weight: float, diffusion_tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        b = self._b[tri_index]
        c = self._c[tri_index]
        a = self._areas[tri_index]
        g = np.stack([b, c], axis=0)
        m = g.T @ diffusion_tensor @ g  # shape (3,3)
        return weight * m / (4 * a)

    def mass(self, tri_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        a = self._areas[tri_index]
        return weight * a / 12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

    def convection(self, tri_index: int, weights: tuple[float, float]) -> NDArray[np.float64]:
        weight_x, weight_y = weights
        b = self._b[tri_index]
        c = self._c[tri_index]
        return 1 / 6.0 * np.outer(np.ones(3), weight_x * b + weight_y * c)

    def supg(self,
             tri_index: int,
             beta: tuple[float, float],
             kappa_eff: float,
             reaction: float = 0.0) -> NDArray[np.float64]:
        """
        "Streamline upwind Petrov–Galerkin pressure-stabilizing" (SUPG)
        Local term for P1 triangles:

            ∫_T τ (β·∇φ_i)(β·∇φ_j) dA

        using b,c coefficients where ∇φ_i = (b_i, c_i) / (2A).
        """
        bx, by = beta
        area = self._areas[tri_index]
        b_i = self._b[tri_index]
        c_i = self._c[tri_index]

        # d_i = βx*b_i + βy*c_i  (note: beta·∇φ_i = d_i / (2A))
        d = bx * b_i + by * c_i

        beta_norm = float(np.hypot(bx, by))
        if beta_norm < 1e-14:
            return np.zeros((3, 3), dtype=np.float64)

        # streamline element length:
        # h = 2|β| / Σ|β·∇φ_i|
        # with β·∇φ_i = d_i/(2A)  =>  h = 4A|β| / Σ|d_i|
        denominator = float(np.sum(np.abs(d)))
        if denominator < 1e-14:
            return np.zeros((3, 3), dtype=np.float64)
        h = 4.0 * area * beta_norm / denominator

        # tau choice (robust):
        # tau = 1/sqrt((2|β|/h)^2 + (4κ/h^2)^2 + r^2)
        inv_tau_sq = (2.0 * beta_norm / h) ** 2 + (4.0 * kappa_eff / (h * h)) ** 2
        if reaction != 0.0:
            inv_tau_sq += reaction ** 2
        tau = 1.0 / np.sqrt(inv_tau_sq)

        # SUPG matrix: tau/(4A) * d d^T
        return (tau / (4.0 * area)) * np.outer(d, d)
