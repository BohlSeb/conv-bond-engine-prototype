from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from finite_elements.constants import EPSILON, EPSILON_BIG

if TYPE_CHECKING:
    from numpy.typing import NDArray


# 1D Linear Interval Finite Elements
class LinearIntervals:
    """
    P1 elements on segments [x0,x1].

    On an element of length h:
      M_e = weight * h/6 * [[2,1],[1,2]]
      K_e = weight * 1/h * [[ 1,-1],[-1, 1]]
      C_e for beta*u_x (Galerkin) with test φ_i:
        C_ij = ∫ φ_i * beta * dφ_j/dx dx
             = beta/2 * [[-1, 1],[-1, 1]]
    """

    def __init__(self, points: NDArray[np.float64]) -> None:
        if points.ndim != 1:
            raise ValueError('Nodes must be 1D')
        if points.shape[0] < 2:
            raise ValueError('Need at least 2 nodes')
        self._h = np.diff(points)
        if not np.all(self._h >= EPSILON_BIG):
            raise ValueError(f'Interval length smaller than {EPSILON_BIG} encountered')
        self._points = points
        self._intervals = np.stack((np.arange(self._points.size - 1, dtype=np.int32),
                                    np.arange(1, self._points.size, dtype=np.int32)),
                                   axis=1)

        self._stiff_m = np.array([[1.0, -1.0],
                                  [-1.0, 1.0]], dtype=np.float64)
        self._mass_m = 1 / 6 * np.array([[2.0, 1.0],
                                         [1.0, 2.0]], dtype=np.float64)
        self._conv_m = 1 / 2 * np.array([[-1.0, 1.0],
                                         [-1.0, 1.0]], dtype=np.float64)

    def points(self) -> NDArray[np.float64]:
        return self._points

    def intervals(self) -> NDArray[np.int32]:
        return self._intervals

    def stiffness(self, elem_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        h = self._h[elem_index]
        return (weight / h) * self._stiff_m

    def mass(self, elem_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        h = self._h[elem_index]
        return weight * h * self._mass_m

    def convection(self, _: int, weight: float) -> NDArray[np.float64]:
        return weight * self._conv_m


# 2D Linear Triangle Finite Elements
class LinearTriangles:

    def __init__(self, points: NDArray[np.float64], triangles: NDArray[np.int32], areas: NDArray[np.float64]) -> None:
        if points.ndim != 2:
            raise ValueError('Nodes must be 2D')
        if points.shape[0] < 2:
            raise ValueError('Need at least 2 nodes')
        if not np.all(areas >= EPSILON_BIG):
            raise ValueError(f'Triangle area smaller than {EPSILON_BIG} encountered')
        self._points = points
        self._triangles = triangles
        self._areas = areas

        x0 = points[triangles[:, 0], 0]
        x1 = points[triangles[:, 1], 0]
        x2 = points[triangles[:, 2], 0]

        y0 = points[triangles[:, 0], 1]
        y1 = points[triangles[:, 1], 1]
        y2 = points[triangles[:, 2], 1]

        # shape-function gradient coefficients
        self._b = np.stack([y1 - y2, y2 - y0, y0 - y1], axis=1)
        self._c = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1)

        # mass matrix doesn't depend on gradients
        self._mass = 1 / 12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

    def points(self) -> NDArray[np.float64]:
        return self._points

    def triangles(self) -> NDArray[np.int32]:
        return self._triangles

    def stiffness(self, elem_index: int, weight: float, diffusion_tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        # 2x2 diffusion tensor, examples:
        # np.eye(2) => u_xx + u_yy
        # matrix [0.0, 1.0; 1.0, 0.0] => u_xy + u_yx = 2 u_xy
        b = self._b[elem_index]
        c = self._c[elem_index]
        a = self._areas[elem_index]
        g = np.stack([b, c], axis=0)
        m = g.T @ diffusion_tensor @ g  # shape (3,3)
        return weight * m / (4 * a)

    def convection(self, elem_index: int, weights: tuple[float, float]) -> NDArray[np.float64]:
        # weights: tuple of (weight_x, weight_y) for convection in x and y directions
        # Examples:
        # weights = (1.0, 0.0) <=> u_x
        # weights = (0.0, 1.0) <=> u_y
        w_x, w_y = weights
        b = self._b[elem_index]
        c = self._c[elem_index]
        return 1 / 6.0 * np.outer(np.ones(3), w_x * b + w_y * c)

    def mass(self, elem_index: int, weight: float = 1.0) -> NDArray[np.float64]:
        a = self._areas[elem_index]
        return weight * a * self._mass

    def supg(self,
             elem_index: int,
             beta: tuple[float, float],
             kappa_eff: float,
             reaction: float = 0.0) -> NDArray[np.float64]:
        """
        Streamline Upwind Petrov–Galerkin (SUPG) convection stabilization term.
        Local term for P1 triangles:

            ∫_T τ (β·∇φ_i)(β·∇φ_j) dA

        using b,c coefficients where ∇φ_i = (b_i, c_i) / (2A).
        """
        bx, by = beta
        area = self._areas[elem_index]
        b_i = self._b[elem_index]
        c_i = self._c[elem_index]

        # d_i = βx*b_i + βy*c_i  (note: beta·∇φ_i = d_i / (2A))
        d = bx * b_i + by * c_i

        beta_norm = float(np.hypot(bx, by))
        if beta_norm < EPSILON:
            return np.zeros((3, 3), dtype=np.float64)

        # streamline element length:
        # h = 2|β| / Σ|β·∇φ_i|
        # with β·∇φ_i = d_i/(2A)  =>  h = 4A|β| / Σ|d_i|
        denominator = float(np.sum(np.abs(d)))
        if denominator < EPSILON:
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
