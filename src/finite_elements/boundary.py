from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from typing import TYPE_CHECKING

from finite_elements.constants import EPSILON_TOL

if TYPE_CHECKING:
    from scipy.sparse import lil_matrix
    from finite_elements.functions import Function


class RectangleHelper:

    def __init__(self, points: NDArray[np.float64]) -> None:
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        x_min_idx = np.where(np.abs(points[:, 0] - x_min) < EPSILON_TOL)[0]
        x_max_idx = np.where(np.abs(points[:, 0] - x_max) < EPSILON_TOL)[0]
        y_min_idx = np.where(np.abs(points[:, 1] - y_min) < EPSILON_TOL)[0]
        y_max_idx = np.where(np.abs(points[:, 1] - y_max) < EPSILON_TOL)[0]

        self._x_min = x_min_idx[np.argsort(points[x_min_idx, 1])]
        self._x_max = x_max_idx[np.argsort(points[x_max_idx, 1])]
        self._y_min = y_min_idx[np.argsort(points[y_min_idx, 0])]
        self._y_max = y_max_idx[np.argsort(points[y_max_idx, 0])]

        self._boundary = np.unique(np.concatenate([self._x_min, self._x_max, self._y_min, self._y_max]))

    def boundary(self) -> NDArray[np.int64]:
        return self._boundary

    def x_min(self) -> NDArray[np.int64]:
        return self._x_min

    def x_max(self) -> NDArray[np.int64]:
        return self._x_max

    def y_min(self) -> NDArray[np.int64]:
        return self._y_min

    def y_max(self) -> NDArray[np.int64]:
        return self._y_max


class ConstBoundaryBC:

    def __init__(self,
                 where: NDArray[np.int64],
                 points: NDArray[np.float64],
                 condition: Function) -> None:
        if where.shape[0] < 1:
            raise ValueError('Boundary condition requires at least one point')
        self._where = where.astype(np.intp, copy=False)
        self._g = condition(points[self._where])


# Dirichlet condition

DirichletDataT = tuple[NDArray[np.int64], NDArray[np.float64]]


class ConstDirichletBC(ConstBoundaryBC):

    def data(self) -> DirichletDataT:
        return self._where, self._g

    # Warning: assumes that self._where is coming from rectangle helper segments
    def as_data_1d(self, where: str) -> list[DirichletDataT]:
        assert where in ('left', 'right')
        idx = 0 if where == 'left' else self._g.size
        return [(np.array([idx]), np.array([g])) for g in self._g]


def merge_dirichlet_last_wins(data: list[DirichletDataT], debug_overwrite: bool = True) -> DirichletDataT:
    if not data:
        # Return empty arrays with correct dtypes and shapes
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    idx_all = np.concatenate([np.asarray(idx, dtype=np.int64) for idx, _ in data])
    g_all = np.concatenate([np.asarray(g, dtype=np.float64) for _, g in data])
    # reverse arrays so "last wins" becomes "first occurrence"
    idx_rev = idx_all[::-1]
    g_rev = g_all[::-1]

    # unique keeps *first* occurrence → corresponds to last in original order
    uniq_idx, uniq_pos = np.unique(idx_rev, return_index=True)
    # recover corresponding g values
    uniq_g = g_rev[uniq_pos]

    if uniq_idx.size < idx_all.size and debug_overwrite:
        print(f'[Dirichlet merge] {idx_all.size - uniq_idx.size} overlaps resolved')

    order = np.argsort(uniq_idx)
    return uniq_idx[order], uniq_g[order]


def apply_dirichlet_sparse(
        left_h_side: sp.spmatrix,
        right_h_side: np.ndarray,
        data: DirichletDataT,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Apply Dirichlet constraints u[where] = values by elimination.
    Steps:
      1) b <- b - A[:,where] @ values  (write constrained DOF contributions to b)
      2) zero columns 'where' (removes constrained DOFs from all other equations A)
      3) overwrite rows 'where' as identity and set b[where] = values
    """
    where = np.asarray(data[0], dtype=np.int64)
    values = np.asarray(data[1], dtype=np.float64)

    lhs = left_h_side.tocsr(copy=True)
    rhs = np.asarray(right_h_side, dtype=np.float64).copy()

    if where.size == 0:
        return lhs, rhs

    rhs -= lhs[:, where] @ values

    lhs_c = lhs.tocsc(copy=False)
    for k in where:
        lhs_c.data[lhs_c.indptr[k]:lhs_c.indptr[k + 1]] = 0.0
    lhs = lhs_c.tocsr(copy=False)

    lhs_l = lhs.tolil(copy=False)
    for k in where:
        lhs_l.rows[k] = [k]
        lhs_l.data[k] = [1.0]

    # watch out for "where" containing duplicates (e.g. when two conditions constrain the same coordinate)
    # in this case, use merge_dirichlet function above before passing data to this function
    rhs[where] = values

    lhs = lhs_l.tocsr(copy=False)
    lhs[where, where] = 1.0

    return lhs, rhs


# This condition can be applied only for a boundary segment (left, right, top, bottom) # todo: assert this
class ConstNeumannBCBase(ConstBoundaryBC):

    def __init__(self,
                 where: NDArray[np.int64],
                 points: NDArray[np.float64],
                 condition: Function,
                 integration_mode: str) -> None:
        super().__init__(where, points, condition)
        if where.shape[0] < 2:
            raise ValueError('Neumann type boundary condition requires at least two points')
        if integration_mode not in ('edge', 'lumped'):
            raise ValueError(f'Unknown integration mode {integration_mode} '
                             f'for Neumann type boundary condition, use "edge" or "lumped"')
        self._lengths = np.linalg.norm(np.diff(points[self._where], axis=0), axis=1)
        self._lumped = integration_mode == 'lumped'
        if self._lumped:
            self._weights: NDArray[np.float64] | None = self._edge_weights_lumped(self._lengths)
        else:
            self._weights = None

    def _assemble_contribution_rhs(self, size: int) -> NDArray[np.float64]:
        b = np.zeros(size, dtype=np.float64)
        if self._lumped:
            np.add.at(b, self._where, self._g * self._weights)
        else:
            g0 = self._g[:-1]
            g1 = self._g[1:]
            i0 = self._lengths * (2 * g0 + g1) / 6.0
            i1 = self._lengths * (2 * g1 + g0) / 6.0
            i_idx = self._where[:-1]
            j_idx = self._where[1:]
            np.add.at(b, i_idx, i0)
            np.add.at(b, j_idx, i1)
        return b

    @staticmethod
    def _edge_weights_lumped(lengths: NDArray[np.float64]) -> NDArray[np.float64]:
        weights = np.empty(lengths.shape[0] + 1, dtype=np.float64)
        weights[0] = lengths[0] / 2
        weights[-1] = lengths[-1] / 2
        if lengths.shape[0] > 1:
            weights[1:-1] = (lengths[:-1] + lengths[1:]) / 2
        return weights


class ConstNeumannBC(ConstNeumannBCBase):

    def __init__(self,
                 where: NDArray[np.int64],
                 points: NDArray[np.float64],
                 condition: Function,
                 integration_mode: str = 'edge') -> None:
        super().__init__(where, points, condition, integration_mode)

    # modifies input
    def apply(self, rhs: NDArray[np.float64]) -> None:
        rhs += self._assemble_contribution_rhs(rhs.shape[0])


class ConstRobinBC(ConstNeumannBCBase):

    def __init__(self,
                 where: NDArray[np.int64],
                 points: NDArray[np.float64],
                 condition: Function,  # g(x) in alpha*u + beta*du/dn = g
                 dirichlet_alpha: Function,  # alpha(x)
                 neumann_beta: Function,  # beta(x)
                 integration_mode: str = 'edge') -> None:
        super().__init__(where, points, condition, integration_mode)
        self._alpha = dirichlet_alpha(points[self._where]).astype(np.float64, copy=False)
        beta = neumann_beta(points[self._where]).astype(np.float64, copy=False)

        # build effective (gamma, h) used by canonical Robin du/dn + gamma*u = h
        # gamma = alpha / beta
        # h = g / beta
        beta_small = np.abs(beta) <= EPSILON_TOL
        both_small = beta_small & (np.abs(self._alpha) <= EPSILON_TOL)
        if np.any(both_small):
            raise ValueError('Robin BC with alpha=beta=0 encountered (ill-posed)')
        # todo: gpt doesn't want to change _g and _alpha in place ("future proof") ... keep in mind
        self._g[~beta_small] /= beta[~beta_small]
        self._alpha[~beta_small] /= beta[~beta_small]

    # modifies input
    def apply(self, matrix: lil_matrix[np.float64], rhs: NDArray[np.float64]) -> None:
        rhs += self._assemble_contribution_rhs(rhs.shape[0])
        matrix += self._assemble_contribution_lhs(matrix.shape[0]).tolil()

    def _assemble_contribution_lhs(self, size: int) -> sp.coo_matrix:
        if self._lumped:
            rows = self._where.astype(np.intp, copy=False)
            cols = self._where.astype(np.intp, copy=False)
            data = self._alpha * self._weights
        else:
            i_idx = self._where[:-1]
            j_idx = self._where[1:]
            alpha_avg = 0.5 * (self._alpha[:-1] + self._alpha[1:])
            coeffs = self._lengths * alpha_avg / 6.0
            rows = np.concatenate([i_idx, j_idx, i_idx, j_idx])
            cols = np.concatenate([i_idx, j_idx, j_idx, i_idx])
            data = np.concatenate([2 * coeffs, 2 * coeffs, coeffs, coeffs])
        return sp.coo_matrix((data, (rows, cols)), shape=(size, size))
