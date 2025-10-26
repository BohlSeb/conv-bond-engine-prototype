from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.finite_elements.elements import TriangleElements


class FEMAssembler:
    """
    Assemble global FEM matrices (stiffness, mass) from triangle elements.
    """

    def __init__(self, elements: TriangleElements):
        self._el = elements

    def assemble_stiffness(self, weights: Callable[[float, float], float] | None = None) -> lil_matrix:
        # "list of lists" sparse matrix for efficient construction / modification
        n = self._el.points().shape[0]
        global_stiff = lil_matrix((n, n), dtype=np.float64)
        for i_tri in range(self._el.triangles().shape[0]):
            i_tri_vertices = self._el.triangles()[i_tri]

            weight = 1.0
            if weights is not None:
                # Evaluate coefficient at triangle centroid
                xy = self._el.points()[i_tri_vertices].mean(axis=0)
                weight = weights(xy[0], xy[1])

            local_stiffness = self._el.local_stiffness(i_tri, weight=weight)

            for i_local, i_global in enumerate(i_tri_vertices):
                for j_local, j_global in enumerate(i_tri_vertices):
                    global_stiff[i_global, j_global] += local_stiffness[i_local, j_local]
        return global_stiff

    def assemble_mass(self, weights: Callable[[float, float], float] | None = None) -> lil_matrix:
        n = self._el.points().shape[0]
        global_mass = lil_matrix((n, n), dtype=np.float64)
        for i_tri in range(self._el.triangles().shape[0]):
            i_tri_vertices = self._el.triangles()[i_tri]
            weight = 1.0
            if weights is not None:
                xy = self._el.points()[i_tri_vertices].mean(axis=0)
                weight = weights(xy[0], xy[1])

            local_mass = self._el.local_mass(i_tri, weight=weight)
            for i_local, i_global in enumerate(i_tri_vertices):
                for j_local, j_global in enumerate(i_tri_vertices):
                    global_mass[i_global, j_global] += local_mass[i_local, j_local]
        return global_mass