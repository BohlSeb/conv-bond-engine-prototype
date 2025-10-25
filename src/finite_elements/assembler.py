from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from src.finite_elements.elements import LinearTriangularElements


class FEMAssembler:
    """
    Assemble global FEM matrices (stiffness, mass) from LinearElement geometry.
    """

    def __init__(self, elements: LinearTriangularElements):
        self._el = elements

    def assemble_stiffness(self, k_func: callable | None = None) -> csr_matrix:
        """
        Assemble global stiffness matrix.
        Optionally scale local K_e by spatially varying coefficient k(x, y).
        """
        n = self._el.points().shape[0]
        global_stiffness = lil_matrix((n, n), dtype=np.float64)

        for i_tri in range(self._el.triangles().shape[0]):
            i_tri_vertices = self._el.triangles()[i_tri]

            if k_func is not None:
                # Evaluate coefficient at triangle centroid
                xy = self._el.points()[i_tri_vertices].mean(axis=0)
                rho = k_func(xy[0], xy[1])
            else:
                rho = 1.0

            local_stiffness = self._el.local_stiffness(i_tri, const_factor=rho)

            for i_local, i_global in enumerate(i_tri_vertices):
                for j_local, j_global in enumerate(i_tri_vertices):
                    global_stiffness[i_global, j_global] += local_stiffness[i_local, j_local]

        return global_stiffness.tocsr()

    def assemble_mass(self, rho_func: callable | None = None) -> csr_matrix:
        """
        Assemble global mass matrix.
        Optionally scale local M_e by spatially varying density rho(x, y).
        """
        el = self._el
        n = self._el.points().shape[0]
        global_mass = lil_matrix((n, n), dtype=np.float64)

        for i_tri in range(self._el.triangles().shape[0]):
            i_tri_vertices = self._el.triangles()[i_tri]
            rho = 1.0
            if rho_func is not None:
                xy = el.points()[i_tri_vertices].mean(axis=0)
                rho = rho_func(xy[0], xy[1])

            local_mass = el.local_mass(i_tri, const_factor=rho)
            for i_local, i_global in enumerate(i_tri_vertices):
                for j_local, j_global in enumerate(i_tri_vertices):
                    global_mass[i_global, j_global] += local_mass[i_local, j_local]

        return global_mass.tocsr()
