from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix

from typing import Callable, TYPE_CHECKING, Optional

from finite_elements.constants import EPSILON_TOL

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from src.finite_elements.elements import LinearTriangles, LinearIntervals


class LinIntervalAssembler:

    def __init__(self, elements: LinearIntervals) -> None:
        """
        Assemble global FEM matrices (stiffness, mass, convection) from interval elements.
        """
        self._el = elements

    def assemble_stiffness(self, weight: Callable[[float], float] | None = None) -> lil_matrix:
        return self._assemble(lambda i, w: self._el.stiffness(i, w), weight=weight)

    def assemble_mass(self, weight: Callable[[float], float] | None = None) -> lil_matrix:
        return self._assemble(lambda i, w: self._el.mass(i, w), weight=weight)

    def assemble_convection(self, beta: Callable[[float], float]) -> lil_matrix:
        return self._assemble(lambda i, w: self._el.convection(i, w), weight=beta)

    def _assemble(self,
                  local_c: Callable[[int, float], NDArray[np.float64]],
                  weight: Callable[[float], float] | None = None) -> lil_matrix:
        n = self._el.points().shape[0]
        # lil_matrix: "list of lists" sparse matrix for efficient construction / modification
        out = lil_matrix((n, n), dtype=np.float64)
        for i0, i1 in self._el.intervals():
            w = 1.0
            if weight is not None:
                xm = 0.5 * (self._el.points()[i0] + self._el.points()[i1])
                w = float(weight(xm))
            local = local_c(i0, w)
            out[i0, i0] += local[0, 0]
            out[i0, i1] += local[0, 1]
            out[i1, i0] += local[1, 0]
            out[i1, i1] += local[1, 1]
        return out


class LinTriangleAssembler:
    """
    Assemble global FEM matrices (stiffness, mass, convection, SUPG stabilization) from triangle elements.
    """

    def __init__(self, elements: LinearTriangles) -> None:
        self._el = elements

    def assemble_stiffness(self,
                           weights: Callable[[float, float], float] | None = None,
                           diffusion_tensor: Optional[NDArray[np.float64]] = None) -> lil_matrix:
        if diffusion_tensor is not None:
            if diffusion_tensor.shape != (2, 2):
                raise ValueError(f'Diffusion tensor must be of shape (2, 2), got {diffusion_tensor.shape}')
        else:
            diffusion_tensor = np.eye(2, dtype=np.float64)
        n = self._el.points().shape[0]
        global_stiff = lil_matrix((n, n), dtype=np.float64)
        for i_tri in range(self._el.triangles().shape[0]):
            i_tri_vertices = self._el.triangles()[i_tri]
            weight = 1.0
            if weights is not None:
                # Evaluate coefficient at triangle centroid
                xy = self._el.points()[i_tri_vertices].mean(axis=0)
                weight = weights(xy[0], xy[1])

            local_stiffness = self._el.stiffness(i_tri, weight, diffusion_tensor)

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

            local_mass = self._el.mass(i_tri, weight)

            for i_local, i_global in enumerate(i_tri_vertices):
                for j_local, j_global in enumerate(i_tri_vertices):
                    global_mass[i_global, j_global] += local_mass[i_local, j_local]
        return global_mass

    def assemble_convection(self,
                            weight_x: Callable[[float, float], float] | None = None,
                            weight_y: Callable[[float, float], float] | None = None) -> lil_matrix:
        n = self._el.points().shape[0]
        global_conv = lil_matrix((n, n), dtype=np.float64)
        for i_tri in range(self._el.triangles().shape[0]):
            i_tri_vertices = self._el.triangles()[i_tri]
            xy = self._el.points()[i_tri_vertices].mean(axis=0)
            w_x, w_y = 1.0, 1.0
            if weight_x is not None:
                w_x = weight_x(xy[0], xy[1])
            if weight_y is not None:
                w_y = weight_y(xy[0], xy[1])

            local_conv = self._el.convection(i_tri, (w_x, w_y))

            for i_local, i_global in enumerate(i_tri_vertices):
                for j_local, j_global in enumerate(i_tri_vertices):
                    global_conv[i_global, j_global] += local_conv[i_local, j_local]
        return global_conv

    def assemble_supg(self,
                      beta_x: Callable[[float, float], float],
                      beta_y: Callable[[float, float], float],
                      diffusion_tensor: NDArray[np.float64],
                      reaction: float = 0.0) -> lil_matrix:
        """
        Assemble global SUPG stabilization matrix:
            Σ_e ∫_Te τ (β·∇φ_i)(β·∇φ_j) dA

        diffusion_tensor is used only to derive κ_eff along the streamline.
        """
        if diffusion_tensor.shape != (2, 2):
            raise ValueError(f"diffusion_tensor must be (2,2), got {diffusion_tensor.shape}")

        n = self._el.points().shape[0]
        global_supg = lil_matrix((n, n), dtype=np.float64)

        a_mat = diffusion_tensor.astype(np.float64, copy=False)

        for e_idx in range(self._el.triangles().shape[0]):
            vertices = self._el.triangles()[e_idx]
            xy = self._el.points()[vertices].mean(axis=0)

            bx = float(beta_x(xy[0], xy[1]))
            by = float(beta_y(xy[0], xy[1]))

            beta = np.array([bx, by], dtype=np.float64)
            beta_norm_sq = float(beta @ beta)

            if beta_norm_sq < EPSILON_TOL:
                raise RuntimeError(f'Too small beta encountered at triangle {e_idx}')

            # κ_eff = (β^T A β) / (β^T β)
            kappa_eff = float(beta @ (a_mat @ beta)) / beta_norm_sq
            local_supg = self._el.supg(e_idx, (bx, by), kappa_eff=kappa_eff, reaction=reaction)

            for i_local, i_global in enumerate(vertices):
                for j_local, j_global in enumerate(vertices):
                    global_supg[i_global, j_global] += local_supg[i_local, j_local]

        return global_supg
