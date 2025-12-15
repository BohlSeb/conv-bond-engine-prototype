from __future__ import annotations

import unittest
import numpy as np
from matplotlib import pyplot as plt

from finite_elements.functions import Constant
from finite_elements.interval import ConcentratingInterval
from finite_elements.triangulation import DelaunayMesh2D
from finite_elements.elements import LinearTriElements
from finite_elements.assembler import FEMAssembler
from finite_elements.boundary import RectangleHelper, ConstDirichletBC, ConstRobinBC, merge_dirichlet_last_wins, \
    apply_dirichlet_sparse

from test.utils import plot_solution

PLOT = False

class TestConvection(unittest.TestCase):

    def test_convection(self):
        for kappa in [0.3, 0.1, 0.01, 0.001, 0.0001]:
            self._run_Jay_Gopalakrishnan_Con_fusion(kappa, False, False)

    def test_convection_weak_dirichlet(self):
        for kappa in [0.3, 0.1, 0.01, 0.001, 0.0001]:
            self._run_Jay_Gopalakrishnan_Con_fusion(kappa, True, False)

    def test_convection_concentrating(self):
        for kappa in [0.3, 0.1, 0.01, 0.001, 0.0001]:
            self._run_Jay_Gopalakrishnan_Con_fusion(kappa, False, True)

    @staticmethod
    def _run_Jay_Gopalakrishnan_Con_fusion(kappa: float,
                                           weak_dirichlet: bool = False,
                                           concentrating: bool = False) -> None:
        # This example is given here and is very nicely explained:
        # https://web.pdx.edu/~gjay/teaching/mth651_2023/651-jupyterlite/FEMnotebooks/D_Confusion.html
        # https://web.pdx.edu/~gjay/

        f = 0
        g = 1 / 2
        dirichlet_beta = (1, 0)

        y = np.linspace(0, 1.0, 20)
        if not concentrating:
            x = y
        else:
            x = np.array(ConcentratingInterval(0.0, 1.0, 20, 1.0 - kappa, 2 * kappa).grid())
        mesh = DelaunayMesh2D(x, y)
        elements = LinearTriElements(mesh.points(), mesh.triangles(), mesh.areas())
        b_help = RectangleHelper(elements.points())

        dirichlet_bcs = []
        robin_bcs = []

        # left
        dirichlet_bcs.append(ConstDirichletBC(b_help.x_min(), elements.points(), Constant(g)))

        # right
        if not weak_dirichlet:
            dirichlet_bcs.append(ConstDirichletBC(b_help.x_max(), elements.points(), Constant(0.0)))
        else:
            bc_right = ConstRobinBC(b_help.x_max(),
                                    elements.points(),
                                    Constant(0.0 * kappa),  # rhs
                                    Constant(kappa),  # "dirichlet alpha"
                                    Constant(1))  # "neumann beta"
            robin_bcs.append(bc_right)

        bc_top = ConstRobinBC(b_help.y_max(),
                              elements.points(),
                              Constant(0.0),  # rhs
                              Constant(0.0),  # "dirichlet alpha" = <dirichlet_beta,n> = <(1,0),(0,1)> = 0
                              Constant(kappa))  # "neumann beta"
        bc_bottom = ConstRobinBC(b_help.y_min(),
                                 elements.points(),
                                 Constant(0.0),
                                 Constant(0.0),
                                 Constant(kappa))
        robin_bcs.append(bc_top)
        robin_bcs.append(bc_bottom)

        assembler = FEMAssembler(elements)

        f_mass = assembler.assemble_mass()
        rhs = f_mass.toarray() @ np.full(elements.points().shape[0], f)

        lhs_stiff = assembler.assemble_stiffness()
        lhs_conv = assembler.assemble_convection(weight_x=lambda _x, _y: dirichlet_beta[0],
                                                 weight_y=lambda _x, _y: dirichlet_beta[1])
        # Note: We convert both lhs_stiff and lhs_conv to CSR format for efficient arithmetic,
        # then convert the result to LIL format for efficient modification (e.g., applying boundary conditions).
        # If performance or memory usage becomes an issue, revisit the choice of formats here.
        # See: https://docs.scipy.org/doc/scipy/reference/sparse.html for details on sparse matrix formats.
        lhs = (kappa * lhs_stiff.tocsr() + lhs_conv.tocsr()).tolil()

        for robin_bc in robin_bcs:
            robin_bc.apply(lhs, rhs)
        dirichlet_data = merge_dirichlet_last_wins([bc.data() for bc in dirichlet_bcs])
        lhs, rhs = apply_dirichlet_sparse(lhs, rhs, dirichlet_data)
        u_approx = np.linalg.solve(lhs.toarray(), rhs)
        if PLOT:
            import QuantLib as ql

            plot_solution(elements, u_approx,
                          title=fr"FEM Solution of Jay's Con-fusion Problem: $\kappa$ = {kappa}" + f'\n{weak_dirichlet = }, {concentrating = }')

            x_plt = np.linspace(0.0, 1.0, 100)
            u = (1 - np.exp((x_plt - 1) / kappa)) / (1 - np.exp(-1 / kappa)) / 2
            plt.plot(x_plt, u)
            u_x = ql.LinearInterpolation(x.tolist(), u_approx[b_help.y_min()].tolist())
            u_x_plt = [u_x(xi) for xi in x_plt]
            plt.plot(x_plt, u_x_plt, 'r--')
            plt.xlabel('$x$')
            plt.grid(True)
            plt.title(
                fr"Exact solution $u$ vs approx, $\kappa$ = {kappa}" + f'\n{weak_dirichlet = }, {concentrating = }')
            plt.show()
