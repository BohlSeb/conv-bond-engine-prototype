from __future__ import annotations

import unittest

import numpy as np

from finite_elements.boundary import ConstRobinBC
from finite_elements.functions import Scalar, Constant
from finite_elements.triangulation import DelaunayMesh2D
from finite_elements.elements import LinearTriElements
from finite_elements.assembler import FEMAssembler
from finite_elements.boundary import RectangleHelper, ConstStrongDirichletBC, ConstNeumannBC

from test.utils import plot_solution

PLOT = False


def test_condition_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def test_condition_trig(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) - np.cos(np.pi * y)


class BoundaryConditionTest(unittest.TestCase):

    def setUp(self):
        # i_x = ConcentratingInterval(-1.0, 1.0, 40, 0.5, 0.1)
        i_x = np.linspace(-1.0, 1.0, 20)
        tri = DelaunayMesh2D(i_x, i_x)
        self._elements = LinearTriElements(tri.points(), tri.triangles(), tri.areas())

    # most basic elliptic equation: \div \grad u = 0 with Dirichlet BC
    # The linear function defined on the boundary is also the solution inside the domain
    def test_laplace_dirichlet(self) -> None:
        elements = self._elements
        helper = RectangleHelper(elements.points())
        linear_condition = ConstStrongDirichletBC(helper.boundary(), elements.points(), Scalar(lambda x, y: x + y))
        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_stiffness()
        rhs = np.zeros(elements.points().shape[0])
        linear_condition.apply(lhs, rhs)
        u = np.linalg.solve(lhs.toarray(), rhs)
        u_expected = test_condition_linear(elements.points()[:, 0], elements.points()[:, 1])
        np.testing.assert_allclose(u, u_expected, atol=1e-14)
        if PLOT:
            plot_solution(elements, u, "laplace u = 0 with linear Dirichlet BC")

    def test_poisson_const(self) -> None:
        elements = self._elements
        helper = RectangleHelper(elements.points())
        f = 10 * np.ones(elements.points().shape[0])
        linear_condition = ConstStrongDirichletBC(helper.boundary(), elements.points(),
                                                  Scalar(lambda x, y: np.sin(np.pi * x) - np.cos(np.pi * y)))
        assembler = FEMAssembler(elements)
        lhs_f = assembler.assemble_stiffness()
        rhs_f = assembler.assemble_mass().toarray() @ f
        linear_condition.apply(lhs_f, rhs_f)
        u_approx_f = np.linalg.solve(lhs_f.toarray(), rhs_f)
        if PLOT:
            plot_solution(elements, u_approx_f, "laplace u = 10 with trig. Dirichlet BC")

    def test_poisson_trigonometric(self) -> None:
        elements = self._elements
        helper = RectangleHelper(elements.points())
        x = elements.points()[:, 0]
        y = elements.points()[:, 1]
        f = 2 * np.pi ** 2 * (np.sin(np.pi * x) - np.cos(np.pi * y))
        trig_condition = ConstStrongDirichletBC(helper.boundary(), elements.points(), Scalar(lambda _x, _y: _x + _y))
        assembler = FEMAssembler(elements)
        lhs_f = assembler.assemble_stiffness()
        rhs_f = assembler.assemble_mass().toarray() @ f
        trig_condition.apply(lhs_f, rhs_f)
        u_approx_f = np.linalg.solve(lhs_f.toarray(), rhs_f)
        if PLOT:
            plot_solution(elements, u_approx_f, "laplace u = trigonometric f with linear Dirichlet BC")

    def test_laplace_neumann(self) -> None:
        elements = self._elements
        helper = RectangleHelper(elements.points())
        dirichlet_1 = ConstStrongDirichletBC(helper.x_min(), elements.points(), Scalar(lambda x, y: x + y))
        dirichlet_2 = ConstStrongDirichletBC(helper.x_max(), elements.points(), Scalar(lambda x, y: x + y))
        dirichlet_3 = ConstStrongDirichletBC(helper.y_max(), elements.points(), Scalar(lambda x, y: x + y))
        neumann = ConstNeumannBC(helper.y_min(), elements.points(), Constant(-1.0))
        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_stiffness()
        rhs = np.zeros(elements.points().shape[0])
        neumann.apply(rhs)
        dirichlet_1.apply(lhs, rhs)
        dirichlet_2.apply(lhs, rhs)
        dirichlet_3.apply(lhs, rhs)
        u_approx = np.linalg.solve(lhs.toarray(), rhs)
        if PLOT:
            plot_solution(elements, u_approx, "laplace u = 0, with Neumann for x_min and linear Dirichlet elsewhere")

    def test_robin_neumann_sanity(self) -> None:
        elements = self._elements
        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_stiffness()

        lhs_robin = assembler.assemble_stiffness()
        rhs = np.zeros(elements.points().shape[0])
        rhs_robin = np.zeros(elements.points().shape[0])

        helper = RectangleHelper(elements.points())

        boundaries = [b for b in [helper.x_min(), helper.x_max(), helper.y_min(), helper.y_max()]]

        for i, b_points in enumerate(boundaries):
            neumann = ConstNeumannBC(b_points, elements.points(), Constant(0.5))
            robin = ConstRobinBC(b_points, elements.points(), Constant(0.5),
                                 Constant(0.0), Constant(1.0))
            neumann.apply(rhs)
            robin.apply(lhs_robin, rhs_robin)
        np.testing.assert_allclose(lhs.toarray(), lhs_robin.toarray(), atol=1e-14)
        np.testing.assert_allclose(rhs, rhs_robin, atol=1e-14)
