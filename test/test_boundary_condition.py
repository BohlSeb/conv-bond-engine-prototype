from __future__ import annotations

import unittest

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from finite_elements.boundary import ConstNeumannCondition
from src.finite_elements.triangulation import RectangleMesher
from src.finite_elements.elements import LinearTriElements
from src.finite_elements.assembler import FEMAssembler
from src.finite_elements.boundary import RectangleHelper, XYCallBack, ConstDirichletCondition


def test_condition_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def test_condition_trig(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) - np.cos(np.pi * y)


def test_linear_flux_ymin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -1 * np.ones_like(x)


class DirichletConditionTest(unittest.TestCase):

    def setUp(self):
        x = np.linspace(0, 1, 30)
        y = np.linspace(0, 1, 30)
        tri = RectangleMesher(x, y)
        self._elements = LinearTriElements(tri.points(), tri.triangles(), tri.areas())

    @staticmethod
    def plot_solution(elements: LinearTriElements, u: np.ndarray, title: str = "FEM Solution") -> None:
        """
        Plot a solution vector u on the triangular mesh defined by LinearTriElements.
        """
        points = elements.points()
        triangles = elements.triangles()

        x = points[:, 0]
        y = points[:, 1]

        # create triangulation
        triang = mtri.Triangulation(x, y, triangles)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_trisurf(triang, u, cmap='viridis', edgecolor='k', linewidth=0.3, alpha=0.9)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(title)
        plt.show()

    # most basic poisson equation
    def test_zero_constant_laplace_with_dirichlet(self) -> None:
        elements = self._elements

        helper = RectangleHelper(elements.points())
        linear_condition = ConstDirichletCondition(helper.boundary(), XYCallBack(test_condition_linear), elements.points())

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_stiffness()

        # f = 0
        rhs = np.zeros(elements.points().shape[0])

        linear_condition.apply(lhs, rhs)

        u_exact = np.linalg.solve(lhs.toarray(), rhs)
        u_expected = test_condition_linear(elements.points()[:, 0], elements.points()[:, 1])
        np.testing.assert_allclose(u_exact, u_expected, rtol=1e-12)

        trig_condition = ConstDirichletCondition(helper.boundary(), XYCallBack(test_condition_trig), elements.points())
        trig_condition.apply(lhs, rhs)
        u_approx = np.linalg.solve(lhs.toarray(), rhs)

        f = 0.05
        rhs_f = f * np.ones(elements.points().shape[0])
        trig_condition.apply(lhs, rhs_f)
        u_approx_f = np.linalg.solve(lhs.toarray(), rhs_f)

        plot = False
        if plot:
            self.plot_solution(elements, u_exact, title="FEM Solution laplace u = 0")
            self.plot_solution(elements, u_approx, title="FEM Solution laplace u = 0")
            self.plot_solution(elements, u_approx_f, title="FEM Solution laplace u = constant f")

    # most basic poisson equation
    def test_zero_constant_laplace_with_neumann(self) -> None:
        elements = self._elements

        helper = RectangleHelper(elements.points())

        dirichlet = ConstDirichletCondition(helper.x_min(), XYCallBack(test_condition_linear), elements.points())
        neumann = ConstNeumannCondition(helper.y_min(), XYCallBack(test_linear_flux_ymin), elements.points())

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_stiffness()

        # f = 0
        rhs = np.zeros(elements.points().shape[0])

        neumann.apply(rhs)
        dirichlet.apply(lhs, rhs)

        u_approx = np.linalg.solve(lhs.toarray(), rhs)

        plot = False
        if plot:
            self.plot_solution(elements, u_approx, title="FEM Solution poisson f = constant")

