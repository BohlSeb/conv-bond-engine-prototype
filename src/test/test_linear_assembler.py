from __future__ import annotations

import unittest
from timeit import default_timer as timer
import random

import numpy as np

from finite_elements.triangulation import DelaunayMesh2D
from finite_elements.elements import LinearTriElements
from finite_elements.assembler import FEMAssembler


class FEAssemblerTest(unittest.TestCase):

    @staticmethod
    def setup_regular(n: int) -> LinearTriElements:
        x_vals = np.linspace(-1, 2, n)
        y_vals = np.linspace(-1.5, 1, n)
        tri = DelaunayMesh2D(x_vals, y_vals)
        elements = LinearTriElements(tri.points(), tri.triangles(), tri.areas())
        return elements

    @staticmethod
    def setup_irregular() -> tuple[list[float], list[float]]:
        random.seed(42)
        h_x = [random.uniform(0.2, 0.8) for _ in range(10)]
        h_y = [random.uniform(0.2, 0.8) for _ in range(15)]
        intervals_x = [-1.0]
        intervals_y = [2.3]
        for i, _h_x in enumerate(h_x):
            intervals_x.append(intervals_x[i] + _h_x)
        for i, _h_y in enumerate(h_y):
            intervals_y.append(intervals_y[i] + _h_y)
        intervals_x.append(7.0)
        intervals_y.append(16.0)
        return intervals_x, intervals_y

    def test_mass_rectangular_area(self) -> None:
        elements = self.setup_regular(14)
        integrand = np.ones(elements.points().shape[0])
        area = 3.0 * 2.5

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_mass().tocsr()

        result = integrand @ lhs @ integrand
        self.assertAlmostEqual(float(result), area)

    def test_mass_linear_exact_regular(self) -> None:
        elements = self.setup_regular(27)
        integrand = np.array([x[0] + x[1] for x in elements.points()])
        integral = 1.875

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_mass().tocsr()

        result = np.ones_like(integrand) @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_mass_linear_exact_irregular(self) -> None:
        x_knots, y_knots = self.setup_irregular()
        tri = DelaunayMesh2D(x_knots, y_knots)
        elements = LinearTriElements(tri.points(), tri.triangles(), tri.areas())
        integrand = np.array([x[0] + x[1] for x in elements.points()])
        integral = 1331.64

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_mass().tocsr()

        result = np.ones_like(integrand) @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_mass_times(self) -> None:
        print('Testing mass assembler...')

        def f(x: float, y: float) -> float:
            return np.sin(np.pi * x) + np.cos(np.pi * y)

        integral = -2.54648

        for n in [10, 20, 40, 80, 100]:
            elements = self.setup_regular(n)
            integrand = np.array([f(x[0], x[1]) for x in elements.points()])

            start = timer()
            assembler = FEMAssembler(elements)
            lhs = assembler.assemble_mass().tocsr()
            result = integrand @ lhs @ np.ones_like(integrand)
            end = timer()

            print(f'Analytic={integral}, result={result}, n*m={n * n}, time={end - start}')

    def test_stiffness_exact_regular(self) -> None:
        def f(x, y):
            return 3 * x - 2 * y + 1

        elements = self.setup_regular(10)
        integrand = np.array([f(x[0], x[1]) for x in elements.points()])

        grad_norm_squared = 3 ** 2 + (-2) ** 2
        area = 3.0 * 2.5
        integral = area * grad_norm_squared

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_stiffness().tocsr()

        result = integrand @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_stiffness_exact_irregular(self) -> None:
        def f(x, y):
            return 3 * x - 2 * y + 1

        x_knots, y_knots = self.setup_irregular()
        tri = DelaunayMesh2D(x_knots, y_knots)
        elements = LinearTriElements(tri.points(), tri.triangles(), tri.areas())
        integrand = np.array([f(x[0], x[1]) for x in elements.points()])
        grad_norm_squared = 3 ** 2 + (-2) ** 2
        area = 8.0 * 13.7
        integral = area * grad_norm_squared

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_stiffness().tocsr()

        result = integrand @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_stiffness_times(self) -> None:
        print('Testing stiffness assembler...')

        def f(x, y):
            return x ** 2 + y ** 2

        integral = 47.5

        for n in [10, 20, 40, 80, 100]:
            elements = self.setup_regular(n)
            integrand = np.array([f(x[0], x[1]) for x in elements.points()])

            start = timer()
            assembler = FEMAssembler(elements)
            lhs = assembler.assemble_stiffness().tocsr()
            result = integrand @ lhs @ integrand
            end = timer()
            print(f'Analytic={integral}, result={result}, n*m={n * n}, time={end - start}')

    def test_mass_and_stiffness(self) -> None:
        def f(x, y):
            return 3 * x - 2 * y + 1

        x_knots, y_knots = self.setup_irregular()
        tri = DelaunayMesh2D(x_knots, y_knots)
        elements = LinearTriElements(tri.points(), tri.triangles(), tri.areas())
        integrand = np.array([f(x[0], x[1]) for x in elements.points()])

        grad_norm_squared = 3 ** 2 + (-2) ** 2
        area = 8.0 * 13.7
        integral_grad_squared = area * grad_norm_squared
        integral_f_squared = 19668.1
        integral = integral_f_squared + integral_grad_squared

        assembler = FEMAssembler(elements)
        lhs_m = assembler.assemble_mass().tocsr()
        lhs_s = assembler.assemble_stiffness().tocsr()

        result = integrand @ lhs_m @ integrand + integrand @ lhs_s @ integrand
        err = float(abs(result - integral) / integral)
        self.assertLess(err, 1e-6)  # error too big?

    def test_convection(self) -> None:
        # f(x,y) = 2*x^2 + 3*y^2 + 3
        # beta = (w1, w2)
        # Integrate <beta, grad(f)> dxdy over the domain
        def _f(x, y):
            return 2 * x * x + 3 * y * y + 3

        beta = (-3.0, 1.3)

        # wolfram alpha: integrate -3*4*x + 1.3*6*y dx dy, x=-1..2, y=-1.5..1
        expected = -59.625
        # integrate -3*4*x + 1.3*6*y dx dy, x=-1..7, y=2.3..16
        expected_irregular = 3876.55

        elements = self.setup_regular(17)
        f = np.array([_f(x[0], x[1]) for x in elements.points()])
        ones = np.ones_like(f)

        assembler = FEMAssembler(elements)
        lhs = assembler.assemble_convection(weight_x=lambda x, y: beta[0], weight_y=lambda x, y: beta[1]).tocsr()
        f = np.array([_f(x[0], x[1]) for x in elements.points()])
        result = ones @ lhs @ f
        self.assertAlmostEqual(float(result), expected)

        x_grid, y_grid = self.setup_irregular()
        tri_irr = DelaunayMesh2D(x_grid, y_grid)
        elements_irr = LinearTriElements(tri_irr.points(), tri_irr.triangles(), tri_irr.areas())
        f_irr = np.array([_f(x[0], x[1]) for x in elements_irr.points()])
        ones_irr = np.ones_like(f_irr)

        assembler_irr = FEMAssembler(elements_irr)
        lhs_irr = assembler_irr.assemble_convection(weight_x=lambda x, y: beta[0],
                                                    weight_y=lambda x, y: beta[1]).tocsr()
        result_irr = ones_irr @ lhs_irr @ f_irr
        err = float(abs(result_irr - expected_irregular) / expected_irregular)
        self.assertLess(err, 1e-6)  # error too big?
