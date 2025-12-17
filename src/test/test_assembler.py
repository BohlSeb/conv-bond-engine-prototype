from __future__ import annotations

import unittest
from timeit import default_timer as timer
import random

import numpy as np

from finite_elements.triangulation import DelaunayMesh2D
from finite_elements.elements import LinearTriangles, LinearIntervals
from finite_elements.assembler import LinTriangleAssembler, LinIntervalAssembler


class FEAssembler2DTest(unittest.TestCase):

    @staticmethod
    def setup_regular(n: int) -> LinearTriangles:
        x_vals = np.linspace(-1, 2, n)
        y_vals = np.linspace(-1.5, 1, n)
        tri = DelaunayMesh2D(x_vals, y_vals)
        elements = LinearTriangles(tri.points(), tri.triangles(), tri.areas())
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

        assembler = LinTriangleAssembler(elements)
        lhs = assembler.assemble_mass().tocsr()

        result = integrand @ lhs @ integrand
        self.assertAlmostEqual(float(result), area)

    def test_mass_linear_exact_regular(self) -> None:
        elements = self.setup_regular(27)
        integrand = np.array([x[0] + x[1] for x in elements.points()])
        integral = 1.875

        assembler = LinTriangleAssembler(elements)
        lhs = assembler.assemble_mass().tocsr()

        result = np.ones_like(integrand) @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_mass_linear_exact_irregular(self) -> None:
        x_knots, y_knots = self.setup_irregular()
        tri = DelaunayMesh2D(x_knots, y_knots)
        elements = LinearTriangles(tri.points(), tri.triangles(), tri.areas())
        integrand = np.array([x[0] + x[1] for x in elements.points()])
        integral = 1331.64

        assembler = LinTriangleAssembler(elements)
        lhs = assembler.assemble_mass().tocsr()

        result = np.ones_like(integrand) @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_mass_times(self) -> None:
        print('Testing 2d mass assembler...')

        def f(x: float, y: float) -> float:
            return np.sin(np.pi * x) + np.cos(np.pi * y)

        integral = -2.54648

        for n in [10, 20, 40, 80, 100]:
            elements = self.setup_regular(n)
            integrand = np.array([f(x[0], x[1]) for x in elements.points()])

            start = timer()
            assembler = LinTriangleAssembler(elements)
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

        assembler = LinTriangleAssembler(elements)
        lhs = assembler.assemble_stiffness().tocsr()

        result = integrand @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_stiffness_exact_irregular(self) -> None:
        def f(x, y):
            return 3 * x - 2 * y + 1

        x_knots, y_knots = self.setup_irregular()
        tri = DelaunayMesh2D(x_knots, y_knots)
        elements = LinearTriangles(tri.points(), tri.triangles(), tri.areas())
        integrand = np.array([f(x[0], x[1]) for x in elements.points()])
        grad_norm_squared = 3 ** 2 + (-2) ** 2
        area = 8.0 * 13.7
        integral = area * grad_norm_squared

        assembler = LinTriangleAssembler(elements)
        lhs = assembler.assemble_stiffness().tocsr()

        result = integrand @ lhs @ integrand
        self.assertAlmostEqual(float(result), integral)

    def test_stiffness_times(self) -> None:
        print('Testing 2d stiffness assembler...')

        def f(x, y):
            return x ** 2 + y ** 2

        integral = 47.5

        for n in [10, 20, 40, 80, 100]:
            elements = self.setup_regular(n)
            integrand = np.array([f(x[0], x[1]) for x in elements.points()])

            start = timer()
            assembler = LinTriangleAssembler(elements)
            lhs = assembler.assemble_stiffness().tocsr()
            result = integrand @ lhs @ integrand
            end = timer()
            print(f'Analytic={integral}, result={result}, n*m={n * n}, time={end - start}')

    def test_mass_and_stiffness(self) -> None:
        def f(x, y):
            return 3 * x - 2 * y + 1

        x_knots, y_knots = self.setup_irregular()
        tri = DelaunayMesh2D(x_knots, y_knots)
        elements = LinearTriangles(tri.points(), tri.triangles(), tri.areas())
        integrand = np.array([f(x[0], x[1]) for x in elements.points()])

        grad_norm_squared = 3 ** 2 + (-2) ** 2
        area = 8.0 * 13.7
        integral_grad_squared = area * grad_norm_squared
        integral_f_squared = 19668.1
        integral = integral_f_squared + integral_grad_squared

        assembler = LinTriangleAssembler(elements)
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

        assembler = LinTriangleAssembler(elements)
        lhs = assembler.assemble_convection(weight_x=lambda x, y: beta[0], weight_y=lambda x, y: beta[1]).tocsr()
        f = np.array([_f(x[0], x[1]) for x in elements.points()])
        result = ones @ lhs @ f
        self.assertAlmostEqual(float(result), expected)

        x_grid, y_grid = self.setup_irregular()
        tri_irr = DelaunayMesh2D(x_grid, y_grid)
        elements_irr = LinearTriangles(tri_irr.points(), tri_irr.triangles(), tri_irr.areas())
        f_irr = np.array([_f(x[0], x[1]) for x in elements_irr.points()])
        ones_irr = np.ones_like(f_irr)

        assembler_irr = LinTriangleAssembler(elements_irr)
        lhs_irr = assembler_irr.assemble_convection(weight_x=lambda x, y: beta[0],
                                                    weight_y=lambda x, y: beta[1]).tocsr()
        result_irr = ones_irr @ lhs_irr @ f_irr
        err = float(abs(result_irr - expected_irregular) / expected_irregular)
        self.assertLess(err, 1e-6)  # error too big?


class FEAssembler1DTest(unittest.TestCase):

    @staticmethod
    def setup_regular(n: int) -> LinearIntervals:
        # 1D domain [a,b]
        a, b = -1.0, 2.0
        x = np.linspace(a, b, n)
        return LinearIntervals(x)

    @staticmethod
    def setup_irregular() -> LinearIntervals:
        random.seed(42)
        a, b = -1.0, 7.0

        # random positive steps, then rescale to hit [a,b] exactly
        h = np.array([random.uniform(0.2, 0.8) for _ in range(30)], dtype=np.float64)
        h *= (b - a) / float(np.sum(h))
        x = a + np.concatenate([[0.0], np.cumsum(h)])
        x[-1] = b  # exact endpoint
        return LinearIntervals(x)

    def test_mass_constant_length(self) -> None:
        elements = self.setup_regular(200)
        x = elements.points()
        ones = np.ones_like(x)

        assembler = LinIntervalAssembler(elements)
        mass = assembler.assemble_mass().tocsr()
        expected = float(x[-1] - x[0])
        result = float(ones @ mass @ ones)
        self.assertAlmostEqual(result, expected, places=12)

    def test_mass_linear_exact(self) -> None:
        elements = self.setup_regular(301)
        elements_irr = self.setup_irregular()
        x = elements.points()
        x_irr = elements_irr.points()

        # u(x) = x + 1 (linear) -> exactly represented by P1
        u = x + 1.0
        u_irr = x_irr + 1.0
        ones = np.ones_like(x)
        ones_irr = np.ones_like(x_irr)

        assembler = LinIntervalAssembler(elements)
        assembler_irr = LinIntervalAssembler(elements_irr)
        mass = assembler.assemble_mass().tocsr()
        mass_irr = assembler_irr.assemble_mass().tocsr()

        # ∫ (x+1) dx from a to b = 0.5(b^2-a^2) + (b-a)
        a, b = float(x[0]), float(x[-1])
        expected = 0.5 * (b * b - a * a) + (b - a)
        result = float(ones @ mass @ u)
        self.assertAlmostEqual(result, expected, places=12)

        a_irr, b_irr = float(x_irr[0]), float(x_irr[-1])
        expected_irr = 0.5 * (b_irr * b_irr - a_irr * a_irr) + (b_irr - a_irr)
        result_irr = float(ones_irr @ mass_irr @ u_irr)
        self.assertAlmostEqual(result_irr, expected_irr, places=12)


    def test_stiffness_exact_linear(self) -> None:
        elements = self.setup_regular(301)
        elements_irr = self.setup_irregular()
        x = elements.points()
        x_irr = elements_irr.points()

        # u(x) = 3x + 1 -> u' = 3
        u = 3.0 * x + 1.0
        u_irr = 3.0 * x_irr + 1.0

        assembler = LinIntervalAssembler(elements)
        assembler_irr = LinIntervalAssembler(elements_irr)

        stiff = assembler.assemble_stiffness().tocsr()
        stiff_irr = assembler_irr.assemble_stiffness().tocsr()

        a, b = float(x[0]), float(x[-1])
        expected = (3.0 ** 2) * (b - a)  # ∫ (u')^2 dx
        result = float(u @ stiff @ u)
        self.assertAlmostEqual(result, expected, places=10)

        a_irr, b_irr = float(x_irr[0]), float(x_irr[-1])
        expected_irr = (3.0 ** 2) * (b_irr - a_irr)
        result_irr = float(u_irr @ stiff_irr @ u_irr)
        self.assertAlmostEqual(result_irr, expected_irr, places=10)

    def test_convection_invariant(self) -> None:
        """
        With your Galerkin convection matrix C for beta * u_x:
            ones^T C u  ≈  ∫ beta * u'(x) dx = beta*(u(b)-u(a))

        This is a great structural test, and it holds regardless of whether u is linear,
        because both sides depend only on endpoint values when integrated exactly.
        """
        beta = -3.0

        elements = self.setup_regular(250)
        elements_irr = self.setup_irregular()
        x = elements.points()
        x_irr = elements_irr.points()

        def u_fun(xx: np.ndarray) -> np.ndarray:
            return 2.0 * xx * xx + 3.0 * xx + 3.0  # any smooth function

        u = u_fun(x)
        ones = np.ones_like(x)
        u_irr = u_fun(x_irr)
        ones_irr = np.ones_like(x_irr)

        assembler = LinIntervalAssembler(elements)
        assembler_irr = LinIntervalAssembler(elements_irr)
        conv = assembler.assemble_convection(beta=lambda _: beta).tocsr()
        conv_irr = assembler_irr.assemble_convection(beta=lambda _: beta).tocsr()

        a, b = float(x[0]), float(x[-1])
        expected = beta * (u_fun(np.array([b]))[0] - u_fun(np.array([a]))[0])
        result = float(ones @ conv @ u)
        self.assertAlmostEqual(result, expected, places=10)

        a_irr, b_irr = float(x_irr[0]), float(x_irr[-1])
        expected_irr = beta * (u_fun(np.array([b_irr]))[0] - u_fun(np.array([a_irr]))[0])
        result_irr= float(ones_irr @ conv_irr @ u_irr)
        self.assertAlmostEqual(result_irr, expected_irr, places=10)
