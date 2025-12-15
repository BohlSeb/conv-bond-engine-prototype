from __future__ import annotations

import json
import unittest
import math
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt
import QuantLib as ql

from typing import TYPE_CHECKING, Any

from finite_elements.interval import ConcentratingInterval
from finite_elements.triangulation import DelaunayMesh2D
from finite_elements.elements import LinearTriElements
from finite_elements.assembler import FEMAssembler
from finite_elements.boundary import RectangleHelper
from option.european import MarketData, OptionData, BSTransformHelper, EuropeanOptionBCs

from test.utils import plot_solution

if TYPE_CHECKING:
    from numpy.typing import NDArray

PLOT = False


class TestEuropeanOption(unittest.TestCase):

    @staticmethod
    def initialize_today() -> ql.Date:
        today = ql.Date(12, 12, 2025)
        ql.Settings.instance().evaluationDate = today
        return today

    def test_european_call_option(self) -> None:
        today = self.initialize_today()
        market_data = MarketData(spot=100.0, sigma=0.2, r=0.02, q=0.025)
        option_data = OptionData(val_date=today, period2mat=ql.Period('4Y'), strike=90.0, cal=ql.TARGET(),
                                 dc=ql.Actual365Fixed(), put_call=ql.Option.Call)
        size = 50
        res, exact = self._do_test_european_option(option_data, market_data, size, concentrating=True,
                                                   weak_boundary=False)
        print(
            f'Testing European Call Option: analytic {exact:4.2f}, calculated {res:4.2f}, error: {(res - exact) / exact:2.6f}')

    def test_european_put_option(self) -> None:
        today = self.initialize_today()
        market_data = MarketData(spot=100.0, sigma=0.2, r=0.02, q=0.025)
        option_data = OptionData(val_date=today, period2mat=ql.Period('4Y'), strike=110.0, cal=ql.TARGET(),
                                 dc=ql.Actual365Fixed(), put_call=ql.Option.Put)
        size = 50
        res, exact = self._do_test_european_option(option_data, market_data, size, concentrating=False,
                                                   weak_boundary=False)
        print(
            f'Testing European Put Option: analytic {exact:4.2f}, calculated {res:4.2f}, error: {(res - exact) / exact:2.6f}')

    def test_european_option(self, cache_mode: bool = False) -> None:
        if PLOT:
            raise RuntimeError(
                "PLOT=True detected during test execution. "
                "Plotting should not be enabled in CI/CD or automated test runs. "
                "Set PLOT=False to run tests."
            )
        print('Testing european vanilla option with different modes...')
        today = self.initialize_today()
        sizes = [25, 50, 75]
        market_data = MarketData(spot=100.0, sigma=0.2, r=0.02, q=0.025)
        print(market_data)
        cal = ql.TARGET()
        dc = ql.Actual365Fixed()
        calls = [OptionData(today, ql.Period('4Y'), k, cal, dc, ql.Option.Call) for k in (75.0, 100.0, 125.0)]
        puts = [OptionData(today, ql.Period('4Y'), k, cal, dc, ql.Option.Put) for k in (75.0, 100.0, 125.0)]
        modes = [
            (False, False),
            (True, False),
            (False, True),
            (True, True)
        ]

        cached_out: dict[str, list[dict[str, Any]]] = {}
        cache_in: dict[str, dict[tuple[int, bool, bool], tuple[float, float]]] = {}
        if not cache_mode:
            with open(Path(__file__).parent / 'test_data' / 'european_option_reg_test.json', 'r') as h:
                cached_data = json.load(h)
            for option_id, data in cached_data.items():
                cache_in[option_id] = {tuple(v['keys']): (v['analytic'], v['calculated']) for v in data}

        for (option_id, option) in zip(['ITM C', 'ATM C', 'OTM C', 'OTM P', 'ATM P', 'ITM P'], calls + puts):
            print(option_id, option.strike)
            if cache_mode:
                cached_out[option_id] = []
            for n in sizes:
                print(f'Size: {n}')
                for concentrating, weak_bc in modes:
                    start = timer()
                    calculated, expected = self._do_test_european_option(option, market_data, n,
                                                                         concentrating=concentrating,
                                                                         weak_boundary=weak_bc)
                    cal_time = timer() - start
                    err = calculated - expected
                    rel_err = err / expected
                    print(
                        f'Expected {expected:10.6f}, Calculated: {calculated:10.6f}, Error: {err:10.6f}, RelError: {rel_err:10.6f}, time: {cal_time:2.2f}, concentrating: {concentrating}, weak_bc: {weak_bc}')

                    if cache_mode:
                        keys = [n, concentrating, weak_bc]
                        cached_out[option_id].append(
                            {'keys': keys, 'analytic': round(expected, 8), 'calculated': round(calculated, 10)})
                    else:
                        key = (n, concentrating, weak_bc)
                        cached_exp, cached_cal = cache_in[option_id][key]
                        self.assertLess(abs(cached_exp - expected) / cached_exp, 1e-3,
                                        msg=f'Expected analytic: {cached_exp}, analytic: {expected}')
                        self.assertLess(abs(cached_cal - calculated), 0.1,
                                        msg=f'Expected calculated: {cached_cal}, calculated: {calculated}')
        if cache_mode:
            with open('european_option_reg_test.json', 'w') as h:
                json.dump(cached_out, h, indent=4)

    @staticmethod
    def _compare_analytic(option_data: OptionData,
                          market_data: MarketData,
                          elements: LinearTriElements,
                          boundary_h: RectangleHelper,
                          transform_h: BSTransformHelper,
                          u_approx: NDArray[np.float64],
                          concentrating: bool,
                          robin_boundary: bool) -> tuple[float, float]:
        if PLOT:
            plot_solution(elements,
                          u_approx,
                          title=f"European {'Call' if option_data.put_call == ql.Option.Call else 'Put'} Option FEM Solution, {concentrating = }, {robin_boundary = }")
        ql_option = ql.EuropeanOption(ql.PlainVanillaPayoff(option_data.put_call, option_data.strike),
                                      ql.EuropeanExercise(option_data.expiry()))
        spot_quote = ql.SimpleQuote(0.0)
        val_date, dc = option_data.val_date, option_data.dc
        bs_process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(spot_quote),
            ql.YieldTermStructureHandle(ql.FlatForward(val_date, market_data.q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(val_date, market_data.r, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, option_data.cal, market_data.sigma, dc))
        )
        ql_option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))

        if PLOT:
            spot_eps = 1e-14
            s_plt = np.linspace(option_data.strike * math.exp(transform_h.x_min_coord) + spot_eps,
                                option_data.strike * math.exp(transform_h.x_max_coord) - spot_eps, 100)
            u_exact = []
            for s in s_plt:
                spot_quote.setValue(s)
                u_exact.append(ql_option.NPV())
            plt.plot(s_plt, u_exact, label='Analytical Solution', color='black')

        t_max = boundary_h.x_max()
        log_s_k = elements.points()[t_max][:, 1]
        u_intp = ql.LinearInterpolation(log_s_k.tolist(), u_approx[t_max].tolist())

        if PLOT:
            u_approx = []
            for s in s_plt:
                x = math.log(s / option_data.strike)
                u_approx.append(u_intp(x) * option_data.strike)
            plt.plot(s_plt, u_approx, 'r--', label='FEM Solution at t=0')
            plt.legend()
            plt.grid()
            plt.show()

        spot_quote.setValue(market_data.spot)
        exact = ql_option.NPV()
        return u_intp(math.log(market_data.spot / option_data.strike)) * option_data.strike, exact

    def _do_test_european_option(self,
                                 option_data: OptionData,
                                 market_data: MarketData,
                                 size: int,
                                 concentrating: bool = False,
                                 weak_boundary: bool = False) -> tuple[float, float]:

        std_devs = 6
        tf_helper = BSTransformHelper(market_data, option_data, std_devs=std_devs)

        beta = size * 1e-4
        i_x = np.linspace(0.0, option_data.time2maturity(), size)
        if concentrating:
            i_y = np.array(ConcentratingInterval(tf_helper.x_min_coord, tf_helper.x_max_coord, size, 0.0, beta).grid())
        else:
            i_y = np.linspace(tf_helper.x_min_coord, tf_helper.x_max_coord, size)
        # TODO: Consider refining the mesh or adjusting the grid spacing in i_y so that all triangles
        # in the Delaunay triangulation have a minimum angle above a specified threshold (e.g., 20 degrees).
        # This helps prevent poorly shaped (sliver) triangles, which can negatively affect numerical accuracy.
        # To implement: add a mesh quality check after mesh generation, and if any triangle has an angle
        # below the threshold, refine the mesh or adjust the interval accordingly.

        grid = DelaunayMesh2D(i_x, i_y)
        tris = LinearTriElements(grid.points(), grid.triangles(), grid.areas())
        rec_helper = RectangleHelper(tris.points())

        assembler = FEMAssembler(tris)

        lhs = assembler.assemble_stiffness(diffusion_tensor=tf_helper.A).copy()
        lhs += assembler.assemble_convection(weight_x=lambda _x, _y: tf_helper.beta()[0],
                                             weight_y=lambda _x, _y: tf_helper.beta()[1])
        lhs += tf_helper.mass_weight * assembler.assemble_mass()

        rhs = np.zeros_like(tris.points()[:, 0])

        bc = EuropeanOptionBCs(market_data, tf_helper, rec_helper, tris.points())
        p_c = option_data.put_call
        if weak_boundary:
            bc.bc_nonzero_weak(p_c).apply(lhs, rhs)
            bc.bc_outflow_neumann0().apply(lhs, rhs)
        else:
            bc.bc_nonzero_strong(p_c).apply(lhs, rhs)
            bc.bc_zero_strong(p_c).apply(lhs, rhs)
        bc.bc_maturity(p_c).apply(lhs, rhs)
        u = np.linalg.solve(lhs.toarray(), rhs)
        approx, exact = self._compare_analytic(option_data, market_data, tris, rec_helper, tf_helper, u, concentrating,
                                               weak_boundary)
        return approx, exact
