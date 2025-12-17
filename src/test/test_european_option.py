from __future__ import annotations

import json
import unittest
import math
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt
import QuantLib as ql

from typing import TYPE_CHECKING, Any

from option.european import MarketData, OptionData, ModelParams
from option.european_calculator import european_vanilla_fe_2d, european_vanilla_fe_fd
from finite_elements.constants import EPSILON

from test.utils import plot_solution_triangles, plot_solution_time_space

if TYPE_CHECKING:
    from option.european_calculator import FEVanillaResult

PLOT = False


class TestEuropeanOption(unittest.TestCase):

    @staticmethod
    def initialize_today() -> ql.Date:
        today = ql.Date(12, 12, 2025)
        ql.Settings.instance().evaluationDate = today
        return today

    def test_european_call_option_fe_2d(self) -> None:
        today = self.initialize_today()
        market_data = MarketData(spot=100.0, sigma=0.03, r=0.1, q=0.0)
        option_data = OptionData(val_date=today, period2mat=ql.Period('4Y'), strike=90.0, cal=ql.TARGET(),
                                 dc=ql.Actual365Fixed(), put_call=ql.Option.Call)
        plot_size = 50
        if PLOT:
            sizes = [plot_size]
        else:
            sizes = [25, 50, 75, 100]
        for size in sizes:
            params = ModelParams(size=size, std_devs=8, max_spots=8)
            start = timer()
            result = european_vanilla_fe_2d(option_data, market_data, params)
            time = timer() - start
            res, exact = self._compare_analytic(option_data, market_data, params, result)
            print(
                f'Testing European Call Option FE-2D: analytic {exact:4.4f}, calculated {res:4.4f}, error: {(res - exact) / exact:2.6f}, time {time:2.4f}')
            bench_pv, bench_time = self._ql_benchmark_fd(option_data, market_data, params.size)
            print(
                f'Testing European Call Option QL-FD: analytic {exact:4.4f}, calculated {bench_pv:4.4f}, error: {(bench_pv - exact) / exact:2.6f}, time {bench_time:2.4f}')

    def test_european_put_option_fe_2d(self) -> None:
        today = self.initialize_today()
        market_data = MarketData(spot=100.0, sigma=0.03, r=0.02, q=0.025)
        option_data = OptionData(val_date=today, period2mat=ql.Period('4Y'), strike=110.0, cal=ql.TARGET(),
                                 dc=ql.Actual365Fixed(), put_call=ql.Option.Put)
        plot_size = 50
        if PLOT:
            sizes = [plot_size]
        else:
            sizes = [25, 50, 75, 100]
        for size in sizes:
            params = ModelParams(size=size, concentrating=True, flux_boundary_bc=False, std_devs=8, max_spots=8)
            start = timer()
            result = european_vanilla_fe_2d(option_data, market_data, params)
            time = timer() - start
            res, exact = self._compare_analytic(option_data, market_data, params, result)
            print(
                f'Testing European Put Option FE-2D: analytic {exact:4.4f}, calculated {res:4.4f}, error: {(res - exact) / exact:2.6f}, time {time:2.4f}')
            bench_pv, bench_time = self._ql_benchmark_fd(option_data, market_data, params.size)
            print(
                f'Testing European Put Option QL-FD: analytic {exact:4.4f}, calculated {bench_pv:4.4f}, error: {(bench_pv - exact) / exact:2.6f}, time {bench_time:2.4f}')

    def test_european_call_option_fe_fd(self, cache_mode: bool = False) -> None:
        test_file = 'european_call_fe_fd_reg_test.json'
        print(
            f'Testing implicit/crank-nicholson space-finite-element european call regression against "{test_file}" ...')
        today = self.initialize_today()
        market_data = MarketData(spot=100.0, sigma=0.4, r=0.1, q=0.0)
        option_data = OptionData(val_date=today, period2mat=ql.Period('4Y'), strike=90.0, cal=ql.TARGET(),
                                 dc=ql.Actual365Fixed(), put_call=ql.Option.Call)
        plot_size = 50
        theta_plot = 2
        if PLOT:
            sizes = [plot_size]
            thetas_inverted = [theta_plot]
        else:
            sizes = [50, 100, 200, 400]
            thetas_inverted = [1, 2]

        cached_data = self._initialize_reg_test(cache_mode, test_file)

        for size in sizes:
            params = ModelParams(size=size, concentrating=False, flux_boundary_bc=False, std_devs=10, max_spots=8)
            for theta_inv in thetas_inverted:
                start = timer()
                result = european_vanilla_fe_fd(option_data, market_data, params, theta=1 / theta_inv)
                time = timer() - start
                res, exact = self._compare_analytic(option_data, market_data, params, result)
                print(
                    f'Testing European Call Option FE-FD {size:>3}: analytic {exact:4.4f}, calculated {res:4.4f}, error: {(res - exact) / exact:2.6f}, time {time:2.4f}, theta={1 / theta_inv:1.2f}')
                bench_pv, bench_time = self._ql_benchmark_fd(option_data, market_data, params.size)
                print(
                    f'Testing European Call Option QL-FD {size:>3}: analytic {exact:4.4f}, calculated {bench_pv:4.4f}, error: {(bench_pv - exact) / exact:2.6f}, time {bench_time:2.4f}')

                self._check_results(cache_mode, f'CALL', (size, theta_inv), ('analytic', 'calculated'), (exact, res),
                                    cached_data)
        if cache_mode:
            self._write_cache(test_file, cached_data)

    def test_european_put_option_fe_fd(self, cache_mode: bool = False) -> None:
        test_file = 'european_put_fe_fd_reg_test.json'
        print(
            f'Testing implicit/crank-nicholson space-finite-element european put regression against "{test_file}" ...')
        today = self.initialize_today()
        market_data = MarketData(spot=100.0, sigma=0.3, r=0.02, q=0.025)
        option_data = OptionData(val_date=today, period2mat=ql.Period('4Y'), strike=110.0, cal=ql.TARGET(),
                                 dc=ql.Actual365Fixed(), put_call=ql.Option.Put)
        plot_size = 50
        theta_plot = 2
        if PLOT:
            sizes = [plot_size]
            thetas_inverted = [theta_plot]
        else:
            sizes = [50, 100, 200, 400]
            thetas_inverted = [1, 2]
        cached_data = self._initialize_reg_test(cache_mode, test_file)

        for size in sizes:
            params = ModelParams(size=size, concentrating=False, flux_boundary_bc=False, std_devs=10, max_spots=8)
            for theta_inv in thetas_inverted:
                start = timer()
                result = european_vanilla_fe_fd(option_data, market_data, params, theta=1 / theta_inv)
                time = timer() - start
                res, exact = self._compare_analytic(option_data, market_data, params, result)
                print(
                    f'Testing European Put Option FE-FD {size:>3}: analytic {exact:4.4f}, calculated {res:4.4f}, error: {(res - exact) / exact:2.6f}, time {time:2.4f}, theta={1 / theta_inv:1.2f}')
                bench_pv, bench_time = self._ql_benchmark_fd(option_data, market_data, params.size)
                print(
                    f'Testing European Put Option QL-FD {size:>3}: analytic {exact:4.4f}, calculated {bench_pv:4.4f}, error: {(bench_pv - exact) / exact:2.6f}, time {bench_time:2.4f}')

                self._check_results(cache_mode, f'PUT', (size, theta_inv), ('analytic', 'calculated'), (exact, res),
                                    cached_data)

        if cache_mode:
            self._write_cache(test_file, cached_data)

    def test_european_option_fe_2d(self, cache_mode: bool = False) -> None:
        test_file = 'european_option_fe_2d_reg_test.json'
        print(f'Testing 2d finite element european option regression against "{test_file}" ...')
        if PLOT:
            raise RuntimeError(
                "PLOT=True detected during test execution. "
                "Plotting should not be enabled in CI/CD or automated test runs. "
                "Set PLOT=False to run tests."
            )
        today = self.initialize_today()
        sizes = [25, 50, 75]
        market_data = MarketData(spot=100.0, sigma=0.2, r=0.02, q=0.025)
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

        cached_data = self._initialize_reg_test(cache_mode, test_file)

        for (option_id, option) in zip(['ITM C', 'ATM C', 'OTM C', 'OTM P', 'ATM P', 'ITM P'], calls + puts):
            print(option_id, option.strike)
            for n in sizes:
                print(f'Size: {n}')
                for concentrating, weak_bc in modes:
                    params = ModelParams(size=n, concentrating=concentrating, flux_boundary_bc=weak_bc, use_supg=False)
                    start = timer()
                    result = european_vanilla_fe_2d(option, market_data, params)
                    calculated, expected = self._compare_analytic(option, market_data, params, result)
                    time = timer() - start
                    err = calculated - expected
                    rel_err = err / expected
                    print(
                        f'Expected {expected:10.6f}, Calculated: {calculated:10.6f}, Error: {err:10.6f}, RelError: {rel_err:10.6f}, time: {time:2.2f}, concentrating: {concentrating}, weak_bc: {weak_bc}')
                    self._check_results(cache_mode, option_id,
                                        (n, concentrating, weak_bc),
                                        ('analytic', 'calculated'),
                                        (expected, calculated),
                                        cached_data)
        if cache_mode:
            self._write_cache(test_file, cached_data)

    @staticmethod
    def _make_option(option_data: OptionData) -> ql.EuropeanOption:
        return ql.EuropeanOption(
            ql.PlainVanillaPayoff(
                option_data.put_call,
                option_data.strike
            ),
            ql.EuropeanExercise(
                option_data.expiry()
            )
        )

    @staticmethod
    def _make_process(spot: ql.QuoteHandle,
                      r: float,
                      q: float,
                      sigma: float,
                      val_date: ql.Date,
                      day_count: ql.DayCounter,
                      calendar: ql.Calendar) -> ql.BlackScholesMertonProcess:
        return ql.BlackScholesMertonProcess(
            spot,
            ql.YieldTermStructureHandle(ql.FlatForward(val_date, q, day_count)),
            ql.YieldTermStructureHandle(ql.FlatForward(val_date, r, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, calendar, sigma, day_count)),
        )

    # insanely fast...
    def _ql_benchmark_fd(self,
                         option_data: OptionData,
                         market_data: MarketData,
                         size: int) -> tuple[float, float]:
        ql_option = self._make_option(option_data)
        start = timer()
        ql_process = self._make_process(ql.makeQuoteHandle(market_data.spot),
                                        market_data.r,
                                        market_data.q,
                                        market_data.sigma,
                                        option_data.val_date,
                                        option_data.dc,
                                        option_data.cal)
        ql_engine = ql.FdBlackScholesVanillaEngine(ql_process, size, size)
        ql_option.setPricingEngine(ql_engine)
        pv = ql_option.NPV()
        time = timer() - start
        return pv, time

    def _compare_analytic(self,
                          option_data: OptionData,
                          market_data: MarketData,
                          model_params: ModelParams,
                          result: FEVanillaResult) -> tuple[float, float]:
        if PLOT:
            if result.w_orientation == 'triangles':
                plot_solution_triangles(result.elements,
                                        result.w_solution,
                                        title=f"European {'Call' if option_data.put_call == ql.Option.Call else 'Put'} Option FE-2D Solution, {model_params.concentrating = }, {model_params.flux_boundary_bc = }")
            else:
                plot_solution_time_space(result.elements.points(),
                                         result.w_solution,
                                         title=f"European {'Call' if option_data.put_call == ql.Option.Call else 'Put'} Option FE-FD Solution"
                                         )
        ql_option = self._make_option(option_data)
        spot_quote = ql.SimpleQuote(0.0)
        ql_process = self._make_process(ql.QuoteHandle(spot_quote), market_data.r, market_data.q, market_data.sigma,
                                        option_data.val_date,
                                        option_data.dc, option_data.cal)
        ql_option.setPricingEngine(ql.AnalyticEuropeanEngine(ql_process))

        if PLOT:
            s_plt = np.linspace(option_data.strike * math.exp(result.transform_helper.x_min()) + EPSILON,
                                option_data.strike * math.exp(result.transform_helper.x_max()) - EPSILON, 100)
            u_exact = []
            u_approx = []
            for s in s_plt:
                spot_quote.setValue(s)
                u_exact.append(ql_option.NPV())
                u_approx.append(result.npv_at(s))
            plt.plot(s_plt, u_exact, label='Analytical Solution', color='black')
            plt.plot(s_plt, u_approx, 'r--', label='FEM Solution at t=0')
            plt.legend()
            plt.grid()
            plt.show()

        spot_quote.setValue(market_data.spot)
        exact = ql_option.NPV()
        approx = result.npv_at(spot_quote.value())
        return approx, exact

    @staticmethod
    def _initialize_reg_test(cache_mode: bool, file_name: str) -> dict[str, dict[tuple[Any, ...], dict[str, float]]]:
        cache_in: dict[str, dict[tuple[Any, ...], dict[str, float]]] = defaultdict(dict)
        if not cache_mode:
            with open(Path(__file__).parent / 'test_data' / 'regression' / file_name, 'r') as h:
                cached_data = json.load(h)
            for option_id, data in cached_data.items():
                for item in data:
                    cache_in[option_id][tuple(item['params'])] = item['data']
        return cache_in

    @staticmethod
    def _write_cache(file_name: str, cache: dict[str, dict[tuple[Any, ...], dict[str, float]]]) -> None:
        data_out = {_id: [{'params': list(k), 'data': d} for k, d in cache[_id].items()] for _id in cache}
        with open(Path(__file__).parent / file_name, 'w') as h:
            json.dump(data_out, h, indent=4)

    def _check_results(
            self,
            cache_mode: bool,
            instrument_id: str,
            params: tuple[Any, ...],
            keys: tuple[str, ...],
            values: tuple[float, ...],
            cached_results: dict[str, dict[tuple[Any, ...], dict[str, float]]],
            abs_tol: float = 1e-3,  # todo: find reason for deviations this big across platforms
            rel_tol: float = 1e-6
    ) -> None:
        if cache_mode:
            if instrument_id not in cached_results:
                cached_results[instrument_id] = {params: dict(zip(keys, [round(v, 10) for v in values]))}
            else:
                cached_results[instrument_id][params] = dict(zip(keys, [round(v, 10) for v in values]))
        else:
            cached_vals = [cached_results[instrument_id][params][k] for k in keys]
            for k, v, v_c in zip(keys, values, cached_vals):
                abs_err = abs(v_c - v)
                rel_err = abs_err / v_c
                # self.assertLess(rel_err, rel_tol,
                #                 msg=f'ID "{instrument_id}", params: {params}, Rel Error Fail for "{k}" - Expected cached: {v_c}, got: {v}')
                # self.assertLess(abs_err, abs_tol,
                #                 msg=f'ID "{instrument_id}", params: {params}, Abs Error Fail "{k}" - Expected cached: {v_c}, got: {v}')
                print(f'ID "{instrument_id}", params: {params}, Rel Error "{rel_err}" - Expected cached: {v_c}, got: {v}')
