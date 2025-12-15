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

from option.european import MarketData, OptionData, ModelParams
from option.european_calculator import fe_european_vanilla

from test.utils import plot_solution

if TYPE_CHECKING:
    from option.european_calculator import FEVanillaResult

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
        params = ModelParams(size=50, concentrating=True, flux_boundary_bc=False)
        result = fe_european_vanilla(option_data, market_data, params)
        res, exact = self._compare_analytic(option_data, market_data, params, result)
        print(
            f'Testing European Call Option: analytic {exact:4.2f}, calculated {res:4.2f}, error: {(res - exact) / exact:2.6f}')

    def test_european_put_option(self) -> None:
        today = self.initialize_today()
        market_data = MarketData(spot=100.0, sigma=0.2, r=0.02, q=0.025)
        option_data = OptionData(val_date=today, period2mat=ql.Period('4Y'), strike=110.0, cal=ql.TARGET(),
                                 dc=ql.Actual365Fixed(), put_call=ql.Option.Put)
        params = ModelParams(size=50, concentrating=True, flux_boundary_bc=False)
        result = fe_european_vanilla(option_data, market_data, params)
        res, exact = self._compare_analytic(option_data, market_data, params, result)
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
                    params = ModelParams(size=n, concentrating=concentrating, flux_boundary_bc=weak_bc)
                    start = timer()
                    result = fe_european_vanilla(option, market_data, params)
                    calculated, expected = self._compare_analytic(option, market_data, params, result)
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
                          model_params: ModelParams,
                          result: FEVanillaResult) -> tuple[float, float]:
        if PLOT:
            plot_solution(result.elements,
                          result.w_solution,
                          title=f"European {'Call' if option_data.put_call == ql.Option.Call else 'Put'} Option FEM Solution, {model_params.concentrating = }, {model_params.flux_boundary_bc = }")
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
            s_plt = np.linspace(option_data.strike * math.exp(result.transform_helper.x_min()) + spot_eps,
                                option_data.strike * math.exp(result.transform_helper.x_max()) - spot_eps, 100)
            u_exact = []
            u_approx = []
            for s in s_plt:
                spot_quote.setValue(s)
                u_exact.append(ql_option.NPV())
                u_approx.append(result.npv(s))
            plt.plot(s_plt, u_exact, label='Analytical Solution', color='black')
            plt.plot(s_plt, u_approx, 'r--', label='FEM Solution at t=0')
            plt.legend()
            plt.grid()
            plt.show()

        spot_quote.setValue(market_data.spot)
        exact = ql_option.NPV()
        approx = result.npv(spot_quote.value())
        return approx, exact
