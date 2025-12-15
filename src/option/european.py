from __future__ import annotations

import math
from dataclasses import dataclass

import QuantLib as ql
import numpy as np
from numpy.typing import NDArray

from finite_elements.boundary import RectangleHelper, ConstDirichletBC, ConstRobinBC
from finite_elements.functions import Constant, Scalar


@dataclass
class MarketData:
    spot: float
    sigma: float
    r: float
    q: float


@dataclass
class OptionData:
    val_date: ql.Date
    period2mat: ql.Period
    strike: float
    cal: ql.Calendar
    dc: ql.DayCounter
    put_call: int = ql.Option.Call

    def time2maturity(self) -> float:
        return self.dc.yearFraction(self.val_date, self.expiry())

    def expiry(self) -> ql.Date:
        return self.cal.advance(self.val_date, self.period2mat)


class BSTransformHelper:
    # x = log(S/K)
    # tau  = T - t

    def __init__(self, market_data: MarketData, option_data: OptionData, std_devs: int = 6) -> None:
        self._kappa = 0.5 * market_data.sigma ** 2

        # diffusion tensor
        # x1 := x := tau (reversed time), x2 := y := log(S)
        self.A = np.array(
            [
                [0.0, 0.0],
                [0.0, self._kappa]
            ]
        )
        # convection beta = (1, beta_y)
        self._beta_y = -1 * (market_data.r - market_data.q - self._kappa)
        # right-hand side
        self.mass_weight = market_data.r

        self.x_origin = math.log(market_data.spot / option_data.strike)

        self.x_max_coord = std_devs * market_data.sigma * math.sqrt(option_data.time2maturity())
        self.x_min_coord = -self.x_max_coord

    def beta(self) -> tuple[float, float]:
        return 1.0, self._beta_y

    def beta_y(self) -> float:
        return self._beta_y

    def kappa(self) -> float:
        return self._kappa


INFLOW_DIRECTION_TOL = 1e-14


class EuropeanOptionBCs:

    def __init__(self,
                 market_data: MarketData,
                 bs_helper: BSTransformHelper,
                 bc_helper: RectangleHelper,
                 points: NDArray[np.float64]) -> None:
        self._params = market_data
        self._transform_h = bs_helper
        self._boundary_h = bc_helper
        self._points = points

    def bc_nonzero_strong(self, option_type: int) -> ConstDirichletBC:
        if option_type == ql.Option.Call:
            boundary = self._boundary_h.y_max()
            g = self._w_call_xmax(1.0)
        else:
            boundary = self._boundary_h.y_min()
            g = self._w_put_xmin(1.0)
        return ConstDirichletBC(boundary, self._points, g)

    def bc_zero_strong(self, option_type: int) -> ConstDirichletBC:
        boundary = self._boundary_h.y_min() if option_type == ql.Option.Call else self._boundary_h.y_max()
        return ConstDirichletBC(boundary, self._points, Constant(0.0))

    def bc_nonzero_weak(self, option_type: int) -> ConstRobinBC:
        """
        Sign-aware *inflow flux* Robin on whichever side is inflow.

        Implements (in x-direction):
            F_x(w) = -kappa * w_x + a * w
        and enforces at inflow:
            F_x(w) = F_x(w_D)

        For your far-field asymptotics w_D(tau) has no x-derivative, so:
            -kappa*w_x + a*w = a*w_D

        Rewritten into alpha*w + beta*dw/dn = g with outward normal n_x:
            alpha = a
            beta  = -kappa * n_x
            g     = a * w_D
        """
        a = self._transform_h.beta_y()
        boundary, n_x = self._inflow_edge()
        beta = -self._transform_h.kappa() * n_x

        # pick the asymptotic appropriate for THIS inflow side
        # - call's nonzero asymptotic lives at x_max
        # - put's  nonzero asymptotic lives at x_min
        if option_type == ql.Option.Call:
            # if inflow is at x_max, use nonzero asymptotic; else inflow is at x_min => w_D = 0
            g = self._w_call_xmax(a) if n_x > 0 else Constant(0.0)
        else:
            g = self._w_put_xmin(a) if n_x < 0 else Constant(0.0)

        # ConstRobinBC expects g in alpha*u + beta*du/dn = g
        return ConstRobinBC(
            boundary,
            self._points,
            condition=g,
            dirichlet_alpha=Constant(a),
            neumann_beta=Constant(beta)
        )

    def bc_outflow_neumann0(self) -> ConstRobinBC:
        boundary, _ = self._outflow_edge()
        return ConstRobinBC(
            boundary,
            self._points,
            condition=Constant(0.0),
            dirichlet_alpha=Constant(0.0),
            neumann_beta=Constant(1.0)
        )

    def bc_maturity(self, option_type: int) -> ConstDirichletBC:
        bc = self._boundary_h.x_min()
        if option_type == ql.Option.Call:
            g = Scalar(lambda _, x: np.maximum(np.exp(x) - 1.0, 0.0))
        else:
            g = Scalar(lambda _, x: np.maximum(1.0 - np.exp(x), 0.0))
        return ConstDirichletBC(bc, self._points, g)

    def _inflow_edge(self) -> tuple[NDArray[np.int64], float]:  # return inflow edge and outward normal
        if self._transform_h.beta_y() > INFLOW_DIRECTION_TOL:
            return self._boundary_h.y_min(), -1.0
        return self._boundary_h.y_max(), 1.0

    def _outflow_edge(self) -> tuple[NDArray[np.int64], float]:
        if self._transform_h.beta_y() > INFLOW_DIRECTION_TOL:
            return self._boundary_h.y_max(), 1.0
        return self._boundary_h.y_min(), -1.0

    def _w_call_xmax(self, alpha: float) -> Scalar:
        # scaled: w = V/K, x=log(S/K)
        # call as x->+inf: w ~ exp(x) e^{-q tau} - e^{-r tau}
        r = self._params.r
        q = self._params.q
        e_x_max = math.exp(self._transform_h.x_max_coord)
        return Scalar(lambda tau, _: alpha * (e_x_max * np.exp(-q * tau) - np.exp(-r * tau)))

    def _w_put_xmin(self, alpha: float) -> Scalar:
        # put as x->-inf: w ~ e^{-r tau}
        r = self._params.r
        return Scalar(lambda tau, _: alpha * np.exp(-r * tau))
