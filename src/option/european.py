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
    """
    Black–Scholes space–time transformation helper.

    This helper encodes the standard log-price and time-reversal transform
    used to cast the Black–Scholes PDE into a constant-coefficient
    convection–diffusion–reaction equation suitable for space–time FEM.

    Original Black–Scholes PDE (with dividend yield q):

        ∂V/∂t
        + (r - q) S ∂V/∂S
        + 0.5 σ² S² ∂²V/∂S²
        - r V = 0

    Time reversal:
        τ = T - t

    Log-moneyness variable:
        x = log(S / K)

    Scaled unknown (no discounting applied here):
        w(τ, x) = V(T - τ, K e^x) / K

    After transformation, w(τ, x) satisfies the linear parabolic PDE:

        ∂w/∂τ
        - κ ∂²w/∂x²
        + a ∂w/∂x
        + r w = 0

    where:
        κ = 0.5 σ²
        a = -(r - q - κ)

    This is written in convection–diffusion–reaction form:

        β · ∇w - ∇·(A ∇w) + r w = 0

    on the space–time domain:
        τ ∈ [0, T],   x ∈ [x_min, x_max]

    with coefficients:
        β = (1, a)
        A = [[0, 0],
             [0, κ]]

    Notes
    -----
    • The first coordinate is time-to-maturity τ, the second is log strike price x.
    • Diffusion acts only in the spatial (x) direction.
    • The time derivative is represented as convection with unit velocity.
    • Boundary conditions encode payoff at τ = 0 and asymptotic behavior as x → ±∞.

    Parameters
    ----------
    market_data : MarketData
        Spot, volatility, interest rate r, dividend yield q.
    option_data : OptionData
        Strike, maturity, calendar, day counter.
    std_devs : int
        Half-width of the spatial domain in standard deviations:
            x ∈ [−std_devs σ √T, +std_devs σ √T].
    """

    def __init__(self, market_data: MarketData, option_data: OptionData, std_devs: int = 6) -> None:
        self._kappa = 0.5 * market_data.sigma ** 2
        # diffusion tensor
        # x1 := x := tau (reversed time), x2 := y := log(S)
        self._a = np.array(
            [
                [0.0, 0.0],
                [0.0, self._kappa]
            ]
        )
        # convection beta = (1, beta_y)
        self._beta_y = -1 * (market_data.r - market_data.q - self._kappa)
        self._mass = market_data.r
        self._x_max = std_devs * market_data.sigma * math.sqrt(option_data.time2maturity())
        self._x_min = -self._x_max

    def diffusion(self) -> NDArray[np.float64]:
        return self._a

    def beta(self) -> tuple[float, float]:
        return 1.0, self._beta_y

    def beta_y(self) -> float:
        return self._beta_y

    def kappa(self) -> float:
        return self._kappa

    def mass(self) -> float:
        return self._mass

    def x_min(self) -> float:
        return self._x_min

    def x_max(self) -> float:
        return self._x_max


INFLOW_DIRECTION_TOL = 1e-14


class EuropeanOptionBCs:
    """
       Boundary-condition factory for space–time FEM discretization of
       European vanilla options in log-moneyness coordinates.

       This class constructs all boundary conditions required for the
       transformed Black–Scholes PDE solved on the space–time domain:

           τ ∈ [0, T]      (time-to-maturity)
           x ∈ [x_min, x_max],   x = log(S / K)

       for the scaled unknown:
           w(τ, x) = V(T - τ, K e^x) / K

       The governing PDE is:

           ∂w/∂τ
           - κ ∂²w/∂x²
           + a ∂w/∂x
           + r w = 0

       where:
           κ = 0.5 σ²
           a = -(r - q - κ)

       Boundary Structure
       ------------------
       The rectangular space–time domain has four boundaries:

           τ = 0      : maturity (payoff)
           τ = T      : terminal time (no condition required)
           x = x_min  : small underlying price (S → 0)
           x = x_max  : large underlying price (S → ∞)

       Boundary conditions depend on:
           • option type (call vs put)
           • drift direction (sign of a)
           • whether the boundary is inflow or outflow

       Inflow vs Outflow
       -----------------
       Convection acts in the x-direction with velocity a.

       • If a > 0:
           inflow  at x = x_min
           outflow at x = x_max

       • If a < 0:
           inflow  at x = x_max
           outflow at x = x_min

       Inflow boundaries require stabilization or Dirichlet data.
       Outflow boundaries should *not* be over-constrained.

       Implemented Boundary Conditions
       -------------------------------

       1) Strong Dirichlet (far-field asymptotics)
          ---------------------------------------
          bc_nonzero_strong():
              Call at x → +∞:
                  w ~ e^x e^{-q τ} - e^{-r τ}
              Put at x → −∞:
                  w ~ e^{-r τ}

          bc_zero_strong():
              Complementary boundary where option value → 0.

          These are robust and physically correct, but may induce
          oscillations in convection-dominated regimes.

       2) Weak inflow boundary (Robin / flux-based)
          -----------------------------------------
          bc_nonzero_weak():

          Enforces equality of *normal flux* at inflow:

              F_x(w) = F_x(w_D)

          where the physical flux is:
              F_x(w) = -κ ∂w/∂x + a w

          Since far-field asymptotics w_D(τ) are x-independent:
              ∂w_D/∂x = 0

          leading to the Robin condition:
              -κ ∂w/∂n + a w = a w_D

          This is rewritten in canonical form:
              α w + β ∂w/∂n = g

          with:
              α = a
              β = -κ n_x
              g = a w_D

          This boundary condition is:
              • sign-aware (depends on drift direction)
              • applied only on inflow edges
              • consistent with SUPG and convection-dominated regimes

       3) Outflow Neumann (do-nothing)
          -----------------------------
          bc_outflow_neumann0():

          Applies zero normal derivative:
              ∂w/∂n = 0

          This avoids artificial reflection at outflow boundaries.

       4) Maturity condition
          -------------------
          bc_maturity():

          At τ = 0, enforces the payoff:
              Call: w(0, x) = max(e^x - 1, 0)
              Put : w(0, x) = max(1 - e^x, 0)

       Design Notes
       ------------
       • Boundary selection is centralized and sign-aware.
       • Weak inflow BCs and SUPG stabilization are compatible.
       • Strong BCs are provided as a reference / fallback.
       • No artificial conditions are imposed at τ = T.

       This class is intentionally policy-only: it does not assemble
       matrices itself, but returns boundary-condition objects that
       modify the global system.
       """

    def __init__(self,
                 market_data: MarketData,
                 transform_helper: BSTransformHelper,
                 rectangle_helper: RectangleHelper,
                 points: NDArray[np.float64]) -> None:
        self._params = market_data
        self._transform_h = transform_helper
        self._boundary_h = rectangle_helper
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
        e_x_max = math.exp(self._transform_h.x_max())
        return Scalar(lambda tau, _: alpha * (e_x_max * np.exp(-q * tau) - np.exp(-r * tau)))

    def _w_put_xmin(self, alpha: float) -> Scalar:
        # put as x->-inf: w ~ e^{-r tau}
        r = self._params.r
        return Scalar(lambda tau, _: alpha * np.exp(-r * tau))
