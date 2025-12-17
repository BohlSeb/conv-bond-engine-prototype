from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np
import QuantLib as ql
import scipy.sparse.linalg as spl

from finite_elements.assembler import LinTriangleAssembler, LinIntervalAssembler
from finite_elements.boundary import RectangleHelper, merge_dirichlet_last_wins, apply_dirichlet_sparse
from finite_elements.elements import LinearTriangles, LinearIntervals
from finite_elements.triangulation import DelaunayMesh2D
from option.european import BSTransformHelper, EuropeanOptionBCs
from finite_elements.interval import ConcentratingInterval
from time_stepping.finite_differences import step_theta

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from option.european import OptionData, MarketData, ModelParams


@dataclass
class FEVanillaResult:
    w_solution: NDArray[np.float64]
    w_orientation: str
    strike: float
    rectangle_helper: RectangleHelper
    elements: LinearTriangles

    # debug
    transform_helper: BSTransformHelper

    def npv_at(self, spot: float) -> float:
        x = np.log(spot / self.strike)

        t_max = self.rectangle_helper.x_max()
        xs = self.elements.points()[t_max, 1]
        if self.w_orientation == 'triangles':
            ys = self.w_solution[t_max]
        elif self.w_orientation == 'time_space':
            ys = self.w_solution[-1]
        else:
            raise ValueError('Unknown orientation')

        npv = np.interp(x, xs, ys) * self.strike
        return npv


def basic_grid(transform_h: BSTransformHelper, time2mat: float, model_params: ModelParams) -> DelaunayMesh2D:
    i_x = np.linspace(0.0, time2mat, model_params.size)
    if model_params.concentrating:
        beta = model_params.size * 1e-4  # TODO: Find good concentrating beta
        i_y = np.array(ConcentratingInterval(transform_h.x_min(),
                                             transform_h.x_max(),
                                             model_params.size,
                                             0.0,  # concentrating on S = K
                                             beta).grid())
    else:
        i_y = np.linspace(transform_h.x_min(), transform_h.x_max(), model_params.size)
    return DelaunayMesh2D(i_x, i_y)


def european_vanilla_fe_2d(option_data: OptionData,
                           market_data: MarketData,
                           model_params: ModelParams) -> FEVanillaResult:
    transform_h = BSTransformHelper(market_data, option_data, model_params)
    grid = basic_grid(transform_h, option_data.time2maturity(), model_params)

    elements = LinearTriangles(grid.points(), grid.triangles(), grid.areas())

    assembler = LinTriangleAssembler(elements)
    lhs = assembler.assemble_stiffness(diffusion_tensor=transform_h.diffusion())
    lhs += assembler.assemble_convection(weight_x=lambda _x, _y: 1.0, weight_y=lambda _x, _y: transform_h.beta_y())
    lhs += transform_h.mass() * assembler.assemble_mass()
    if model_params.use_supg:
        supg = assembler.assemble_supg(lambda _x, _y: 1.0, lambda _x, _y: transform_h.beta_y(), transform_h.diffusion())
        lhs += model_params.supg_scale * supg
    rhs = np.zeros_like(elements.points()[:, 0])

    robin_bcs = []
    dirichlet_bcs = []
    rectangle_h = RectangleHelper(elements.points())
    bc_helper = EuropeanOptionBCs(market_data, transform_h, rectangle_h, elements.points())

    if model_params.flux_boundary_bc:
        robin_bcs.append(bc_helper.bc_nonzero_weak(option_data.put_call))
        robin_bcs.append(bc_helper.bc_outflow_neumann0())
    else:
        dirichlet_bcs.append(bc_helper.bc_nonzero_strong(option_data.put_call).data())
        dirichlet_bcs.append(bc_helper.bc_zero_strong(option_data.put_call).data())
    dirichlet_bcs.append(bc_helper.bc_maturity(option_data.put_call).data())

    for robin_bc in robin_bcs:
        robin_bc.apply(lhs, rhs)
    dirichlet_data = merge_dirichlet_last_wins(dirichlet_bcs)
    lhs, rhs = apply_dirichlet_sparse(lhs, rhs, dirichlet_data)
    w = spl.spsolve(lhs, rhs)
    result = FEVanillaResult(w_solution=w,
                             w_orientation='triangles',
                             strike=option_data.strike,
                             rectangle_helper=rectangle_h,
                             elements=elements,
                             transform_helper=transform_h)
    return result


def european_vanilla_fe_fd(option_data: OptionData,
                           market_data: MarketData,
                           model_params: ModelParams,
                           theta: float = 1.0) -> FEVanillaResult:
    assert model_params.use_supg == False and model_params.flux_boundary_bc == False
    transform_h = BSTransformHelper(market_data, option_data, model_params)
    grid = basic_grid(transform_h, option_data.time2maturity(), model_params)

    rectangle_h = RectangleHelper(grid.points())

    s_grid = grid.points()[rectangle_h.x_min(), 1]
    elements = LinearIntervals(s_grid)
    assembler = LinIntervalAssembler(elements)

    kappa = transform_h.diffusion()[1][1]
    lhs = assembler.assemble_stiffness(weight=lambda _: kappa)
    lhs += assembler.assemble_convection(lambda _: transform_h.beta_y())
    lhs += transform_h.mass() * assembler.assemble_mass()

    time_mass = assembler.assemble_mass()
    time_steps = np.diff(grid.points()[rectangle_h.y_min(), 0])

    bc_helper = EuropeanOptionBCs(market_data, transform_h, rectangle_h, grid.points())

    bc_non_zero = bc_helper.bc_nonzero_strong(option_data.put_call)
    bc_zero = bc_helper.bc_zero_strong(option_data.put_call)

    if option_data.put_call == ql.Option.Call:
        bc_left = bc_zero.as_data_1d('left')
        bc_right = bc_non_zero.as_data_1d('right')
    else:
        bc_left = bc_non_zero.as_data_1d('left')
        bc_right = bc_zero.as_data_1d('right')
    assert len(bc_left) == len(bc_right) == (len(time_steps) + 1)

    w = np.zeros((len(time_steps) + 1, s_grid.size))

    bc_left = bc_left[1:]
    bc_right = bc_right[1:]
    w_n = bc_helper.bc_maturity(option_data.put_call).data()[1]
    w[0] = w_n

    for i, (dt, bc_l, bc_r) in enumerate(zip(time_steps, bc_left, bc_right)):
        lhs_dt, rhs_dt = step_theta(time_mass, dt, lhs, w_n, theta)
        bc_data = merge_dirichlet_last_wins([bc_l, bc_r], skip_overwrite=True)
        lhs_bc, rhs_bc = apply_dirichlet_sparse(lhs_dt, rhs_dt, bc_data)
        w_n = spl.spsolve(lhs_bc, rhs_bc)  # todo: can use band solver
        w[i + 1] = w_n

    plot_elems = LinearTriangles(grid.points(), grid.triangles(), grid.areas())
    result = FEVanillaResult(w_solution=w,
                             w_orientation='time_space',
                             strike=option_data.strike,
                             rectangle_helper=rectangle_h,
                             elements=plot_elems,
                             transform_helper=transform_h)
    return result
