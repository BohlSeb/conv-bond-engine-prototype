from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse.linalg as spl

from finite_elements.assembler import LinTriangleAssembler
from finite_elements.boundary import RectangleHelper, merge_dirichlet_last_wins, apply_dirichlet_sparse
from finite_elements.elements import LinearTriangles
from finite_elements.triangulation import DelaunayMesh2D
from option.european import BSTransformHelper, EuropeanOptionBCs
from finite_elements.interval import ConcentratingInterval

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from option.european import OptionData, MarketData, ModelParams


@dataclass
class FEVanillaResult:
    w_solution: NDArray[np.float64]
    strike: float
    rectangle_helper: RectangleHelper
    elements: LinearTriangles

    # debug
    transform_helper: BSTransformHelper

    def npv(self, spot: float) -> float:
        x = np.log(spot / self.strike)

        t_max = self.rectangle_helper.x_max()
        xs = self.elements.points()[t_max][:, 1]
        ys = self.w_solution[t_max]
        order = np.argsort(xs)

        npv = np.interp(x, xs[order], ys[order]) * self.strike
        return npv


def european_vanilla_fe2d(option_data: OptionData,
                          market_data: MarketData,
                          model_params: ModelParams) -> FEVanillaResult:
    transform_h = BSTransformHelper(market_data, option_data, std_devs=model_params.std_devs)

    i_x = np.linspace(0.0, option_data.time2maturity(), model_params.size)

    if model_params.concentrating:
        beta = model_params.size * 1e-4  # TODO: Find good concentrating beta
        i_y = np.array(ConcentratingInterval(transform_h.x_min(),
                                             transform_h.x_max(),
                                             model_params.size,
                                             0.0,  # concentrating on S = K
                                             beta).grid())
    else:
        i_y = np.linspace(transform_h.x_min(), transform_h.x_max(), model_params.size)

    grid = DelaunayMesh2D(i_x, i_y)
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
                             strike=option_data.strike,
                             rectangle_helper=rectangle_h,
                             elements=elements,
                             transform_helper=transform_h)
    return result
