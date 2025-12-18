from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri

from typing import TYPE_CHECKING

from finite_elements.boundary import RectangleHelper

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from finite_elements.elements import LinearTriangles


def plot_solution_triangles(elements: LinearTriangles, u: NDArray[np.float64], title: str) -> None:
    points = elements.points()
    triangles = elements.triangles()
    x = points[:, 0]
    y = points[:, 1]
    triangles = mtri.Triangulation(x, y, triangles)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(triangles, u, cmap='viridis', edgecolor='k', linewidth=0.3, alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u')
    ax.set_xlabel('tau')
    ax.set_ylabel('log(S/K)')
    ax.set_zlabel('w')
    ax.set_title(title)
    plt.show()


def plot_solution_time_space(points: NDArray[np.float64], u: NDArray[np.float64], title: str) -> None:
    rectangle_h = RectangleHelper(points)
    time = points[rectangle_h.y_min(), 0]
    space = points[rectangle_h.x_min(), 1]
    tau, log_s_k = np.meshgrid(time, space, indexing='ij')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(tau, log_s_k, u, linewidth=0.0, antialiased=True, alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="u")
    ax.set_xlabel('tau')
    ax.set_ylabel('log(S/K)')
    ax.set_zlabel('w')
    ax.set_title(title)
    plt.show()
