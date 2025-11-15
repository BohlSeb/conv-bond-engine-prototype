from __future__ import annotations

from matplotlib import pyplot as plt
import matplotlib.tri as mtri

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from finite_elements.elements import LinearTriElements


def plot_solution(elements: LinearTriElements, u: NDArray[np.float64], title: str) -> None:
    points = elements.points()
    triangles = elements.triangles()
    x = points[:, 0]
    y = points[:, 1]
    triangles = mtri.Triangulation(x, y, triangles)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(triangles, u, cmap='viridis', edgecolor='k', linewidth=0.3, alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(title)
    plt.show()