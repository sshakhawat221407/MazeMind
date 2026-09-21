"""Feature engineering for the learned A* heuristic.

The neural heuristic has to answer "how far is cell *c* from the goal, really?"
using only information a heuristic is *allowed* to look at - cheap, local
structure - never the answer itself. The sixteen features below are all
computable in closed form from the wall grid and the terrain grid.

Speed matters here in a way it usually doesn't for ML features: the heuristic is
called once per node expansion inside A*'s inner loop. Rather than pay a
Python-level function call and a matrix multiply per node, every feature is
computed for **all cells at once** with numpy, and the network is evaluated in a
single batched forward pass before the search starts. The heuristic then becomes
an O(1) array lookup. That is what makes a learned heuristic practical at 60 FPS.

The two structurally interesting features:

``blocked_ray``
    Walls crossed by the L-shaped ray from the cell to the goal (horizontal leg
    then vertical leg), divided by the Manhattan distance. This is the single
    strongest signal for "the direct line lies, you'll have to go around".

``box_wall_density``
    Mean wall count per cell inside the bounding box spanned by the cell and the
    goal. Computed for every cell in O(1) via an integral image, despite each
    cell having a differently sized box.
"""

from __future__ import annotations

import numpy as np

from ..config import COST_MUD, N_FEATURES
from ..core.maze import Cell, Maze

# Bits set per 4-bit wall mask, so popcount is a table lookup.
_POPCOUNT = np.array([bin(i).count("1") for i in range(16)], dtype=np.float32)

# Rough upper bound used to normalise maze area into ~[0, 1].
_AREA_SCALE = 2048.0


def _integral(arr: np.ndarray) -> np.ndarray:
    """2-D prefix sum padded with a zero row and column.

    ``I[y, x]`` holds the sum of ``arr[:y, :x]``, so any axis-aligned box sum is
    four lookups regardless of its size.
    """
    out = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)
    return out


def _box_sum(integral: np.ndarray, y0, y1, x0, x1) -> np.ndarray:
    """Inclusive box sum ``arr[y0:y1+1, x0:x1+1]`` for arrays of corners."""
    return (
        integral[y1 + 1, x1 + 1]
        - integral[y0, x1 + 1]
        - integral[y1 + 1, x0]
        + integral[y0, x0]
    )


def _window_mean(arr: np.ndarray, integral: np.ndarray, k: int) -> np.ndarray:
    """Mean of ``arr`` over a k x k window centred on each cell, edges clamped."""
    h, w = arr.shape
    r = k // 2
    ys, xs = np.mgrid[0:h, 0:w]
    y0 = np.clip(ys - r, 0, h - 1)
    y1 = np.clip(ys + r, 0, h - 1)
    x0 = np.clip(xs - r, 0, w - 1)
    x1 = np.clip(xs + r, 0, w - 1)
    counts = (y1 - y0 + 1) * (x1 - x0 + 1)
    return (_box_sum(integral, y0, y1, x0, x1) / counts).astype(np.float32)


def feature_grid(maze: Maze, goal: Cell) -> np.ndarray:
    """Features for every cell relative to ``goal``.

    Returns an array of shape ``(height * width, N_FEATURES)`` in row-major
    order, so cell ``(x, y)`` maps to row ``y * width + x``.
    """
    h, w = maze.height, maze.width
    gx, gy = goal
    walls = maze.walls
    cost = maze.cost.astype(np.float32)

    ys, xs = np.mgrid[0:h, 0:w]
    dx = (gx - xs).astype(np.float32)
    dy = (gy - ys).astype(np.float32)
    adx = np.abs(dx)
    ady = np.abs(dy)

    # --- geometric distance family (normalised to be scale-free) ----------
    f_dx_signed = dx / w
    f_dy_signed = dy / h
    f_dx_abs = adx / w
    f_dy_abs = ady / h
    manhattan_steps = adx + ady
    f_manhattan = manhattan_steps / (w + h)
    f_euclidean = np.sqrt(dx * dx + dy * dy) / np.sqrt(w * w + h * h)
    f_chebyshev = np.maximum(adx, ady) / max(w, h)
    # How diagonal the offset is. A purely axis-aligned offset has far fewer
    # distinct shortest routes than a diagonal one, so it detours more often.
    f_axis_skew = (np.minimum(adx, ady) / np.maximum(manhattan_steps, 1.0)).astype(np.float32)

    # --- local wall texture ----------------------------------------------
    wall_count = _POPCOUNT[walls]  # 0..4 walls per cell
    wall_integral = _integral(wall_count)
    f_w3 = _window_mean(wall_count, wall_integral, 3) / 4.0
    f_w7 = _window_mean(wall_count, wall_integral, 7) / 4.0
    f_degree = (4.0 - wall_count) / 4.0

    # --- walls crossed by the straight-ish ray to the goal ----------------
    # East bit (2) of cell (x, y) is the wall between x and x+1 on row y.
    east = ((walls >> 1) & 1).astype(np.float64)
    south = ((walls >> 2) & 1).astype(np.float64)
    # Row-wise prefix: ce[y, i] = number of east walls in columns [0, i).
    ce = np.zeros((h, w + 1), dtype=np.float64)
    ce[:, 1:] = east.cumsum(axis=1)
    # Column-wise prefix: cs[i, x] = number of south walls in rows [0, i).
    cs = np.zeros((h + 1, w), dtype=np.float64)
    cs[1:, :] = south.cumsum(axis=0)

    x_lo = np.minimum(xs, gx)
    x_hi = np.maximum(xs, gx)
    y_lo = np.minimum(ys, gy)
    y_hi = np.maximum(ys, gy)
    # Horizontal leg travelled on the cell's own row, vertical leg on the
    # goal's column - together an L-shaped ray from cell to goal.
    horiz = ce[ys, x_hi] - ce[ys, x_lo]
    vert = cs[y_hi, gx] - cs[y_lo, gx]
    crossed = horiz + vert
    f_ray = (crossed / np.maximum(manhattan_steps, 1.0)).astype(np.float32)
    # The same quantity un-normalised (scaled by maze size instead): a cell can
    # have a low walls-per-step ratio yet still sit behind a great many walls.
    f_ray_abs = (crossed / float(w + h)).astype(np.float32)

    # --- bounding-box statistics -----------------------------------------
    box_walls = _box_sum(wall_integral, y_lo, y_hi, x_lo, x_hi)
    box_cells = (y_hi - y_lo + 1) * (x_hi - x_lo + 1)
    f_box_walls = (box_walls / box_cells / 4.0).astype(np.float32)
    f_box_area = (box_cells / float(w * h)).astype(np.float32)

    # --- terrain ----------------------------------------------------------
    cost_integral = _integral(cost)
    f_box_cost = (_box_sum(cost_integral, y_lo, y_hi, x_lo, x_hi)
                  / box_cells / COST_MUD).astype(np.float32)
    f_cell_cost = (cost / COST_MUD).astype(np.float32)

    # --- global maze character -------------------------------------------
    # Constant across the grid, so they contribute nothing to the *ordering* of
    # cells within one search - but they calibrate how twisty this maze is
    # overall, which is what sets the scale of the detour ratio. Without them
    # the model cannot tell a loopy braided maze from a perfect one.
    degree_grid = 4.0 - wall_count
    f_maze_area = np.full((h, w), (w * h) / _AREA_SCALE, dtype=np.float32)
    f_mean_degree = np.full((h, w), degree_grid.mean() / 4.0, dtype=np.float32)
    f_dead_ends = np.full((h, w), float((degree_grid <= 1).mean()), dtype=np.float32)
    # How boxed in the goal itself is: a goal down a long dead-end corridor
    # inflates the detour ratio for every cell in the maze.
    goal_walls = _window_mean(wall_count, wall_integral, 5)[gy, gx] / 4.0
    f_goal_enclosure = np.full((h, w), float(goal_walls), dtype=np.float32)

    stack = np.stack(
        [
            f_dx_signed, f_dy_signed, f_dx_abs, f_dy_abs,
            f_manhattan, f_euclidean, f_chebyshev, f_axis_skew,
            f_w3, f_w7, f_degree,
            f_ray, f_ray_abs, f_box_walls, f_box_area,
            f_box_cost, f_cell_cost,
            f_maze_area, f_mean_degree, f_dead_ends, f_goal_enclosure,
        ],
        axis=-1,
    ).astype(np.float32)

    assert stack.shape == (h, w, N_FEATURES), stack.shape
    return stack.reshape(h * w, N_FEATURES)


def features_for_cell(maze: Maze, cell: Cell, goal: Cell) -> np.ndarray:
    """Single-cell features. Convenient for tests; the grid form is faster."""
    grid = feature_grid(maze, goal)
    return grid[cell[1] * maze.width + cell[0]]


def manhattan_grid(maze: Maze, goal: Cell) -> np.ndarray:
    """Manhattan distance from every cell to ``goal``, shape ``(h, w)``."""
    ys, xs = np.mgrid[0:maze.height, 0:maze.width]
    return (np.abs(xs - goal[0]) + np.abs(ys - goal[1])).astype(np.float64)


def distance_scale(maze: Maze) -> float:
    """Legacy normaliser, kept for the ASCII/debug paths."""
    return float(maze.width + maze.height)
