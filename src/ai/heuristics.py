"""Heuristic functions for informed search.

A heuristic estimates the remaining cost from a cell to the goal. It is
*admissible* when it never overestimates that cost, which is the condition A*
needs to guarantee an optimal path.

The three classical heuristics here are admissible by construction on this grid,
because the cheapest possible move costs exactly 1 (see ``config.COST_FLOOR``)
and movement is 4-directional. The learned heuristic is admissible only
*empirically*, and :mod:`src.ai.train` measures exactly how often it isn't.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Protocol

import numpy as np

from ..config import HEURISTIC_WEIGHTS
from ..core.maze import Cell, Maze
from .features import feature_grid, manhattan_grid
from .nn import MLP


class Heuristic(Protocol):
    """Any callable of ``(maze, cell, goal) -> estimated remaining cost``."""

    def __call__(self, maze: Maze, cell: Cell, goal: Cell) -> float: ...


# ---------------------------------------------------------------------------
# Classical heuristics
# ---------------------------------------------------------------------------
def zero(maze: Maze, cell: Cell, goal: Cell) -> float:
    """h = 0. Turns A* into Dijkstra; the baseline for "no information"."""
    return 0.0


def manhattan(maze: Maze, cell: Cell, goal: Cell) -> float:
    """Grid distance ignoring walls. The standard admissible choice here."""
    return float(abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))


def euclidean(maze: Maze, cell: Cell, goal: Cell) -> float:
    """Straight-line distance. Admissible but weaker than Manhattan on a
    4-connected grid, since diagonal shortcuts do not actually exist."""
    dx = cell[0] - goal[0]
    dy = cell[1] - goal[1]
    return math.sqrt(dx * dx + dy * dy)


def chebyshev(maze: Maze, cell: Cell, goal: Cell) -> float:
    """max(|dx|, |dy|). Weakest of the three; included for comparison."""
    return float(max(abs(cell[0] - goal[0]), abs(cell[1] - goal[1])))


# ---------------------------------------------------------------------------
# Learned heuristic
# ---------------------------------------------------------------------------
class LearnedHeuristic:
    """Neural-network heuristic with whole-grid batched inference.

    The network does not predict distance directly. It predicts the **detour
    ratio** ``h* / manhattan``, and the heuristic is then::

        h(c) = manhattan(c, goal) * (1 + (ratio_pred - 1) * alpha)

    Two properties fall out of that shape. The network's output layer is
    ``1 + softplus(.)``, so ``ratio_pred >= 1`` always, and the shrink is
    anchored at 1 rather than 0 - together these mean the learned heuristic is
    **never less informed than Manhattan**, whatever the network does. And
    ``alpha``, calibrated on held-out mazes, pulls the estimate back toward
    Manhattan until overestimates are rare, which is what keeps A* returning
    optimal paths.

    Calling a network once per node expansion would make A* slower than the
    search it is meant to speed up. Instead, the first call for a given
    ``(maze, goal)`` pair evaluates every cell in one matrix multiply and caches
    the resulting grid; subsequent calls are array lookups.
    """

    def __init__(self, model: MLP, safety: bool = True, name: str = "A*-learned"):
        self.model = model
        self.safety = safety
        self.name = name
        self._maze: Maze | None = None
        self._goal: Cell | None = None
        self._grid: np.ndarray | None = None
        self.last_prepare_ms = 0.0

    @classmethod
    def load(
        cls, path: str = HEURISTIC_WEIGHTS, safety: bool = True
    ) -> "LearnedHeuristic | None":
        """Load trained weights, or return ``None`` if training hasn't run."""
        if not os.path.exists(path):
            return None
        try:
            return cls(MLP.load(path), safety=safety)
        except Exception:  # corrupt or stale weights - fall back gracefully
            return None

    def _prepare(self, maze: Maze, goal: Cell) -> None:
        import time

        t0 = time.perf_counter()
        ratio = self.model.predict(feature_grid(maze, goal))
        if self.safety:
            # Shrink toward Manhattan by the factor calibrated on held-out
            # mazes. Anchored at 1, so this can only ever move the estimate
            # closer to Manhattan - never below it.
            ratio = 1.0 + (ratio - 1.0) * self.model.alpha
        ratio = np.maximum(ratio, 1.0)
        grid = manhattan_grid(maze, goal).ravel() * ratio
        grid[goal[1] * maze.width + goal[0]] = 0.0
        self._grid = grid
        self._maze = maze
        self._goal = goal
        self.last_prepare_ms = (time.perf_counter() - t0) * 1000.0

    def __call__(self, maze: Maze, cell: Cell, goal: Cell) -> float:
        if maze is not self._maze or goal != self._goal:
            self._prepare(maze, goal)
        assert self._grid is not None
        return float(self._grid[cell[1] * maze.width + cell[0]])

    def grid_for(self, maze: Maze, goal: Cell) -> np.ndarray:
        """Expose the (h, w) heuristic field so the UI can draw it as a map."""
        if maze is not self._maze or goal != self._goal:
            self._prepare(maze, goal)
        assert self._grid is not None
        return self._grid.reshape(maze.height, maze.width)


CLASSICAL: dict[str, Callable[..., float]] = {
    "zero": zero,
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
}

DISPLAY_NAMES = {
    "zero": "None (h=0)",
    "manhattan": "Manhattan",
    "euclidean": "Euclidean",
    "chebyshev": "Chebyshev",
    "learned": "Learned (NN)",
}


def get_heuristic(name: str) -> Heuristic:
    """Look up a heuristic by name; ``'learned'`` falls back to Manhattan."""
    if name == "learned":
        learned = LearnedHeuristic.load()
        return learned if learned is not None else manhattan
    return CLASSICAL.get(name, manhattan)


# Re-exported so UI code can pull both heat maps from one module.
__all__ = [
    "Heuristic", "zero", "manhattan", "euclidean", "chebyshev",
    "LearnedHeuristic", "CLASSICAL", "DISPLAY_NAMES", "get_heuristic",
    "manhattan_grid",
]
