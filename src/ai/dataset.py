"""Supervised training data for the learned heuristic.

The label we want is ``h*(c)`` - the true cheapest cost from cell ``c`` to the
goal. Computing that per sample would mean one search per sample, which is
absurd. Instead, a single Dijkstra sweep *outward from the goal* labels every
cell in the maze at once, so one maze yields thousands of exact labels for the
price of one search. This is the whole reason the dataset is cheap to build.

Mazes are drawn across a range of sizes, carving algorithms, braid factors and
mud densities so the network sees varied structure and cannot simply memorise
one layout. Targets are divided by ``W + H`` so a model trained on small mazes
transfers to large ones.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from ..config import N_FEATURES, VAL_SPLIT
from ..core.generator import CARVERS, generate
from .features import feature_grid, manhattan_grid

ProgressFn = Callable[[float, str], None]

# Sizes span roughly 12x12 up to 45x29 so the model learns scale-free structure.
_SIZE_POOL = [
    (13, 11), (17, 13), (21, 15), (25, 17),
    (29, 19), (35, 23), (41, 27), (45, 29),
]


def sample_maze(rng: random.Random):
    """A random maze with randomised structure, plus a random goal cell."""
    width, height = rng.choice(_SIZE_POOL)
    maze = generate(
        width,
        height,
        algorithm=rng.choice(list(CARVERS)),
        braid_factor=rng.uniform(0.0, 0.35),
        mud_fraction=rng.uniform(0.0, 0.20),
        seed=rng.randrange(1 << 30),
    )
    goal = (rng.randrange(width), rng.randrange(height))
    return maze, goal


def build_dataset(
    n_mazes: int,
    samples_per_maze: int,
    seed: int = 0,
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, manhattan)``.

    ``X`` has shape ``(N, N_FEATURES)``, ``y`` shape ``(N, 1)`` holding the
    **detour ratio** ``h* / manhattan``, and ``manhattan`` shape ``(N, 1)``
    holding the raw Manhattan distance in cells, so a ratio error can be
    converted back into a real distance error for reporting.

    Cells whose Manhattan distance to the goal is zero (the goal itself) are
    excluded: the ratio is undefined there, and ``h(goal) = 0`` is hard-coded in
    the heuristic anyway.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    manhattans: list[np.ndarray] = []

    for i in range(n_mazes):
        maze, goal = sample_maze(rng)
        # One reverse sweep => exact h* (cost *to* the goal) for every cell.
        dist = maze.cost_to(goal).ravel()
        manh = manhattan_grid(maze, goal).ravel()
        feats = feature_grid(maze, goal)

        valid = np.flatnonzero(np.isfinite(dist) & (manh >= 1.0))
        if valid.size == 0:
            continue
        take = min(samples_per_maze, valid.size)
        picks = np_rng.choice(valid, size=take, replace=False)

        ratio = dist[picks] / manh[picks]
        xs.append(feats[picks])
        ys.append(ratio.reshape(-1, 1))
        manhattans.append(manh[picks].reshape(-1, 1))

        if progress and (i % 10 == 0 or i == n_mazes - 1):
            progress((i + 1) / n_mazes, f"Generating mazes  {i + 1}/{n_mazes}")

    X = np.concatenate(xs, axis=0).astype(np.float64)
    y = np.concatenate(ys, axis=0).astype(np.float64)
    m = np.concatenate(manhattans, axis=0).astype(np.float64)
    assert X.shape[1] == N_FEATURES
    # Manhattan is admissible on this grid (min move cost is 1), so the true
    # ratio can never dip below 1. If it does, something upstream is broken.
    assert y.min() >= 1.0 - 1e-9, f"detour ratio below 1: {y.min()}"
    return X, y, m


def three_way_split(
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    holdout: float = VAL_SPLIT,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Shuffle into train / calibration / test.

    The split is three-way rather than two-way for a specific reason: the
    admissibility margin is *fitted* on the calibration slice, so measuring the
    resulting violation rate on that same slice would be circular. The test
    slice is never touched during training or calibration, so the violation
    rate reported from it is honest.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    X, y, s = X[order], y[order], s[order]

    n = X.shape[0]
    cut_train = int(n * (1.0 - holdout))
    cut_cal = cut_train + (n - cut_train) // 2
    return {
        "train": (X[:cut_train], y[:cut_train], s[:cut_train]),
        "calib": (X[cut_train:cut_cal], y[cut_train:cut_cal], s[cut_train:cut_cal]),
        "test": (X[cut_cal:], y[cut_cal:], s[cut_cal:]),
    }
