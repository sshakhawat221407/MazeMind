"""Head-to-head benchmarking of the search algorithms.

Two questions this answers, both of which the AI Lab screen plots:

1. How do the five algorithms compare on **work done** (nodes expanded) versus
   **path quality** (cost relative to the true optimum)?
2. Does the learned heuristic actually beat Manhattan, and at what cost to
   optimality?

Every maze is scored against ground truth from a Dijkstra sweep, so
"optimal" here means genuinely optimal, not "same as the other algorithm".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.generator import generate, CARVERS
from ..core.maze import Cell, Maze
from . import search
from .heuristics import LearnedHeuristic, manhattan

ProgressFn = Callable[[float, str], None]


@dataclass
class Contender:
    """One algorithm configuration under test."""

    key: str
    label: str
    algorithm: str
    heuristic: object | None = None
    weight: float = 1.0


@dataclass
class Row:
    """Aggregated results for one contender."""

    key: str
    label: str
    expanded: list[int] = field(default_factory=list)
    cost_ratio: list[float] = field(default_factory=list)
    millis: list[float] = field(default_factory=list)
    frontier: list[int] = field(default_factory=list)
    solved: int = 0
    optimal: int = 0
    n: int = 0

    @property
    def mean_expanded(self) -> float:
        return float(np.mean(self.expanded)) if self.expanded else 0.0

    @property
    def mean_cost_ratio(self) -> float:
        return float(np.mean(self.cost_ratio)) if self.cost_ratio else 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.millis)) if self.millis else 0.0

    @property
    def mean_frontier(self) -> float:
        return float(np.mean(self.frontier)) if self.frontier else 0.0

    @property
    def optimal_rate(self) -> float:
        return self.optimal / self.n if self.n else 0.0


def default_contenders(include_learned: bool = True) -> list[Contender]:
    out = [
        Contender("bfs", "BFS", "bfs"),
        Contender("dfs", "DFS", "dfs"),
        Contender("dijkstra", "Dijkstra", "dijkstra"),
        Contender("greedy", "Greedy", "greedy", manhattan),
        Contender("astar", "A* Manhattan", "astar", manhattan),
    ]
    if include_learned:
        learned = LearnedHeuristic.load()
        if learned is not None:
            out.append(Contender("astar_learned", "A* Learned", "astar", learned))
    return out


def endpoints(maze: Maze, rng: random.Random) -> tuple[Cell, Cell]:
    """Opposite corners, nudged to guarantee a genuinely long route."""
    return (0, 0), (maze.width - 1, maze.height - 1)


def run_benchmark(
    n_mazes: int = 40,
    size: tuple[int, int] = (31, 21),
    contenders: Sequence[Contender] | None = None,
    seed: int = 0,
    progress: ProgressFn | None = None,
) -> dict[str, Row]:
    """Run every contender over ``n_mazes`` freshly generated mazes."""
    contenders = list(contenders or default_contenders())
    rows = {c.key: Row(c.key, c.label) for c in contenders}
    rng = random.Random(seed)
    w, h = size

    for i in range(n_mazes):
        maze = generate(
            w, h,
            algorithm=rng.choice(list(CARVERS)),
            seed=rng.randrange(1 << 30),
        )
        start, goal = endpoints(maze, rng)
        best = search.optimal_cost(maze, start, goal)
        if not np.isfinite(best) or best <= 0:
            continue

        for c in contenders:
            res = search.solve(
                maze, start, goal, c.algorithm,
                heuristic=c.heuristic, weight=c.weight,
            )
            row = rows[c.key]
            row.n += 1
            row.expanded.append(res.nodes_expanded)
            row.millis.append(res.elapsed_ms)
            row.frontier.append(res.max_frontier)
            if res.found:
                row.solved += 1
                row.cost_ratio.append(res.cost / best)
                if abs(res.cost - best) < 1e-6:
                    row.optimal += 1

        if progress:
            progress((i + 1) / n_mazes, f"Benchmarking maze {i + 1}/{n_mazes}")

    return rows


def format_table(rows: dict[str, Row]) -> str:
    """Fixed-width summary table for the terminal."""
    header = (
        f"{'algorithm':<16}{'expanded':>10}{'cost/opt':>10}"
        f"{'optimal':>10}{'frontier':>10}{'time':>10}"
    )
    lines = ["", header, "-" * len(header)]
    for row in rows.values():
        lines.append(
            f"{row.label:<16}"
            f"{row.mean_expanded:>10.1f}"
            f"{row.mean_cost_ratio:>10.3f}"
            f"{row.optimal_rate * 100:>9.0f}%"
            f"{row.mean_frontier:>10.1f}"
            f"{row.mean_ms:>9.2f}ms"
        )
    lines.append("")
    return "\n".join(lines)


def heuristic_showdown(
    n_mazes: int = 30,
    size: tuple[int, int] = (31, 21),
    seed: int = 1,
) -> dict[str, object] | None:
    """Focused Manhattan-vs-learned comparison for the AI Lab screen.

    Returns ``None`` when the network has not been trained yet.
    """
    learned = LearnedHeuristic.load()
    if learned is None:
        return None

    rng = random.Random(seed)
    man_exp: list[int] = []
    lrn_exp: list[int] = []
    man_opt = 0
    lrn_opt = 0
    lrn_ratio: list[float] = []

    for _ in range(n_mazes):
        maze = generate(
            *size,
            algorithm=rng.choice(list(CARVERS)),
            seed=rng.randrange(1 << 30),
        )
        start, goal = endpoints(maze, rng)
        best = search.optimal_cost(maze, start, goal)
        if not np.isfinite(best) or best <= 0:
            continue

        a = search.solve(maze, start, goal, "astar", heuristic=manhattan)
        b = search.solve(maze, start, goal, "astar", heuristic=learned)
        man_exp.append(a.nodes_expanded)
        lrn_exp.append(b.nodes_expanded)
        man_opt += int(abs(a.cost - best) < 1e-6)
        lrn_opt += int(abs(b.cost - best) < 1e-6)
        lrn_ratio.append(b.cost / best)

    n = len(man_exp)
    if n == 0:
        return None
    return {
        "n": n,
        "manhattan_expanded": man_exp,
        "learned_expanded": lrn_exp,
        "manhattan_mean": float(np.mean(man_exp)),
        "learned_mean": float(np.mean(lrn_exp)),
        "reduction_pct": 100.0 * (1.0 - float(np.mean(lrn_exp)) / float(np.mean(man_exp))),
        "manhattan_optimal_rate": man_opt / n,
        "learned_optimal_rate": lrn_opt / n,
        "learned_cost_ratio": float(np.mean(lrn_ratio)),
    }
