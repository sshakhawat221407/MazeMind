"""Maze generation.

Three classical carving algorithms are implemented from scratch, each with a
different structural signature:

* **Recursive backtracker** - a randomised DFS. Produces long, winding corridors
  and few junctions. This is the layout where an uninformed DFS solver looks
  deceptively good and where Greedy Best-First gets badly trapped.
* **Randomised Prim** - grows a single region outward. Produces many short
  branches and a high junction count, which punishes DFS heavily.
* **Randomised Kruskal** - merges disjoint sets via a union-find structure.
  Produces the most "uniform" texture of the three.

After carving, two post-processing passes shape the gameplay:

* **Braiding** removes a fraction of dead ends, introducing loops. A perfect
  maze has exactly one path between any two cells, which makes racing dull and
  makes every search algorithm return the same path. Loops restore genuine
  choice and make the solvers measurably different from each other.
* **Terrain** sprinkles mud patches, giving cells a movement cost above 1 so
  that Dijkstra and A* diverge from plain BFS.
"""

from __future__ import annotations

import random
from typing import Callable

from ..config import (
    BRAID_FACTOR,
    COST_FLOOR,
    COST_MUD,
    DELTA,
    DIRECTIONS,
    MUD_FRACTION,
    OPPOSITE,
)
from .maze import Cell, Maze

Rng = random.Random


# ---------------------------------------------------------------------------
# Carving algorithms
# ---------------------------------------------------------------------------
def recursive_backtracker(maze: Maze, rng: Rng) -> None:
    """Randomised depth-first carving with an explicit stack."""
    start = (rng.randrange(maze.width), rng.randrange(maze.height))
    visited = {start}
    stack = [start]
    while stack:
        x, y = stack[-1]
        options = []
        for d in DIRECTIONS:
            dx, dy = DELTA[d]
            nb = (x + dx, y + dy)
            if maze.in_bounds(nb) and nb not in visited:
                options.append((d, nb))
        if not options:
            stack.pop()
            continue
        direction, nxt = rng.choice(options)
        maze.carve((x, y), direction)
        visited.add(nxt)
        stack.append(nxt)


def randomised_prim(maze: Maze, rng: Rng) -> None:
    """Grow a spanning tree by repeatedly carving a random frontier edge."""
    start = (rng.randrange(maze.width), rng.randrange(maze.height))
    inside = {start}
    # Frontier holds (cell_inside, direction) edges pointing outward.
    frontier: list[tuple[Cell, int]] = []

    def push(cell: Cell) -> None:
        x, y = cell
        for d in DIRECTIONS:
            dx, dy = DELTA[d]
            nb = (x + dx, y + dy)
            if maze.in_bounds(nb) and nb not in inside:
                frontier.append((cell, d))

    push(start)
    while frontier:
        idx = rng.randrange(len(frontier))
        cell, direction = frontier.pop(idx)
        dx, dy = DELTA[direction]
        nxt = (cell[0] + dx, cell[1] + dy)
        if nxt in inside:
            continue
        maze.carve(cell, direction)
        inside.add(nxt)
        push(nxt)


def randomised_kruskal(maze: Maze, rng: Rng) -> None:
    """Union-find over every interior wall, shuffled."""
    parent: dict[Cell, Cell] = {c: c for c in maze.cells()}

    def find(c: Cell) -> Cell:
        root = c
        while parent[root] != root:
            root = parent[root]
        while parent[c] != root:  # path compression
            parent[c], c = root, parent[c]
        return root

    edges: list[tuple[Cell, int]] = []
    for cell in maze.cells():
        x, y = cell
        # Only consider E and S to visit each shared wall exactly once.
        for d in (2, 4):  # E, S
            dx, dy = DELTA[d]
            if maze.in_bounds((x + dx, y + dy)):
                edges.append((cell, d))
    rng.shuffle(edges)

    for cell, direction in edges:
        dx, dy = DELTA[direction]
        nxt = (cell[0] + dx, cell[1] + dy)
        ra, rb = find(cell), find(nxt)
        if ra != rb:
            parent[ra] = rb
            maze.carve(cell, direction)


CARVERS: dict[str, Callable[[Maze, Rng], None]] = {
    "backtracker": recursive_backtracker,
    "prim": randomised_prim,
    "kruskal": randomised_kruskal,
}


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def braid(maze: Maze, rng: Rng, factor: float = BRAID_FACTOR) -> int:
    """Remove a fraction of dead ends by carving one extra opening each.

    Returns the number of dead ends removed.
    """
    if factor <= 0:
        return 0
    ends = maze.dead_ends()
    rng.shuffle(ends)
    target = int(len(ends) * factor)
    removed = 0
    for cell in ends:
        if removed >= target:
            break
        if maze.degree(cell) != 1:
            continue  # an earlier carve already opened this one up
        x, y = cell
        walled = []
        for d in DIRECTIONS:
            if not (maze.walls[y, x] & d):
                continue
            dx, dy = DELTA[d]
            if maze.in_bounds((x + dx, y + dy)):
                walled.append(d)
        if not walled:
            continue
        # Prefer opening toward another dead end so loops stay interesting.
        walled.sort(key=lambda d: maze.degree((x + DELTA[d][0], y + DELTA[d][1])))
        maze.carve(cell, walled[0])
        removed += 1
    return removed


def sprinkle_terrain(maze: Maze, rng: Rng, fraction: float = MUD_FRACTION) -> None:
    """Grow small mud blobs so costly terrain forms patches, not noise."""
    if fraction <= 0:
        return
    target = int(maze.area * fraction)
    placed = 0
    guard = 0
    while placed < target and guard < target * 40:
        guard += 1
        seed = (rng.randrange(maze.width), rng.randrange(maze.height))
        blob = rng.randint(2, 6)
        frontier = [seed]
        while frontier and placed < target and blob > 0:
            cell = frontier.pop(rng.randrange(len(frontier)))
            if maze.is_mud(cell):
                continue
            maze.set_cost(cell, COST_MUD)
            placed += 1
            blob -= 1
            frontier.extend(maze.neighbours(cell))


def clear_terrain(maze: Maze, cells) -> None:
    """Force specific cells back to normal floor (used for spawns/goal)."""
    for cell in cells:
        maze.set_cost(cell, COST_FLOOR)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate(
    width: int,
    height: int,
    algorithm: str = "backtracker",
    braid_factor: float = BRAID_FACTOR,
    mud_fraction: float = MUD_FRACTION,
    seed: int | None = None,
) -> Maze:
    """Build a fully connected maze with loops and terrain."""
    rng = random.Random(seed)
    maze = Maze(width, height)
    carver = CARVERS.get(algorithm)
    if carver is None:
        raise ValueError(f"unknown generation algorithm: {algorithm!r}")
    carver(maze, rng)
    braid(maze, rng, braid_factor)
    sprinkle_terrain(maze, rng, mud_fraction)
    # Every carver above builds a spanning tree, so connectivity is guaranteed;
    # assert it anyway because a silent disconnect would break every solver.
    assert maze.is_fully_connected(), "generator produced a disconnected maze"
    return maze


def random_algorithm(rng: Rng | None = None) -> str:
    rng = rng or random.Random()
    return rng.choice(list(CARVERS))
