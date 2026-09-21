"""Uninformed and informed search algorithms, implemented from scratch.

Every algorithm is written twice-over in a single place: as a **generator** that
yields one ``SearchStep`` per node expansion, and (via :func:`run`) as a plain
function returning a ``SearchResult``. The generator form is what the AI Lab
screen consumes to animate the frontier growing in real time; the plain form is
what the bots and the benchmark harness use.

All five share the same skeleton - pop a node, test the goal, expand
successors - and differ only in *which* node the container hands back next:

===============  ===========================  ==============================
Algorithm        Container / priority         Guarantee
===============  ===========================  ==============================
BFS              FIFO queue                   optimal iff all costs equal
DFS              LIFO stack                   none (any path)
Dijkstra         min-heap on ``g``            optimal for any costs >= 0
Greedy           min-heap on ``h``            none (fast but easily fooled)
A*               min-heap on ``g + w*h``      optimal for w=1 and admissible h
===============  ===========================  ==============================
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Generator, Iterable

from ..core.maze import Cell, Maze, path_cost
from .heuristics import Heuristic, manhattan, zero

# A generator that yields progress and finally *returns* a SearchResult.
SearchGen = Generator["SearchStep", None, "SearchResult"]


@dataclass
class SearchStep:
    """One expansion, handed to the UI so it can draw the search in flight."""

    current: Cell
    frontier: tuple[Cell, ...]
    visited_count: int


@dataclass
class SearchResult:
    """Everything the benchmark and the UI need to compare two algorithms."""

    algorithm: str
    found: bool
    path: list[Cell] = field(default_factory=list)
    cost: float = float("inf")
    nodes_expanded: int = 0
    max_frontier: int = 0
    elapsed_ms: float = 0.0
    visited: set[Cell] = field(default_factory=set)
    expansion_order: list[Cell] = field(default_factory=list)

    @property
    def path_length(self) -> int:
        return len(self.path)

    def summary(self) -> str:
        if not self.found:
            return f"{self.algorithm}: no path"
        return (
            f"{self.algorithm}: cost {self.cost:.0f} | "
            f"expanded {self.nodes_expanded} | {self.elapsed_ms:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def reconstruct(came_from: dict[Cell, Cell], goal: Cell, start: Cell) -> list[Cell]:
    """Walk parent pointers backwards from the goal to the start."""
    path = [goal]
    node = goal
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def run(gen: SearchGen) -> SearchResult:
    """Drain a search generator and return its final result.

    Timing is measured *here* rather than inside the algorithm. A generator is
    suspended between yields, so a clock started inside it measures wall-clock
    time across every pause as well - which for the animated Lab view means it
    reports the length of the animation (hundreds of ms) instead of the
    algorithm's compute time (a fraction of one). Draining in a tight loop, as
    this does, makes elapsed time and compute time the same thing.
    """
    t0 = time.perf_counter()
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        result = stop.value
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result


def _result(
    name: str,
    maze: Maze,
    start: Cell,
    goal: Cell,
    came_from: dict[Cell, Cell],
    found: bool,
    expanded: list[Cell],
    visited: set[Cell],
    max_frontier: int,
    t0: float,
) -> SearchResult:
    elapsed = (time.perf_counter() - t0) * 1000.0
    if not found:
        return SearchResult(
            algorithm=name,
            found=False,
            nodes_expanded=len(expanded),
            max_frontier=max_frontier,
            elapsed_ms=elapsed,
            visited=visited,
            expansion_order=expanded,
        )
    path = reconstruct(came_from, goal, start)
    return SearchResult(
        algorithm=name,
        found=True,
        path=path,
        cost=path_cost(maze, path),
        nodes_expanded=len(expanded),
        max_frontier=max_frontier,
        elapsed_ms=elapsed,
        visited=visited,
        expansion_order=expanded,
    )


# ---------------------------------------------------------------------------
# Uninformed search
# ---------------------------------------------------------------------------
def bfs_iter(maze: Maze, start: Cell, goal: Cell, **_: object) -> SearchGen:
    """Breadth-first search: explores in rings of equal *step count*.

    Optimal in steps, but blind to terrain cost - it will happily route a
    player through a mud field because mud is still just one step.
    """
    t0 = time.perf_counter()
    queue: deque[Cell] = deque([start])
    came_from: dict[Cell, Cell] = {}
    visited: set[Cell] = {start}
    expanded: list[Cell] = []
    max_frontier = 1

    while queue:
        current = queue.popleft()
        expanded.append(current)
        if current == goal:
            return _result("BFS", maze, start, goal, came_from, True,
                           expanded, visited, max_frontier, t0)
        for nb in maze.neighbours(current):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                queue.append(nb)
        max_frontier = max(max_frontier, len(queue))
        yield SearchStep(current, tuple(queue), len(visited))

    return _result("BFS", maze, start, goal, came_from, False,
                   expanded, visited, max_frontier, t0)


def dfs_iter(maze: Maze, start: Cell, goal: Cell, **_: object) -> SearchGen:
    """Depth-first search: dives down one corridor until it dead-ends.

    Complete on a finite grid but the path it returns is essentially arbitrary.
    On a braided maze it routinely returns a path several times longer than
    optimal, which is exactly why the "Rookie" bot uses it.
    """
    t0 = time.perf_counter()
    # The stack carries (node, parent) pairs, and the parent pointer is only
    # committed when a node is actually popped and expanded. Recording parents
    # at push time instead would let a later push overwrite the parent of an
    # already-expanded node, producing a "path" that is not a path at all.
    stack: list[tuple[Cell, Cell | None]] = [(start, None)]
    came_from: dict[Cell, Cell] = {}
    visited: set[Cell] = set()
    expanded: list[Cell] = []
    max_frontier = 1

    while stack:
        current, parent = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if parent is not None:
            came_from[current] = parent
        expanded.append(current)
        if current == goal:
            return _result("DFS", maze, start, goal, came_from, True,
                           expanded, visited, max_frontier, t0)
        # Reversed so the first neighbour is explored first once popped.
        for nb in reversed(maze.neighbours(current)):
            if nb not in visited:
                stack.append((nb, current))
        max_frontier = max(max_frontier, len(stack))
        yield SearchStep(current, tuple(c for c, _ in stack), len(visited))

    return _result("DFS", maze, start, goal, came_from, False,
                   expanded, visited, max_frontier, t0)


def dijkstra_iter(maze: Maze, start: Cell, goal: Cell, **_: object) -> SearchGen:
    """Uniform-cost search: expands by cheapest accumulated cost ``g``.

    This is A* with h = 0, and it is the reference for "cheapest path" once mud
    makes the grid weighted.
    """
    t0 = time.perf_counter()
    counter = 0
    heap: list[tuple[float, int, Cell]] = [(0.0, counter, start)]
    best_g: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    closed: set[Cell] = set()
    expanded: list[Cell] = []
    max_frontier = 1

    while heap:
        g, _, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)
        expanded.append(current)
        if current == goal:
            return _result("Dijkstra", maze, start, goal, came_from, True,
                           expanded, closed, max_frontier, t0)
        for nb in maze.neighbours(current):
            ng = g + maze.move_cost(current, nb)
            if ng < best_g.get(nb, float("inf")):
                best_g[nb] = ng
                came_from[nb] = current
                counter += 1
                heapq.heappush(heap, (ng, counter, nb))
        max_frontier = max(max_frontier, len(heap))
        yield SearchStep(current, tuple(c for _, _, c in heap), len(closed))

    return _result("Dijkstra", maze, start, goal, came_from, False,
                   expanded, closed, max_frontier, t0)


# ---------------------------------------------------------------------------
# Informed search
# ---------------------------------------------------------------------------
def greedy_iter(
    maze: Maze,
    start: Cell,
    goal: Cell,
    heuristic: Heuristic = manhattan,
    **_: object,
) -> SearchGen:
    """Greedy best-first: expands whatever *looks* closest, ignoring cost so far.

    Very fast on open layouts, and spectacularly bad when a wall sits between it
    and the goal - it will press against that wall before backing out.
    """
    t0 = time.perf_counter()
    counter = 0
    heap: list[tuple[float, int, Cell]] = [(heuristic(maze, start, goal), counter, start)]
    came_from: dict[Cell, Cell] = {}
    seen: set[Cell] = {start}
    closed: set[Cell] = set()
    expanded: list[Cell] = []
    max_frontier = 1

    while heap:
        _, _, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)
        expanded.append(current)
        if current == goal:
            return _result("Greedy", maze, start, goal, came_from, True,
                           expanded, seen, max_frontier, t0)
        for nb in maze.neighbours(current):
            if nb not in seen:
                seen.add(nb)
                came_from[nb] = current
                counter += 1
                heapq.heappush(heap, (heuristic(maze, nb, goal), counter, nb))
        max_frontier = max(max_frontier, len(heap))
        yield SearchStep(current, tuple(c for _, _, c in heap), len(closed))

    return _result("Greedy", maze, start, goal, came_from, False,
                   expanded, seen, max_frontier, t0)


def astar_iter(
    maze: Maze,
    start: Cell,
    goal: Cell,
    heuristic: Heuristic = manhattan,
    weight: float = 1.0,
    label: str | None = None,
    **_: object,
) -> SearchGen:
    """A*: expands by ``f = g + w*h``.

    With ``w = 1`` and an admissible heuristic (one that never overestimates the
    true remaining cost) A* is guaranteed to return an optimal path while
    expanding no more nodes than Dijkstra. Raising ``w`` above 1 trades that
    guarantee for speed: the returned path is at most ``w`` times optimal.
    """
    name = label or ("A*" if weight == 1.0 else f"A*(w={weight:g})")
    t0 = time.perf_counter()
    counter = 0
    h0 = heuristic(maze, start, goal)
    # Heap entries are (f, -g, tie, cell). The second slot breaks f-ties toward
    # *larger* g, i.e. toward nodes deeper along a promising path, which cuts
    # plateau thrashing across wide-open regions. Note it is stored negated, so
    # it has to be un-negated on the way out.
    heap: list[tuple[float, float, int, Cell]] = [(h0 * weight, -0.0, counter, start)]
    best_g: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    closed: set[Cell] = set()
    expanded: list[Cell] = []
    max_frontier = 1

    while heap:
        _, neg_g, _, current = heapq.heappop(heap)
        g = -neg_g
        if current in closed:
            continue
        # A stale heap entry: a cheaper route to this cell was found later.
        if g > best_g.get(current, float("inf")):
            continue
        closed.add(current)
        expanded.append(current)
        if current == goal:
            return _result(name, maze, start, goal, came_from, True,
                           expanded, closed, max_frontier, t0)
        for nb in maze.neighbours(current):
            ng = g + maze.move_cost(current, nb)
            if ng < best_g.get(nb, float("inf")):
                best_g[nb] = ng
                came_from[nb] = current
                counter += 1
                f = ng + weight * heuristic(maze, nb, goal)
                heapq.heappush(heap, (f, -ng, counter, nb))
        max_frontier = max(max_frontier, len(heap))
        yield SearchStep(current, tuple(c for *_r, c in heap), len(closed))

    return _result(name, maze, start, goal, came_from, False,
                   expanded, closed, max_frontier, t0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
AlgorithmFn = Callable[..., SearchGen]

ALGORITHMS: dict[str, AlgorithmFn] = {
    "bfs": bfs_iter,
    "dfs": dfs_iter,
    "dijkstra": dijkstra_iter,
    "greedy": greedy_iter,
    "astar": astar_iter,
}

DISPLAY_NAMES = {
    "bfs": "BFS",
    "dfs": "DFS",
    "dijkstra": "Dijkstra",
    "greedy": "Greedy",
    "astar": "A*",
}


def solve(
    maze: Maze,
    start: Cell,
    goal: Cell,
    algorithm: str = "astar",
    heuristic: Heuristic | None = None,
    weight: float = 1.0,
) -> SearchResult:
    """Convenience wrapper: run ``algorithm`` to completion and return stats."""
    fn = ALGORITHMS.get(algorithm)
    if fn is None:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    kwargs: dict[str, object] = {}
    if algorithm in ("astar", "greedy"):
        kwargs["heuristic"] = heuristic or manhattan
    if algorithm == "astar":
        kwargs["weight"] = weight
    return run(fn(maze, start, goal, **kwargs))


def optimal_cost(maze: Maze, start: Cell, goal: Cell) -> float:
    """Ground-truth cheapest cost, used to score every other algorithm.

    Uses the reverse field so the number is directly comparable to
    ``path_cost`` of a start-to-goal path (see ``Maze.distance_field``).
    """
    return float(maze.cost_to(goal)[start[1], start[0]])


def iterate(
    maze: Maze,
    start: Cell,
    goal: Cell,
    algorithm: str,
    heuristic: Heuristic | None = None,
    weight: float = 1.0,
) -> SearchGen:
    """Return the *generator* form, for step-by-step visualisation."""
    fn = ALGORITHMS[algorithm]
    kwargs: dict[str, object] = {}
    if algorithm in ("astar", "greedy"):
        kwargs["heuristic"] = heuristic or manhattan
    if algorithm == "astar":
        kwargs["weight"] = weight
    return fn(maze, start, goal, **kwargs)


def multi_leg_path(
    maze: Maze,
    start: Cell,
    waypoints: Iterable[Cell],
    algorithm: str = "astar",
    heuristic: Heuristic | None = None,
) -> list[Cell]:
    """Chain searches through an ordered list of waypoints into one path."""
    full: list[Cell] = [start]
    cur = start
    for wp in waypoints:
        res = solve(maze, cur, wp, algorithm, heuristic)
        if not res.found:
            break
        full.extend(res.path[1:])
        cur = wp
    return full
