"""Where the locking doors go.

A door is a gate on one passage - the edge between two adjacent open cells. The
interesting question is *which* passages get one. Scattering them at random
mostly lands them in dead-end corridors nobody walks down, where they never
lock and never matter. So placement is **route-aware**:

1. Trace the optimal route between every pair of objectives - each spawn and
   each key to every key and to the exit - using the exact cost-to fields the
   rest of the game already relies on.
2. Count how many of those routes use each passage. A high count means a
   corridor that almost every sensible plan funnels through.
3. Fill most of the gate quota from the busiest passages, and the rest at
   random, so the layout rewards reading the maze without being perfectly
   predictable from where the keys are.

Three constraints keep it fair: no gate touching the exit, no gate so close to
a spawn that a player starts boxed in, and a minimum spacing so gates never
cluster into a single choke that seals off a region.

Tracing uses the directed cost model from ``Maze.distance_field``: from cell
``c`` the optimal next step is the neighbour ``v`` minimising
``cost(v) + D(v)``, where ``D`` is the cost-to-target field.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Sequence

from ..config import (
    DOOR_CELLS_PER_DOOR,
    DOOR_MIN_COUNT,
    DOOR_MIN_SPACING,
    DOOR_ROUTE_SHARE,
    DOOR_SPAWN_CLEARANCE,
)
from .maze import Cell, Maze

Edge = tuple[Cell, Cell]


def edge_key(a: Cell, b: Cell) -> Edge:
    """Unordered key for the passage between two adjacent cells."""
    return (a, b) if a <= b else (b, a)


def all_edges(maze: Maze) -> list[Edge]:
    """Every open passage in the maze, each listed exactly once."""
    out: list[Edge] = []
    for cell in maze.cells():
        for nb in maze.neighbours(cell):
            if cell < nb:
                out.append((cell, nb))
    return out


def door_count(maze: Maze) -> int:
    """How many gates a maze of this size gets."""
    return max(DOOR_MIN_COUNT, round(maze.area / DOOR_CELLS_PER_DOOR))


def trace_route(maze: Maze, field, start: Cell) -> list[Cell]:
    """Follow a cost-to field downhill from ``start`` to its zero point.

    Returns an empty list when ``start`` cannot reach the field's target.
    """
    if not math.isfinite(field[start[1], start[0]]):
        return []
    route = [start]
    current = start
    # A correct field always strictly decreases along the route, so the path
    # can never be longer than the maze; the guard only catches a bad field.
    for _ in range(maze.area):
        if field[current[1], current[0]] == 0.0:
            return route
        best: Cell | None = None
        best_value = math.inf
        for nb in maze.neighbours(current):
            value = maze.cell_cost(nb) + field[nb[1], nb[0]]
            if value < best_value:
                best, best_value = nb, value
        if best is None:
            return []
        route.append(best)
        current = best
    return []


def route_traffic(
    maze: Maze, sources: Sequence[Cell], targets: Sequence[Cell]
) -> Counter:
    """How many optimal source-to-target routes use each passage."""
    traffic: Counter = Counter()
    for target in targets:
        field = maze.cost_to(target)          # one sweep serves every source
        for source in sources:
            if source == target:
                continue
            route = trace_route(maze, field, source)
            for a, b in zip(route, route[1:]):
                traffic[edge_key(a, b)] += 1
    return traffic


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def edge_distance(e: Edge, f: Edge) -> int:
    """Manhattan distance between the nearest cells of two passages."""
    return min(_manhattan(p, q) for p in e for q in f)


def place_doors(
    maze: Maze,
    spawns: Sequence[Cell],
    keys: Iterable[Cell],
    goal: Cell,
    rng: random.Random | None = None,
    count: int | None = None,
) -> list[Edge]:
    """Choose which passages become locking gates.

    May return fewer than ``count`` gates when the constraints leave no room -
    a tiny maze simply has fewer places a fair gate can go.
    """
    rng = rng or random.Random()
    count = door_count(maze) if count is None else count
    keys = list(keys)

    def allowed(edge: Edge, chosen: list[Edge]) -> bool:
        if goal in edge:
            return False
        for cell in edge:
            if any(_manhattan(cell, s) <= DOOR_SPAWN_CLEARANCE for s in spawns):
                return False
        return all(edge_distance(edge, other) >= DOOR_MIN_SPACING
                   for other in chosen)

    chosen: list[Edge] = []

    # -- phase 1: the corridors that sensible routes actually funnel through --
    traffic = route_traffic(maze, list(spawns) + keys, keys + [goal])
    busy = list(traffic)
    rng.shuffle(busy)                       # random order among equal counts
    busy.sort(key=lambda e: traffic[e], reverse=True)
    route_quota = round(count * DOOR_ROUTE_SHARE)
    for edge in busy:
        if len(chosen) >= route_quota:
            break
        if allowed(edge, chosen):
            chosen.append(edge)

    # -- phase 2: scatter the rest so gates are not fully predictable ---------
    taken = set(chosen)
    rest = [e for e in all_edges(maze) if e not in taken]
    rng.shuffle(rest)
    for edge in rest:
        if len(chosen) >= count:
            break
        if allowed(edge, chosen):
            chosen.append(edge)

    return chosen
