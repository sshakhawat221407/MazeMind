"""Door placement: counts, fairness constraints, and route-awareness."""

import random

import pytest

from src.ai.csp import build_layout
from src.config import (
    DIFFICULTY_SIZES,
    DOOR_MIN_COUNT,
    DOOR_MIN_SPACING,
    DOOR_SPAWN_CLEARANCE,
)
from src.core.doors import (
    all_edges,
    door_count,
    edge_distance,
    edge_key,
    place_doors,
    route_traffic,
    trace_route,
)
from src.core.game import spawn_points
from src.core.generator import generate
from src.core.maze import Maze, path_cost


def _layout(seed, size=(31, 21)):
    maze = generate(*size, seed=seed)
    spawns = spawn_points(maze)
    layout = build_layout(maze, spawns, n_keys=3, n_powerups=2,
                          rng=random.Random(seed))
    return maze, spawns, layout


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def test_edge_key_is_unordered():
    assert edge_key((3, 4), (3, 5)) == edge_key((3, 5), (3, 4))


def test_all_edges_lists_each_open_passage_once():
    maze = generate(15, 11, seed=1)
    edges = all_edges(maze)
    assert len(edges) == len(set(edges))
    for a, b in edges:
        assert b in maze.neighbours(a)
    # Every generator carves a spanning tree (area - 1 passages); braiding
    # only ever adds more.
    assert len(edges) >= maze.area - 1


def test_door_count_grows_with_maze_size():
    counts = [door_count(Maze(w, h)) for w, h in DIFFICULTY_SIZES.values()]
    assert counts == sorted(counts)
    assert counts[0] >= DOOR_MIN_COUNT
    assert counts[-1] > counts[0]


def test_trace_route_is_walkable_and_optimal():
    maze = generate(25, 17, seed=3)
    start, goal = (0, 0), (24, 16)
    field = maze.cost_to(goal)
    route = trace_route(maze, field, start)
    assert route[0] == start and route[-1] == goal
    for a, b in zip(route, route[1:]):
        assert b in maze.neighbours(a)
    assert path_cost(maze, route) == pytest.approx(field[start[1], start[0]])


def test_trace_route_from_an_unreachable_cell_is_empty():
    maze = Maze(4, 4)  # nothing carved
    field = maze.cost_to((3, 3))
    assert trace_route(maze, field, (0, 0)) == []


# ---------------------------------------------------------------------------
# Placement constraints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_doors_respect_every_placement_constraint(seed):
    maze, spawns, layout = _layout(seed)
    doors = place_doors(maze, spawns, layout.keys, layout.goal, random.Random(seed))

    assert doors, "no doors were placed at all"
    assert len(doors) == len(set(doors))
    for a, b in doors:
        assert b in maze.neighbours(a), "door placed on a wall"
        assert layout.goal not in (a, b), "door touching the exit"
        for cell in (a, b):
            for s in spawns:
                distance = abs(cell[0] - s[0]) + abs(cell[1] - s[1])
                assert distance > DOOR_SPAWN_CLEARANCE, "door boxes in a spawn"
    for i, e in enumerate(doors):
        for f in doors[i + 1:]:
            assert edge_distance(e, f) >= DOOR_MIN_SPACING, "doors bunched together"


def test_placement_is_reproducible_for_a_seed():
    maze, spawns, layout = _layout(4)
    first = place_doors(maze, spawns, layout.keys, layout.goal, random.Random(99))
    again = place_doors(maze, spawns, layout.keys, layout.goal, random.Random(99))
    assert first == again


def test_a_tiny_maze_returns_what_fits_instead_of_hanging():
    maze = generate(7, 5, seed=2)
    spawns = spawn_points(maze)
    layout = build_layout(maze, spawns, n_keys=1, n_powerups=0,
                          rng=random.Random(0))
    doors = place_doors(maze, spawns, layout.keys, layout.goal,
                        random.Random(0), count=50)
    assert len(doors) < 50


# ---------------------------------------------------------------------------
# The point of it: doors land where players actually go
# ---------------------------------------------------------------------------
def test_doors_sit_on_busy_routes_far_more_than_chance():
    """Route-aware placement must beat scattering doors at random.

    Measured over 20 mazes during development: 80% of doors lie on some optimal
    route against a 35% base rate, carrying 3.1x the route traffic of a random
    passage. The thresholds below leave a wide margin under those figures.
    """
    door_hits = door_total = 0
    edge_hits = edge_total = 0
    door_traffic = random_traffic = 0.0

    for seed in range(10):
        maze, spawns, layout = _layout(seed)
        traffic = route_traffic(maze, list(spawns) + layout.keys,
                                layout.keys + [layout.goal])
        edges = all_edges(maze)
        doors = place_doors(maze, spawns, layout.keys, layout.goal,
                            random.Random(seed))

        door_total += len(doors)
        door_hits += sum(1 for d in doors if traffic[d] > 0)
        edge_total += len(edges)
        edge_hits += sum(1 for e in edges if traffic[e] > 0)

        door_traffic += sum(traffic[d] for d in doors) / len(doors)
        sample = random.Random(seed).sample(edges, len(doors))
        random_traffic += sum(traffic[e] for e in sample) / len(sample)

    door_share = door_hits / door_total
    base_share = edge_hits / edge_total
    assert door_share >= base_share + 0.25, (door_share, base_share)
    assert door_traffic >= 2.0 * random_traffic, (door_traffic, random_traffic)
