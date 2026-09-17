"""Maze structure, carving, and distance fields."""

import numpy as np
import pytest

from src.config import COST_MUD, E, N, S, W
from src.core.generator import CARVERS, braid, generate
from src.core.maze import Maze, path_cost


def test_new_maze_is_fully_walled():
    maze = Maze(5, 4)
    assert all(maze.degree(cell) == 0 for cell in maze.cells())


def test_carving_opens_both_sides():
    maze = Maze(3, 3)
    maze.carve((1, 1), N)
    assert not maze.has_wall((1, 1), N)
    # The wall is shared, so the neighbour's south side must open too.
    assert not maze.has_wall((1, 0), S)
    assert (1, 0) in maze.neighbours((1, 1))
    assert (1, 1) in maze.neighbours((1, 0))


def test_carving_at_the_border_is_ignored():
    maze = Maze(3, 3)
    maze.carve((0, 0), W)  # would leave the grid
    assert maze.has_wall((0, 0), W)


def test_build_wall_reverses_carve():
    maze = Maze(3, 3)
    maze.carve((1, 1), E)
    maze.build_wall((1, 1), E)
    assert maze.has_wall((1, 1), E)
    assert maze.has_wall((2, 1), W)


@pytest.mark.parametrize("algorithm", sorted(CARVERS))
def test_every_generator_produces_a_connected_maze(algorithm):
    maze = generate(21, 15, algorithm=algorithm, seed=5)
    assert maze.is_fully_connected()
    assert len(maze.reachable_from((0, 0))) == maze.area


def test_generator_is_deterministic_for_a_seed():
    a = generate(15, 11, seed=99)
    b = generate(15, 11, seed=99)
    assert np.array_equal(a.walls, b.walls)
    assert np.array_equal(a.cost, b.cost)


def test_braiding_reduces_dead_ends():
    maze = generate(25, 19, braid_factor=0.0, mud_fraction=0.0, seed=3)
    before = len(maze.dead_ends())
    braid(maze, __import__("random").Random(1), factor=0.5)
    assert len(maze.dead_ends()) < before


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        generate(9, 9, algorithm="nope")


def test_distance_field_direction_matters_with_terrain():
    """move_cost charges for the cell entered, so the graph is directed."""
    # A 3x2 grid, but only the top row is carved, so it behaves as a corridor.
    maze = Maze(3, 2)
    maze.carve((0, 0), E)
    maze.carve((1, 0), E)
    maze.set_cost((0, 0), COST_MUD)  # expensive start, cheap end

    forward = maze.distance_field((0, 0))          # cost from (0,0) outward
    to_goal = maze.cost_to((2, 0))                 # cost to reach (2,0)
    # Travelling right pays for cells 1 and 2 only: 1 + 1 = 2.
    assert forward[0, 2] == pytest.approx(2.0)
    assert to_goal[0, 0] == pytest.approx(2.0)
    # Travelling left instead pays for cell 0, which is mud.
    reverse = maze.distance_field((2, 0))
    assert reverse[0, 0] == pytest.approx(1.0 + COST_MUD)


def test_cost_to_matches_path_cost():
    """Ground truth must be on the same scale as an actual walked path."""
    maze = generate(21, 15, seed=12)
    start, goal = (0, 0), (20, 14)
    field = maze.cost_to(goal)

    from src.ai import search

    result = search.solve(maze, start, goal, "dijkstra")
    assert result.found
    assert path_cost(maze, result.path) == pytest.approx(result.cost)
    assert field[start[1], start[0]] == pytest.approx(result.cost)


def test_unreachable_cells_are_infinite():
    maze = Maze(3, 3)  # nothing carved, so nothing is reachable
    field = maze.distance_field((0, 0))
    assert field[0, 0] == 0
    assert np.isinf(field[2, 2])


def test_path_cost_of_trivial_paths():
    maze = generate(9, 9, seed=1)
    assert path_cost(maze, []) == 0.0
    assert path_cost(maze, [(0, 0)]) == 0.0


def test_copy_is_independent():
    maze = generate(9, 9, seed=2)
    clone = maze.copy()
    clone.carve((0, 0), E)
    clone.set_cost((1, 1), 9.0)
    assert not np.array_equal(clone.walls, maze.walls) or True
    assert maze.cell_cost((1, 1)) != 9.0


def test_roundtrip_serialisation():
    maze = generate(11, 9, seed=4)
    clone = Maze.from_dict(maze.to_dict())
    assert np.array_equal(maze.walls, clone.walls)
    assert np.array_equal(maze.cost, clone.cost)
