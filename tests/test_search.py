"""Search algorithms: correctness, optimality, and the guarantees they claim."""

import pytest

from src.ai import search
from src.ai.heuristics import chebyshev, euclidean, manhattan, zero
from src.core.generator import generate
from src.core.maze import path_cost

ALGORITHMS = ["bfs", "dfs", "dijkstra", "greedy", "astar"]
OPTIMAL = ["dijkstra", "astar"]


@pytest.fixture(scope="module")
def problem():
    maze = generate(25, 19, seed=77)
    return maze, (0, 0), (24, 18)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_finds_a_path(problem, algorithm):
    maze, start, goal = problem
    result = search.solve(maze, start, goal, algorithm)
    assert result.found
    assert result.path[0] == start
    assert result.path[-1] == goal


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_returned_path_is_actually_walkable(problem, algorithm):
    """Every consecutive pair must be adjacent with no wall between them."""
    maze, start, goal = problem
    result = search.solve(maze, start, goal, algorithm)
    for a, b in zip(result.path, result.path[1:]):
        assert b in maze.neighbours(a), f"{algorithm}: {a} -> {b} is not a legal step"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_reported_cost_matches_the_path(problem, algorithm):
    maze, start, goal = problem
    result = search.solve(maze, start, goal, algorithm)
    assert result.cost == pytest.approx(path_cost(maze, result.path))


@pytest.mark.parametrize("algorithm", OPTIMAL)
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_optimal_algorithms_are_optimal(algorithm, seed):
    maze = generate(21, 15, seed=seed)
    start, goal = (0, 0), (20, 14)
    best = search.optimal_cost(maze, start, goal)
    result = search.solve(maze, start, goal, algorithm)
    assert result.cost == pytest.approx(best)


@pytest.mark.parametrize("heuristic", [manhattan, euclidean, chebyshev, zero])
def test_astar_is_optimal_for_any_admissible_heuristic(problem, heuristic):
    maze, start, goal = problem
    best = search.optimal_cost(maze, start, goal)
    result = search.solve(maze, start, goal, "astar", heuristic=heuristic)
    assert result.cost == pytest.approx(best)


def test_astar_with_zero_heuristic_equals_dijkstra(problem):
    """A* with h = 0 *is* Dijkstra, so the work done must match."""
    maze, start, goal = problem
    a = search.solve(maze, start, goal, "astar", heuristic=zero)
    d = search.solve(maze, start, goal, "dijkstra")
    assert a.cost == pytest.approx(d.cost)
    assert a.nodes_expanded == d.nodes_expanded


def test_astar_expands_no_more_than_dijkstra(problem):
    """A useful heuristic must not make A* do strictly more work."""
    maze, start, goal = problem
    informed = search.solve(maze, start, goal, "astar", heuristic=manhattan)
    blind = search.solve(maze, start, goal, "dijkstra")
    assert informed.nodes_expanded <= blind.nodes_expanded


def test_weighted_astar_bounds_suboptimality(problem):
    """Weighted A* promises a path at most `weight` times optimal."""
    maze, start, goal = problem
    best = search.optimal_cost(maze, start, goal)
    weight = 1.6
    result = search.solve(maze, start, goal, "astar", weight=weight)
    assert result.found
    assert result.cost <= best * weight + 1e-6


def test_generator_form_matches_the_batch_form(problem):
    """The animated path and the drained path must agree."""
    maze, start, goal = problem
    gen = search.iterate(maze, start, goal, "astar", heuristic=manhattan)
    steps = 0
    try:
        while True:
            next(gen)
            steps += 1
    except StopIteration as stop:
        animated = stop.value
    batch = search.solve(maze, start, goal, "astar", heuristic=manhattan)
    assert animated.cost == pytest.approx(batch.cost)
    assert animated.nodes_expanded == batch.nodes_expanded
    assert steps == batch.nodes_expanded - 1  # the goal expansion does not yield


def test_start_equals_goal():
    maze = generate(9, 9, seed=8)
    result = search.solve(maze, (0, 0), (0, 0), "astar")
    assert result.found
    assert result.path == [(0, 0)]
    assert result.cost == 0.0


def test_unknown_algorithm_raises(problem):
    maze, start, goal = problem
    with pytest.raises(ValueError):
        search.solve(maze, start, goal, "telepathy")


def test_multi_leg_path_is_contiguous():
    maze = generate(21, 15, seed=6)
    route = search.multi_leg_path(maze, (0, 0), [(10, 7), (20, 14)])
    for a, b in zip(route, route[1:]):
        assert b in maze.neighbours(a)


def test_dfs_is_usually_worse_than_optimal():
    """Not a guarantee, but across many mazes DFS must clearly lag."""
    penalties = []
    for seed in range(12):
        maze = generate(25, 19, seed=seed)
        start, goal = (0, 0), (24, 18)
        best = search.optimal_cost(maze, start, goal)
        dfs = search.solve(maze, start, goal, "dfs")
        penalties.append(dfs.cost / best)
    assert sum(penalties) / len(penalties) > 1.1
