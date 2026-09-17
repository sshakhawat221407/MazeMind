"""CSP solver machinery and the maze layout problem it drives."""

import random

import numpy as np
import pytest

from src.ai.csp import CSP, ac3, backtracking_search, build_layout, solve
from src.core.game import spawn_points
from src.core.generator import generate


# ---------------------------------------------------------------------------
# Generic solver
# ---------------------------------------------------------------------------
def test_solves_a_simple_all_different():
    csp = CSP()
    for name in "abc":
        csp.add_variable(name, [1, 2, 3])
    for i, a in enumerate("abc"):
        for b in "abc"[i + 1:]:
            csp.add_binary(a, b, lambda x, y: x != y, "!=")
    assignment, stats = solve(csp)
    assert assignment is not None
    assert len(set(assignment.values())) == 3
    assert stats.solved


def test_detects_an_unsatisfiable_problem():
    """Four variables, three values, all different - impossible."""
    csp = CSP()
    for name in "abcd":
        csp.add_variable(name, [1, 2, 3])
    for i, a in enumerate("abcd"):
        for b in "abcd"[i + 1:]:
            csp.add_binary(a, b, lambda x, y: x != y, "!=")
    assignment, stats = solve(csp)
    assert assignment is None
    assert not stats.solved


def test_unary_constraints_prune_domains():
    csp = CSP()
    csp.add_variable("x", list(range(10)))
    csp.add_unary("x", lambda v: v % 2 == 0, "even")
    removed = csp.apply_unary()
    assert removed == 5
    assert csp.domains["x"] == [0, 2, 4, 6, 8]


def test_ac3_removes_unsupported_values():
    csp = CSP()
    csp.add_variable("x", [1, 2, 3])
    csp.add_variable("y", [1])
    # x must be strictly greater than y, so x=1 has no support.
    csp.add_binary("x", "y", lambda a, b: a > b, ">")
    domains, removed = ac3(csp)
    assert domains is not None
    assert 1 not in domains["x"]
    assert removed >= 1


def test_ac3_reports_a_wipeout():
    csp = CSP()
    csp.add_variable("x", [1])
    csp.add_variable("y", [1])
    csp.add_binary("x", "y", lambda a, b: a > b, ">")
    domains, _ = ac3(csp)
    assert domains is None


def test_binary_constraints_are_symmetric():
    csp = CSP()
    csp.add_variable("a", [1])
    csp.add_variable("b", [2])
    csp.add_binary("a", "b", lambda x, y: x < y, "<")
    assert csp.binary_ok("a", 1, "b", 2)
    assert csp.binary_ok("b", 2, "a", 1)      # the flipped direction too
    assert not csp.binary_ok("b", 1, "a", 2)


def test_node_limit_is_respected():
    csp = CSP()
    for i in range(12):
        csp.add_variable(f"v{i}", list(range(11)))
    for i in range(12):
        for j in range(i + 1, 12):
            csp.add_binary(f"v{i}", f"v{j}", lambda x, y: x != y, "!=")
    from src.ai.csp import SolveStats

    stats = SolveStats()
    # 12 variables over 11 values with all-different is unsatisfiable, so the
    # search would run a long time; the limit must cut it off.
    result = backtracking_search(csp, None, stats, node_limit=200)
    assert result is None
    assert stats.nodes <= 400


# ---------------------------------------------------------------------------
# Maze layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_layout_places_everything_legally(seed):
    maze = generate(31, 21, seed=seed)
    starts = spawn_points(maze)
    layout = build_layout(maze, starts, n_keys=3, n_powerups=2,
                          rng=random.Random(seed))

    items = [layout.goal] + layout.keys + layout.powerups
    assert len(items) == 6
    assert len(set(items)) == 6, "two objectives share a cell"
    for cell in items:
        assert cell not in starts, "an objective sits on a spawn"

    reachable = maze.reachable_from(starts[0])
    for cell in items:
        assert cell in reachable, "an objective is unreachable"


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_layout_separates_objectives(seed):
    maze = generate(31, 21, seed=seed)
    layout = build_layout(maze, spawn_points(maze), n_keys=3, n_powerups=2,
                          rng=random.Random(seed))
    items = [layout.goal] + layout.keys + layout.powerups
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 1


def test_layout_is_fairer_than_random_placement():
    """The whole point of the CSP: comparable routes for both players."""
    csp_gaps, random_gaps = [], []
    for seed in range(10):
        maze = generate(31, 21, seed=seed)
        starts = spawn_points(maze)
        fields = [maze.distance_field(s) for s in starts]

        layout = build_layout(maze, starts, n_keys=3, n_powerups=2,
                              rng=random.Random(seed))
        csp_gaps.append(layout.fairness_gap)

        rng = random.Random(seed)
        cells = [c for c in maze.cells()
                 if np.isfinite(fields[0][c[1], c[0]])]
        for _ in range(20):
            c = rng.choice(cells)
            random_gaps.append(abs(fields[0][c[1], c[0]] - fields[1][c[1], c[0]]))

    assert np.mean(csp_gaps) < np.mean(random_gaps) * 0.6


def test_layout_always_returns_something_on_a_tiny_maze():
    """Tight constraints on a small board must relax, not crash or hang."""
    maze = generate(7, 5, seed=1)
    layout = build_layout(maze, spawn_points(maze), n_keys=2, n_powerups=1,
                          rng=random.Random(0))
    assert layout.goal is not None
    assert len(layout.keys) == 2
