"""Constraint satisfaction: a general solver, plus the maze layout problem.

**Why a CSP is the right tool here.** In a same-screen race, *where* the keys
and the exit sit decides the match before either player moves. Scattering them
randomly regularly hands one player a three-second win. What we actually want to
express is a set of simultaneous requirements - "the exit is far from both
spawns", "it is roughly equally far from each", "no two keys are on top of each
other", "the keys are spread across the maze" - and then find any placement
satisfying all of them at once. That is a constraint satisfaction problem, and
solving it properly is what makes the game fair.

The solver implements the standard machinery from scratch:

* **AC-3** arc consistency as a preprocessing pass, pruning values that cannot
  participate in any solution before search begins.
* **MRV** (minimum remaining values) variable ordering, with **degree** as the
  tie-break: assign the most constrained variable first, so dead ends surface
  near the root of the tree rather than at its leaves.
* **LCV** (least constraining value) value ordering: try the value that rules
  out the fewest options for the neighbours.
* **Forward checking** after each assignment, to fail fast.

If the strictest version of the problem has no solution - easy to arrange on a
small maze with tight fairness bounds - the layout builder walks down a
**relaxation ladder**, loosening tolerances a step at a time rather than giving
up or hanging.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Sequence

from ..core.maze import Cell, Maze

Value = Hashable
Variable = str
UnaryPred = Callable[[Value], bool]
BinaryPred = Callable[[Value, Value], bool]


@dataclass
class SolveStats:
    """Instrumentation, so the UI can show that the solver actually worked."""

    nodes: int = 0
    backtracks: int = 0
    prunes: int = 0
    unary_removed: int = 0
    ac3_removed: int = 0
    domain_before: int = 0
    domain_after: int = 0
    relaxation_level: int = 0
    solved: bool = False

    def summary(self) -> str:
        return (
            f"domain {self.domain_before}->{self.domain_after} | "
            f"unary -{self.unary_removed} | AC-3 -{self.ac3_removed} | "
            f"nodes {self.nodes} | backtracks {self.backtracks} | "
            f"relax L{self.relaxation_level}"
        )


class CSP:
    """Variables with finite domains, unary and binary constraints."""

    def __init__(self) -> None:
        self.variables: list[Variable] = []
        self.domains: dict[Variable, list[Value]] = {}
        self.unary: dict[Variable, list[tuple[str, UnaryPred]]] = defaultdict(list)
        self.binary: dict[tuple[Variable, Variable], list[tuple[str, BinaryPred]]] = (
            defaultdict(list)
        )
        self.neighbours: dict[Variable, set[Variable]] = defaultdict(set)

    # -- construction ------------------------------------------------------
    def add_variable(self, name: Variable, domain: Iterable[Value]) -> None:
        self.variables.append(name)
        self.domains[name] = list(domain)

    def add_unary(self, var: Variable, pred: UnaryPred, name: str = "") -> None:
        self.unary[var].append((name, pred))

    def add_binary(
        self, a: Variable, b: Variable, pred: BinaryPred, name: str = ""
    ) -> None:
        """Register a symmetric binary constraint in both arc directions."""
        self.binary[(a, b)].append((name, pred))
        self.binary[(b, a)].append((name, lambda y, x, _p=pred: _p(x, y)))
        self.neighbours[a].add(b)
        self.neighbours[b].add(a)

    # -- checking ----------------------------------------------------------
    def apply_unary(self) -> int:
        """Filter every domain by its unary constraints. Returns values removed."""
        removed = 0
        for var in self.variables:
            preds = self.unary.get(var)
            if not preds:
                continue
            before = len(self.domains[var])
            self.domains[var] = [
                v for v in self.domains[var] if all(p(v) for _, p in preds)
            ]
            removed += before - len(self.domains[var])
        return removed

    def binary_ok(self, a: Variable, av: Value, b: Variable, bv: Value) -> bool:
        preds = self.binary.get((a, b))
        if not preds:
            return True
        return all(p(av, bv) for _, p in preds)

    def consistent(
        self, var: Variable, value: Value, assignment: dict[Variable, Value]
    ) -> bool:
        for other, other_value in assignment.items():
            if other == var:
                continue
            if not self.binary_ok(var, value, other, other_value):
                return False
        return True


# ---------------------------------------------------------------------------
# AC-3
# ---------------------------------------------------------------------------
def revise(
    csp: CSP, domains: dict[Variable, list[Value]], xi: Variable, xj: Variable
) -> int:
    """Drop values of ``xi`` with no supporting value in ``xj``. Returns count."""
    keep: list[Value] = []
    removed = 0
    dj = domains[xj]
    for x in domains[xi]:
        if any(csp.binary_ok(xi, x, xj, y) for y in dj):
            keep.append(x)
        else:
            removed += 1
    if removed:
        domains[xi] = keep
    return removed


def ac3(
    csp: CSP, domains: dict[Variable, list[Value]] | None = None
) -> tuple[dict[Variable, list[Value]] | None, int]:
    """Enforce arc consistency. Returns ``(domains, removed)``.

    ``domains`` is ``None`` when some domain was wiped out, meaning the problem
    is unsatisfiable and search need not even start.
    """
    doms = {v: list(d) for v, d in (domains or csp.domains).items()}
    queue: deque[tuple[Variable, Variable]] = deque(
        (a, b) for a in csp.variables for b in csp.neighbours[a]
    )
    total = 0
    while queue:
        xi, xj = queue.popleft()
        removed = revise(csp, doms, xi, xj)
        if removed:
            total += removed
            if not doms[xi]:
                return None, total
            # xi shrank, so every arc pointing *into* xi may now be revisable.
            for xk in csp.neighbours[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return doms, total


# ---------------------------------------------------------------------------
# Backtracking search with MRV / degree / LCV / forward checking
# ---------------------------------------------------------------------------
def _select_variable(
    csp: CSP, assignment: dict[Variable, Value], domains: dict[Variable, list[Value]]
) -> Variable:
    """MRV, tie-broken by degree (most constrained, then most constraining)."""
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(
        unassigned,
        key=lambda v: (len(domains[v]), -len(csp.neighbours[v])),
    )


def _order_values(
    csp: CSP,
    var: Variable,
    assignment: dict[Variable, Value],
    domains: dict[Variable, list[Value]],
) -> list[Value]:
    """LCV: prefer values that eliminate the fewest neighbour options."""
    neighbours = [n for n in csp.neighbours[var] if n not in assignment]
    if not neighbours:
        return list(domains[var])

    def conflicts(value: Value) -> int:
        total = 0
        for n in neighbours:
            for nv in domains[n]:
                if not csp.binary_ok(var, value, n, nv):
                    total += 1
        return total

    return sorted(domains[var], key=conflicts)


def _forward_check(
    csp: CSP,
    var: Variable,
    value: Value,
    domains: dict[Variable, list[Value]],
    assignment: dict[Variable, Value],
) -> dict[Variable, list[Value]] | None:
    """Prune neighbour domains against ``var = value``; ``None`` on wipe-out."""
    pruned = {v: list(d) for v, d in domains.items()}
    pruned[var] = [value]
    for n in csp.neighbours[var]:
        if n in assignment:
            continue
        survivors = [nv for nv in pruned[n] if csp.binary_ok(var, value, n, nv)]
        if not survivors:
            return None
        pruned[n] = survivors
    return pruned


def backtracking_search(
    csp: CSP,
    domains: dict[Variable, list[Value]] | None = None,
    stats: SolveStats | None = None,
    node_limit: int = 20000,
) -> dict[Variable, Value] | None:
    """Depth-first assignment search with the heuristics described above."""
    stats = stats or SolveStats()
    doms = domains if domains is not None else {v: list(d) for v, d in csp.domains.items()}

    def backtrack(
        assignment: dict[Variable, Value], doms: dict[Variable, list[Value]]
    ) -> dict[Variable, Value] | None:
        if len(assignment) == len(csp.variables):
            return assignment
        if stats.nodes > node_limit:
            return None

        var = _select_variable(csp, assignment, doms)
        for value in _order_values(csp, var, assignment, doms):
            stats.nodes += 1
            if stats.nodes > node_limit:
                return None
            if not csp.consistent(var, value, assignment):
                continue
            assignment[var] = value
            reduced = _forward_check(csp, var, value, doms, assignment)
            if reduced is not None:
                result = backtrack(assignment, reduced)
                if result is not None:
                    return result
            else:
                stats.prunes += 1
            del assignment[var]
            stats.backtracks += 1
        return None

    result = backtrack({}, doms)
    stats.solved = result is not None
    return result


def solve(
    csp: CSP,
    node_limit: int = 20000,
    max_domain: int | None = None,
    rng: random.Random | None = None,
) -> tuple[dict[Variable, Value] | None, SolveStats]:
    """Full pipeline: unary filter -> domain cap -> AC-3 -> backtracking."""
    stats = SolveStats()
    stats.domain_before = sum(len(d) for d in csp.domains.values())
    stats.unary_removed = csp.apply_unary()
    if any(not d for d in csp.domains.values()):
        return None, stats

    if max_domain:
        # AC-3 revises every arc pairwise, so an uncapped domain of ~700 cells
        # across six variables is tens of millions of predicate calls. Sampling
        # down after unary filtering keeps propagation interactive; shuffling
        # first means successive matches get different layouts.
        for var in csp.variables:
            domain = csp.domains[var]
            if len(domain) > max_domain:
                if rng is not None:
                    rng.shuffle(domain)
                csp.domains[var] = domain[:max_domain]

    doms, removed = ac3(csp)
    stats.ac3_removed = removed
    if doms is None:
        return None, stats
    stats.domain_after = sum(len(d) for d in doms.values())
    assignment = backtracking_search(csp, doms, stats, node_limit)
    return assignment, stats


# ---------------------------------------------------------------------------
# The maze layout problem
# ---------------------------------------------------------------------------
@dataclass
class Layout:
    """Where the objectives ended up, plus how hard it was to get there."""

    goal: Cell
    keys: list[Cell] = field(default_factory=list)
    powerups: list[Cell] = field(default_factory=list)
    stats: SolveStats = field(default_factory=SolveStats)
    fairness_gap: float = 0.0  # |cost(P1 -> goal) - cost(P2 -> goal)|


def _quadrant(cell: Cell, maze: Maze) -> int:
    return (0 if cell[0] < maze.width / 2 else 1) + 2 * (
        0 if cell[1] < maze.height / 2 else 1
    )


# Progressively looser parameter sets. Index 0 is the layout we actually want;
# later entries trade fairness and spread for guaranteed solvability.
_RELAXATIONS = [
    # goal_min_frac, fair_frac, key_min_frac, key_fair_frac, sep, quadrants
    (0.60, 0.08, 0.20, 0.30, 5, True),
    (0.50, 0.14, 0.15, 0.45, 4, True),
    (0.40, 0.22, 0.10, 0.60, 3, False),
    (0.25, 0.40, 0.05, 1.00, 2, False),
    (0.00, 9.99, 0.00, 9.99, 1, False),
]

MAX_DOMAIN = 110  # cap domain size so AC-3 stays interactive


def build_layout(
    maze: Maze,
    starts: Sequence[Cell],
    n_keys: int = 3,
    n_powerups: int = 2,
    rng: random.Random | None = None,
) -> Layout:
    """Place the exit, keys and power-ups so the race is winnable and fair.

    Distances are true in-maze costs from a Dijkstra sweep per spawn, not
    straight-line approximations - a cell can be ten tiles away on screen and
    eighty tiles away through the corridors.
    """
    rng = rng or random.Random()
    # One sweep per spawn point gives the real travel cost to every cell.
    fields = [maze.distance_field(s, weighted=True) for s in starts]
    reach = [f[maze.height - 1, maze.width - 1] for f in fields]  # touch, keep numpy warm
    del reach

    def dist(cell: Cell, i: int) -> float:
        return float(fields[i][cell[1], cell[0]])

    finite = [
        c for c in maze.cells()
        if c not in starts and all(dist(c, i) != float("inf") for i in range(len(starts)))
    ]
    if not finite:
        raise ValueError("maze has no cell reachable from every spawn")

    max_dist = max(max(dist(c, i) for i in range(len(starts))) for c in finite)

    for level, (goal_min, fair, key_min, key_fair, sep, quads) in enumerate(_RELAXATIONS):
        csp = CSP()
        variables: list[str] = ["goal"]

        # -- variables over the full board ----------------------------------
        # Domains start as every reachable cell and are narrowed by the solver's
        # own unary pass, rather than being pre-filtered here. That keeps the
        # constraints declarative and lets SolveStats report honestly on how
        # much work propagation actually did.
        csp.add_variable("goal", finite)
        for k in range(n_keys):
            name = f"key{k}"
            variables.append(name)
            csp.add_variable(name, finite)
        for p in range(n_powerups):
            name = f"pow{p}"
            variables.append(name)
            csp.add_variable(name, finite)

        # -- unary constraints ----------------------------------------------
        def goal_far(c: Cell) -> bool:
            return all(dist(c, i) >= goal_min * max_dist for i in range(len(starts)))

        def goal_fair(c: Cell) -> bool:
            return abs(dist(c, 0) - dist(c, 1)) <= fair * max_dist

        csp.add_unary("goal", goal_far, "far from both spawns")
        csp.add_unary("goal", goal_fair, "equidistant from spawns")

        for k in range(n_keys):
            csp.add_unary(
                f"key{k}",
                lambda c: min(dist(c, i) for i in range(len(starts))) >= key_min * max_dist,
                "not on a doorstep",
            )
            csp.add_unary(
                f"key{k}",
                lambda c: abs(dist(c, 0) - dist(c, 1)) <= key_fair * max_dist,
                "reachable fairly by both",
            )

        # -- binary constraints ---------------------------------------------
        def far_enough(a: Cell, b: Cell, _sep: int = sep) -> bool:
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) >= _sep

        for i, va in enumerate(variables):
            for vb in variables[i + 1:]:
                csp.add_binary(va, vb, far_enough, f"sep>={sep}")

        if quads and n_keys > 1:
            def different_quadrant(a: Cell, b: Cell) -> bool:
                return _quadrant(a, maze) != _quadrant(b, maze)

            key_vars = [f"key{k}" for k in range(n_keys)]
            for i, va in enumerate(key_vars):
                for vb in key_vars[i + 1:]:
                    csp.add_binary(va, vb, different_quadrant, "distinct quadrant")

        assignment, stats = solve(csp, max_domain=MAX_DOMAIN, rng=rng)
        stats.relaxation_level = level
        if assignment is not None:
            goal = assignment["goal"]
            return Layout(
                goal=goal,
                keys=[assignment[f"key{k}"] for k in range(n_keys)],
                powerups=[assignment[f"pow{p}"] for p in range(n_powerups)],
                stats=stats,
                fairness_gap=abs(dist(goal, 0) - dist(goal, 1)),
            )

    # Every relaxation failed (pathological maze). Fall back to random cells so
    # the game always starts; the stats make clear this path was taken.
    pool = list(finite)
    rng.shuffle(pool)
    need = 1 + n_keys + n_powerups
    picks = pool[:need]
    while len(picks) < need:
        picks.append(pool[0])
    stats = SolveStats(relaxation_level=len(_RELAXATIONS), solved=False)
    goal = picks[0]
    return Layout(
        goal=goal,
        keys=picks[1:1 + n_keys],
        powerups=picks[1 + n_keys:need],
        stats=stats,
        fairness_gap=abs(dist(goal, 0) - dist(goal, 1)),
    )
