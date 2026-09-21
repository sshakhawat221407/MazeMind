"""The AI opponent.

Each difficulty tier is literally a different search algorithm, so the
difficulty slider is a live demonstration of what the algorithms are worth:

======================  ==================================================
Rookie    (DFS)         dives down corridors, takes ~50% longer routes
Scout     (BFS)         shortest in *steps*, so it walks straight into mud
Hunter    (Greedy)      beelines for the target and gets trapped by walls
Tactician (A*/Manhattan) optimal routes
Oracle    (A*/learned)   optimal routes, fewer nodes expanded, fastest cadence
======================  ==================================================

Two further things separate the tiers. Weak bots pick their next key by
straight-line distance, which is frequently the wrong key; strong bots price
every remaining key with a real search and go for the genuinely cheapest one.
And weak bots have a per-step mistake rate that knocks them off their plan and
forces a replan.

Locked gates split the tiers a third way. A weak bot that reaches a locked gate
simply waits for it. A strong bot prices both options in seconds - the time left
on the lock plus the rest of its plan, against the best route that avoids every
locked gate - and takes whichever is faster.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import BOT_PROFILES, DELTA, DIRECTIONS, DOOR_DETOUR_MARGIN
from ..core.game import Game, Player
from ..core.maze import Cell, Maze, path_cost
from . import search
from .heuristics import LearnedHeuristic, Heuristic, manhattan


def direction_between(a: Cell, b: Cell) -> int | None:
    """The direction bit that steps from ``a`` to adjacent ``b``."""
    delta = (b[0] - a[0], b[1] - a[1])
    for d in DIRECTIONS:
        if DELTA[d] == delta:
            return d
    return None


@dataclass
class Bot:
    """Steers one player. Call :meth:`think` once per frame."""

    player: Player
    profile: str = "Tactician"
    rng: random.Random = field(default_factory=random.Random)

    algorithm: str = "astar"
    delay: float = 0.13
    mistake_rate: float = 0.0
    heuristic: Heuristic | None = None
    smart_targeting: bool = True

    path: list[Cell] = field(default_factory=list)
    target: Cell | None = None
    next_act_at: float = 0.0
    # Exposed so the UI can draw what the bot is currently planning.
    last_result: search.SearchResult | None = None
    replans: int = 0
    nodes_expanded_total: int = 0
    # How the bot dealt with locked gates, for the results screen.
    door_waits: int = 0
    door_detours: int = 0

    def __post_init__(self) -> None:
        algo, delay, mistakes = BOT_PROFILES.get(
            self.profile, BOT_PROFILES["Tactician"]
        )
        self.delay = delay
        self.mistake_rate = mistakes
        if algo == "astar_learned":
            self.algorithm = "astar"
            learned = LearnedHeuristic.load()
            # Falling back to Manhattan keeps the top tier playable before the
            # network has ever been trained.
            self.heuristic = learned if learned is not None else manhattan
        else:
            self.algorithm = algo
            self.heuristic = manhattan if algo in ("greedy", "astar") else None
        # Only the two strongest tiers price their targets with a real search.
        self.smart_targeting = self.profile in ("Tactician", "Oracle")

    # -- target selection --------------------------------------------------
    def choose_target(self, game: Game) -> Cell:
        """Which objective to head for next."""
        if game.exit_open_for(self.player):
            return game.goal
        remaining = game.remaining_keys(self.player)
        if not remaining:
            return game.goal
        if not self.smart_targeting:
            # Straight-line nearest: cheap, and frequently wrong, because a key
            # ten tiles away on screen can be eighty tiles away by corridor.
            return min(
                remaining,
                key=lambda c: abs(c[0] - self.player.cell[0])
                + abs(c[1] - self.player.cell[1]),
            )
        # Price each candidate with an actual search and take the cheapest.
        best: Cell = remaining[0]
        best_cost = float("inf")
        for cand in remaining:
            res = search.solve(
                game.maze, self.player.cell, cand,
                self.algorithm, heuristic=self.heuristic,
            )
            if res.found and res.cost < best_cost:
                best_cost, best = res.cost, cand
        return best

    # -- planning ----------------------------------------------------------
    def replan(self, game: Game, target: Cell) -> None:
        result = search.solve(
            game.maze, self.player.cell, target,
            self.algorithm, heuristic=self.heuristic,
        )
        self.last_result = result
        self.replans += 1
        self.nodes_expanded_total += result.nodes_expanded
        self.path = list(result.path) if result.found else []
        self.target = target

    def _resync(self) -> None:
        """Drop the part of the plan already walked; clear it if we strayed."""
        if not self.path:
            return
        try:
            idx = self.path.index(self.player.cell)
        except ValueError:
            self.path = []
            return
        self.path = self.path[idx:]

    # -- per-frame decision ------------------------------------------------
    def think(self, game: Game, now: float) -> int | None:
        if self.player.finished or self.player.is_frozen(now):
            return None
        if now < self.next_act_at:
            return None

        # Any early exit below must still push next_act_at forward. Returning
        # without doing so re-enters this method on the very next frame, and
        # since every path here runs a fresh search, the bot burns thousands of
        # searches per second while standing still.
        def hold() -> None:
            self.next_act_at = now + self.delay

        target = self.choose_target(game)
        self._resync()
        if target != self.target or len(self.path) < 2:
            self.replan(game, target)
        if len(self.path) < 2:
            hold()
            return None

        direction = direction_between(self.player.cell, self.path[1])
        if direction is None:
            self.path = []
            hold()
            return None

        # Blunder: take some other legal exit and abandon the plan.
        if self.mistake_rate and self.rng.random() < self.mistake_rate:
            options = [
                d for d in DIRECTIONS
                if not game.maze.has_wall(self.player.cell, d)
                and game.maze.in_bounds(
                    (self.player.cell[0] + DELTA[d][0],
                     self.player.cell[1] + DELTA[d][1])
                )
                and d != direction
            ]
            if options:
                direction = self.rng.choice(options)
                self.path = []

        dx, dy = DELTA[direction]
        nxt = (self.player.cell[0] + dx, self.player.cell[1] + dy)

        if game.is_door(self.player.cell, nxt) and game.is_door_locked(self.player.cell, nxt):
            return self._at_locked_gate(game, now, target, nxt)

        step_cost = 1.0
        if game.maze.in_bounds(nxt):
            step_cost = game.maze.cell_cost(nxt)
        self.next_act_at = now + self.delay * step_cost
        return direction

    # -- locked gates ------------------------------------------------------
    def _at_locked_gate(self, game: Game, now: float, target: Cell,
                        far_side: Cell) -> int | None:
        """Decide between waiting at a locked gate and going round it."""
        remaining = game.lock_remaining(self.player.cell, far_side)
        # Look again when the lock runs out, but at least once a step, so a
        # gate that unlocks is walked through straight away.
        recheck = now + max(0.05, min(remaining, self.delay))
        # True only on the first look at this door, not on every recheck.
        new_stop = game.note_blocked(self.player, self.player.cell, far_side)

        if not self.smart_targeting:
            self.door_waits += int(new_stop)
            self.next_act_at = recheck
            return None

        # Both options in seconds, since that is what decides a race:
        #   wait   = lock time left + time to walk the rest of the current plan
        #   detour = time to walk the best route avoiding every locked gate
        wait_time = remaining + path_cost(game.maze, self.path) * self.delay
        detour = self._route_around_locks(game, target)
        if detour is not None:
            detour_time = detour.cost * self.delay
            # The margin stops the bot abandoning a good plan to save a
            # fraction of a second, which looks like dithering on screen.
            if detour_time + DOOR_DETOUR_MARGIN < wait_time:
                self.door_detours += 1
                self.path = list(detour.path)
                nxt = self.path[1]
                direction = direction_between(self.player.cell, nxt)
                self.next_act_at = now + self.delay * game.maze.cell_cost(nxt)
                return direction

        self.door_waits += int(new_stop)
        self.next_act_at = recheck
        return None

    def _route_around_locks(self, game: Game, target: Cell):
        """Best route to ``target`` treating every locked gate as a wall.

        Searches a copy of the maze with those gates bricked up, so the search
        algorithms need no knowledge of doors at all.
        """
        sealed = game.maze.copy()
        for a, b in list(game.locked_doors):
            if game.is_door_locked(a, b):
                direction = direction_between(a, b)
                if direction is not None:
                    sealed.build_wall(a, direction)
        result = search.solve(sealed, self.player.cell, target,
                              self.algorithm, heuristic=self.heuristic)
        self.nodes_expanded_total += result.nodes_expanded
        if not result.found or len(result.path) < 2:
            return None
        return result

    # -- reporting ---------------------------------------------------------
    def status(self) -> str:
        algo = search.DISPLAY_NAMES.get(self.algorithm, self.algorithm)
        if self.profile == "Oracle":
            algo += " + learned h"
        return f"{self.profile} ({algo})"


def plan_preview(
    maze: Maze, start: Cell, goal: Cell, algorithm: str = "astar"
) -> list[Cell]:
    """Route used by the REVEAL power-up. Always optimal, whoever picked it up."""
    result = search.solve(maze, start, goal, algorithm, heuristic=manhattan)
    return list(result.path) if result.found else []
