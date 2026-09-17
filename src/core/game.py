"""Match state for a two-player same-screen race.

Both players run on one shared maze and one shared clock. The rules:

* Collect all keys, then reach the exit. The exit stays locked until a player
  holds every key, so the race is a route-planning problem rather than a dash.
* Terrain costs are real. Stepping into mud takes three times as long as
  stepping onto clear floor, which is the same number ``Dijkstra`` and ``A*``
  optimise against - so the cheapest path on screen is genuinely the fastest
  path to walk.
* Power-ups are picked up by walking over them and fire immediately.

Objective placement is delegated to the CSP solver so that neither player spawns
with a materially shorter route than the other.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from ..config import (
    DELTA,
    DOOR_LOCK_DURATION,
    DOORS_ENABLED_DEFAULT,
    FINISH_GRACE_PERIOD,
    FREEZE_DURATION,
    KEYS_TO_COLLECT,
    MATCH_TIME_LIMIT,
    PLAYER_MOVE_COOLDOWN,
    POWERUP_COUNT,
    REVEAL_DURATION,
)
from .doors import Edge, edge_key, place_doors
from .generator import clear_terrain
from .maze import Cell, Maze


class PlayerKind(Enum):
    HUMAN = "human"
    BOT = "bot"


class MatchState(Enum):
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    FINISHED = "finished"


class PowerUp(Enum):
    REVEAL = "reveal"   # briefly draws the optimal route to your next objective
    FREEZE = "freeze"   # stops the opponent where they stand


@dataclass
class Player:
    pid: int
    name: str
    kind: PlayerKind
    spawn: Cell
    cell: Cell
    keys: int = 0
    steps: int = 0
    distance_cost: float = 0.0
    next_move_at: float = 0.0
    frozen_until: float = 0.0
    reveal_until: float = 0.0
    finished_at: float | None = None
    bot_profile: str | None = None

    # -- run statistics, shown on the results screen ----------------------
    doors_blocked: int = 0        # times a locked gate refused a move
    doors_locked: int = 0         # gates this player locked by passing through
    wall_bumps: int = 0           # moves into a wall
    mud_steps: int = 0            # steps onto costly terrain
    times_frozen: int = 0
    frozen_seconds: float = 0.0
    powerups_taken: int = 0
    first_key_at: float | None = None
    last_key_at: float | None = None
    backtracks: int = 0           # steps onto a cell already visited
    visited: set[Cell] = field(default_factory=set)
    # The locked door currently holding this player up, if any. Used so that a
    # player pushing against one door for four seconds counts as one stop,
    # not as a stop for every retry the input loop makes in that time.
    blocked_at: Edge | None = None

    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    @property
    def unique_cells(self) -> int:
        return len(self.visited)

    def exploration_ratio(self) -> float:
        """Share of this player's steps that broke new ground."""
        return self.unique_cells / self.steps if self.steps else 0.0

    def is_frozen(self, now: float) -> bool:
        return now < self.frozen_until

    def revealing(self, now: float) -> bool:
        return now < self.reveal_until


@dataclass
class Game:
    """Everything that changes during a match."""

    maze: Maze
    goal: Cell
    players: list[Player]
    # Cell -> set of player ids that have already picked this key up.
    #
    # Keys are per-player, not first-come-first-served. Shared keys look
    # natural but make the match unwinnable: with three keys on the board and
    # two racers, a 2-1 split leaves *neither* player able to reach three, and
    # both wander until the clock runs out. Every player must visit every key
    # cell, so the race is a routing problem and always has a winner.
    keys: dict[Cell, set[int]] = field(default_factory=dict)
    powerups: dict[Cell, PowerUp] = field(default_factory=dict)
    total_keys: int = KEYS_TO_COLLECT
    elapsed: float = 0.0
    state: MatchState = MatchState.COUNTDOWN
    countdown: float = 3.0
    winner: Player | None = None
    time_limit: float = MATCH_TIME_LIMIT
    events: list[tuple[float, str]] = field(default_factory=list)
    csp_summary: str = ""
    fairness_gap: float = 0.0

    # -- locking doors -----------------------------------------------------
    doors_enabled: bool = DOORS_ENABLED_DEFAULT
    # The fixed gates, chosen once when the match is built. Only these
    # passages ever lock; every other passage is always open.
    doors: set[Edge] = field(default_factory=set)
    # Gate -> time it unlocks. Keyed on the unordered pair of cells a gate
    # joins, so a locked gate is shut in both directions for every player.
    locked_doors: dict[Edge, float] = field(default_factory=dict)

    # -- finishing ---------------------------------------------------------
    # Set when the first player finishes. Everyone still running has until this
    # moment to complete the maze; the match is called when they all finish or
    # the clock runs out, whichever comes first.
    grace_deadline: float | None = None
    finish_order: list[int] = field(default_factory=list)

    # -- queries -----------------------------------------------------------
    def remaining_keys(self, player: Player) -> list[Cell]:
        """Key cells this particular player has still to visit."""
        return [c for c, taken in self.keys.items() if player.pid not in taken]

    def key_fully_taken(self, cell: Cell) -> bool:
        """True once every player has collected this key (used for fading)."""
        return len(self.keys.get(cell, ())) >= len(self.players)

    def exit_open_for(self, player: Player) -> bool:
        return player.keys >= self.total_keys

    # -- doors -------------------------------------------------------------
    door_key = staticmethod(edge_key)

    def is_door(self, a: Cell, b: Cell) -> bool:
        """Whether the passage between two cells is one of the gates."""
        return self.doors_enabled and edge_key(a, b) in self.doors

    def is_door_locked(self, a: Cell, b: Cell) -> bool:
        until = self.locked_doors.get(edge_key(a, b))
        return until is not None and self.elapsed < until

    def lock_remaining(self, a: Cell, b: Cell) -> float:
        """Seconds until a gate unlocks; 0 when it is already open."""
        until = self.locked_doors.get(edge_key(a, b))
        if until is None:
            return 0.0
        return max(0.0, until - self.elapsed)

    def lock_progress(self, a: Cell, b: Cell) -> float:
        """How much of a gate's lock has run, in ``[0, 1]``; 1 when open."""
        remaining = self.lock_remaining(a, b)
        if remaining <= 0.0:
            return 1.0
        return 1.0 - remaining / DOOR_LOCK_DURATION

    def note_blocked(self, player: Player, a: Cell, b: Cell) -> bool:
        """Record a locked door stopping ``player``. Returns True for a new stop.

        Humans and bots meet a locked door differently - a human keeps pressing
        against it, a bot decides to wait - so both route through here to be
        counted the same way: once per door per stop.
        """
        key = edge_key(a, b)
        if player.blocked_at == key:
            return False
        player.blocked_at = key
        player.doors_blocked += 1
        return True

    def open_neighbours(self, cell: Cell) -> list[Cell]:
        """Neighbours reachable *right now* - walls and locked gates excluded."""
        return [n for n in self.maze.neighbours(cell)
                if not self.is_door_locked(cell, n)]

    def _expire_locks(self) -> None:
        if not self.locked_doors:
            return
        now = self.elapsed
        for key in [k for k, until in self.locked_doors.items() if until <= now]:
            del self.locked_doors[key]

    # -- finishing ---------------------------------------------------------
    def grace_remaining(self) -> float | None:
        """Seconds left for stragglers, or ``None`` when nobody has finished."""
        if self.grace_deadline is None:
            return None
        return max(0.0, self.grace_deadline - self.elapsed)

    def still_running(self) -> list[Player]:
        return [p for p in self.players if not p.finished]

    def placement_of(self, player: Player) -> int | None:
        """1-based finishing position, or ``None`` if they never finished."""
        if player.pid not in self.finish_order:
            return None
        return self.finish_order.index(player.pid) + 1

    def opponent_of(self, player: Player) -> Player | None:
        for other in self.players:
            if other is not player:
                return other
        return None

    def next_objective(self, player: Player) -> Cell:
        """Where this player should head next: nearest key, else the exit."""
        if self.exit_open_for(player):
            return self.goal
        remaining = self.remaining_keys(player)
        if not remaining:
            return self.goal
        # Straight-line nearest is enough for target *selection*; the actual
        # route is planned with a real search by whoever is steering.
        return min(
            remaining,
            key=lambda c: abs(c[0] - player.cell[0]) + abs(c[1] - player.cell[1]),
        )

    # -- simulation --------------------------------------------------------
    def update(self, dt: float, intents: dict[int, int | None]) -> None:
        """Advance the match by ``dt`` seconds given each player's held direction."""
        if self.state is MatchState.COUNTDOWN:
            self.countdown -= dt
            if self.countdown <= 0:
                self.state = MatchState.PLAYING
                self.log("Go!")
            return

        if self.state is MatchState.FINISHED:
            return

        self.elapsed += dt
        self._expire_locks()

        if self.elapsed >= self.time_limit:
            self.state = MatchState.FINISHED
            self.log("Time limit reached")
            return

        for player in self.players:
            if player.finished:
                continue
            if player.is_frozen(self.elapsed):
                # Charge the freeze to this player's stats while it lasts.
                player.frozen_seconds += dt
                continue
            direction = intents.get(player.pid)
            if direction is None:
                continue
            if self.elapsed < player.next_move_at:
                continue
            self.step(player, direction)

        # The match ends when everyone is home, or when the stragglers' grace
        # period expires - not the moment the first player arrives.
        if not self.still_running():
            self.state = MatchState.FINISHED
            self.log("Everyone finished")
        elif self.grace_deadline is not None and self.elapsed >= self.grace_deadline:
            self.state = MatchState.FINISHED
            names = ", ".join(p.name for p in self.still_running())
            self.log(f"Time up for {names}")

    def step(self, player: Player, direction: int) -> bool:
        """Attempt one grid move. Returns whether it happened."""
        if self.maze.has_wall(player.cell, direction):
            player.wall_bumps += 1
            return False
        dx, dy = DELTA[direction]
        origin = player.cell
        target = (origin[0] + dx, origin[1] + dy)
        if not self.maze.in_bounds(target):
            player.wall_bumps += 1
            return False

        if self.is_door(origin, target) and self.is_door_locked(origin, target):
            self.note_blocked(player, origin, target)
            player.next_move_at = self.elapsed + PLAYER_MOVE_COOLDOWN
            return False

        cost = self.maze.cell_cost(target)
        if not player.visited:
            player.visited.add(origin)
        if target in player.visited:
            player.backtracks += 1
        player.visited.add(target)
        if self.maze.is_mud(target):
            player.mud_steps += 1

        player.blocked_at = None
        player.cell = target
        player.steps += 1
        player.distance_cost += cost
        # Mud slows you by exactly the factor the search algorithms price it at,
        # so "cheapest path" and "fastest path" are the same thing.
        player.next_move_at = self.elapsed + PLAYER_MOVE_COOLDOWN * cost

        if self.is_door(origin, target):
            # Passing through a gate locks it behind you - for everyone,
            # yourself included - until the lock runs out.
            self.locked_doors[edge_key(origin, target)] = (
                self.elapsed + DOOR_LOCK_DURATION
            )
            player.doors_locked += 1

        self._on_enter(player, target)
        return True

    def _on_enter(self, player: Player, cell: Cell) -> None:
        holders = self.keys.get(cell)
        if holders is not None and player.pid not in holders:
            holders.add(player.pid)
            player.keys += 1
            if player.first_key_at is None:
                player.first_key_at = self.elapsed
            player.last_key_at = self.elapsed
            remaining = self.total_keys - player.keys
            self.log(
                f"{player.name} took a key "
                + (f"({remaining} to go)" if remaining else "- exit unlocked!")
            )

        power = self.powerups.pop(cell, None)
        if power is not None:
            player.powerups_taken += 1
        if power is PowerUp.REVEAL:
            player.reveal_until = self.elapsed + REVEAL_DURATION
            self.log(f"{player.name} revealed the route")
        elif power is PowerUp.FREEZE:
            other = self.opponent_of(player)
            if other and not other.finished:
                other.frozen_until = self.elapsed + FREEZE_DURATION
                other.times_frozen += 1
                self.log(f"{player.name} froze {other.name}")

        if cell == self.goal and self.exit_open_for(player) and not player.finished:
            player.finished_at = self.elapsed
            self.finish_order.append(player.pid)
            if self.winner is None:
                # First one home wins, but the match keeps running so everyone
                # else can complete their own attempt.
                self.winner = player
                self.grace_deadline = self.elapsed + FINISH_GRACE_PERIOD
                self.log(
                    f"{player.name} wins in {self.elapsed:.1f}s - "
                    f"{FINISH_GRACE_PERIOD:.0f}s for the rest"
                )
            else:
                place = self.placement_of(player)
                self.log(f"{player.name} finished {_ordinal(place)} "
                         f"in {self.elapsed:.1f}s")

    def log(self, message: str) -> None:
        self.events.append((self.elapsed, message))
        del self.events[:-6]  # keep only the most recent handful


def _ordinal(n: int | None) -> str:
    if n is None:
        return "-"
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def spawn_points(maze: Maze) -> list[Cell]:
    """Opposite corners, so neither player starts nearer the middle."""
    return [(0, 0), (maze.width - 1, maze.height - 1)]


def new_game(
    maze: Maze,
    player_specs: Sequence[tuple[str, PlayerKind, str | None]],
    n_keys: int = KEYS_TO_COLLECT,
    n_powerups: int = POWERUP_COUNT,
    rng: random.Random | None = None,
    doors_enabled: bool = DOORS_ENABLED_DEFAULT,
) -> Game:
    """Build a match: place objectives via CSP, then seat the players.

    ``player_specs`` is a sequence of ``(name, kind, bot_profile)``.
    """
    from ..ai.csp import build_layout  # local import keeps core/ai decoupled

    rng = rng or random.Random()
    starts = spawn_points(maze)
    # Spawns must never be mud, or a player can lose the race to a dice roll
    # before they have pressed anything.
    clear_terrain(maze, starts)

    layout = build_layout(maze, starts, n_keys=n_keys, n_powerups=n_powerups, rng=rng)
    clear_terrain(maze, [layout.goal])

    players = [
        Player(
            pid=i,
            name=name,
            kind=kind,
            spawn=starts[i],
            cell=starts[i],
            bot_profile=profile,
            visited={starts[i]},
        )
        for i, (name, kind, profile) in enumerate(player_specs)
    ]

    powers = [PowerUp.REVEAL, PowerUp.FREEZE]
    game = Game(
        maze=maze,
        goal=layout.goal,
        players=players,
        keys={c: set() for c in layout.keys},
        powerups={c: powers[i % len(powers)] for i, c in enumerate(layout.powerups)},
        total_keys=len(layout.keys),
        csp_summary=layout.stats.summary(),
        fairness_gap=layout.fairness_gap,
        doors_enabled=doors_enabled,
    )
    if doors_enabled:
        # Placed after the CSP layout, because the gates go on the routes
        # *between* the objectives it just chose.
        game.doors = set(place_doors(maze, starts, layout.keys, layout.goal, rng))
    game.log("Collect every key, then reach the exit")
    if doors_enabled:
        game.log(f"{len(game.doors)} locking doors - each locks for "
                 f"{DOOR_LOCK_DURATION:.0f}s after someone passes through")
    return game
