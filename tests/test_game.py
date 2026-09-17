"""Match rules: movement, keys, power-ups, and how a match ends."""

import random

import pytest

from src.ai.bot import Bot, direction_between
from src.config import (
    DOOR_LOCK_DURATION,
    E,
    FINISH_GRACE_PERIOD,
    N,
    PLAYER_MOVE_COOLDOWN,
    S,
    W,
)
from src.core.game import (
    Game,
    MatchState,
    Player,
    PlayerKind,
    PowerUp,
    new_game,
    spawn_points,
)
from src.core.generator import generate
from src.core.maze import Maze

HUMANS = [("Player 1", PlayerKind.HUMAN, None), ("Player 2", PlayerKind.HUMAN, None)]


@pytest.fixture
def game():
    maze = generate(21, 15, seed=42)
    g = new_game(maze, HUMANS, n_keys=3, rng=random.Random(0))
    g.state = MatchState.PLAYING
    return g


def test_players_start_on_opposite_corners(game):
    assert game.players[0].cell == (0, 0)
    assert game.players[1].cell == (game.maze.width - 1, game.maze.height - 1)


def test_spawns_are_never_mud():
    maze = generate(21, 15, seed=3)
    g = new_game(maze, HUMANS, rng=random.Random(1))
    for cell in spawn_points(maze):
        assert not maze.is_mud(cell)


def test_cannot_walk_through_a_wall(game):
    player = game.players[0]
    for direction in (N, E, S, W):
        if game.maze.has_wall(player.cell, direction):
            before = player.cell
            assert not game.step(player, direction)
            assert player.cell == before
            return
    pytest.skip("no walled side on the spawn cell")


def test_moving_costs_time_proportional_to_terrain(game):
    player = game.players[0]
    target = game.maze.neighbours(player.cell)[0]
    cost = game.maze.cell_cost(target)
    assert game.step(player, direction_between(player.cell, target))
    assert player.next_move_at == pytest.approx(
        game.elapsed + PLAYER_MOVE_COOLDOWN * cost
    )


def test_keys_are_per_player(game):
    """Both players must be able to collect the same key cell."""
    cell = next(iter(game.keys))
    p1, p2 = game.players
    p1.cell = cell
    game._on_enter(p1, cell)
    p2.cell = cell
    game._on_enter(p2, cell)
    assert p1.keys == 1
    assert p2.keys == 1
    assert game.key_fully_taken(cell)


def test_a_key_cannot_be_collected_twice_by_one_player(game):
    cell = next(iter(game.keys))
    player = game.players[0]
    game._on_enter(player, cell)
    game._on_enter(player, cell)
    assert player.keys == 1


def test_remaining_keys_is_per_player(game):
    cell = next(iter(game.keys))
    p1, p2 = game.players
    game._on_enter(p1, cell)
    assert cell not in game.remaining_keys(p1)
    assert cell in game.remaining_keys(p2)


def test_exit_stays_locked_until_every_key_is_held(game):
    player = game.players[0]
    assert not game.exit_open_for(player)
    game._on_enter(player, game.goal)
    assert player.finished_at is None, "exited without the keys"

    for cell in list(game.keys):
        game._on_enter(player, cell)
    assert game.exit_open_for(player)
    game._on_enter(player, game.goal)
    assert player.finished_at is not None
    assert game.winner is player
    # The match deliberately keeps running so the other player can finish their
    # own attempt; a grace clock starts instead of an immediate end.
    assert game.state is MatchState.PLAYING
    assert game.grace_remaining() == pytest.approx(FINISH_GRACE_PERIOD)


# ---------------------------------------------------------------------------
# Finishing: grace period
# ---------------------------------------------------------------------------
def _finish(game, player):
    """Give a player every key and walk them onto the exit."""
    for cell in list(game.keys):
        game._on_enter(player, cell)
    player.cell = game.goal
    game._on_enter(player, game.goal)


def test_first_finisher_wins_but_match_continues(game):
    p1, p2 = game.players
    _finish(game, p1)
    assert game.winner is p1
    assert game.state is MatchState.PLAYING
    assert p2 in game.still_running()
    assert game.placement_of(p1) == 1
    assert game.placement_of(p2) is None


def test_match_ends_when_everyone_finishes(game):
    p1, p2 = game.players
    _finish(game, p1)
    _finish(game, p2)
    game.update(0.01, {0: None, 1: None})
    assert game.state is MatchState.FINISHED
    assert game.placement_of(p2) == 2
    assert game.winner is p1  # still the first one home


def test_stragglers_are_cut_off_when_grace_expires(game):
    p1, p2 = game.players
    _finish(game, p1)
    game.update(FINISH_GRACE_PERIOD + 1.0, {0: None, 1: None})
    assert game.state is MatchState.FINISHED
    assert not p2.finished


def test_grace_remaining_is_none_before_anyone_finishes(game):
    assert game.grace_remaining() is None


def test_freeze_powerup_stops_the_opponent(game):
    p1, p2 = game.players
    cell = next(c for c, kind in game.powerups.items() if kind is PowerUp.FREEZE)
    game._on_enter(p1, cell)
    assert p2.is_frozen(game.elapsed)
    assert cell not in game.powerups  # consumed


def test_reveal_powerup_applies_to_the_collector(game):
    p1 = game.players[0]
    cell = next((c for c, k in game.powerups.items() if k is PowerUp.REVEAL), None)
    if cell is None:
        pytest.skip("no reveal power-up in this layout")
    game._on_enter(p1, cell)
    assert p1.revealing(game.elapsed)


def test_frozen_players_cannot_move(game):
    p1 = game.players[0]
    p1.frozen_until = game.elapsed + 5.0
    before = p1.cell
    game.update(0.5, {p1.pid: E, game.players[1].pid: None})
    assert p1.cell == before


def test_countdown_blocks_movement_then_releases():
    maze = generate(15, 11, seed=5)
    g = new_game(maze, HUMANS, rng=random.Random(2))
    assert g.state is MatchState.COUNTDOWN
    before = g.players[0].cell
    g.update(0.5, {0: E, 1: None})
    assert g.players[0].cell == before
    g.update(5.0, {0: None, 1: None})
    assert g.state is MatchState.PLAYING


def test_match_ends_at_the_time_limit(game):
    game.time_limit = 1.0
    game.update(2.0, {0: None, 1: None})
    assert game.state is MatchState.FINISHED
    assert game.winner is None


def test_next_objective_switches_to_the_exit_when_keys_are_done(game):
    player = game.players[0]
    for cell in list(game.keys):
        game._on_enter(player, cell)
    assert game.next_objective(player) == game.goal


# ---------------------------------------------------------------------------
# Locking doors
# ---------------------------------------------------------------------------
@pytest.fixture
def doors_game():
    maze = generate(25, 17, seed=42)
    g = new_game(maze, HUMANS, n_keys=3, rng=random.Random(0), doors_enabled=True)
    g.state = MatchState.PLAYING
    return g


def _beside(player, edge):
    """Stand a player on one side of a door, ready to walk through it."""
    a, b = edge
    player.cell = a
    player.next_move_at = 0.0
    return a, b


def _plain_passage(game):
    """An open passage that is not a door."""
    for cell in game.maze.cells():
        for nb in game.maze.neighbours(cell):
            if not game.is_door(cell, nb):
                return cell, nb
    raise AssertionError("maze has no ordinary passages")


def test_doors_are_off_by_default(game):
    assert not game.doors_enabled
    assert not game.doors
    player = game.players[0]
    target = game.maze.neighbours(player.cell)[0]
    game.step(player, direction_between(player.cell, target))
    assert not game.locked_doors


def test_enabling_doors_places_a_fixed_set(doors_game):
    assert doors_game.doors
    for a, b in doors_game.doors:
        assert b in doors_game.maze.neighbours(a), "a door was placed on a wall"


def test_ordinary_passages_never_lock(doors_game):
    g = doors_game
    player = g.players[0]
    a, b = _plain_passage(g)
    player.cell = a
    assert g.step(player, direction_between(a, b))
    assert not g.locked_doors
    assert player.doors_locked == 0
    player.next_move_at = 0.0
    assert g.step(player, direction_between(b, a)), "could not walk straight back"


def test_passing_through_a_door_locks_it(doors_game):
    g = doors_game
    player = g.players[0]
    a, b = _beside(player, next(iter(g.doors)))
    assert g.step(player, direction_between(a, b))
    assert g.is_door_locked(a, b)
    assert g.lock_remaining(a, b) == pytest.approx(DOOR_LOCK_DURATION)
    assert player.doors_locked == 1


def test_a_locked_door_blocks_the_way_back(doors_game):
    g = doors_game
    player = g.players[0]
    a, b = _beside(player, next(iter(g.doors)))
    g.step(player, direction_between(a, b))
    player.next_move_at = 0.0
    assert not g.step(player, direction_between(b, a))
    assert player.cell == b
    assert player.doors_blocked == 1


def test_a_locked_door_blocks_the_opponent_too(doors_game):
    g = doors_game
    p1, p2 = g.players
    a, b = _beside(p1, next(iter(g.doors)))
    g.step(p1, direction_between(a, b))
    p2.cell = a
    p2.next_move_at = 0.0
    assert not g.step(p2, direction_between(a, b))
    assert p2.doors_blocked == 1


def test_door_unlocks_after_the_lock_duration(doors_game):
    g = doors_game
    player = g.players[0]
    a, b = _beside(player, next(iter(g.doors)))
    g.step(player, direction_between(a, b))
    g.elapsed += DOOR_LOCK_DURATION - 0.01
    assert g.is_door_locked(a, b)
    g.elapsed += 0.02
    assert not g.is_door_locked(a, b)
    g._expire_locks()
    assert not g.locked_doors
    assert g.is_door(a, b), "a door must stay a door after unlocking"
    player.next_move_at = 0.0
    assert g.step(player, direction_between(b, a))


def test_lock_progress_runs_from_zero_to_one(doors_game):
    g = doors_game
    player = g.players[0]
    a, b = _beside(player, next(iter(g.doors)))
    g.step(player, direction_between(a, b))
    assert g.lock_progress(a, b) == pytest.approx(0.0, abs=1e-6)
    g.elapsed += DOOR_LOCK_DURATION / 2
    assert g.lock_progress(a, b) == pytest.approx(0.5, abs=1e-6)
    g.elapsed += DOOR_LOCK_DURATION
    assert g.lock_progress(a, b) == 1.0
    assert g.lock_remaining(a, b) == 0.0


def test_update_expires_locks(doors_game):
    g = doors_game
    player = g.players[0]
    a, b = _beside(player, next(iter(g.doors)))
    g.step(player, direction_between(a, b))
    g.update(DOOR_LOCK_DURATION + 0.1, {0: None, 1: None})
    assert not g.locked_doors


def test_door_key_is_order_independent():
    a, b = (1, 2), (1, 3)
    assert Game.door_key(a, b) == Game.door_key(b, a)


def test_open_neighbours_excludes_locked_doors(doors_game):
    g = doors_game
    player = g.players[0]
    a, b = _beside(player, next(iter(g.doors)))
    g.step(player, direction_between(a, b))
    assert a in g.maze.neighbours(b)
    assert a not in g.open_neighbours(b)


def test_bots_still_finish_with_doors_on():
    """Doors must raise the difficulty, not deadlock the AI."""
    for trial in range(3):
        maze = generate(21, 15, seed=400 + trial)
        g = new_game(maze, [("A", PlayerKind.BOT, "Tactician"),
                            ("B", PlayerKind.BOT, "Scout")],
                     rng=random.Random(trial), doors_enabled=True)
        g.state = MatchState.PLAYING
        bots = [Bot(p, p.bot_profile, random.Random(trial + i))
                for i, p in enumerate(g.players)]
        guard = 0
        while g.state is MatchState.PLAYING and guard < 60 * 300:
            guard += 1
            g.update(1 / 60, {b.player.pid: b.think(g, g.elapsed) for b in bots})
        assert g.winner is not None, f"trial {trial} deadlocked with doors on"


# -- how the AI deals with a locked door ------------------------------------
def _ring_game(profile, lock_seconds):
    """A 5x3 ring: a short top route with a door on it, a long bottom route.

        (0,0)-(1,0)=(2,0)-(3,0)-(4,0)      = marks the door
          |                     |
        (0,1)                 (4,1)
          |                     |
        (0,2)-(1,2)-(2,2)-(3,2)-(4,2)

    The bot stands at (1,0) heading for (4,0), facing the locked door. Waiting
    costs the lock time plus 3 steps; going round costs 9 steps.
    """
    maze = Maze(5, 3)
    for x in range(4):
        maze.carve((x, 0), E)
        maze.carve((x, 2), E)
    for y in range(2):
        maze.carve((0, y), S)
        maze.carve((4, y), S)

    player = Player(pid=0, name="bot", kind=PlayerKind.BOT, spawn=(1, 0), cell=(1, 0))
    door = Game.door_key((1, 0), (2, 0))
    g = Game(maze=maze, goal=(4, 0), players=[player], total_keys=0,
             doors_enabled=True, doors={door})
    g.state = MatchState.PLAYING
    g.elapsed = 10.0
    g.locked_doors[door] = g.elapsed + lock_seconds

    bot = Bot(player, profile, random.Random(0))
    bot.mistake_rate = 0.0           # keep the decision deterministic
    return g, bot


def test_strong_bot_goes_round_when_that_is_faster():
    # 4s left on the lock: waiting costs ~4.4s, the 9-step detour ~1.2s.
    g, bot = _ring_game("Tactician", DOOR_LOCK_DURATION)
    direction = bot.think(g, g.elapsed)
    assert direction == W, "should turn back and take the long way round"
    assert bot.door_detours == 1
    assert bot.door_waits == 0


def test_strong_bot_waits_when_the_door_is_about_to_open():
    # 0.3s left: waiting costs ~0.7s, cheaper than the ~1.2s detour.
    g, bot = _ring_game("Tactician", 0.3)
    assert bot.think(g, g.elapsed) is None
    assert bot.door_waits == 1
    assert bot.door_detours == 0


def test_weak_bot_just_waits_at_a_locked_door():
    g, bot = _ring_game("Scout", DOOR_LOCK_DURATION)
    assert bot.think(g, g.elapsed) is None
    assert bot.door_waits == 1
    assert bot.door_detours == 0


def test_waiting_bot_walks_through_once_the_door_opens():
    g, bot = _ring_game("Scout", 0.5)
    assert bot.think(g, g.elapsed) is None
    g.elapsed += 0.6
    g._expire_locks()
    assert bot.think(g, g.elapsed) == E


# -- a stop at a door is counted once, however long it lasts ----------------
def test_pushing_against_a_locked_door_counts_as_one_stop(doors_game):
    """Regression: every retry used to count, ~12 a second while a key was held."""
    g = doors_game
    p1, p2 = g.players
    a, b = _beside(p1, next(iter(g.doors)))
    g.step(p1, direction_between(a, b))           # p1 locks the door
    p2.cell, p2.next_move_at = a, 0.0
    # Hold the key against it for two seconds of simulated frames.
    for _ in range(120):
        g.update(1 / 60, {p1.pid: None, p2.pid: direction_between(a, b)})
    assert p2.cell == a
    assert p2.doors_blocked == 1


def test_a_second_visit_to_a_locked_door_is_a_new_stop(doors_game):
    g = doors_game
    p1, p2 = g.players
    a, b = _beside(p1, next(iter(g.doors)))
    g.step(p1, direction_between(a, b))
    p2.cell, p2.next_move_at = a, 0.0
    g.step(p2, direction_between(a, b))
    assert p2.doors_blocked == 1
    # Walk away along an ordinary passage, come back, and push again.
    away = next(n for n in g.maze.neighbours(a) if n != b)
    p2.next_move_at = 0.0
    assert g.step(p2, direction_between(a, away))
    p2.next_move_at = 0.0
    assert g.step(p2, direction_between(away, a))
    p2.next_move_at = 0.0
    g.step(p2, direction_between(a, b))
    assert p2.doors_blocked == 2


def test_a_waiting_bot_counts_one_stop_across_every_recheck():
    """Regression: bot rechecks used to count as separate waits (40 in 6s)."""
    g, bot = _ring_game("Scout", DOOR_LOCK_DURATION)
    door = next(iter(g.doors))
    rechecks = 0
    while g.is_door_locked(*door):
        assert bot.think(g, g.elapsed) is None, "walked through a locked door"
        rechecks += 1
        g.elapsed = bot.next_act_at
    assert rechecks > 5, "the scenario should involve many rechecks"
    assert bot.door_waits == 1
    assert bot.player.doors_blocked == 1


# ---------------------------------------------------------------------------
# Run statistics
# ---------------------------------------------------------------------------
def test_stats_track_movement(game):
    player = game.players[0]
    origin = player.cell
    target = game.maze.neighbours(origin)[0]
    game.step(player, direction_between(origin, target))
    assert player.steps == 1
    assert origin in player.visited and target in player.visited
    assert player.unique_cells == 2
    # Walking back over known ground counts as a repeat step.
    game.step(player, direction_between(target, origin))
    assert player.backtracks == 1
    assert player.unique_cells == 2


def test_wall_bumps_are_counted(game):
    player = game.players[0]
    for direction in (N, E, S, W):
        if game.maze.has_wall(player.cell, direction):
            game.step(player, direction)
            assert player.wall_bumps == 1
            return
    pytest.skip("no walled side on the spawn cell")


def test_key_timestamps_are_recorded(game):
    player = game.players[0]
    game.elapsed = 5.0
    cells = list(game.keys)
    game._on_enter(player, cells[0])
    assert player.first_key_at == pytest.approx(5.0)
    game.elapsed = 9.0
    game._on_enter(player, cells[1])
    assert player.first_key_at == pytest.approx(5.0)
    assert player.last_key_at == pytest.approx(9.0)


def test_freeze_is_counted_and_timed(game):
    p1, p2 = game.players
    cell = next(c for c, kind in game.powerups.items() if kind is PowerUp.FREEZE)
    game._on_enter(p1, cell)
    assert p2.times_frozen == 1
    assert p1.powerups_taken == 1
    game.update(0.5, {0: None, 1: None})
    assert p2.frozen_seconds > 0


def test_direction_between_adjacent_cells():
    assert direction_between((1, 1), (1, 0)) is N
    assert direction_between((1, 1), (2, 1)) is E
    assert direction_between((1, 1), (1, 2)) is S
    assert direction_between((1, 1), (0, 1)) is W
    assert direction_between((1, 1), (3, 3)) is None


# ---------------------------------------------------------------------------
# Bots
# ---------------------------------------------------------------------------
def test_bot_does_not_replan_every_frame():
    """A bot that replans each tick burns thousands of searches per second."""
    maze = generate(21, 15, seed=11)
    g = new_game(maze, [("A", PlayerKind.BOT, "Tactician"),
                        ("B", PlayerKind.BOT, "Scout")], rng=random.Random(1))
    g.state = MatchState.PLAYING
    bots = [Bot(p, p.bot_profile, random.Random(i)) for i, p in enumerate(g.players)]
    for _ in range(600):  # ten seconds at 60 FPS
        g.update(1 / 60, {b.player.pid: b.think(g, g.elapsed) for b in bots})
    for bot in bots:
        assert bot.replans < 60, f"{bot.profile} replanned {bot.replans} times"


@pytest.mark.parametrize("strong,weak", [("Oracle", "Rookie"),
                                         ("Tactician", "Rookie")])
def test_stronger_bots_beat_weaker_ones(strong, weak):
    wins = 0
    trials = 5
    for trial in range(trials):
        maze = generate(21, 15, seed=200 + trial)
        g = new_game(maze, [(strong, PlayerKind.BOT, strong),
                            (weak, PlayerKind.BOT, weak)], rng=random.Random(trial))
        g.state = MatchState.PLAYING
        bots = [Bot(p, p.bot_profile, random.Random(trial + i))
                for i, p in enumerate(g.players)]
        guard = 0
        while g.state is MatchState.PLAYING and guard < 60 * 300:
            guard += 1
            g.update(1 / 60, {b.player.pid: b.think(g, g.elapsed) for b in bots})
        if g.winner and g.winner.name == strong:
            wins += 1
    assert wins >= trials - 1, f"{strong} only won {wins}/{trials} against {weak}"


def test_matches_between_bots_always_finish():
    """Regression: shared keys once made matches mathematically unwinnable."""
    for trial in range(4):
        maze = generate(21, 15, seed=300 + trial)
        g = new_game(maze, [("A", PlayerKind.BOT, "Tactician"),
                            ("B", PlayerKind.BOT, "Hunter")],
                     rng=random.Random(trial))
        g.state = MatchState.PLAYING
        bots = [Bot(p, p.bot_profile, random.Random(trial + i))
                for i, p in enumerate(g.players)]
        guard = 0
        while g.state is MatchState.PLAYING and guard < 60 * 300:
            guard += 1
            g.update(1 / 60, {b.player.pid: b.think(g, g.elapsed) for b in bots})
        assert g.winner is not None, f"trial {trial} ended in a draw"
