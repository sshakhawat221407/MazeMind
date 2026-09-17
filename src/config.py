"""Central tuning constants for MazeMind.

Everything that a grader or a player might reasonably want to tweak lives here
so the rest of the codebase stays free of magic numbers.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
HEURISTIC_WEIGHTS = os.path.join(MODELS_DIR, "heuristic.npz")
TRAINING_HISTORY = os.path.join(MODELS_DIR, "training_history.npz")
BENCHMARK_CACHE = os.path.join(MODELS_DIR, "benchmark.npz")

# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
WINDOW_W = 1280
WINDOW_H = 780
FPS = 60
TITLE = "MazeMind - Multiplayer Maze Solver"

# --------------------------------------------------------------------------
# Maze
# --------------------------------------------------------------------------
# Wall bit flags. A set bit means "there IS a wall on that side".
N, E, S, W = 1, 2, 4, 8
OPPOSITE = {N: S, S: N, E: W, W: E}
# (dx, dy) for each direction. y grows downward, matching screen coordinates.
DELTA = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}
DIRECTIONS = (N, E, S, W)

# Terrain movement costs. Keep the minimum at 1.0 so that Manhattan distance
# stays an admissible heuristic (h never exceeds the true remaining cost).
COST_FLOOR = 1.0
COST_MUD = 3.0
MUD_FRACTION = 0.10  # share of open cells turned to mud

DIFFICULTY_SIZES = {
    "Small": (17, 13),
    "Medium": (25, 17),
    "Large": (35, 23),
    "Huge": (45, 29),
}

# Fraction of dead ends removed to create loops. Loops matter: they are what
# makes DFS visibly worse than BFS/A*, and they give racing players choices.
BRAID_FACTOR = 0.18

# --------------------------------------------------------------------------
# Gameplay
# --------------------------------------------------------------------------
PLAYER_MOVE_COOLDOWN = 0.085  # seconds between steps when a key is held
KEYS_TO_COLLECT = 3
MATCH_TIME_LIMIT = 300.0  # seconds
POWERUP_COUNT = 2
REVEAL_DURATION = 3.5  # seconds the "reveal path" powerup stays on screen
FREEZE_DURATION = 2.0  # seconds an opponent is frozen

# Locking doors. When enabled, a fixed set of gates is placed in the maze at the
# start of the match. Walking through a gate locks it for everyone - yourself
# included - for DOOR_LOCK_DURATION seconds, then it unlocks again. Ordinary
# passages never lock.
#
# An earlier version locked *every* passage behind every step, which turned the
# whole maze into a trail of barriers: it stopped being tactical and became
# noise. A handful of visible gates on routes that matter gives each lock a
# decision behind it - go through now and shut your rival out, or wait.
DOORS_ENABLED_DEFAULT = False
DOOR_LOCK_DURATION = 4.0
# One gate per this many cells, so a Huge maze gets more gates than a Small one
# while the density - and so the feel - stays the same.
DOOR_CELLS_PER_DOOR = 60
DOOR_MIN_COUNT = 3
# Gates at least this far apart (Manhattan, between their nearest cells), so
# they never bunch into one choke that seals off a whole region.
DOOR_MIN_SPACING = 3
# No gate within this distance of a spawn, so nobody starts boxed in.
DOOR_SPAWN_CLEARANCE = 2
# Share of gates placed on the busiest optimal routes; the rest are scattered
# at random, so the layout is not perfectly predictable from the objectives.
DOOR_ROUTE_SHARE = 0.7
# A strong bot at a locked gate only reroutes when the detour beats waiting by
# at least this many seconds; smaller gains are not worth abandoning its plan.
DOOR_DETOUR_MARGIN = 0.4

# Once someone finishes, the others get this long to complete the maze before
# the match is called. Without it a race ends the instant one player touches
# the exit, and the loser never gets to finish their own run or see their own
# stats - which is most of what makes a rematch interesting.
FINISH_GRACE_PERIOD = 30.0

BOT_PROFILES = {
    #  label          algorithm          step delay   mistake rate
    "Rookie": ("dfs", 0.34, 0.25),
    "Scout": ("bfs", 0.24, 0.10),
    "Hunter": ("greedy", 0.17, 0.04),
    "Tactician": ("astar", 0.13, 0.0),
    # The learned heuristic buys planning speed at ~1% path suboptimality, so
    # Oracle needs a genuinely faster cadence to stay the top tier - otherwise
    # Tactician's exactly-optimal routes cancel out the advantage.
    "Oracle": ("astar_learned", 0.085, 0.0),
}

# --------------------------------------------------------------------------
# Learned heuristic / neural network
# --------------------------------------------------------------------------
# The network predicts a DETOUR RATIO  h* / manhattan,  not a raw distance.
# Raw distance normalised by (W + H) turned out to be unlearnable from local
# features - true corridor distance depends on global maze topology, and the
# resulting model landed at 0.15x Manhattan, i.e. far weaker than the baseline
# it was supposed to beat. The ratio is a much better-behaved target: it lives
# in roughly [1, 10], it is scale-free, and because Manhattan is admissible on
# this grid the true ratio is always >= 1. Clamping the prediction at 1 then
# guarantees the learned heuristic is never *worse* informed than Manhattan.
FEATURE_NAMES = (
    # geometry relative to the goal
    "dx_signed",
    "dy_signed",
    "dx_abs",
    "dy_abs",
    "manhattan",
    "euclidean",
    "chebyshev",
    "axis_skew",
    # local structure around the cell
    "wall_ratio_3x3",
    "wall_ratio_7x7",
    "open_degree",
    # structure between the cell and the goal
    "blocked_ray",
    "blocked_ray_abs",
    "box_wall_density",
    "box_area",
    # terrain
    "box_mean_cost",
    "cell_cost",
    # global maze character (constant per maze, calibrates overall twistiness)
    "maze_area",
    "maze_mean_degree",
    "maze_dead_end_ratio",
    "goal_enclosure",
)
N_FEATURES = len(FEATURE_NAMES)

NN_HIDDEN = (96, 64)
NN_LEARNING_RATE = 3e-3
NN_EPOCHS = 90
NN_BATCH_SIZE = 256
NN_QUANTILE = 0.25  # pinball-loss quantile: low => conservative => admissible
NN_SEED = 7

# Admissibility calibration. The shrink factor alpha pulls the predicted ratio
# back toward 1 (plain Manhattan): alpha=1 trusts the network fully, alpha=0
# collapses to Manhattan.
#
# Calibrating alpha to force a zero violation rate does not work, and the reason
# is structural rather than a tuning failure: plenty of cells genuinely have a
# detour ratio of exactly 1 (straight shot to the goal), so *any* prediction
# above Manhattan at those cells is an overestimate. Demanding zero violations
# therefore drives alpha to 0 and throws the model away.
#
# So alpha is chosen by measuring the thing that actually matters - running real
# A* searches on held-out mazes and picking the most aggressive alpha whose
# mean path cost stays within NN_MAX_COST_RATIO of optimal. That is a genuine
# speed/optimality trade-off, measured rather than assumed, and the whole sweep
# is recorded in the training report and plotted in the AI Lab screen.
NN_ALPHA_GRID = (0.0, 0.15, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0)
NN_MAX_COST_RATIO = 1.01  # accept at most 1% above optimal on average
NN_CALIB_MAZES = 24
NN_CALIB_SIZE = (31, 21)

TRAIN_MAZES = 320
SAMPLES_PER_MAZE = 110
VAL_SPLIT = 0.20
