# Architecture

How the pieces fit together, and why several of them are shaped the way they
are. The README covers *what* the project achieves; this covers *how*.

---

## Layering

```
                    main.py
                       |
        +--------------+--------------+
        |                             |
     src/ui                       src/ai
   (pygame, scenes)        (search, csp, nn, bots)
        |                             |
        +--------------+--------------+
                       |
                    src/core
              (maze, generator, game)
```

`core` knows nothing about `ai` or `ui`. `ai` depends on `core` only. `ui`
depends on both. The one deliberate exception is `core/game.py`, which imports
the CSP layout builder *inside a function* rather than at module scope — the
game rules genuinely need objective placement, and a local import keeps the
module-level dependency graph acyclic.

`src/viz/charts.py` sits beside the UI and converts matplotlib figures into
pygame surfaces.

---

## Data flow for one match

```
 settings (setup screen)
        |
        v
 generate(w, h, algorithm)          src/core/generator.py
        |   carve -> braid -> mud
        v
 build_layout(maze, spawns)         src/ai/csp.py
        |   unary filter -> AC-3 -> backtracking (MRV/LCV/FC)
        v
 new_game(maze, players)            src/core/game.py
        |
        v
 GameScene.update(dt)               src/ui/scenes/game.py
    |         |
    |         +--> Bot.think() ---> search.solve() ---> LearnedHeuristic
    |                                                        |
    +--> Game.update(dt, intents)                    feature_grid + MLP
             |                                       (one batched pass)
             v
        MatchState.FINISHED -> ResultsScene
```

---

## Key design decisions

### The maze graph is directed

`Maze.move_cost(src, dst)` returns the cost of the cell being *entered*. That is
the natural model for a game — you pay for the ground you land on — but it makes
the graph directed whenever two adjacent cells have different terrain.

The consequence is easy to get wrong. A distance field computed *outward from*
the goal gives the cost of travelling *from the goal to* each cell, which is not
the same as `h*`, the cost of travelling *from each cell to the goal*. Writing
`D(c)` for the latter:

```
D(goal) = 0
D(c)    = min over neighbours v of [ cost(v) + D(v) ]
```

so the reverse sweep relaxes with the cost of the node being **expanded**, not
the neighbour being reached. `Maze.distance_field(source, reverse=True)`,
exposed as `Maze.cost_to(goal)`, does this.

Getting it backwards inflates or deflates every label by
`cost(goal) - cost(cell)`. During development this showed up as Dijkstra scoring
`0.999` against its own ground truth — a "better than optimal" result, which is
the tell that ground truth is measured on a different scale than the thing being
scored. `tests/test_maze.py::test_cost_to_matches_path_cost` pins it.

### Search algorithms are generators

Each algorithm is written once as a generator that yields a `SearchStep` per
expansion and `return`s a `SearchResult`. `search.run()` drains it; the AI Lab
steps it one expansion at a time.

Timing lives in `run()`, not inside the algorithms. A generator is suspended
between yields, so a clock started inside one measures wall-clock across every
pause — for the animated view that meant reporting *the length of the animation*
(around 950 ms) instead of compute time (about 2 ms). Draining in a tight loop
makes elapsed time and compute time the same thing.

### DFS commits parent pointers on pop, not on push

Recording a node's parent when it is *pushed* lets a later push overwrite the
parent of a node that has already been expanded, and the reconstructed "path"
then contains steps between non-adjacent cells. The stack carries
`(node, parent)` pairs and the parent is committed only when the node is popped.
`tests/test_search.py::test_returned_path_is_actually_walkable` checks every
algorithm against this class of bug.

### The heuristic runs once per search, not once per node

A per-node numpy forward pass would make A\* slower than the search it is meant
to accelerate. `LearnedHeuristic` instead computes features for **every cell at
once** (`features.feature_grid`, fully vectorised, integral images for the
variable-size bounding-box statistics) and evaluates the network in a single
batched pass on the first call for a given `(maze, goal)`. After that the
heuristic is an array lookup.

### The CSP works on real domains

Objective domains start as *every reachable cell* and are narrowed by the
solver's own unary pass, rather than being pre-filtered before the CSP is built.
This keeps the constraints declarative and lets `SolveStats` report honestly on
how much propagation actually did.

Because AC-3 revises arcs pairwise, an uncapped domain of ~700 cells across six
variables is tens of millions of predicate calls. Domains are therefore sampled
down to `MAX_DOMAIN` *after* unary filtering, which keeps propagation
interactive; shuffling before the cut means successive matches get different
layouts.

Typical run on a 31×21 maze: domain 3894 → 574 values, 1663 removed by unary
constraints, solved in 6 nodes with 0 backtracks. AC-3 usually prunes nothing
further here — after unary filtering the separation constraints are easily
satisfiable, so arc consistency has little left to find. That is reported as it
is rather than dressed up.

### Doors are fixed, route-aware, and keyed on the passage

The first version locked *every* passage behind every step. It turned the maze
into a trail of barriers — lots of red, few decisions. Doors are now a fixed set
chosen once per match by `core/doors.py`, and only those passages ever lock.

**Placement is route-aware.** Scattered at random, most doors land in dead-end
corridors nobody walks, where they never lock and never matter. Instead
`place_doors` traces the optimal route between every pair of objectives (each
spawn and key to every key and the exit), counts how many routes use each
passage, and fills 70% of the quota from the busiest passages and the rest at
random. Tracing reuses the cost-to fields: from cell `c` the optimal next step
is the neighbour minimising `cost(v) + D(v)`. Constraints keep it fair — no door
touching the exit, none within two cells of a spawn, and a minimum spacing so
doors never bunch into one choke that seals off a region. Measured over 20
mazes: 80% of doors sit on an optimal route against a 35% base rate, carrying
3.1× the traffic of a random passage.

**A door belongs to the edge, not a cell.** `doors` and `locked_doors` are keyed
on the unordered pair `edge_key(a, b)`. Keying on "the cell you left" would lock
only one direction; keying on "the cell you entered" would seal every exit from
that cell. The unordered pair locks exactly one passage, for everybody, both
ways.

**A stop is counted once.** A human pushing against a locked door retries every
movement tick, and a waiting bot rechecks every step — so counting attempts
reported a dozen "blocks" per second. `Game.note_blocked` records a stop only
when the player's `blocked_at` door changes, and a successful move clears it.
Humans and bots both go through it, so their stats are comparable.

**Search never learns about doors.** When a strong bot weighs a detour it
searches a *copy* of the maze with every locked door bricked up as a wall, so
the five search algorithms stay exactly as they were.

### Static maze rendering is cached

Floor, mud and walls never change during a match, so they are drawn once into a
surface keyed on `(maze, cell size, wall checksum)` and blitted per frame. Only
players, pickups and search overlays are redrawn. Without the split, a 45×29
maze costs several thousand draw calls per frame.

### Long work runs on a worker thread

Benchmarks and training would freeze the window for seconds. Both run on a
daemon thread that reports progress through a shared attribute; the UI keeps
repainting. Only the *computation* is threaded — matplotlib is not thread-safe,
so figures are always built on the main thread once the data is ready.

---

## Testing

146 tests, ~12 seconds. The ones that carry the most weight:

| Test | What it protects |
| --- | --- |
| `test_analytic_gradients_match_finite_differences` | Backprop is correct — the whole learning claim rests on it |
| `test_astar_with_zero_heuristic_equals_dijkstra` | Same cost *and* same expansion count; a strong joint check |
| `test_returned_path_is_actually_walkable` | Every step is legal, for every algorithm |
| `test_cost_to_matches_path_cost` | Ground truth and walked paths share a scale |
| `test_layout_is_fairer_than_random_placement` | The CSP earns its place |
| `test_bot_does_not_replan_every_frame` | Regression: bots once ran thousands of searches per second |
| `test_matches_between_bots_always_finish` | Regression: shared keys made matches unwinnable |
| `test_bots_still_finish_with_doors_on` | Locking doors must raise difficulty, not deadlock the AI |
| `test_doors_sit_on_busy_routes_far_more_than_chance` | Route-aware door placement earns its place |
| `test_strong_bot_goes_round_when_that_is_faster` | The wait-versus-detour decision picks correctly |
| `test_stragglers_are_cut_off_when_grace_expires` | The 30s grace period actually ends the match |

Run with `pytest tests -q`.

---

## Extending it

- **A new search algorithm** — write a generator in `src/ai/search.py` yielding
  `SearchStep`, register it in `ALGORITHMS`, add a colour in `ui/theme.py`.
  It appears in the AI Lab and the benchmark automatically.
- **A new bot tier** — add an entry to `BOT_PROFILES` in `src/config.py`.
- **New heuristic features** — extend `FEATURE_NAMES` and the stack in
  `features.feature_grid`; the assertion there catches a mismatch immediately.
  Retrain afterwards.
- **Networked multiplayer** — `Game.update(dt, intents)` already takes all
  input as one dictionary and holds no rendering state, so it can be driven by
  a server loop rather than the keyboard.
