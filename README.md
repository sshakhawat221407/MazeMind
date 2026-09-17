# MazeMind — Multiplayer Maze Solver

A two-player maze racing game, built entirely in Python, that doubles as a
working demonstration of **search algorithms**, **constraint satisfaction** and a
**neural network written from first principles**.

Two players share one keyboard and one maze. Collect every key, then reach the
exit. Your opponent is either the person next to you or one of five AI
difficulty tiers — and each tier is literally a different search algorithm, so
the difficulty slider is a live comparison of what those algorithms are worth.

---

## Contents

- [Features](#features)
- [Course requirements](#course-requirements)
- [Getting started](#getting-started)
- [How to play](#how-to-play)
- [The AI](#the-ai)
- [Project structure](#project-structure)
- [Team and branches](#team-and-branches)
- [Git workflow](#git-workflow)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Features

- **Same-screen multiplayer** — two humans on one keyboard, or a human against
  an AI opponent with five difficulty tiers.
- **Procedural mazes** — three generation algorithms, loops for real route
  choice, and mud terrain that costs three times as much to cross.
- **Fair layouts** — keys and the exit are placed by a CSP solver so neither
  player starts with a shorter route.
- **Locking doors** (optional) — fixed doors on busy routes that lock for 4
  seconds after anyone passes through.
- **Everybody finishes** — the first player home wins, and the rest get 30
  seconds to complete their own run.
- **Detailed results** — click either player afterwards for a full stat sheet
  and a map of the ground they covered against the optimal tour.
- **AI Lab** — watch BFS, DFS, Dijkstra, Greedy and A\* search a maze step by
  step, then benchmark them against each other.
- **Learned heuristic** — a neural network trained in-app that makes A\* expand
  fewer nodes, with live training curves.

---

## Course requirements

| Requirement | How MazeMind meets it |
| --- | --- |
| Built entirely in Python | 100% Python. Libraries: `pygame` (window and drawing), `numpy` (array maths), `matplotlib` (charts). |
| UI design | Seven pygame screens with animated widgets, a custom icon set and in-app charts. |
| Git repository | Developed in three feature branches, merged into `main` — see [Git workflow](#git-workflow). |
| AI/ML, not a black box | The neural network, backpropagation, optimiser, CSP solver and every search algorithm are written from scratch. No ML or AI library is used. |
| A\*, BFS, DFS, CSP | All four implemented, plus Dijkstra and Greedy best-first, and benchmarked against each other. |

---

## Getting started

### Windows — easiest

Double-click **`Play MazeMind.bat`**.

The first time on a machine it finds your Python, creates a virtual
environment, installs the dependencies and launches the game. After that it
just launches. You need Python 3.10 or newer installed, and an internet
connection the first time.

### Manual setup — any platform

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe main.py
```

On macOS or Linux use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

### Moving the project to another PC

Copy the project folder **without** the `.venv` folder — or leave it in, since
the launcher detects a virtual environment built on a different machine and
rebuilds it. A `.venv` cannot be moved between computers: it stores the absolute
path of the Python that created it, so on another PC every command fails with
`No Python at '...'`.

### Other commands

| Command | What it does |
| --- | --- |
| `python main.py` | Launch the game |
| `python main.py --scene lab` | Open straight into the AI Lab |
| `python main.py train` | Retrain the learned heuristic and print a report |
| `python main.py bench` | Benchmark the search algorithms in the terminal |
| `python -m pytest tests` | Run the 146-test suite |

The learned heuristic ships pre-trained in `models/`, so training is optional.

---

## How to play

### Controls

| Key | Action |
| --- | --- |
| `W` `A` `S` `D` | Player 1 moves |
| Arrow keys | Player 2 moves |
| `ESC` | Pause / resume |
| `F11` | Fullscreen |

### Rules

- **Collect every key, then reach the exit.** The exit stays locked until you
  hold them all.
- **Keys are per-player** — both racers must visit every key cell.
- **Mud** (brown tiles) takes three times as long to cross as clear floor.
- **Power-ups:** the eye briefly shows your best route; the snowflake freezes
  your opponent for 2 seconds.
- **Everybody finishes.** The first player home wins, then everyone still
  running has 30 seconds to finish before the match ends.

### Locking doors

An optional setting on the match setup screen. A fixed set of doors is placed in
the maze, drawn as silver posts with a dashed line. **Walk through a door and it
locks for 4 seconds** — for both players, including you — then unlocks again. A
locked door turns red, with a bar that drains as the lock runs down.

Doors go on the corridors that the best routes actually use, so each one is a
real decision: go through now and shut your rival out, or go around. Bigger
mazes get more doors — 4 on Small, 7 on Medium, 13 on Large, 22 on Huge.

### Opponents

| Tier | Plans with | Notes |
| --- | --- | --- |
| Rookie | Depth-first search | Takes long, winding routes and makes mistakes |
| Scout | Breadth-first search | Shortest in steps, but walks straight through mud |
| Hunter | Greedy best-first | Fast, but easily trapped behind walls |
| Tactician | A\* with Manhattan distance | Optimal routes |
| Oracle | A\* with the learned heuristic | Near-optimal routes, fastest pace |

---

## The AI

Everything below is implemented from scratch. `numpy` is used for array
arithmetic only.

### 1. Search algorithms

`src/ai/search.py` — BFS, DFS, Dijkstra, Greedy best-first and A\*. Measured over
30 freshly generated 31×21 mazes, scored against the true optimum:

| Algorithm | Nodes expanded | Path cost ÷ optimal | Optimal paths |
| --- | --- | --- | --- |
| BFS | 595.4 | 1.041 | 57% |
| DFS | 279.8 | 1.786 | 7% |
| Dijkstra | 579.0 | 1.000 | 100% |
| Greedy | 146.4 | 1.221 | 20% |
| A\* (Manhattan) | 365.8 | 1.000 | 100% |
| A\* (learned) | 314.3 | 1.024 | 83% |

- **A\* matches Dijkstra's optimality while expanding 37% fewer nodes** — the
  value of a good heuristic, in one row.
- **BFS is not optimal here.** It minimises *steps*, but mud costs more, so the
  shortest path in steps is often not the cheapest. That is exactly the
  difference Dijkstra exists to handle.

### 2. Constraint satisfaction

`src/ai/csp.py` — where the keys and exit go is a constraint satisfaction
problem: far from both spawns, roughly equidistant from both, spread apart and
across different quadrants. Solved with **AC-3** arc consistency, **MRV**
variable ordering, **LCV** value ordering and **forward checking**. If the strict
version has no solution, a relaxation ladder loosens it step by step.

| Placement | Gap between the two players' route lengths |
| --- | --- |
| Random | 101.9 |
| CSP | 21.7 |

**79% fairer** than random placement, measured over 10 mazes.

### 3. Machine learning — a learned heuristic for A\*

`src/ai/nn.py`, `features.py`, `dataset.py`, `train.py` — a neural network with
every layer's forward pass and gradient written by hand, a hand-written Adam
optimiser and a quantile loss:

```
21 features -> Dense(96) -> ReLU -> Dense(64) -> ReLU -> Dense(1) -> Softplus -> +1
```

- **Correctness:** the hand-written gradients match numerical gradients to
  within **7.9 × 10⁻⁸**, and a test fails if they ever stop matching.
- **What it predicts:** the *detour ratio* between a cell and the goal — how
  much longer the real route is than the straight-line distance. A first
  version predicted raw distance and ended up weaker than the simple Manhattan
  heuristic it was meant to beat; the ratio is far easier to learn.
- **Result:** A\* with the learned heuristic expands **16.7% fewer nodes** than
  with Manhattan, while 87% of paths stay exactly optimal and the average path
  is only 0.4% longer.
- **Honest trade-off:** a learned heuristic cannot guarantee optimal paths, so a
  shrink factor is tuned by running real searches. The full speed-versus-
  optimality curve is plotted in the AI Lab.

### 4. AI opponent and door placement

- `src/ai/bot.py` — each difficulty tier plans with a different algorithm. At a
  locked door, weak tiers wait; **Tactician** and **Oracle** compare the time to
  wait against the time to go around, and take the faster option.
- `src/core/doors.py` — doors are placed on the corridors the optimal routes
  funnel through. Measured over 20 mazes, **80% of doors sit on an optimal
  route**, against 35% for random placement.

---

## Project structure

```
MazeMind/
├── Play MazeMind.bat        one-click launcher and first-time setup (Windows)
├── main.py                  entry point: play / train / bench
├── requirements.txt
├── docs/
│   └── ARCHITECTURE.md      design decisions in depth
├── models/                  trained neural-network weights and training report
├── src/
│   ├── config.py            every tunable constant, in one place
│   ├── core/                GAME ENGINE
│   │   ├── maze.py          grid, walls, terrain, distance fields
│   │   ├── generator.py     backtracker / Prim / Kruskal, loops, mud
│   │   ├── game.py          players, keys, power-ups, doors, finishing
│   │   └── doors.py         route-aware locking door placement
│   ├── ai/                  ARTIFICIAL INTELLIGENCE
│   │   ├── search.py        BFS, DFS, Dijkstra, Greedy, A*
│   │   ├── heuristics.py    Manhattan, Euclidean, Chebyshev, learned
│   │   ├── csp.py           AC-3, MRV, LCV, forward checking, layout
│   │   ├── bot.py           the AI opponent
│   │   ├── nn.py            neural network: layers, backprop, Adam
│   │   ├── features.py      21 features computed for every cell at once
│   │   ├── dataset.py       training data from Dijkstra sweeps
│   │   ├── train.py         training, calibration, audit
│   │   └── benchmark.py     head-to-head algorithm measurement
│   ├── ui/                  USER INTERFACE
│   │   ├── app.py           window, main loop, screen switching
│   │   ├── theme.py         colours, fonts, drawing helpers
│   │   ├── widgets.py       buttons, toggles, sliders
│   │   ├── maze_view.py     maze renderer and icons
│   │   └── scenes/          menu, setup, game, results, lab, train, help
│   └── viz/
│       └── charts.py        matplotlib charts drawn inside the game
└── tests/                   146 automated tests
```

---

## Team and branches

The project is split into three parts. Each part was developed in its own branch
and merged into `main`. No file belongs to more than one part, so the branches
merge without conflicts.

| Part | Branch | Member | Responsibility |
| --- | --- | --- | --- |
| Foundation | `main` | Error 404 | README, shared config, launcher, docs |
| Part 1 | `part-1-game-engine` | Samsad Ibne Shakhawat Nuhash | Game engine, AI opponent, CSP |
| Part 2 | `part-2-search-ml` | Shreedhara Datta | Search algorithms, machine learning, AI Lab |
| Part 3 | `part-3-user-interface` | Rifah Tasniea Katha | User interface and application |

### Foundation — `main`

Shared files every part depends on, committed first so all three branches start
from the same base.

`README.md` · `requirements.txt` · `Play MazeMind.bat` · `docs/ARCHITECTURE.md` ·
`src/config.py` · `.gitignore` · `.gitattributes` · package markers

### Part 1 — Game Engine, AI Opponent and CSP

**Branch:** `part-1-game-engine` · **AI techniques:** constraint satisfaction,
agent decision-making

- Maze data structure, terrain and distance fields
- Three maze generation algorithms, loops and mud
- Game rules: movement, keys, power-ups, locking doors, finishing and the
  30-second grace period, player statistics
- Route-aware door placement
- CSP solver (AC-3, MRV, LCV, forward checking) for fair key and exit placement
- AI opponent: difficulty tiers, target selection, the wait-or-go-around
  decision at locked doors

**Files:** `src/core/` (all) · `src/ai/csp.py` · `src/ai/bot.py` ·
`tests/test_maze.py` · `tests/test_game.py` · `tests/test_doors.py` ·
`tests/test_csp.py`

### Part 2 — Search Algorithms and Machine Learning

**Branch:** `part-2-search-ml` · **AI techniques:** BFS, DFS, Dijkstra, Greedy,
A\*, neural network, learned heuristic

- BFS, DFS, Dijkstra, Greedy best-first and A\*, with step-by-step animation
- Classical heuristics: Manhattan, Euclidean, Chebyshev
- Neural network from scratch: layers, backpropagation, Adam, gradient check
- Feature extraction, training data, training, calibration and audit
- Benchmarking and the trained model
- The AI Lab and Training screens, and their charts

**Files:** `src/ai/` (all except `csp.py` and `bot.py`) · `models/` ·
`src/ui/scenes/lab.py` · `src/ui/scenes/train.py` · `src/viz/` ·
`tests/test_search.py` · `tests/test_nn.py`

### Part 3 — User Interface and Application

**Branch:** `part-3-user-interface` · **Focus:** UI design

- Application entry point and command-line options
- Window, main loop and animated screen transitions
- Visual theme, fonts and reusable widgets (buttons, toggles, sliders)
- Maze renderer, icon set, door and player animation
- Menu, match setup, game, results and help screens

**Files:** `main.py` · `src/ui/app.py` · `src/ui/theme.py` ·
`src/ui/widgets.py` · `src/ui/maze_view.py` · `src/ui/scenes/` (menu, setup,
game, results, help)

---

## Git workflow

```
main ──●──────────────────────────●──────────●──────────●──▶ complete game
       │                          ▲          ▲          ▲
       ├── part-1-game-engine ────●          │          │
       ├── part-2-search-ml ─────────────────●          │
       └── part-3-user-interface ───────────────────────●
```

1. The foundation is committed to `main`.
2. Each member creates their branch from `main` and commits their part.
3. The branches are merged into `main` in order: **Part 1 → Part 2 → Part 3**.

The parts depend on each other — the interface uses the engine and the AI, and
the AI uses the engine — so **the game runs once all three are merged**, not from
any single branch on its own.

---

## Testing

```bash
python -m pytest tests
```

146 tests, about 12 seconds. They cover every search algorithm's correctness and
optimality, the neural network's gradients, CSP fairness, door placement and
locking, the finishing rules, and regressions for bugs found during development.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit together,
  and the reasoning behind the key design decisions.
