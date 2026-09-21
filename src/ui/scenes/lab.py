"""The AI Lab: watch the algorithms think, then measure them.

Three tabs:

* **Visualise** - step a chosen algorithm through a maze and watch the frontier
  spread. This is where the difference between the algorithms stops being a
  claim and becomes something you can see.
* **Benchmark** - run every algorithm over many fresh mazes and chart the
  results. The run happens on a worker thread; only the *computation* does, not
  the plotting, because matplotlib is not thread-safe.
* **Heuristic** - Manhattan against the learned heuristic as heat maps, beside
  the trade-off curve that chose the shrink factor.
"""

from __future__ import annotations

import random
import threading

import numpy as np
import pygame

from ...ai import benchmark as bench
from ...ai import search
from ...ai.heuristics import LearnedHeuristic, manhattan, manhattan_grid
from ...ai.train import load_report
from ...config import WINDOW_H, WINDOW_W
from ...core.generator import generate
from ...viz import charts
from .. import theme as T
from ..app import Scene, draw_header
from ..maze_view import MazeView
from ..widgets import Button, SegmentedControl, Slider

ALGOS = ["bfs", "dfs", "dijkstra", "greedy", "astar", "astar_learned"]
ALGO_LABELS = ["BFS", "DFS", "Dijkstra", "Greedy", "A*", "A* learned"]
ALGO_BLURB = {
    "bfs": "Expands in rings of equal step count. Optimal in steps, blind to mud.",
    "dfs": "Dives down one corridor to the end. Fast to give up, poor routes.",
    "dijkstra": "Cheapest-cost-first. Optimal with terrain, but explores widely.",
    "greedy": "Chases whatever looks closest. Quick, and easily walled in.",
    "astar": "g + h with Manhattan. Optimal, and far tighter than Dijkstra.",
    "astar_learned": "g + h with the neural heuristic. Fewer nodes, ~1% longer paths.",
}


class LabScene(Scene):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.view = MazeView(pygame.Rect(60, 210, 640, 500))
        self.maze = None
        self.start = (0, 0)
        self.goal = (0, 0)

        self.gen = None
        self.expanded: list = []
        self.frontier: tuple = ()
        self.result = None
        self.running = False
        self.accum = 0.0
        self.speed = 400.0  # expansions per second

        # Benchmark state, shared with the worker thread.
        self.bench_rows = None
        self.bench_thread: threading.Thread | None = None
        self.bench_progress = 0.0
        self.bench_message = ""
        self.bench_token = 0

        self.tabs = SegmentedControl(
            pygame.Rect(60, 116, 420, 40),
            ["Visualise", "Benchmark", "Heuristic"], 0,
            on_change=lambda i: None,
        )
        self.algo_ctl = SegmentedControl(
            pygame.Rect(724, 210, 496, 38), ALGO_LABELS, 4,
            on_change=lambda i: self.reset(), size=13,
        )
        self.speed_slider = Slider(
            pygame.Rect(724, 320, 496, 20), 20, 2000, self.speed,
            on_change=self._set_speed, fmt="{:.0f} nodes/s",
            label="Animation speed",
        )

        self.btn_run = Button(pygame.Rect(724, 366, 150, 46), "Run",
                              self.toggle_run, primary=True, icon="▶")
        self.btn_step = Button(pygame.Rect(886, 366, 120, 46), "Step",
                               self.single_step)
        self.btn_reset = Button(pygame.Rect(1018, 366, 120, 46), "Reset",
                                self.reset)
        self.btn_maze = Button(pygame.Rect(1150, 366, 70, 46), "New",
                               self.new_maze)
        self.btn_bench = Button(pygame.Rect(60, 176, 220, 44),
                                "Run benchmark", self.start_benchmark,
                                primary=True)
        self.btn_back = Button(pygame.Rect(WINDOW_W - 170, 116, 110, 40),
                               "Back", lambda: app.go("menu"))

        self.widgets = [self.tabs, self.btn_back]
        self.new_maze()

    # -- setup -------------------------------------------------------------
    def _set_speed(self, value: float) -> None:
        self.speed = value

    def new_maze(self) -> None:
        self.maze = generate(35, 25, algorithm=random.choice(
            ["backtracker", "prim", "kruskal"]))
        self.start = (0, 0)
        self.goal = (self.maze.width - 1, self.maze.height - 1)
        self.view.set_maze(self.maze)
        self.reset()

    def _heuristic(self):
        key = ALGOS[self.algo_ctl.index]
        if key == "astar_learned":
            learned = LearnedHeuristic.load()
            return learned if learned is not None else manhattan
        return manhattan

    def reset(self) -> None:
        key = ALGOS[self.algo_ctl.index]
        algorithm = "astar" if key == "astar_learned" else key
        # Run the same search once, uninterrupted, purely to get an honest
        # compute time. Timing the animated generator would measure how long
        # the animation took, not how long the algorithm took.
        self.reference = search.solve(self.maze, self.start, self.goal,
                                      algorithm, heuristic=self._heuristic())
        self.gen = search.iterate(self.maze, self.start, self.goal, algorithm,
                                  heuristic=self._heuristic())
        self.expanded = []
        self.frontier = ()
        self.result = None
        self.running = False
        self.accum = 0.0

    def toggle_run(self) -> None:
        if self.result is not None:
            self.reset()
        self.running = not self.running
        self.btn_run.label = "Pause" if self.running else "Run"
        self.btn_run.icon = "❚❚" if self.running else "▶"

    def single_step(self) -> None:
        self.running = False
        self.btn_run.label, self.btn_run.icon = "Run", "▶"
        self._advance(1)

    def _advance(self, count: int) -> None:
        for _ in range(count):
            if self.result is not None:
                return
            try:
                step = next(self.gen)
                self.expanded.append(step.current)
                self.frontier = step.frontier
            except StopIteration as stop:
                self.result = stop.value
                self.running = False
                self.btn_run.label, self.btn_run.icon = "Run", "▶"
                return

    # -- benchmark ---------------------------------------------------------
    def start_benchmark(self) -> None:
        if self.bench_thread is not None and self.bench_thread.is_alive():
            return
        self.bench_rows = None
        self.bench_progress = 0.0
        self.bench_token += 1

        def work() -> None:
            def progress(frac: float, message: str) -> None:
                self.bench_progress = frac
                self.bench_message = message

            rows = bench.run_benchmark(n_mazes=30, size=(31, 21),
                                       progress=progress)
            self.bench_rows = rows
            self.bench_progress = 1.0

        self.bench_thread = threading.Thread(target=work, daemon=True)
        self.bench_thread.start()

    # -- frame -------------------------------------------------------------
    def handle(self, event: pygame.event.Event) -> None:
        for widget in self._active_widgets():
            if widget.handle(event):
                return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.go("menu")
            elif event.key == pygame.K_SPACE and self.tabs.index == 0:
                self.toggle_run()

    def _active_widgets(self) -> list:
        common = [self.tabs, self.btn_back]
        if self.tabs.index == 0:
            return common + [self.algo_ctl, self.speed_slider, self.btn_run,
                             self.btn_step, self.btn_reset, self.btn_maze]
        if self.tabs.index == 1:
            return common + [self.btn_bench]
        return common

    def update(self, dt: float) -> None:
        for widget in self._active_widgets():
            widget.update(dt)
        if self.running and self.tabs.index == 0:
            self.accum += dt * self.speed
            count = int(self.accum)
            if count:
                self.accum -= count
                self._advance(count)

    def draw(self, surface: pygame.Surface) -> None:
        draw_header(surface, "AI Lab",
                    "Watch the algorithms search, then measure them against each other.")
        for widget in self._active_widgets():
            widget.draw(surface)

        if self.tabs.index == 0:
            self._draw_visualise(surface)
        elif self.tabs.index == 1:
            self._draw_benchmark(surface)
        else:
            self._draw_heuristic(surface)

    # -- tab 1 -------------------------------------------------------------
    def _draw_visualise(self, surface: pygame.Surface) -> None:
        self.view.draw_maze(surface)
        self.view.draw_cells(surface, self.expanded, T.VISITED_HI, alpha=125)
        self.view.draw_cells(surface, self.frontier, T.FRONTIER, alpha=190)
        if self.result is not None and self.result.found:
            self.view.draw_path(surface, self.result.path, T.PATH)
        for cell, colour in ((self.start, T.P1), (self.goal, T.GOAL)):
            r = self.view.cell_rect(cell).inflate(-self.view.cell // 3,
                                                  -self.view.cell // 3)
            pygame.draw.rect(surface, colour, r, border_radius=3)
        self.view.draw_border(surface)

        panel = pygame.Rect(724, 430, 496, 280)
        T.panel(surface, panel, T.PANEL, T.PANEL_EDGE, radius=14)
        key = ALGOS[self.algo_ctl.index]
        T.draw_text(surface, ALGO_LABELS[self.algo_ctl.index],
                    (panel.x + 20, panel.y + 18), 20, T.ALGO_COLORS[key], bold=True)
        self._wrap(surface, ALGO_BLURB[key],
                   pygame.Rect(panel.x + 20, panel.y + 48, panel.w - 40, 60),
                   13, T.TEXT_DIM)

        optimal = search.optimal_cost(self.maze, self.start, self.goal)
        rows = [
            ("nodes expanded", str(len(self.expanded))),
            ("frontier size", str(len(self.frontier))),
            ("optimal cost", f"{optimal:.0f}"),
        ]
        if self.result is not None:
            gap = self.result.cost - optimal
            rows += [
                ("path cost", f"{self.result.cost:.0f}"
                              + ("  (optimal)" if abs(gap) < 1e-6 else f"  (+{gap:.0f})")),
                ("path length", str(self.result.path_length)),
                ("compute time", f"{self.reference.elapsed_ms:.2f} ms"),
            ]
        y = panel.y + 108
        for label, value in rows:
            T.draw_text(surface, label, (panel.x + 20, y), 14, T.TEXT_DIM)
            T.draw_text(surface, value, (panel.right - 20, y + 7), 15, T.TEXT,
                        mono=True, right=True)
            y += 27

        legend = [("expanded", T.VISITED_HI), ("frontier", T.FRONTIER),
                  ("path", T.PATH)]
        x = 60
        for label, colour in legend:
            pygame.draw.rect(surface, colour, pygame.Rect(x, WINDOW_H - 52, 14, 14),
                             border_radius=3)
            T.draw_text(surface, label, (x + 22, WINDOW_H - 52), 13, T.TEXT_DIM)
            x += 22 + T.font(13).size(label)[0] + 24
        T.draw_text(surface, "SPACE to run/pause", (700, WINDOW_H - 52), 13,
                    T.TEXT_FAINT, right=True)

    # -- tab 2 -------------------------------------------------------------
    def _draw_benchmark(self, surface: pygame.Surface) -> None:
        busy = self.bench_thread is not None and self.bench_thread.is_alive()
        self.btn_bench.enabled = not busy

        if busy:
            bar = pygame.Rect(300, 190, 500, 16)
            T.progress_bar(surface, bar, self.bench_progress)
            T.draw_text(surface, self.bench_message or "working...",
                        (816, 198), 14, T.TEXT_DIM)

        if self.bench_rows is None:
            if not busy:
                T.draw_text(surface,
                            "Run the benchmark to compare all five algorithms "
                            "over 30 freshly generated mazes.",
                            (60, 260), 16, T.TEXT_DIM)
                T.draw_text(surface,
                            "Every result is scored against a ground-truth "
                            "Dijkstra sweep, so 'optimal' means genuinely optimal.",
                            (60, 288), 14, T.TEXT_FAINT)
            return

        rows = self.bench_rows
        chart = charts.cached(("bench-bars", self.bench_token),
                              lambda size: charts.algorithm_comparison(rows, size),
                              (700, 300))
        surface.blit(chart, (56, 236))
        spread = charts.cached(("bench-box", self.bench_token),
                               lambda size: charts.expansion_histogram(rows, size),
                               (450, 300))
        surface.blit(spread, (770, 236))

        table = pygame.Rect(56, 552, 1164, 168)
        T.panel(surface, table, T.PANEL, T.PANEL_EDGE, radius=14)
        headers = ["algorithm", "expanded", "cost / optimal", "optimal paths",
                   "peak frontier", "time"]
        xs = [80, 340, 500, 700, 900, 1080]
        for label, x in zip(headers, xs):
            T.draw_text(surface, label.upper(), (x, table.y + 16), 11,
                        T.TEXT_FAINT, bold=True)
        y = table.y + 42
        for key, row in rows.items():
            colour = T.ALGO_COLORS.get(key, T.TEXT)
            pygame.draw.rect(surface, colour, pygame.Rect(62, y + 4, 4, 14),
                             border_radius=2)
            values = [
                row.label,
                f"{row.mean_expanded:.1f}",
                f"{row.mean_cost_ratio:.3f}",
                f"{row.optimal_rate * 100:.0f}%",
                f"{row.mean_frontier:.1f}",
                f"{row.mean_ms:.2f} ms",
            ]
            for value, x in zip(values, xs):
                T.draw_text(surface, value, (x, y), 14,
                            T.TEXT if x == xs[0] else T.TEXT_DIM, mono=(x != xs[0]))
            y += 22

    # -- tab 3 -------------------------------------------------------------
    def _draw_heuristic(self, surface: pygame.Surface) -> None:
        learned = LearnedHeuristic.load()
        report = load_report()

        if learned is None or report is None:
            T.draw_text(surface, "The heuristic has not been trained yet.",
                        (60, 220), 20, T.WARNING)
            T.draw_text(surface,
                        "Run  python main.py train  or open the Train screen "
                        "from the menu.",
                        (60, 254), 15, T.TEXT_DIM)
            return

        man = manhattan_grid(self.maze, self.goal)
        lrn = learned.grid_for(self.maze, self.goal)
        true = self.maze.cost_to(self.goal)

        cell = min(560 // self.maze.width, 240 // self.maze.height)
        for i, (title, field_) in enumerate((("Manhattan h", man),
                                             ("Learned h", lrn),
                                             ("True h*", true))):
            box = pygame.Rect(60 + i * 390, 200, 370, 250)
            T.panel(surface, box, T.PANEL, T.PANEL_EDGE, radius=12)
            T.draw_text(surface, title, (box.x + 16, box.y + 12), 15, T.TEXT,
                        bold=True)
            view = MazeView(pygame.Rect(box.x + 12, box.y + 40, box.w - 24,
                                        box.h - 56))
            view.set_maze(self.maze)
            # Deliberately no wall layer: at this size the wall strokes are
            # thicker than the cells and swamp the colour, and the field itself
            # already traces the corridors.
            view.draw_heatmap(surface, field_, alpha=255)
            view.draw_border(surface)
            finite = field_[np.isfinite(field_)]
            T.draw_text(surface, f"max {finite.max():.0f}",
                        (box.right - 16, box.bottom - 22), 12, T.TEXT_FAINT,
                        right=True)

        if report.alpha_sweep:
            chart = charts.cached(
                ("alpha", report.alpha),
                lambda size: charts.alpha_tradeoff(report.alpha_sweep, size),
                (620, 250),
            )
            surface.blit(chart, (60, 468))

        info = pygame.Rect(700, 468, 520, 250)
        T.panel(surface, info, T.PANEL, T.PANEL_EDGE, radius=14)
        T.draw_text(surface, "Learned heuristic", (info.x + 20, info.y + 16),
                    17, T.TEXT, bold=True)
        chosen = next((r for r in report.alpha_sweep if r.get("chosen")), None)
        rows = [
            ("shrink alpha", f"{report.alpha:.2f}"),
            ("strength vs Manhattan", f"{report.mean_learned_over_manhattan:.2f}x"),
            ("true detour ratio", f"{report.mean_true_ratio:.2f}x"),
            ("validation MAE", f"{report.val_mae_cells[-1]:.1f} cells"
                               if report.val_mae_cells else "-"),
            ("gradient check", f"{report.gradient_check_error:.1e}"),
        ]
        if chosen:
            rows.insert(1, ("expansions saved", f"{chosen['reduction_pct']:.1f}%"))
            rows.insert(2, ("optimal paths", f"{chosen['optimal_rate'] * 100:.0f}%"))
        y = info.y + 50
        for label, value in rows:
            T.draw_text(surface, label, (info.x + 20, y), 14, T.TEXT_DIM)
            T.draw_text(surface, value, (info.right - 20, y + 7), 14, T.TEXT,
                        mono=True, right=True)
            y += 26

    @staticmethod
    def _wrap(surface, text, rect, size, colour) -> None:
        font = T.font(size)
        words = text.split()
        line, y = "", rect.y
        for word in words:
            probe = f"{line} {word}".strip()
            if font.size(probe)[0] <= rect.w:
                line = probe
                continue
            T.draw_text(surface, line, (rect.x, y), size, colour)
            y += size + 6
            line = word
        if line:
            T.draw_text(surface, line, (rect.x, y), size, colour)
