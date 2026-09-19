"""Title screen.

The right-hand panel runs a live A* search on a small maze, looping forever.
It is the cheapest possible demonstration of what the project is: the frontier
spreads, the path resolves, and the counters tick up while the player decides
what to click.
"""

from __future__ import annotations

import random

import pygame

from ...config import WINDOW_H, WINDOW_W
from ...ai import search
from ...ai.heuristics import LearnedHeuristic, manhattan
from ...ai.train import load_report
from ...core.generator import generate
from .. import theme as T
from ..app import Scene
from ..maze_view import MazeView
from ..widgets import Button


class MenuScene(Scene):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.view = MazeView(pygame.Rect(690, 172, 520, 436))
        self.demo_maze = None
        self.demo_gen = None
        self.demo_result = None
        self.expanded: list = []
        self.path: list = []
        self.accum = 0.0
        self.hold = 0.0
        self.rng = random.Random()

        x, y, w, h = 80, 250, 300, 54
        self.widgets = [
            Button(pygame.Rect(x, y, w, h), "Play a match",
                   lambda: app.go("setup"), primary=True, icon="▶", size=19),
            Button(pygame.Rect(x, y + 68, w, h), "AI Lab",
                   lambda: app.go("lab"), icon="◉"),
            Button(pygame.Rect(x, y + 68 * 2, w, h), "Train heuristic",
                   lambda: app.go("train"), icon="⚙"),
            Button(pygame.Rect(x, y + 68 * 3, w, h), "How it works",
                   lambda: app.go("help"), icon="?"),
            Button(pygame.Rect(x, y + 68 * 4, w, h), "Quit",
                   app.quit, icon="×"),
        ]
        self.new_demo()

    # -- demo animation ----------------------------------------------------
    def new_demo(self) -> None:
        self.demo_maze = generate(21, 19, algorithm=self.rng.choice(
            ["backtracker", "prim", "kruskal"]), seed=self.rng.randrange(1 << 30))
        self.view.set_maze(self.demo_maze)
        start, goal = (0, 0), (20, 18)
        self.demo_start, self.demo_goal = start, goal
        learned = LearnedHeuristic.load()
        heuristic = learned if learned is not None else manhattan
        # Uninterrupted run, only for an honest compute time (see search.run).
        self.reference = search.solve(self.demo_maze, start, goal, "astar",
                                      heuristic=heuristic)
        self.demo_gen = search.iterate(self.demo_maze, start, goal, "astar",
                                       heuristic=heuristic)
        self.demo_label = "A* + learned h" if learned else "A* + Manhattan"
        self.expanded = []
        self.path = []
        self.demo_result = None
        self.hold = 0.0

    def update(self, dt: float) -> None:
        super().update(dt)
        if self.demo_result is not None:
            self.hold += dt
            if self.hold > 2.4:
                self.new_demo()
            return

        self.accum += dt
        # Fixed number of expansions per second, independent of frame rate.
        budget = int(self.accum / 0.006)
        if budget <= 0:
            return
        self.accum -= budget * 0.006
        for _ in range(budget):
            try:
                step = next(self.demo_gen)
                self.expanded.append(step.current)
            except StopIteration as stop:
                self.demo_result = stop.value
                self.path = list(self.demo_result.path)
                break

    # -- drawing -----------------------------------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        T.draw_text(surface, "MazeMind", (80, 96), 66, T.TEXT, bold=True)
        T.draw_text(surface, "Multiplayer maze solver", (84, 172), 20, T.ACCENT_HI)
        T.draw_text(
            surface,
            "Race a friend or an algorithm through a maze built by a constraint solver.",
            (84, 200), 15, T.TEXT_DIM,
        )
        super().draw(surface)

        panel = pygame.Rect(660, 120, 560, 530)
        T.panel(surface, panel, T.PANEL, T.PANEL_EDGE, radius=16)
        T.draw_text(surface, self.demo_label, (panel.x + 22, panel.y + 18),
                    16, T.TEXT, bold=True)
        T.draw_text(surface, "searching live", (panel.right - 22, panel.y + 26),
                    14, T.TEXT_FAINT, right=True)

        self.view.draw_maze(surface)
        self.view.draw_cells(surface, self.expanded, T.VISITED_HI, alpha=120)
        if self.path:
            self.view.draw_path(surface, self.path, T.PATH)
        for cell, colour in ((self.demo_start, T.P1), (self.demo_goal, T.GOAL)):
            r = self.view.cell_rect(cell).inflate(-self.view.cell // 3,
                                                  -self.view.cell // 3)
            pygame.draw.rect(surface, colour, r, border_radius=3)
        self.view.draw_border(surface)

        stats = f"expanded {len(self.expanded)}"
        if self.demo_result is not None:
            stats += (f"   cost {self.demo_result.cost:.0f}"
                      f"   {self.reference.elapsed_ms:.2f} ms")
        T.draw_text(surface, stats, (panel.x + 22, panel.bottom - 32), 14,
                    T.TEXT_DIM, mono=True)

        self._draw_status(surface)

    def _draw_status(self, surface: pygame.Surface) -> None:
        report = load_report()
        y = WINDOW_H - 118
        T.draw_text(surface, "MODEL STATUS", (80, y), 12, T.TEXT_FAINT, bold=True)
        if report is None:
            T.draw_text(surface, "Heuristic not trained - bots fall back to Manhattan",
                        (80, y + 20), 14, T.WARNING)
            T.draw_text(surface, "Open 'Train heuristic' to build it (about a minute)",
                        (80, y + 42), 13, T.TEXT_FAINT)
        else:
            saved = report.alpha_sweep or []
            chosen = next((r for r in saved if r.get("chosen")), None)
            gain = f"{chosen['reduction_pct']:.1f}% fewer expansions" if chosen else ""
            T.draw_text(surface, f"Learned heuristic ready  (alpha {report.alpha:.2f})",
                        (80, y + 20), 14, T.SUCCESS)
            T.draw_text(surface,
                        f"{gain}   |   gradient check {report.gradient_check_error:.1e}",
                        (80, y + 42), 13, T.TEXT_FAINT, mono=True)
