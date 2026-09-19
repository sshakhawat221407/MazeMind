"""Match configuration, with a live preview of the maze you are about to play.

The preview is a real maze built with the chosen settings and laid out by the
real CSP solver, so the fairness numbers shown underneath it are the ones that
will actually apply. Regenerating is cheap enough to do on every change.
"""

from __future__ import annotations

import random

import pygame

from ...config import (
    BOT_PROFILES,
    DIFFICULTY_SIZES,
    DOOR_LOCK_DURATION,
    WINDOW_H,
    WINDOW_W,
)
from ...core.game import PlayerKind, new_game, spawn_points
from ...core.generator import generate
from .. import theme as T
from ..app import Scene, draw_header
from ..maze_view import MazeView
from ..widgets import Button, SegmentedControl, Slider, Toggle

GENERATORS = ["backtracker", "prim", "kruskal"]
GENERATOR_LABELS = ["Backtracker", "Prim", "Kruskal"]
GENERATOR_BLURB = {
    "backtracker": "Long winding corridors, few junctions.",
    "prim": "Bushy and branch-heavy - punishes depth-first search.",
    "kruskal": "Uniform texture, no strong grain.",
}


class SetupScene(Scene):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.view = MazeView(pygame.Rect(660, 150, 560, 420))
        self.preview = None
        self.preview_game = None
        self.dirty = True

        s = app.settings
        x, w = 60, 520

        self.size_ctl = SegmentedControl(
            pygame.Rect(x, 174, w, 40), list(DIFFICULTY_SIZES),
            index=list(DIFFICULTY_SIZES).index(s["size"]),
            on_change=self._on_size,
        )
        self.gen_ctl = SegmentedControl(
            pygame.Rect(x, 256, w, 40), GENERATOR_LABELS,
            index=GENERATORS.index(s["generator"]),
            on_change=self._on_generator,
        )
        self.opp_ctl = SegmentedControl(
            pygame.Rect(x, 360, w, 40), ["Player 2 (human)", "AI opponent"],
            index=0 if s["opponent"] == "Human" else 1,
            on_change=self._on_opponent,
        )
        self.diff_ctl = SegmentedControl(
            pygame.Rect(x, 442, w, 40), list(BOT_PROFILES),
            index=list(BOT_PROFILES).index(s["difficulty"]),
            on_change=self._on_difficulty, size=13,
        )
        # The slider draws its own caption 22px above its track, so it needs
        # clear air above it as well as below.
        self.keys_slider = Slider(
            pygame.Rect(x, 546, w, 20), 1, 5, float(s["keys"]),
            on_change=self._on_keys, step=1, fmt="{:.0f}", label="Keys to collect",
        )
        self.plan_toggle = Toggle(
            pygame.Rect(x, 590, 320, 28), "Show the bot's planned route",
            bool(s["show_bot_plan"]), on_change=self._on_plan,
        )
        self.doors_toggle = Toggle(
            pygame.Rect(x, 632, 380, 28), "Locking doors",
            bool(s["doors"]), on_change=self._on_doors,
        )

        self.widgets = [
            self.size_ctl, self.gen_ctl, self.opp_ctl, self.diff_ctl,
            self.keys_slider, self.plan_toggle, self.doors_toggle,
            Button(pygame.Rect(x, WINDOW_H - 92, 210, 52), "Start match",
                   self._start, primary=True, icon="▶", size=18),
            Button(pygame.Rect(x + 226, WINDOW_H - 92, 130, 52), "Back",
                   lambda: app.go("menu")),
            Button(pygame.Rect(x + 372, WINDOW_H - 92, 148, 52), "New maze",
                   self._regen, icon="⟳"),
        ]
        self._sync_enabled()

    # -- settings callbacks ------------------------------------------------
    def _on_size(self, i: int) -> None:
        self.app.settings["size"] = list(DIFFICULTY_SIZES)[i]
        self.dirty = True

    def _on_generator(self, i: int) -> None:
        self.app.settings["generator"] = GENERATORS[i]
        self.dirty = True

    def _on_opponent(self, i: int) -> None:
        self.app.settings["opponent"] = "Human" if i == 0 else "Bot"
        self._sync_enabled()

    def _on_difficulty(self, i: int) -> None:
        self.app.settings["difficulty"] = list(BOT_PROFILES)[i]

    def _on_keys(self, value: float) -> None:
        self.app.settings["keys"] = int(value)
        self.dirty = True

    def _on_plan(self, value: bool) -> None:
        self.app.settings["show_bot_plan"] = value

    def _on_doors(self, value: bool) -> None:
        self.app.settings["doors"] = value
        # Rebuild so the preview shows where the doors would actually go.
        self.dirty = True

    def _sync_enabled(self) -> None:
        is_bot = self.app.settings["opponent"] == "Bot"
        self.diff_ctl.enabled = is_bot
        self.plan_toggle.enabled = is_bot

    def _regen(self) -> None:
        self.dirty = True

    def _start(self) -> None:
        self.app.go("game")

    def on_enter(self, **kwargs) -> None:
        self.dirty = True

    # -- preview -----------------------------------------------------------
    def _rebuild(self) -> None:
        s = self.app.settings
        w, h = DIFFICULTY_SIZES[s["size"]]
        maze = generate(w, h, algorithm=s["generator"])
        specs = [("Player 1", PlayerKind.HUMAN, None),
                 ("Preview", PlayerKind.BOT, "Tactician")]
        self.preview_game = new_game(maze, specs, n_keys=int(s["keys"]),
                                     rng=random.Random(),
                                     doors_enabled=bool(s.get("doors", False)))
        self.preview = maze
        self.view.set_maze(maze)
        self.dirty = False

    def update(self, dt: float) -> None:
        super().update(dt)
        if self.dirty:
            self._rebuild()

    # -- drawing -----------------------------------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        draw_header(surface, "Match setup",
                    "Objectives are placed by the CSP solver, so both routes are comparable.")

        for label, y in (("MAZE SIZE", 150), ("GENERATOR", 232),
                         ("OPPONENT", 336), ("BOT DIFFICULTY", 418)):
            T.draw_text(surface, label, (60, y), 12, T.TEXT_FAINT, bold=True)

        super().draw(surface)

        gen = self.app.settings["generator"]
        T.draw_text(surface, GENERATOR_BLURB[gen], (60, 304), 13, T.TEXT_DIM)

        if self.app.settings["opponent"] == "Bot":
            profile = self.app.settings["difficulty"]
            algo, delay, mistakes = BOT_PROFILES[profile]
            names = {"dfs": "depth-first search", "bfs": "breadth-first search",
                     "greedy": "greedy best-first", "astar": "A* + Manhattan",
                     "astar_learned": "A* + the learned heuristic"}
            detail = (f"Plans with {names.get(algo, algo)}   |   "
                      f"{delay * 1000:.0f} ms per step")
            if mistakes:
                detail += f"   |   {mistakes * 100:.0f}% blunder rate"
            T.draw_text(surface, detail, (60, 492), 13, T.TEXT_DIM)
        else:
            T.draw_text(surface, "Two players share the keyboard.",
                        (60, 492), 13, T.TEXT_DIM)

        if self.app.settings["doors"]:
            count = len(self.preview_game.doors) if self.preview_game else 0
            T.draw_text(surface,
                        f"{count} doors on busy routes. Pass one and it locks "
                        f"for {DOOR_LOCK_DURATION:.0f}s - for both players.",
                        (60, 664), 12, T.WARNING)
        else:
            T.draw_text(surface,
                        "No doors. Turn this on for a harder, more tactical match.",
                        (60, 664), 12, T.TEXT_FAINT)

        self._draw_preview(surface)

    def _draw_preview(self, surface: pygame.Surface) -> None:
        panel = pygame.Rect(640, 120, 580, 500)
        T.panel(surface, panel, T.PANEL, T.PANEL_EDGE, radius=16)
        T.draw_text(surface, "Preview", (panel.x + 22, panel.y + 16), 16,
                    T.TEXT, bold=True)

        if self.preview_game is None:
            return
        game = self.preview_game
        self.view.draw_maze(surface)
        self.view.draw_spawn_markers(surface, game)
        self.view.draw_doors(surface, game)
        self.view.draw_objects(surface, game, self.app.time)
        self.view.draw_border(surface)

        gap = game.fairness_gap
        y = panel.bottom - 62
        T.draw_text(surface, "CSP placement", (panel.x + 22, y), 12,
                    T.TEXT_FAINT, bold=True)
        colour = T.SUCCESS if gap < 30 else T.WARNING
        T.draw_text(surface,
                    f"route-length gap between players: {gap:.0f} cost units",
                    (panel.x + 22, y + 18), 13, colour)
        T.draw_text(surface, game.csp_summary, (panel.x + 22, y + 38), 11,
                    T.TEXT_FAINT, mono=True)
