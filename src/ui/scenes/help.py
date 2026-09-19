"""Explains the controls, the rules, and what each AI technique is doing."""

from __future__ import annotations

import pygame

from ...config import WINDOW_H, WINDOW_W
from .. import theme as T
from ..app import Scene, draw_header
from ..widgets import Button, key_hint

RULES = [
    "Both players race on one shared maze and one shared clock.",
    "Collect every key, then reach the exit - it stays locked until you hold them all.",
    "Keys are per-player: you both have to visit every key cell.",
    "Brown mud tiles cost three times as much to cross as clear floor.",
    "Diamonds are power-ups: one reveals your route, the other freezes your rival.",
]

TECHNIQUES = [
    ("Search", "BFS, DFS, Dijkstra, Greedy best-first and A* are all implemented "
               "from scratch. Each bot difficulty is literally a different one of "
               "them, so the difficulty slider shows you what they are worth."),
    ("CSP", "Where the keys and exit go is a constraint satisfaction problem: "
            "far from both spawns, roughly equidistant, well separated, spread "
            "across quadrants. Solved with AC-3 arc consistency, MRV and degree "
            "ordering, LCV, and forward checking - which is what stops one "
            "player getting a much shorter route than the other."),
    ("Learning", "A neural network learns a heuristic for A*. It is written by "
                 "hand in numpy - forward pass, backpropagation and Adam - and "
                 "its gradients are checked against finite differences. It "
                 "predicts the detour ratio between a cell and the goal, which "
                 "makes A* expand measurably fewer nodes than Manhattan does."),
]


class HelpScene(Scene):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.widgets = [
            Button(pygame.Rect(60, WINDOW_H - 92, 150, 50), "Back",
                   lambda: app.go("menu")),
            Button(pygame.Rect(226, WINDOW_H - 92, 190, 50), "Open the AI Lab",
                   lambda: app.go("lab"), primary=True),
        ]

    def draw(self, surface: pygame.Surface) -> None:
        draw_header(surface, "How it works",
                    "A maze race that doubles as a search-algorithm demonstration.")

        # -- left column: controls and rules -------------------------------
        T.draw_text(surface, "CONTROLS", (60, 130), 12, T.TEXT_FAINT, bold=True)
        y = 158
        rows = [("W A S D", "Player 1 moves"),
                ("Arrows", "Player 2 moves"),
                ("ESC", "Pause / resume"),
                ("F11", "Fullscreen")]
        for keys, description in rows:
            key_hint(surface, (60, y), keys, description)
            y += 34

        T.draw_text(surface, "RULES", (60, y + 14), 12, T.TEXT_FAINT, bold=True)
        y += 42
        for rule in RULES:
            T.draw_text(surface, "-", (60, y), 14, T.ACCENT)
            self._wrap(surface, rule, pygame.Rect(78, y, 500, 40), 14, T.TEXT_DIM)
            y += 40

        # -- right column: the AI ------------------------------------------
        panel = pygame.Rect(640, 120, 580, 540)
        T.panel(surface, panel, T.PANEL, T.PANEL_EDGE, radius=16)
        y = panel.y + 24
        for title, body in TECHNIQUES:
            T.draw_text(surface, title.upper(), (panel.x + 24, y), 12,
                        T.ACCENT_HI, bold=True)
            y += 24
            used = self._wrap(surface, body,
                              pygame.Rect(panel.x + 24, y, panel.w - 48, 200),
                              14, T.TEXT_DIM)
            y += used + 22

        super().draw(surface)

    @staticmethod
    def _wrap(surface: pygame.Surface, text: str, rect: pygame.Rect,
              size: int, colour) -> int:
        """Greedy word wrap. Returns the pixel height consumed."""
        font = T.font(size)
        words = text.split()
        line = ""
        y = rect.y
        line_h = size + 7
        for word in words:
            probe = f"{line} {word}".strip()
            if font.size(probe)[0] <= rect.w:
                line = probe
                continue
            T.draw_text(surface, line, (rect.x, y), size, colour)
            y += line_h
            line = word
        if line:
            T.draw_text(surface, line, (rect.x, y), size, colour)
            y += line_h
        return y - rect.y
