"""Window, main loop and scene management.

A single pygame window hosts every screen. Scenes are created lazily and cached,
so returning to the menu does not rebuild it, but a scene can opt into a reset by
implementing :meth:`Scene.on_enter`.

Scene changes are deferred to the end of the frame and cross-faded. Switching
mid-event would mutate the scene list while it is being iterated, and an
instant cut between two dark screens is hard to read.
"""

from __future__ import annotations

from typing import Callable

import pygame

from ..config import DOORS_ENABLED_DEFAULT, FPS, TITLE, WINDOW_H, WINDOW_W
from . import theme as T
from .widgets import ToastStack


class Scene:
    """Base screen. Subclasses override the hooks they need."""

    def __init__(self, app: "App") -> None:
        self.app = app
        self.widgets: list = []

    # -- lifecycle ---------------------------------------------------------
    def on_enter(self, **kwargs) -> None:
        pass

    def on_exit(self) -> None:
        pass

    # -- frame -------------------------------------------------------------
    def handle(self, event: pygame.event.Event) -> None:
        for widget in self.widgets:
            if widget.handle(event):
                return

    def update(self, dt: float) -> None:
        for widget in self.widgets:
            widget.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        for widget in self.widgets:
            widget.draw(surface)


class App:
    """Owns the window, the clock, and the scene stack."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.running = True
        self.time = 0.0

        self.factories: dict[str, Callable[[App], Scene]] = {}
        self.scenes: dict[str, Scene] = {}
        self.scene: Scene | None = None
        self.scene_name = ""

        self._pending: tuple[str, dict] | None = None
        self._fade = 0.0        # 0 = clear, 1 = fully covered
        self._fading_out = False

        self.toasts = ToastStack()
        # Capture mode: run a fixed number of frames then save a PNG. Used by
        # ``main.py --shot`` to verify rendering without a human at the keyboard.
        self.max_frames: int | None = None
        self.shot_path: str | None = None
        self._frames = 0
        # Shared match configuration, edited on the setup screen.
        self.settings: dict[str, object] = {
            "size": "Medium",
            "generator": "backtracker",
            "opponent": "Bot",
            "difficulty": "Tactician",
            "keys": 3,
            "show_bot_plan": False,
            "doors": DOORS_ENABLED_DEFAULT,
        }
        self._background: pygame.Surface | None = None

    # -- registration ------------------------------------------------------
    def register(self, name: str, factory: Callable[["App"], Scene]) -> None:
        self.factories[name] = factory

    def go(self, name: str, **kwargs) -> None:
        """Request a scene change; applied after the current frame."""
        self._pending = (name, kwargs)
        self._fading_out = True

    def _activate(self, name: str, kwargs: dict) -> None:
        if name not in self.scenes:
            self.scenes[name] = self.factories[name](self)
        if self.scene is not None:
            self.scene.on_exit()
        self.scene = self.scenes[name]
        self.scene_name = name
        self.scene.on_enter(**kwargs)

    # -- rendering helpers -------------------------------------------------
    def background(self) -> pygame.Surface:
        if self._background is None:
            surf = T.vertical_gradient((WINDOW_W, WINDOW_H), T.BG, T.BG_DEEP)
            # Faint grid, so large flat areas are not perfectly dead.
            for x in range(0, WINDOW_W, 40):
                pygame.draw.line(surf, (18, 22, 36), (x, 0), (x, WINDOW_H))
            for y in range(0, WINDOW_H, 40):
                pygame.draw.line(surf, (18, 22, 36), (0, y), (WINDOW_W, y))
            self._background = surf
        return self._background

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        while self.running:
            dt = min(0.05, self.clock.tick(FPS) / 1000.0)
            self.time += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif self.scene is not None:
                    self.scene.handle(event)

            if self.scene is not None:
                self.scene.update(dt)
            self.toasts.update(dt)

            # Fade out, swap, fade back in.
            if self._fading_out:
                self._fade = min(1.0, self._fade + dt * 6.0)
                if self._fade >= 1.0 and self._pending is not None:
                    name, kwargs = self._pending
                    self._pending = None
                    self._fading_out = False
                    self._activate(name, kwargs)
            else:
                self._fade = max(0.0, self._fade - dt * 5.0)

            self.screen.blit(self.background(), (0, 0))
            if self.scene is not None:
                self.scene.draw(self.screen)
            self.toasts.draw(self.screen, (28, WINDOW_H - 40))

            if self._fade > 0.001:
                veil = pygame.Surface((WINDOW_W, WINDOW_H))
                veil.fill(T.BG_DEEP)
                veil.set_alpha(int(255 * self._fade))
                self.screen.blit(veil, (0, 0))

            pygame.display.flip()

            self._frames += 1
            if self.max_frames is not None and self._frames >= self.max_frames:
                if self.shot_path:
                    pygame.image.save(self.screen, self.shot_path)
                self.running = False

        pygame.quit()

    def quit(self) -> None:
        self.running = False


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------
def draw_header(surface: pygame.Surface, title: str, subtitle: str = "") -> pygame.Rect:
    """Standard page header. Returns the rect below it for content."""
    T.draw_text(surface, title, (36, 28), 30, T.TEXT, bold=True)
    if subtitle:
        T.draw_text(surface, subtitle, (36, 66), 15, T.TEXT_DIM)
    pygame.draw.line(surface, T.PANEL_EDGE, (36, 96), (WINDOW_W - 36, 96))
    return pygame.Rect(36, 112, WINDOW_W - 72, WINDOW_H - 150)
