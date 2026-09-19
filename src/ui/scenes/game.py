"""The race itself.

Input is read from the held-key state each frame rather than from key events.
A maze racer needs "keep walking while held", and event-driven input would
either fire once per press or repeat at the OS key-repeat rate, neither of which
matches a movement cooldown.

Player 1 uses WASD, player 2 the arrow keys, and both are polled independently
so simultaneous input from two people on one keyboard works.
"""

from __future__ import annotations

import random

import pygame

from ...ai.bot import Bot, plan_preview
from ...config import DIFFICULTY_SIZES, E, N, S, W, WINDOW_H, WINDOW_W
from ...core.game import MatchState, PlayerKind, PowerUp, new_game
from ...core.generator import generate
from .. import theme as T
from ..app import Scene
from ..maze_view import MazeView
from ..widgets import Button, key_hint

P1_KEYS = {pygame.K_w: N, pygame.K_d: E, pygame.K_s: S, pygame.K_a: W}
P2_KEYS = {pygame.K_UP: N, pygame.K_RIGHT: E, pygame.K_DOWN: S, pygame.K_LEFT: W}


class GameScene(Scene):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.view = MazeView(pygame.Rect(276, 104, 728, 620))
        self.game = None
        self.bots: list[Bot] = []
        self.paused = False
        self.reveal_phase = 0.0
        self.pause_buttons: list[Button] = [
            Button(pygame.Rect(WINDOW_W // 2 - 150, 360, 300, 50), "Resume",
                   self._resume, primary=True),
            Button(pygame.Rect(WINDOW_W // 2 - 150, 422, 300, 50), "Restart match",
                   self._restart),
            Button(pygame.Rect(WINDOW_W // 2 - 150, 484, 300, 50), "Back to menu",
                   lambda: app.go("menu")),
        ]

    # -- lifecycle ---------------------------------------------------------
    def on_enter(self, **kwargs) -> None:
        self._build()

    def _build(self) -> None:
        s = self.app.settings
        w, h = DIFFICULTY_SIZES[s["size"]]
        maze = generate(w, h, algorithm=s["generator"])

        vs_bot = s["opponent"] == "Bot"
        profile = str(s["difficulty"])
        specs = [
            ("Player 1", PlayerKind.HUMAN, None),
            (profile if vs_bot else "Player 2",
             PlayerKind.BOT if vs_bot else PlayerKind.HUMAN,
             profile if vs_bot else None),
        ]
        self.game = new_game(maze, specs, n_keys=int(s["keys"]),
                             rng=random.Random(),
                             doors_enabled=bool(s.get("doors", False)))
        self.view.set_maze(maze)
        self.bots = [
            Bot(p, p.bot_profile or "Tactician", random.Random())
            for p in self.game.players if p.kind is PlayerKind.BOT
        ]
        self.paused = False

    def _resume(self) -> None:
        self.paused = False

    def _restart(self) -> None:
        self._build()

    # -- input -------------------------------------------------------------
    def handle(self, event: pygame.event.Event) -> None:
        if self.paused:
            for button in self.pause_buttons:
                button.handle(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.paused = not self.paused
            elif event.key == pygame.K_r and self.paused:
                self._restart()

    def _intents(self) -> dict[int, int | None]:
        """Read held keys into one direction per human player."""
        pressed = pygame.key.get_pressed()
        out: dict[int, int | None] = {}
        for i, player in enumerate(self.game.players):
            if player.kind is not PlayerKind.HUMAN:
                continue
            mapping = P1_KEYS if i == 0 else P2_KEYS
            direction = None
            for key, value in mapping.items():
                if pressed[key]:
                    direction = value
                    break
            out[player.pid] = direction
        return out

    # -- frame -------------------------------------------------------------
    def update(self, dt: float) -> None:
        if self.paused:
            for button in self.pause_buttons:
                button.update(dt)
            return
        if self.game is None:
            return

        self.reveal_phase += dt * 6.0
        intents = self._intents()
        for bot in self.bots:
            intents[bot.player.pid] = bot.think(self.game, self.game.elapsed)
        self.game.update(dt, intents)

        if self.game.state is MatchState.FINISHED:
            self.app.go("results", game=self.game, bots=self.bots)

    def draw(self, surface: pygame.Surface) -> None:
        if self.game is None:
            return
        game = self.game

        self._draw_topbar(surface)
        self.view.draw_maze(surface)
        self.view.draw_spawn_markers(surface, game)

        # The bot's current plan, when the player asked to see it.
        if self.app.settings.get("show_bot_plan"):
            for bot in self.bots:
                if len(bot.path) > 1:
                    self.view.draw_path(surface, bot.path, T.P2_DIM,
                                        width=max(2, self.view.cell // 6),
                                        alpha=150)

        # REVEAL power-up: marching ants toward the holder's next objective.
        for player in game.players:
            if player.revealing(game.elapsed):
                route = plan_preview(game.maze, player.cell,
                                     game.next_objective(player))
                self.view.draw_dashed_path(surface, route, T.POWER_REVEAL,
                                           self.reveal_phase)

        self.view.draw_doors(surface, game)
        self.view.draw_objects(surface, game, self.app.time)
        self.view.draw_players(surface, game, 1 / 60.0, self.app.time)
        self.view.draw_border(surface)

        self._draw_hud(surface, 0, pygame.Rect(20, 104, 244, 620))
        self._draw_hud(surface, 1, pygame.Rect(1016, 104, 244, 620))
        self._draw_events(surface)

        if game.state is MatchState.COUNTDOWN:
            self._draw_countdown(surface)
        if self.paused:
            self._draw_pause(surface)

    # -- chrome ------------------------------------------------------------
    def _draw_topbar(self, surface: pygame.Surface) -> None:
        game = self.game
        bar = pygame.Rect(20, 20, WINDOW_W - 40, 68)
        T.panel(surface, bar, T.PANEL, T.PANEL_EDGE, radius=12)

        grace = game.grace_remaining()
        if grace is not None:
            # Someone is already home. What matters now is how long the rest
            # have left, so that replaces the match clock entirely.
            flash = grace < 10 and int(grace * 3) % 2 == 0
            colour = T.WARNING if flash else T.DANGER
            T.draw_text(surface, f"{grace:04.1f}", (bar.centerx, bar.centery),
                        32, colour, bold=True, center=True)
            label = f"{game.winner.name.upper()} HOME - RACE TO FINISH"
            T.draw_text(surface, label, (bar.centerx, bar.centery + 25), 10,
                        T.WARNING, bold=True, center=True)
        else:
            remaining = max(0.0, game.time_limit - game.elapsed)
            urgent = remaining < 30
            T.draw_text(surface, f"{int(remaining) // 60}:{int(remaining) % 60:02d}",
                        (bar.centerx, bar.centery), 30,
                        T.DANGER if urgent else T.TEXT, bold=True, center=True)
            T.draw_text(surface, "TIME LEFT", (bar.centerx, bar.centery + 24), 10,
                        T.TEXT_FAINT, bold=True, center=True)

        T.draw_text(surface, f"{game.maze.width}x{game.maze.height}",
                    (bar.x + 22, bar.y + 14), 14, T.TEXT_DIM)
        T.draw_text(surface, f"{game.total_keys} keys",
                    (bar.x + 22, bar.y + 36), 14, T.TEXT_DIM)
        if game.doors_enabled:
            locked = sum(1 for a, b in game.doors if game.is_door_locked(a, b))
            label = f"{len(game.doors)} LOCKING DOORS"
            if locked:
                label += f"  -  {locked} LOCKED"
            T.draw_text(surface, label, (bar.x + 140, bar.y + 36), 11,
                        T.GATE_LOCKED if locked else T.GATE, bold=True)
        T.draw_text(surface, "ESC  pause", (bar.right - 22, bar.centery), 13,
                    T.TEXT_FAINT, right=True)

    def _draw_hud(self, surface: pygame.Surface, index: int,
                  rect: pygame.Rect) -> None:
        game = self.game
        if index >= len(game.players):
            return
        player = game.players[index]
        colour = T.PLAYER_COLORS[index]

        T.panel(surface, rect, T.PANEL, T.PANEL_EDGE, radius=14)
        pygame.draw.rect(surface, colour,
                         pygame.Rect(rect.x, rect.y, rect.w, 4),
                         border_top_left_radius=14, border_top_right_radius=14)

        T.draw_text(surface, player.name, (rect.x + 18, rect.y + 22), 19,
                    T.TEXT, bold=True)
        controls = "WASD" if index == 0 else "Arrow keys"
        subtitle = controls if player.kind is PlayerKind.HUMAN else "AI opponent"
        T.draw_text(surface, subtitle, (rect.x + 18, rect.y + 48), 13, T.TEXT_DIM)

        # Key pips.
        T.draw_text(surface, "KEYS", (rect.x + 18, rect.y + 84), 11,
                    T.TEXT_FAINT, bold=True)
        for k in range(game.total_keys):
            cx = rect.x + 26 + k * 26
            cy = rect.y + 116
            if k < player.keys:
                pygame.draw.circle(surface, T.KEY, (cx, cy), 9)
                pygame.draw.circle(surface, T.BG_DEEP, (cx, cy), 4)
            else:
                pygame.draw.circle(surface, T.PANEL_HI, (cx, cy), 9)

        if player.finished:
            place = game.placement_of(player)
            badge = "FINISHED 1st" if place == 1 else f"FINISHED {place}"
            T.draw_text(surface, f"{badge}  {player.finished_at:.1f}s",
                        (rect.x + 18, rect.y + 142), 12, T.SUCCESS, bold=True)
        else:
            unlocked = game.exit_open_for(player)
            T.draw_text(surface, "EXIT UNLOCKED" if unlocked else "exit locked",
                        (rect.x + 18, rect.y + 142), 12,
                        T.SUCCESS if unlocked else T.TEXT_FAINT, bold=unlocked)

        rows = [
            ("steps", str(player.steps)),
            ("distance cost", f"{player.distance_cost:.0f}"),
        ]
        if game.doors_enabled:
            rows.append(("blocked by doors", str(player.doors_blocked)))
        if player.kind is PlayerKind.BOT:
            bot = next((b for b in self.bots if b.player is player), None)
            if bot:
                rows += [
                    ("replans", str(bot.replans)),
                    ("nodes expanded", str(bot.nodes_expanded_total)),
                ]
        y = rect.y + 186
        for label, value in rows:
            T.draw_text(surface, label, (rect.x + 18, y), 13, T.TEXT_DIM)
            T.draw_text(surface, value, (rect.right - 18, y + 7), 14, T.TEXT,
                        mono=True, right=True)
            y += 26

        # Status effects.
        y += 8
        if player.is_frozen(game.elapsed):
            left = player.frozen_until - game.elapsed
            T.draw_text(surface, f"FROZEN  {left:.1f}s", (rect.x + 18, y), 13,
                        T.POWER_FREEZE, bold=True)
            y += 22
        if player.revealing(game.elapsed):
            left = player.reveal_until - game.elapsed
            T.draw_text(surface, f"ROUTE SHOWN  {left:.1f}s", (rect.x + 18, y),
                        13, T.POWER_REVEAL, bold=True)
            y += 22

        if player.kind is PlayerKind.BOT:
            bot = next((b for b in self.bots if b.player is player), None)
            if bot:
                T.draw_text(surface, bot.status(), (rect.x + 18, rect.bottom - 38),
                            12, T.TEXT_FAINT)

    def _draw_events(self, surface: pygame.Surface) -> None:
        y = WINDOW_H - 44
        for _, message in list(self.game.events)[-2:]:
            T.draw_text(surface, message, (WINDOW_W // 2, y), 14, T.TEXT_DIM,
                        center=True)
            y += 20

    def _draw_countdown(self, surface: pygame.Surface) -> None:
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill(T.with_alpha(T.BG_DEEP, 170))
        surface.blit(veil, (0, 0))
        value = self.game.countdown
        label = str(int(value) + 1) if value > 0 else "GO"
        T.draw_text(surface, label, (WINDOW_W // 2, WINDOW_H // 2 - 30), 96,
                    T.ACCENT_HI, bold=True, center=True)
        T.draw_text(surface, "Collect every key, then reach the exit",
                    (WINDOW_W // 2, WINDOW_H // 2 + 50), 18, T.TEXT_DIM,
                    center=True)

        x = WINDOW_W // 2 - 210
        y = WINDOW_H // 2 + 96
        x += key_hint(surface, (x, y), "WASD", "Player 1")
        if len(self.game.players) > 1 and \
                self.game.players[1].kind is PlayerKind.HUMAN:
            key_hint(surface, (x, y), "Arrows", "Player 2")

    def _draw_pause(self, surface: pygame.Surface) -> None:
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill(T.with_alpha(T.BG_DEEP, 200))
        surface.blit(veil, (0, 0))
        T.draw_text(surface, "Paused", (WINDOW_W // 2, 290), 44, T.TEXT,
                    bold=True, center=True)
        for button in self.pause_buttons:
            button.draw(surface)
