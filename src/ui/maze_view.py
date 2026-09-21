"""Renders a maze, its contents, and search state on top of it.

The static layer - floor, mud, walls - never changes during a match, so it is
drawn once into a cached surface and blitted per frame. Only the moving parts
(players, pickups, search overlays) are redrawn. Without that split, a 45x29
maze costs several thousand draw calls every frame and the frame rate collapses.

Player positions are interpolated. The simulation moves players a whole cell at
a time, but stepping the sprite discretely looks broken at 60 FPS, so the view
eases a display position toward the logical one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pygame

from ..config import DIRECTIONS, E, N, S, W
from ..core.game import Game, PowerUp
from ..core.maze import Cell, Maze
from . import theme as T


# ---------------------------------------------------------------------------
# Icon sprites
# ---------------------------------------------------------------------------
# Pickups used to be flat circles and diamonds, which read as "debug markers"
# rather than as objects. These draw recognisable shapes instead - a key with a
# bow and teeth, a panelled door, an eye, a snowflake.
#
# Two details make them survive being 14 pixels wide. Each sprite is drawn at
# four times its final size and then smooth-scaled down, which antialiases every
# curve for free; and the result is cached per (kind, size, colour), because
# rasterising them every frame for every pickup would cost more than the rest of
# the maze put together.
_SS = 4
_icon_cache: dict[tuple, pygame.Surface] = {}
_CLEAR = (0, 0, 0, 0)


def _shade(colour, factor: float):
    """Darken (factor < 1) or lighten (factor > 1) a colour."""
    return tuple(max(0, min(255, int(c * factor))) for c in colour[:3])


def _draw_key(surf: pygame.Surface, s: float, colour) -> None:
    dark = _shade(colour, 0.55)
    # Bow: the round grip, with a hole punched through it. Drawing with a fully
    # transparent colour *replaces* the pixels on an SRCALPHA surface rather
    # than blending, so this cuts a real hole.
    bow = (s * 0.30, s * 0.50)
    pygame.draw.circle(surf, colour, bow, s * 0.21)
    pygame.draw.circle(surf, dark, bow, s * 0.21, max(1, int(s * 0.035)))
    pygame.draw.circle(surf, _CLEAR, bow, s * 0.085)

    # Shaft.
    shaft = pygame.Rect(s * 0.45, s * 0.445, s * 0.44, s * 0.11)
    pygame.draw.rect(surf, colour, shaft, border_radius=int(s * 0.04))

    # Two teeth on the underside of the business end.
    for ox in (0.66, 0.78):
        pygame.draw.rect(surf, colour,
                         pygame.Rect(s * ox, s * 0.55, s * 0.075, s * 0.16))
    # A highlight along the top of the shaft gives it a little metal.
    pygame.draw.line(surf, _shade(colour, 1.35),
                     (s * 0.47, s * 0.47), (s * 0.86, s * 0.47),
                     max(1, int(s * 0.03)))


def _draw_door(surf: pygame.Surface, s: float, colour) -> None:
    dark = _shade(colour, 0.5)
    frame = pygame.Rect(s * 0.16, s * 0.14, s * 0.68, s * 0.78)
    pygame.draw.rect(surf, dark, frame, border_radius=int(s * 0.1))
    inner = frame.inflate(-s * 0.14, -s * 0.1)
    inner.bottom = frame.bottom
    pygame.draw.rect(surf, colour, inner,
                     border_top_left_radius=int(s * 0.18),
                     border_top_right_radius=int(s * 0.18))
    # Recessed panel and a knob, so it reads as a door and not a tombstone.
    panel = inner.inflate(-s * 0.16, -s * 0.22)
    panel.bottom = inner.bottom - s * 0.06
    pygame.draw.rect(surf, dark, panel,
                     max(1, int(s * 0.035)),
                     border_top_left_radius=int(s * 0.12),
                     border_top_right_radius=int(s * 0.12))
    pygame.draw.circle(surf, dark, (inner.right - s * 0.1, inner.centery),
                       s * 0.045)


def _draw_padlock(surf: pygame.Surface, s: float, colour) -> None:
    dark = _shade(colour, 0.45)
    width = max(2, int(s * 0.07))
    # Shackle: an arc for the top half, plus two straight legs down to the
    # body. Without the legs the arc floats above the lock and reads as a
    # broken ring rather than a shackle.
    pygame.draw.arc(surf, colour,
                    pygame.Rect(s * 0.33, s * 0.18, s * 0.34, s * 0.36),
                    0.0, math.pi, width)
    for lx in (s * 0.345, s * 0.655):
        pygame.draw.line(surf, colour, (lx, s * 0.36), (lx, s * 0.45), width)
    body = pygame.Rect(s * 0.26, s * 0.43, s * 0.48, s * 0.38)
    pygame.draw.rect(surf, colour, body, border_radius=int(s * 0.09))
    pygame.draw.circle(surf, dark, body.center, s * 0.065)


def _draw_eye(surf: pygame.Surface, s: float, colour) -> None:
    dark = _shade(colour, 0.4)
    # Almond outline built from two opposing quadratic-ish arcs.
    top = [(s * 0.10, s * 0.5)]
    bottom = []
    steps = 18
    for i in range(steps + 1):
        t = i / steps
        x = s * (0.10 + 0.80 * t)
        bulge = math.sin(math.pi * t) * s * 0.26
        top.append((x, s * 0.5 - bulge))
        bottom.append((x, s * 0.5 + bulge))
    pygame.draw.polygon(surf, colour, top + list(reversed(bottom)))
    pygame.draw.circle(surf, dark, (s * 0.5, s * 0.5), s * 0.17)
    pygame.draw.circle(surf, (14, 18, 30), (s * 0.5, s * 0.5), s * 0.085)
    pygame.draw.circle(surf, (255, 255, 255), (s * 0.56, s * 0.44), s * 0.045)


def _draw_snowflake(surf: pygame.Surface, s: float, colour) -> None:
    cx, cy = s * 0.5, s * 0.5
    arm = s * 0.36
    width = max(1, int(s * 0.06))
    for k in range(6):
        angle = math.pi * k / 3.0
        ex, ey = cx + arm * math.cos(angle), cy + arm * math.sin(angle)
        pygame.draw.line(surf, colour, (cx, cy), (ex, ey), width)
        # Two small branches per arm, at two-thirds out.
        bx, by = cx + arm * 0.62 * math.cos(angle), cy + arm * 0.62 * math.sin(angle)
        for side in (-1, 1):
            a2 = angle + side * math.pi / 4.0
            pygame.draw.line(surf, colour, (bx, by),
                             (bx + arm * 0.3 * math.cos(a2),
                              by + arm * 0.3 * math.sin(a2)), width)
    pygame.draw.circle(surf, colour, (cx, cy), s * 0.075)


_DRAWERS = {
    "key": _draw_key,
    "door": _draw_door,
    "padlock": _draw_padlock,
    "eye": _draw_eye,
    "snowflake": _draw_snowflake,
}


def icon(kind: str, size: int, colour) -> pygame.Surface:
    """Cached, antialiased sprite of ``kind`` at ``size`` pixels square."""
    size = max(6, int(size))
    key = (kind, size, tuple(colour[:3]))
    hit = _icon_cache.get(key)
    if hit is not None:
        return hit
    big = pygame.Surface((size * _SS, size * _SS), pygame.SRCALPHA)
    _DRAWERS[kind](big, size * _SS, colour)
    img = pygame.transform.smoothscale(big, (size, size))
    _icon_cache[key] = img
    return img


def blit_icon(surface: pygame.Surface, kind: str, centre, size: int,
              colour, alpha: int = 255) -> None:
    img = icon(kind, size, colour)
    if alpha < 255:
        img = img.copy()
        img.set_alpha(alpha)
    surface.blit(img, img.get_rect(center=(int(centre[0]), int(centre[1]))))


@dataclass
class MazeView:
    """Maps maze cells to screen pixels and draws everything inside them."""

    rect: pygame.Rect
    maze: Maze | None = None
    cell: int = 16
    origin: tuple[int, int] = (0, 0)
    _static: pygame.Surface | None = field(default=None, repr=False)
    _static_key: object = None
    _display: dict[int, tuple[float, float]] = field(default_factory=dict)

    # -- geometry ----------------------------------------------------------
    def set_maze(self, maze: Maze) -> None:
        self.maze = maze
        self.layout()
        self._static = None
        self._display.clear()

    def layout(self) -> None:
        if self.maze is None:
            return
        self.cell = max(6, min(self.rect.w // self.maze.width,
                               self.rect.h // self.maze.height))
        grid_w = self.cell * self.maze.width
        grid_h = self.cell * self.maze.height
        self.origin = (self.rect.x + (self.rect.w - grid_w) // 2,
                       self.rect.y + (self.rect.h - grid_h) // 2)

    def cell_rect(self, cell: Cell) -> pygame.Rect:
        return pygame.Rect(self.origin[0] + cell[0] * self.cell,
                           self.origin[1] + cell[1] * self.cell,
                           self.cell, self.cell)

    def centre(self, cell: Cell) -> tuple[int, int]:
        r = self.cell_rect(cell)
        return r.centerx, r.centery

    def point(self, fx: float, fy: float) -> tuple[int, int]:
        """Pixel centre for a fractional cell position (used for tweening)."""
        return (int(self.origin[0] + (fx + 0.5) * self.cell),
                int(self.origin[1] + (fy + 0.5) * self.cell))

    # -- static layer ------------------------------------------------------
    def _build_static(self) -> pygame.Surface:
        assert self.maze is not None
        maze = self.maze
        surf = pygame.Surface((self.cell * maze.width, self.cell * maze.height),
                              pygame.SRCALPHA)
        c = self.cell

        for y in range(maze.height):
            for x in range(maze.width):
                r = pygame.Rect(x * c, y * c, c, c)
                if maze.is_mud((x, y)):
                    pygame.draw.rect(surf, T.MUD, r)
                    # Stipple so mud reads as texture rather than a flat block.
                    for i in range(2):
                        px = r.x + 4 + (i * 7 + (x * 5 + y * 3) % 6) % max(1, c - 8)
                        py = r.y + 4 + (i * 5 + (x * 3 + y * 7) % 6) % max(1, c - 8)
                        pygame.draw.circle(surf, T.MUD_EDGE, (px, py), 1)
                else:
                    # Faint checker so long corridors stay readable.
                    tint = T.FLOOR if (x + y) % 2 == 0 else T.FLOOR_ALT
                    pygame.draw.rect(surf, tint, r)

        thickness = max(2, c // 7)
        for y in range(maze.height):
            for x in range(maze.width):
                px, py = x * c, y * c
                mask = maze.walls[y, x]
                if mask & N:
                    pygame.draw.line(surf, T.WALL, (px, py), (px + c, py), thickness)
                if mask & W:
                    pygame.draw.line(surf, T.WALL, (px, py), (px, py + c), thickness)
                # Only the last row/column need their far edges drawn, since
                # every other wall is already covered by a neighbour's N/W.
                if x == maze.width - 1 and mask & E:
                    pygame.draw.line(surf, T.WALL, (px + c, py), (px + c, py + c),
                                     thickness)
                if y == maze.height - 1 and mask & S:
                    pygame.draw.line(surf, T.WALL, (px, py + c), (px + c, py + c),
                                     thickness)
        return surf

    def static_layer(self) -> pygame.Surface:
        key = (id(self.maze), self.cell,
               None if self.maze is None else int(self.maze.walls.sum()))
        if self._static is None or self._static_key != key:
            self._static = self._build_static()
            self._static_key = key
        return self._static

    def draw_maze(self, surface: pygame.Surface) -> None:
        if self.maze is None:
            return
        surface.blit(self.static_layer(), self.origin)

    # -- overlays ----------------------------------------------------------
    def draw_cells(self, surface: pygame.Surface, cells, color, alpha: int = 90,
                   inset: int = 2) -> None:
        if not cells:
            return
        c = self.cell
        layer = pygame.Surface((self.cell * self.maze.width,
                                self.cell * self.maze.height), pygame.SRCALPHA)
        col = T.with_alpha(color, alpha)
        for cx, cy in cells:
            pygame.draw.rect(layer, col,
                             pygame.Rect(cx * c + inset, cy * c + inset,
                                         c - inset * 2, c - inset * 2))
        surface.blit(layer, self.origin)

    def draw_heatmap(self, surface: pygame.Surface, field_: np.ndarray,
                     alpha: int = 150) -> None:
        """Tint every cell by a scalar field (used for heuristic maps)."""
        if self.maze is None:
            return
        finite = field_[np.isfinite(field_)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        span = max(1e-6, hi - lo)
        c = self.cell
        layer = pygame.Surface((c * self.maze.width, c * self.maze.height),
                               pygame.SRCALPHA)
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                v = field_[y, x]
                if not np.isfinite(v):
                    continue
                t = (v - lo) / span
                colour = T.lerp_color(T.SUCCESS, T.DANGER, t)
                pygame.draw.rect(layer, T.with_alpha(colour, alpha),
                                 pygame.Rect(x * c, y * c, c, c))
        surface.blit(layer, self.origin)

    def draw_path(self, surface: pygame.Surface, path, color=T.PATH,
                  width: int | None = None, alpha: int = 235) -> None:
        if not path or len(path) < 2:
            return
        w = width or max(2, self.cell // 4)
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        points = [self.centre(cell) for cell in path]
        pygame.draw.lines(layer, T.with_alpha(color, alpha), False, points, w)
        surface.blit(layer, (0, 0))

    def draw_dashed_path(self, surface: pygame.Surface, path, color=T.PATH,
                         phase: float = 0.0) -> None:
        """Animated marching-ants path, used for the REVEAL power-up."""
        if not path or len(path) < 2:
            return
        w = max(2, self.cell // 5)
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for i in range(len(path) - 1):
            if (i + int(phase)) % 3 == 0:
                continue
            pygame.draw.line(layer, T.with_alpha(color, 220),
                             self.centre(path[i]), self.centre(path[i + 1]), w)
        surface.blit(layer, (0, 0))

    # -- game contents -----------------------------------------------------
    def draw_objects(self, surface: pygame.Surface, game: Game, t: float) -> None:
        c = self.cell
        pulse = 0.5 + 0.5 * math.sin(t * 3.0)
        bob = int(math.sin(t * 2.4) * max(1, c * 0.06))

        # Exit: a door, locked until a player holds every key.
        cx, cy = self.centre(game.goal)
        any_open = any(game.exit_open_for(p) for p in game.players)
        if any_open:
            T.glow_rect(surface, self.cell_rect(game.goal).inflate(-c // 4, -c // 4),
                        T.GOAL, radius=4, strength=int(30 + 40 * pulse))
        blit_icon(surface, "door", (cx, cy), int(c * 0.92),
                  T.GOAL if any_open else T.GOAL_LOCKED)
        if not any_open:
            # A padlock hung over the door says "locked" far faster than a
            # colour change does.
            blit_icon(surface, "padlock", (cx, cy + c * 0.06), int(c * 0.5),
                      T.WARNING)

        # Keys: dim once every player has taken this one.
        for cell in game.keys:
            taken_by_all = game.key_fully_taken(cell)
            kx, ky = self.centre(cell)
            if taken_by_all:
                blit_icon(surface, "key", (kx, ky), int(c * 0.8),
                          T.lerp_color(T.KEY, T.PANEL_HI, 0.8), alpha=90)
                continue
            glow = pygame.Surface((c * 2, c * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, T.with_alpha(T.KEY, int(30 + 34 * pulse)),
                               (c, c), int(c * 0.44))
            surface.blit(glow, (kx - c, ky - c))
            blit_icon(surface, "key", (kx, ky + bob), int(c * 0.86), T.KEY)

        # Power-ups: an eye for reveal, a snowflake for freeze.
        for cell, kind in game.powerups.items():
            px, py = self.centre(cell)
            reveal = kind is PowerUp.REVEAL
            colour = T.POWER_REVEAL if reveal else T.POWER_FREEZE
            glow = pygame.Surface((c * 2, c * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, T.with_alpha(colour, int(26 + 30 * pulse)),
                               (c, c), int(c * 0.42))
            surface.blit(glow, (px - c, py - c))
            blit_icon(surface, "eye" if reveal else "snowflake",
                      (px, py + bob), int(c * 0.8), colour)

    def gate_line(self, a: Cell, b: Cell) -> tuple[tuple[int, int], tuple[int, int]]:
        """Screen endpoints of the edge a gate sits on.

        Cells side by side share a vertical edge; cells stacked vertically
        share a horizontal one. The gate is drawn *on* that edge, exactly where
        a wall would be if the passage were closed.
        """
        c = self.cell
        ox, oy = self.origin
        if a[0] != b[0]:
            x = ox + max(a[0], b[0]) * c
            y = oy + a[1] * c
            return (x, y), (x, y + c)
        x = ox + a[0] * c
        y = oy + max(a[1], b[1]) * c
        return (x, y), (x + c, y)

    def draw_doors(self, surface: pygame.Surface, game: Game) -> None:
        """Every gate, open or locked.

        Gates are drawn even while open, because their whole point is that you
        can plan around them. An open gate is a pair of silver posts with a
        dashed threshold. A locked gate fills with a red bar that drains toward
        its first post as the lock runs down, with a padlock on top - so which
        gates are about to open can be read at a glance, not guessed.
        """
        if not game.doors_enabled or not game.doors:
            return
        c = self.cell
        post = max(3, int(c * 0.2))
        bar = max(3, int(c * 0.26))

        for a, b in game.doors:
            p1, p2 = self.gate_line(a, b)
            vertical = p1[0] == p2[0]
            locked = game.is_door_locked(a, b)

            if locked:
                # Full-length dark track, then the time still left in bright
                # red, anchored at the first post so it visibly counts down.
                remaining = 1.0 - game.lock_progress(a, b)
                pygame.draw.line(surface, T.GATE_LOCKED_DIM, p1, p2, bar)
                end = (p1[0], int(p1[1] + (p2[1] - p1[1]) * remaining)) if vertical \
                    else (int(p1[0] + (p2[0] - p1[0]) * remaining), p1[1])
                if end != p1:
                    pygame.draw.line(surface, T.GATE_LOCKED, p1, end, bar)
                post_colour = T.GATE_LOCKED
            else:
                # Dashed threshold between the posts: three short segments.
                for i in range(3):
                    t0 = 0.18 + i * 0.24
                    t1 = t0 + 0.14
                    s = (int(p1[0] + (p2[0] - p1[0]) * t0),
                         int(p1[1] + (p2[1] - p1[1]) * t0))
                    e = (int(p1[0] + (p2[0] - p1[0]) * t1),
                         int(p1[1] + (p2[1] - p1[1]) * t1))
                    pygame.draw.line(surface, T.GATE_DIM, s, e, max(2, bar // 2))
                post_colour = T.GATE

            for p in (p1, p2):
                pygame.draw.rect(surface, post_colour,
                                 pygame.Rect(p[0] - post // 2, p[1] - post // 2,
                                             post, post),
                                 border_radius=max(1, post // 3))

            if locked:
                # A dark disc behind the padlock, so it reads against the bar
                # instead of dissolving into it at small cell sizes.
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                pygame.draw.circle(surface, T.BG_DEEP, mid, max(4, int(c * 0.3)))
                pygame.draw.circle(surface, T.GATE_LOCKED, mid,
                                   max(4, int(c * 0.3)), max(1, c // 14))
                blit_icon(surface, "padlock", mid, int(c * 0.46), T.TEXT)

    def draw_players(self, surface: pygame.Surface, game: Game, dt: float,
                     t: float) -> None:
        for i, player in enumerate(game.players):
            target = (float(player.cell[0]), float(player.cell[1]))
            cur = self._display.get(player.pid, target)
            # Ease toward the logical cell so movement looks continuous.
            k = min(1.0, dt * 18.0)
            cur = (cur[0] + (target[0] - cur[0]) * k,
                   cur[1] + (target[1] - cur[1]) * k)
            self._display[player.pid] = cur

            colour = T.PLAYER_COLORS[i % len(T.PLAYER_COLORS)]
            px, py = self.point(*cur)
            radius = max(4, int(self.cell * 0.33))

            if player.is_frozen(game.elapsed):
                colour = T.lerp_color(colour, T.POWER_FREEZE, 0.6)
                ring = radius + 4 + int(2 * math.sin(t * 9))
                pygame.draw.circle(surface, T.POWER_FREEZE, (px, py), ring, 2)

            glow = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
            pygame.draw.circle(glow, T.with_alpha(colour, 55),
                               (radius * 3, radius * 3), radius * 2)
            surface.blit(glow, (px - radius * 3, py - radius * 3))

            pygame.draw.circle(surface, colour, (px, py), radius)
            pygame.draw.circle(surface, T.BG_DEEP, (px, py), radius, 2)
            if player.finished:
                pygame.draw.circle(surface, T.SUCCESS, (px, py), radius + 4, 2)

    def draw_spawn_markers(self, surface: pygame.Surface, game: Game) -> None:
        for i, player in enumerate(game.players):
            colour = T.PLAYER_DIM[i % len(T.PLAYER_DIM)]
            r = self.cell_rect(player.spawn).inflate(-self.cell // 3,
                                                     -self.cell // 3)
            pygame.draw.rect(surface, colour, r, 2, border_radius=3)

    def draw_border(self, surface: pygame.Surface) -> None:
        if self.maze is None:
            return
        frame = pygame.Rect(self.origin[0] - 2, self.origin[1] - 2,
                            self.cell * self.maze.width + 4,
                            self.cell * self.maze.height + 4)
        pygame.draw.rect(surface, T.PANEL_EDGE, frame, 2, border_radius=6)

    def cell_at(self, pos: tuple[int, int]) -> Cell | None:
        """Which cell a screen point falls in, or ``None`` if outside."""
        if self.maze is None:
            return None
        x = (pos[0] - self.origin[0]) // self.cell
        y = (pos[1] - self.origin[1]) // self.cell
        if 0 <= x < self.maze.width and 0 <= y < self.maze.height:
            return (int(x), int(y))
        return None
