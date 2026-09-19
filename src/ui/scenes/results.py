"""Post-match summary, with a full stat sheet per player.

Each player's walked distance is scored against the true optimal tour - the
cheapest route that visits every key and then the exit. With at most five keys
the tour is solved exactly by trying every ordering, using real search costs
between waypoints, so the efficiency figure is a genuine optimum rather than a
greedy approximation.

Click either player card to inspect that player: the maze redraws showing the
ground they actually covered against the route they should have taken, and the
stat sheet below it breaks down where the time went.
"""

from __future__ import annotations

from itertools import permutations

import pygame

from ...ai import search
from ...config import WINDOW_H, WINDOW_W
from ...core.game import Game, PlayerKind
from .. import theme as T
from ..app import Scene, draw_header
from ..maze_view import MazeView
from ..widgets import Button

_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def ordinal(n: int | None) -> str:
    if n is None:
        return "-"
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{_SUFFIX.get(n % 10, 'th')}"


def optimal_tour(game: Game, start) -> tuple[float, list]:
    """Cheapest route visiting every key cell then the exit.

    Exhaustive over key orderings. That is factorial in the number of keys, but
    the game caps keys at five (120 orderings), so exact beats approximate here.
    """
    keys = list(game.keys)
    if not keys:
        result = search.solve(game.maze, start, game.goal, "astar")
        return (result.cost if result.found else float("inf")), list(result.path)

    # Pairwise costs between every waypoint, computed once.
    points = [start] + keys + [game.goal]
    cost: dict[tuple, float] = {}
    path: dict[tuple, list] = {}
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            result = search.solve(game.maze, a, b, "astar")
            cost[(a, b)] = result.cost if result.found else float("inf")
            path[(a, b)] = list(result.path)
            # The graph is directed (cost is charged for the cell entered), so
            # the reverse leg is priced with its own search.
            back = search.solve(game.maze, b, a, "astar")
            cost[(b, a)] = back.cost if back.found else float("inf")
            path[(b, a)] = list(back.path)

    best = float("inf")
    best_route: list = []
    for order in permutations(keys):
        legs = [start, *order, game.goal]
        total = sum(cost[(legs[i], legs[i + 1])] for i in range(len(legs) - 1))
        if total < best:
            best = total
            route: list = [start]
            for i in range(len(legs) - 1):
                route.extend(path[(legs[i], legs[i + 1])][1:])
            best_route = route
    return best, best_route


class ResultsScene(Scene):
    CARD = pygame.Rect(60, 150, 600, 132)
    CARD_GAP = 150
    PANEL = pygame.Rect(688, 120, 532, 540)

    def __init__(self, app) -> None:
        super().__init__(app)
        self.game: Game | None = None
        self.bots: list = []
        self.summary: list[dict] = []
        self.selected = 0
        self.view = MazeView(pygame.Rect(700, 172, 508, 184))

        self.widgets = [
            Button(pygame.Rect(60, WINDOW_H - 96, 200, 52), "Rematch",
                   lambda: app.go("game"), primary=True, icon="↻"),
            Button(pygame.Rect(276, WINDOW_H - 96, 170, 52), "Change setup",
                   lambda: app.go("setup")),
            Button(pygame.Rect(462, WINDOW_H - 96, 150, 52), "Menu",
                   lambda: app.go("menu")),
        ]

    # -- lifecycle ---------------------------------------------------------
    def on_enter(self, **kwargs) -> None:
        self.game = kwargs.get("game")
        self.bots = kwargs.get("bots", [])
        if self.game is None:
            return
        self.view.set_maze(self.game.maze)
        self.selected = self.game.winner.pid if self.game.winner else 0
        self.summary = []
        for player in self.game.players:
            best, route = optimal_tour(self.game, player.spawn)
            walked = player.distance_cost
            # Efficiency only means something for a completed tour. Scoring a
            # player who stopped two keys in against the cost of the *whole*
            # tour flatters them - they can show 97% while having walked barely
            # half the route.
            efficiency = (best / walked) if (player.finished and walked > 0) else None
            self.summary.append({
                "player": player,
                "optimal": best,
                "route": route,
                "walked": walked,
                "efficiency": efficiency,
            })

    def card_rect(self, index: int) -> pygame.Rect:
        return pygame.Rect(self.CARD.x, self.CARD.y + index * self.CARD_GAP,
                           self.CARD.w, self.CARD.h)

    # -- input -------------------------------------------------------------
    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i in range(len(self.summary)):
                if self.card_rect(i).collidepoint(event.pos):
                    self.selected = i
                    return
        elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_TAB, pygame.K_LEFT, pygame.K_RIGHT):
            if self.summary:
                self.selected = (self.selected + 1) % len(self.summary)
            return
        super().handle(event)

    @staticmethod
    def _set_cursor(shape) -> None:
        """Set the mouse cursor, tolerating backends that have none.

        The dummy SDL video driver used for headless screenshots raises here,
        and so do some remote-desktop setups. A cursor is a nicety; it must
        never be the reason a screen fails to draw.
        """
        try:
            pygame.mouse.set_cursor(shape)
        except (pygame.error, TypeError, AttributeError):
            pass

    def update(self, dt: float) -> None:
        super().update(dt)
        # Make the cards feel clickable rather than leaving it to be discovered.
        mouse = pygame.mouse.get_pos()
        over = any(self.card_rect(i).collidepoint(mouse)
                   for i in range(len(self.summary)))
        self._set_cursor(pygame.SYSTEM_CURSOR_HAND if over
                         else pygame.SYSTEM_CURSOR_ARROW)

    def on_exit(self) -> None:
        self._set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    # -- drawing -----------------------------------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        if self.game is None:
            return
        game = self.game
        winner = game.winner
        if winner is None:
            title, subtitle = "No winner", "nobody reached the exit in time"
        else:
            done = sum(1 for p in game.players if p.finished)
            title = f"{winner.name} wins"
            subtitle = (f"finished in {winner.finished_at:.1f}s  -  "
                        f"{done} of {len(game.players)} players completed the maze")
        draw_header(surface, title, subtitle)

        for i, row in enumerate(self.summary):
            self._draw_card(surface, i, row)

        hint_y = self.CARD.y + len(self.summary) * self.CARD_GAP + 2
        T.draw_text(surface, "click a player for their full stats",
                    (self.CARD.x + 4, hint_y), 12, T.TEXT_FAINT)

        self._draw_detail(surface)
        super().draw(surface)

    def _draw_card(self, surface, index: int, row: dict) -> None:
        rect = self.card_rect(index)
        player = row["player"]
        colour = T.PLAYER_COLORS[index]
        is_winner = self.game.winner is player
        active = index == self.selected

        T.panel(surface, rect,
                T.PANEL_HI if active else T.PANEL,
                colour if active else T.PANEL_EDGE,
                radius=14, width=2 if active else 1)
        if active:
            pygame.draw.rect(surface, colour,
                             pygame.Rect(rect.x, rect.y + 14, 4, rect.h - 28),
                             border_radius=2)

        pygame.draw.circle(surface, colour, (rect.x + 40, rect.y + 40), 14)
        T.draw_text(surface, player.name, (rect.x + 68, rect.y + 26), 20,
                    T.TEXT, bold=True)
        role = ("AI opponent" if player.kind is PlayerKind.BOT
                else ("WASD" if index == 0 else "Arrow keys"))
        T.draw_text(surface, role, (rect.x + 68, rect.y + 52), 13, T.TEXT_DIM)

        place = self.game.placement_of(player)
        if is_winner:
            T.draw_text(surface, "WINNER", (rect.right - 20, rect.y + 32), 13,
                        T.SUCCESS, bold=True, right=True)
        elif place:
            T.draw_text(surface, f"{ordinal(place)} place",
                        (rect.right - 20, rect.y + 32), 13, T.TEXT_DIM, right=True)
        else:
            T.draw_text(surface, "did not finish", (rect.right - 20, rect.y + 32),
                        13, T.TEXT_FAINT, right=True)

        stats = [
            ("keys", f"{player.keys}/{self.game.total_keys}"),
            ("steps", str(player.steps)),
            ("walked", f"{row['walked']:.0f}"),
            ("optimal", f"{row['optimal']:.0f}"),
        ]
        x = rect.x + 22
        for label, value in stats:
            T.draw_text(surface, value, (x, rect.y + 84), 19, T.TEXT, mono=True)
            T.draw_text(surface, label, (x, rect.y + 110), 11, T.TEXT_FAINT)
            x += 110

        eff = row["efficiency"]
        bar = pygame.Rect(rect.right - 168, rect.y + 92, 146, 10)
        if eff is None:
            progress = player.keys / max(1, self.game.total_keys)
            note = ("had every key, never reached the exit"
                    if player.keys >= self.game.total_keys
                    else f"{player.keys} of {self.game.total_keys} keys collected")
            T.draw_text(surface, note, (rect.right - 22, rect.y + 78), 12,
                        T.TEXT_FAINT, right=True)
            T.progress_bar(surface, bar, progress, T.PANEL_EDGE)
        else:
            T.draw_text(surface, f"route efficiency  {eff * 100:.0f}%",
                        (rect.right - 22, rect.y + 78), 12, T.TEXT_DIM, right=True)
            colour_eff = (T.SUCCESS if eff > 0.8
                          else (T.WARNING if eff > 0.55 else T.DANGER))
            T.progress_bar(surface, bar, eff, colour_eff)

    # -- detail panel ------------------------------------------------------
    def _draw_detail(self, surface: pygame.Surface) -> None:
        if not self.summary:
            return
        row = self.summary[self.selected]
        player = row["player"]
        colour = T.PLAYER_COLORS[self.selected]
        game = self.game

        panel = self.PANEL
        T.panel(surface, panel, T.PANEL, T.PANEL_EDGE, radius=16)
        pygame.draw.rect(surface, colour,
                         pygame.Rect(panel.x, panel.y, panel.w, 4),
                         border_top_left_radius=16, border_top_right_radius=16)

        T.draw_text(surface, player.name, (panel.x + 22, panel.y + 18), 19,
                    T.TEXT, bold=True)
        place = game.placement_of(player)
        headline = (f"{ordinal(place)} - {player.finished_at:.1f}s"
                    if player.finished else "did not finish")
        T.draw_text(surface, headline, (panel.right - 22, panel.y + 25), 14,
                    T.SUCCESS if player.finished else T.TEXT_FAINT, right=True)

        # Ground actually covered, against the route they should have walked.
        self.view.draw_maze(surface)
        self.view.draw_cells(surface, player.visited, colour, alpha=80)
        self.view.draw_path(surface, row["route"], T.PATH,
                            width=max(2, self.view.cell // 5), alpha=190)
        self.view.draw_objects(surface, game, 0.0)
        self.view.draw_border(surface)

        legend_y = panel.y + 240
        pygame.draw.rect(surface, colour,
                         pygame.Rect(panel.x + 22, legend_y, 12, 12),
                         border_radius=3)
        T.draw_text(surface, "where they went", (panel.x + 42, legend_y - 2), 12,
                    T.TEXT_DIM)
        pygame.draw.rect(surface, T.PATH,
                         pygame.Rect(panel.x + 200, legend_y, 12, 12),
                         border_radius=3)
        T.draw_text(surface, "optimal tour", (panel.x + 220, legend_y - 2), 12,
                    T.TEXT_DIM)

        self._draw_stat_grid(surface, row, pygame.Rect(
            panel.x + 22, panel.y + 266, panel.w - 44, panel.bottom - panel.y - 282))

    def _draw_stat_grid(self, surface, row: dict, area: pygame.Rect) -> None:
        player = row["player"]
        game = self.game
        eff = row["efficiency"]

        left: list[tuple[str, str, object]] = [
            ("keys collected", f"{player.keys} / {game.total_keys}", T.TEXT),
            ("steps taken", str(player.steps), T.TEXT),
            ("distance walked", f"{player.distance_cost:.0f}", T.TEXT),
            ("optimal tour", f"{row['optimal']:.0f}", T.TEXT_DIM),
            ("route efficiency",
             f"{eff * 100:.0f}%" if eff is not None else "-",
             T.SUCCESS if (eff or 0) > 0.8 else T.WARNING),
            ("cells explored", str(player.unique_cells), T.TEXT),
            ("repeat steps", str(player.backtracks), T.TEXT),
        ]
        right: list[tuple[str, str, object]] = [
            ("mud crossings", str(player.mud_steps), T.TEXT),
            ("wall bumps", str(player.wall_bumps), T.TEXT),
            ("power-ups taken", str(player.powerups_taken), T.TEXT),
            ("times frozen",
             f"{player.times_frozen}  ({player.frozen_seconds:.1f}s)", T.TEXT),
        ]
        if game.doors_enabled:
            right.append(("blocked by doors", str(player.doors_blocked),
                          T.DANGER if player.doors_blocked else T.TEXT))
            right.append(("doors locked", str(player.doors_locked), T.TEXT_DIM))
        if player.first_key_at is not None:
            right.append(("first key at", f"{player.first_key_at:.1f}s", T.TEXT_DIM))
        if player.keys >= game.total_keys and player.last_key_at is not None:
            right.append(("all keys by", f"{player.last_key_at:.1f}s", T.TEXT_DIM))

        bot = next((b for b in self.bots if b.player is player), None)
        if bot is not None:
            algo = bot.status()
            if "(" in algo:
                algo = algo.split("(", 1)[1].rstrip(")")
            right.append(("algorithm", algo, T.ACCENT_HI))
            right.append(("searches run", str(bot.replans), T.TEXT_DIM))
            if game.doors_enabled and (bot.door_waits or bot.door_detours):
                right.append(("door: waited / rerouted",
                              f"{bot.door_waits} / {bot.door_detours}", T.TEXT_DIM))
            right.append(("nodes expanded", str(bot.nodes_expanded_total),
                          T.TEXT_DIM))

        col_w = area.w // 2
        for col, rows in ((0, left), (1, right)):
            y = area.y
            for label, value, colour in rows:
                if y + 20 > area.bottom:
                    break
                x = area.x + col * col_w
                T.draw_text(surface, label, (x, y), 13, T.TEXT_DIM)
                T.draw_text(surface, value, (x + col_w - 28, y + 7), 13, colour,
                            mono=True, right=True)
                y += 21
