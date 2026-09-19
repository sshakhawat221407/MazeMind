"""Interactive widgets drawn directly with pygame.

Pygame ships no widget toolkit, so buttons, toggles and sliders are built here
from primitives. Every widget animates its hover and press state through an
eased interpolation rather than snapping, because instant state changes read as
a rendering glitch rather than as feedback.

All widgets follow the same contract:

* ``handle(event)`` returns ``True`` when it consumed the event
* ``update(dt)`` advances animation
* ``draw(surface)`` renders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import pygame

from . import theme as T


class Widget:
    """Base widget: a rect, an enabled flag, and the three-method contract."""

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)
        self.enabled = True
        self.visible = True

    def handle(self, event: pygame.event.Event) -> bool:
        return False

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        pass


class Button(Widget):
    """Rounded button with hover glow and a press-depth animation."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        on_click: Callable[[], None] | None = None,
        *,
        primary: bool = False,
        icon: str = "",
        size: int = 17,
        tooltip: str = "",
    ) -> None:
        super().__init__(rect)
        self.label = label
        self.on_click = on_click
        self.primary = primary
        self.icon = icon
        self.size = size
        self.tooltip = tooltip
        self._hover = 0.0
        self._press = 0.0
        self._hovered = False
        self._pressed = False

    def handle(self, event: pygame.event.Event) -> bool:
        if not (self.enabled and self.visible):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was = self._pressed
            self._pressed = False
            if was and self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        return False

    def update(self, dt: float) -> None:
        # Approach the target at a rate independent of frame time.
        speed = min(1.0, dt * 14.0)
        self._hover += ((1.0 if self._hovered else 0.0) - self._hover) * speed
        self._press += ((1.0 if self._pressed else 0.0) - self._press) * speed

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        rect = self.rect.copy()
        rect.y += int(self._press * 2)

        if self.primary:
            base = T.lerp_color(T.ACCENT, T.ACCENT_HI, self._hover)
            fg = (255, 255, 255)
            edge = None
        else:
            base = T.lerp_color(T.PANEL, T.PANEL_HI, self._hover)
            fg = T.lerp_color(T.TEXT_DIM, T.TEXT, self._hover)
            edge = T.lerp_color(T.PANEL_EDGE, T.ACCENT, self._hover * 0.6)

        if not self.enabled:
            base = T.PANEL
            fg = T.TEXT_FAINT
            edge = T.PANEL_EDGE

        if self.primary and self._hover > 0.02 and self.enabled:
            T.glow_rect(surface, rect, T.ACCENT, radius=10,
                        strength=int(30 * self._hover))

        T.panel(surface, rect, base, edge, radius=10)
        # Drop the icon rather than render a tofu box on fonts that lack it.
        label = (f"{self.icon}  {self.label}"
                 if self.icon and T.supports(self.icon) else self.label)
        T.draw_text(surface, label, rect.center, self.size, fg,
                    bold=self.primary, center=True)


class SegmentedControl(Widget):
    """A row of mutually exclusive options with a sliding selection pill."""

    def __init__(
        self,
        rect: pygame.Rect,
        options: Sequence[str],
        index: int = 0,
        on_change: Callable[[int], None] | None = None,
        size: int = 15,
    ) -> None:
        super().__init__(rect)
        self.options = list(options)
        self.index = index
        self.on_change = on_change
        self.size = size
        self._pill = float(index)
        self._hover_index = -1

    @property
    def value(self) -> str:
        return self.options[self.index]

    def _slot(self, i: int) -> pygame.Rect:
        w = self.rect.w / len(self.options)
        return pygame.Rect(int(self.rect.x + w * i), self.rect.y,
                           int(w), self.rect.h)

    def handle(self, event: pygame.event.Event) -> bool:
        if not (self.enabled and self.visible):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover_index = -1
            for i in range(len(self.options)):
                if self._slot(i).collidepoint(event.pos):
                    self._hover_index = i
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i in range(len(self.options)):
                if self._slot(i).collidepoint(event.pos):
                    if i != self.index:
                        self.index = i
                        if self.on_change:
                            self.on_change(i)
                    return True
        return False

    def update(self, dt: float) -> None:
        self._pill += (self.index - self._pill) * min(1.0, dt * 16.0)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        T.panel(surface, self.rect, T.BG_DEEP, T.PANEL_EDGE, radius=10)
        w = self.rect.w / len(self.options)
        pill = pygame.Rect(int(self.rect.x + w * self._pill) + 3, self.rect.y + 3,
                           int(w) - 6, self.rect.h - 6)
        T.panel(surface, pill, T.ACCENT if self.enabled else T.PANEL_HI,
                None, radius=8)
        for i, option in enumerate(self.options):
            slot = self._slot(i)
            if i == self.index:
                color = (255, 255, 255)
            elif i == self._hover_index and self.enabled:
                color = T.TEXT
            else:
                color = T.TEXT_DIM if self.enabled else T.TEXT_FAINT
            T.draw_text(surface, option, slot.center, self.size, color,
                        bold=(i == self.index), center=True)


class Slider(Widget):
    """Horizontal slider with a draggable knob and a live value label."""

    def __init__(
        self,
        rect: pygame.Rect,
        minimum: float,
        maximum: float,
        value: float,
        on_change: Callable[[float], None] | None = None,
        *,
        step: float = 0.0,
        fmt: str = "{:.2f}",
        label: str = "",
    ) -> None:
        super().__init__(rect)
        self.minimum = minimum
        self.maximum = maximum
        self.value = value
        self.on_change = on_change
        self.step = step
        self.fmt = fmt
        self.label = label
        self._dragging = False
        self._hover = 0.0
        self._hovered = False

    @property
    def fraction(self) -> float:
        span = self.maximum - self.minimum
        return 0.0 if span <= 0 else (self.value - self.minimum) / span

    def _track(self) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.w, 6)

    def _set_from_x(self, x: int) -> None:
        frac = (x - self.rect.x) / max(1, self.rect.w)
        frac = max(0.0, min(1.0, frac))
        value = self.minimum + frac * (self.maximum - self.minimum)
        if self.step:
            value = round(value / self.step) * self.step
        value = max(self.minimum, min(self.maximum, value))
        if value != self.value:
            self.value = value
            if self.on_change:
                self.on_change(value)

    def handle(self, event: pygame.event.Event) -> bool:
        if not (self.enabled and self.visible):
            return False
        grab = self.rect.inflate(0, 18)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if grab.collidepoint(event.pos):
                self._dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION:
            self._hovered = grab.collidepoint(event.pos)
            if self._dragging:
                self._set_from_x(event.pos[0])
                return True
        return False

    def update(self, dt: float) -> None:
        target = 1.0 if (self._hovered or self._dragging) else 0.0
        self._hover += (target - self._hover) * min(1.0, dt * 14.0)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        if self.label:
            T.draw_text(surface, self.label, (self.rect.x, self.rect.y - 22),
                        14, T.TEXT_DIM)
            T.draw_text(surface, self.fmt.format(self.value),
                        (self.rect.right, self.rect.y - 22 + 7), 14, T.TEXT,
                        bold=True, right=True)
        track = self._track()
        pygame.draw.rect(surface, T.PANEL_HI, track, border_radius=3)
        filled = pygame.Rect(track.x, track.y, int(track.w * self.fraction), track.h)
        pygame.draw.rect(surface, T.ACCENT, filled, border_radius=3)
        cx = self.rect.x + int(self.rect.w * self.fraction)
        radius = 8 + int(self._hover * 2)
        pygame.draw.circle(surface, T.ACCENT_HI, (cx, self.rect.centery), radius)
        pygame.draw.circle(surface, (255, 255, 255), (cx, self.rect.centery),
                           radius - 4)


class Toggle(Widget):
    """A labelled on/off switch."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        value: bool = False,
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(rect)
        self.label = label
        self.value = value
        self.on_change = on_change
        self._knob = 1.0 if value else 0.0

    def handle(self, event: pygame.event.Event) -> bool:
        if not (self.enabled and self.visible):
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value
                if self.on_change:
                    self.on_change(self.value)
                return True
        return False

    def update(self, dt: float) -> None:
        self._knob += ((1.0 if self.value else 0.0) - self._knob) * min(1.0, dt * 16.0)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        switch = pygame.Rect(self.rect.x, self.rect.centery - 12, 46, 24)
        track = T.lerp_color(T.PANEL_HI, T.SUCCESS, self._knob)
        pygame.draw.rect(surface, track, switch, border_radius=12)
        cx = switch.x + 12 + int(self._knob * 22)
        pygame.draw.circle(surface, (255, 255, 255), (cx, switch.centery), 9)
        # Label sits vertically centred against the switch, not the row rect.
        img = T.text_surface(self.label, 15, T.TEXT if self.value else T.TEXT_DIM)
        surface.blit(img, img.get_rect(midleft=(switch.right + 12, switch.centery)))


@dataclass
class Toast:
    """Transient status message that fades out on its own."""

    message: str
    color: tuple[int, int, int] = T.TEXT
    life: float = 2.6
    age: float = 0.0

    @property
    def alive(self) -> bool:
        return self.age < self.life

    @property
    def alpha(self) -> int:
        remaining = self.life - self.age
        if remaining > 0.6:
            return 255
        return max(0, int(255 * remaining / 0.6))


class ToastStack:
    """Bottom-up stack of fading messages."""

    def __init__(self) -> None:
        self.items: list[Toast] = []

    def push(self, message: str, color: tuple[int, int, int] = T.TEXT) -> None:
        self.items.append(Toast(message, color))
        del self.items[:-4]

    def update(self, dt: float) -> None:
        for item in self.items:
            item.age += dt
        self.items = [i for i in self.items if i.alive]

    def draw(self, surface: pygame.Surface, anchor: tuple[int, int]) -> None:
        x, y = anchor
        for i, item in enumerate(reversed(self.items)):
            img = T.text_surface(item.message, 15, item.color)
            img.set_alpha(item.alpha)
            surface.blit(img, (x, y - i * 24))


class ScrollList(Widget):
    """Vertically scrollable text rows with a slim scrollbar."""

    def __init__(self, rect: pygame.Rect, rows: Sequence[str] | None = None,
                 size: int = 14, mono: bool = True) -> None:
        super().__init__(rect)
        self.rows = list(rows or [])
        self.size = size
        self.mono = mono
        self.offset = 0.0
        self.row_h = size + 8

    @property
    def max_offset(self) -> float:
        return max(0.0, len(self.rows) * self.row_h - self.rect.h + 8)

    def handle(self, event: pygame.event.Event) -> bool:
        if not (self.enabled and self.visible):
            return False
        if event.type == pygame.MOUSEWHEEL:
            mouse = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse):
                self.offset = max(0.0, min(self.max_offset,
                                           self.offset - event.y * 42))
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        prev = surface.get_clip()
        surface.set_clip(self.rect)
        y = self.rect.y + 4 - int(self.offset)
        for row in self.rows:
            if y + self.row_h >= self.rect.y and y <= self.rect.bottom:
                color = T.TEXT_DIM
                if row.startswith("#"):
                    color = T.TEXT
                    row = row[1:]
                T.draw_text(surface, row, (self.rect.x + 10, y), self.size,
                            color, mono=self.mono)
            y += self.row_h
        surface.set_clip(prev)

        if self.max_offset > 0:
            frac_visible = self.rect.h / (len(self.rows) * self.row_h + 8)
            bar_h = max(24, int(self.rect.h * frac_visible))
            travel = self.rect.h - bar_h
            pos = int(travel * (self.offset / self.max_offset))
            bar = pygame.Rect(self.rect.right - 6, self.rect.y + pos, 4, bar_h)
            pygame.draw.rect(surface, T.PANEL_EDGE, bar, border_radius=2)


def key_hint(surface: pygame.Surface, pos: tuple[int, int], keys: str,
             description: str) -> int:
    """Draw a keycap chip followed by its description. Returns width used."""
    img = T.text_surface(keys, 13, T.TEXT, bold=True, mono=True)
    cap = pygame.Rect(pos[0], pos[1], img.get_width() + 16, 24)
    T.panel(surface, cap, T.PANEL_HI, T.PANEL_EDGE, radius=6)
    surface.blit(img, img.get_rect(center=cap.center))
    label = T.text_surface(description, 14, T.TEXT_DIM)
    surface.blit(label, label.get_rect(midleft=(cap.right + 8, cap.centery)))
    return cap.w + 8 + label.get_width() + 18
