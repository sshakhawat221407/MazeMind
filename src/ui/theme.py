"""Visual language: palette, typography and small drawing primitives.

One dark palette, defined once. Hues are chosen so that the two players are
distinguishable at a glance (cyan versus amber - far apart in hue *and* in
lightness, so they stay separable for red-green colour blindness), and so the
search-visualisation colours read as a progression from cold "seen" to hot
"chosen".
"""

from __future__ import annotations

import os

import pygame

Color = tuple[int, int, int]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = (12, 15, 26)
BG_DEEP = (8, 10, 18)
PANEL = (20, 25, 41)
PANEL_HI = (29, 36, 58)
PANEL_EDGE = (45, 55, 84)

TEXT = (228, 233, 245)
TEXT_DIM = (146, 158, 187)
TEXT_FAINT = (96, 107, 134)

ACCENT = (124, 106, 255)      # violet - primary actions
ACCENT_HI = (154, 140, 255)
SUCCESS = (86, 222, 149)
WARNING = (247, 191, 74)
DANGER = (243, 98, 118)

# Locking gates. Open gates are a cool silver so they read as fixtures rather
# than hazards; a locked gate turns red, which is the only red on the board
# apart from urgent timers.
GATE = (196, 206, 226)
GATE_DIM = (112, 124, 152)
GATE_LOCKED = (226, 72, 96)
GATE_LOCKED_DIM = (104, 40, 56)

WALL = (78, 93, 133)
WALL_HI = (104, 122, 168)
FLOOR = (22, 28, 45)
FLOOR_ALT = (26, 33, 52)
MUD = (74, 58, 38)
MUD_EDGE = (104, 82, 52)

P1 = (56, 214, 224)           # cyan
P1_DIM = (28, 110, 118)
P2 = (255, 163, 72)           # amber
P2_DIM = (128, 82, 36)
PLAYER_COLORS = (P1, P2)
PLAYER_DIM = (P1_DIM, P2_DIM)

GOAL = (110, 231, 148)
GOAL_LOCKED = (86, 104, 120)
KEY = (250, 206, 84)
POWER_REVEAL = (128, 190, 255)
POWER_FREEZE = (168, 226, 255)

# Search visualisation: cold -> warm as a node goes from seen to chosen.
VISITED = (40, 54, 92)
VISITED_HI = (52, 71, 122)
FRONTIER = (72, 118, 208)
PATH = (176, 132, 255)

# Per-algorithm identity colour, reused in the charts so the legend and the
# maze animation always agree.
ALGO_COLORS = {
    "bfs": (86, 168, 255),
    "dfs": (243, 98, 118),
    "dijkstra": (247, 191, 74),
    "greedy": (86, 222, 149),
    "astar": (124, 106, 255),
    "astar_learned": (255, 128, 208),
}


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
_FONT_STACK = ["Segoe UI", "Inter", "Calibri", "DejaVu Sans", "Arial"]
_MONO_STACK = ["Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Courier New"]
_cache: dict[tuple[str, int, bool], pygame.font.Font] = {}
_faces: dict[str, tuple[str | None, str | None]] = {}

# pygame's fuzzy name matching sorts "segoeuil.ttf" (Light) ahead of
# "segoeui.ttf", so asking for "Segoe UI" hands back the Light weight - which
# is far too thin to read on a dark background. These suffixes let us find the
# regular and bold files sitting next to whatever match_font returned.
_LIGHT_SUFFIXES = ("sl.ttf", "l.ttf", "th.ttf", "li.ttf")


def _resolve(stack: list[str]) -> tuple[str | None, str | None]:
    """Locate (regular, bold) font files for the first name that resolves."""
    for name in stack:
        path = pygame.font.match_font(name)
        if not path:
            continue
        folder = os.path.dirname(path)
        base = os.path.basename(path).lower()
        regular = path
        for suffix in _LIGHT_SUFFIXES:
            if base.endswith(suffix):
                candidate = os.path.join(folder, base[: -len(suffix)] + ".ttf")
                if os.path.exists(candidate):
                    regular = candidate
                break
        stem = os.path.basename(regular)[:-4]
        bold_path = os.path.join(folder, stem + "b.ttf")
        return regular, (bold_path if os.path.exists(bold_path) else None)
    return None, None


def font(size: int, bold: bool = False, mono: bool = False) -> pygame.font.Font:
    """Cached font lookup. Falls back down the stack until something exists."""
    kind = "mono" if mono else "sans"
    key = (kind, size, bold)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    if kind not in _faces:
        _faces[kind] = _resolve(_MONO_STACK if mono else _FONT_STACK)
    regular, bold_path = _faces[kind]

    if regular is None:
        made = pygame.font.SysFont(
            ",".join(_MONO_STACK if mono else _FONT_STACK), size, bold=bold)
    elif bold and bold_path:
        made = pygame.font.Font(bold_path, size)
    else:
        made = pygame.font.Font(regular, size)
        if bold:
            made.set_bold(True)  # synthesised, when no bold file exists
    _cache[key] = made
    return made


_glyph_cache: dict[str, bool] = {}
_tofu: bytes | None = None
_tofu_size: tuple[int, int] = (0, 0)

# U+FFFF is a permanent noncharacter, so no font will ever have a glyph for it.
# Whatever it renders as *is* this font's "missing glyph" box.
_MISSING = "￿"


def supports(text: str) -> bool:
    """Whether the UI font can actually draw ``text``, or would show tofu.

    Decorative glyphs are a portability trap: an arrow on one machine is an
    empty box on another, depending on which fonts happen to be installed.

    ``Font.metrics`` cannot answer this - it happily reports plausible metrics
    for characters the face lacks, returning the identical tuple for every
    missing glyph because they all resolve to the same .notdef box. So the test
    is done by rendering: draw the candidate, draw a character guaranteed to be
    missing, and compare pixels. Identical output means tofu.
    """
    global _tofu, _tofu_size
    hit = _glyph_cache.get(text)
    if hit is not None:
        return hit

    face = font(16)
    if _tofu is None:
        box = face.render(_MISSING, True, (255, 255, 255))
        _tofu_size = box.get_size()
        _tofu = pygame.image.tostring(box, "RGBA")

    img = face.render(text, True, (255, 255, 255))
    ok = (img.get_size() != _tofu_size
          or pygame.image.tostring(img, "RGBA") != _tofu)
    _glyph_cache[text] = ok
    return ok


def text_surface(
    value: str, size: int = 16, color: Color = TEXT,
    bold: bool = False, mono: bool = False,
) -> pygame.Surface:
    return font(size, bold, mono).render(value, True, color)


def draw_text(
    surface: pygame.Surface,
    value: str,
    pos: tuple[int, int],
    size: int = 16,
    color: Color = TEXT,
    bold: bool = False,
    mono: bool = False,
    center: bool = False,
    right: bool = False,
) -> pygame.Rect:
    img = text_surface(value, size, color, bold, mono)
    rect = img.get_rect()
    if center:
        rect.center = pos
    elif right:
        rect.midright = pos
    else:
        rect.topleft = pos
    surface.blit(img, rect)
    return rect


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def lerp_color(a: Color, b: Color, t: float) -> Color:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def with_alpha(color: Color, alpha: int) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], max(0, min(255, alpha)))


def panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fill: Color = PANEL,
    border: Color | None = PANEL_EDGE,
    radius: int = 12,
    width: int = 1,
) -> None:
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def glow_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: Color,
    radius: int = 10,
    layers: int = 4,
    strength: int = 26,
) -> None:
    """Cheap outer glow: a few expanding translucent rounded rects."""
    for i in range(layers, 0, -1):
        pad = i * 3
        glow = pygame.Rect(rect.x - pad, rect.y - pad,
                           rect.w + pad * 2, rect.h + pad * 2)
        layer = pygame.Surface((glow.w, glow.h), pygame.SRCALPHA)
        pygame.draw.rect(
            layer, with_alpha(color, strength // i),
            layer.get_rect(), border_radius=radius + pad,
        )
        surface.blit(layer, glow.topleft)


def vertical_gradient(size: tuple[int, int], top: Color, bottom: Color) -> pygame.Surface:
    """A one-pixel-wide gradient stretched to size - cheap and smooth."""
    w, h = size
    strip = pygame.Surface((1, h))
    for y in range(h):
        strip.set_at((0, y), lerp_color(top, bottom, y / max(1, h - 1)))
    return pygame.transform.smoothscale(strip, (w, h))


def progress_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fraction: float,
    color: Color = ACCENT,
    track: Color = PANEL_HI,
) -> None:
    pygame.draw.rect(surface, track, rect, border_radius=rect.h // 2)
    filled = max(0.0, min(1.0, fraction))
    if filled > 0:
        inner = pygame.Rect(rect.x, rect.y, max(rect.h, int(rect.w * filled)), rect.h)
        pygame.draw.rect(surface, color, inner, border_radius=rect.h // 2)


def ease_out(t: float) -> float:
    """Cubic ease-out, for hover and transition animation."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3
