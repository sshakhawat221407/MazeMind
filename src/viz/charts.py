"""Matplotlib figures rendered into pygame surfaces.

Matplotlib is used head-less (the ``Agg`` backend, no GUI toolkit) and its RGBA
buffer is handed straight to ``pygame.image.frombuffer``. That keeps a single
window and a single event loop while still getting real charts.

Figures are cached by a caller-supplied key, because rendering one costs tens of
milliseconds and would otherwise be repeated every frame at 60 FPS.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # must precede pyplot; no display is available

import matplotlib.pyplot as plt
import numpy as np
import pygame

from ..ui import theme as T


def _rgb(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


BG = _rgb(T.PANEL)
FG = _rgb(T.TEXT)
DIM = _rgb(T.TEXT_DIM)
GRID = _rgb(T.PANEL_EDGE)

_STYLE = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": DIM,
    "axes.titlecolor": FG,
    "text.color": FG,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "grid.color": GRID,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.45,
    "grid.linewidth": 0.6,
    "figure.autolayout": True,
    "legend.frameon": False,
    "legend.labelcolor": DIM,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_cache: dict[object, pygame.Surface] = {}


def figure_to_surface(fig) -> pygame.Surface:
    """Rasterise a figure and wrap the buffer as a pygame surface."""
    canvas = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    surface = pygame.image.frombuffer(bytes(buf), (w, h), "RGBA").convert_alpha()
    plt.close(fig)
    return surface


def _new_fig(size: tuple[int, int], dpi: int = 100):
    return plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)


def cached(key: object, builder, size: tuple[int, int]) -> pygame.Surface:
    """Return a cached chart, building it on first request."""
    full_key = (key, size)
    hit = _cache.get(full_key)
    if hit is not None:
        return hit
    with plt.rc_context(_STYLE):
        surface = builder(size)
    _cache[full_key] = surface
    return surface


def invalidate(prefix: str | None = None) -> None:
    """Drop cached charts, optionally only those whose key starts with prefix."""
    if prefix is None:
        _cache.clear()
        return
    for key in [k for k in _cache if isinstance(k[0], tuple)
                and k[0] and str(k[0][0]).startswith(prefix)]:
        _cache.pop(key, None)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def algorithm_comparison(rows, size: tuple[int, int]) -> pygame.Surface:
    """Grouped bars: nodes expanded and path cost relative to optimal."""
    fig = _new_fig(size)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    labels = [r.label for r in rows.values()]
    expanded = [r.mean_expanded for r in rows.values()]
    ratios = [r.mean_cost_ratio for r in rows.values()]
    colors = [_rgb(T.ALGO_COLORS.get(k, T.ACCENT)) for k in rows]
    y = np.arange(len(labels))

    ax1.barh(y, expanded, color=colors, height=0.62)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_title("Nodes expanded (lower is better)")
    ax1.grid(axis="y", alpha=0)

    ax2.barh(y, ratios, color=colors, height=0.62)
    ax2.axvline(1.0, color=_rgb(T.SUCCESS), lw=1.2, ls="--")
    ax2.set_yticks(y, [""] * len(labels))
    ax2.invert_yaxis()
    ax2.set_xlim(0.95, max(1.6, max(ratios) * 1.08))
    ax2.set_title("Path cost / optimal")
    ax2.grid(axis="y", alpha=0)
    return figure_to_surface(fig)


def training_curve(train: Sequence[float], val: Sequence[float],
                   mae: Sequence[float], size: tuple[int, int]) -> pygame.Surface:
    """Loss curves plus the validation error expressed in maze cells."""
    fig = _new_fig(size)
    ax = fig.add_subplot(111)
    epochs = np.arange(1, len(train) + 1)
    ax.plot(epochs, train, color=_rgb(T.ACCENT), lw=1.8, label="train (pinball)")
    ax.plot(epochs, val, color=_rgb(T.WARNING), lw=1.8, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("pinball loss")
    ax.legend(loc="upper right")

    if len(mae):
        twin = ax.twinx()
        twin.plot(epochs, mae, color=_rgb(T.SUCCESS), lw=1.2, alpha=0.75)
        twin.set_ylabel("val MAE (cells)", color=_rgb(T.SUCCESS))
        twin.tick_params(colors=_rgb(T.SUCCESS))
        twin.grid(False)
    ax.set_title("Learned heuristic - training")
    return figure_to_surface(fig)


def alpha_tradeoff(sweep: Sequence[dict], size: tuple[int, int]) -> pygame.Surface:
    """The speed/optimality curve that picks the shrink factor alpha."""
    fig = _new_fig(size)
    ax = fig.add_subplot(111)

    rows = [r for r in sweep if r.get("alpha") is not None]
    alphas = [r["alpha"] for r in rows]
    reduction = [r["reduction_pct"] for r in rows]
    optimal = [r["optimal_rate"] * 100 for r in rows]

    ax.plot(alphas, reduction, "o-", color=_rgb(T.ACCENT), lw=2,
            label="expansions saved %")
    ax.set_xlabel("alpha  (0 = Manhattan, 1 = full network)")
    ax.set_ylabel("expansions saved %", color=_rgb(T.ACCENT))
    ax.tick_params(axis="y", colors=_rgb(T.ACCENT))

    twin = ax.twinx()
    twin.plot(alphas, optimal, "s--", color=_rgb(T.WARNING), lw=1.6,
              label="optimal paths %")
    twin.set_ylabel("optimal paths %", color=_rgb(T.WARNING))
    twin.tick_params(colors=_rgb(T.WARNING))
    twin.set_ylim(0, 105)
    twin.grid(False)

    chosen = next((r for r in rows if r.get("chosen")), None)
    if chosen:
        ax.axvline(chosen["alpha"], color=_rgb(T.SUCCESS), lw=1.4, ls=":")
        ax.annotate(f"chosen a={chosen['alpha']:.2f}",
                    (chosen["alpha"], max(reduction) * 0.35),
                    color=_rgb(T.SUCCESS), fontsize=9,
                    ha="center", va="bottom")
    ax.set_title("Speed vs optimality trade-off")
    return figure_to_surface(fig)


def heuristic_showdown(data: dict, size: tuple[int, int]) -> pygame.Surface:
    """Per-maze scatter of Manhattan vs learned expansion counts."""
    fig = _new_fig(size)
    ax = fig.add_subplot(111)
    man = np.array(data["manhattan_expanded"], dtype=float)
    lrn = np.array(data["learned_expanded"], dtype=float)

    ax.scatter(man, lrn, s=26, color=_rgb(T.ACCENT), alpha=0.85,
               edgecolors="none")
    lo = 0
    hi = max(man.max(), lrn.max()) * 1.06
    ax.plot([lo, hi], [lo, hi], color=_rgb(T.TEXT_FAINT), lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("A* with Manhattan - nodes expanded")
    ax.set_ylabel("A* with learned h")
    ax.set_title(
        f"Below the line = learned wins   "
        f"({data['reduction_pct']:.1f}% fewer on average)"
    )
    return figure_to_surface(fig)


def expansion_histogram(rows, size: tuple[int, int]) -> pygame.Surface:
    """Distribution of expansions per algorithm, not just the mean."""
    fig = _new_fig(size)
    ax = fig.add_subplot(111)
    data = [r.expanded for r in rows.values() if r.expanded]
    labels = [r.label for r in rows.values() if r.expanded]
    colors = [_rgb(T.ALGO_COLORS.get(k, T.ACCENT))
              for k, r in rows.items() if r.expanded]

    # Tick labels are set separately rather than via boxplot's own keyword:
    # matplotlib renamed it from `labels` to `tick_labels` in 3.10, so passing
    # either one breaks on the other version.
    parts = ax.boxplot(data, patch_artist=True, widths=0.55,
                       medianprops={"color": FG, "linewidth": 1.4},
                       flierprops={"markersize": 3,
                                   "markerfacecolor": DIM,
                                   "markeredgecolor": "none"})
    ax.set_xticks(range(1, len(labels) + 1), labels)
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)
    for whisker in parts["whiskers"] + parts["caps"]:
        whisker.set_color(GRID)
    ax.set_ylabel("nodes expanded")
    ax.set_title("Spread across mazes")
    ax.tick_params(axis="x", rotation=18)
    return figure_to_surface(fig)


def feature_importance(names: Sequence[str], weights: np.ndarray,
                       size: tuple[int, int]) -> pygame.Surface:
    """First-layer weight magnitude per input feature.

    A crude but honest read on what the network leans on: the L2 norm of each
    input's outgoing weights in layer one. It shows which features carry signal,
    without pretending to be a causal attribution.
    """
    fig = _new_fig(size)
    ax = fig.add_subplot(111)
    strength = np.linalg.norm(weights, axis=1)
    order = np.argsort(strength)
    y = np.arange(len(order))
    ax.barh(y, strength[order], color=_rgb(T.ACCENT), height=0.7)
    ax.set_yticks(y, [names[i] for i in order], fontsize=7)
    ax.set_xlabel("|W| of input row (layer 1)")
    ax.set_title("What the network leans on")
    ax.grid(axis="y", alpha=0)
    return figure_to_surface(fig)
