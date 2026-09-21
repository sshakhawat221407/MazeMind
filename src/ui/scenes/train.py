"""Train the learned heuristic from inside the game.

Training runs on a worker thread so the window keeps repainting and the progress
bar actually moves. The thread only trains; charts are built on the main thread
afterwards, since matplotlib is not thread-safe.
"""

from __future__ import annotations

import threading

import pygame

from ...ai.train import load_report, train_heuristic
from ...config import (
    NN_EPOCHS,
    NN_QUANTILE,
    N_FEATURES,
    TRAIN_MAZES,
    WINDOW_H,
    WINDOW_W,
)
from ...viz import charts
from .. import theme as T
from ..app import Scene, draw_header
from ..widgets import Button, Slider

STEPS = [
    "Generate mazes and label every cell with one Dijkstra sweep",
    "Verify the hand-written gradients against finite differences",
    "Train the network with Adam on the pinball (quantile) loss",
    "Calibrate the shrink factor on live A* searches",
    "Audit admissibility on mazes never seen during training",
]


class TrainScene(Scene):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.thread: threading.Thread | None = None
        self.progress = 0.0
        self.message = ""
        self.report = load_report()
        self.error: str | None = None
        self.token = 0

        # The pipeline list above ends around y=252, and each slider draws its
        # own caption 22px above its track, so these start well clear of it.
        self.mazes = Slider(pygame.Rect(60, 302, 380, 20), 60, 600,
                            float(TRAIN_MAZES), step=20, fmt="{:.0f}",
                            label="Training mazes")
        self.epochs = Slider(pygame.Rect(60, 372, 380, 20), 10, 200,
                             float(NN_EPOCHS), step=10, fmt="{:.0f}",
                             label="Epochs")
        self.btn_train = Button(pygame.Rect(60, 412, 220, 50), "Train network",
                                self.start, primary=True, icon="⚙")
        self.btn_back = Button(pygame.Rect(296, 412, 130, 50), "Back",
                               lambda: app.go("menu"))
        self.widgets = [self.mazes, self.epochs, self.btn_train, self.btn_back]

    def on_enter(self, **kwargs) -> None:
        self.report = load_report()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        if self.busy:
            return
        self.progress = 0.0
        self.message = "starting..."
        self.error = None

        def work() -> None:
            def progress(frac: float, message: str) -> None:
                self.progress = frac
                self.message = message

            try:
                _, report = train_heuristic(
                    n_mazes=int(self.mazes.value),
                    epochs=int(self.epochs.value),
                    progress=progress,
                )
                self.report = report
                self.token += 1
                # Charts and the cached heuristic both need to be re-read.
                charts.invalidate()
            except Exception as exc:  # surface it rather than dying silently
                self.error = f"{type(exc).__name__}: {exc}"

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def update(self, dt: float) -> None:
        super().update(dt)
        self.btn_train.enabled = not self.busy
        self.mazes.enabled = not self.busy
        self.epochs.enabled = not self.busy

    def draw(self, surface: pygame.Surface) -> None:
        draw_header(
            surface, "Train the heuristic",
            "A neural network written from scratch learns how much a maze "
            "makes you detour.",
        )

        T.draw_text(surface, "PIPELINE", (60, 128), 12, T.TEXT_FAINT, bold=True)
        y = 152
        for i, step in enumerate(STEPS):
            done = self.progress > (i + 1) / len(STEPS) * 0.95
            colour = T.SUCCESS if (done and self.report) else T.TEXT_DIM
            T.draw_text(surface, f"{i + 1}.", (60, y), 13, T.TEXT_FAINT)
            T.draw_text(surface, step, (82, y), 13, colour)
            y += 20

        super().draw(surface)

        if self.busy or self.progress > 0:
            bar = pygame.Rect(60, 492, 380, 14)
            T.progress_bar(surface, bar, self.progress)
            T.draw_text(surface, self.message, (60, 516), 13, T.TEXT_DIM,
                        mono=True)

        if self.error:
            T.draw_text(surface, self.error, (60, 548), 14, T.DANGER)

        T.draw_text(surface, f"architecture   {N_FEATURES} inputs -> 96 -> 64 -> 1",
                    (60, WINDOW_H - 96), 13, T.TEXT_FAINT, mono=True)
        T.draw_text(surface, f"loss           pinball at quantile {NN_QUANTILE}",
                    (60, WINDOW_H - 74), 13, T.TEXT_FAINT, mono=True)
        T.draw_text(surface, "optimiser      Adam (hand-rolled, bias-corrected)",
                    (60, WINDOW_H - 52), 13, T.TEXT_FAINT, mono=True)

        self._draw_results(surface)

    def _draw_results(self, surface: pygame.Surface) -> None:
        report = self.report
        panel = pygame.Rect(478, 116, 742, 604)
        T.panel(surface, panel, T.PANEL, T.PANEL_EDGE, radius=16)

        if report is None or not report.train_loss:
            T.draw_text(surface, "No trained model yet",
                        (panel.centerx, panel.centery - 20), 20, T.TEXT_DIM,
                        center=True)
            T.draw_text(surface,
                        "Press 'Train network'. It takes roughly a minute.",
                        (panel.centerx, panel.centery + 12), 14, T.TEXT_FAINT,
                        center=True)
            return

        curve = charts.cached(
            ("curve", self.token, len(report.train_loss)),
            lambda size: charts.training_curve(
                report.train_loss, report.val_loss, report.val_mae_cells, size),
            (700, 250),
        )
        surface.blit(curve, (panel.x + 20, panel.y + 20))

        if report.alpha_sweep:
            sweep = charts.cached(
                ("sweep", self.token, report.alpha),
                lambda size: charts.alpha_tradeoff(report.alpha_sweep, size),
                (700, 230),
            )
            surface.blit(sweep, (panel.x + 20, panel.y + 282))

        chosen = next((r for r in report.alpha_sweep if r.get("chosen")), None)
        y = panel.bottom - 76
        cells = [
            ("gradient check", f"{report.gradient_check_error:.1e}",
             T.SUCCESS if report.gradient_check_error < 1e-4 else T.DANGER),
            ("val MAE", f"{report.val_mae_cells[-1]:.1f} cells", T.TEXT),
            ("alpha", f"{report.alpha:.2f}", T.TEXT),
            ("expansions saved",
             f"{chosen['reduction_pct']:.1f}%" if chosen else "-", T.ACCENT_HI),
            ("optimal paths",
             f"{chosen['optimal_rate'] * 100:.0f}%" if chosen else "-", T.TEXT),
        ]
        x = panel.x + 24
        for label, value, colour in cells:
            T.draw_text(surface, value, (x, y), 18, colour, mono=True, bold=True)
            T.draw_text(surface, label, (x, y + 26), 11, T.TEXT_FAINT)
            x += 146
