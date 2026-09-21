"""Training loop and admissibility audit for the learned heuristic.

Three things happen here, in order:

1. **Train** the network with mini-batch Adam on the pinball loss, which biases
   predictions below the true distance.
2. **Calibrate** an additive safety margin on a held-out slice, so the heuristic
   is empirically admissible.
3. **Audit** on a second, never-touched slice and report the violation rate.
   Fitting the margin and measuring the violation rate on the same data would
   be circular, so they use different splits.

A gradient check runs first: finite differences against the analytic gradients
from :mod:`src.ai.nn`. If the hand-derived chain rule were wrong, training would
still "work" in the sense of producing numbers, so this is worth proving.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np

from ..config import (
    HEURISTIC_WEIGHTS,
    MODELS_DIR,
    NN_ALPHA_GRID,
    NN_BATCH_SIZE,
    NN_CALIB_MAZES,
    NN_CALIB_SIZE,
    NN_EPOCHS,
    NN_HIDDEN,
    NN_LEARNING_RATE,
    NN_MAX_COST_RATIO,
    NN_QUANTILE,
    NN_SEED,
    N_FEATURES,
    SAMPLES_PER_MAZE,
    TRAIN_MAZES,
    TRAINING_HISTORY,
)
from .dataset import build_dataset, three_way_split
from .nn import MLP, Adam, gradient_check, pinball_loss

ProgressFn = Callable[[float, str], None]
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")


@dataclass
class TrainingReport:
    """Everything worth writing down about a training run."""

    epochs: int = 0
    n_train: int = 0
    n_calib: int = 0
    n_test: int = 0
    quantile: float = NN_QUANTILE
    hidden: tuple[int, ...] = NN_HIDDEN
    gradient_check_error: float = 0.0

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mae_cells: list[float] = field(default_factory=list)

    # Before the admissibility shrink is applied.
    raw_violation_rate: float = 0.0
    raw_mae_cells: float = 0.0
    # After it.
    alpha: float = 1.0
    safe_violation_rate: float = 0.0
    safe_mae_cells: float = 0.0
    # Mean predicted detour ratio == how many times stronger than Manhattan the
    # heuristic is, since h = manhattan * ratio. 1.0 means "no better than
    # Manhattan"; the true mean ratio is the ceiling worth aiming at.
    mean_learned_over_manhattan: float = 0.0
    mean_true_ratio: float = 0.0
    # One row per candidate alpha from the live-search calibration sweep.
    alpha_sweep: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


def _batches(n: int, size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def calibrate_alpha(
    model: MLP,
    n_mazes: int = NN_CALIB_MAZES,
    size: tuple[int, int] = NN_CALIB_SIZE,
    grid: Sequence[float] = NN_ALPHA_GRID,
    max_cost_ratio: float = NN_MAX_COST_RATIO,
    seed: int = 4242,
    progress: ProgressFn | None = None,
) -> tuple[float, list[dict]]:
    """Pick the shrink factor by running actual A* searches.

    For each candidate alpha, A* is run over a fixed set of fresh mazes (a seed
    unrelated to the training seed, so these layouts were never learned from)
    and three numbers are recorded: nodes expanded, how often the returned path
    was exactly optimal, and the mean cost relative to optimal.

    The chosen alpha is the most aggressive one whose mean path cost stays
    within ``max_cost_ratio`` of optimal. Selecting on measured search
    behaviour rather than on a proxy statistic is the point: a heuristic that
    looks slightly inadmissible on paper can still return optimal paths, and a
    heuristic that looks safe can still be useless.
    """
    # Local import: benchmark -> heuristics -> nn would otherwise be circular
    # at module load time.
    from ..core.generator import CARVERS, generate
    from . import search as search_mod
    from .heuristics import LearnedHeuristic, manhattan

    rng = random.Random(seed)
    problems = []
    for _ in range(n_mazes):
        maze = generate(
            *size,
            algorithm=rng.choice(list(CARVERS)),
            seed=rng.randrange(1 << 30),
        )
        start, goal = (0, 0), (size[0] - 1, size[1] - 1)
        best = search_mod.optimal_cost(maze, start, goal)
        if np.isfinite(best) and best > 0:
            problems.append((maze, start, goal, best))

    if not problems:
        return 1.0, []

    baseline = [
        search_mod.solve(m, s, g, "astar", heuristic=manhattan).nodes_expanded
        for m, s, g, _ in problems
    ]
    base_mean = float(np.mean(baseline))

    original_alpha = model.alpha
    sweep: list[dict] = []
    for i, alpha in enumerate(grid):
        model.alpha = alpha
        h = LearnedHeuristic(model, safety=True)
        expanded, optimal, ratios = [], 0, []
        for maze, start, goal, best in problems:
            res = search_mod.solve(maze, start, goal, "astar", heuristic=h)
            expanded.append(res.nodes_expanded)
            if res.found:
                ratios.append(res.cost / best)
                optimal += int(abs(res.cost - best) < 1e-6)
        mean_exp = float(np.mean(expanded))
        sweep.append({
            "alpha": float(alpha),
            "expanded": mean_exp,
            "reduction_pct": 100.0 * (1.0 - mean_exp / base_mean),
            "optimal_rate": optimal / len(problems),
            "cost_ratio": float(np.mean(ratios)) if ratios else float("inf"),
        })
        if progress:
            progress(0.94 + 0.04 * (i + 1) / len(grid),
                     f"alpha={alpha:.2f}  expanded {mean_exp:.0f}")
    model.alpha = original_alpha

    acceptable = [r for r in sweep if r["cost_ratio"] <= max_cost_ratio]
    # Most aggressive alpha that still respects the optimality budget; if none
    # qualifies, fall back to 0 (pure Manhattan, guaranteed optimal).
    best_row = max(acceptable, key=lambda r: r["alpha"]) if acceptable else None
    chosen = float(best_row["alpha"]) if best_row else 0.0

    for row in sweep:
        row["chosen"] = row["alpha"] == chosen
    sweep.append({"alpha": None, "expanded": base_mean, "reduction_pct": 0.0,
                  "optimal_rate": 1.0, "cost_ratio": 1.0, "chosen": False,
                  "label": "Manhattan baseline"})
    return chosen, sweep


def train_heuristic(
    n_mazes: int = TRAIN_MAZES,
    samples_per_maze: int = SAMPLES_PER_MAZE,
    epochs: int = NN_EPOCHS,
    lr: float = NN_LEARNING_RATE,
    batch_size: int = NN_BATCH_SIZE,
    tau: float = NN_QUANTILE,
    seed: int = NN_SEED,
    progress: ProgressFn | None = None,
    save: bool = True,
) -> tuple[MLP, TrainingReport]:
    """Train, calibrate and audit the heuristic network."""

    def say(frac: float, msg: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    # -- data --------------------------------------------------------------
    say(0.0, "Generating training mazes...")
    X, y, m = build_dataset(
        n_mazes,
        samples_per_maze,
        seed=seed,
        progress=lambda f, msg: say(f * 0.30, msg),
    )
    splits = three_way_split(X, y, m, seed=seed)
    Xtr, ytr, _ = splits["train"]
    Xca, yca, mca = splits["calib"]
    Xte, yte, mte = splits["test"]

    report = TrainingReport(
        epochs=epochs,
        n_train=Xtr.shape[0],
        n_calib=Xca.shape[0],
        n_test=Xte.shape[0],
        quantile=tau,
        hidden=tuple(NN_HIDDEN),
    )

    # -- model -------------------------------------------------------------
    model = MLP(N_FEATURES, NN_HIDDEN, seed=seed)
    model.fit_normaliser(Xtr)

    # -- prove the backprop is right before trusting it --------------------
    say(0.31, "Verifying analytic gradients...")
    probe = MLP(N_FEATURES, (8, 8), seed=seed)
    probe.fit_normaliser(Xtr)
    report.gradient_check_error = gradient_check(probe, Xtr[:24], ytr[:24])

    # -- training loop -----------------------------------------------------
    opt = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for idx in _batches(n, batch_size, rng):
            xb, yb = Xtr[idx], ytr[idx]
            pred = model.forward(xb)
            loss, dpred = pinball_loss(pred, yb, tau)
            model.backward(dpred)
            opt.step(model.gradients())
            epoch_loss += loss
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        val_pred = model.forward(Xca)
        val_loss = pinball_loss(val_pred, yca, tau)[0]
        # A ratio error times the Manhattan distance is an error in real cells.
        val_mae_cells = float(np.mean(np.abs(val_pred - yca) * mca))

        report.train_loss.append(train_loss)
        report.val_loss.append(val_loss)
        report.val_mae_cells.append(val_mae_cells)

        say(
            0.33 + 0.60 * (epoch + 1) / epochs,
            f"Epoch {epoch + 1}/{epochs}   train {train_loss:.4f}   "
            f"val {val_loss:.4f}   MAE {val_mae_cells:.2f} cells",
        )

    # -- calibrate the admissibility shrink on real searches ---------------
    say(0.94, "Calibrating alpha on live A* searches...")
    alpha, sweep = calibrate_alpha(model, progress=say)
    model.alpha = alpha
    report.alpha = alpha
    report.alpha_sweep = sweep

    # -- audit on the untouched test slice ---------------------------------
    say(0.985, "Auditing on held-out mazes...")
    raw = model.forward(Xte)
    report.raw_violation_rate = float(np.mean(raw > yte))
    report.raw_mae_cells = float(np.mean(np.abs(raw - yte) * mte))

    safe = 1.0 + (raw - 1.0) * alpha
    report.safe_violation_rate = float(np.mean(safe > yte + 1e-12))
    report.safe_mae_cells = float(np.mean(np.abs(safe - yte) * mte))

    # h = manhattan * ratio, so the mean predicted ratio *is* the factor by
    # which the learned heuristic beats Manhattan. The true mean ratio is the
    # ceiling: a perfect predictor would reach it exactly.
    report.mean_learned_over_manhattan = float(np.mean(safe))
    report.mean_true_ratio = float(np.mean(yte))

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model.save(HEURISTIC_WEIGHTS)
        np.savez_compressed(
            TRAINING_HISTORY,
            train_loss=np.array(report.train_loss),
            val_loss=np.array(report.val_loss),
            val_mae_cells=np.array(report.val_mae_cells),
        )
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(report.as_dict(), fh, indent=2)

    say(1.0, "Training complete.")
    return model, report


def load_report() -> TrainingReport | None:
    """Read the last training report, if training has ever been run."""
    if not os.path.exists(REPORT_PATH):
        return None
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["hidden"] = tuple(data.get("hidden", NN_HIDDEN))
        return TrainingReport(**data)
    except Exception:
        return None


def format_report(report: TrainingReport) -> str:
    """Human-readable summary, printed by ``main.py train``."""
    lines = [
        "",
        "=" * 62,
        "  LEARNED HEURISTIC - TRAINING REPORT",
        "=" * 62,
        f"  architecture           {N_FEATURES} -> "
        f"{' -> '.join(map(str, report.hidden))} -> 1 (softplus, +1 offset)",
        f"  samples                {report.n_train} train / {report.n_calib} calib / {report.n_test} test",
        f"  pinball quantile       {report.quantile}",
        f"  gradient check         {report.gradient_check_error:.2e}  "
        f"({'PASS' if report.gradient_check_error < 1e-4 else 'FAIL'})",
        "",
        f"  final train loss       {report.train_loss[-1]:.5f}" if report.train_loss else "",
        f"  final val loss         {report.val_loss[-1]:.5f}" if report.val_loss else "",
        f"  val MAE                {report.val_mae_cells[-1]:.2f} cells" if report.val_mae_cells else "",
        "",
        "  ADMISSIBILITY AUDIT (held-out test mazes)",
        f"    raw  violation rate  {report.raw_violation_rate * 100:5.2f}%   "
        f"MAE {report.raw_mae_cells:.2f} cells",
        f"    shrink alpha         {report.alpha:.2f}  (chosen on live searches)",
        f"    safe violation rate  {report.safe_violation_rate * 100:5.2f}%   "
        f"MAE {report.safe_mae_cells:.2f} cells",
        "",
        f"  h_learned / h_manhattan  {report.mean_learned_over_manhattan:.3f}x "
        f"(true detour ratio averages {report.mean_true_ratio:.3f}x)",
    ]

    if report.alpha_sweep:
        lines += [
            "",
            "  ALPHA SWEEP - A* on held-out mazes",
            f"    {'alpha':>7}{'expanded':>11}{'vs manh':>10}{'optimal':>10}{'cost/opt':>10}",
            "    " + "-" * 48,
        ]
        for row in report.alpha_sweep:
            tag = "  <-- chosen" if row.get("chosen") else ""
            label = "manh" if row["alpha"] is None else f"{row['alpha']:.2f}"
            red = "--" if row["alpha"] is None else f"{row['reduction_pct']:.1f}%"
            lines.append(
                f"    {label:>7}{row['expanded']:>11.1f}{red:>10}"
                f"{row['optimal_rate'] * 100:>9.0f}%{row['cost_ratio']:>10.3f}{tag}"
            )

    lines += ["=" * 62, ""]
    return "\n".join(l for l in lines if l != "")
