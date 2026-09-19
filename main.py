"""MazeMind - entry point.

    python main.py                 launch the game
    python main.py train           train the learned heuristic and print a report
    python main.py bench           run the algorithm benchmark in the terminal
    python main.py --scene lab     launch straight into a screen
    python main.py --shot out.png  render N frames headlessly and save a PNG
"""

from __future__ import annotations

import argparse
import os
import sys


def _register_scenes(app) -> None:
    from src.ui.scenes.game import GameScene
    from src.ui.scenes.help import HelpScene
    from src.ui.scenes.lab import LabScene
    from src.ui.scenes.menu import MenuScene
    from src.ui.scenes.results import ResultsScene
    from src.ui.scenes.setup import SetupScene
    from src.ui.scenes.train import TrainScene

    app.register("menu", MenuScene)
    app.register("setup", SetupScene)
    app.register("game", GameScene)
    app.register("results", ResultsScene)
    app.register("lab", LabScene)
    app.register("train", TrainScene)
    app.register("help", HelpScene)


def cmd_play(args: argparse.Namespace) -> int:
    if args.shot:
        # No window manager needed when we only want pixels on disk.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from src.ui.app import App

    app = App()
    _register_scenes(app)
    if args.shot:
        app.max_frames = args.frames
        app.shot_path = args.shot
    app.go(args.scene)
    # The first go() is deferred behind a fade; activate immediately instead.
    app._activate(args.scene, {})
    app._pending = None
    app._fading_out = False
    app.run()
    if args.shot:
        print(f"saved {args.shot}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from src.ai.train import format_report, train_heuristic

    def progress(frac: float, message: str) -> None:
        print(f"[{frac * 100:5.1f}%] {message}", flush=True)

    _, report = train_heuristic(
        n_mazes=args.mazes,
        epochs=args.epochs,
        progress=progress,
    )
    print(format_report(report))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from src.ai.benchmark import (
        format_table,
        heuristic_showdown,
        run_benchmark,
    )

    rows = run_benchmark(n_mazes=args.mazes, size=(args.width, args.height))
    print(format_table(rows))

    duel = heuristic_showdown(n_mazes=max(10, args.mazes // 2),
                              size=(args.width, args.height))
    if duel is None:
        print("Learned heuristic not trained yet - run `python main.py train`.\n")
    else:
        print("Manhattan vs learned heuristic")
        print(f"  mean nodes expanded : {duel['manhattan_mean']:.1f}  ->  "
              f"{duel['learned_mean']:.1f}   ({duel['reduction_pct']:.1f}% fewer)")
        print(f"  optimal path rate   : {duel['manhattan_optimal_rate'] * 100:.0f}%"
              f"  ->  {duel['learned_optimal_rate'] * 100:.0f}%")
        print(f"  mean cost / optimal : {duel['learned_cost_ratio']:.4f}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mazemind",
        description="MazeMind - a multiplayer maze solver with learned search.",
    )
    sub = parser.add_subparsers(dest="command")

    play = sub.add_parser("play", help="launch the game (default)")
    play.add_argument("--scene", default="menu",
                      choices=["menu", "setup", "game", "lab", "train", "help"])
    play.add_argument("--shot", metavar="PNG", help="render headlessly to a PNG")
    play.add_argument("--frames", type=int, default=90,
                      help="frames to render before capturing")
    play.set_defaults(func=cmd_play)

    train = sub.add_parser("train", help="train the learned heuristic")
    train.add_argument("--mazes", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.set_defaults(func=cmd_train)

    bench = sub.add_parser("bench", help="benchmark the search algorithms")
    bench.add_argument("--mazes", type=int, default=40)
    bench.add_argument("--width", type=int, default=31)
    bench.add_argument("--height", type=int, default=21)
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Bare `python main.py`, or flags with no subcommand, means "play".
    if not argv or argv[0].startswith("-"):
        argv = ["play"] + argv

    args = parser.parse_args(argv)
    if args.command == "train":
        from src.config import NN_EPOCHS, TRAIN_MAZES

        if args.mazes is None:
            args.mazes = TRAIN_MAZES
        if args.epochs is None:
            args.epochs = NN_EPOCHS
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
