"""WUMPUS: THE MIND GAMES

    python main.py                         # Pygame interface: you play the Wumpus
    python main.py --seed 12               # another reproducible dungeon
    python main.py --wumpus minimax        # watch the AI against an AI Wumpus (random/hunter/minimax/idle)
    python main.py --terminal              # terminal version
"""
from __future__ import annotations

import argparse

from wumpus.config import GameConfig

MODES = ["human", "idle", "random", "hunter", "minimax"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="WUMPUS: THE MIND GAMES")
    ap.add_argument("--seed", type=int, default=7, help="dungeon seed (reproducible)")
    ap.add_argument("--wumpus", default="human", choices=MODES, help="who controls the Wumpus")
    ap.add_argument("--terminal", action="store_true", help="play in the terminal instead of the window")
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--no-board", action="store_true", help="terminal: do not print the board each turn")
    args = ap.parse_args(argv)
    if args.terminal:
        from wumpus.human_log import HUMAN_LOG
        from wumpus.terminal import configure_console, run
        configure_console()
        run(GameConfig(seed=args.seed, max_turns=args.max_turns), args.wumpus, show_board=not args.no_board,
            record_to=HUMAN_LOG)
    else:
        from wumpus.ui.app import run_gui
        run_gui(args.seed, args.wumpus)


if __name__ == "__main__":
    main()
