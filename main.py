"""WUMPUS: THE MIND GAMES

    python main.py                       # you play the Wumpus in the terminal
    python main.py --seed 12             # a different (reproducible) dungeon
    python main.py --wumpus hunter       # watch the AI against a scripted Wumpus (idle/random/hunter)
"""
from __future__ import annotations

import argparse

from wumpus.config import GameConfig
from wumpus.terminal import configure_console, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="WUMPUS: THE MIND GAMES")
    ap.add_argument("--seed", type=int, default=7, help="dungeon seed (reproducible)")
    ap.add_argument("--wumpus", default="human", choices=["human", "idle", "random", "hunter"],
                    help="who controls the Wumpus")
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--no-board", action="store_true", help="do not print the board each turn")
    args = ap.parse_args(argv)
    configure_console()
    run(GameConfig(seed=args.seed, max_turns=args.max_turns), args.wumpus, show_board=not args.no_board)


if __name__ == "__main__":
    main()
