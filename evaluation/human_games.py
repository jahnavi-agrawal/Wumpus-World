"""Summarise games played by a human Wumpus (recorded by the GUI and the terminal version).

    python main.py                           # play as the Wumpus; every finished game is recorded
    python -m evaluation.human_games         # summary table of all recorded games

Records live in evaluation/results/human_games.jsonl (one JSON line per game, see wumpus/human_log.py).
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

from evaluation.benchmark import wilson
from wumpus.human_log import HUMAN_LOG


def load(path=HUMAN_LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(games: list[dict]) -> str:
    n = len(games)
    if n == 0:
        return (f"No human games recorded yet ({HUMAN_LOG}).\n"
                "Play with `python main.py` (you control the Wumpus); each finished game is recorded.")
    wins = sum(g["outcome"] == "EXPLORER_WIN" for g in games)
    lo, hi = wilson(wins, n)
    m = [g["metrics"] for g in games]
    tot = lambda k: sum(x[k] for x in m)
    share = lambda f: f"{sum(map(f, games)) / n:.0%}"
    lines = [f"Human Wumpus games: {n} (seeds {sorted({g['seed'] for g in games})})", "",
             "| Metric | Value |", "|---|---|",
             f"| Explorer win [95% CI] | {wins / n:.0%} [{lo:.0%}, {hi:.0%}] |",
             f"| Draw (explorer retreated) | {share(lambda g: g['outcome'] == 'DRAW')} |",
             f"| Explorer eaten | {share(lambda g: g['reason'] == 'EATEN')} |",
             f"| Explorer fell into a pit | {share(lambda g: g['reason'] == 'FELL_INTO_PIT')} |",
             f"| Turn limit | {share(lambda g: g['reason'] == 'turn limit reached')} |",
             f"| Median score | {statistics.median(g['score'] for g in games):.0f} |",
             f"| Avg explorer steps | {statistics.mean(g['steps'] for g in games):.1f} |",
             f"| Tracker: mean P(your true cell) | {tot('w_mass') / max(tot('w_turns'), 1):.0%} |",
             f"| Tracker: top cell correct | {tot('w_top1') / max(tot('w_turns'), 1):.0%} |",
             f"| Danger turns (you adjacent) | {tot('danger_turns')}, explorer eaten on {tot('danger_deaths')} |",
             f"| Minimax decisions per game | {tot('minimax_turns') / n:.1f} |",
             f"| Arrows shot (hits) | {tot('shots')} ({tot('hits')}) |", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(load(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HUMAN_LOG)))
